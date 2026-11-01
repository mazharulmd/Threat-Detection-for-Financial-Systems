"""Neural encoders: HALX Attention-LSTM and the deep baselines."""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Maps a flow record to a sequence of d-dimensional feature tokens.

    numeric feature j:     t_j = x_j * w_j + b_j
    categorical feature c: t_c = E_c[code] + b_c
    Tokens follow a fixed feature order, so a recurrent encoder reads the
    flow record as a sequence (one step per feature).
    """

    def __init__(self, n_num: int, cat_cards: list[int], d: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.1)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.emb = nn.ModuleList([nn.Embedding(c, d) for c in cat_cards])
        self.cb = nn.Parameter(torch.zeros(len(cat_cards), d))
        self.n_tokens = n_num + len(cat_cards)

    def forward(self, xn, xc):
        t = xn.unsqueeze(-1) * self.w + self.b
        if len(self.emb):
            tc = torch.stack([e(xc[:, i]) for i, e in enumerate(self.emb)], 1) + self.cb
            t = torch.cat([t, tc], 1)
        return t


class AttnPool(nn.Module):
    """Additive attention pooling; returns pooled vector and token weights."""

    def __init__(self, d):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d, d), nn.Tanh(), nn.Linear(d, 1, bias=False))

    def forward(self, z):
        a = torch.softmax(self.proj(z).squeeze(-1), dim=1)
        return (a.unsqueeze(-1) * z).sum(1), a


class HALXEncoder(nn.Module):
    """Stacked BiLSTM -> multi-head self-attention -> attention pooling.

    ``use_attention=False`` gives the attention-free ablation (BiLSTM + mean
    pooling). The pooled embedding e (dim 2h) feeds the XGBoost head; the
    linear ``head`` is used only to train the encoder (and for the
    "w/o XGBoost" ablation).
    """

    def __init__(self, n_num, cat_cards, n_classes, d_tok=16, hidden=32,
                 layers=2, heads=4, dropout=0.1, use_attention=True):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cat_cards, d_tok)
        self.lstm = nn.LSTM(d_tok, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        D = 2 * hidden
        self.use_attention = use_attention
        if use_attention:
            self.mha = nn.MultiheadAttention(D, heads, dropout=dropout, batch_first=True)
            self.ln = nn.LayerNorm(D)
            self.pool = AttnPool(D)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(D, n_classes)
        self.emb_dim = D

    def embed(self, xn, xc, return_attn=False):
        h, _ = self.lstm(self.tok(xn, xc))
        if self.use_attention:
            a_out, _ = self.mha(h, h, h, need_weights=False)
            z = self.ln(h + self.drop(a_out))
            e, a = self.pool(z)
        else:
            e, a = h.mean(1), None
        return (e, a) if return_attn else e

    def forward(self, xn, xc):
        return self.head(self.drop(self.embed(xn, xc)))


class MLP(nn.Module):
    def __init__(self, n_num, cat_cards, n_classes, hidden=256, dropout=0.1):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(c, 8) for c in cat_cards])
        d = n_num + 8 * len(cat_cards)
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, n_classes))

    def forward(self, xn, xc):
        parts = [xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)]
        return self.net(torch.cat(parts, 1))


class CNNLSTM(nn.Module):
    """Conv1d feature extractor over feature tokens followed by an LSTM."""

    def __init__(self, n_num, cat_cards, n_classes, d_tok=16, ch=64, hidden=64, dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cat_cards, d_tok)
        self.conv = nn.Sequential(nn.Conv1d(d_tok, ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv1d(ch, ch, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.lstm = nn.LSTM(ch, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, xn, xc):
        t = self.conv(self.tok(xn, xc).transpose(1, 2)).transpose(1, 2)
        _, (h, _) = self.lstm(t)
        return self.head(h[-1])


class TransformerIDS(nn.Module):
    """Transformer encoder over feature tokens with a [CLS] token."""

    def __init__(self, n_num, cat_cards, n_classes, d=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, cat_cards, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_classes))

    def forward(self, xn, xc):
        t = self.tok(xn, xc)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        return self.head(self.enc(t)[:, 0])
