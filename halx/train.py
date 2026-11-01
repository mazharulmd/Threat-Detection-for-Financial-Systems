"""Training utilities, the HALX pipeline, baselines and metrics."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix)
import xgboost as xgb

from .data import Split
from . import models as M

import os
NJ = int(os.environ.get("HALX_THREADS", 4))


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def binary_metrics(yb, p_attack, y_multi=None, pred_multi=None, tau=0.5):
    yhat = (p_attack >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(yb, yhat, labels=[0, 1]).ravel()
    out = dict(acc=accuracy_score(yb, yhat), precision=precision_score(yb, yhat, zero_division=0),
               recall=recall_score(yb, yhat), f1=f1_score(yb, yhat), fpr=fp / (fp + tn),
               auc=roc_auc_score(yb, p_attack))
    if pred_multi is not None:
        out["macro_f1"] = f1_score(y_multi, pred_multi, average="macro")
    return {k: float(v) for k, v in out.items()}


# --------------------------------------------------------------------------
# Torch training
# --------------------------------------------------------------------------
def _tensors(s: Split):
    c = np.ascontiguousarray
    return torch.from_numpy(c(s.Xn)), torch.from_numpy(c(s.Xc)), torch.from_numpy(c(s.y))


@torch.no_grad()
def predict_proba(model, s: Split, bs=4096):
    model.eval()
    xn, xc, _ = _tensors(s)
    out = [F.softmax(model(xn[i:i + bs], xc[i:i + bs]), 1) for i in range(0, len(xn), bs)]
    return torch.cat(out).numpy()


def train_torch(model, tr: Split, va: Split, epochs=15, bs=1024, lr=3e-3, wd=1e-4,
                patience=3, seed=0, log=print):
    torch.manual_seed(seed)
    xn, xc, y = _tensors(tr)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = -1.0, None, 0
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(y), generator=g)
        t0, tot = time.time(), 0.0
        for i in range(0, len(y), bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(model(xn[b], xc[b]), y[b])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        pv = predict_proba(model, va)
        score = f1_score(va.y, pv.argmax(1), average="macro")
        log(f"  ep{ep:02d} loss={tot/len(y):.4f} val_macroF1={score:.4f} ({time.time()-t0:.0f}s)")
        if score > best + 1e-4:
            best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def embed(enc: M.HALXEncoder, s: Split, bs=4096, return_attn=False):
    enc.eval()
    xn, xc, _ = _tensors(s)
    E, A = [], []
    for i in range(0, len(xn), bs):
        e, a = enc.embed(xn[i:i + bs], xc[i:i + bs], return_attn=True)
        E.append(e)
        if a is not None:
            A.append(a)
    E = torch.cat(E).numpy()
    return (E, torch.cat(A).numpy() if A else None) if return_attn else E


# --------------------------------------------------------------------------
# XGBoost head
# --------------------------------------------------------------------------
def xgb_fit(Xtr, ytr, Xva, yva, n_classes, seed=0):
    params = dict(n_estimators=800, max_depth=8, learning_rate=0.08, subsample=0.9,
                  colsample_bytree=0.8, min_child_weight=1, tree_method="hist",
                  n_jobs=NJ, random_state=seed, early_stopping_rounds=40)
    if n_classes == 2:
        clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
    else:
        clf = xgb.XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", **params)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return clf


def tab(s: Split):
    return np.hstack([s.Xn, s.Xc.astype(np.float32)])


# --------------------------------------------------------------------------
# HALX
# --------------------------------------------------------------------------
class HALX:
    """Attention-LSTM encoder + XGBoost scorer (binary) and classifier (multi)."""

    def __init__(self, n_num, cat_cards, n_classes, use_attention=True, use_raw=True,
                 use_xgb=True, enc_kw=None, seed=0):
        self.enc = M.HALXEncoder(n_num, cat_cards, n_classes, use_attention=use_attention, **(enc_kw or {}))
        self.use_raw, self.use_xgb, self.n_classes, self.seed = use_raw, use_xgb, n_classes, seed

    def _feat(self, s):
        e = embed(self.enc, s)
        return np.hstack([e, tab(s)]) if self.use_raw else e

    def fit(self, tr: Split, va: Split, epochs=20, log=print):
        torch.manual_seed(self.seed)
        train_torch(self.enc, tr, va, epochs=epochs, seed=self.seed, log=log)
        if self.use_xgb:
            Ftr, Fva = self._feat(tr), self._feat(va)
            self.xb = xgb_fit(Ftr, tr.yb, Fva, va.yb, 2, self.seed)
            self.xm = xgb_fit(Ftr, tr.y, Fva, va.y, self.n_classes, self.seed)
        return self

    def predict(self, s: Split):
        """Returns (P(attack), multi-class prediction)."""
        if not self.use_xgb:
            p = predict_proba(self.enc, s)
            return 1 - p[:, 0], p.argmax(1)
        Fs = self._feat(s)
        return self.xb.predict_proba(Fs)[:, 1], self.xm.predict_proba(Fs).argmax(1)


def severity_band(p, tau=0.5):
    """0 = benign, 1 = low, 2 = medium, 3 = high severity alert."""
    return np.digitize(p, [tau, 0.8, 0.95])


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def run_baseline(name, ds, tr, seed=0, epochs=20, log=print):
    va, te = ds.val, ds.test
    C = len(ds.classes)
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        rb = RandomForestClassifier(300, n_jobs=NJ, random_state=seed).fit(tab(tr), tr.yb)
        rm = RandomForestClassifier(300, n_jobs=NJ, random_state=seed).fit(tab(tr), tr.y)
        return rb.predict_proba(tab(te))[:, 1], rm.predict(tab(te)), (rb, rm)
    if name == "XGBoost":
        xb = xgb_fit(tab(tr), tr.yb, tab(va), va.yb, 2, seed)
        xm = xgb_fit(tab(tr), tr.y, tab(va), va.y, C, seed)
        return xb.predict_proba(tab(te))[:, 1], xm.predict_proba(tab(te)).argmax(1), (xb, xm)
    if name == "IsolationForest":
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(n_estimators=300, random_state=seed, n_jobs=NJ).fit(tab(tr)[tr.yb == 0])
        sv, st = -iso.score_samples(tab(va)), -iso.score_samples(tab(te))
        # threshold chosen on validation to maximise F1, then mapped to [0,1]
        qs = np.quantile(sv, np.linspace(0.01, 0.99, 197))
        best = max(qs, key=lambda q: f1_score(va.yb, (sv >= q).astype(int)))
        lo, hi = np.quantile(sv, [0.001, 0.999])
        p = np.clip(0.5 + (st - best) / (2 * (hi - lo)), 0, 1)
        return p, None, iso
    ctor = {"MLP": M.MLP, "CNN-LSTM": M.CNNLSTM, "Transformer": M.TransformerIDS}[name]
    torch.manual_seed(seed)
    model = ctor(tr.Xn.shape[1], ds.cat_cards, C)
    train_torch(model, tr, va, epochs=epochs, seed=seed, log=log)
    p = predict_proba(model, te)
    return 1 - p[:, 0], p.argmax(1), model
