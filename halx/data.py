"""Dataset loading and splitting for UNSW-NB15 and the CICIDS2017 sample."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .privacy import PrivacyLayer, signed_log1p

DATA_DIR = os.environ.get("HALX_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))

# Quasi-identifiers: attributes that, combined, can fingerprint a host/session
# (volumes ~ transaction sizes, durations, TTL / window ~ OS fingerprint).
QI = {
    "unsw": ["dur", "sbytes", "dbytes", "sttl"],
    "cic": ["Flow Duration", "Total Length of Fwd Packets",
            "Total Length of Bwd Packets", "Init_Win_bytes_forward"],
}
# Direct identifiers suppressed if present (absent in the public partitions).
DIRECT_IDS = ["id", "srcip", "dstip", "sport", "dsport", "stime", "ltime",
              "Flow ID", "Source IP", "Destination IP", "Source Port", "Timestamp"]


@dataclass
class Split:
    Xn: np.ndarray      # numeric features (privatized, standardized) float32
    Xc: np.ndarray      # categorical codes int64 (may have 0 columns)
    y: np.ndarray       # multi-class labels int64
    yb: np.ndarray      # binary labels (1 = attack)


@dataclass
class Dataset:
    name: str
    train: Split
    val: Split
    test: Split
    num_names: list
    cat_names: list
    cat_cards: list
    classes: list
    benign: int
    privacy: PrivacyLayer


def _load_raw(name):
    if name == "unsw":
        tr = pd.read_csv(os.path.join(DATA_DIR, "unsw/UNSW_NB15_training-set.csv"), encoding="utf-8-sig")
        te = pd.read_csv(os.path.join(DATA_DIR, "unsw/UNSW_NB15_testing-set.csv"), encoding="utf-8-sig")
        for d in (tr, te):
            d["attack_cat"] = d["attack_cat"].str.strip().replace({"Backdoors": "Backdoor"})
        return tr, te, "attack_cat", "Normal", ["proto", "service", "state"]
    if name == "cic":
        df = pd.read_csv(os.path.join(DATA_DIR, "cic/CICIDS2017_sample.csv"))
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.T.duplicated()]  # drops 'Fwd Header Length.1'
        return df, None, "Label", "BENIGN", []
    raise ValueError(name)


def load(name: str, seed: int = 0, k: int = 10, eps: float | None = 1.0,
         delta: float = 1e-5, privacy: bool = True) -> Dataset:
    tr, te, lab, benign_name, cats = _load_raw(name)
    if te is None:  # stratified 70/15/15
        tr, te = train_test_split(tr, test_size=0.15, stratify=tr[lab], random_state=seed)
        tr, va = train_test_split(tr, test_size=0.15 / 0.85, stratify=tr[lab], random_state=seed)
    else:  # official test partition kept untouched; 15% of train -> val
        tr, va = train_test_split(tr, test_size=0.15, stratify=tr[lab], random_state=seed)

    drop = [c for c in DIRECT_IDS if c in tr.columns] + [lab] + (["label"] if "label" in tr.columns else [])
    num = [c for c in tr.columns if c not in drop and c not in cats]
    # remove constant columns (decided on training data only)
    num = [c for c in num if tr[c].nunique() > 1]

    classes = sorted(tr[lab].unique(), key=lambda c: (c != benign_name, c))
    cmap = {c: i for i, c in enumerate(classes)}

    def num_mat(d):
        X = d[num].to_numpy(np.float64)
        X[~np.isfinite(X)] = np.nan
        return X

    Xtr = num_mat(tr)
    med = np.nanmedian(Xtr, axis=0)
    def impute(X):
        m = np.isnan(X)
        X[m] = np.take(med, np.where(m)[1])
        return signed_log1p(X)

    Xtr, Xva, Xte = impute(Xtr), impute(num_mat(va)), impute(num_mat(te))

    qi_idx = [num.index(c) for c in QI[name]]
    layer = PrivacyLayer(qi_idx, k=k if privacy else 1, eps=eps if privacy else None,
                         delta=delta, seed=seed).fit(Xtr)
    layer.audit_k = layer.gen.min_group_size(Xtr)  # smallest QI equivalence class on train

    # categorical codes: top-31 levels on train + "other"
    cat_maps, cards = [], []
    for c in cats:
        top = tr[c].value_counts().index[:31]
        cat_maps.append({v: i + 1 for i, v in enumerate(top)})
        cards.append(len(top) + 1)

    def mk(d, X):
        Xc = (np.stack([d[c].map(m).fillna(0).to_numpy(np.int64) for c, m in zip(cats, cat_maps)], 1)
              if cats else np.zeros((len(d), 0), np.int64))
        y = d[lab].map(cmap).to_numpy(np.int64)
        return Split(layer.transform(X), Xc, y, (y != 0).astype(np.int64))

    return Dataset(name, mk(tr, Xtr), mk(va, Xva), mk(te, Xte), num, cats, cards,
                   classes, 0, layer)


def smote(split: Split, n_cat_cards: list, seed: int = 0, min_frac: float = 0.02) -> Split:
    """SMOTE-NC on the training split only: lift each class to >= min_frac*N."""
    from imblearn.over_sampling import SMOTE, SMOTENC
    n = len(split.y)
    counts = np.bincount(split.y)
    target = {c: max(int(min_frac * n), cnt) for c, cnt in enumerate(counts) if cnt > 0}
    if all(target[c] == counts[c] for c in target):
        return split
    X = np.hstack([split.Xn, split.Xc.astype(np.float32)])
    nn = max(1, min(5, counts[counts > 0].min() - 1))
    if split.Xc.shape[1]:
        cat_idx = list(range(split.Xn.shape[1], X.shape[1]))
        sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=target, k_neighbors=nn, random_state=seed)
    else:
        sm = SMOTE(sampling_strategy=target, k_neighbors=nn, random_state=seed)
    Xr, yr = sm.fit_resample(X, split.y)
    d = split.Xn.shape[1]
    return Split(Xr[:, :d].astype(np.float32), Xr[:, d:].astype(np.int64), yr, (yr != 0).astype(np.int64))
