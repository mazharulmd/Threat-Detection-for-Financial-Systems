# Usage: PYTHONPATH=. python3 experiments/design_selection.py cic
# Design selection for the HALX head, using validation data ONLY (test never touched).
import sys, numpy as np, torch
torch.set_num_threads(2)
from halx import data as D, train as T
from sklearn.metrics import f1_score
name = sys.argv[1]
ds = D.load(name, seed=0); tr = D.smote(ds.train, ds.cat_cards, 0)
rng = np.random.default_rng(0); idx = rng.permutation(len(ds.val.y)); h = len(idx)//2
sub = lambda s, i: D.Split(s.Xn[i], s.Xc[i], s.y[i], s.yb[i])
v1, v2 = sub(ds.val, idx[:h]), sub(ds.val, idx[h:])   # v1: early stopping, v2: selection
C = len(ds.classes)
def rep(tag, p, pm):
    m = T.binary_metrics(v2.yb, p, v2.y, pm); print(tag, {k: round(v, 4) for k, v in m.items()}, flush=True)
# XGB raw reference
xb = T.xgb_fit(T.tab(tr), tr.yb, T.tab(v1), v1.yb, 2); xm = T.xgb_fit(T.tab(tr), tr.y, T.tab(v1), v1.y, C)
rep("XGB-raw", xb.predict_proba(T.tab(v2))[:, 1], xm.predict_proba(T.tab(v2)).argmax(1))
# full encoder
enc = T.M.HALXEncoder(tr.Xn.shape[1], ds.cat_cards, C); T.train_torch(enc, tr, v1, log=lambda m: None)
P = lambda e, s: T.predict_proba(e, s)
rep("enc-dense", 1 - P(enc, v2)[:, 0], P(enc, v2).argmax(1))
for tag, f in [("A emb+raw", lambda s: np.hstack([T.embed(enc, s), T.tab(s)])),
               ("C prob+raw", lambda s: np.hstack([P(enc, s), T.tab(s)]))]:
    b = T.xgb_fit(f(tr), tr.yb, f(v1), v1.yb, 2); m = T.xgb_fit(f(tr), tr.y, f(v1), v1.y, C)
    rep(tag, b.predict_proba(f(v2))[:, 1], m.predict_proba(f(v2)).argmax(1))
# B: 2-fold out-of-fold stacking (probabilities + embeddings are fold-specific -> use probs)
perm = rng.permutation(len(tr.y)); fA, fB = perm[: len(perm)//2], perm[len(perm)//2:]
encs, oof = [], np.zeros((len(tr.y), C), np.float32)
for fit_i, oof_i in [(fA, fB), (fB, fA)]:
    e = T.M.HALXEncoder(tr.Xn.shape[1], ds.cat_cards, C); T.train_torch(e, sub(tr, fit_i), v1, log=lambda m: None)
    oof[oof_i] = P(e, sub(tr, oof_i)); encs.append(e)
avg = lambda s: np.mean([P(e, s) for e in encs], 0)
Ftr = np.hstack([oof, T.tab(tr)]); f = lambda s: np.hstack([avg(s), T.tab(s)])
b = T.xgb_fit(Ftr, tr.yb, f(v1), v1.yb, 2); m = T.xgb_fit(Ftr, tr.y, f(v1), v1.y, C)
rep("B oof-prob+raw", b.predict_proba(f(v2))[:, 1], m.predict_proba(f(v2)).argmax(1))
