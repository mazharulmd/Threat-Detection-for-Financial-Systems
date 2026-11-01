"""Experiment driver.

  python -m experiments.run main     --dataset unsw --seeds 0 1 2
  python -m experiments.run ablation --dataset cic  --seeds 0 1 2
  python -m experiments.run privacy  --dataset cic  --seeds 0
  python -m experiments.run latency  --dataset unsw

Every (experiment, dataset, config, seed) result is written to
results/raw/<exp>_<dataset>_<config>_s<seed>.json; existing files are skipped,
so runs can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from halx import data as D
from halx import train as T

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "results", "raw")
EXTRA = os.path.join(ROOT, "results", "extra")
os.makedirs(RAW, exist_ok=True)
os.makedirs(EXTRA, exist_ok=True)
torch.set_num_threads(int(os.environ.get("HALX_THREADS", 4)))

_EPOCHS = {"cic": 15, "unsw": 8}  # max epochs (early stopping, patience 3)


def epochs_for(ds):
    return int(os.environ.get("HALX_EPOCHS", _EPOCHS[ds]))


def out_path(exp, ds, cfg, seed):
    return os.path.join(RAW, f"{exp}_{ds}_{cfg}_s{seed}.json")


def save(path, rec):
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)


def log_to(path):
    fh = open(path, "a")
    def log(msg):
        print(msg, flush=True)
        fh.write(msg + "\n")
        fh.flush()
    return log


def evaluate(p, pm, ds):
    return T.binary_metrics(ds.test.yb, p, ds.test.y, pm)


def fit_halx(ds, tr, seed, log, **kw):
    m = T.HALX(tr.Xn.shape[1], ds.cat_cards, len(ds.classes), seed=seed, **kw)
    t0 = time.time()
    m.fit(tr, ds.val, epochs=epochs_for(ds.name), log=log)
    return m, time.time() - t0


def run_main(dsname, seeds, methods, log):
    for seed in seeds:
        ds = None
        for meth in methods:
            path = out_path("main", dsname, meth, seed)
            if os.path.exists(path):
                continue
            if ds is None:
                ds = D.load(dsname, seed=seed)
                tr = D.smote(ds.train, ds.cat_cards, seed)
                log(f"[{dsname} s{seed}] train={len(tr.y)} (after SMOTE) val={len(ds.val.y)} test={len(ds.test.y)}")
            log(f"== main {dsname} {meth} seed={seed}")
            t0 = time.time()
            if meth == "HALX":
                m, _ = fit_halx(ds, tr, seed, log)
                p, pm = m.predict(ds.test)
                if seed == seeds[0]:
                    _, A = T.embed(m.enc, ds.test, return_attn=True)
                    names = ds.num_names + ds.cat_names
                    np.savez_compressed(os.path.join(EXTRA, f"halx_{dsname}_s{seed}.npz"),
                                        p=p, yb=ds.test.yb, y=ds.test.y, pm=pm,
                                        attn_mean_attack=A[ds.test.yb == 1].mean(0),
                                        attn_mean_benign=A[ds.test.yb == 0].mean(0),
                                        xgb_gain=_gain(m.xb, m.enc.emb_dim, names),
                                        names=np.array(names), classes=np.array(ds.classes))
            elif meth == "LSTM":
                m, _ = fit_halx(ds, tr, seed, log, use_attention=False, use_xgb=False)
                p, pm = m.predict(ds.test)
            else:
                p, pm, _ = T.run_baseline(meth, ds, tr, seed, epochs_for(dsname), log)
            rec = evaluate(p, pm, ds)
            rec.update(method=meth, dataset=dsname, seed=seed, train_time_s=time.time() - t0)
            if seed == seeds[0]:
                np.save(os.path.join(EXTRA, f"scores_{dsname}_{meth}_s{seed}.npy"), np.stack([ds.test.yb, p]))
            log(json.dumps(rec))
            save(path, rec)


def _gain(clf, emb_dim, names):
    g = clf.get_booster().get_score(importance_type="gain")
    v = np.zeros(emb_dim + len(names))
    for k, val in g.items():
        v[int(k[1:])] = val
    return v


ABLATIONS = {
    "full": dict(),
    "no_attention": dict(kw=dict(use_attention=False)),
    "no_xgboost": dict(kw=dict(use_xgb=False)),
    "no_raw_features": dict(kw=dict(use_raw=False)),
    "no_smote": dict(smote=False),
    "no_privacy": dict(privacy=False),
    "lstm_only": dict(kw=dict(use_attention=False, use_xgb=False)),
}


def run_ablation(dsname, seeds, log):
    for seed in seeds:
        for cfg, spec in ABLATIONS.items():
            path = out_path("ablation", dsname, cfg, seed)
            if os.path.exists(path):
                continue
            src = out_path("main", dsname, "HALX", seed) if cfg == "full" else None
            if src and os.path.exists(src):  # reuse identical main run
                rec = json.load(open(src)); rec.update(config=cfg); save(path, rec); continue
            log(f"== ablation {dsname} {cfg} seed={seed}")
            ds = D.load(dsname, seed=seed, privacy=spec.get("privacy", True))
            tr = D.smote(ds.train, ds.cat_cards, seed) if spec.get("smote", True) else ds.train
            m, tt = fit_halx(ds, tr, seed, log, **spec.get("kw", {}))
            p, pm = m.predict(ds.test)
            rec = evaluate(p, pm, ds)
            rec.update(config=cfg, dataset=dsname, seed=seed, train_time_s=tt)
            log(json.dumps(rec))
            save(path, rec)


def run_privacy(dsname, seeds, log, model="HALX"):
    grid = [("eps", e) for e in [0.01, 0.1, 1.0, 10.0, None]] + \
           [("k", k) for k in [1, 5, 25, 100, 500]]
    for seed in seeds:
        for kind, val in grid:
            cfg = f"{model}_{kind}{val}"
            path = out_path("privacy", dsname, cfg, seed)
            if os.path.exists(path):
                continue
            eps = val if kind == "eps" else 1.0
            k = val if kind == "k" else 10
            log(f"== privacy {dsname} {cfg} seed={seed}")
            ds = D.load(dsname, seed=seed, k=k, eps=eps)
            tr = D.smote(ds.train, ds.cat_cards, seed)
            if model == "HALX":
                m, _ = fit_halx(ds, tr, seed, log)
                p, pm = m.predict(ds.test)
            else:
                p, pm, _ = T.run_baseline(model, ds, tr, seed, epochs_for(dsname), log)
            rec = evaluate(p, pm, ds)
            rec.update(kind=kind, value=val, k=k, eps=eps, dataset=dsname, seed=seed, model=model,
                       n_leaves=int(getattr(ds.privacy.gen, "n_leaves_", 0)), min_class_size=ds.privacy.audit_k,
                       mean_noise_std=float(ds.privacy.std.noise_std_mean_))
            log(json.dumps(rec))
            save(path, rec)


def run_latency(dsname, log, n_single=500):
    path = os.path.join(RAW, f"latency_{dsname}.json")
    if os.path.exists(path):
        return
    ds = D.load(dsname, seed=0)
    tr = D.smote(ds.train, ds.cat_cards, 0)
    m, tt = fit_halx(ds, tr, 0, log)
    te = ds.test
    # batch throughput
    t0 = time.perf_counter(); m.predict(te); tb = time.perf_counter() - t0
    # single-record latency (privacy transform excluded: already applied; measured separately)
    idx = np.random.default_rng(0).choice(len(te.y), n_single, replace=False)
    lat = []
    for i in idx:
        s = D.Split(te.Xn[i:i + 1], te.Xc[i:i + 1], te.y[i:i + 1], te.yb[i:i + 1])
        t0 = time.perf_counter(); m.predict(s); lat.append(time.perf_counter() - t0)
    # privacy layer cost on raw rows
    raw = np.random.default_rng(0).normal(size=(len(te.y), te.Xn.shape[1]))
    t0 = time.perf_counter(); ds.privacy.transform(raw); tp = time.perf_counter() - t0
    n_params = sum(p.numel() for p in m.enc.parameters())
    rec = dict(dataset=dsname, n_test=len(te.y), batch_throughput_flows_per_s=len(te.y) / tb,
               single_latency_ms_median=1e3 * float(np.median(lat)),
               single_latency_ms_p95=1e3 * float(np.percentile(lat, 95)),
               privacy_layer_us_per_flow=1e6 * tp / len(te.y), encoder_params=int(n_params),
               xgb_trees_binary=int(m.xb.best_iteration + 1), train_time_s=tt,
               threads=torch.get_num_threads())
    log(json.dumps(rec))
    save(path, rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=["main", "ablation", "privacy", "latency"])
    ap.add_argument("--dataset", required=True, choices=["unsw", "cic"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+",
                    default=["XGBoost", "RF", "IsolationForest", "MLP", "CNN-LSTM", "LSTM", "Transformer", "HALX"])
    ap.add_argument("--model", default="HALX")
    a = ap.parse_args()
    log = log_to(os.path.join(ROOT, "results", f"log_{a.exp}_{a.dataset}.txt"))
    if a.exp == "main":
        run_main(a.dataset, a.seeds, a.methods, log)
    elif a.exp == "ablation":
        run_ablation(a.dataset, a.seeds, log)
    elif a.exp == "privacy":
        run_privacy(a.dataset, a.seeds, log, a.model)
    else:
        run_latency(a.dataset, log)
