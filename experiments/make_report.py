"""Turn results/raw/*.json into LaTeX tables and pgfplots data for the paper.

All numbers that appear in paper/generated/*.tex come from this script, so the
paper can be regenerated after re-running experiments:

    python -m experiments.make_report
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from sklearn.metrics import roc_curve

ROOT = os.path.join(os.path.dirname(__file__), "..")
RAW = os.path.join(ROOT, "results", "raw")
EXTRA = os.path.join(ROOT, "results", "extra")
GEN = os.path.join(ROOT, "paper", "generated")
os.makedirs(GEN, exist_ok=True)

DS = {"cic": "CICIDS2017 (stratified subset)", "unsw": "UNSW-NB15 (official split)"}
MAIN_ORDER = ["IsolationForest", "RF", "XGBoost", "MLP", "CNN-LSTM", "LSTM", "Transformer", "HALX"]
LABEL = {"IsolationForest": "Isolation Forest~\\cite{liu2008iforest}", "RF": "Random Forest~\\cite{breiman2001rf}",
         "XGBoost": "XGBoost~\\cite{chen2016xgboost}", "MLP": "MLP", "CNN-LSTM": "CNN-LSTM~\\cite{halbouni2022cnnlstm}",
         "LSTM": "BiLSTM (no attention)", "Transformer": "Transformer~\\cite{gorishniy2021revisiting}",
         "HALX": "\\textbf{HALX (ours)}"}
METRICS = [("acc", "Acc."), ("precision", "Prec."), ("recall", "Rec."), ("f1", "F1"), ("fpr", "FPR"),
           ("auc", "AUC"), ("macro_f1", "M-F1")]


def load(pattern):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, pattern)))]


def agg(recs, key):
    v = np.array([r[key] for r in recs if r.get(key) is not None], float)
    return (v.mean(), v.std(ddof=0), len(v)) if len(v) else (np.nan, np.nan, 0)


def fmt(m, s, n, pct=True, best=False, digits=2):
    if np.isnan(m):
        return "--"
    k = 100 if pct else 1
    body = f"{m*k:.{digits}f}"
    if n > 1:
        body += f"{{\\tiny$\\pm${s*k:.{digits}f}}}"
    return f"\\textbf{{{body}}}" if best else body


def main_table():
    lines = []
    summary = {}
    for ds in ["cic", "unsw"]:
        rows = {}
        for meth in MAIN_ORDER:
            recs = load(f"main_{ds}_{meth}_s*.json")
            if recs:
                rows[meth] = {k: agg(recs, k) for k, _ in METRICS}
        if not rows:
            continue
        best = {}
        for k, _ in METRICS:
            vals = {m: r[k][0] for m, r in rows.items() if not np.isnan(r[k][0])}
            if vals:
                best[k] = (min if k == "fpr" else max)(vals, key=vals.get)
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{DS[ds]}}}}} \\\\")
        for meth in MAIN_ORDER:
            if meth not in rows:
                continue
            r = rows[meth]
            cells = [fmt(*r[k], best=(best.get(k) == meth)) for k, _ in METRICS]
            lines.append(f"{LABEL[meth]} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
        summary[ds] = {m: {k: rows[m][k][0] for k, _ in METRICS} for m in rows}
        summary[ds]["_n"] = {m: rows[m]["f1"][2] for m in rows}
    if lines:
        lines = lines[:-1]
    head = " & ".join(["\\textbf{Method}"] + [f"\\textbf{{{h}}}" for _, h in METRICS]) + " \\\\"
    tex = ("\\begin{tabular}{@{}l@{\\hspace{3pt}}ccccccc@{}}\n\\toprule\n" + head + "\n\\midrule\n" +
           "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(GEN, "table_main.tex"), "w").write(tex)
    json.dump(summary, open(os.path.join(GEN, "summary_main.json"), "w"), indent=1)


ABL = [("full", "Full HALX"), ("no_attention", "w/o attention (mean pooling)"),
       ("no_xgboost", "w/o XGBoost (dense softmax head)"), ("no_raw_features", "w/o raw features in head"),
       ("no_smote", "w/o SMOTE"), ("no_privacy", "w/o privacy layer"), ("lstm_only", "BiLSTM only")]


def ablation_table():
    have = {ds: {c: load(f"ablation_{ds}_{c}_s*.json") for c, _ in ABL} for ds in ["cic", "unsw"]}
    lines = []
    for c, name in ABL:
        cells = []
        for ds in ["cic", "unsw"]:
            recs = have[ds][c]
            for k in ["f1", "fpr", "macro_f1"]:
                cells.append(fmt(*agg(recs, k)) if recs else "--")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    n = {ds: max([len(v) for v in have[ds].values()] + [0]) for ds in have}
    tex = ("\\begin{tabular}{@{}l@{\\hspace{2pt}}cccccc@{}}\n\\toprule\n"
           " & \\multicolumn{3}{c}{\\textbf{CICIDS2017}} & \\multicolumn{3}{c}{\\textbf{UNSW-NB15}} \\\\\n"
           "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n"
           "\\textbf{Configuration} & F1 & FPR & M-F1 & F1 & FPR & M-F1 \\\\\n\\midrule\n" +
           "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(GEN, "table_ablation.tex"), "w").write(tex)
    json.dump(n, open(os.path.join(GEN, "ablation_seeds.json"), "w"))


def privacy_data():
    for ds in ["cic", "unsw"]:
        for model in ["HALX", "XGBoost"]:
            recs = load(f"privacy_{ds}_{model}_*_s*.json")
            if not recs:
                continue
            for kind in ["eps", "k"]:
                pts = sorted([r for r in recs if r["kind"] == kind],
                             key=lambda r: (r["value"] is None, r["value"] or 0))
                with open(os.path.join(GEN, f"priv_{ds}_{model}_{kind}.dat"), "w") as f:
                    f.write("x label f1 fpr macro_f1 auc\n")
                    for i, r in enumerate(pts):
                        lab = "inf" if r["value"] is None else r["value"]
                        f.write(f"{i} {lab} {100*r['f1']:.3f} {100*r['fpr']:.3f} {100*r['macro_f1']:.3f} {100*r['auc']:.4f}\n")


def roc_data(n_pts=200):
    for ds in ["cic", "unsw"]:
        for meth in ["HALX", "XGBoost", "RF", "Transformer", "CNN-LSTM", "IsolationForest"]:
            f = os.path.join(EXTRA, f"scores_{ds}_{meth}_s0.npy")
            if not os.path.exists(f):
                continue
            yb, p = np.load(f)
            fpr, tpr, _ = roc_curve(yb, p)
            grid = np.unique(np.concatenate([np.logspace(-4, 0, n_pts), [0.0]]))
            t = np.interp(grid, fpr, tpr)
            np.savetxt(os.path.join(GEN, f"roc_{ds}_{meth}.dat"), np.c_[np.maximum(grid, 1e-4) * 100, t * 100],
                       header="fpr tpr", comments="", fmt="%.5f")


def attention_data(top=10):
    out = {}
    for ds in ["cic", "unsw"]:
        f = os.path.join(EXTRA, f"halx_{ds}_s0.npz")
        if not os.path.exists(f):
            continue
        z = np.load(f, allow_pickle=True)
        a, b, names = z["attn_mean_attack"], z["attn_mean_benign"], list(z["names"])
        idx = np.argsort(-a)[:top][::-1]
        with open(os.path.join(GEN, f"attn_{ds}.dat"), "w") as fh:
            fh.write("i name attack benign\n")
            for j, i in enumerate(idx):
                nm = names[i].replace(" ", "~").replace("_", "\\_").replace("/", "/")
                fh.write(f"{j} {{{nm}}} {100*a[i]:.3f} {100*b[i]:.3f}\n")
        # severity bands on the test set
        p, yb = z["p"], z["yb"]
        bands = np.digitize(p, [0.5, 0.8, 0.95])
        out[ds] = {name: dict(n=int((bands == i).sum()), attack_frac=float(yb[bands == i].mean()) if (bands == i).any() else None)
                   for i, name in enumerate(["benign", "low", "medium", "high"])}
        out[ds]["uniform_attention"] = 1.0 / len(names)
    json.dump(out, open(os.path.join(GEN, "severity_attention.json"), "w"), indent=1)


def _mname(*parts):
    """LaTeX macro name from parts, letters only (digits spelled out)."""
    digits = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))
    out = "".join("".join(digits.get(ch, ch) for ch in p if ch.isalnum()) for p in parts)
    return "\\" + out


def numbers():
    """Macros like \\cicHALXfOne{99.51} for every metric quoted in the prose."""
    L = ["% auto-generated by experiments/make_report.py -- do not edit"]
    keyname = {"acc": "acc", "precision": "prec", "recall": "rec", "f1": "fOne", "fpr": "fpr", "auc": "auc", "macro_f1": "mfOne"}
    for ds in ["cic", "unsw"]:
        for meth in MAIN_ORDER:
            recs = load(f"main_{ds}_{meth}_s*.json")
            if not recs:
                continue
            for k, kn in keyname.items():
                m, sd, n = agg(recs, k)
                if not np.isnan(m):
                    L.append(f"\\newcommand{{{_mname(ds, meth, kn)}}}{{{100*m:.2f}}}")
                    L.append(f"\\newcommand{{{_mname(ds, meth, kn, 'sd')}}}{{{100*sd:.2f}}}")
            tt = agg(recs, "train_time_s")[0]
            L.append(f"\\newcommand{{{_mname(ds, meth, 'time')}}}{{{tt/60:.1f}}}")
        for c, _ in ABL:
            recs = load(f"ablation_{ds}_{c}_s*.json")
            for k in ["f1", "fpr", "macro_f1"]:
                if recs:
                    L.append(f"\\newcommand{{{_mname(ds, 'abl', c, keyname[k])}}}{{{100*agg(recs, k)[0]:.2f}}}")
    for f in glob.glob(os.path.join(RAW, "latency_*.json")):
        r = json.load(open(f))
        ds = r["dataset"]
        L.append(f"\\newcommand{{{_mname(ds, 'thru')}}}{{{r['batch_throughput_flows_per_s']:,.0f}}}".replace(",", "{,}"))
        L.append(f"\\newcommand{{{_mname(ds, 'latmed')}}}{{{r['single_latency_ms_median']:.2f}}}")
        L.append(f"\\newcommand{{{_mname(ds, 'latpn')}}}{{{r['single_latency_ms_p95']:.2f}}}")
        L.append(f"\\newcommand{{{_mname(ds, 'privus')}}}{{{r['privacy_layer_us_per_flow']:.2f}}}")
        L.append(f"\\newcommand{{{_mname(ds, 'params')}}}{{{r['encoder_params']:,}}}".replace(",", "{,}"))
    sev = os.path.join(GEN, "severity_attention.json")
    if os.path.exists(sev):
        for ds, d in json.load(open(sev)).items():
            for band in ["low", "medium", "high", "benign"]:
                b = d[band]
                if b["attack_frac"] is not None:
                    L.append(f"\\newcommand{{{_mname(ds, 'sev', band)}}}{{{100*b['attack_frac']:.1f}}}")
                    L.append(f"\\newcommand{{{_mname(ds, 'sevn', band)}}}{{{b['n']:,}}}".replace(",", "{,}"))
    open(os.path.join(GEN, "numbers.tex"), "w").write("\n".join(L) + "\n")


def latency():
    out = {}
    for f in glob.glob(os.path.join(RAW, "latency_*.json")):
        r = json.load(open(f))
        out[r["dataset"]] = r
    json.dump(out, open(os.path.join(GEN, "latency.json"), "w"), indent=1)


if __name__ == "__main__":
    main_table()
    ablation_table()
    privacy_data()
    roc_data()
    attention_data()
    latency()
    numbers()
    print("written to", GEN)
    for f in sorted(os.listdir(GEN)):
        print(" ", f)
