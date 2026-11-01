"""Add tpr_at_1fpr to result JSONs written before the metric existed (needs saved scores)."""
import glob, json, os
import numpy as np
from halx.train import tpr_at_fpr
ROOT = os.path.join(os.path.dirname(__file__), "..")
for f in glob.glob(os.path.join(ROOT, "results/raw/main_*_s0.json")):
    r = json.load(open(f))
    if "tpr_at_1fpr" in r:
        continue
    sc = os.path.join(ROOT, "results/extra", f"scores_{r['dataset']}_{r['method']}_s0.npy")
    if os.path.exists(sc):
        yb, p = np.load(sc)
        r["tpr_at_1fpr"] = tpr_at_fpr(yb.astype(int), p, 0.01)
        json.dump(r, open(f, "w"), indent=1)
        print("backfilled", os.path.basename(f))
