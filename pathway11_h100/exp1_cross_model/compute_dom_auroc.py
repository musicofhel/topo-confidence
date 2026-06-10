"""Cross-architecture prefill DoM AUROC (F-2 generalization test).

Does the L~2/3-depth prefill direction-of-means predict K=1 correctness on
Phi-3-mini and Llama-3.2-1B, the way L19 does on Qwen-2.5-1.5B (AUROC 0.7731)?

Uses cached NPZs in exp1_cross_model/data/{phi3mini,llama32_1b}/.
Same OOF protocol as FE459: StratifiedKFold(5, seed=9999), mean-center on train,
DoM = mu_correct - mu_incorrect, unit-norm, project test, rank-AUROC.

Output: pathway11_h100/exp1_cross_model/dom_auroc_results.json
"""
from __future__ import annotations
import json, glob, os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
BASE = ROOT / "pathway11_h100/exp1_cross_model"
OUT_JSON = BASE / "dom_auroc_results.json"
SEED, N_FOLDS = 9999, 5

# Qwen-1.5B reference: L19 of 28 layers = 0.679 depth fraction.
MODELS = {
    "phi3mini":   {"n_layers": 33, "twothirds": 21},  # 21/33 = 0.636
    "llama32_1b": {"n_layers": 17, "twothirds": 11},   # 11/17 = 0.647
}


def auroc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def oof_dom_auroc(X, y):
    """5-fold OOF DoM AUROC for one (n, d) prefill matrix."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = np.zeros(len(X), dtype=np.float64)
    for tr, te in skf.split(X, y):
        mu = X[tr].mean(0)
        Xc = X[tr] - mu
        dom = Xc[y[tr]].mean(0) - Xc[~y[tr]].mean(0)
        dom /= (np.linalg.norm(dom) + 1e-30)
        scores[te] = (X[te] - mu) @ dom
    return auroc(scores, y), scores


def load_model(name, n_layers):
    fs = sorted(glob.glob(str(BASE / f"data/{name}/problem_*.npz")))
    n = len(fs)
    d0 = np.load(fs[0], allow_pickle=True)
    hid = d0["prefill_all_layers"].shape[1]
    X = np.zeros((n, n_layers, hid), dtype=np.float32)
    y = np.zeros(n, dtype=bool)
    for i, fp in enumerate(fs):
        with np.load(fp, allow_pickle=True) as d:
            X[i] = d["prefill_all_layers"].astype(np.float32)
            y[i] = bool(d["correct"])
    return X, y


def main():
    out = {"experiment": "cross_arch_prefill_dom_auroc",
           "reference_qwen15b_L19": 0.7731, "n_folds": N_FOLDS, "seed": SEED,
           "models": {}}
    for name, cfg in MODELS.items():
        X, y = load_model(name, cfg["n_layers"])
        n_correct = int(y.sum())
        # full prefill-layer sweep
        sweep = {}
        for L in range(cfg["n_layers"]):
            a, _ = oof_dom_auroc(X[:, L, :], y)
            sweep[L] = round(a, 4)
        peak_L = max(sweep, key=sweep.get)
        tt = cfg["twothirds"]
        out["models"][name] = {
            "n": len(X), "n_correct": n_correct,
            "accuracy": round(n_correct / len(X), 4),
            "hidden_dim": int(X.shape[2]), "n_layers": cfg["n_layers"],
            "twothirds_layer": tt,
            "auroc_twothirds": sweep[tt],
            "peak_layer": peak_L, "auroc_peak": sweep[peak_L],
            "layer_sweep": sweep,
        }
        print(f"\n{name}: acc={n_correct}/{len(X)}={n_correct/len(X):.3f}")
        print(f"  2/3-depth L{tt} AUROC = {sweep[tt]:.4f}")
        print(f"  peak       L{peak_L} AUROC = {sweep[peak_L]:.4f}")

    # verdict
    aus = [out["models"][m]["auroc_peak"] for m in MODELS]
    out["verdict"] = {
        "min_peak_auroc": min(aus), "max_peak_auroc": max(aus),
        "f2_generalizes": all(a >= 0.65 for a in aus),
        "note": "F-2 generalizes if both architectures show prefill DoM AUROC >= 0.65 "
                "(well above 0.5). Qwen-1.5B reference is 0.7731 at L19.",
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nf2_generalizes={out['verdict']['f2_generalizes']}  "
          f"(min peak {min(aus):.4f}, max peak {max(aus):.4f})")
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
