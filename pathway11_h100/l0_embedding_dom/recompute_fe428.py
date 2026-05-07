"""FE428 — L0 embedding-layer DoM baseline.

Extracts prefill-position activations at layer 0 (embedding output) and
computes OOF DoM AUROC. If L0 AUROC > 0.6, F-2's L19 specificity is
partly inherited from input geometry.

Output: pathway11_h100/results/fe428_l0_embedding_dom.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
DATA_DIR = ROOT / "pathway8_layerwise/data/math500"
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe428_l0_embedding_dom.json"

N_PROBLEMS = 500
HIDDEN_DIM = 1536
SEED = 9999
N_FOLDS = 5


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def participation_ratio(X: np.ndarray) -> float:
    X_c = X - X.mean(axis=0)
    G = X_c @ X_c.T
    eigvals = np.linalg.eigvalsh(G)
    eigvals = np.maximum(eigvals, 0)
    sum_sq = (eigvals ** 2).sum()
    if sum_sq < 1e-30:
        return 0.0
    return float(eigvals.sum() ** 2 / sum_sq)


def main() -> int:
    print("Loading L0 prefill activations from per-problem NPZs...")
    X_L0 = np.zeros((N_PROBLEMS, HIDDEN_DIM), dtype=np.float32)
    correct = np.zeros(N_PROBLEMS, dtype=bool)

    for i in range(N_PROBLEMS):
        fp = DATA_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            raise SystemExit(f"missing: {fp}")
        with np.load(fp, allow_pickle=True) as d:
            X_L0[i] = d["states"][0, 0, :].astype(np.float32)  # L0, prefill pos
            correct[i] = bool(d["correct"])
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{N_PROBLEMS}")

    print(f"Correct: {correct.sum()}/{N_PROBLEMS}")

    # L19 DoM from convenience cache (for cosine comparison)
    cache = np.load(CACHE)
    X_L19 = cache["prefill"].astype(np.float32)
    X_L19_c = X_L19 - X_L19.mean(axis=0)
    dom_L19 = X_L19_c[correct].mean(0) - X_L19_c[~correct].mean(0)
    dom_L19 = dom_L19 / (np.linalg.norm(dom_L19) + 1e-30)

    # L0 DoM (full sample)
    X_L0_c = X_L0 - X_L0.mean(axis=0)
    dom_L0 = X_L0_c[correct].mean(0) - X_L0_c[~correct].mean(0)
    dom_L0 = dom_L0 / (np.linalg.norm(dom_L0) + 1e-30)

    cos_L0_L19 = float(dom_L0 @ dom_L19)
    print(f"cos(DoM_L0, DoM_L19) = {cos_L0_L19:.4f}")

    # OOF DoM AUROC at L0
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_scores = np.zeros(N_PROBLEMS, dtype=np.float64)

    for train_idx, test_idx in skf.split(X_L0, correct):
        X_train = X_L0[train_idx]
        y_train = correct[train_idx]
        X_train_c = X_train - X_train.mean(axis=0)
        dom = X_train_c[y_train].mean(0) - X_train_c[~y_train].mean(0)
        dom = dom / (np.linalg.norm(dom) + 1e-30)
        X_test_c = X_L0[test_idx] - X_train.mean(axis=0)
        oof_scores[test_idx] = X_test_c @ dom

    auroc_L0 = auroc(oof_scores, correct)
    print(f"L0 OOF DoM AUROC = {auroc_L0:.4f}")

    # L0 PR
    pr_L0 = participation_ratio(X_L0)
    print(f"L0 PR = {pr_L0:.2f}")

    out = {
        "experiment": "FE428",
        "description": "L0 embedding-layer DoM baseline (deconfounding control for F-2)",
        "n": N_PROBLEMS,
        "n_correct": int(correct.sum()),
        "auroc_L0_dom_oof": float(auroc_L0),
        "auroc_L19_dom_reference": 0.7731,
        "cos_dom_L0_L19": cos_L0_L19,
        "pr_L0": pr_L0,
        "pr_L19_reference": 19.86,
        "interpretation": (
            "L0 AUROC near 0.5 confirms F-2's L19 specificity is NOT inherited "
            "from input geometry. L0 AUROC > 0.6 would suggest the embedding "
            "already separates correct/incorrect, weakening L19's unique role."
        ),
        "fold_structure": "StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)",
        "meta": {
            "data_dir": str(DATA_DIR),
            "cache": str(CACHE),
            "seed": SEED,
            "method": "oof_dom_at_layer_0",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
