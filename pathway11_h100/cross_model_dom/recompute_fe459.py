"""FE459 — Cross-model 1.5B↔7B DoM score correlation.

Computes 7B L19 prefill DoM OOF scores and correlates with 1.5B OOF scores.
Tests whether "difficulty geometry" is consistent across model sizes despite
different capabilities (1.5B: 48.6%, 7B: 73.2%).

Output: pathway11_h100/results/fe459_cross_model_dom.json
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
from scipy.stats import spearmanr, kendalltau

ROOT = Path("/home/musicofhel/topo-confidence")
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
DATA_7B = ROOT / "pathway11_h100/data/math500_7b"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe459_cross_model_dom.json"

N_PROBLEMS = 500
HIDDEN_DIM_7B = 3584
SEED = 9999
N_FOLDS = 5


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def main() -> int:
    # 1.5B OOF scores (already computed)
    dom_data = np.load(DOM_NPZ)
    scores_15b = dom_data["prefill_score"].astype(np.float64)
    correct_15b = dom_data["correct_k1"].astype(bool)
    print(f"1.5B: {correct_15b.sum()}/{N_PROBLEMS} correct")

    # Load 7B L19 prefill activations (one at a time to save memory)
    print("Loading 7B L19 prefill activations (500 files, sequential)...")
    X_7b = np.zeros((N_PROBLEMS, HIDDEN_DIM_7B), dtype=np.float32)
    correct_7b = np.zeros(N_PROBLEMS, dtype=bool)

    for i in range(N_PROBLEMS):
        fp = DATA_7B / f"problem_{i:03d}.npz"
        if not fp.exists():
            raise SystemExit(f"missing: {fp}")
        with np.load(fp, allow_pickle=True) as d:
            s = d["states"]  # (29, T, 3584)
            X_7b[i] = s[19, 0, :].astype(np.float32)
            correct_7b[i] = bool(d["correct"])
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{N_PROBLEMS}")

    print(f"7B: {correct_7b.sum()}/{N_PROBLEMS} correct (expected 366)")

    # 7B OOF DoM
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores_7b = np.zeros(N_PROBLEMS, dtype=np.float64)

    for train_idx, test_idx in skf.split(X_7b, correct_7b):
        X_train = X_7b[train_idx]
        y_train = correct_7b[train_idx]
        mu = X_train.mean(axis=0)
        X_train_c = X_train - mu
        dom = X_train_c[y_train].mean(0) - X_train_c[~y_train].mean(0)
        dom = dom / (np.linalg.norm(dom) + 1e-30)
        scores_7b[test_idx] = (X_7b[test_idx] - mu) @ dom

    auroc_7b = auroc(scores_7b, correct_7b)
    auroc_15b = auroc(scores_15b, correct_15b)
    print(f"\n1.5B OOF DoM AUROC = {auroc_15b:.4f} (reference: 0.7731)")
    print(f"7B OOF DoM AUROC   = {auroc_7b:.4f}")

    # Cross-model correlation
    rho_spearman, p_spearman = spearmanr(scores_15b, scores_7b)
    tau_kendall, p_kendall = kendalltau(scores_15b, scores_7b)

    # Concordance: fraction of problem pairs where both models agree on ordering
    concordant = 0
    discordant = 0
    for i in range(N_PROBLEMS):
        for j in range(i + 1, N_PROBLEMS):
            s1 = (scores_15b[i] - scores_15b[j]) * (scores_7b[i] - scores_7b[j])
            if s1 > 0:
                concordant += 1
            elif s1 < 0:
                discordant += 1
    total_pairs = concordant + discordant
    concordance = concordant / total_pairs if total_pairs > 0 else 0.0

    print(f"\nSpearman(1.5B, 7B) = {rho_spearman:.4f} (p={p_spearman:.2e})")
    print(f"Kendall tau        = {tau_kendall:.4f} (p={p_kendall:.2e})")
    print(f"Concordance        = {concordance:.4f}")

    # Agreement on correctness labels
    both_correct = (correct_15b & correct_7b).sum()
    both_incorrect = (~correct_15b & ~correct_7b).sum()
    only_15b_correct = (correct_15b & ~correct_7b).sum()
    only_7b_correct = (~correct_15b & correct_7b).sum()

    print(f"\nLabel agreement:")
    print(f"  Both correct:      {both_correct}")
    print(f"  Both incorrect:    {both_incorrect}")
    print(f"  Only 1.5B correct: {only_15b_correct}")
    print(f"  Only 7B correct:   {only_7b_correct}")

    out = {
        "experiment": "FE459",
        "description": "Cross-model 1.5B↔7B DoM score correlation at L19 prefill",
        "n": N_PROBLEMS,
        "model_15b": {
            "n_correct": int(correct_15b.sum()),
            "auroc_oof": float(auroc_15b),
            "auroc_reference": 0.7731,
        },
        "model_7b": {
            "n_correct": int(correct_7b.sum()),
            "auroc_oof": float(auroc_7b),
            "hidden_dim": HIDDEN_DIM_7B,
        },
        "cross_model": {
            "spearman_rho": float(rho_spearman),
            "spearman_p": float(p_spearman),
            "kendall_tau": float(tau_kendall),
            "kendall_p": float(p_kendall),
            "concordance_fraction": float(concordance),
        },
        "label_agreement": {
            "both_correct": int(both_correct),
            "both_incorrect": int(both_incorrect),
            "only_15b_correct": int(only_15b_correct),
            "only_7b_correct": int(only_7b_correct),
        },
        "cross_check": {
            "n_correct_7b_is_366": int(correct_7b.sum()) == 366,
        },
        "interpretation": (
            "High Spearman (>0.5) means both models rank problem difficulty "
            "similarly in activation space, despite different capabilities. "
            "This would support F-2 as a general geometric phenomenon, not "
            "1.5B-specific."
        ),
        "fold_structure": "StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)",
        "meta": {
            "dom_npz": str(DOM_NPZ),
            "data_7b": str(DATA_7B),
            "seed": SEED,
            "method": "oof_dom_cross_model_spearman",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
