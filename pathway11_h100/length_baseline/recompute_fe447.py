"""FE447 — Length-as-correctness baseline + OOF residualized DoM AUROC.

Existing in-sample result: length AUROC=0.7986, residualized DoM=0.6647.
New contribution: OOF residualization via 5-fold + Spearman(dom, seq_len).

Output: pathway11_h100/results/fe447_length_baseline.json
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
from scipy.stats import spearmanr

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe447_length_baseline.json"

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
    cache = np.load(CACHE)
    seq_len = cache["seq_len"].astype(np.float64)
    correct = cache["correct"].astype(bool)
    n = len(correct)

    dom_data = np.load(DOM_NPZ)
    dom_score = dom_data["prefill_score"].astype(np.float64)

    # Raw length AUROCs
    auroc_length_pos = auroc(seq_len, correct)
    auroc_length_neg = auroc(-seq_len, correct)
    print(f"AUROC(seq_len, correct)  = {auroc_length_pos:.4f}")
    print(f"AUROC(-seq_len, correct) = {auroc_length_neg:.4f}")
    print(f"  Expected: 0.7986 (for -seq_len, i.e. shorter = more correct)")

    # Spearman(dom, seq_len)
    rho_dom_len, p_dom_len = spearmanr(dom_score, seq_len)
    print(f"Spearman(dom, seq_len) = {rho_dom_len:.4f} (p={p_dom_len:.2e})")

    # Raw DoM AUROC (cross-check)
    auroc_dom_raw = auroc(dom_score, correct)
    print(f"AUROC(dom_score, correct) = {auroc_dom_raw:.4f}")

    # OOF residualized DoM AUROC
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_residual = np.zeros(n, dtype=np.float64)

    for train_idx, test_idx in skf.split(seq_len, correct):
        sl_train = seq_len[train_idx]
        dom_train = dom_score[train_idx]
        sl_test = seq_len[test_idx]
        dom_test = dom_score[test_idx]

        # OLS: dom ~ a*seq_len + b
        A = np.column_stack([sl_train, np.ones(len(sl_train))])
        coeffs, _, _, _ = np.linalg.lstsq(A, dom_train, rcond=None)
        predicted_test = sl_test * coeffs[0] + coeffs[1]
        oof_residual[test_idx] = dom_test - predicted_test

    auroc_oof_residual = auroc(oof_residual, correct)
    print(f"\nOOF residualized DoM AUROC = {auroc_oof_residual:.4f}")
    print(f"  In-sample reference:       0.6647")
    print(f"  Raw DoM reference:         0.7731")

    # Per-class length stats
    len_correct = seq_len[correct]
    len_incorrect = seq_len[~correct]

    out = {
        "experiment": "FE447",
        "description": "Length-as-correctness baseline + OOF residualized DoM AUROC",
        "n": n,
        "n_correct": int(correct.sum()),
        "n_incorrect": int((~correct).sum()),
        "auroc_length_positive": float(auroc_length_pos),
        "auroc_length_negative": float(auroc_length_neg),
        "auroc_dom_raw": float(auroc_dom_raw),
        "auroc_oof_residualized_dom": float(auroc_oof_residual),
        "auroc_insample_residualized_reference": 0.6647,
        "spearman_dom_seqlen": float(rho_dom_len),
        "spearman_dom_seqlen_pvalue": float(p_dom_len),
        "length_stats": {
            "correct_mean": float(len_correct.mean()),
            "correct_median": float(np.median(len_correct)),
            "incorrect_mean": float(len_incorrect.mean()),
            "incorrect_median": float(np.median(len_incorrect)),
        },
        "cross_check": {
            "expected_length_auroc": 0.7986,
            "matches": abs(auroc_length_neg - 0.7986) < 0.001,
        },
        "interpretation": (
            "OOF residualized AUROC isolates DoM signal that is NOT explained by "
            "sequence length. If OOF << 0.6647, in-sample residualization was "
            "optimistic. The gap (raw - residualized) quantifies length confounding."
        ),
        "fold_structure": "StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)",
        "meta": {
            "cache": str(CACHE),
            "dom_npz": str(DOM_NPZ),
            "seed": SEED,
            "method": "oof_ols_residualization",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
