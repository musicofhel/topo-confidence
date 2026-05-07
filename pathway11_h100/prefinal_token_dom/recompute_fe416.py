"""FE416 — Pre-final token DoM at L19 (position -2).

Tests whether F-3's orthogonality (cos(prefill_DoM, final_DoM)=0.046) is
a positional artifact of the answer token. If cos(prefinal_DoM, prefill_DoM)
> 0.3, the DoM direction rotates specifically at the LAST token.

Output: pathway11_h100/results/fe416_prefinal_token_dom.json
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
RESULTS_DIR = ROOT / "pathway11_h100/results"
OUT_JSON = RESULTS_DIR / "fe416_prefinal_token_dom.json"

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


def oof_dom_auroc(X: np.ndarray, y: np.ndarray) -> float:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n = len(y)
    oof_scores = np.zeros(n, dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        y_train = y[train_idx]
        mu = X_train.mean(axis=0)
        X_train_c = X_train - mu
        dom = X_train_c[y_train].mean(0) - X_train_c[~y_train].mean(0)
        dom = dom / (np.linalg.norm(dom) + 1e-30)
        oof_scores[test_idx] = (X[test_idx] - mu) @ dom
    return float(auroc(oof_scores, y))


def main() -> int:
    print("Loading L19 activations at positions 0, -2, -1 from per-problem NPZs...")
    X_prefill = np.zeros((N_PROBLEMS, HIDDEN_DIM), dtype=np.float32)
    X_prefinal = np.zeros((N_PROBLEMS, HIDDEN_DIM), dtype=np.float32)
    X_final = np.zeros((N_PROBLEMS, HIDDEN_DIM), dtype=np.float32)
    correct = np.zeros(N_PROBLEMS, dtype=bool)

    for i in range(N_PROBLEMS):
        fp = DATA_DIR / f"problem_{i:03d}.npz"
        if not fp.exists():
            raise SystemExit(f"missing: {fp}")
        with np.load(fp, allow_pickle=True) as d:
            s = d["states"]  # (29, T, 1536)
            X_prefill[i] = s[19, 0, :].astype(np.float32)
            X_prefinal[i] = s[19, -2, :].astype(np.float32)
            X_final[i] = s[19, -1, :].astype(np.float32)
            correct[i] = bool(d["correct"])
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{N_PROBLEMS}")

    print(f"Correct: {correct.sum()}/{N_PROBLEMS}")

    # Full-sample DoM directions (centered)
    def get_dom(X):
        X_c = X - X.mean(axis=0)
        d = X_c[correct].mean(0) - X_c[~correct].mean(0)
        return d / (np.linalg.norm(d) + 1e-30)

    dom_prefill = get_dom(X_prefill)
    dom_prefinal = get_dom(X_prefinal)
    dom_final = get_dom(X_final)

    cos_prefinal_prefill = float(dom_prefinal @ dom_prefill)
    cos_prefinal_final = float(dom_prefinal @ dom_final)
    cos_prefill_final = float(dom_prefill @ dom_final)

    print(f"\nCosine similarities:")
    print(f"  cos(prefinal, prefill) = {cos_prefinal_prefill:.4f}  (>0.3 = positional artifact)")
    print(f"  cos(prefinal, final)   = {cos_prefinal_final:.4f}")
    print(f"  cos(prefill, final)    = {cos_prefill_final:.4f}  (reference: 0.046)")

    # OOF DoM AUROCs
    print("\nComputing OOF DoM AUROCs...")
    auroc_prefill = oof_dom_auroc(X_prefill, correct)
    auroc_prefinal = oof_dom_auroc(X_prefinal, correct)
    auroc_final = oof_dom_auroc(X_final, correct)

    print(f"  prefill (pos 0):  AUROC = {auroc_prefill:.4f}")
    print(f"  prefinal (pos -2): AUROC = {auroc_prefinal:.4f}")
    print(f"  final (pos -1):   AUROC = {auroc_final:.4f}")

    out = {
        "experiment": "FE416",
        "description": "Pre-final token DoM at L19 (position -2) — F-3 positional artifact test",
        "n": N_PROBLEMS,
        "n_correct": int(correct.sum()),
        "cosines": {
            "prefinal_vs_prefill": cos_prefinal_prefill,
            "prefinal_vs_final": cos_prefinal_final,
            "prefill_vs_final": cos_prefill_final,
            "prefill_vs_final_reference": 0.046,
        },
        "aurocs": {
            "prefill_pos0": float(auroc_prefill),
            "prefinal_pos_minus2": float(auroc_prefinal),
            "final_pos_minus1": float(auroc_final),
            "prefill_reference": 0.7731,
            "final_reference": 0.7186,
        },
        "f3_test": {
            "cos_prefinal_prefill_above_0_3": cos_prefinal_prefill > 0.3,
            "verdict": (
                "positional artifact" if cos_prefinal_prefill > 0.3
                else "NOT a positional artifact — orthogonality is gradual"
            ),
        },
        "interpretation": (
            "If cos(prefinal_DoM, prefill_DoM) > 0.3, the DoM direction rotates "
            "specifically at the last token (answer token), making F-3 orthogonality "
            "a positional artifact. If < 0.3, the rotation is gradual."
        ),
        "fold_structure": "StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)",
        "meta": {
            "data_dir": str(DATA_DIR),
            "seed": SEED,
            "method": "oof_dom_at_prefinal_position",
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
