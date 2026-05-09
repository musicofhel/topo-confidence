"""System prompt for recompute script generation."""

RECOMPUTE_SYSTEM = r"""You write recompute scripts for topo-confidence experiments on cached NPZ activations.

Every script follows an identical pattern. Your output must be a SINGLE Python file that:
1. Has a module docstring describing the experiment
2. Imports only: json, sys, os, pathlib.Path, numpy. scipy and sklearn are allowed if needed.
3. Defines ROOT, CACHE (or domain-specific NPZ paths), and OUT_JSON constants
4. Includes an auroc() helper and optionally stratified_kfold()
5. Has a main() function that returns 0 on success, nonzero on failure
6. Writes results to OUT_JSON as JSON
7. Ends with: if __name__ == "__main__": raise SystemExit(main())

HARD CONSTRAINTS:
- NO network access (no requests, urllib, httpx)
- NO GPU (no torch, tensorflow, jax, transformers, accelerate)
- CPU only. Set OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS to 4 if using scipy/sklearn.
- All paths must be under /home/musicofhel/topo-confidence/
- Single file, no external dependencies beyond numpy/scipy/sklearn
- Must handle missing cache files gracefully (print MISSING_REGEN_INPUT, return 2)

NPZ SCHEMA:
- Main cache: pathway11_h100/prefill_inversion/cache/m15b_prefill.npz
  - prefill: (500, 1536) float32 — L19 prefill hidden states
  - correct: (500,) bool — ground truth correctness labels
  - seq_len: (500,) int — sequence lengths
- DoM scores: pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz
  - prefill_score: (500,) float64 — DoM projection scores
- K=8 cache: pathway11_h100/data/k8_selfconsistency/problem_NNN.npz (500 files)
  - correct: (8,) bool — per-sample correctness across 8 generations

Output ONLY the Python code. No markdown fences, no explanation.
"""

EXEMPLAR_FE101 = r'''"""FE101 — LEACE linear-erasure null test for F-2.

Per-fold LEACE: fit μ, Σ, DoM on train; apply to test; refit DoM on erased
train fold; score on erased test fold; aggregate AUROC over 5 OOF folds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/leace_erasure/results.json"

N_FOLDS = 5
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def fit_leace(X_train, y_train, alpha_rel=1e-3):
    mu = X_train.mean(axis=0)
    Xc = X_train - mu
    n, d = Xc.shape
    Sigma = Xc.T @ Xc / max(n - 1, 1)
    trace = float(np.trace(Sigma))
    alpha = alpha_rel * trace / d
    Sigma_ridge = Sigma + alpha * np.eye(d, dtype=Xc.dtype)
    mu_pos = X_train[y_train].mean(axis=0)
    mu_neg = X_train[~y_train].mean(axis=0)
    d_vec = mu_pos - mu_neg
    w = np.linalg.solve(Sigma_ridge, d_vec)
    return mu, d_vec, w, alpha


def apply_leace(X, mu, d_vec, w):
    denom = float(d_vec @ w)
    if abs(denom) < 1e-12:
        return X.copy()
    coeff = ((X - mu) @ w) / denom
    return X - np.outer(coeff, d_vec)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    folds = stratified_kfold(y, N_FOLDS, SEED)
    n = len(y)
    raw_scores = np.zeros(n, dtype=np.float64)
    erased_scores = np.zeros(n, dtype=np.float64)

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr, yte = y[train_mask], y[test_idx]
        d_raw = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        raw_scores[test_idx] = Xte @ d_raw
        mu, d_vec, w, alpha = fit_leace(Xtr, ytr)
        Xte_e = apply_leace(Xte, mu, d_vec, w)
        d_erased = apply_leace(Xtr, mu, d_vec, w)[ytr].mean(0) - apply_leace(Xtr, mu, d_vec, w)[~ytr].mean(0)
        erased_scores[test_idx] = Xte_e @ d_erased

    out = {
        "auroc_raw_oof": float(auroc(raw_scores, y)),
        "auroc_erased_oof": float(auroc(erased_scores, y)),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

EXEMPLAR_FE308 = r'''"""FE308 — Adaptive best-of-k Damani allocator.

Uses DoM scores as per-problem confidence to allocate variable K across
problems. Compares adaptive majority-vote accuracy against uniform-K baselines.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from scipy.stats import rankdata

ROOT = Path("/home/musicofhel/topo-confidence")
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/results/fe308_adaptive_bestofk.json"

SEED = 9999


def auroc(scores, labels):
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def main() -> int:
    dom_data = np.load(DOM_NPZ)
    dom_score = dom_data["prefill_score"].astype(np.float64)
    lambda_hat = (rankdata(dom_score) - 1) / 499
    # ... allocator logic, majority vote, write results ...
    out = {"experiment": "FE308", "auroc_dom_raw": float(auroc(dom_score, np.load(ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz")["correct"]))}
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

EXEMPLAR_FE447 = r'''"""FE447 — Length-as-correctness baseline + OOF residualized DoM AUROC."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/results/fe447_length_baseline.json"

SEED = 9999
N_FOLDS = 5


def auroc(scores, labels):
    pos = scores[labels]; neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def main() -> int:
    cache = np.load(CACHE)
    seq_len = cache["seq_len"].astype(np.float64)
    correct = cache["correct"].astype(bool)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_residual = np.zeros(len(correct), dtype=np.float64)
    for train_idx, test_idx in skf.split(seq_len, correct):
        A = np.column_stack([seq_len[train_idx], np.ones(len(train_idx))])
        coeffs, _, _, _ = np.linalg.lstsq(A, dom_score[train_idx], rcond=None)
        oof_residual[test_idx] = dom_score[test_idx] - (seq_len[test_idx] * coeffs[0] + coeffs[1])

    out = {"auroc_oof_residualized_dom": float(auroc(oof_residual, correct))}
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
