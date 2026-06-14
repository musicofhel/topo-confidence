"""P11-FE390 — Self-Sim label-free monitor floor on cached prefill NPZs.

Self-Sim baseline: on each OOF fold, compute the mean L19 prefill activation
over K=1-correct training problems and the mean over K=1-incorrect training
problems, then score every holdout problem by its cosine similarity to each
mean. The monitor score is cos(x, mean_correct) - cos(x, mean_incorrect).
Report OOF AUROC against ground-truth K=1 correctness plus precision (accuracy
of the predicted-correct set) at multiple coverages. Run for Qwen2.5-1.5B
(required cache) and Qwen2.5-7B (optional cache, skipped if absent).

Rationale: establishes the trivial mean-cosine floor below the supervised DoM
(0.7731 on 1.5B). High Self-Sim AUROC => low TELLME headroom; low Self-Sim
AUROC => supervised DoM is doing real work and TELLME has room to lift.
No new extraction.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE_15B = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
CACHE_7B = ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/selfsim_monitor/results.json"

N_FOLDS = 5
SEED = 9999
COVERAGES = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
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


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def selfsim_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Out-of-fold Self-Sim score: cos(x, mean_correct) - cos(x, mean_incorrect)."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    Xn = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    for test_idx in stratified_kfold(y, N_FOLDS, SEED):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        ytr = y[train_mask]
        if ytr.all() or (~ytr).all():
            continue
        mean_pos = _unit(X[train_mask][ytr].mean(axis=0))
        mean_neg = _unit(X[train_mask][~ytr].mean(axis=0))
        scores[test_idx] = Xn[test_idx] @ mean_pos - Xn[test_idx] @ mean_neg
    return scores


def coverage_accuracy(scores: np.ndarray, y: np.ndarray) -> dict:
    """Precision of the top-coverage predicted-correct set, by descending score."""
    order = np.argsort(-scores)
    y_sorted = y[order].astype(np.float64)
    n = len(y)
    out = {}
    for cov in COVERAGES:
        k = max(1, int(round(cov * n)))
        out[f"{cov:.2f}"] = float(y_sorted[:k].mean())
    return out


def evaluate(cache_path: Path) -> dict | None:
    if not cache_path.exists():
        return None
    blob = np.load(cache_path)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    scores = selfsim_oof(X, y)
    return {
        "n": int(len(y)),
        "n_correct": int(y.sum()),
        "base_rate": float(y.mean()),
        "auroc_selfsim_oof": float(auroc(scores, y)),
        "accuracy_at_coverage": coverage_accuracy(scores, y),
    }


def main() -> int:
    if not CACHE_15B.exists():
        print("MISSING_REGEN_INPUT", CACHE_15B, file=sys.stderr)
        return 2

    res_15b = evaluate(CACHE_15B)
    res_7b = evaluate(CACHE_7B)

    out = {
        "experiment": "P11-FE390",
        "description": "Self-Sim label-free monitor floor (OOF cosine to class means).",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "qwen2_5_1_5b": res_15b,
        "qwen2_5_7b": res_7b if res_7b is not None else "MISSING_CACHE",
        "supervised_dom_auroc_1_5b": 0.7731,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("auroc_selfsim_1.5b =", res_15b["auroc_selfsim_oof"])
    if res_7b is not None:
        print("auroc_selfsim_7b  =", res_7b["auroc_selfsim_oof"])
    else:
        print("7B cache absent:", CACHE_7B)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())