"""FE740 — Length-balanced resampling control on prefill DoM AUROC.

F-2's headline prefill L19 DoM AUROC of 0.7731 could in principle be driven by
sequence length acting as a correctness proxy (longer / shorter solutions
correlate with both DoM projection and ground-truth correctness). This control
replicates Sun et al.'s length-balance protocol: bin MATH-500 by sequence
length (a step-count proxy), then for each length bin subsample the majority
class down to the minority count so the length distribution is matched between
the correct and incorrect classes. On each length-balanced subset we recompute
the OOF 5-fold prefill DoM AUROC and aggregate across many resamples. Sun et
al. observe their trajectory-feature AUC drop only from 0.852 to 0.847 under
the same protocol — a small length contribution but preserved signal. We report
the mean / std balanced AUROC and the delta from the unbalanced 0.7731 baseline.
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
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/length_balance/results.json"

N_FOLDS = 5
N_BINS = 10
N_RESAMPLES = 200
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


def oof_dom_auroc(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    """OOF 5-fold supervised DoM (mean-difference direction) AUROC."""
    if y.sum() < k or (~y).sum() < k:
        return float("nan")
    n = len(y)
    folds = stratified_kfold(y, k, seed)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        if ytr.sum() == 0 or (~ytr).sum() == 0:
            return float("nan")
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d_vec
    return auroc(scores, y)


def length_balanced_indices(length: np.ndarray, y: np.ndarray,
                            bin_edges: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per length-bin, subsample the majority class to the minority class count."""
    bin_id = np.clip(np.digitize(length, bin_edges[1:-1]), 0, len(bin_edges) - 2)
    keep = []
    for b in range(len(bin_edges) - 1):
        in_bin = np.flatnonzero(bin_id == b)
        if len(in_bin) == 0:
            continue
        pos = in_bin[y[in_bin]]
        neg = in_bin[~y[in_bin]]
        m = min(len(pos), len(neg))
        if m == 0:
            continue
        keep.append(rng.choice(pos, size=m, replace=False))
        keep.append(rng.choice(neg, size=m, replace=False))
    if not keep:
        return np.array([], dtype=int)
    return np.concatenate(keep)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    seq_len = blob["seq_len"].astype(np.float64)
    assert X.shape == (500, 1536) and y.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    # Unbalanced baselines (refit OOF DoM + the cached prefill_score) on the full set.
    auroc_full_oof = oof_dom_auroc(X, y, N_FOLDS, SEED)
    auroc_cached_full = auroc(dom_score, y)

    # Quantile bin edges over sequence length (step-count proxy).
    qs = np.linspace(0.0, 1.0, N_BINS + 1)
    bin_edges = np.quantile(seq_len, qs)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    rng = np.random.default_rng(SEED)
    balanced_oof = []
    balanced_cached = []
    subset_sizes = []
    for _ in range(N_RESAMPLES):
        idx = length_balanced_indices(seq_len, y, bin_edges, rng)
        if len(idx) < 2 * N_FOLDS:
            continue
        subset_sizes.append(int(len(idx)))
        a_oof = oof_dom_auroc(X[idx], y[idx], N_FOLDS, int(rng.integers(1, 2**31 - 1)))
        if not np.isnan(a_oof):
            balanced_oof.append(a_oof)
        balanced_cached.append(auroc(dom_score[idx], y[idx]))

    balanced_oof = np.asarray(balanced_oof, dtype=np.float64)
    balanced_cached = np.asarray(balanced_cached, dtype=np.float64)

    # Mean step-count by class, before/after balancing (sanity that imbalance was removed).
    mean_len_pos_full = float(seq_len[y].mean())
    mean_len_neg_full = float(seq_len[~y].mean())

    out = {
        "experiment": "FE740",
        "n_resamples": int(N_RESAMPLES),
        "n_bins": int(N_BINS),
        "n_folds": int(N_FOLDS),
        "auroc_full_oof_dom": float(auroc_full_oof),
        "auroc_full_cached_dom": float(auroc_cached_full),
        "auroc_balanced_oof_dom_mean": float(balanced_oof.mean()) if balanced_oof.size else float("nan"),
        "auroc_balanced_oof_dom_std": float(balanced_oof.std(ddof=1)) if balanced_oof.size > 1 else float("nan"),
        "auroc_balanced_cached_dom_mean": float(balanced_cached.mean()) if balanced_cached.size else float("nan"),
        "auroc_balanced_cached_dom_std": float(balanced_cached.std(ddof=1)) if balanced_cached.size > 1 else float("nan"),
        "delta_oof_vs_full": (float(balanced_oof.mean() - auroc_full_oof)
                              if balanced_oof.size else float("nan")),
        "mean_subset_size": float(np.mean(subset_sizes)) if subset_sizes else float("nan"),
        "mean_seq_len_correct": mean_len_pos_full,
        "mean_seq_len_incorrect": mean_len_neg_full,
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())