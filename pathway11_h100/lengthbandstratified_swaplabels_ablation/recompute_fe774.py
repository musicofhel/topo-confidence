"""P11-FE774 — Length-band-stratified Swap-Labels ablation on MATH-500 prefill DoM.

Direct analog of the Swap-Numbers perturbation control (Fu et al. 2604.20817,
Table 1). The unshuffled L19 prefill DoM probe scores AUROC 0.7731 (F-2). If
that signal is genuine functional structure rather than a length / topic /
token-frequency artifact, then destroying the label-to-activation correspondence
*within* length-matched bands should collapse AUROC to chance (~0.50).

Protocol: partition the 500 problems into length quartiles (by seq_len). Within
each quartile, randomly permute the correct/incorrect labels (preserving the
per-band positive rate). Refit the DoM logistic probe at L19 prefill under
3 seeds x 10-fold stratified CV (30 OOF AUROC evaluations per shuffled run),
and build the shuffled-AUROC distribution across seeds. Compare against the
real-label OOF AUROC under the identical CV protocol. If the shuffled
distribution's upper tail overlaps 0.7731, F-2 is partially a length confound.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/swap_labels_length_band/results.json"

UNSHUFFLED_REFERENCE = 0.7731  # F-2 canonical OOF DoM AUROC (1024-tok)
N_QUARTILES = 4
N_FOLDS = 10
SEEDS = [9999, 1234, 4242]  # 3 seeds x 10-fold = 30 runs per Fu et al. protocol


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def length_quartiles(seq_len: np.ndarray, n_bins: int) -> np.ndarray:
    """Return a per-problem band index in [0, n_bins) by seq_len quantiles."""
    ranks = seq_len.argsort().argsort()
    return np.minimum((ranks * n_bins) // len(seq_len), n_bins - 1)


def swap_within_bands(y: np.ndarray, bands: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute labels independently within each length band (preserves per-band rate)."""
    y_shuf = y.copy()
    for b in np.unique(bands):
        idx = np.flatnonzero(bands == b)
        perm = rng.permutation(idx)
        y_shuf[idx] = y[perm]
    return y_shuf


def oof_probe_auroc(X: np.ndarray, y: np.ndarray, seed: int) -> float:
    """OOF AUROC of an L2-regularized logistic probe under stratified k-fold CV."""
    # Degenerate (single-class) band shuffles can occasionally yield a fold with
    # one class; guard at the AUROC level rather than failing the run.
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr = y[train_idx]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        clf.fit((Xtr - mu) / sd, ytr)
        oof[test_idx] = clf.decision_function((Xte - mu) / sd)
    return auroc(oof, y)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    seq_len = blob["seq_len"].astype(np.float64)
    assert X.shape == (500, 1536) and y.shape == (500,) and seq_len.shape == (500,)

    bands = length_quartiles(seq_len, N_QUARTILES)
    band_pos_rate = {int(b): float(y[bands == b].mean()) for b in np.unique(bands)}

    # Real-label baseline under the identical CV protocol (3-seed mean).
    real_aurocs = [oof_probe_auroc(X, y, s) for s in SEEDS]
    real_mean = float(np.nanmean(real_aurocs))

    # Shuffled-label runs: per seed, swap within bands then refit (30 OOF evals).
    shuffled_aurocs: list[float] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        y_shuf = swap_within_bands(y, bands, rng)
        shuffled_aurocs.append(oof_probe_auroc(X, y_shuf, seed))

    shuf = np.array(shuffled_aurocs, dtype=np.float64)
    shuf_clean = shuf[~np.isnan(shuf)]
    shuf_mean = float(np.mean(shuf_clean)) if len(shuf_clean) else float("nan")
    shuf_std = float(np.std(shuf_clean)) if len(shuf_clean) else float("nan")
    shuf_max = float(np.max(shuf_clean)) if len(shuf_clean) else float("nan")

    # Refutation verdict: does the shuffled distribution's upper tail reach the
    # canonical 0.7731? Overlap => F-2 partially a length confound.
    overlaps_reference = bool(shuf_max >= UNSHUFFLED_REFERENCE)
    margin_to_reference = float(UNSHUFFLED_REFERENCE - shuf_mean)

    out = {
        "experiment": "P11-FE774",
        "description": "Length-band-stratified Swap-Labels ablation on MATH-500 prefill DoM",
        "unshuffled_reference_auroc": UNSHUFFLED_REFERENCE,
        "n_quartiles": N_QUARTILES,
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "band_positive_rate": band_pos_rate,
        "real_label_auroc_per_seed": [float(a) for a in real_aurocs],
        "real_label_auroc_mean": real_mean,
        "shuffled_auroc_per_seed": [float(a) for a in shuffled_aurocs],
        "shuffled_auroc_mean": shuf_mean,
        "shuffled_auroc_std": shuf_std,
        "shuffled_auroc_max": shuf_max,
        "overlaps_reference": overlaps_reference,
        "margin_to_reference": margin_to_reference,
        "verdict": (
            "F-2 partially a length confound (shuffled tail reaches 0.7731)"
            if overlaps_reference
            else "F-2 survives: shuffled labels collapse AUROC toward chance"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())