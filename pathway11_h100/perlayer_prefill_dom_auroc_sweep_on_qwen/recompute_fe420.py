"""FE420 — Per-layer prefill DoM/logistic AUROC sweep on Qwen-2.5-1.5B.

Trains a probe at each transformer layer using cached per-layer prefill
activations and 5-fold stratified OOF on MATH-500 correctness, then reports
AUROC vs depth-fraction. Two readouts per layer: a difference-of-means (DoM)
projection and an L2-regularized logistic probe, both scored out-of-fold.

Refutation test for F-2's L19 anchor: if AUROC peaks before L19, the
L19-specificity is a property of the grounding plateau (cf. TinySQL two-phase
prediction) rather than of layer 19 itself. The script locates the peak layer
and flags whether it falls at/after L19.
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
# L19 single-layer cache — used only for the correctness labels (canonical).
LABEL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# Candidate locations for the full 28-layer prefill activation cache.
LAYERWISE_NPZ_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layerwise.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz",
    ROOT / "pathway11_h100/layerwise_logitlens_trajectory_on_qwen25/m15b_prefill_layerwise.npz",
    ROOT / "pathway11_h100/layerwise_dom_sweep/cache/m15b_prefill_layerwise.npz",
]
# Candidate directories holding per-layer files (layer_NN.npz with "prefill").
LAYERWISE_DIR_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/layerwise",
    ROOT / "pathway11_h100/layerwise_logitlens_trajectory_on_qwen25/cache",
    ROOT / "pathway11_h100/layerwise_dom_sweep/cache",
]
OUT_JSON = ROOT / "pathway11_h100/layerwise_dom_sweep/results.json"

N_FOLDS = 5
SEED = 9999
L19_INDEX = 19  # the anchored layer under test


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


def _stack_from_npz(blob) -> np.ndarray | None:
    """Coerce an npz into a (n_layers, 500, 1536) float32 stack, or None."""
    keys = list(blob.keys())
    # Case A: a single 3-D array under a plausible key.
    for k in ("prefill", "hidden", "activations", "prefill_layers", "layers"):
        if k in keys:
            arr = blob[k]
            if arr.ndim == 3:
                return arr.astype(np.float64)
    # Case B: per-layer keys like "layer_00"/"L0"/"l19".
    layer_keys = []
    for k in keys:
        kl = k.lower().lstrip("layer_").lstrip("l").lstrip("_")
        if kl.isdigit():
            layer_keys.append((int(kl), k))
    if layer_keys:
        layer_keys.sort()
        mats = [blob[k] for _, k in layer_keys]
        if all(m.ndim == 2 for m in mats):
            return np.stack(mats, axis=0).astype(np.float64)
    return None


def load_layerwise() -> np.ndarray | None:
    """Return prefill activations as (n_layers, 500, 1536) or None if absent."""
    for path in LAYERWISE_NPZ_CANDIDATES:
        if path.exists():
            stack = _stack_from_npz(np.load(path))
            if stack is not None:
                return stack
    for d in LAYERWISE_DIR_CANDIDATES:
        if d.is_dir():
            files = sorted(d.glob("layer_*.npz"))
            if not files:
                files = sorted(d.glob("L*.npz"))
            mats = []
            for f in files:
                blob = np.load(f)
                key = "prefill" if "prefill" in blob else list(blob.keys())[0]
                mats.append(blob[key].astype(np.float64))
            if mats and all(m.ndim == 2 for m in mats):
                return np.stack(mats, axis=0)
    return None


def logistic_oof(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """OOF decision-function scores from an L2 logistic probe (standardized)."""
    from sklearn.linear_model import LogisticRegression

    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0); sd[sd < 1e-8] = 1.0
        Xtr_z = (Xtr - mu) / sd
        Xte_z = (Xte - mu) / sd
        clf = LogisticRegression(C=0.01, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr_z, ytr)
        scores[test_idx] = clf.decision_function(Xte_z)
    return scores


def dom_oof(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """OOF difference-of-means projection scores."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = Xte @ d_vec
    return scores


def main() -> int:
    if not LABEL_CACHE.exists():
        print("MISSING_REGEN_INPUT", LABEL_CACHE, file=sys.stderr); return 2
    labels = np.load(LABEL_CACHE)["correct"].astype(bool)
    assert labels.shape == (500,)

    stack = load_layerwise()
    if stack is None:
        print("MISSING_REGEN_INPUT", "no layerwise prefill cache found among:",
              file=sys.stderr)
        for p in LAYERWISE_NPZ_CANDIDATES + LAYERWISE_DIR_CANDIDATES:
            print("  ", p, file=sys.stderr)
        return 2

    n_layers, n_samples, _ = stack.shape
    if n_samples != 500:
        print("MISSING_REGEN_INPUT", f"expected 500 samples, got {n_samples}",
              file=sys.stderr); return 2

    folds = stratified_kfold(labels, N_FOLDS, SEED)
    depth_fracs, dom_aurocs, logit_aurocs = [], [], []
    for li in range(n_layers):
        Xl = stack[li]
        dom_aurocs.append(auroc(dom_oof(Xl, labels, folds), labels))
        logit_aurocs.append(auroc(logistic_oof(Xl, labels, folds), labels))
        depth_fracs.append(li / max(n_layers - 1, 1))

    dom_arr = np.array(dom_aurocs)
    logit_arr = np.array(logit_aurocs)
    dom_peak = int(np.nanargmax(dom_arr))
    logit_peak = int(np.nanargmax(logit_arr))

    out = {
        "experiment": "FE420",
        "n_layers": n_layers,
        "l19_index": L19_INDEX,
        "depth_fraction": depth_fracs,
        "dom_auroc_per_layer": [float(v) for v in dom_arr],
        "logistic_auroc_per_layer": [float(v) for v in logit_arr],
        "dom_peak_layer": dom_peak,
        "dom_peak_auroc": float(dom_arr[dom_peak]),
        "logistic_peak_layer": logit_peak,
        "logistic_peak_auroc": float(logit_arr[logit_peak]),
        "dom_auroc_at_l19": float(dom_arr[L19_INDEX]) if L19_INDEX < n_layers else None,
        "logistic_auroc_at_l19": float(logit_arr[L19_INDEX]) if L19_INDEX < n_layers else None,
        # F-2 refutation flag: True if the probe peaks strictly before L19.
        "peaks_before_l19_dom": bool(dom_peak < L19_INDEX),
        "peaks_before_l19_logistic": bool(logit_peak < L19_INDEX),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())