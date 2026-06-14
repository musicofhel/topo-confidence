"""P11-FE687 — Multi-layer prefill DoM ensemble (TADA layer-localization analog).

Tests the TADA-implied prior that a functional concept (here: correctness) may
live in a small 2–4 layer subset of the residual stream rather than a single
layer. We refit OOF 5-fold logistic probes on cached 1.5B prefill activations
across a focus window (L17–L21) and a wider available sweep, then report:

  - per-layer single-layer OOF AUROC,
  - L19-alone baselines (logistic + precomputed DoM score),
  - concatenated focus-window (L17–L21) OOF AUROC,
  - full-sweep concatenation OOF AUROC,
  - best-of-k over every non-empty subset of the focus window,
  - delta of the best concatenated probe vs L19-alone.

This requires a *multi-layer* prefill cache. The canonical headline cache
(m15b_prefill.npz) only stores L19, so this script looks for a multi-layer
companion cache and prints MISSING_REGEN_INPUT (return 2) if absent — the L19
labels are read from the canonical cache, which always exists.

Counterargument it must survive: an ensemble can only *look* better because it
has more parameters; OOF folds + identical regularization across all probes
keep the L19-alone vs ensemble comparison honest.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
# Canonical L19 cache — always present; source of labels + L19 baseline.
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
# Multi-layer companion cache candidates (per-layer prefill hidden states).
MULTILAYER_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_multilayer.npz",
    ROOT / "pathway11_h100/data/m15b_prefill_layers.npz",
]
OUT_JSON = ROOT / "pathway11_h100/multilayer_dom_ensemble/results.json"

SEED = 9999
N_FOLDS = 5
C_REG = 0.1                     # identical regularization across all probes
FOCUS_WINDOW = [17, 18, 19, 20, 21]
N_SAMPLES = 500
D_LAYER = 1536


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def oof_logistic_scores(X: np.ndarray, y: np.ndarray, seed: int, C: float) -> np.ndarray:
    """OOF decision-function scores from a standardized L2 logistic probe."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    scores = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[train_idx])
        scores[test_idx] = clf.decision_function(Xte)
    return scores


def load_multilayer(path: Path, n: int) -> dict[int, np.ndarray] | None:
    """Return {layer_index: (n, D)} from a multi-layer prefill cache, or None."""
    blob = np.load(path, allow_pickle=False)
    layers: dict[int, np.ndarray] = {}

    # Format A: a 3D stack + an explicit layer-index vector.
    stack_key = next((k for k in ("prefill_layers", "prefill_stack", "hidden_layers")
                      if k in blob.files), None)
    if stack_key is not None:
        stack = np.asarray(blob[stack_key])
        if stack.ndim == 3:
            # Normalize to (num_layers, n, D).
            if stack.shape[1] == n:
                pass
            elif stack.shape[0] == n:
                stack = np.transpose(stack, (1, 0, 2))
            idx_key = next((k for k in ("layer_indices", "layers", "layer_ids")
                            if k in blob.files), None)
            if idx_key is not None:
                idxs = [int(v) for v in np.asarray(blob[idx_key]).ravel()]
            else:
                idxs = list(range(stack.shape[0]))
            for li, arr in zip(idxs, stack):
                if arr.shape[0] == n:
                    layers[int(li)] = arr.astype(np.float64)
            return layers or None

    # Format B: one keyed array per layer, e.g. "L17" / "prefill_L17" / "layer_17".
    for key in blob.files:
        m = re.search(r"(?:^|[_/])l(?:ayer)?[_]?(\d+)$", key, flags=re.IGNORECASE)
        if m is None:
            m = re.search(r"(\d+)$", key) if re.fullmatch(r"\D*\d+", key) else None
        if m is None:
            continue
        arr = np.asarray(blob[key])
        if arr.ndim == 2 and arr.shape[0] == n and arr.shape[1] == D_LAYER:
            layers[int(m.group(1))] = arr.astype(np.float64)
    return layers or None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    base = np.load(CACHE)
    y = base["correct"].astype(bool)
    X_l19 = base["prefill"].astype(np.float64)
    n = len(y)
    assert X_l19.shape == (N_SAMPLES, D_LAYER) and n == N_SAMPLES

    ml_path = next((p for p in MULTILAYER_CANDIDATES if p.exists()), None)
    if ml_path is None:
        print("MISSING_REGEN_INPUT", "no multi-layer prefill cache; tried:",
              *[str(p) for p in MULTILAYER_CANDIDATES], file=sys.stderr)
        return 2

    layers = load_multilayer(ml_path, n)
    if not layers:
        print("MISSING_REGEN_INPUT", "multi-layer cache has no usable layer arrays:",
              ml_path, file=sys.stderr)
        return 2

    # Ensure L19 is present; fall back to canonical cache for it.
    if 19 not in layers:
        layers[19] = X_l19

    available = sorted(layers)

    # --- L19-alone baselines ---
    l19_logistic = auroc(oof_logistic_scores(layers[19], y, SEED, C_REG), y)
    dom_auroc = float("nan")
    if DOM_NPZ.exists():
        dom_auroc = auroc(np.load(DOM_NPZ)["prefill_score"].astype(np.float64), y)

    # --- per-layer single-layer OOF AUROC ---
    per_layer = {}
    for li in available:
        per_layer[str(li)] = auroc(oof_logistic_scores(layers[li], y, SEED, C_REG), y)

    # --- concatenated focus window (L17-L21, intersected with availability) ---
    window = [li for li in FOCUS_WINDOW if li in layers]
    window_concat_auroc = float("nan")
    if len(window) >= 2:
        Xw = np.concatenate([layers[li] for li in window], axis=1)
        window_concat_auroc = auroc(oof_logistic_scores(Xw, y, SEED, C_REG), y)

    # --- full available-sweep concatenation ---
    Xfull = np.concatenate([layers[li] for li in available], axis=1)
    full_concat_auroc = auroc(oof_logistic_scores(Xfull, y, SEED, C_REG), y)

    # --- best-of-k over every non-empty subset of the focus window ---
    best_subset: list[int] = []
    best_subset_auroc = -1.0
    subset_scores = {}
    nw = len(window)
    for mask in range(1, 1 << nw):
        subset = [window[i] for i in range(nw) if (mask >> i) & 1]
        if len(subset) == 1:
            a = per_layer[str(subset[0])]
        else:
            Xs = np.concatenate([layers[li] for li in subset], axis=1)
            a = auroc(oof_logistic_scores(Xs, y, SEED, C_REG), y)
        subset_scores["+".join(f"L{li}" for li in subset)] = a
        if a > best_subset_auroc:
            best_subset_auroc = a
            best_subset = subset

    out = {
        "experiment": "P11-FE687",
        "description": "Multi-layer prefill DoM ensemble (TADA layer-localization analog)",
        "multilayer_cache": str(ml_path),
        "available_layers": available,
        "focus_window": window,
        "n_samples": int(n),
        "n_folds": N_FOLDS,
        "C_reg": C_REG,
        "seed": SEED,
        "l19_logistic_oof_auroc": l19_logistic,
        "l19_dom_precomputed_auroc": dom_auroc,
        "per_layer_oof_auroc": per_layer,
        "window_concat_oof_auroc": window_concat_auroc,
        "full_concat_oof_auroc": full_concat_auroc,
        "subset_oof_auroc": subset_scores,
        "best_subset": [f"L{li}" for li in best_subset],
        "best_subset_oof_auroc": float(best_subset_auroc),
        "delta_best_vs_l19_logistic": float(best_subset_auroc - l19_logistic),
        "delta_window_vs_l19_logistic": float(window_concat_auroc - l19_logistic),
        "verdict_localized_subset": bool(
            len(best_subset) >= 2 and (best_subset_auroc - l19_logistic) >= 0.01
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())