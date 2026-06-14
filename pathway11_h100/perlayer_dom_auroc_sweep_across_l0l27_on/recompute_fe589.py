"""P11-FE589 — Per-layer DoM/logistic AUROC sweep across L0–L27 (Qwen-2.5-1.5B prefill).

Trains a 5-fold OOF logistic probe per layer on prefill hidden states, using the
same MATH-500 K=1 correctness labels as F-2, and (as a robustness secondary) the
difference-of-means (DoM) OOF score per layer.

Locates the AUROC peak across depth and adjudicates two Mix-Compress-Refine
predictions for generation-correctness signal:

  * F-2's layer-locality claim (L19 ≈ 67.8% depth > L27/final) is refuted by a
    monotonically rising curve into late layers.
  * Mix-Compress-Refine predicts generation-correctness signal should rise
    monotonically into Phase 3 (depth ≥ 85% → L23–L27 on 28 layers). A peak in
    Phase 1/2 (depth < 0.85) refutes that generation-task prediction.

Requires a per-layer prefill cache (500, n_layers, 1536). The canonical single-
layer cache only stores L19; if no all-layer cache is present this script reports
MISSING_REGEN_INPUT and exits 2.
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
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
# Per-layer prefill cache candidates: (path, key). First existing 3-D array wins.
ALL_LAYER_CANDIDATES = [
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all_layers.npz", None),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz", None),
    (ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all.npz", None),
    (ROOT / "pathway11_h100/data/m15b_prefill_all_layers.npz", None),
]
CANDIDATE_KEYS = ["prefill_all", "prefill_layers", "prefill", "hidden", "hidden_states", "activations"]

OUT_JSON = ROOT / "pathway11_h100/layer_sweep_dom/results.json"

N_FOLDS = 5
SEED = 9999
N_LAYERS_EXPECTED = 28
PHASE3_DEPTH = 0.85  # Mix-Compress-Refine Phase 3 threshold


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _load_all_layers():
    """Return (X_all, source_str) where X_all is (500, n_layers, 1536), or (None, None)."""
    for path, forced_key in ALL_LAYER_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        keys = [forced_key] if forced_key else CANDIDATE_KEYS + list(blob.files)
        for key in keys:
            if key is None or key not in blob.files:
                continue
            arr = blob[key]
            if arr.ndim == 3 and arr.shape[0] == 500 and arr.shape[2] == 1536:
                return arr.astype(np.float64), f"{path.name}:{key}"
    return None, None


def _logistic_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5-fold stratified OOF decision-function scores from an L2 logistic probe."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr = scaler.transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y[train_idx])
        oof[test_idx] = clf.decision_function(Xte)
    return oof


def _dom_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5-fold stratified OOF difference-of-means projection scores."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in skf.split(X, y):
        ytr = y[train_idx]
        d = X[train_idx][ytr].mean(axis=0) - X[train_idx][~ytr].mean(axis=0)
        oof[test_idx] = X[test_idx] @ d
    return oof


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    X_all, source = _load_all_layers()
    if X_all is None:
        print("MISSING_REGEN_INPUT per-layer prefill cache (500, n_layers, 1536) not found", file=sys.stderr)
        for path, _ in ALL_LAYER_CANDIDATES:
            print("  tried:", path, file=sys.stderr)
        return 2

    y = np.load(CACHE)["correct"].astype(bool)
    assert y.shape == (500,), f"unexpected label shape {y.shape}"
    n_layers = X_all.shape[1]

    layers = list(range(n_layers))
    depths = [(L / (n_layers - 1)) for L in layers]
    logit_aurocs = []
    dom_aurocs = []
    for L in layers:
        XL = X_all[:, L, :]
        logit_aurocs.append(auroc(_logistic_oof(XL, y), y))
        dom_aurocs.append(auroc(_dom_oof(XL, y), y))

    logit_aurocs = np.asarray(logit_aurocs, dtype=np.float64)
    dom_aurocs = np.asarray(dom_aurocs, dtype=np.float64)

    # Primary metric = logistic OOF AUROC.
    peak_idx = int(np.nanargmax(logit_aurocs))
    peak_layer = layers[peak_idx]
    peak_depth = float(depths[peak_idx])
    peak_auroc = float(logit_aurocs[peak_idx])

    # Monotonicity of the AUROC-vs-layer curve.
    valid = ~np.isnan(logit_aurocs)
    rho, pval = spearmanr(np.asarray(layers)[valid], logit_aurocs[valid])

    # F-2 reference layers (L19 ≈ 67.8% depth; L27/final).
    def _at(layer):
        return float(logit_aurocs[layer]) if 0 <= layer < n_layers else float("nan")

    l19_auroc = _at(19)
    final_auroc = float(logit_aurocs[-1])

    # Adjudication.
    peak_in_phase3 = peak_depth >= PHASE3_DEPTH
    monotonic_rising = bool(rho > 0.7 and (pval == 0 or pval < 0.05))

    # F-2 layer-locality: L19 > final. Refuted if curve rises monotonically to the end.
    refutes_f2_locality = bool(monotonic_rising and final_auroc >= l19_auroc - 1e-9)
    # Mix-Compress-Refine generation prediction: peak should be in Phase 3.
    refutes_mcr_generation = bool(not peak_in_phase3)

    out = {
        "experiment": "P11-FE589",
        "source": source,
        "n_layers": int(n_layers),
        "n_problems": int(len(y)),
        "n_correct": int(y.sum()),
        "phase3_depth_threshold": PHASE3_DEPTH,
        "layers": layers,
        "depths": [float(d) for d in depths],
        "logistic_oof_auroc": [float(a) for a in logit_aurocs],
        "dom_oof_auroc": [float(a) for a in dom_aurocs],
        "peak_layer": int(peak_layer),
        "peak_depth": peak_depth,
        "peak_auroc": peak_auroc,
        "peak_in_phase3": bool(peak_in_phase3),
        "spearman_rho_layer_vs_auroc": (float(rho) if np.isfinite(rho) else None),
        "spearman_p": (float(pval) if np.isfinite(pval) else None),
        "monotonic_rising": monotonic_rising,
        "l19_auroc": l19_auroc,
        "final_layer_auroc": final_auroc,
        "f2_l19_minus_final": float(l19_auroc - final_auroc),
        "refutes_f2_layer_locality": refutes_f2_locality,
        "refutes_mcr_generation_prediction": refutes_mcr_generation,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"peak L{peak_layer} (depth {peak_depth:.3f}) AUROC={peak_auroc:.4f} "
          f"rho={rho:.3f} L19={l19_auroc:.4f} final={final_auroc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())