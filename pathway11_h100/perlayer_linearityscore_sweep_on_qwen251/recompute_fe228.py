"""P11-FE228 — Per-layer linearity_score sweep vs per-layer prefill DoM AUROC.

Refutation #1 for F-2 (L19-uniqueness). Computes a Razzhigaev §3.1
generalized-Procrustes linearity score between every pair of consecutive
transformer layers (lstsq linear fit on Frobenius-normalized, mean-centered
hidden states; score = 1 - residual Frobenius energy) and overlays the
per-layer prefill DoM OOF AUROC.

Question: does the AUROC peak at L19 coincide with a *non-linearity* peak
(low linearity_score — i.e. signal living in the residual of the ~0.99 linear
fit), or is L19 just one of many high-linearity middle-layer plateau picks?

Requires a per-layer prefill cache with a 3-D (n_layers, 500, 1536) array.
The single-layer m15b_prefill.npz only stores L19, so a dedicated per-layer
NPZ must exist; if absent we exit cleanly with MISSING_REGEN_INPUT.
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
# Per-layer hidden-state cache (n_layers, 500, 1536). Several plausible names
# are probed so this survives minor regen-naming drift.
PERLAYER_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_perlayer.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_perlayer.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz",
    ROOT / "pathway11_h100/data/perlayer/m15b_prefill_perlayer.npz",
]
# Single-layer L19 cache — used only for the correctness labels (and as a
# sanity cross-check on the L19 slice of the per-layer tensor).
L19_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/perlayer_linearity/results.json"

L19_LAYER = 19
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


def _find_perlayer() -> Path | None:
    for p in PERLAYER_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_perlayer(path: Path) -> np.ndarray:
    """Return hidden states as (n_layers, n_samples, d_model) float64."""
    blob = np.load(path)
    arr = None
    for key in ("hidden", "hidden_states", "layers", "prefill_perlayer",
                "prefill", "states"):
        if key in blob.files and blob[key].ndim == 3:
            arr = blob[key]
            break
    if arr is None:
        # fall back to the first 3-D array present
        for key in blob.files:
            if blob[key].ndim == 3:
                arr = blob[key]
                break
    if arr is None:
        raise ValueError(f"no 3-D hidden-state array in {path}: {blob.files}")
    a = arr.astype(np.float64)
    # Normalize orientation to (n_layers, n_samples, d_model). The sample
    # dimension is the one equal to 500.
    if a.shape[1] == 500 and a.shape[0] != 500:
        return a
    if a.shape[0] == 500:
        return np.transpose(a, (1, 0, 2))
    return a


def linearity_score(X: np.ndarray, Y: np.ndarray) -> float:
    """Razzhigaev §3.1 generalized-Procrustes linearity between two layers.

    Center each, scale to unit Frobenius norm, fit the optimal linear map
    W via least squares (X W ≈ Y), and report 1 - ||Y - X W||_F^2. A value
    near 1 means consecutive representations are nearly linearly related.
    """
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    xn = float(np.linalg.norm(Xc))
    yn = float(np.linalg.norm(Yc))
    if xn < 1e-12 or yn < 1e-12:
        return float("nan")
    Xc /= xn
    Yc /= yn
    W, _, _, _ = np.linalg.lstsq(Xc, Yc, rcond=None)
    resid = Yc - Xc @ W
    return float(1.0 - np.sum(resid * resid))


def perlayer_dom_auroc(X: np.ndarray, y: np.ndarray,
                       folds: list[np.ndarray]) -> float:
    """OOF DoM (mean_pos - mean_neg) projection AUROC for one layer."""
    n = len(y)
    oof = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        d_vec = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        oof[test_idx] = X[test_idx] @ d_vec
    return auroc(oof, y)


def main() -> int:
    if not L19_CACHE.exists():
        print("MISSING_REGEN_INPUT", L19_CACHE, file=sys.stderr)
        return 2
    perlayer_path = _find_perlayer()
    if perlayer_path is None:
        print("MISSING_REGEN_INPUT", PERLAYER_CANDIDATES[0],
              "(per-layer hidden-state cache required for FE228)",
              file=sys.stderr)
        return 2

    labels = np.load(L19_CACHE)["correct"].astype(bool)
    assert labels.shape == (500,), labels.shape

    H = _load_perlayer(perlayer_path)  # (n_layers, 500, 1536)
    n_layers = H.shape[0]
    if H.shape[1] != 500:
        print("MISSING_REGEN_INPUT", perlayer_path,
              f"(expected 500 samples, got shape {H.shape})", file=sys.stderr)
        return 2

    folds = stratified_kfold(labels, N_FOLDS, SEED)

    # Per-layer linearity between consecutive layers i -> i+1.
    lin_pairs = []   # (from_layer, to_layer, score)
    for i in range(n_layers - 1):
        s = linearity_score(H[i], H[i + 1])
        lin_pairs.append((i, i + 1, s))
    lin_scores = np.array([p[2] for p in lin_pairs], dtype=np.float64)

    # Per-layer prefill DoM OOF AUROC.
    layer_auroc = np.array(
        [perlayer_dom_auroc(H[i], labels, folds) for i in range(n_layers)],
        dtype=np.float64,
    )

    # Locate the AUROC peak and ask whether it lands on a linearity trough.
    valid_auroc = np.where(np.isfinite(layer_auroc), layer_auroc, -np.inf)
    peak_auroc_layer = int(np.argmax(valid_auroc))
    valid_lin = np.where(np.isfinite(lin_scores), lin_scores, np.inf)
    min_lin_pair = int(np.argmin(valid_lin))      # index into lin_pairs
    max_lin_pair = int(np.argmax(np.where(np.isfinite(lin_scores),
                                          lin_scores, -np.inf)))

    # Linearity "at" L19: the incoming transition (L18->L19) and outgoing
    # (L19->L20), if present.
    def lin_at(layer: int):
        incoming = lin_scores[layer - 1] if 1 <= layer <= n_layers - 1 else None
        outgoing = lin_scores[layer] if 0 <= layer <= n_layers - 2 else None
        vals = [v for v in (incoming, outgoing) if v is not None and np.isfinite(v)]
        return {
            "incoming_lin": (float(incoming) if incoming is not None
                             and np.isfinite(incoming) else None),
            "outgoing_lin": (float(outgoing) if outgoing is not None
                             and np.isfinite(outgoing) else None),
            "mean_lin": float(np.mean(vals)) if vals else None,
        }

    finite_lin = lin_scores[np.isfinite(lin_scores)]
    l19_lin = lin_at(L19_LAYER)
    # Rank of L19's mean transition-linearity among all transitions (0 = most
    # non-linear). A *low* rank supports the "L19 is a non-linearity peak" read.
    if l19_lin["mean_lin"] is not None and finite_lin.size:
        l19_lin_percentile = float(
            (finite_lin < l19_lin["mean_lin"]).mean()
        )
    else:
        l19_lin_percentile = None

    l19_in_range = 0 <= L19_LAYER < n_layers
    l19_auroc = float(layer_auroc[L19_LAYER]) if l19_in_range else None

    # Verdict heuristic: non-linearity feature iff AUROC peaks at/near L19 AND
    # L19 sits in the lower tail of the linearity distribution.
    auroc_peaks_at_l19 = abs(peak_auroc_layer - L19_LAYER) <= 1
    l19_is_nonlinearity = (
        l19_lin_percentile is not None and l19_lin_percentile <= 0.25
    )
    if auroc_peaks_at_l19 and l19_is_nonlinearity:
        verdict = "NONLINEARITY_FEATURE"
    elif auroc_peaks_at_l19 and l19_lin_percentile is not None and l19_lin_percentile >= 0.5:
        verdict = "HIGH_LINEARITY_PLATEAU_PICK"
    else:
        verdict = "INCONCLUSIVE"

    out = {
        "experiment": "P11-FE228",
        "perlayer_cache": str(perlayer_path),
        "n_layers": int(n_layers),
        "n_transitions": int(n_layers - 1),
        "l19_layer_index": L19_LAYER,
        "linearity_transitions": [
            {"from": int(a), "to": int(b), "linearity_score": float(s)}
            for (a, b, s) in lin_pairs
        ],
        "linearity_scores": [float(x) for x in lin_scores],
        "perlayer_dom_auroc": [float(x) for x in layer_auroc],
        "linearity_mean": float(np.nanmean(lin_scores)),
        "linearity_min": float(np.nanmin(lin_scores)),
        "linearity_max": float(np.nanmax(lin_scores)),
        "min_linearity_transition": {
            "from": int(lin_pairs[min_lin_pair][0]),
            "to": int(lin_pairs[min_lin_pair][1]),
            "linearity_score": float(lin_scores[min_lin_pair]),
        },
        "max_linearity_transition": {
            "from": int(lin_pairs[max_lin_pair][0]),
            "to": int(lin_pairs[max_lin_pair][1]),
            "linearity_score": float(lin_scores[max_lin_pair]),
        },
        "peak_auroc_layer": peak_auroc_layer,
        "peak_auroc_value": float(valid_auroc[peak_auroc_layer]),
        "l19_auroc": l19_auroc,
        "l19_linearity": l19_lin,
        "l19_linearity_percentile": l19_lin_percentile,
        "auroc_peaks_at_l19": bool(auroc_peaks_at_l19),
        "verdict": verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"FE228 done: peak AUROC layer={peak_auroc_layer} "
          f"(L19 AUROC={l19_auroc}), verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())