"""P11-FE543 — D²HScore (label-free, all-layer) vs single-layer L19 DoM at 7B scale.

Repeat of P11-FE542 on the cached P11 Qwen2.5-7B Stage-2 all-layer NPZs. Direct
test of whether the label-free, all-layer D²HScore (a curvature/dispersion metric
over the per-layer prefill trajectory) beats a single-layer L19 supervised DoM
probe at 7B scale.

F-9 claims all-layer signals collapse to single-layer informativeness at 1.5B.
This is the critical refutation test at the scale we publish at: if
D²HScore_AUROC_7B > L19-DoM_AUROC_7B, F-9 is contradicted at 7B.

D²HScore (label-free): for each problem, take the all-layer prefill trajectory
H (L, D), L2-normalize per layer, take the second difference across layers
(curvature of the representational path), and score the sample by the mean
curvature magnitude. No labels are used to compute it. Orientation against
correctness is reported both raw and oriented (max(a, 1-a)) for an honest
upper-bound comparison.

Falls back to MISSING_REGEN_INPUT (exit 2) if the 7B all-layer cache is absent.
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
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")

# 7B Stage-2 all-layer prefill cache. Try the documented candidate locations;
# the regen producer (P11 7B Stage 2 extract) writes one of these.
CACHE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill_alllayers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_alllayers.npz",
    ROOT / "pathway11_h100/data/7b_stage2/m7b_prefill_alllayers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m7b_prefill.npz",
]
OUT_JSON = ROOT / "pathway11_h100/d2hscore_7b/results.json"

L19 = 19
SEED = 9999
N_FOLDS = 5


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _resolve_cache() -> Path | None:
    for p in CACHE_CANDIDATES:
        if p.exists():
            return p
    return None


def _load_alllayers(blob) -> tuple[np.ndarray, np.ndarray]:
    """Return (H, correct) with H of shape (N, L, D)."""
    correct = blob["correct"].astype(bool)
    if "hidden" in blob:
        H = blob["hidden"].astype(np.float64)
    elif "prefill_alllayers" in blob:
        H = blob["prefill_alllayers"].astype(np.float64)
    elif "prefill" in blob:
        # Single-layer fallback: cannot compute an all-layer trajectory.
        P = blob["prefill"].astype(np.float64)
        H = P[:, None, :]
    else:
        raise KeyError("no all-layer hidden tensor in cache")
    if H.ndim != 3:
        raise ValueError(f"expected (N, L, D) hidden, got {H.shape}")
    return H, correct


def d2hscore(H: np.ndarray) -> np.ndarray:
    """Label-free per-sample D²HScore = mean curvature of the layer trajectory.

    H: (N, L, D). L2-normalize each layer vector, take the second difference
    across layers, and score each sample by the mean L2 magnitude of that
    curvature. Uses no labels.
    """
    N, L, D = H.shape
    norms = np.linalg.norm(H, axis=2, keepdims=True)
    norms[norms == 0] = 1.0
    Hn = H / norms
    if L < 3:
        # Not enough layers for a second difference; fall back to first diff.
        if L < 2:
            return np.zeros(N, dtype=np.float64)
        d1 = np.diff(Hn, axis=1)
        return np.linalg.norm(d1, axis=2).mean(axis=1)
    d2 = np.diff(Hn, n=2, axis=1)  # (N, L-2, D)
    return np.linalg.norm(d2, axis=2).mean(axis=1)


def oof_dom_auroc(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Out-of-fold supervised DoM probe AUROC on a single-layer feature block."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    for tr, te in skf.split(X, y):
        d = X[tr][y[tr]].mean(axis=0) - X[tr][~y[tr]].mean(axis=0)
        oof[te] = X[te] @ d
    return auroc(oof, y), oof


def main() -> int:
    cache = _resolve_cache()
    if cache is None:
        print("MISSING_REGEN_INPUT", CACHE_CANDIDATES[0], file=sys.stderr)
        return 2

    blob = np.load(cache)
    try:
        H, y = _load_alllayers(blob)
    except (KeyError, ValueError) as e:
        print("MISSING_REGEN_INPUT", cache, str(e), file=sys.stderr)
        return 2

    N, L, D = H.shape
    if L < 3:
        print("MISSING_REGEN_INPUT", cache, f"all-layer trajectory unavailable (L={L})",
              file=sys.stderr)
        return 2

    # Single-layer L19 supervised DoM (the F-9 incumbent at 7B).
    l19_idx = L19 if L19 < L else L - 1
    X_l19 = H[:, l19_idx, :]
    dom_auroc, _ = oof_dom_auroc(X_l19, y)

    # Label-free all-layer D²HScore.
    d2h = d2hscore(H)
    d2h_auroc_raw = auroc(d2h, y)
    d2h_auroc_oriented = (
        float(max(d2h_auroc_raw, 1.0 - d2h_auroc_raw))
        if not np.isnan(d2h_auroc_raw) else float("nan")
    )

    d2h_beats_dom = (
        bool(d2h_auroc_oriented > dom_auroc)
        if not (np.isnan(d2h_auroc_oriented) or np.isnan(dom_auroc)) else False
    )

    out = {
        "experiment": "P11-FE543",
        "cache": str(cache.relative_to(ROOT)),
        "n_samples": int(N),
        "n_layers": int(L),
        "hidden_dim": int(D),
        "l19_index_used": int(l19_idx),
        "n_correct": int(y.sum()),
        "l19_dom_auroc_oof_7b": dom_auroc,
        "d2hscore_auroc_raw_7b": d2h_auroc_raw,
        "d2hscore_auroc_oriented_7b": d2h_auroc_oriented,
        "d2hscore_label_free": True,
        "d2hscore_all_layer": True,
        "d2hscore_beats_l19_dom": d2h_beats_dom,
        "f9_collapse_contradicted_at_7b": d2h_beats_dom,
        "note": (
            "D²HScore is label-free (curvature of L2-normalized per-layer prefill "
            "trajectory); L19 DoM is supervised single-layer OOF. d2hscore_beats_l19_dom "
            "uses the orientation-corrected D²HScore AUROC as an honest upper bound."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())