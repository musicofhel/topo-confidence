"""P11-FE542 — D²HScore (intra-layer attention-weighted centroid dispersion +
inter-layer drift) on cached P11 1.5B Stage 2 per-layer NPZs.

D²HScore is a label-free uncertainty signal: for each problem we read the
per-layer, per-token hidden states, form an attention-weighted centroid at every
layer, measure (a) the intra-layer dispersion of token states about that
centroid and (b) the inter-layer drift of consecutive centroids, then sum the
two. We AUROC that score against MATH-500 correctness and run it head-to-head
against the supervised L19 prefill DoM (0.7731), a Chain-of-Embedding (CoE)
proxy, and final-token DoM (0.7186).

Two refutations are adjudicated:
  #2  label-free (D²HScore) vs supervised (L19 DoM) at comparable model scale.
  #4  single-layer (L19-only dispersion) vs all-layer (full D²HScore).

Per-layer states are not part of the committed L19 cache, so the per-layer
directory is the gating regen input — absent it the script reports
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
PERLAYER_DIR = ROOT / "pathway11_h100/data/perlayer_states"
OUT_JSON = ROOT / "pathway11_h100/d2hscore/results.json"

N_PROBLEMS = 500
L19_INDEX = 19  # canonical prefill layer for the single-layer refutation arm


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def auroc_abs(scores: np.ndarray, labels: np.ndarray) -> float:
    """Orientation-agnostic AUROC (max of a score and its flip).

    D²HScore is unsigned uncertainty, so report the best of either polarity.
    """
    a = auroc(scores, labels)
    if a != a:  # NaN
        return a
    return max(a, 1.0 - a)


def _problem_path(i: int) -> Path:
    return PERLAYER_DIR / f"problem_{i:03d}.npz"


def _load_states(blob) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (hidden, attn) with hidden shaped (L, T, D); attn (L, T) or None.

    Tolerant of a few plausible key names across the per-layer cache schema.
    """
    keys = set(blob.files)
    hkey = next((k for k in ("hidden", "states", "hidden_states", "h") if k in keys), None)
    if hkey is None:
        raise KeyError("no hidden-states array in per-layer NPZ")
    hidden = np.asarray(blob[hkey], dtype=np.float64)
    if hidden.ndim != 3:
        raise ValueError(f"expected (L,T,D) hidden states, got {hidden.shape}")
    akey = next((k for k in ("attn", "attention", "attn_weights", "w") if k in keys), None)
    attn = np.asarray(blob[akey], dtype=np.float64) if akey is not None else None
    return hidden, attn


def _centroids_and_dispersion(hidden: np.ndarray, attn: np.ndarray | None):
    """Attention-weighted per-layer centroids and dispersions.

    hidden: (L, T, D). attn: (L, T) or None — falls back to L2-norm weights,
    which approximate the attention mass each token receives.
    """
    L, T, D = hidden.shape
    if attn is None:
        w = np.linalg.norm(hidden, axis=2)  # (L, T)
    else:
        if attn.shape != (L, T):
            # broadcast a single shared weighting across layers if needed
            attn = np.broadcast_to(attn.reshape(-1)[:T], (L, T))
        w = np.clip(attn, 0.0, None)
    wsum = w.sum(axis=1, keepdims=True)
    wsum[wsum == 0] = 1.0
    wn = w / wsum  # (L, T)
    centroids = np.einsum("lt,ltd->ld", wn, hidden)  # (L, D)
    diff = hidden - centroids[:, None, :]  # (L, T, D)
    disp = np.einsum("lt,ltd->l", wn, diff * diff)  # (L,) weighted variance
    return centroids, disp


def _inter_layer_drift(centroids: np.ndarray) -> float:
    if centroids.shape[0] < 2:
        return 0.0
    steps = np.diff(centroids, axis=0)  # (L-1, D)
    return float(np.linalg.norm(steps, axis=1).sum())


def _coe_proxy(centroids: np.ndarray) -> float:
    """Chain-of-Embedding magnitude+angle proxy over per-layer centroids."""
    if centroids.shape[0] < 2:
        return 0.0
    steps = np.diff(centroids, axis=0)
    mags = np.linalg.norm(steps, axis=1)
    mag_term = float(mags.sum())
    ang_term = 0.0
    for a, b in zip(steps[:-1], steps[1:]):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-12 and nb > 1e-12:
            cos = float(np.clip((a @ b) / (na * nb), -1.0, 1.0))
            ang_term += np.arccos(cos)
    return mag_term + ang_term


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not PERLAYER_DIR.exists():
        print("MISSING_REGEN_INPUT", PERLAYER_DIR, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    assert correct.shape == (N_PROBLEMS,)

    # Baseline: supervised L19 prefill DoM score (signed, label-derived).
    dom_score = None
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    d2h = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    intra = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    inter = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    l19_disp = np.full(N_PROBLEMS, np.nan, dtype=np.float64)
    coe = np.full(N_PROBLEMS, np.nan, dtype=np.float64)

    n_loaded = 0
    for i in range(N_PROBLEMS):
        p = _problem_path(i)
        if not p.exists():
            continue
        try:
            hidden, attn = _load_states(np.load(p))
            centroids, disp = _centroids_and_dispersion(hidden, attn)
        except (KeyError, ValueError) as exc:
            print(f"SKIP problem_{i:03d}: {exc}", file=sys.stderr)
            continue
        intra[i] = float(disp.mean())
        inter[i] = _inter_layer_drift(centroids)
        d2h[i] = intra[i] + inter[i]
        li = min(L19_INDEX, disp.shape[0] - 1)
        l19_disp[i] = float(disp[li])
        coe[i] = _coe_proxy(centroids)
        n_loaded += 1

    if n_loaded == 0:
        print("MISSING_REGEN_INPUT", PERLAYER_DIR, "(no readable per-layer NPZs)", file=sys.stderr)
        return 2

    valid = np.isfinite(d2h)
    y = correct[valid]

    out = {
        "experiment": "P11-FE542",
        "n_problems_loaded": int(n_loaded),
        "n_valid_scored": int(valid.sum()),
        # All-layer D²HScore (label-free), orientation-agnostic.
        "auroc_d2hscore_alllayer": auroc_abs(d2h[valid], y),
        "auroc_d2h_intra_only": auroc_abs(intra[valid], y),
        "auroc_d2h_inter_only": auroc_abs(inter[valid], y),
        # Single-layer L19 dispersion (refutation #4 arm).
        "auroc_l19_singlelayer_dispersion": auroc_abs(l19_disp[valid], y),
        # CoE proxy baseline (label-free competitor).
        "auroc_coe_proxy": auroc_abs(coe[valid], y),
        # Supervised L19 DoM baseline (refutation #2 arm).
        "auroc_l19_dom_supervised": (
            auroc(dom_score[valid], y) if dom_score is not None else None
        ),
        "reference_aurocs": {
            "l19_prefill_dom": 0.7731,
            "finaltoken_dom": 0.7186,
        },
    }

    # Refutation verdicts.
    d2h_auroc = out["auroc_d2hscore_alllayer"]
    dom_auroc = out["auroc_l19_dom_supervised"]
    out["refutation_2_labelfree_vs_supervised"] = (
        {
            "d2hscore_labelfree": d2h_auroc,
            "l19_dom_supervised": dom_auroc,
            "delta_labelfree_minus_supervised": (
                float(d2h_auroc - dom_auroc)
                if (dom_auroc is not None and d2h_auroc == d2h_auroc)
                else None
            ),
            "labelfree_competitive": (
                bool(d2h_auroc is not None and dom_auroc is not None and d2h_auroc >= dom_auroc - 0.02)
            ),
        }
    )
    out["refutation_4_singlelayer_vs_alllayer"] = {
        "single_layer_l19_dispersion": out["auroc_l19_singlelayer_dispersion"],
        "all_layer_d2hscore": d2h_auroc,
        "alllayer_beats_singlelayer": bool(
            d2h_auroc == d2h_auroc
            and out["auroc_l19_singlelayer_dispersion"] == out["auroc_l19_singlelayer_dispersion"]
            and d2h_auroc > out["auroc_l19_singlelayer_dispersion"]
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())