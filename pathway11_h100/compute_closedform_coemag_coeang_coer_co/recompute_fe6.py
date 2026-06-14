"""P9-FE6 — Closed-form Chain-of-Embedding (CoE) scores vs F-9 redundancy claim.

F-9 calls CoE-60 redundant with the single L19 DoM direction (F-2, AUROC 0.7731),
but EXP-030/036 measured CoE through a *60-dim XGBoost* feature vector, not the
CoE paper's headline *closed-form* scalars. The closed-form CoE-C in particular
aggregates phase-coherent step magnitudes (a complex/polar sum over the
hidden-state trajectory) that a tree ensemble on raw per-layer stats cannot
reproduce. This script computes the four closed-form variants directly and runs
a head-to-head AUROC comparison:

  CoE-Mag : summed L2 magnitude of consecutive hidden-state innovations
  CoE-Ang : summed angular innovation between consecutive hidden states
  CoE-R   : resultant sqrt(CoE-Mag^2 + CoE-Ang^2)
  CoE-C   : |sum_i mag_i * exp(j * cumulative_angle_i)| — phase-coherent energy

Genuine CoE needs the per-layer hidden-state trajectory. If a multi-layer cache
is available we compute the true closed forms over it; otherwise we fall back to
a single-step proxy (population centroid -> L19 prefill vector) and flag the
result as DEGRADED (CoE-C collapses to CoE-Mag with one step, so the
phase-coherence test is only meaningful on the multi-layer path).

Falsification rule: F-9 is FALSIFIED if any closed-form variant reaches an
orientation-free AUROC >= 0.80. AUROC sign is arbitrary for an unsupervised
self-eval score, so we report oriented_auroc = max(a, 1 - a) for the gate.
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
OUT_JSON = ROOT / "pathway11_h100/coe_closed_form/results.json"

# Optional multi-layer trajectory caches (first match with a usable key wins).
LAYER_CACHE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_all_layers.npz",
    ROOT / "pathway11_h100/data/m15b_layer_trajectory.npz",
]
LAYER_KEY_CANDIDATES = ["prefill_layers", "layers", "hidden_states", "trajectory"]

# Reference numbers for the head-to-head (1024-tok canonical).
F2_DOM_AUROC = 0.7731       # F-2: single L19 prefill DoM direction
EXP030_COE60_XGB = 0.811    # EXP-030/036: 60-dim XGBoost CoE feature vector
FALSIFY_THRESHOLD = 0.80    # F-9 falsified if any closed form >= this


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def oriented_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    a = auroc(scores, labels)
    if np.isnan(a):
        return a
    return max(a, 1.0 - a)


def coe_features(H: np.ndarray):
    """Closed-form CoE variants over a hidden-state trajectory.

    H: (n, L, d) — L hidden states per problem (L>=2). Returns four (n,) arrays.
    """
    h_prev = H[:, :-1, :]
    h_next = H[:, 1:, :]
    diffs = h_next - h_prev                                  # (n, L-1, d)

    mag_steps = np.linalg.norm(diffs, axis=2)                # (n, L-1)
    num = (h_prev * h_next).sum(axis=2)
    den = np.linalg.norm(h_prev, axis=2) * np.linalg.norm(h_next, axis=2)
    cos = np.clip(num / (den + 1e-12), -1.0, 1.0)
    ang_steps = np.arccos(cos)                               # (n, L-1)

    coe_mag = mag_steps.sum(axis=1)
    coe_ang = ang_steps.sum(axis=1)
    coe_r = np.sqrt(coe_mag ** 2 + coe_ang ** 2)
    cum_ang = np.cumsum(ang_steps, axis=1)
    z = (mag_steps * np.exp(1j * cum_ang)).sum(axis=1)
    coe_c = np.abs(z)
    return coe_mag, coe_ang, coe_r, coe_c


def load_layer_trajectory():
    """Return (H, source) for a multi-layer cache, or (None, None) if absent."""
    for path in LAYER_CACHE_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        for key in LAYER_KEY_CANDIDATES:
            if key in blob.files:
                H = blob[key].astype(np.float64)
                if H.ndim == 3 and H.shape[0] == 500 and H.shape[1] >= 2:
                    return H, f"{path.name}:{key}"
    return None, None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    dom_auroc = auroc(dom_score, y)

    H, layer_source = load_layer_trajectory()
    if H is not None:
        degraded = False
        n_layers = int(H.shape[1])
        coe_mag, coe_ang, coe_r, coe_c = coe_features(H)
    else:
        # Single-step proxy: chain = [population centroid] -> [L19 prefill].
        # CoE-C degenerates to CoE-Mag with one step; flag the result.
        degraded = True
        n_layers = 2
        mu = X.mean(axis=0)
        H_proxy = np.stack([np.broadcast_to(mu, X.shape), X], axis=1)  # (500,2,1536)
        coe_mag, coe_ang, coe_r, coe_c = coe_features(H_proxy)

    variants = {
        "CoE-Mag": coe_mag,
        "CoE-Ang": coe_ang,
        "CoE-R": coe_r,
        "CoE-C": coe_c,
    }

    results = {}
    best_oriented = 0.0
    best_variant = None
    for name, score in variants.items():
        a = auroc(score, y)
        oa = oriented_auroc(score, y)
        results[name] = {"auroc": a, "oriented_auroc": oa}
        if not np.isnan(oa) and oa > best_oriented:
            best_oriented = oa
            best_variant = name

    falsifies_f9 = bool(best_oriented >= FALSIFY_THRESHOLD)

    out = {
        "experiment": "P9-FE6",
        "description": "Closed-form CoE (Mag/Ang/R/C) head-to-head vs L19 DoM and CoE-60 XGB",
        "n": int(len(y)),
        "n_correct": int(y.sum()),
        "multi_layer_source": layer_source,
        "n_layers_used": n_layers,
        "degraded_single_step_proxy": degraded,
        "closed_form_variants": results,
        "best_variant": best_variant,
        "best_oriented_auroc": best_oriented,
        "reference": {
            "L19_DoM_auroc_F2": F2_DOM_AUROC,
            "L19_DoM_auroc_recomputed": dom_auroc,
            "CoE60_XGB_EXP030": EXP030_COE60_XGB,
            "falsify_threshold": FALSIFY_THRESHOLD,
        },
        "best_closed_form_beats_dom": bool(
            not np.isnan(best_oriented) and best_oriented > F2_DOM_AUROC
        ),
        "best_closed_form_beats_coe60_xgb": bool(
            not np.isnan(best_oriented) and best_oriented > EXP030_COE60_XGB
        ),
        "falsifies_f9": falsifies_f9,
        "verdict": (
            "F-9 FALSIFIED — closed-form CoE >= 0.80" if falsifies_f9
            else "F-9 STANDS — no closed-form CoE variant reaches 0.80"
        ),
    }
    if degraded:
        out["caveat"] = (
            "No multi-layer trajectory cache found; CoE computed via single-step "
            "centroid->L19 proxy. CoE-C is degenerate (== CoE-Mag) on one step, so "
            "the phase-coherence test is NOT exercised. Regenerate per-layer "
            "activations for a genuine closed-form CoE-C comparison."
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("best_variant", "best_oriented_auroc", "falsifies_f9", "degraded_single_step_proxy")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())