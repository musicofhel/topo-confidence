"""P9-FE5 — Closed-form Chain-of-Embedding (CoE) scores vs L19 DoM and CoE-60 XGB.

F-9 calls CoE-60 redundant with the single-layer L19 DoM direction, but the
EXP-030/036 numbers came from a 60-dim XGBoost feature vector — not the CoE
paper's headline *closed-form* scalar (Wang et al., "Latent Space Chain-of-
Embedding"). The closed-form CoE-C in particular aggregates step magnitudes with
their cumulative phase (a coherent complex sum) — a nonlinearity XGB on raw
per-layer features does not reproduce. This script computes the four closed-form
variants directly from the cached per-layer hidden-state trajectory and reports a
single scalar AUROC for each, oriented so the discriminative direction is the one
that helps (max(auroc, 1-auroc)):

  CoE-Mag : mean step magnitude   mean_k ||h_{k+1} - h_k||
  CoE-Ang : mean turning angle    mean_k angle(Δ_k, Δ_{k+1})
  CoE-R   : net radial extent     ||h_last - h_first||
  CoE-C   : phase-coherent sum    |Σ_k ||Δ_k|| · exp(i·φ_k)|, φ_k = cumsum turning angle

Falsifies F-9 if any oriented closed-form AUROC ≥ 0.80 (clearing both the L19 DoM
0.7731 F-2 baseline and the CoE-60 XGB 0.811 EXP-030 number).

Requires a per-layer trajectory cache (500, n_layers, 1536); the single-layer
m15b_prefill.npz alone is insufficient. Prints MISSING_REGEN_INPUT and returns 2
if no layerwise cache is present.
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
OUT_JSON = ROOT / "pathway11_h100/coe_closedform/results.json"

# Candidate per-layer trajectory caches: (500, n_layers, 1536) float arrays.
LAYERWISE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_layerwise.npz",
    ROOT / "pathway11_h100/data/m15b_layerwise.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layerwise.npz",
    ROOT / "pathway8/cache/m15b_layerwise.npz",
]

# Reference baselines (1024-tok canonical).
F2_DOM_AUROC = 0.7731       # L19 prefill DoM, OOF 5-fold (F-2)
COE60_XGB_AUROC = 0.811     # 60-dim XGB CoE-60, EXP-030
FALSIFY_THRESHOLD = 0.80    # F-9 falsified if any closed-form variant clears this

EPS = 1e-12


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def oriented_auroc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (directed_auroc, oriented_auroc) where oriented = max(a, 1-a)."""
    a = auroc(scores, labels)
    if np.isnan(a):
        return a, a
    return a, float(max(a, 1.0 - a))


def find_layerwise() -> Path | None:
    for p in LAYERWISE_CANDIDATES:
        if p.exists():
            return p
    return None


def load_trajectory(path: Path) -> np.ndarray:
    """Load the first 3D float array of shape (500, L, 1536) from the NPZ."""
    blob = np.load(path)
    for key in blob.files:
        arr = blob[key]
        if arr.ndim == 3 and arr.shape[0] == 500 and arr.shape[2] == 1536:
            return arr.astype(np.float64)
    raise ValueError(f"no (500, L, 1536) array in {path}; keys={list(blob.files)}")


def coe_scores(H: np.ndarray) -> dict[str, np.ndarray]:
    """Closed-form CoE-Mag / -Ang / -R / -C per sample. H: (N, L, d)."""
    n, L, d = H.shape
    delta = H[:, 1:, :] - H[:, :-1, :]                 # (N, L-1, d) step vectors
    step_mag = np.linalg.norm(delta, axis=2)           # (N, L-1)

    coe_mag = step_mag.mean(axis=1)                    # mean step magnitude
    coe_r = np.linalg.norm(H[:, -1, :] - H[:, 0, :], axis=1)  # net radial extent

    # Turning angle between consecutive step vectors Δ_k, Δ_{k+1}.
    if L >= 3:
        d0 = delta[:, :-1, :]                          # (N, L-2, d)
        d1 = delta[:, 1:, :]
        num = np.einsum("nkd,nkd->nk", d0, d1)
        den = np.linalg.norm(d0, axis=2) * np.linalg.norm(d1, axis=2)
        cos = np.clip(num / (den + EPS), -1.0, 1.0)
        angle = np.arccos(cos)                         # (N, L-2)
        coe_ang = angle.mean(axis=1)

        # Phase-coherent complex aggregation: weight each step magnitude by the
        # cumulative turning phase, then take the modulus of the coherent sum.
        # Step k (k>=1) carries cumulative phase = sum of angles up to k.
        phase = np.zeros((n, L - 1), dtype=np.float64)
        phase[:, 1:] = np.cumsum(angle, axis=1)
        z = (step_mag * np.exp(1j * phase)).sum(axis=1)
        coe_c = np.abs(z)
    else:
        coe_ang = np.full(n, np.nan)
        coe_c = np.full(n, np.nan)

    return {"CoE-Mag": coe_mag, "CoE-Ang": coe_ang, "CoE-R": coe_r, "CoE-C": coe_c}


def main() -> int:
    layerwise = find_layerwise()
    if layerwise is None:
        print("MISSING_REGEN_INPUT", " ".join(str(p) for p in LAYERWISE_CANDIDATES),
              file=sys.stderr)
        return 2
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    y = np.load(CACHE)["correct"].astype(bool)
    assert y.shape == (500,), y.shape

    try:
        H = load_trajectory(layerwise)
    except ValueError as exc:
        print("MISSING_REGEN_INPUT", exc, file=sys.stderr)
        return 2
    assert H.shape[0] == 500, H.shape

    scores = coe_scores(H)

    variants = {}
    best_oriented = 0.0
    best_variant = None
    for name, s in scores.items():
        directed, oriented = oriented_auroc(s, y)
        variants[name] = {
            "auroc_directed": directed,
            "auroc_oriented": oriented,
        }
        if not np.isnan(oriented) and oriented > best_oriented:
            best_oriented = oriented
            best_variant = name

    # DoM sanity comparison from cached scalar score, if available.
    dom_auroc = None
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        _, dom_auroc = oriented_auroc(dom_score, y)

    falsifies_f9 = bool(best_oriented >= FALSIFY_THRESHOLD)

    out = {
        "experiment": "P9-FE5",
        "description": "Closed-form CoE-Mag/-Ang/-R/-C vs L19 DoM and CoE-60 XGB",
        "layerwise_cache": str(layerwise),
        "n_layers": int(H.shape[1]),
        "n_problems": int(H.shape[0]),
        "variants": variants,
        "best_variant": best_variant,
        "best_oriented_auroc": float(best_oriented),
        "baselines": {
            "l19_dom_f2_auroc": F2_DOM_AUROC,
            "dom_recomputed_oriented_auroc": dom_auroc,
            "coe60_xgb_exp030_auroc": COE60_XGB_AUROC,
        },
        "falsify_threshold": FALSIFY_THRESHOLD,
        "falsifies_f9": falsifies_f9,
        "verdict": (
            f"closed-form {best_variant} oriented AUROC {best_oriented:.4f} "
            + (">=" if falsifies_f9 else "<")
            + f" {FALSIFY_THRESHOLD:.2f} — F-9 "
            + ("FALSIFIED" if falsifies_f9 else "STANDS")
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(out["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())