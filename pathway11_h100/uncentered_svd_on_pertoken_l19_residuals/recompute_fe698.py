"""P11-FE698 — Uncentered SVD of per-token L19 residual-stream trajectories.

Tests whether the L19 prefill DoM direction (F-2's single strong signal) is the
residual-stream image of Xu's optimizer-induced parameter-space "backbone".

For each MATH-500 problem we load its per-token L19 hidden-state trajectory
H (T, 1536), build the Δactivation matrix D[t] = H[t] - H[t-1] across token
positions, and take an UNCENTERED SVD of D. We then report, per problem:

  (a) PC1 energy fraction  s_1^2 / Σ s_i^2     (Xu predicts ~0.60-0.80)
  (b) cos(PC1, prefill_DoM_global)             (predicts ≈1 if DoM is backbone)
  (c) intra-signal gap position k* = argmax_i (s_i - s_{i+1}) + 1
                                                (predicts k*=1 in the majority)

prefill_DoM_global is recomputed as the (normalized) class-mean difference of
the global L19 prefill vectors from the main cache. PC1 sign is arbitrary, so
the alignment is reported as |cos|.

Per-token trajectory NPZs are Stage-2 caches (one file per problem). If they are
absent we emit MISSING_REGEN_INPUT and exit 2 rather than fabricate a result.
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
TRAJ_DIR = ROOT / "pathway11_h100/data/l19_token_trajectories"
OUT_JSON = ROOT / "pathway11_h100/xu_residual_backbone/results.json"

N_PROBLEMS = 500
DIM = 1536
TRAJ_KEYS = ("tokens", "hidden", "traj", "prefill_tokens", "l19", "hidden_states")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def _pick_traj(blob) -> np.ndarray | None:
    """Return the first plausible 2D (T, DIM) array from an NPZ blob."""
    for key in TRAJ_KEYS:
        if key in blob.files:
            arr = np.asarray(blob[key])
            if arr.ndim == 2 and arr.shape[1] == DIM:
                return arr.astype(np.float64)
    for key in blob.files:
        arr = np.asarray(blob[key])
        if arr.ndim == 2 and arr.shape[1] == DIM:
            return arr.astype(np.float64)
    return None


def _dom_global_direction() -> np.ndarray:
    """Recompute normalized class-mean-difference DoM from global prefill cache."""
    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    d = X[y].mean(axis=0) - X[~y].mean(axis=0)
    nrm = np.linalg.norm(d)
    return d / nrm if nrm > 0 else d


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2
    if not TRAJ_DIR.exists():
        print("MISSING_REGEN_INPUT", TRAJ_DIR, file=sys.stderr)
        return 2

    traj_files = sorted(TRAJ_DIR.glob("problem_*.npz"))
    if not traj_files:
        print("MISSING_REGEN_INPUT", TRAJ_DIR, file=sys.stderr)
        return 2

    dom_unit = _dom_global_direction()

    pc1_energy = []
    cos_pc1_dom = []
    k_star = []
    n_tokens = []
    skipped = 0

    for fp in traj_files:
        try:
            blob = np.load(fp)
        except Exception:
            skipped += 1
            continue
        H = _pick_traj(blob)
        if H is None or H.shape[0] < 3:
            skipped += 1
            continue

        # Δactivation matrix across token positions, uncentered SVD.
        D = np.diff(H, axis=0)
        if not np.all(np.isfinite(D)) or D.shape[0] < 2:
            skipped += 1
            continue
        try:
            _, s, Vt = np.linalg.svd(D, full_matrices=False)
        except np.linalg.LinAlgError:
            skipped += 1
            continue
        if s.size < 2 or s[0] <= 0:
            skipped += 1
            continue

        energy = float(s[0] ** 2 / np.sum(s ** 2))
        pc1 = Vt[0]
        cos = float(abs(np.dot(pc1, dom_unit)))
        gaps = s[:-1] - s[1:]
        kstar = int(np.argmax(gaps)) + 1

        pc1_energy.append(energy)
        cos_pc1_dom.append(cos)
        k_star.append(kstar)
        n_tokens.append(int(H.shape[0]))

    n_used = len(pc1_energy)
    if n_used == 0:
        print("MISSING_REGEN_INPUT no usable trajectories", file=sys.stderr)
        return 2

    pc1_energy = np.asarray(pc1_energy)
    cos_pc1_dom = np.asarray(cos_pc1_dom)
    k_star = np.asarray(k_star)

    frac_k1 = float(np.mean(k_star == 1))
    frac_energy_in_xu_band = float(np.mean((pc1_energy >= 0.60) & (pc1_energy <= 0.80)))

    out = {
        "experiment": "P11-FE698",
        "n_problems_used": n_used,
        "n_skipped": skipped,
        "pc1_energy_fraction": {
            "mean": float(np.mean(pc1_energy)),
            "median": float(np.median(pc1_energy)),
            "std": float(np.std(pc1_energy)),
            "min": float(np.min(pc1_energy)),
            "max": float(np.max(pc1_energy)),
            "frac_in_xu_60_80_band": frac_energy_in_xu_band,
        },
        "cos_pc1_prefill_dom_global": {
            "mean": float(np.mean(cos_pc1_dom)),
            "median": float(np.median(cos_pc1_dom)),
            "std": float(np.std(cos_pc1_dom)),
            "frac_above_0_9": float(np.mean(cos_pc1_dom >= 0.9)),
            "frac_above_0_7": float(np.mean(cos_pc1_dom >= 0.7)),
        },
        "k_star": {
            "frac_equal_1": frac_k1,
            "mean": float(np.mean(k_star)),
            "median": float(np.median(k_star)),
            "max": int(np.max(k_star)),
        },
        "n_tokens": {
            "mean": float(np.mean(n_tokens)),
            "median": float(np.median(n_tokens)),
        },
        # Headline verdict booleans for the backbone hypothesis.
        "backbone_supported": bool(
            np.median(pc1_energy) >= 0.60
            and np.median(cos_pc1_dom) >= 0.9
            and frac_k1 >= 0.5
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())