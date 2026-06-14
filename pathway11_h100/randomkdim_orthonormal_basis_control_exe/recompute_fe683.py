"""P11-FE683 — Random-K-dim orthonormal basis control (ρ_exec/ρ_rand) on
L19 hidden-state trajectories.

For each MATH-500 problem the per-token L19 sequence is treated as a trajectory.
We stack its step-to-step deltas D = x_{t+1}-x_t, fit a top-K (K=4) PCA basis on
those deltas, and measure the fraction of delta energy captured by that basis
(ρ_exec). The control ρ_rand is the same fraction averaged over N_RANDOM random
K-dim orthonormal subspaces of R^1536. The ρ_exec/ρ_rand ratio is reported split
by correct vs incorrect outcome and by token-position band (prefill / mid /
final-token third of the trajectory).

If ρ_exec/ρ_rand >> 1 there is concentrated trajectory geometry that the
point-cloud-PH Gaussian-null framing of F-10 cannot see — i.e. F-10's null is
invariant-specific. A ratio near 1 means the trajectory carries no more
low-dimensional structure than a random subspace.

Trajectory inputs are per-problem L19 token sequences. The documented single
1536-d prefill cache does NOT contain these; this script loads a per-problem
trajectory cache and degrades gracefully (MISSING_REGEN_INPUT) if absent.
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
TRAJ_DIR = ROOT / "pathway11_h100/data/l19_trajectories"
OUT_JSON = ROOT / "pathway11_h100/trajectory_random_basis/results.json"

K = 4
N_RANDOM = 10
SEED = 9999
HIDDEN_DIM = 1536
TRAJ_KEYS = ("hidden", "hidden_states", "l19", "traj", "trajectory", "states")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def load_trajectory(npz_path: Path) -> np.ndarray | None:
    """Return a (T, HIDDEN_DIM) float64 trajectory or None if unusable."""
    try:
        blob = np.load(npz_path)
    except Exception:
        return None
    arr = None
    for key in TRAJ_KEYS:
        if key in blob.files:
            cand = np.asarray(blob[key])
            if cand.ndim == 2 and cand.shape[1] == HIDDEN_DIM:
                arr = cand
                break
    if arr is None:
        # Fall back to the first 2D array whose second dim matches.
        for key in blob.files:
            cand = np.asarray(blob[key])
            if cand.ndim == 2 and cand.shape[1] == HIDDEN_DIM:
                arr = cand
                break
    if arr is None:
        return None
    return arr.astype(np.float64)


def proj_fraction(Dc: np.ndarray, B: np.ndarray) -> float:
    """Fraction of squared-Frobenius energy of Dc captured by orthonormal B."""
    if Dc.shape[0] == 0:
        return float("nan")
    total = float(np.sum(Dc * Dc))
    if total <= 0.0:
        return float("nan")
    P = Dc @ B
    return float(np.sum(P * P)) / total


def pca_basis(Dc: np.ndarray, k: int) -> np.ndarray | None:
    """Top-k right singular vectors of centered deltas, as (HIDDEN_DIM, k)."""
    if Dc.shape[0] < k:
        return None
    try:
        _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if Vt.shape[0] < k:
        return None
    return np.ascontiguousarray(Vt[:k].T)


def random_bases(rng: np.random.Generator, k: int, n: int) -> list[np.ndarray]:
    bases = []
    for _ in range(n):
        G = rng.standard_normal((HIDDEN_DIM, k))
        Q, _ = np.linalg.qr(G)
        bases.append(Q[:, :k])
    return bases


def band_slices(n: int) -> dict[str, slice]:
    """Split delta-row indices into prefill / mid / final thirds."""
    c1 = n // 3
    c2 = (2 * n) // 3
    return {"prefill": slice(0, c1), "mid": slice(c1, c2), "final": slice(c2, n)}


def safe_mean(xs: list[float]) -> float:
    arr = np.asarray([x for x in xs if np.isfinite(x)], dtype=np.float64)
    return float(arr.mean()) if arr.size else float("nan")


def ratio_summary(exec_fracs: list[float], rand_fracs: list[float]) -> dict:
    """Per-problem ratio mean plus pooled (mean-exec / mean-rand)."""
    per_ratio = []
    for e, r in zip(exec_fracs, rand_fracs):
        if np.isfinite(e) and np.isfinite(r) and r > 0:
            per_ratio.append(e / r)
    me = safe_mean(exec_fracs)
    mr = safe_mean(rand_fracs)
    return {
        "n": len(per_ratio),
        "rho_exec_mean": me,
        "rho_rand_mean": mr,
        "ratio_mean_of_ratios": safe_mean(per_ratio),
        "ratio_pooled": float(me / mr) if (np.isfinite(me) and np.isfinite(mr) and mr > 0) else float("nan"),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not TRAJ_DIR.exists():
        print("MISSING_REGEN_INPUT", TRAJ_DIR, file=sys.stderr)
        return 2

    correct_all = np.load(CACHE)["correct"].astype(bool)
    n_problems = correct_all.shape[0]

    files = sorted(TRAJ_DIR.glob("problem_*.npz"))
    if not files:
        print("MISSING_REGEN_INPUT", TRAJ_DIR / "problem_*.npz", file=sys.stderr)
        return 2

    rng = np.random.default_rng(SEED)
    rbases = random_bases(rng, K, N_RANDOM)

    bands = ("prefill", "mid", "final")
    # records[band] -> {"exec": [...], "rand": [...], "correct": [bool ...]}
    rec_overall = {"exec": [], "rand": [], "correct": []}
    rec_band = {b: {"exec": [], "rand": [], "correct": []} for b in bands}

    n_used = 0
    n_skipped = 0
    for f in files:
        m = re.search(r"problem_(\d+)", f.stem)
        if m is None:
            continue
        idx = int(m.group(1))
        if not (0 <= idx < n_problems):
            n_skipped += 1
            continue

        traj = load_trajectory(f)
        if traj is None or traj.shape[0] < 2:
            n_skipped += 1
            continue

        D = np.diff(traj, axis=0)            # (T-1, HIDDEN_DIM) step-to-step deltas
        Dc = D - D.mean(axis=0, keepdims=True)
        B_pca = pca_basis(Dc, K)
        if B_pca is None:
            n_skipped += 1
            continue

        is_correct = bool(correct_all[idx])

        # Overall (all deltas).
        e_all = proj_fraction(Dc, B_pca)
        r_all = safe_mean([proj_fraction(Dc, Br) for Br in rbases])
        if np.isfinite(e_all) and np.isfinite(r_all):
            rec_overall["exec"].append(e_all)
            rec_overall["rand"].append(r_all)
            rec_overall["correct"].append(is_correct)

        # Position bands (same per-problem PCA + random bases).
        slices = band_slices(Dc.shape[0])
        for b in bands:
            seg = Dc[slices[b]]
            if seg.shape[0] == 0:
                continue
            eb = proj_fraction(seg, B_pca)
            rb = safe_mean([proj_fraction(seg, Br) for Br in rbases])
            if np.isfinite(eb) and np.isfinite(rb):
                rec_band[b]["exec"].append(eb)
                rec_band[b]["rand"].append(rb)
                rec_band[b]["correct"].append(is_correct)

        n_used += 1

    if n_used == 0:
        print("MISSING_REGEN_INPUT no_usable_trajectories", file=sys.stderr)
        return 2

    def split_by_outcome(rec: dict) -> dict:
        e = np.asarray(rec["exec"], dtype=np.float64)
        r = np.asarray(rec["rand"], dtype=np.float64)
        c = np.asarray(rec["correct"], dtype=bool)
        out = {"all": ratio_summary(list(e), list(r))}
        if c.any():
            out["correct"] = ratio_summary(list(e[c]), list(r[c]))
        if (~c).any():
            out["incorrect"] = ratio_summary(list(e[~c]), list(r[~c]))
        return out

    out = {
        "experiment": "P11-FE683",
        "K": K,
        "n_random_bases": N_RANDOM,
        "seed": SEED,
        "n_problems_used": int(n_used),
        "n_skipped": int(n_skipped),
        "expected_random_baseline_fraction": float(K / HIDDEN_DIM),
        "overall": split_by_outcome(rec_overall),
        "by_band": {b: split_by_outcome(rec_band[b]) for b in bands},
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("OK", OUT_JSON, "n_used=", n_used, "n_skipped=", n_skipped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())