#!/usr/bin/env python3
"""Stage 2: Metric divergence — cosine PH vs Euclidean PH.

Demonstrates that cosine distance captures directional structure that
Euclidean PH misses, using:
  (a) Synthetic radial data (points on unit circle × random radii)
  (b) Real MATH-500 hidden-state trajectories (if available)

Run: python pathway7/validation/metric_divergence.py
"""
from __future__ import annotations

import sys
import json
import numpy as np
from pathlib import Path
from ripser import ripser
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, ".")
from pathway7.distance_metrics import cosine_ph, effective_resistance_ph


def max_persistence_h1(dgm: np.ndarray) -> float:
    if len(dgm) == 0:
        return 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return 0.0
    return float((finite[:, 1] - finite[:, 0]).max())


def test_radial_divergence(n: int = 200, ambient_dim: int = 50, seed: int = 42):
    """Points on a unit circle embedded in high dimensions with varying radii.

    In high dimensions, Euclidean distance concentration makes Rips PH degrade,
    while cosine PH sees through norm variation to recover angular structure.
    In 2D, both metrics recover the circle equally well — the divergence only
    manifests in high-D where Euclidean distances concentrate.
    """
    print(f"\n--- Synthetic: Unit circle × random radii (ambient_dim={ambient_dim}) ---")
    rng = np.random.default_rng(seed)

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Embed circle in first 2 dims of high-D space
    directions = np.zeros((n, ambient_dim))
    directions[:, 0] = np.cos(theta)
    directions[:, 1] = np.sin(theta)
    # Add noise in remaining dims (simulates high-D hidden states)
    directions[:, 2:] = rng.normal(0, 0.1, (n, ambient_dim - 2))
    # Normalize to unit vectors then scale by random radii
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    radii = rng.uniform(1.0, 10.0, n)
    X = directions * radii[:, None]  # (n, ambient_dim) with varying norms

    # Euclidean PH
    euclid_dgms = ripser(X, maxdim=1)["dgms"]
    euclid_h1 = max_persistence_h1(euclid_dgms[1])

    # Cosine PH
    cos_dgms = cosine_ph(X, maxdim=1)
    cosine_h1 = max_persistence_h1(cos_dgms[1])

    # Effective-resistance PH
    er_dgms = effective_resistance_ph(X, k=30, maxdim=1)
    er_h1 = max_persistence_h1(er_dgms[1])

    ratio = cosine_h1 / euclid_h1 if euclid_h1 > 0 else float("inf")
    status = "PASS" if ratio > 3.0 else "FAIL"

    print(f"  Euclidean H1 max persistence: {euclid_h1:.4f}")
    print(f"  Cosine    H1 max persistence: {cosine_h1:.4f}")
    print(f"  Eff-Res   H1 max persistence: {er_h1:.4f}")
    print(f"  Cosine/Euclidean ratio: {ratio:.1f}x [{status}]")
    print(f"  (Expect cosine >> Euclidean because angular structure is clean)")

    return {"euclidean_h1": euclid_h1, "cosine_h1": cosine_h1, "effres_h1": er_h1, "ratio": ratio}


def test_real_trajectories(n_problems: int = 10):
    """Run on real MATH-500 trajectories (if available on disk).

    Compares max H1 persistence across the three metrics on actual
    hidden-state point clouds. No assertions — just reports divergence.
    """
    print(f"\n--- Real MATH-500 trajectories (first {n_problems}) ---")

    traj_path = Path("data/experiment1_v2/trajectories.npz")
    if not traj_path.exists():
        print("  [SKIP] train_trajectories_partial.npz not found (GPU data not on disk)")
        return None

    data = np.load(traj_path, allow_pickle=True)
    keys = sorted([k for k in data.keys() if k.startswith("traj_")])[:n_problems]

    results = []
    for key in keys:
        traj = data[key]  # (n_tokens, hidden_dim)
        if len(traj) < 5:
            continue

        # Subsample to 100 points (matching winning_features.py)
        if len(traj) > 100:
            idx = np.linspace(0, len(traj) - 1, 100, dtype=int)
            traj = traj[idx]

        euclid_dgms = ripser(traj, maxdim=1)["dgms"]
        cos_dgms = cosine_ph(traj, maxdim=1)
        er_dgms = effective_resistance_ph(traj, k=30, maxdim=1)

        e_h1 = max_persistence_h1(euclid_dgms[1])
        c_h1 = max_persistence_h1(cos_dgms[1])
        r_h1 = max_persistence_h1(er_dgms[1])

        results.append({"problem": key, "euclid": e_h1, "cosine": c_h1, "effres": r_h1})

    if results:
        e_mean = np.mean([r["euclid"] for r in results])
        c_mean = np.mean([r["cosine"] for r in results])
        r_mean = np.mean([r["effres"] for r in results])
        print(f"  Mean H1 max persistence across {len(results)} problems:")
        print(f"    Euclidean: {e_mean:.4f}")
        print(f"    Cosine:    {c_mean:.4f}")
        print(f"    Eff-Res:   {r_mean:.4f}")

    return results


def main():
    print("=" * 60)
    print("Stage 2: Metric Divergence — Cosine vs Euclidean PH")
    print("=" * 60)

    synthetic = test_radial_divergence()
    real = test_real_trajectories()

    # Save results
    out = {"synthetic_radial": synthetic}
    if real:
        out["real_math500"] = real

    out_path = Path("pathway7/validation/metric_divergence_results.json")
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
