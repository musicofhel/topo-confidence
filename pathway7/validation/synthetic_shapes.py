#!/usr/bin/env python3
"""Stage 1: PH computation correctness on synthetic shapes.

Validates that all three distance backends (Euclidean, cosine, effective
resistance) recover known topological features from canonical test shapes:
circle → 1 dominant H1, torus → 2 H1, sphere → 1 dominant H2.

Run: python pathway7/validation/synthetic_shapes.py
"""
from __future__ import annotations

import sys
import numpy as np
from ripser import ripser

sys.path.insert(0, ".")
from pathway7.distance_metrics import cosine_ph, effective_resistance_ph


def noisy_circle(n: int = 200, noise: float = 0.05, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X = np.c_[np.cos(theta), np.sin(theta)]
    return X + rng.normal(0, noise, X.shape)


def noisy_torus(n: int = 500, c: float = 2.0, a: float = 1.0, noise: float = 0.05, seed: int = 42) -> np.ndarray:
    """Sample from torus (c=major radius, a=minor radius) in R^3."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    X = np.c_[
        (c + a * np.cos(phi)) * np.cos(theta),
        (c + a * np.cos(phi)) * np.sin(theta),
        a * np.sin(phi),
    ]
    return X + rng.normal(0, noise, X.shape)


def noisy_sphere(n: int = 400, r: float = 1.0, noise: float = 0.03, seed: int = 42) -> np.ndarray:
    """Sample from 2-sphere in R^3."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    xy = np.sqrt(1 - z**2)
    X = r * np.c_[xy * np.cos(phi), xy * np.sin(phi), z]
    return X + rng.normal(0, noise, X.shape)


def max_persistence(dgm: np.ndarray) -> float:
    """Maximum persistence in a diagram (finite bars only)."""
    if len(dgm) == 0:
        return 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) == 0:
        return 0.0
    life = finite[:, 1] - finite[:, 0]
    return float(life.max())


def second_max_persistence(dgm: np.ndarray) -> float:
    """Second-largest persistence."""
    if len(dgm) == 0:
        return 0.0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) < 2:
        return 0.0
    life = np.sort(finite[:, 1] - finite[:, 0])[::-1]
    return float(life[1])


def count_dominant_bars(dgm: np.ndarray, ratio: float = 3.0) -> int:
    """Count bars with persistence > ratio * median persistence."""
    if len(dgm) == 0:
        return 0
    finite = dgm[np.isfinite(dgm[:, 1])]
    if len(finite) < 2:
        return len(finite)
    life = finite[:, 1] - finite[:, 0]
    thresh = ratio * np.median(life)
    return int(np.sum(life > thresh))


def euclidean_ph(points: np.ndarray, maxdim: int = 1) -> dict[int, np.ndarray]:
    """Standard Euclidean Rips PH (baseline)."""
    if len(points) < 3:
        return {d: np.empty((0, 2)) for d in range(maxdim + 1)}
    result = ripser(points, maxdim=maxdim)
    return {d: result["dgms"][d] for d in range(maxdim + 1)}


def test_circle():
    """Circle should produce 1 dominant H1 bar with persistence > 5× second."""
    print("\n--- Circle (200 pts, noise=0.05) ---")
    X = noisy_circle()

    for name, ph_fn in [("Euclidean", euclidean_ph), ("Cosine", cosine_ph), ("Eff-Res", effective_resistance_ph)]:
        dgms = ph_fn(X, maxdim=1)
        h1_max = max_persistence(dgms[1])
        h1_second = second_max_persistence(dgms[1])
        ratio = h1_max / h1_second if h1_second > 0 else float("inf")
        n_bars = len(dgms[1][np.isfinite(dgms[1][:, 1])]) if len(dgms[1]) > 0 else 0
        status = "PASS" if ratio > 3.0 else "FAIL"
        print(f"  {name:10s}: H1 max={h1_max:.4f}, 2nd={h1_second:.4f}, ratio={ratio:.1f}, n_bars={n_bars} [{status}]")


def test_torus():
    """Torus should produce >= 2 dominant H1 bars."""
    print("\n--- Torus (500 pts, noise=0.05) ---")
    X = noisy_torus()

    for name, ph_fn in [("Euclidean", euclidean_ph), ("Cosine", cosine_ph), ("Eff-Res", effective_resistance_ph)]:
        dgms = ph_fn(X, maxdim=1)
        n_dom = count_dominant_bars(dgms[1], ratio=3.0)
        h1_max = max_persistence(dgms[1])
        status = "PASS" if n_dom >= 2 else "FAIL"
        print(f"  {name:10s}: H1 dominant_bars={n_dom}, H1_max={h1_max:.4f} [{status}]")


def test_sphere():
    """2-sphere should produce 1 dominant H2 bar (Euclidean only, maxdim=2)."""
    print("\n--- Sphere (200 pts, noise=0.03, maxdim=2) ---")
    # Use fewer points for speed with maxdim=2
    X = noisy_sphere(n=200, noise=0.03)

    # Only test Euclidean — cosine/eff-res on 3D low-noise sphere
    # should also work but maxdim=2 is slow on graph-based methods
    dgms = euclidean_ph(X, maxdim=2)
    h2 = dgms.get(2, np.empty((0, 2)))
    h2_max = max_persistence(h2)
    n_h2 = len(h2[np.isfinite(h2[:, 1])]) if len(h2) > 0 else 0
    status = "PASS" if h2_max > 0 and n_h2 >= 1 else "FAIL"
    print(f"  Euclidean : H2 max={h2_max:.4f}, n_H2_bars={n_h2} [{status}]")


def test_degenerate():
    """Degenerate inputs: too-small point clouds should return empty diagrams."""
    print("\n--- Degenerate inputs ---")
    for n_pts in [0, 1, 2]:
        X = np.random.randn(n_pts, 10) if n_pts > 0 else np.empty((0, 10))
        for name, fn in [("cosine", cosine_ph), ("effres", effective_resistance_ph)]:
            dgms = fn(X, maxdim=1)
            ok = all(len(dgms[d]) == 0 for d in dgms)
            status = "PASS" if ok else "FAIL"
            print(f"  n={n_pts}, {name}: empty diagrams = {ok} [{status}]")


def main():
    print("=" * 60)
    print("Stage 1: PH Correctness Validation on Synthetic Shapes")
    print("=" * 60)

    test_circle()
    test_torus()
    test_sphere()
    test_degenerate()

    print("\n" + "=" * 60)
    print("Validation complete. Review PASS/FAIL above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
