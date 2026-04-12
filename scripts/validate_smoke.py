"""Smoke test: verify all 13 features produce finite values on synthetic data."""

import time

import numpy as np

from topo_confidence.features import FEATURE_NAMES, TopologicalFeatureExtractor


def make_synthetic_trajectories(n_problems=5, n_tokens=50, hidden_dim=64, seed=42):
    """Create synthetic token trajectories mimicking LLM hidden states."""
    rng = np.random.default_rng(seed)
    trajectories = []
    for i in range(n_problems):
        n_tok = n_tokens + rng.integers(-10, 10)
        # Mix of clusters + noise to give nontrivial topology
        t = rng.standard_normal((n_tok, hidden_dim))
        # Add cluster structure
        labels = rng.integers(0, 2, n_tok)
        t[labels == 0] += 2.0
        t[labels == 1] -= 2.0
        trajectories.append(t.astype(np.float32))
    return trajectories


def run_config(trajectories, max_dim, null_k, label):
    """Run feature extraction with given config and report results."""
    ext = TopologicalFeatureExtractor(max_dim=max_dim, null_k=null_k)

    t0 = time.perf_counter()
    features = ext.extract(token_trajectories=trajectories)
    elapsed = time.perf_counter() - t0

    n_problems = len(trajectories)
    per_sample = elapsed / n_problems

    print(f"\n{'=' * 60}")
    print(f"Config: {label} (max_dim={max_dim}, null_k={null_k})")
    print(f"  Shape: {features.shape} (expected ({n_problems}, {ext.n_features}))")
    print(f"  Time: {elapsed:.3f}s total, {per_sample:.3f}s/sample")

    assert features.shape == (n_problems, ext.n_features), (
        f"Shape mismatch: {features.shape} != ({n_problems}, {ext.n_features})"
    )

    n_nan = np.isnan(features).sum()
    n_inf = np.isinf(features).sum()
    print(f"  NaN count: {n_nan}, Inf count: {n_inf}")
    assert n_nan == 0, f"Found {n_nan} NaN values!"
    assert n_inf == 0, f"Found {n_inf} Inf values!"

    # Per-feature stats
    names = ext.feature_names
    n_zero_features = 0
    for j, name in enumerate(names):
        col = features[:, j]
        mean_v = col.mean()
        std_v = col.std()
        min_v = col.min()
        max_v = col.max()
        is_zero = std_v < 1e-10
        n_zero_features += is_zero
        flag = " [CONSTANT/ZERO]" if is_zero else ""
        print(f"  {name:>30s}: mean={mean_v:>10.4f}  std={std_v:>8.4f}  "
              f"range=[{min_v:.4f}, {max_v:.4f}]{flag}")

    if n_zero_features > 0:
        print(f"  WARNING: {n_zero_features} features are constant/zero")

    return features, elapsed, per_sample


def main():
    print(f"FEATURE_NAMES ({len(FEATURE_NAMES)}):")
    for i, n in enumerate(FEATURE_NAMES):
        print(f"  [{i:2d}] {n}")

    trajectories = make_synthetic_trajectories()
    print(f"\nSynthetic data: {len(trajectories)} problems, "
          f"~50 tokens x 64 hidden dims")

    # Full config (max_dim=2, null_k=100)
    feat_full, t_full, ps_full = run_config(
        trajectories, max_dim=2, null_k=100, label="FULL")

    # No null shuffles
    feat_nonull, t_nonull, ps_nonull = run_config(
        trajectories, max_dim=2, null_k=0, label="NO NULL")

    # max_dim=1 only (old config equivalent)
    feat_old, t_old, ps_old = run_config(
        trajectories, max_dim=1, null_k=0, label="OLD (max_dim=1, no null)")

    # Summary
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print(f"  FULL (max_dim=2, null_k=100): {ps_full:.3f}s/sample")
    print(f"  NO NULL (max_dim=2, null_k=0): {ps_nonull:.3f}s/sample")
    print(f"  OLD (max_dim=1, null_k=0):     {ps_old:.3f}s/sample")
    print(f"  Null overhead: {ps_full - ps_nonull:.3f}s/sample")
    print(f"  H2 overhead:   {ps_nonull - ps_old:.3f}s/sample")

    # Check H2 degeneracy
    h2_idx = FEATURE_NAMES.index("H2_n_features")
    h2_vals = feat_full[:, h2_idx]
    if h2_vals.sum() == 0:
        print("\n  *** H2 DEGENERACY: H2_n_features = 0 for all samples ***")
        print("  This suggests 100 pts in 30D is too sparse for voids.")
    else:
        print(f"\n  H2_n_features range: [{h2_vals.min():.0f}, {h2_vals.max():.0f}] -- OK")

    print("\nAll smoke tests PASSED")


if __name__ == "__main__":
    main()
