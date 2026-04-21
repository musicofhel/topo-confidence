"""Cross-layer single-token trajectory PH (Experiment 5).

For each token, form a 28-point trajectory across layers (layer 1-28).
PH on this trajectory uses H0-only (n=28 is too small for reliable H1).
Bootstrap: sample m=20 tokens from ~100, repeat 50 times, aggregate mean/std.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pathway8_layerwise.config import N_TOTAL_LAYERS
from pathway8_layerwise.layerwise_features import compute_6_ph_features


def compute_crosslayer_token_ph(
    states: np.ndarray,
    n_bootstrap: int = 50,
    m_tokens: int = 20,
    layers: range | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Cross-layer trajectory PH for a single problem.

    For each bootstrap sample:
      1. Sample m_tokens from the available tokens
      2. For each sampled token, extract its 28-layer trajectory: (28, 1536)
      3. PCA to 20 dims on the stacked trajectories
      4. For each token's trajectory: compute H0-only PH (3 features)
      5. Average across tokens → 3 features per bootstrap

    Aggregate across bootstraps: mean/std → 6 features total.

    Parameters
    ----------
    states : (29, n_tokens, 1536)
    n_bootstrap : number of bootstrap repetitions
    m_tokens : tokens per bootstrap sample
    layers : which layers (default: 1-28)

    Returns
    -------
    features : (6,) — mean/std of [H0_total, H0_n_features, H0_entropy]
    names : feature name list
    """
    if layers is None:
        layers = range(1, N_TOTAL_LAYERS)
    L = len(layers)
    n_tokens = states.shape[1]

    names = [
        "xlayer_H0_total_mean", "xlayer_H0_n_feat_mean", "xlayer_H0_entropy_mean",
        "xlayer_H0_total_std", "xlayer_H0_n_feat_std", "xlayer_H0_entropy_std",
    ]

    if n_tokens < 5 or L < 10:
        return np.zeros(6, dtype=np.float64), names

    rng = np.random.default_rng(seed)
    bootstrap_feats = []

    for b in range(n_bootstrap):
        # Sample m tokens (with replacement if n_tokens < m)
        if n_tokens >= m_tokens:
            tok_idx = rng.choice(n_tokens, size=m_tokens, replace=False)
        else:
            tok_idx = rng.choice(n_tokens, size=m_tokens, replace=True)

        # For each sampled token, extract its cross-layer trajectory: (L, 1536)
        # Stack all sampled tokens' trajectories for a shared PCA
        all_trajectories = []
        for t in tok_idx:
            traj = np.array([states[layer, t, :] for layer in layers], dtype=np.float32)
            all_trajectories.append(traj)

        stacked = np.stack(all_trajectories)  # (m_tokens, L, 1536)

        # PCA on the pooled (m_tokens*L, 1536) points
        pooled = stacked.reshape(-1, states.shape[2])
        n_comp = min(20, pooled.shape[0] - 1, pooled.shape[1])
        if n_comp < 2:
            continue

        scaler = StandardScaler().fit(pooled)
        pca = PCA(n_components=n_comp, random_state=0).fit(scaler.transform(pooled))
        pooled_reduced = pca.transform(scaler.transform(pooled))

        # Reshape back to per-token trajectories: (m_tokens, L, n_comp)
        per_token = pooled_reduced.reshape(m_tokens, L, n_comp)

        # H0-only PH per token trajectory (n=28 points, too small for H1)
        token_h0_features = []
        for traj_reduced in per_token:
            # traj_reduced: (L, n_comp) — L=28 points in n_comp dims
            ph = compute_6_ph_features(traj_reduced)
            # Take only H0 features: [H0_total, H0_n_features, H0_entropy]
            token_h0_features.append(ph[:3])

        token_h0_features = np.stack(token_h0_features)  # (m_tokens, 3)
        # Average across tokens for this bootstrap
        bootstrap_feats.append(token_h0_features.mean(axis=0))

    if not bootstrap_feats:
        return np.zeros(6, dtype=np.float64), names

    bootstrap_feats = np.stack(bootstrap_feats)  # (n_bootstrap, 3)
    features = np.concatenate([
        bootstrap_feats.mean(axis=0),  # mean of 3 H0 features
        bootstrap_feats.std(axis=0),   # std of 3 H0 features
    ])

    return features.astype(np.float64), names


def compute_crosslayer_batch(
    states_list: list[np.ndarray | None],
    n_bootstrap: int = 50,
    m_tokens: int = 20,
) -> tuple[np.ndarray, list[str]]:
    """Compute cross-layer trajectory PH for all problems."""
    results = []
    names = None

    for s in states_list:
        if s is None:
            if names is None:
                names = [
                    "xlayer_H0_total_mean", "xlayer_H0_n_feat_mean",
                    "xlayer_H0_entropy_mean", "xlayer_H0_total_std",
                    "xlayer_H0_n_feat_std", "xlayer_H0_entropy_std",
                ]
            results.append(np.zeros(6, dtype=np.float64))
        else:
            feats, names = compute_crosslayer_token_ph(s, n_bootstrap, m_tokens)
            results.append(feats)

    return np.stack(results), names
