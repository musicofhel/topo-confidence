"""D2HScore features (arXiv:2509.11569).

Two independent per-layer scores:
- Intra-layer dispersion (semantic breadth): std of token distances from centroid
- Inter-layer drift (semantic depth): change in layer centroids

Both can be computed from hidden states alone ("lite" version, 56 features).
Full version adds attention entropy (84 features) from separate extraction.
"""
from __future__ import annotations

import numpy as np

from pathway8_layerwise.config import N_TOTAL_LAYERS


def compute_d2h_from_states(
    states: np.ndarray,
    layers: range | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute D2HScore-lite from hidden states only (no attention needed).

    Parameters
    ----------
    states : (29, n_tokens, 1536) — all-layer hidden states

    Returns
    -------
    features : (2 + 2*L,) array — 2 scalar scores + per-layer dispersion/drift
    names : feature name list
    """
    if layers is None:
        layers = range(1, N_TOTAL_LAYERS)
    L = len(layers)

    dispersion = np.zeros(L, dtype=np.float64)
    centroids = np.zeros((L, states.shape[2]), dtype=np.float32)

    for i, layer in enumerate(layers):
        tokens = states[layer].astype(np.float32)  # (n_tokens, 1536)
        if len(tokens) == 0:
            continue
        centroid = tokens.mean(axis=0)
        centroids[i] = centroid
        # Dispersion: mean distance from centroid
        dists = np.linalg.norm(tokens - centroid, axis=1)
        dispersion[i] = float(dists.mean())

    # Inter-layer drift: norm of consecutive centroid differences
    drift = np.zeros(L, dtype=np.float64)
    for i in range(1, L):
        drift[i] = float(np.linalg.norm(centroids[i] - centroids[i - 1]))

    # Scalar summaries
    disp_score = float(dispersion.mean())
    drift_score = float(drift[1:].mean()) if L > 1 else 0.0

    features = np.concatenate([
        [disp_score, drift_score],
        dispersion,
        drift,
    ])

    names = ["D2H_dispersion_score", "D2H_drift_score"]
    for i, layer in enumerate(layers):
        names.append(f"D2H_disp_L{layer:02d}")
    for i, layer in enumerate(layers):
        names.append(f"D2H_drift_L{layer:02d}")

    return features, names


def compute_d2h_full(
    states: np.ndarray,
    d2h_attn_entropy: np.ndarray,
    layers: range | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Full D2HScore with attention entropy.

    Parameters
    ----------
    states : (29, n_tokens, 1536) — all-layer hidden states
    d2h_attn_entropy : (28,) — per-layer mean attention entropy from extraction

    Returns
    -------
    features : (2 + 3*L,) — scalar scores + per-layer dispersion/drift/attn_entropy
    names : feature name list
    """
    if layers is None:
        layers = range(1, N_TOTAL_LAYERS)
    L = len(layers)

    lite_feats, lite_names = compute_d2h_from_states(states, layers)

    # Attention entropy per layer
    attn_ent = np.zeros(L, dtype=np.float64)
    for i, layer in enumerate(layers):
        layer_idx = layer - 1  # d2h_attn_entropy is indexed 0-27 for layers 1-28
        if layer_idx < len(d2h_attn_entropy):
            attn_ent[i] = d2h_attn_entropy[layer_idx]

    attn_names = [f"D2H_attn_entropy_L{layer:02d}" for layer in layers]

    features = np.concatenate([lite_feats, attn_ent])
    names = lite_names + attn_names

    return features, names


def compute_d2h_batch(
    states_list: list[np.ndarray | None],
    d2h_attn_list: list[np.ndarray | None] | None = None,
    layers: range | None = None,
    full: bool = False,
) -> tuple[np.ndarray, list[str]]:
    """Compute D2HScore features for all problems.

    Returns (n_problems, n_features) array and feature names.
    """
    results = []
    names = None

    for i, s in enumerate(states_list):
        if s is None:
            if names is None:
                dummy = np.zeros((N_TOTAL_LAYERS, 10, 1536), dtype=np.float16)
                if full:
                    _, names = compute_d2h_full(dummy, np.zeros(28), layers)
                else:
                    _, names = compute_d2h_from_states(dummy, layers)
            results.append(np.zeros(len(names), dtype=np.float64))
        elif full and d2h_attn_list is not None and d2h_attn_list[i] is not None:
            feats, names = compute_d2h_full(s, d2h_attn_list[i], layers)
            results.append(feats)
        else:
            feats, names = compute_d2h_from_states(s, layers)
            results.append(feats)

    return np.stack(results), names
