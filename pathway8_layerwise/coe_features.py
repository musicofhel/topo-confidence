"""Chain-of-Embedding features (Wang et al., ICLR 2025, arXiv:2410.13640).

Uses mean-pooled token embeddings per layer. Two scalar features of the
L-layer trajectory: magnitude change and angle change. Pure numpy, no PCA.
"""
from __future__ import annotations

import numpy as np

from pathway8_layerwise.config import N_TOTAL_LAYERS


def compute_coe_single(
    states: np.ndarray,
    layers: range | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute CoE features for a single problem.

    Parameters
    ----------
    states : (29, n_tokens, 1536) — all-layer hidden states

    Returns
    -------
    features : (4 + 2*L,) array — 4 scalar scores + per-layer mag/ang
    names : feature name list
    """
    if layers is None:
        layers = range(1, N_TOTAL_LAYERS)
    L = len(layers)

    # Mean-pool tokens per layer: (L, 1536)
    per_layer_mean = np.zeros((L, states.shape[2]), dtype=np.float32)
    for i, layer in enumerate(layers):
        tokens = states[layer].astype(np.float32)
        if len(tokens) == 0:
            continue
        per_layer_mean[i] = tokens.mean(axis=0)

    # Consecutive differences
    dif = per_layer_mean[1:] - per_layer_mean[:-1]  # (L-1, d)

    # Per-layer magnitude of change
    mag_per_layer = np.linalg.norm(dif, axis=1)  # (L-1,)

    # Per-layer angle change (arccos of cosine similarity)
    norms = np.linalg.norm(per_layer_mean, axis=1)  # (L,)
    norms_prev = norms[:-1]
    norms_next = norms[1:]
    denom = norms_prev * norms_next + 1e-12
    cos_sim = np.einsum("ld,ld->l", per_layer_mean[:-1], per_layer_mean[1:]) / denom
    cos_sim = np.clip(cos_sim, -1 + 1e-7, 1 - 1e-7)
    ang_per_layer = np.arccos(cos_sim)  # (L-1,)

    # Scalar summaries
    mag_mean = float(mag_per_layer.mean()) if len(mag_per_layer) > 0 else 0.0
    ang_mean = float(ang_per_layer.mean()) if len(ang_per_layer) > 0 else 0.0

    # CoE-R: normalized magnitude + (1 - normalized angle)
    total_displacement = np.linalg.norm(per_layer_mean[-1] - per_layer_mean[0])
    coe_r = mag_mean / (total_displacement + 1e-12) + (1 - ang_mean / np.pi)

    # CoE-C: complex combination |mean(mag * exp(i*ang))|
    z = mag_per_layer * np.exp(1j * ang_per_layer)
    coe_c = float(np.abs(np.mean(z))) if len(z) > 0 else 0.0

    # Build feature vector: 4 scalars + L-1 magnitudes + L-1 angles
    # Pad to L features each (first layer has 0 mag/ang)
    mag_padded = np.concatenate([[0.0], mag_per_layer])  # (L,)
    ang_padded = np.concatenate([[0.0], ang_per_layer])  # (L,)

    features = np.concatenate([
        [mag_mean, ang_mean, coe_r, coe_c],
        mag_padded,
        ang_padded,
    ])

    names = ["CoE_mag", "CoE_ang", "CoE_R", "CoE_C"]
    for i, layer in enumerate(layers):
        names.append(f"CoE_mag_L{layer:02d}")
    for i, layer in enumerate(layers):
        names.append(f"CoE_ang_L{layer:02d}")

    return features.astype(np.float64), names


def compute_coe_batch(
    states_list: list[np.ndarray | None],
    layers: range | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute CoE features for all problems.

    Returns (n_problems, n_features) array and feature names.
    """
    results = []
    names = None
    for s in states_list:
        if s is None:
            if names is None:
                # Need to determine feature count
                dummy = np.zeros((N_TOTAL_LAYERS, 10, 1536), dtype=np.float16)
                _, names = compute_coe_single(dummy, layers)
            results.append(np.zeros(len(names), dtype=np.float64))
        else:
            feats, names = compute_coe_single(s, layers)
            results.append(feats)
    return np.stack(results), names
