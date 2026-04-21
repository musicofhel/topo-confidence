"""TwoNN intrinsic dimension estimation per layer.

Uses scikit-dimension's TwoNN estimator. Works on n>=20 points in any
ambient dimension. Returns one scalar (estimated intrinsic dim) per layer.
"""
from __future__ import annotations

import numpy as np

from pathway8_layerwise.config import N_TOTAL_LAYERS
from pathway8_layerwise.extraction_utils import subsample_tokens


def compute_twonn_per_layer(
    states: np.ndarray,
    layers: range | None = None,
    subsample_n: int = 100,
) -> tuple[np.ndarray, list[str]]:
    """TwoNN intrinsic dimension per layer for a single problem.

    Parameters
    ----------
    states : (29, n_tokens, 1536) — all-layer hidden states
    layers : which layers (default: 1-28)
    subsample_n : max tokens

    Returns
    -------
    features : (L,) intrinsic dimension per layer
    names : feature name list
    """
    from skdim.id import TwoNN

    if layers is None:
        layers = range(1, N_TOTAL_LAYERS)

    dims = np.full(len(layers), np.nan, dtype=np.float64)
    names = [f"TwoNN_ID_L{layer:02d}" for layer in layers]

    for i, layer in enumerate(layers):
        tokens = states[layer].astype(np.float32)
        tokens = subsample_tokens(tokens, subsample_n)

        if tokens.shape[0] < 10:
            continue

        try:
            est = TwoNN(discard_fraction=0.1)
            est.fit(tokens)
            dims[i] = float(est.dimension_)
        except Exception:
            continue

    return dims, names


def compute_twonn_batch(
    states_list: list[np.ndarray | None],
    layers: range | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute TwoNN ID for all problems. Returns (n_problems, L) array."""
    results = []
    names = None

    for s in states_list:
        if s is None:
            if layers is None:
                layers = range(1, N_TOTAL_LAYERS)
            if names is None:
                names = [f"TwoNN_ID_L{l:02d}" for l in layers]
            results.append(np.full(len(names), np.nan, dtype=np.float64))
        else:
            feats, names = compute_twonn_per_layer(s, layers)
            results.append(feats)

    arr = np.stack(results)
    # Replace NaN with 0 for classifier input
    arr = np.nan_to_num(arr, nan=0.0)
    return arr, names
