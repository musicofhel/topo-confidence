"""Utility functions for topo-confidence."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@contextmanager
def timer(label: str):
    """Context manager that logs elapsed time."""
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    logger.info("%s: %.3fs", label, dt)


def resolve_device(device: str) -> str:
    """Resolve 'auto' to the best available device."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_layers(layers: list[int] | str, n_layers: int) -> list[int]:
    """Convert layer specification to concrete indices.

    Args:
        layers: "last" (final layer only), "terminal" (last 5), or explicit list.
        n_layers: total number of hidden layers in the model.

    Returns:
        List of 0-indexed layer indices.
    """
    if isinstance(layers, list):
        return [i % n_layers for i in layers]
    if layers == "last":
        return [n_layers - 1]
    if layers == "terminal":
        return list(range(max(0, n_layers - 5), n_layers))
    raise ValueError(f"Unknown layer spec: {layers!r}")


def subsample_points(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly subsample a point cloud to at most n points."""
    if len(points) <= n:
        return points
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


def persistence_entropy(lifetimes: np.ndarray) -> float:
    """Shannon entropy of normalized persistence lifetimes.

    Measures how uniformly distributed the lifetimes of topological features are.
    High entropy = many features with similar lifetimes = complex structure.
    Low entropy = one dominant feature = simple structure.
    """
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def load_yaml_config(path: str) -> dict[str, Any]:
    """Load a YAML configuration file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}
