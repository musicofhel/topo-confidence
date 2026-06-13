"""activation_loader.py — one interface over every cache the panel needs.

Wraps the consolidated nocompute/cache/*.npz caches (X_mean / X_prefill /
X_last at all 29 layers + y + mean_logprob + n_gen_tokens + d2h_attn_entropy +
subjects) and the raw per-problem caches (full (29,T,H) trajectories, K=8
self-consistency) behind one loader. Knows each model's 2/3-depth layer.

Cache facts (verified 2026-06-11):
  nocompute/cache/math500_1p5b.npz : 500x(29,1536), subjects = MATH category
  nocompute/cache/math500_7b.npz   : 500x(29,3584), d2h_attn_entropy all-zero
  nocompute/cache/bbh_1p5b.npz     : 750x(29,1536), subsets array (3 BBH tasks)
  pathway8_layerwise/data/math500/problem_*.npz : full (29,T,1536) trajectory
  pathway11_h100/data/math500_7b/problem_*.npz  : full (29,T,3584) trajectory
  pathway11_h100/data/k8_selfconsistency/problem_*.npz : L19_samples (8,1536)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
NC_CACHE = ROOT / "nocompute" / "cache"
TRAJ_1P5B = ROOT / "pathway8_layerwise" / "data" / "math500"
TRAJ_7B = ROOT / "pathway11_h100" / "data" / "math500_7b"
K8_DIR = ROOT / "pathway11_h100" / "data" / "k8_selfconsistency"

N_LAYERS = 29
# 2/3-depth operating point per model (resolves the v1 fixed-layer collision).
# Qwen 29L -> L19; Qwen-7B 29L (HF reports 28 blocks +embed = 29 states) -> L19.
# Cross-arch depths added in Phase 2 when those caches are re-extracted.
TWO_THIRDS_LAYER = {
    "qwen1.5b": 19,
    "qwen7b": 19,
    "phi3": 21,        # 33L
    "llama3.2-1b": 11,  # 17L
}

# Map (model, dataset) -> consolidated cache file.
_CONSOLIDATED = {
    ("qwen1.5b", "math"): "math500_1p5b.npz",
    ("qwen7b", "math"): "math500_7b.npz",
    ("qwen1.5b", "bbh"): "bbh_1p5b.npz",
}


@dataclass
class Cell:
    """One (model, dataset[, subset]) panel cell, materialised as feature arrays."""
    model: str
    dataset: str
    y: np.ndarray              # (n,) bool — True = correct
    mean_logprob: np.ndarray   # (n,)
    n_gen_tokens: np.ndarray   # (n,)
    subjects: np.ndarray       # (n,) MATH category / BBH subset label
    _X_mean: np.ndarray        # (n, L, H)
    _X_prefill: np.ndarray     # (n, L, H)
    _X_last: np.ndarray        # (n, L, H)
    d2h_attn_entropy: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def layer(self) -> int:
        return TWO_THIRDS_LAYER[self.model]

    def X(self, position: str = "prefill", layer: int | None = None) -> np.ndarray:
        """Feature matrix at one layer/position. position in {prefill,last,mean}."""
        layer = self.layer if layer is None else layer
        arr = {"prefill": self._X_prefill, "last": self._X_last,
               "mean": self._X_mean}[position]
        return arr[:, layer, :].astype(np.float64)

    def X_all_layers(self, position: str = "mean") -> np.ndarray:
        """(n, L, H) at one position — for layer-profile / depth-grid features."""
        return {"prefill": self._X_prefill, "last": self._X_last,
                "mean": self._X_mean}[position].astype(np.float64)


def load_cell(model: str, dataset: str) -> Cell:
    """Load a consolidated panel cell (Qwen-1.5B MATH/BBH, Qwen-7B MATH)."""
    key = (model, dataset)
    if key not in _CONSOLIDATED:
        raise KeyError(f"No consolidated cache for {key}; available: "
                       f"{list(_CONSOLIDATED)}")
    d = np.load(NC_CACHE / _CONSOLIDATED[key], allow_pickle=True)
    subj_key = "subjects" if "subjects" in d.files else "subsets"
    entropy = d["d2h_attn_entropy"] if "d2h_attn_entropy" in d.files else None
    return Cell(
        model=model, dataset=dataset,
        y=d["y"].astype(bool),
        mean_logprob=d["mean_logprob"].astype(np.float64),
        n_gen_tokens=d["n_gen_tokens"].astype(np.float64),
        subjects=d[subj_key],
        _X_mean=d["X_mean"], _X_prefill=d["X_prefill"], _X_last=d["X_last"],
        d2h_attn_entropy=entropy,
    )


def loco_cells(cell: Cell) -> dict[str, np.ndarray]:
    """Return {subject: boolean mask} for leave-one-category-out (T1)."""
    return {s: (cell.subjects == s) for s in sorted(set(cell.subjects.tolist()))}


# ---------------------------------------------------------------------------
# Raw trajectory loader (Phase 1B.2 earliest-decision curve)
# ---------------------------------------------------------------------------

def load_trajectory(model: str, idx: int) -> tuple[np.ndarray, bool, float]:
    """Load one full (29, T, H) trajectory + label + mean_logprob."""
    base = {"qwen1.5b": TRAJ_1P5B, "qwen7b": TRAJ_7B}[model]
    with np.load(base / f"problem_{idx:03d}.npz", allow_pickle=True) as d:
        return (d["states"].astype(np.float32), bool(d["correct"]),
                float(d["mean_logprob"]))


def prefix_meanpool(model: str, idx: int, m: int, layer: int) -> np.ndarray:
    """Mean-pooled hidden state over the FIRST m generated tokens at `layer`.

    m=None / m>=T -> full generation. Returns (H,) float64.
    """
    states, _, _ = load_trajectory(model, idx)
    T = states.shape[1]
    mm = T if (m is None or m >= T) else m
    return states[layer, :mm, :].mean(axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# K=8 self-consistency loader (Phase 1B.1 probe-reranking)
# ---------------------------------------------------------------------------

def load_k8(idx: int) -> dict:
    """One K=8 problem: L19_samples (8,1536), texts (8,), correct (8,)."""
    with np.load(K8_DIR / f"problem_{idx:03d}.npz", allow_pickle=True) as d:
        return {"L19_samples": d["L19_samples"].astype(np.float64),
                "texts": d["texts"], "correct": d["correct"].astype(bool)}


def n_k8_problems() -> int:
    return len(list(K8_DIR.glob("problem_*.npz")))
