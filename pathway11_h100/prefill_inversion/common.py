"""Shared utilities for Experiment 3 — 7B prefill inversion."""
from __future__ import annotations

import numpy as np
from pathlib import Path

ROOT = Path("/home/musicofhel/topo-confidence")
STAGE1_7B_DIR = ROOT / "pathway11_h100/data/math500_7b"           # 29 × n_gen × 3584
STAGE2_15B_DIR = ROOT / "pathway8_layerwise/data/math500"          # 29 × n_gen × 1536
BBH_DIR = ROOT / "pathway8_layerwise/data/bbh"                     # 3 subsets × 250
OUT_DIR = ROOT / "pathway11_h100/prefill_inversion"

STEERING_LAYER = 19
N_MATH = 500
N_BBH_PER_SUBSET = 250
BBH_SUBSETS = ["tracking_shuffled_objects_seven_objects",
               "logical_deduction_seven_objects", "web_of_lies"]


def participation_ratio(X: np.ndarray) -> float:
    """Participation ratio of the empirical covariance of X.

    X: (n_samples, d). Returns PR = (sum(lambda))^2 / sum(lambda^2)
    where lambda are eigenvalues of the covariance.

    Equivalently: (tr(C))^2 / tr(C^2) without eigendecomposition.
    Uses SVD of centered X for numerical stability:
      PR = (sum(s^2))^2 / sum(s^4)
    where s are singular values of (X - mean) / sqrt(n-1).
    """
    if len(X) < 2:
        return 0.0
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD gives singular values of Xc. Eigvals of C = Xc^T Xc / (n-1) are s^2 / (n-1).
    # PR is scale-invariant w.r.t. (n-1) so we can just use s^2 directly.
    try:
        s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    except np.linalg.LinAlgError:
        # Fallback: eigendecomp of covariance via smaller side
        C = Xc @ Xc.T if Xc.shape[0] < Xc.shape[1] else Xc.T @ Xc
        eig = np.linalg.eigvalsh(C)
        eig = np.maximum(eig, 0)
        num = eig.sum() ** 2
        den = (eig ** 2).sum()
        return float(num / (den + 1e-30))
    s2 = s ** 2
    num = s2.sum() ** 2
    den = (s2 ** 2).sum()
    return float(num / (den + 1e-30))


def load_prefill_and_labels(
    data_dir: Path,
    n_problems: int,
    hidden_dim: int,
    layer: int = STEERING_LAYER,
    position: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Load (prefill-end L19, final-token L19, correctness, seq_len) arrays.

    position=0  → prefill-end (last prompt token, first entry of `states`)
    position=-1 → final generated token
    Returns prefill, final_tok, correct, seq_len, missing_indices.
    """
    prefill = np.zeros((n_problems, hidden_dim), dtype=np.float32)
    final_tok = np.zeros((n_problems, hidden_dim), dtype=np.float32)
    correct = np.zeros(n_problems, dtype=bool)
    seq_len = np.zeros(n_problems, dtype=int)
    missing = []
    for i in range(n_problems):
        fp = data_dir / f"problem_{i:03d}.npz"
        if not fp.exists():
            missing.append(i)
            continue
        with np.load(fp, allow_pickle=True) as d:
            s = d["states"]  # (29, n_gen, hidden_dim)
            prefill[i] = s[layer, 0, :].astype(np.float32)
            final_tok[i] = s[layer, -1, :].astype(np.float32)
            correct[i] = bool(d["correct"])
            seq_len[i] = s.shape[1]
    return prefill, final_tok, correct, seq_len, missing


def load_7b():
    return load_prefill_and_labels(STAGE1_7B_DIR, N_MATH, 3584)


def load_15b():
    return load_prefill_and_labels(STAGE2_15B_DIR, N_MATH, 1536)


def load_bbh_subset(name: str):
    return load_prefill_and_labels(BBH_DIR / name, N_BBH_PER_SUBSET, 1536)
