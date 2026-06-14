"""P11-FE1303 — Is the DoM correctness direction a principal singular read of L19 MLP weights?

Computes the leading residual-space singular vector v1 of Qwen-2.5-1.5B layer-19
MLP weights (gate_proj, up_proj read directions; down_proj write direction) via
the paper's single-step matrix power-iteration (MPI) update (R·W·W^T, Eq. ~10),
then measures the paper's lambda projection metric (Eq. 11) between v1 and the
cached L19 prefill DoM correctness direction (DoM ≈ PC1, cos 0.9216).

lambda is reported two ways:
  - cosine alignment |cos(DoM, v1)|  (a "principal singular read" should be ~1)
  - captured-energy fraction ||W·DoM||^2 / sigma_1^2   (how much of the top
    singular gain DoM actually exercises)
both compared against an isotropic random-1536d-vector baseline (mean/std/z).

Premise under test (H-740 / H-744 / paper "principal direction is most
expressive"): if DoM is a *dominant* weight read, lambda is high (z >> 0); if it
is an *off-principal tail* read, lambda collapses toward the random baseline.

Weights are NOT in the standard activation caches and cannot be loaded here
(no torch/transformers permitted). They must be pre-extracted offline to:
  pathway11_h100/weight_svd/cache/m15b_L19_mlp.npz
with keys gate_proj (8960,1536), up_proj (8960,1536), down_proj (1536,8960)
(alt key names *_weight / W_gate etc. are accepted). Regenerate with a one-off
offline dump of model.model.layers[19].mlp.{gate,up,down}_proj.weight.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
WEIGHTS = ROOT / "pathway11_h100/weight_svd/cache/m15b_L19_mlp.npz"
OUT_JSON = ROOT / "pathway11_h100/weight_svd/results.json"

HIDDEN = 1536
SEED = 9999
N_RANDOM = 4000  # random unit vectors for the isotropic baseline


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def _get(blob, *names):
    for n in names:
        if n in blob.files:
            return blob[n].astype(np.float64)
    return None


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def mpi_single_step(W: np.ndarray, side: str, seed: int) -> np.ndarray:
    """Paper's single-step power iteration for the dominant residual-space
    singular vector. side='right' (read dir, dim=n_cols) or 'left' (write dir,
    dim=n_rows). Starts from a random vector R and applies one W·W^T pass."""
    rng = np.random.default_rng(seed)
    if side == "right":  # v ← W^T (W R)
        r = rng.standard_normal(W.shape[1])
        v = W.T @ (W @ r)
    else:  # left:  v ← W (W^T R)
        r = rng.standard_normal(W.shape[0])
        v = W @ (W.T @ r)
    return _unit(v)


def exact_top_singular(W: np.ndarray, side: str):
    """Exact dominant residual-space singular vector + singular value, for
    reference (the 1536×1536 gram is cheap). Returns (v1_unit, sigma1)."""
    gram = (W.T @ W) if side == "right" else (W @ W.T)  # 1536×1536
    evals, evecs = np.linalg.eigh(gram)
    v1 = evecs[:, -1]
    sigma1 = float(np.sqrt(max(evals[-1], 0.0)))
    return _unit(v1), sigma1


def lambda_metrics(d_unit, W, side, v1_exact, sigma1):
    """Eq.11 lambda projection metrics of unit direction d against W's top
    singular structure: cosine alignment with v1, and captured-energy fraction."""
    cos_v1 = float(abs(d_unit @ v1_exact))
    if side == "right":
        Wd = W @ d_unit
    else:  # for write side, exercise W^T d (pull d back through write map)
        Wd = W.T @ d_unit
    gain_sq = float(Wd @ Wd)
    energy_frac = gain_sq / (sigma1 ** 2) if sigma1 > 0 else float("nan")
    return cos_v1, energy_frac, float(np.sqrt(gain_sq))


def random_baseline(v1_exact, W, side, sigma1, seed):
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((N_RANDOM, HIDDEN))
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    cos = np.abs(R @ v1_exact)
    if side == "right":
        WR = R @ W.T  # (N, out)
    else:
        WR = R @ W      # apply W^T·r since W is (1536, in) -> r@W gives (N,in)
    gain_sq = np.einsum("ij,ij->i", WR, WR)
    energy = gain_sq / (sigma1 ** 2) if sigma1 > 0 else np.full(N_RANDOM, np.nan)
    return {
        "cos_v1_mean": float(cos.mean()),
        "cos_v1_std": float(cos.std()),
        "energy_frac_mean": float(np.nanmean(energy)),
        "energy_frac_std": float(np.nanstd(energy)),
    }


def analyze(name, W, side, d_unit, seed):
    v1_mpi = mpi_single_step(W, side, seed)
    v1_exact, sigma1 = exact_top_singular(W, side)
    mpi_vs_exact = float(abs(v1_mpi @ v1_exact))  # single-step convergence quality

    cos_v1, energy_frac, gain = lambda_metrics(d_unit, W, side, v1_exact, sigma1)
    cos_v1_mpi = float(abs(d_unit @ v1_mpi))
    base = random_baseline(v1_exact, W, side, sigma1, seed + 1)

    z_cos = ((cos_v1 - base["cos_v1_mean"]) / base["cos_v1_std"]
             if base["cos_v1_std"] > 0 else float("nan"))
    z_energy = ((energy_frac - base["energy_frac_mean"]) / base["energy_frac_std"]
                if base["energy_frac_std"] > 0 else float("nan"))
    return {
        "matrix": name,
        "side": side,
        "shape": list(W.shape),
        "sigma1": sigma1,
        "mpi_single_step_vs_exact_cos": mpi_vs_exact,
        "lambda_cos_v1_exact": cos_v1,
        "lambda_cos_v1_mpi": cos_v1_mpi,
        "lambda_energy_frac": energy_frac,
        "dom_gain": gain,
        "random_baseline": base,
        "z_cos": float(z_cos),
        "z_energy": float(z_energy),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not WEIGHTS.exists():
        print("MISSING_REGEN_INPUT", WEIGHTS, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, HIDDEN) and y.shape == (500,)

    # DoM correctness direction (≈ PC1, cos 0.9216) — and its sanity AUROC.
    dom = X[y].mean(axis=0) - X[~y].mean(axis=0)
    dom_unit = _unit(dom)
    dom_auroc = auroc(X @ dom_unit, y)

    wb = np.load(WEIGHTS)
    gate = _get(wb, "gate_proj", "gate_proj_weight", "W_gate", "gate")
    up = _get(wb, "up_proj", "up_proj_weight", "W_up", "up")
    down = _get(wb, "down_proj", "down_proj_weight", "W_down", "down")
    if gate is None or up is None or down is None:
        print("MISSING_REGEN_INPUT", "weight keys gate/up/down not found in", WEIGHTS,
              file=sys.stderr)
        return 2

    # Orient each so the residual (1536) axis is the read/write axis.
    def orient_in(W):   # want shape (out, 1536): right singular = read direction
        return W if W.shape[1] == HIDDEN else W.T

    def orient_out(W):  # want shape (1536, in): left singular = write direction
        return W if W.shape[0] == HIDDEN else W.T

    gate, up, down = orient_in(gate), orient_in(up), orient_out(down)

    results = [
        analyze("gate_proj", gate, "right", dom_unit, SEED),
        analyze("up_proj", up, "right", dom_unit, SEED + 100),
        analyze("down_proj", down, "left", dom_unit, SEED + 200),
    ]

    # Verdict: is DoM a principal singular read of any L19 MLP map?
    max_cos = max(r["lambda_cos_v1_exact"] for r in results)
    max_z = max((r["z_cos"] for r in results if np.isfinite(r["z_cos"])), default=float("nan"))
    if max_cos >= 0.30:
        verdict = "PRINCIPAL_READ"  # DoM aligns with a dominant weight singular vector
    elif np.isfinite(max_z) and max_z >= 3.0:
        verdict = "WEAK_NONRANDOM"  # above isotropic noise but off-principal
    else:
        verdict = "OFF_PRINCIPAL_TAIL"  # refutes 'principal direction is most expressive'

    out = {
        "experiment": "P11-FE1303",
        "hidden_dim": HIDDEN,
        "dom_self_auroc": float(dom_auroc),
        "dom_norm": float(np.linalg.norm(dom)),
        "n_random_baseline": N_RANDOM,
        "per_matrix": results,
        "max_lambda_cos_v1_exact": float(max_cos),
        "max_z_cos": float(max_z),
        "verdict": verdict,
        "interpretation": (
            "PRINCIPAL_READ supports H-740/H-744 (DoM is a dominant singular read "
            "of L19 weights); OFF_PRINCIPAL_TAIL refutes the paper's 'principal "
            "direction is most expressive' premise for this setting."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("verdict:", verdict, "max_cos:", round(max_cos, 4), "max_z:", round(max_z, 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())