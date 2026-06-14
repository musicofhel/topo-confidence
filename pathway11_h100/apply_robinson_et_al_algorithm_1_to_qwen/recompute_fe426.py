"""P11-FE426 — Robinson et al. Algorithm 1 on the Qwen-2.5-1.5B token embedding matrix.

Robinson, Dey & Chiang ("Token embeddings violate the manifold hypothesis") give a
deterministic per-token local-geometry test on the *input* embedding matrix W_E. This
script is a CPU reimplementation of that Algorithm-1 idea on a 20K stratified subsample
of the ~152K-token vocabulary (full pairwise NN against the subsample as the ambient
point cloud), producing for every sampled token:

  * local intrinsic dimension  (Levina–Bickel MLE over k nearest neighbours)
  * a *manifold* p-value        — curvature test of the log(rank)~log(radius) volume law
                                   (under a manifold the power law is straight; a
                                   significant quadratic term => violation)
  * a *fiber-bundle* p-value    — two-scale dimension-consistency test (inner ball vs
                                   outer annulus MLE dimension equality, z-test on the
                                   inverse-dimension statistic whose variance is known
                                   in closed form: Var(1/d_hat) = 1/((n-1) d^2))

Both p-values are Bonferroni-corrected over the number of valid tokens. The per-token
irregularity score is -log10(min(manifold_p_corr, fiber_p_corr)); larger => the token's
neighbourhood is less manifold-like. This table is the substrate for FE101 and the
L0-probe refutation of F-2.

Requires a cached embedding matrix at EMBED_NPZ (key 'embedding'/'weight'/'wte' ...,
shape (V, 1536)); extracted offline since this harness is CPU/no-transformers. If the
cache is absent the script prints MISSING_REGEN_INPUT and returns 2.

Full per-token arrays are also written to a companion NPZ for downstream FEs; the JSON
keeps summary statistics plus the most-irregular tokens.
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
from scipy import stats

ROOT = Path("/home/musicofhel/topo-confidence")
EMBED_NPZ = ROOT / "pathway11_h100/robinson_manifold/qwen15b_embeddings.npz"
OUT_JSON = ROOT / "pathway11_h100/robinson_manifold/results.json"
PER_TOKEN_NPZ = ROOT / "pathway11_h100/robinson_manifold/per_token_irregularity.npz"

SEED = 9999
N_SAMPLE = 20_000        # stratified subsample of the vocabulary
N_STRATA = 20            # strata by embedding-norm quantile
K_NEIGHBORS = 64         # neighbours used for the local-dimension MLE
KQ = K_NEIGHBORS + 16    # query buffer to survive exact-duplicate / zero distances
CHUNK = 1000             # query rows per distance block
ALPHA = 0.05             # rejection level (post Bonferroni)
TOP_K = 256              # most-irregular tokens reported in the JSON
EMBED_KEYS = ("embedding", "embeddings", "embed", "weight", "W_E", "wte", "arr_0")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def load_embedding() -> np.ndarray | None:
    if not EMBED_NPZ.exists():
        return None
    blob = np.load(EMBED_NPZ)
    for key in EMBED_KEYS:
        if key in blob.files:
            W = blob[key]
            break
    else:
        # fall back to the first 2-D array in the archive
        W = None
        for key in blob.files:
            arr = blob[key]
            if arr.ndim == 2:
                W = arr
                break
        if W is None:
            return None
    W = np.asarray(W)
    if W.ndim != 2 or W.shape[1] != 1536:
        return None
    return W.astype(np.float32, copy=False)


def stratified_subsample(norms: np.ndarray, n_target: int, seed: int,
                         n_strata: int) -> np.ndarray:
    """Sample n_target vocab indices stratified into n_strata norm-quantile bands."""
    rng = np.random.default_rng(seed)
    V = len(norms)
    n_target = min(n_target, V)
    edges = np.quantile(norms, np.linspace(0.0, 1.0, n_strata + 1))
    edges[-1] = np.inf
    strata = np.searchsorted(edges, norms, side="right") - 1
    strata = np.clip(strata, 0, n_strata - 1)
    picks: list[np.ndarray] = []
    for s in range(n_strata):
        idx = np.flatnonzero(strata == s)
        if idx.size == 0:
            continue
        take = int(round(n_target * idx.size / V))
        take = max(1, min(take, idx.size))
        picks.append(rng.choice(idx, size=take, replace=False))
    chosen = np.unique(np.concatenate(picks))
    # trim / pad to exactly n_target deterministically
    if chosen.size > n_target:
        chosen = rng.choice(chosen, size=n_target, replace=False)
    elif chosen.size < n_target:
        remaining = np.setdiff1d(np.arange(V), chosen, assume_unique=False)
        extra = rng.choice(remaining, size=n_target - chosen.size, replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen)


def neighbour_distances(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, valid): R is (n, K) sorted positive NN distances; valid masks tokens
    with at least K_NEIGHBORS strictly-positive neighbours."""
    n = A.shape[0]
    A_T = np.ascontiguousarray(A.T)               # (d, n) float32
    anorm = (A.astype(np.float64) ** 2).sum(axis=1)
    R = np.full((n, K_NEIGHBORS), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    eps2 = 1e-18

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        Q = A[start:stop]
        d2 = anorm[start:stop, None] + anorm[None, :] - 2.0 * (Q.astype(np.float64) @ A_T)
        np.maximum(d2, 0.0, out=d2)
        kq = min(KQ, n)
        part = np.argpartition(d2, kth=kq - 1, axis=1)[:, :kq]
        for r in range(stop - start):
            cand = d2[r, part[r]]
            cand.sort()
            pos = cand[cand > eps2]               # drop self + exact duplicates
            if pos.size >= K_NEIGHBORS:
                R[start + r] = np.sqrt(pos[:K_NEIGHBORS])
                valid[start + r] = True
    return R, valid


def main() -> int:
    W = load_embedding()
    if W is None:
        print("MISSING_REGEN_INPUT", EMBED_NPZ, file=sys.stderr)
        return 2

    V = W.shape[0]
    norms = np.linalg.norm(W.astype(np.float64), axis=1)
    sample_idx = stratified_subsample(norms, N_SAMPLE, SEED, N_STRATA)
    A = W[sample_idx]
    n = A.shape[0]

    R, valid = neighbour_distances(A)             # (n, K), bool
    logR = np.log(R)                              # nan-safe where invalid (nan in, nan out)

    K = K_NEIGHBORS
    h = K // 2

    # --- Levina-Bickel MLE local dimension (full k-neighbourhood) -------------
    ref_lg = logR[:, K - 1][:, None]
    inv_large = np.mean(ref_lg - logR[:, : K - 1], axis=1)      # = 1/d_hat
    local_dim = 1.0 / inv_large

    # --- Fiber-bundle two-scale dimension-consistency test --------------------
    n_in = h
    n_out = (K - 1) - h
    inv_in = np.mean(logR[:, h][:, None] - logR[:, :h], axis=1)
    inv_out = np.mean(logR[:, K - 1][:, None] - logR[:, h : K - 1], axis=1)
    var = inv_large ** 2 * (1.0 / n_in + 1.0 / n_out)
    z = (inv_in - inv_out) / np.sqrt(var)
    p_fiber = 2.0 * stats.norm.sf(np.abs(z))

    # --- Manifold curvature test on the log(rank) ~ log(radius) power law -----
    x = logR                                       # (n, K)
    y = np.log(np.arange(1, K + 1, dtype=np.float64))   # shared across rows
    Sy = y.sum()
    Sy2 = (y * y).sum()
    x2 = x * x
    Sx = x.sum(axis=1)
    Sx2 = x2.sum(axis=1)
    Sx3 = (x2 * x).sum(axis=1)
    Sx4 = (x2 * x2).sum(axis=1)
    Sxy = (x * y).sum(axis=1)
    Sx2y = (x2 * y).sum(axis=1)

    M = np.empty((n, 3, 3), dtype=np.float64)
    M[:, 0, 0] = Sx4; M[:, 0, 1] = Sx3; M[:, 0, 2] = Sx2
    M[:, 1, 0] = Sx3; M[:, 1, 1] = Sx2; M[:, 1, 2] = Sx
    M[:, 2, 0] = Sx2; M[:, 2, 1] = Sx;  M[:, 2, 2] = K
    M += 1e-9 * np.eye(3)[None, :, :]
    b = np.stack([Sx2y, Sxy, np.full(n, Sy)], axis=1)          # (n, 3)

    p_manifold = np.ones(n, dtype=np.float64)
    ok = valid & np.isfinite(M).all(axis=(1, 2)) & np.isfinite(b).all(axis=1)
    if ok.any():
        Mo, bo = M[ok], b[ok]
        coeffs = np.linalg.solve(Mo, bo)                       # [a, b, c] per row
        Minv = np.linalg.inv(Mo)
        sse = Sy2 - np.einsum("ij,ij->i", coeffs, bo)
        sse = np.clip(sse, 0.0, None)
        s2 = sse / (K - 3)
        var_a = s2 * Minv[:, 0, 0]
        se_a = np.sqrt(np.clip(var_a, 1e-300, None))
        t = coeffs[:, 0] / se_a
        p_manifold[ok] = 2.0 * stats.t.sf(np.abs(t), df=K - 3)

    # invalid tokens -> non-informative p-values / nan dimension
    p_fiber = np.where(valid, p_fiber, 1.0)
    p_manifold = np.where(valid, p_manifold, 1.0)
    local_dim = np.where(valid, local_dim, np.nan)

    # --- Bonferroni correction over valid tokens ------------------------------
    n_valid = int(valid.sum())
    m = max(n_valid, 1)
    p_manifold_corr = np.minimum(1.0, p_manifold * m)
    p_fiber_corr = np.minimum(1.0, p_fiber * m)
    min_corr = np.minimum(p_manifold_corr, p_fiber_corr)
    irregularity = -np.log10(np.clip(min_corr, 1e-300, 1.0))
    irregularity = np.where(valid, irregularity, 0.0)

    rej_manifold = int((p_manifold_corr[valid] < ALPHA).sum())
    rej_fiber = int((p_fiber_corr[valid] < ALPHA).sum())
    rej_any = int((min_corr[valid] < ALPHA).sum())

    # --- companion per-token table (substrate for FE101) ----------------------
    PER_TOKEN_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PER_TOKEN_NPZ,
        token_idx=sample_idx.astype(np.int64),
        valid=valid,
        local_dim=local_dim.astype(np.float64),
        embed_norm=norms[sample_idx].astype(np.float64),
        p_manifold=p_manifold.astype(np.float64),
        p_fiber=p_fiber.astype(np.float64),
        p_manifold_corr=p_manifold_corr.astype(np.float64),
        p_fiber_corr=p_fiber_corr.astype(np.float64),
        irregularity=irregularity.astype(np.float64),
    )

    order = np.argsort(-irregularity)[:TOP_K]
    top_tokens = [
        {
            "vocab_index": int(sample_idx[i]),
            "local_dim": float(local_dim[i]) if np.isfinite(local_dim[i]) else None,
            "embed_norm": float(norms[sample_idx[i]]),
            "p_manifold_corr": float(p_manifold_corr[i]),
            "p_fiber_corr": float(p_fiber_corr[i]),
            "irregularity": float(irregularity[i]),
        }
        for i in order
    ]

    vd = local_dim[valid & np.isfinite(local_dim)]
    hist_counts, hist_edges = np.histogram(vd, bins=30, range=(0.0, 30.0))

    out = {
        "experiment": "P11-FE426",
        "method": "Robinson et al. Algorithm 1 (CPU reimpl) on Qwen-2.5-1.5B W_E",
        "vocab_size": int(V),
        "embed_dim": int(W.shape[1]),
        "n_sample": int(n),
        "n_valid": n_valid,
        "k_neighbors": K_NEIGHBORS,
        "n_strata": N_STRATA,
        "alpha": ALPHA,
        "seed": SEED,
        "local_dim_mean": float(np.nanmean(local_dim)),
        "local_dim_median": float(np.nanmedian(local_dim)),
        "local_dim_std": float(np.nanstd(local_dim)),
        "local_dim_min": float(np.nanmin(local_dim)),
        "local_dim_max": float(np.nanmax(local_dim)),
        "ambient_dim": int(W.shape[1]),
        "bonferroni_m": m,
        "rej_manifold_count": rej_manifold,
        "rej_manifold_frac": rej_manifold / m,
        "rej_fiber_count": rej_fiber,
        "rej_fiber_frac": rej_fiber / m,
        "rej_any_count": rej_any,
        "rej_any_frac": rej_any / m,
        "local_dim_hist_counts": [int(c) for c in hist_counts],
        "local_dim_hist_edges": [float(e) for e in hist_edges],
        "top_irregular_tokens": top_tokens,
        "per_token_npz": str(PER_TOKEN_NPZ.relative_to(ROOT)),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())