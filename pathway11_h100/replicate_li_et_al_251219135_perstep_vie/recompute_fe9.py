"""P8-FE9 — Per-step Vietoris–Rips persistent homology on cached MATH-500 1.5B K=1 traces.

Replicates Li et al. (2512.19135): each generation is segmented by '\\n\\n' into reasoning
steps, every step embedded with all-mpnet-base-v2 (768-d), and a Vietoris–Rips filtration is
built over the per-step embedding point cloud. We extract |H0| (number of finite H0 bars),
|H1| (number of H1 loops), and the combined persistent entropy, then:

  1. Score each topology feature's AUROC for K=1 correctness.
  2. Refit a logistic regression with (L19_DoM, |H0|, |H1|, persistent_entropy) and test
     whether the topology coefficients remain nonzero once DoM is controlled (bootstrap CIs +
     OOF AUROC of DoM-only vs DoM+topology).

This is a substrate-shifted refutation test for F-10's PH null (per-token residuals) and F-9's
"CoE redundant with DoM" claim — here the substrate is per-step *semantic* embeddings, not
per-token residuals. If |H0| survives partialling DoM, F-9 needs a substrate qualifier and we
have a third axis besides DoM and participation ratio.

NOTE ON INPUTS: the per-step mpnet embeddings are produced offline (GPU/network step) and cached
as pathway11_h100/data/perstep_mpnet/problem_NNN.npz with key 'emb' (n_steps, 768). This recompute
script is CPU-only and consumes those cached embeddings; the PH (H0 via union-find, H1 via Z/2
boundary reduction) is implemented in pure numpy/scipy — no ripser dependency. If the embedding
cache is absent it prints MISSING_REGEN_INPUT and returns 2.
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
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
EMB_DIR = ROOT / "pathway11_h100/data/perstep_mpnet"
OUT_JSON = ROOT / "pathway11_h100/perstep_ph/results.json"

N_PROBLEMS = 500
N_FOLDS = 5
SEED = 9999
MAX_STEPS = 48          # cap point-cloud size; VR triangles are O(n^3)
EPS = 1e-9              # persistence noise floor


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def persistent_entropy(lengths: np.ndarray) -> float:
    lengths = lengths[lengths > EPS]
    if lengths.size == 0:
        return 0.0
    p = lengths / lengths.sum()
    return float(-(p * np.log(p)).sum())


def vr_persistence(points: np.ndarray):
    """Vietoris–Rips persistence up to H1 on an (m, d) point cloud.

    Returns (h0_deaths, h1_bars) where h0_deaths are finite H0 bar deaths (births at 0) and
    h1_bars is an array of (birth, death) for H1 loops. Pure Z/2 boundary-matrix reduction.
    """
    m = points.shape[0]
    if m < 2:
        return np.zeros(0), np.zeros((0, 2))

    # Subsample evenly along the trajectory to bound combinatorial cost.
    if m > MAX_STEPS:
        sel = np.unique(np.linspace(0, m - 1, MAX_STEPS).round().astype(int))
        points = points[sel]
        m = points.shape[0]

    D = squareform(pdist(points))

    # Build simplices: vertices (dim0), edges (dim1), triangles (dim2).
    key_to_idx: dict = {}
    simplices = []  # (filt, dim, key)
    for v in range(m):
        key_to_idx[("v", v)] = len(simplices)
        simplices.append((0.0, 0, ("v", v)))
    for i in range(m):
        for j in range(i + 1, m):
            key_to_idx[("e", i, j)] = len(simplices)
            simplices.append((float(D[i, j]), 1, ("e", i, j)))
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                f = float(max(D[i, j], D[i, k], D[j, k]))
                simplices.append((f, 2, ("t", i, j, k)))

    n_simp = len(simplices)
    order = sorted(range(n_simp), key=lambda s: (simplices[s][0], simplices[s][1]))
    rank = np.empty(n_simp, dtype=np.int64)
    for pos, s in enumerate(order):
        rank[s] = pos
    filt_by_pos = np.array([simplices[order[p]][0] for p in range(n_simp)])
    dim_by_pos = np.array([simplices[order[p]][1] for p in range(n_simp)])

    def boundary_ranks(simp_idx):
        _, dim, key = simplices[simp_idx]
        if dim == 0:
            return set()
        if dim == 1:
            _, i, j = key
            return {int(rank[key_to_idx[("v", i)]]), int(rank[key_to_idx[("v", j)]])}
        _, i, j, k = key
        return {
            int(rank[key_to_idx[("e", i, j)]]),
            int(rank[key_to_idx[("e", i, k)]]),
            int(rank[key_to_idx[("e", j, k)]]),
        }

    R: dict = {}          # pivot position -> column (set of positions)
    h0_deaths = []
    h1_bars = []
    for p in range(n_simp):
        col = boundary_ranks(order[p])
        while col:
            piv = max(col)
            if piv in R:
                col ^= R[piv]
            else:
                break
        if col:
            piv = max(col)
            R[piv] = col
            b_dim = int(dim_by_pos[piv])
            birth = float(filt_by_pos[piv])
            death = float(filt_by_pos[p])
            if b_dim == 0:
                if death - birth > EPS:
                    h0_deaths.append(death)          # birth is 0 (vertex)
            elif b_dim == 1:
                if death - birth > EPS:
                    h1_bars.append((birth, death))

    return np.asarray(h0_deaths, dtype=np.float64), np.asarray(h1_bars, dtype=np.float64).reshape(-1, 2)


def load_embeddings(idx: int):
    for name in (f"problem_{idx:03d}.npz", f"problem_{idx + 1:03d}.npz"):
        path = EMB_DIR / name
        if path.exists():
            blob = np.load(path)
            key = "emb" if "emb" in blob.files else blob.files[0]
            emb = blob[key].astype(np.float64)
            if emb.ndim == 2 and emb.shape[0] >= 1:
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                return emb / norms
            return None
    return None


def main() -> int:
    for required in (CACHE, DOM_NPZ):
        if not required.exists():
            print("MISSING_REGEN_INPUT", required, file=sys.stderr)
            return 2
    if not EMB_DIR.exists() or not any(EMB_DIR.glob("problem_*.npz")):
        print("MISSING_REGEN_INPUT", EMB_DIR, file=sys.stderr)
        return 2

    correct = np.load(CACHE)["correct"].astype(bool)
    dom = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert correct.shape == (N_PROBLEMS,) and dom.shape == (N_PROBLEMS,)

    h0_count = np.full(N_PROBLEMS, np.nan)
    h1_count = np.full(N_PROBLEMS, np.nan)
    pers_entropy = np.full(N_PROBLEMS, np.nan)
    n_steps = np.full(N_PROBLEMS, np.nan)

    for idx in range(N_PROBLEMS):
        emb = load_embeddings(idx)
        if emb is None:
            continue
        n_steps[idx] = emb.shape[0]
        h0_deaths, h1_bars = vr_persistence(emb)
        h0_count[idx] = float(h0_deaths.size)
        h1_count[idx] = float(h1_bars.shape[0])
        bar_lengths = np.concatenate([
            h0_deaths,
            (h1_bars[:, 1] - h1_bars[:, 0]) if h1_bars.size else np.zeros(0),
        ])
        pers_entropy[idx] = persistent_entropy(bar_lengths)

    valid = ~np.isnan(h0_count)
    n_valid = int(valid.sum())
    if n_valid < 2 * N_FOLDS:
        print("MISSING_REGEN_INPUT too few valid problems:", n_valid, file=sys.stderr)
        return 2

    y = correct[valid]
    dom_v = dom[valid]
    h0_v = h0_count[valid]
    h1_v = h1_count[valid]
    pe_v = pers_entropy[valid]

    # Univariate AUROCs (sign-agnostic — report raw orientation).
    uni = {
        "dom": auroc(dom_v, y),
        "h0_count": auroc(h0_v, y),
        "h1_count": auroc(h1_v, y),
        "persistent_entropy": auroc(pe_v, y),
    }

    # Standardized feature matrix for logistic regression.
    feats = np.column_stack([dom_v, h0_v, h1_v, pe_v])
    feat_names = ["dom", "h0_count", "h1_count", "persistent_entropy"]
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (feats - mu) / sd

    # OOF AUROC: DoM-only vs DoM+topology.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_dom = np.zeros(n_valid)
    oof_full = np.zeros(n_valid)
    for tr, te in skf.split(Z, y):
        m_dom = LogisticRegression(max_iter=2000, C=1.0)
        m_dom.fit(Z[tr][:, :1], y[tr])
        oof_dom[te] = m_dom.predict_proba(Z[te][:, :1])[:, 1]
        m_full = LogisticRegression(max_iter=2000, C=1.0)
        m_full.fit(Z[tr], y[tr])
        oof_full[te] = m_full.predict_proba(Z[te])[:, 1]

    auroc_dom_oof = auroc(oof_dom, y)
    auroc_full_oof = auroc(oof_full, y)

    # Full-data standardized coefficients + bootstrap CIs (are topo coeffs nonzero given DoM?).
    full = LogisticRegression(max_iter=2000, C=1.0)
    full.fit(Z, y)
    coefs = full.coef_.ravel()

    rng = np.random.default_rng(SEED)
    n_boot = 1000
    boot = np.zeros((n_boot, Z.shape[1]))
    for b in range(n_boot):
        bi = rng.integers(0, n_valid, n_valid)
        if y[bi].sum() == 0 or y[bi].sum() == len(bi):
            boot[b] = np.nan
            continue
        mb = LogisticRegression(max_iter=2000, C=1.0)
        mb.fit(Z[bi], y[bi])
        boot[b] = mb.coef_.ravel()

    coef_report = {}
    for j, name in enumerate(feat_names):
        col = boot[:, j]
        col = col[~np.isnan(col)]
        lo, hi = np.percentile(col, [2.5, 97.5])
        same_sign = float(np.mean(np.sign(col) == np.sign(coefs[j]))) if col.size else float("nan")
        coef_report[name] = {
            "coef_std": float(coefs[j]),
            "boot_ci95": [float(lo), float(hi)],
            "nonzero_at_95": bool(lo > 0 or hi < 0),
            "frac_same_sign": same_sign,
        }

    topo_names = ["h0_count", "h1_count", "persistent_entropy"]
    any_topo_nonzero = any(coef_report[n]["nonzero_at_95"] for n in topo_names)

    out = {
        "experiment": "P8-FE9",
        "description": "Per-step VR-PH (mpnet embeddings) vs F-10 null / F-9 redundancy",
        "n_valid_problems": n_valid,
        "max_steps_cap": MAX_STEPS,
        "n_steps_mean": float(np.nanmean(n_steps)),
        "n_steps_max": float(np.nanmax(n_steps)),
        "univariate_auroc": {k: float(v) for k, v in uni.items()},
        "logreg_oof_auroc_dom_only": float(auroc_dom_oof),
        "logreg_oof_auroc_dom_plus_topo": float(auroc_full_oof),
        "logreg_oof_lift": float(auroc_full_oof - auroc_dom_oof),
        "standardized_coefficients": coef_report,
        "any_topology_coef_nonzero_given_dom": bool(any_topo_nonzero),
        "verdict": (
            "TOPO_SURVIVES_DOM_CONTROL" if any_topo_nonzero
            else "TOPO_REDUNDANT_WITH_DOM"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())