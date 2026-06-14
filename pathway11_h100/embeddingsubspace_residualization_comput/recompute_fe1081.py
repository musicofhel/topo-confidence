"""P11-FE1081 — Embedding-subspace residualization of the L19 DoM direction.

Tests whether the L19 prefill DoM direction is aligned with the token-identity
subspace spanned by the Qwen-2.5-1.5B input-embedding matrix restricted to the
tokens that appear in MATH-500.

Procedure
---------
1. Load the embedding rows for MATH-500 tokens (precomputed NPZ — the model
   cannot be loaded under the CPU-only / no-transformers constraint).
2. Compute the top-k principal components of the centered embedding matrix.
   These span the "token-identity" subspace.
3. Project the L19 prefill (and, if available, final-token) activations onto the
   *complement* of that subspace: X_resid = X - (X @ V) @ V.T.
4. Recompute the DoM AUROC (OOF, 5-fold) in the residualized space and sweep k.
5. Recompute cos(prefill_DoM_resid, final_DoM_resid).

Reads:
- prefill / final cache + DoM scores (documented schema)
- embedding-subspace cache (the only novel input; MISSING_REGEN_INPUT if absent)

Interpretation: if AUROC is preserved after removing the embedding subspace, the
correctness signal is genuinely contextual (not identity-confounded). If
cos(prefill_DoM_resid, final_DoM) rises toward 1, F-3's prefill/final
orthogonality was an identity-vs-context artifact.
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
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
# Embedding rows for MATH-500 tokens (vocab_subset, 1536). The only novel input;
# must be precomputed since the model cannot be loaded under CPU-only limits.
EMB_NPZ = ROOT / "pathway11_h100/embedding_subspace/qwen15b_math500_embeddings.npz"
# Optional final-token L19 cache for the orthogonality recomputation.
FINAL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
OUT_JSON = ROOT / "pathway11_h100/embedding_subspace/results.json"

SEED = 9999
N_FOLDS = 5
K_LIST = [10, 25, 50, 100, 200]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def emb_basis(E: np.ndarray, k: int) -> np.ndarray:
    """Top-k right singular vectors of centered E as an orthonormal (d, k) basis."""
    Ec = E - E.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k].T  # (d, k)


def residualize(X: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project X onto the complement of the column span of `basis` (orthonormal)."""
    return X - (X @ basis) @ basis.T


def dom_oof(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    n = X.shape[0]
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        d = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d
    return scores


def load_final(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    blob = np.load(path)
    for key in ("final", "final_hidden", "hidden", "prefill"):
        if key in blob.files:
            arr = blob[key].astype(np.float64)
            if arr.ndim == 2 and arr.shape[1] == 1536 and arr.shape[0] == 500:
                return arr
    return None


def main() -> int:
    for required in (CACHE, EMB_NPZ):
        if not required.exists():
            print("MISSING_REGEN_INPUT", required, file=sys.stderr)
            return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    emb_blob = np.load(EMB_NPZ)
    emb_key = next((k for k in ("embeddings", "embedding", "emb", "E") if k in emb_blob.files), None)
    if emb_key is None:
        print("MISSING_REGEN_INPUT", EMB_NPZ, "(no embeddings array)", file=sys.stderr)
        return 2
    E = emb_blob[emb_key].astype(np.float64)
    if E.ndim != 2 or E.shape[1] != 1536:
        print("MISSING_REGEN_INPUT", EMB_NPZ, f"(bad shape {E.shape})", file=sys.stderr)
        return 2

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Baseline: raw DoM (no residualization).
    raw_scores = dom_oof(X, y, folds)
    auroc_raw = auroc(raw_scores, y)
    prefill_dom_raw = X[y].mean(axis=0) - X[~y].mean(axis=0)

    # Optional final-token activations for the orthogonality recomputation.
    Xf = load_final(FINAL_CACHE)
    final_available = Xf is not None
    final_dom_raw = (Xf[y].mean(axis=0) - Xf[~y].mean(axis=0)) if final_available else None
    cos_raw = cosine(prefill_dom_raw, final_dom_raw) if final_available else None

    sweep = []
    for k in K_LIST:
        basis = emb_basis(E, k)
        Xr = residualize(X, basis)
        resid_scores = dom_oof(Xr, y, folds)
        prefill_dom_resid = Xr[y].mean(axis=0) - Xr[~y].mean(axis=0)

        entry = {
            "k": int(basis.shape[1]),
            "auroc_resid_oof": auroc(resid_scores, y),
            "auroc_delta_vs_raw": auroc(resid_scores, y) - auroc_raw,
            "fraction_prefill_norm_removed": float(
                1.0 - (np.linalg.norm(Xr) ** 2) / (np.linalg.norm(X) ** 2)
            ),
            "cos_prefill_dom_raw_vs_resid": cosine(prefill_dom_raw, prefill_dom_resid),
        }
        if final_available:
            Xfr = residualize(Xf, basis)
            final_dom_resid = Xfr[y].mean(axis=0) - Xfr[~y].mean(axis=0)
            entry["cos_prefill_resid_vs_final_resid"] = cosine(prefill_dom_resid, final_dom_resid)
            entry["cos_prefill_resid_vs_final_raw"] = cosine(prefill_dom_resid, final_dom_raw)
        sweep.append(entry)

    out = {
        "experiment": "P11-FE1081",
        "n": int(len(y)),
        "embedding_tokens": int(E.shape[0]),
        "auroc_raw_oof": auroc_raw,
        "final_available": final_available,
        "cos_prefill_dom_raw_vs_final_dom_raw": cos_raw,
        "k_sweep": sweep,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())