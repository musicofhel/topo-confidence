"""P11-FE640 — Pooled-position contrastive activation means (CAA Eq 22, survey 2601.14004).

The survey formalizes our DoM as the canonical contrastive-activation-addition (CAA)
protocol of Eq 22: v = mean(x_correct over all positions) - mean(x_incorrect over all
positions), using a SINGLE global position pool rather than our split prefill/final
binning. This script computes that pooled direction over the cached L19 prefill+final
position activations and measures its OOF AUROC against our prefill-only and final-only
DoMs. If the pooled v aligns with the prefill DoM (high cosine, matching AUROC), then
F-3's reported prefill/final orthogonality (cos ~= 0.046) is plausibly a
position-binning artifact rather than a structural property of the representation.
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

# The final-token L19 activations live in a sibling cache. Try the symmetric
# names/locations; bail with MISSING_REGEN_INPUT if none resolve.
FINAL_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/final_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltoken.npz",
]
FINAL_KEYS = ("final", "final_token", "finaltoken", "hidden", "prefill")

OUT_JSON = ROOT / "pathway11_h100/pooled_caa/results.json"

N_FOLDS = 5
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


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


def _extract_500x1536(blob) -> np.ndarray | None:
    for key in FINAL_KEYS:
        if key in blob.files:
            arr = np.asarray(blob[key])
            if arr.shape == (500, 1536):
                return arr.astype(np.float64)
    for key in blob.files:
        arr = np.asarray(blob[key])
        if arr.shape == (500, 1536):
            return arr.astype(np.float64)
    return None


def resolve_final() -> np.ndarray | None:
    for path in FINAL_CANDIDATES:
        if path.exists():
            arr = _extract_500x1536(np.load(path))
            if arr is not None:
                return arr
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    prefill = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert prefill.shape == (500, 1536) and y.shape == (500,)

    final = resolve_final()
    if final is None:
        print("MISSING_REGEN_INPUT", "no m15b_final L19 cache found among:",
              *[str(p) for p in FINAL_CANDIDATES], file=sys.stderr)
        return 2

    n = len(y)
    folds = stratified_kfold(y, N_FOLDS, SEED)

    prefill_oof = np.zeros(n, dtype=np.float64)
    final_oof = np.zeros(n, dtype=np.float64)
    pooled_oof = np.zeros(n, dtype=np.float64)

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = y[train_mask]

        d_pre = prefill[train_mask][ytr].mean(0) - prefill[train_mask][~ytr].mean(0)
        d_fin = final[train_mask][ytr].mean(0) - final[train_mask][~ytr].mean(0)

        # CAA Eq 22 global pool: every position (prefill + final) of every train
        # problem is a sample, labelled by that problem's correctness.
        X_pool = np.vstack([prefill[train_mask], final[train_mask]])
        y_pool = np.concatenate([ytr, ytr])
        v_pool = X_pool[y_pool].mean(0) - X_pool[~y_pool].mean(0)

        prefill_oof[test_idx] = prefill[test_idx] @ d_pre
        final_oof[test_idx] = final[test_idx] @ d_fin
        # Score a held-out problem by projecting its own pooled (mean) position.
        pooled_rep = 0.5 * (prefill[test_idx] + final[test_idx])
        pooled_oof[test_idx] = pooled_rep @ v_pool

    # Full-data directions for cosine geometry.
    d_pre_full = prefill[y].mean(0) - prefill[~y].mean(0)
    d_fin_full = final[y].mean(0) - final[~y].mean(0)
    X_pool_full = np.vstack([prefill, final])
    y_pool_full = np.concatenate([y, y])
    v_pool_full = X_pool_full[y_pool_full].mean(0) - X_pool_full[~y_pool_full].mean(0)

    # Canonical committed prefill DoM score cross-check.
    canon_auroc = float("nan")
    if DOM_NPZ.exists():
        canon = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        canon_auroc = auroc(canon, y)

    auroc_pooled = auroc(pooled_oof, y)
    auroc_prefill = auroc(prefill_oof, y)
    auroc_final = auroc(final_oof, y)
    cos_pooled_prefill = cosine(v_pool_full, d_pre_full)
    cos_pooled_final = cosine(v_pool_full, d_fin_full)
    cos_prefill_final = cosine(d_pre_full, d_fin_full)

    out = {
        "experiment": "P11-FE640",
        "n_problems": int(n),
        "n_correct": int(y.sum()),
        "auroc_pooled_caa_oof": auroc_pooled,
        "auroc_prefill_dom_oof": auroc_prefill,
        "auroc_final_dom_oof": auroc_final,
        "auroc_canonical_prefill_score": canon_auroc,
        "cos_pooled_vs_prefill": cos_pooled_prefill,
        "cos_pooled_vs_final": cos_pooled_final,
        "cos_prefill_vs_final": cos_prefill_final,
        # F-3 orthogonality is an artifact only if the global pool both collapses
        # toward prefill AND recovers prefill-level AUROC.
        "pooled_aligns_with_prefill": bool(
            (not np.isnan(cos_pooled_prefill)) and abs(cos_pooled_prefill) > 0.90
        ),
        "pooled_matches_prefill_auroc": bool(
            (not np.isnan(auroc_pooled)) and (not np.isnan(auroc_prefill))
            and abs(auroc_pooled - auroc_prefill) < 0.01
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())