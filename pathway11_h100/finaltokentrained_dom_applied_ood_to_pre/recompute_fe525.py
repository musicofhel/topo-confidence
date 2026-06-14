"""P11-FE525 — Final-token-trained DoM applied OOD to prefill activations (MATH-500).

Tests whether F-2's "prefill > final-token" privilege is a property of the
*direction* or of the *measurement position*. Following Chan et al. 2507.12428
§5.2 ("present-trained" probes outperform position-specific probes when applied
OOD to early positions), we train the L19 difference-of-means (DoM) direction on
FINAL-token activations and apply it OOD to PREFILL activations, with proper 5-fold
OOF cross-fitting on the shared correctness labels.

Decision rule (vs the two anchor AUROCs):
  * if AUROC(final-trained DoM @ prefill) matches the prefill-trained 0.7731
    within 0.02 -> F-2's prefill privilege COLLAPSES (same direction, just
    measured at the easiest position).
  * if it stays near the final-token 0.7186 -> F-2 HOLDS (the privilege is real
    and tied to where the direction is learned).

Inputs:
  * prefill cache (prefill acts + shared correctness labels)
  * final-token L19 cache (final acts) — resolved from candidate paths
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
RESULTS_JSON = ROOT / "pathway11_h100/prefill_gated_compute/results.json"
OUT_JSON = ROOT / "pathway11_h100/finaltrained_dom_ood/results.json"

# Candidate locations / key names for the final-token L19 activation cache.
FINAL_CACHE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_finaltoken.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway11_h100/final_token/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_gated_compute/m15b_final.npz",
]
FINAL_KEY_CANDIDATES = ["final", "finaltoken", "final_token", "hidden", "acts"]

PREFILL_ANCHOR = 0.7731  # prefill-trained DoM AUROC (F-2 headline, 1024tok)
FINAL_ANCHOR = 0.7186    # final-token DoM AUROC (1024tok)
COLLAPSE_TOL = 0.02

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


def resolve_final_cache() -> tuple[np.ndarray | None, str | None, str | None]:
    """Return (final_acts, path_used, key_used) or (None, None, None)."""
    for path in FINAL_CACHE_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        for key in FINAL_KEY_CANDIDATES:
            if key in blob.files and blob[key].ndim == 2 and blob[key].shape == (500, 1536):
                return blob[key].astype(np.float64), str(path), key
        # fall back to any (500,1536) array in the file
        for key in blob.files:
            arr = blob[key]
            if getattr(arr, "ndim", 0) == 2 and arr.shape == (500, 1536):
                return arr.astype(np.float64), str(path), key
    return None, None, None


def dom_oof_scores(X_train_src: np.ndarray, X_apply: np.ndarray, y: np.ndarray,
                   folds: list[np.ndarray]) -> np.ndarray:
    """OOF: per fold, fit DoM on X_train_src train rows, score X_apply test rows."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = y[train_mask]
        d_vec = X_train_src[train_mask][ytr].mean(axis=0) - X_train_src[train_mask][~ytr].mean(axis=0)
        scores[test_idx] = X_apply[test_idx] @ d_vec
    return scores


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    blob = np.load(CACHE)
    Xp = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert Xp.shape == (500, 1536) and y.shape == (500,)

    Xf, final_path, final_key = resolve_final_cache()
    if Xf is None:
        print("MISSING_REGEN_INPUT (final-token L19 activation cache not found)",
              "; tried:", [str(p) for p in FINAL_CACHE_CANDIDATES], file=sys.stderr)
        return 2

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Core OOD test: final-token-trained DoM, applied to prefill activations.
    final_dom_on_prefill = dom_oof_scores(Xf, Xp, y, folds)
    auroc_final_on_prefill = auroc(final_dom_on_prefill, y)

    # References: in-position OOF AUROCs (anchors should reproduce ~0.7731 / ~0.7186).
    prefill_dom_on_prefill = dom_oof_scores(Xp, Xp, y, folds)
    auroc_prefill_on_prefill = auroc(prefill_dom_on_prefill, y)

    final_dom_on_final = dom_oof_scores(Xf, Xf, y, folds)
    auroc_final_on_final = auroc(final_dom_on_final, y)

    # Symmetric companion: prefill-trained DoM applied OOD to final-token acts.
    prefill_dom_on_final = dom_oof_scores(Xp, Xf, y, folds)
    auroc_prefill_on_final = auroc(prefill_dom_on_final, y)

    delta_vs_prefill = abs(auroc_final_on_prefill - PREFILL_ANCHOR)
    delta_vs_final = abs(auroc_final_on_prefill - FINAL_ANCHOR)
    collapses = bool(delta_vs_prefill <= COLLAPSE_TOL)

    if collapses:
        verdict = ("F2_PRIVILEGE_COLLAPSES: final-trained DoM scores prefill acts at "
                   f"{auroc_final_on_prefill:.4f}, within {COLLAPSE_TOL} of prefill anchor "
                   f"{PREFILL_ANCHOR}. Same direction, easiest measurement position.")
    elif delta_vs_final <= COLLAPSE_TOL:
        verdict = ("F2_HOLDS: final-trained DoM on prefill stays near final-token anchor "
                   f"{FINAL_ANCHOR} ({auroc_final_on_prefill:.4f}); prefill privilege is "
                   "tied to where the direction is learned.")
    else:
        verdict = ("INTERMEDIATE: final-trained DoM on prefill = "
                   f"{auroc_final_on_prefill:.4f}, between anchors {FINAL_ANCHOR} and "
                   f"{PREFILL_ANCHOR}; partial transfer of the direction.")

    out = {
        "experiment": "P11-FE525",
        "description": "final-token-trained L19 DoM applied OOD to prefill activations",
        "n_folds": N_FOLDS,
        "seed": SEED,
        "final_cache_path": final_path,
        "final_cache_key": final_key,
        "auroc_finaltrained_dom_on_prefill_oof": auroc_final_on_prefill,
        "auroc_prefilltrained_dom_on_prefill_oof": auroc_prefill_on_prefill,
        "auroc_finaltrained_dom_on_final_oof": auroc_final_on_final,
        "auroc_prefilltrained_dom_on_final_oof": auroc_prefill_on_final,
        "prefill_anchor": PREFILL_ANCHOR,
        "final_anchor": FINAL_ANCHOR,
        "collapse_tolerance": COLLAPSE_TOL,
        "delta_vs_prefill_anchor": delta_vs_prefill,
        "delta_vs_final_anchor": delta_vs_final,
        "privilege_collapses": collapses,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())