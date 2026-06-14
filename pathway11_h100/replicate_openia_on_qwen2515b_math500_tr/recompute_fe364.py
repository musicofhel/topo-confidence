"""P11-FE364 — OPENIA replication on Qwen-2.5-1.5B MATH-500.

OPENIA's headline claim is that a small MLP probe on the *last-decoded-token at
the last layer* (here L27) is the strongest single internal-state correctness
probe in the code domain. F-2 claims the opposite ordering on math: a *prefill*
L19 DoM direction (AUROC 0.7731 OOF) beats the final-token readout (0.7186).

This is a direct head-to-head test of whether F-2's prefill > final-token
ordering is a Qwen+math idiosyncrasy or a genuine cross-domain regularity. We
train the OPENIA-style 128/64 MLP probe on the cached last-layer last-decoded
token activations (OOF 5-fold) and compare its AUROC against:
  (a) the prefill L19 DoM direction (F-2, recomputed OOF for a fair split), and
  (b) an identical 128/64 MLP trained on the prefill L19 features.

All inputs are cached pathway11_h100 NPZs; CPU only.
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

# OPENIA probes the last-decoded-token at the last layer (L27). Candidate cache
# paths, tried in order; the first that exists and carries a (500, D) hidden
# array is used.
FINAL_CACHE_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_l27.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_lasttoken_l27.npz",
    ROOT / "pathway11_h100/data/m15b_final_l27.npz",
]

OUT_JSON = ROOT / "pathway11_h100/openia_replication/results.json"

SEED = 9999
N_FOLDS = 5
HIDDEN = (128, 64)

# F-2 canonical OOF prefill DoM AUROC (1024-tok labels), for reference.
F2_PREFILL_DOM_AUROC = 0.7731


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]; neg = scores[~labels]
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


def _find_hidden_key(blob, n_expected: int):
    """Return the name of the first (n_expected, D) float-ish array key, else None."""
    preferred = ["final", "hidden", "last_token", "lasttoken", "final_hidden", "l27", "states"]
    keys = list(blob.files)
    for name in preferred:
        if name in keys:
            arr = blob[name]
            if arr.ndim == 2 and arr.shape[0] == n_expected:
                return name
    for name in keys:
        arr = blob[name]
        if arr.ndim == 2 and arr.shape[0] == n_expected and arr.shape[1] >= 8:
            return name
    return None


def oof_mlp_scores(X: np.ndarray, y: np.ndarray, folds, seed: int) -> np.ndarray:
    """Out-of-fold positive-class probabilities from a 128/64 MLP per fold."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        scaler = StandardScaler().fit(X[train_mask])
        Xtr = scaler.transform(X[train_mask])
        Xte = scaler.transform(X[test_idx])
        clf = MLPClassifier(
            hidden_layer_sizes=HIDDEN,
            activation="relu",
            alpha=1e-3,
            max_iter=500,
            early_stopping=False,
            random_state=seed,
        )
        clf.fit(Xtr, y[train_mask].astype(int))
        proba = clf.predict_proba(Xte)
        pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else proba.shape[1] - 1
        scores[test_idx] = proba[:, pos_col]
    return scores


def oof_dom_scores(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """Out-of-fold difference-of-means projection (F-2 style)."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        ytr = y[train_mask]
        d = X[train_mask][ytr].mean(axis=0) - X[train_mask][~ytr].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d
    return scores


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X_prefill = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X_prefill.shape == (500, 1536) and y.shape == (500,)

    # Locate the OPENIA last-layer last-token cache.
    final_path = next((p for p in FINAL_CACHE_CANDIDATES if p.exists()), None)
    if final_path is None:
        print("MISSING_REGEN_INPUT", FINAL_CACHE_CANDIDATES[0], file=sys.stderr)
        return 2
    fblob = np.load(final_path)
    hkey = _find_hidden_key(fblob, n_expected=500)
    if hkey is None:
        print("MISSING_REGEN_INPUT", f"{final_path} (no (500,D) hidden array)", file=sys.stderr)
        return 2
    X_final = fblob[hkey].astype(np.float64)
    # Prefer the cache's own labels if present, else fall back to prefill labels.
    if "correct" in fblob.files and fblob["correct"].shape[0] == 500:
        y_final = fblob["correct"].astype(bool)
    else:
        y_final = y
    assert X_final.shape[0] == 500

    folds = stratified_kfold(y, N_FOLDS, SEED)
    folds_final = folds if np.array_equal(y_final, y) else stratified_kfold(y_final, N_FOLDS, SEED)

    # (1) OPENIA MLP on last-layer last-decoded token.
    final_mlp = oof_mlp_scores(X_final, y_final, folds_final, SEED)
    auroc_openia_mlp = auroc(final_mlp, y_final)

    # (2) F-2 prefill L19 DoM, recomputed OOF on the same split, plus the
    #     cached canonical DoM score for cross-check.
    prefill_dom_oof = oof_dom_scores(X_prefill, y, folds)
    auroc_prefill_dom_oof = auroc(prefill_dom_oof, y)
    cached_dom = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    auroc_cached_dom = auroc(cached_dom, y)

    # (3) Apples-to-apples: identical 128/64 MLP on prefill L19 features.
    prefill_mlp = oof_mlp_scores(X_prefill, y, folds, SEED)
    auroc_prefill_mlp = auroc(prefill_mlp, y)

    prefill_wins = auroc_prefill_dom_oof > auroc_openia_mlp
    out = {
        "experiment": "P11-FE364",
        "description": "OPENIA last-layer-last-token MLP probe vs F-2 prefill L19 DoM on Qwen-2.5-1.5B MATH-500",
        "n": 500,
        "final_cache_used": str(final_path),
        "final_cache_hidden_key": hkey,
        "final_cache_dim": int(X_final.shape[1]),
        "auroc_openia_mlp_final_l27": auroc_openia_mlp,
        "auroc_prefill_dom_oof": auroc_prefill_dom_oof,
        "auroc_prefill_dom_cached": auroc_cached_dom,
        "auroc_prefill_mlp": auroc_prefill_mlp,
        "f2_prefill_dom_canonical": F2_PREFILL_DOM_AUROC,
        "delta_prefill_dom_minus_openia": auroc_prefill_dom_oof - auroc_openia_mlp,
        "prefill_beats_openia": bool(prefill_wins),
        "verdict": (
            "F2_ORDERING_HOLDS prefill DoM > OPENIA final-token MLP on math"
            if prefill_wins
            else "F2_ORDERING_OVERTURNED OPENIA final-token MLP >= prefill DoM on math"
        ),
        "n_folds": N_FOLDS,
        "mlp_hidden_layer_sizes": list(HIDDEN),
        "seed": SEED,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())