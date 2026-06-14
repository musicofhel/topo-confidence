"""P11-FE292 — CAST full grid search for the correct/incorrect prefill contrast.

F-2 privileges L19 as *the* correctness layer, but the original analysis fit only
the DoM direction and never searched comparison directions/operators. CAST
(Conditional Activation Steering) routinely finds optimal condition layers in the
early-mid range. Here we run CAST's operator grid — (layer, threshold theta,
comparator > vs <) — across all per-layer prefill activations of Qwen-2.5-1.5B,
maximising F1 on the binary correct/incorrect target. If a layer != L19 wins, the
privileging of L19 is an artifact of an incomplete operator search.

Projection scores are computed out-of-fold (5-fold, DoM contrast direction fit on
train, projected on test) so the per-layer separability is not in-sample inflated;
the threshold/comparator grid (CAST's operator search) is then optimised on the OOF
scores. We report the winning layer, theta, comparator, its F1, and its directional
OOF AUROC, plus the full per-layer table.
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
# Per-layer prefill activations (all hidden-state layers). Several historical
# names exist; we probe candidates and adapt to whichever is present.
PERLAYER_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_alllayers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_perlayer.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_layers.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_all_layers.npz",
    ROOT / "pathway11_h100/data/perlayer/m15b_prefill_perlayer.npz",
]
PERLAYER_KEYS = ["prefill_all", "prefill_layers", "prefill_perlayer", "hidden",
                 "hidden_states", "layers", "prefill"]
OUT_JSON = ROOT / "pathway11_h100/cast_grid_search/results.json"

N_FOLDS = 5
SEED = 9999
HIDDEN = 1536
N_PROBLEMS = 500


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


def load_perlayer() -> tuple[np.ndarray, str, str]:
    """Return (X_all, path, key) with X_all shape (N_PROBLEMS, n_layers, HIDDEN)."""
    for path in PERLAYER_CANDIDATES:
        if not path.exists():
            continue
        blob = np.load(path)
        key = None
        for k in PERLAYER_KEYS:
            if k in blob.files:
                arr = blob[k]
                if arr.ndim == 3:
                    key = k
                    break
        if key is None:
            # fall back to the first 3-D array present
            for k in blob.files:
                if blob[k].ndim == 3:
                    key = k
                    break
        if key is None:
            continue
        arr = blob[key].astype(np.float64)
        arr = orient(arr)
        return arr, str(path), key
    raise FileNotFoundError


def orient(arr: np.ndarray) -> np.ndarray:
    """Reorder a 3-D array to (N_PROBLEMS, n_layers, HIDDEN)."""
    shp = arr.shape
    # identify hidden axis (== HIDDEN) and problem axis (== N_PROBLEMS)
    hid_ax = next((i for i, s in enumerate(shp) if s == HIDDEN), None)
    prob_ax = next((i for i, s in enumerate(shp) if s == N_PROBLEMS), None)
    if hid_ax is None or prob_ax is None or hid_ax == prob_ax:
        # cannot disambiguate; assume already (N, L, H)
        return arr
    layer_ax = ({0, 1, 2} - {hid_ax, prob_ax}).pop()
    return np.transpose(arr, (prob_ax, layer_ax, hid_ax))


def oof_projection(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    """OOF DoM-contrast projection scores for one layer's activations (N, H)."""
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        d = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        nrm = np.linalg.norm(d)
        if nrm > 0:
            d = d / nrm
        scores[test_idx] = X[test_idx] @ d
    return scores


def best_f1_grid(scores: np.ndarray, y: np.ndarray) -> dict:
    """CAST operator search: pick (theta, comparator) maximising F1 for the
    positive (correct) class on OOF projection scores."""
    order = np.sort(np.unique(scores))
    if len(order) > 1:
        thetas = (order[:-1] + order[1:]) / 2.0
        thetas = np.concatenate([[order[0] - 1e-9], thetas, [order[-1] + 1e-9]])
    else:
        thetas = np.array([order[0] - 1e-9, order[0] + 1e-9])
    pos_total = int(y.sum())
    best = {"f1": -1.0, "theta": float("nan"), "direction": ">"}
    for comp in (">", "<"):
        if comp == ">":
            preds = scores[None, :] > thetas[:, None]
        else:
            preds = scores[None, :] < thetas[:, None]
        tp = (preds & y[None, :]).sum(axis=1).astype(np.float64)
        pp = preds.sum(axis=1).astype(np.float64)
        prec = np.divide(tp, pp, out=np.zeros_like(tp), where=pp > 0)
        rec = tp / pos_total if pos_total > 0 else np.zeros_like(tp)
        denom = prec + rec
        f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
        bi = int(np.argmax(f1))
        if f1[bi] > best["f1"]:
            best = {"f1": float(f1[bi]), "theta": float(thetas[bi]),
                    "direction": comp}
    return best


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    y = np.load(CACHE)["correct"].astype(bool)
    assert y.shape == (N_PROBLEMS,)

    try:
        X_all, perlayer_path, perlayer_key = load_perlayer()
    except FileNotFoundError:
        print("MISSING_REGEN_INPUT", "per-layer prefill cache (tried: "
              + ", ".join(str(p) for p in PERLAYER_CANDIDATES) + ")",
              file=sys.stderr)
        return 2

    if X_all.shape[0] != N_PROBLEMS or X_all.shape[2] != HIDDEN:
        print("MISSING_REGEN_INPUT",
              f"unexpected per-layer shape {X_all.shape}", file=sys.stderr)
        return 2

    n_layers = X_all.shape[1]
    folds = stratified_kfold(y, N_FOLDS, SEED)

    per_layer = []
    for li in range(n_layers):
        scores = oof_projection(X_all[:, li, :], y, folds)
        grid = best_f1_grid(scores, y)
        # directional AUROC matching the winning comparator
        if grid["direction"] == ">":
            au = auroc(scores, y)
        else:
            au = auroc(-scores, y)
        per_layer.append({
            "layer": li,
            "best_f1": grid["f1"],
            "theta": grid["theta"],
            "direction": grid["direction"],
            "auroc": float(au),
        })

    best_idx = int(np.argmax([d["best_f1"] for d in per_layer]))
    best = per_layer[best_idx]

    # F1 of L19 specifically (if present) for the head-to-head with F-2
    l19 = next((d for d in per_layer if d["layer"] == 19), None)

    out = {
        "experiment": "P11-FE292",
        "perlayer_cache": perlayer_path,
        "perlayer_key": perlayer_key,
        "n_layers": int(n_layers),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "best_layer": best["layer"],
        "best_f1": best["best_f1"],
        "best_theta": best["theta"],
        "best_direction": best["direction"],
        "best_layer_auroc": best["auroc"],
        "l19": l19,
        "l19_is_winner": (best["layer"] == 19),
        "per_layer": per_layer,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())