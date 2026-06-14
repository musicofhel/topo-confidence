"""P11-FE141 — Label-free mean-centred direction vs supervised DoM at L19.

Tests whether F-2's supervised correct-vs-incorrect prefill direction (DoM,
AUROC=0.7731) can be recovered from a label-free target/training mean
difference in the spirit of Jorgensen et al. 2312.03813.

We form f_mc = µ(correct prefill activations at L19) − µ(OpenWebText L19
activations), report its AUROC against the 1024-tok MATH-500 correctness
labels (both in-sample and OOF 5-fold, recomputing the correct-mean per
train fold while the OWT mean is a fixed label-free anchor), and measure
cos(f_mc, supervised_DoM_L19) where DoM = µ(correct) − µ(incorrect).

Decision rule (from the FE rationale): AUROC(f_mc) ≥ 0.74 and
cos(f_mc, DoM) > 0.9 ⇒ F-2's framing must be re-stated and H-12 gains a
cheap label-free probe.
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
OWT_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/owt_l19.npz"
OUT_JSON = ROOT / "pathway11_h100/label_free_meancentred/results.json"

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


def _load_owt_mean(d: int) -> np.ndarray:
    """Load OWT L19 activations and return their mean (1, d).

    The OWT NPZ schema is not fixed; accept any 2-D float array whose second
    axis matches the model width, or a precomputed mean vector of length d.
    """
    blob = np.load(OWT_CACHE)
    # Preferred explicit keys first.
    for key in ("owt", "hidden", "activations", "prefill", "l19"):
        if key in blob.files:
            arr = np.asarray(blob[key], dtype=np.float64)
            break
    else:
        # Fall back to the first array in the archive.
        arr = np.asarray(blob[blob.files[0]], dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != d:
            raise ValueError(f"OWT mean vector dim {arr.shape[0]} != {d}")
        return arr
    if arr.ndim == 2:
        if arr.shape[1] != d:
            raise ValueError(f"OWT activation width {arr.shape[1]} != {d}")
        return arr.mean(axis=0)
    raise ValueError(f"unexpected OWT array shape {arr.shape}")


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not OWT_CACHE.exists():
        print("MISSING_REGEN_INPUT", OWT_CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)
    n, d = X.shape

    owt_mean = _load_owt_mean(d)  # (d,)

    # Supervised DoM direction at L19: µ(correct) − µ(incorrect).
    dom = X[y].mean(axis=0) - X[~y].mean(axis=0)

    # Label-free mean-centred direction: µ(correct) − µ(OWT).
    f_mc = X[y].mean(axis=0) - owt_mean

    # cos(f_mc, DoM).
    def cos(a, b):
        na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return float("nan")
        return float((a @ b) / (na * nb))

    cos_fmc_dom = cos(f_mc, dom)

    # In-sample AUROCs (directions fit on all data; optimistic).
    auroc_fmc_insample = auroc(X @ f_mc, y)
    auroc_dom_insample = auroc(X @ dom, y)

    # OOF 5-fold: recompute the correct-mean per train fold; the OWT mean is a
    # fixed label-free anchor, so f_mc remains label-light out of sample.
    folds = stratified_kfold(y, N_FOLDS, SEED)
    oof_fmc = np.zeros(n, dtype=np.float64)
    oof_dom = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        f_mc_tr = Xtr[ytr].mean(axis=0) - owt_mean
        dom_tr = Xtr[ytr].mean(axis=0) - Xtr[~ytr].mean(axis=0)
        oof_fmc[test_idx] = X[test_idx] @ f_mc_tr
        oof_dom[test_idx] = X[test_idx] @ dom_tr

    auroc_fmc_oof = auroc(oof_fmc, y)
    auroc_dom_oof = auroc(oof_dom, y)

    decision = bool(auroc_fmc_oof >= 0.74 and cos_fmc_dom > 0.9)

    out = {
        "experiment": "P11-FE141",
        "n": int(n),
        "n_correct": int(y.sum()),
        "n_incorrect": int((~y).sum()),
        "cos_fmc_dom": cos_fmc_dom,
        "auroc_fmc_insample": float(auroc_fmc_insample),
        "auroc_dom_insample": float(auroc_dom_insample),
        "auroc_fmc_oof": float(auroc_fmc_oof),
        "auroc_dom_oof": float(auroc_dom_oof),
        "norm_fmc": float(np.linalg.norm(f_mc)),
        "norm_dom": float(np.linalg.norm(dom)),
        "decision_label_free_matches_supervised": decision,
        "decision_rule": "auroc_fmc_oof >= 0.74 AND cos_fmc_dom > 0.9",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())