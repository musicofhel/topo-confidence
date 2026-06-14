"""P11-FE478 — STA-style atom selection on cached L19 prefill activations.

Treats each of the 1536 residual-stream dimensions as a candidate "atom" (no
SAE — raw residual coordinates are the atom basis). For each atom we compute,
between correct and incorrect MATH-500 problems on Qwen-2.5-1.5B:

  - Δa (amplitude diff): |mean(correct) - mean(incorrect)| of the standardized
    coordinate (mass-mean-style amplitude).
  - Δf (frequency diff): |P(coord > θ | correct) - P(coord > θ | incorrect)|,
    the difference in activation frequency above a per-atom median gate θ.

Atoms surviving a joint threshold (Δa ≥ α-quantile AND Δf ≥ β-quantile) are fed
to an L2 logistic probe. Selection + standardization + θ are fit on the training
fold only and applied OOF (5-fold) to avoid leakage. We sweep (α, β) over a
quantile grid and report the selected-atom probe AUROC against the single
L19-DoM direction baseline (0.7731). A sparse multi-atom probe beating DoM by
≥0.03 would force F-2's "single direction" framing toward "low-dim sparse code".
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
from sklearn.linear_model import LogisticRegression

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/sta_atom_selection/results.json"

N_FOLDS = 5
SEED = 9999
DOM_BASELINE = 0.7731
PROBE_C = 0.01

# Quantile grid for the joint (Δa, Δf) threshold sweep. A value of q keeps the
# top (1-q) fraction of atoms on that criterion; (0.0, 0.0) keeps all 1536.
ALPHA_QUANTILES = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]
BETA_QUANTILES = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99]


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


def atom_stats(Xz: np.ndarray, y: np.ndarray, theta: np.ndarray):
    """Per-atom amplitude diff Δa and frequency diff Δf on standardized coords."""
    pos = Xz[y]; neg = Xz[~y]
    da = np.abs(pos.mean(axis=0) - neg.mean(axis=0))
    fpos = (pos > theta[None, :]).mean(axis=0)
    fneg = (neg > theta[None, :]).mean(axis=0)
    df = np.abs(fpos - fneg)
    return da, df


def oof_probe_auroc(X, y, folds, aq, bq):
    """OOF selected-atom probe AUROC for one (α-quantile, β-quantile) pair.

    Standardization, the activation gate θ, atom selection, and the probe are
    all fit on the training fold only. Returns (auroc, mean_n_atoms)."""
    n = len(y)
    oof = np.full(n, np.nan, dtype=np.float64)
    n_atoms = []
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = y[train_mask]

        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0); sd[sd < 1e-8] = 1.0
        Ztr = (Xtr - mu) / sd
        Zte = (Xte - mu) / sd

        theta = np.median(Ztr, axis=0)  # per-atom activation gate (train only)
        da, df = atom_stats(Ztr, ytr, theta)

        a_thr = np.quantile(da, aq) if aq > 0 else -np.inf
        b_thr = np.quantile(df, bq) if bq > 0 else -np.inf
        sel = np.flatnonzero((da >= a_thr) & (df >= b_thr))
        if sel.size == 0:
            sel = np.array([int(np.argmax(da))])  # never select zero atoms
        n_atoms.append(int(sel.size))

        clf = LogisticRegression(C=PROBE_C, max_iter=2000, solver="lbfgs")
        clf.fit(Ztr[:, sel], ytr)
        oof[test_idx] = clf.decision_function(Zte[:, sel])

    return auroc(oof, y), float(np.mean(n_atoms))


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    dom_auroc = auroc(dom_score, y)

    folds = stratified_kfold(y, N_FOLDS, SEED)

    grid = []
    best = {"auroc": -1.0, "alpha_q": None, "beta_q": None, "n_atoms": None}
    for aq in ALPHA_QUANTILES:
        for bq in BETA_QUANTILES:
            au, na = oof_probe_auroc(X, y, folds, aq, bq)
            grid.append({"alpha_q": aq, "beta_q": bq, "auroc": au, "n_atoms": na})
            if not np.isnan(au) and au > best["auroc"]:
                best = {"auroc": au, "alpha_q": aq, "beta_q": bq, "n_atoms": na}

    # Full-coordinate L2 probe (no selection) as an upper reference point.
    full_au, full_na = oof_probe_auroc(X, y, folds, 0.0, 0.0)

    delta_vs_dom = best["auroc"] - DOM_BASELINE
    out = {
        "experiment": "P11-FE478",
        "description": "STA-style atom selection (raw L19 residual coords as atoms) vs single-direction DoM",
        "n_problems": int(len(y)),
        "n_atoms_total": int(X.shape[1]),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "probe_C": PROBE_C,
        "dom_baseline_reported": DOM_BASELINE,
        "auroc_dom_oof_recomputed": float(dom_auroc),
        "auroc_full_coord_probe_oof": float(full_au),
        "best_selected": best,
        "delta_best_vs_dom_baseline": float(delta_vs_dom),
        "beats_dom_by_0p03": bool(delta_vs_dom >= 0.03),
        "grid": grid,
        "verdict": (
            "REFRAME_F2_SPARSE_CODE" if delta_vs_dom >= 0.03
            else "F2_SINGLE_DIRECTION_HOLDS"
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"best STA-probe AUROC={best['auroc']:.4f} "
          f"(α_q={best['alpha_q']}, β_q={best['beta_q']}, "
          f"~{best['n_atoms']:.0f} atoms) vs DoM {DOM_BASELINE} "
          f"→ Δ={delta_vs_dom:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())