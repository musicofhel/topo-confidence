"""P11-FE533 — Probe-transfers-across-positions test (FASB vs F-3 falsifier).

Trains FASB-style per-head linear probes on Qwen2.5-1.5B L19 activations at the
prefill-end position (the 1536-d residual split into 12 attention-head subspaces
of head_dim 128, per-head logistic probe + meta-logistic aggregator), then
evaluates the *same frozen probes* at two non-prefill positions:
  (a) final-token-of-CoT
  (b) every-fifth mid-CoT position.

A single-direction DoM probe (the F-3 construction) is carried alongside as a
contrast. F-3 predicts cross-position AUROC collapses toward chance
(cos(prefill_DoM, final_DoM) = 0.046 → near-orthogonal subspaces); FASB predicts
AUROC stays > 0.7 because the same theta serves both prefill classification and
per-token deviation detection. If FASB transfers, F-3's orthogonality claim is a
single-layer-DoM construction artifact, not a deep statement about generation
geometry.

Requires a per-position activation cache (final-token + stacked mid-CoT L19
states). If absent, prints MISSING_REGEN_INPUT and returns 2.
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
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
POS_CACHE = ROOT / "pathway11_h100/per_position/cache/m15b_per_position.npz"
OUT_JSON = ROOT / "pathway11_h100/probe_transfer_positions/results.json"

N_FOLDS = 5
SEED = 9999
N_HEADS = 12          # Qwen2.5-1.5B attention heads (12 * head_dim 128 = 1536)
LOGREG_C = 1.0
THRESH_HIGH = 0.70    # FASB-supported transfer floor
THRESH_CHANCE = 0.55  # F-3-supported (near-chance) ceiling


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
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


def head_slices(d: int, n_heads: int) -> list[slice]:
    if d % n_heads != 0:
        raise ValueError(f"dim {d} not divisible by {n_heads} heads")
    hd = d // n_heads
    return [slice(h * hd, (h + 1) * hd) for h in range(n_heads)]


def fit_fasb(X: np.ndarray, y: np.ndarray, slices: list[slice]) -> dict:
    """Per-head logistic probes (frozen StandardScaler) + meta-logistic aggregator."""
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    heads = []
    train_scores = np.zeros((len(y), len(slices)), dtype=np.float64)
    for j, sl in enumerate(slices):
        lr = LogisticRegression(max_iter=2000, C=LOGREG_C)
        lr.fit(Xs[:, sl], y)
        heads.append(lr)
        train_scores[:, j] = lr.decision_function(Xs[:, sl])
    meta = LogisticRegression(max_iter=2000, C=LOGREG_C).fit(train_scores, y)
    return {"scaler": scaler, "heads": heads, "meta": meta, "slices": slices}


def apply_fasb(model: dict, X: np.ndarray) -> np.ndarray:
    Xs = model["scaler"].transform(X)
    hs = np.column_stack(
        [m.decision_function(Xs[:, sl]) for m, sl in zip(model["heads"], model["slices"])]
    )
    return model["meta"].decision_function(hs)


def per_head_scores(model: dict, X: np.ndarray) -> np.ndarray:
    Xs = model["scaler"].transform(X)
    return np.column_stack(
        [m.decision_function(Xs[:, sl]) for m, sl in zip(model["heads"], model["slices"])]
    )


def fit_dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def verdict(auc: float) -> str:
    if np.isnan(auc):
        return "NO_DATA"
    if auc >= THRESH_HIGH:
        return "FASB_SUPPORTED"
    if auc <= THRESH_CHANCE:
        return "F3_SUPPORTED"
    return "AMBIGUOUS"


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not POS_CACHE.exists():
        print("MISSING_REGEN_INPUT", POS_CACHE, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X_prefill = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X_prefill.shape == (500, 1536) and y.shape == (500,)

    pos = np.load(POS_CACHE, allow_pickle=True)
    if "final_token" not in pos.files:
        print("MISSING_REGEN_INPUT", f"{POS_CACHE}:final_token", file=sys.stderr); return 2
    X_final = pos["final_token"].astype(np.float64)
    if X_final.shape != X_prefill.shape:
        print("MISSING_REGEN_INPUT", f"{POS_CACHE}:final_token shape {X_final.shape}",
              file=sys.stderr); return 2

    # Mid-CoT: stacked every-fifth positions with a per-row problem index.
    have_mid = "mid_cot" in pos.files and "mid_cot_idx" in pos.files
    if have_mid:
        X_mid = pos["mid_cot"].astype(np.float64)
        mid_idx = pos["mid_cot_idx"].astype(int)
        if X_mid.ndim != 2 or X_mid.shape[1] != 1536 or X_mid.shape[0] != mid_idx.shape[0]:
            have_mid = False
    if have_mid:
        y_mid = y[mid_idx]

    slices = head_slices(X_prefill.shape[1], N_HEADS)

    # ---- In-distribution prefill-end (OOF, both probes) ----
    n = len(y)
    fasb_oof = np.zeros(n, dtype=np.float64)
    dom_oof = np.zeros(n, dtype=np.float64)
    for test_idx in stratified_kfold(y, N_FOLDS, SEED):
        tr = np.ones(n, dtype=bool); tr[test_idx] = False
        model = fit_fasb(X_prefill[tr], y[tr], slices)
        fasb_oof[test_idx] = apply_fasb(model, X_prefill[test_idx])
        d = fit_dom(X_prefill[tr], y[tr])
        dom_oof[test_idx] = X_prefill[test_idx] @ d

    auc_fasb_prefill = auroc(fasb_oof, y)
    auc_dom_prefill = auroc(dom_oof, y)

    # ---- Frozen probes trained on ALL prefill data, transferred to other positions ----
    full_model = fit_fasb(X_prefill, y, slices)
    full_dom = fit_dom(X_prefill, y)

    # cos(prefill_DoM, final_DoM) sanity — the F-3 orthogonality quantity.
    final_dom = fit_dom(X_final, y)
    cos_dom = float(
        full_dom @ final_dom
        / (np.linalg.norm(full_dom) * np.linalg.norm(final_dom) + 1e-12)
    )

    # Final-token transfer.
    auc_fasb_final = auroc(apply_fasb(full_model, X_final), y)
    auc_dom_final = auroc(X_final @ full_dom, y)
    head_final = per_head_scores(full_model, X_final)
    per_head_final_auc = [auroc(head_final[:, j], y) for j in range(N_HEADS)]

    out = {
        "experiment": "P11-FE533",
        "description": "FASB per-head probe cross-position transfer vs F-3 orthogonality",
        "n_heads": N_HEADS,
        "n_problems": int(n),
        "cos_prefill_dom_final_dom": cos_dom,
        "prefill_end": {
            "auroc_fasb_oof": auc_fasb_prefill,
            "auroc_dom_oof": auc_dom_prefill,
        },
        "final_token": {
            "auroc_fasb_transfer": auc_fasb_final,
            "auroc_dom_transfer": auc_dom_final,
            "per_head_auroc": per_head_final_auc,
            "verdict": verdict(auc_fasb_final),
        },
    }

    # Mid-CoT transfer (pooled over positions + aggregated per problem).
    if have_mid:
        fasb_mid = apply_fasb(full_model, X_mid)
        dom_mid = X_mid @ full_dom
        auc_fasb_mid_pooled = auroc(fasb_mid, y_mid)
        auc_dom_mid_pooled = auroc(dom_mid, y_mid)

        # Per-problem: mean probe score across that problem's mid-CoT rows.
        agg_fasb = np.full(n, np.nan)
        agg_dom = np.full(n, np.nan)
        for p in np.unique(mid_idx):
            rows = mid_idx == p
            agg_fasb[p] = fasb_mid[rows].mean()
            agg_dom[p] = dom_mid[rows].mean()
        seen = ~np.isnan(agg_fasb)
        auc_fasb_mid_prob = auroc(agg_fasb[seen], y[seen])
        auc_dom_mid_prob = auroc(agg_dom[seen], y[seen])

        out["mid_cot"] = {
            "n_positions": int(X_mid.shape[0]),
            "n_problems_covered": int(seen.sum()),
            "auroc_fasb_pooled": auc_fasb_mid_pooled,
            "auroc_dom_pooled": auc_dom_mid_pooled,
            "auroc_fasb_per_problem": auc_fasb_mid_prob,
            "auroc_dom_per_problem": auc_dom_mid_prob,
            "verdict": verdict(auc_fasb_mid_pooled),
        }
    else:
        out["mid_cot"] = {"status": "MISSING_REGEN_INPUT",
                          "note": "POS_CACHE lacks mid_cot/mid_cot_idx; final-token transfer only"}

    # Overall adjudication: FASB falsifies F-3 only if it transfers at BOTH
    # available non-prefill positions; F-3 stands if transfer collapses to chance.
    transfer_aucs = [auc_fasb_final]
    if have_mid:
        transfer_aucs.append(out["mid_cot"]["auroc_fasb_pooled"])
    valid = [a for a in transfer_aucs if not np.isnan(a)]
    if valid and all(a >= THRESH_HIGH for a in valid):
        overall = "FASB_SUPPORTED_F3_ARTIFACT"
    elif valid and all(a <= THRESH_CHANCE for a in valid):
        overall = "F3_SUPPORTED_ORTHOGONAL_SUBSPACES"
    else:
        overall = "AMBIGUOUS"
    out["overall_verdict"] = overall

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"prefill FASB OOF={auc_fasb_prefill:.4f}  final FASB={auc_fasb_final:.4f}  "
          f"verdict={overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())