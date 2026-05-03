"""FE181 — Token-probability baseline vs prefill L19 DoM.

Direct head-to-head: does sequence-likelihood predict correctness as well as
F-2's 0.7731 prefill-DoM AUROC? Tests whether DoM is "just restating output
entropy."

Cached scope (this script): each per-problem NPZ already stores the scalar
`mean_logprob = mean_t log P(y_t | y<t, x)` over the answer span. We compute
AUROC of mean_logprob against the same `correct` labels used for F-2.

Out of scope (defer to fresh forward pass): per-token min, sum, and product
aggregations require per-token logprob arrays, which the per-problem cache
does not store. The cheap-cached mean-only test is sufficient for the
headline "DoM-vs-likelihood" question — additional aggregations are
follow-ups, not gates.

Reference: PLAN_cheap_wins.md §"Phase 2" — FE181 spec.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
NPZ_15B = ROOT / "pathway8_layerwise/data/math500"
NPZ_7B = ROOT / "pathway11_h100/data/math500_7b"
OUT_JSON = ROOT / "pathway11_h100/token_prob/results.json"

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


def oof_1d_oriented(x: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    """Single-feature DoM probe with fold-safe orientation."""
    folds = stratified_kfold(y, k, seed)
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train = np.ones(n, dtype=bool); train[test_idx] = False
        s = (x[train & y].mean() - x[train & ~y].mean())
        scores[test_idx] = x[test_idx] * np.sign(s + 1e-12)
    return auroc(scores, y)


def oof_2feat(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    folds = stratified_kfold(y, k, seed)
    n = len(y); scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train = np.ones(n, dtype=bool); train[test_idx] = False
        Xtr = X[train]
        d = X[train & y].mean(0) - X[train & ~y].mean(0)
        Xc = Xtr - Xtr.mean(0)
        Sigma = Xc.T @ Xc / max(len(Xtr) - 1, 1)
        Sigma += 1e-3 * np.trace(Sigma) / 2 * np.eye(2)
        w = np.linalg.solve(Sigma, d)
        scores[test_idx] = X[test_idx] @ w
    return auroc(scores, y)


def load_mean_logprob(npz_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(npz_dir.glob("problem_*.npz"))
    n = len(files)
    mean_logp = np.zeros(n, dtype=np.float64)
    correct = np.zeros(n, dtype=bool)
    seq_len = np.zeros(n, dtype=np.int64)
    for i, f in enumerate(files):
        d = np.load(f)
        mean_logp[i] = float(d["mean_logprob"])
        correct[i] = bool(d["correct"])
        seq_len[i] = int(d["states"].shape[1])
    return mean_logp, correct, seq_len


def main() -> int:
    if not NPZ_15B.exists():
        print(f"MISSING_REGEN_INPUT {NPZ_15B}", file=sys.stderr); return 2

    # ---- 1.5B ----
    mlp_15, y_15, len_15 = load_mean_logprob(NPZ_15B)
    auroc_mean_15 = oof_1d_oriented(mlp_15, y_15, N_FOLDS, SEED)
    # Sum-logprob = mean_logprob × seq_len.
    sum_15 = mlp_15 * len_15
    auroc_sum_15 = oof_1d_oriented(sum_15.astype(np.float64), y_15, N_FOLDS, SEED)

    # Joint [mean_logp, prefill_DoM_proj_L19] AUROC.
    prefill_path = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
    joint_auroc_15 = float("nan")
    if prefill_path.exists():
        pf = np.load(prefill_path)
        Xp = pf["prefill"].astype(np.float64); yp = pf["correct"].astype(bool)
        if len(yp) == len(y_15) and (yp == y_15).all():
            d_full = Xp[yp].mean(0) - Xp[~yp].mean(0)
            dom_proj = Xp @ d_full
            joint_auroc_15 = oof_2feat(np.column_stack([mlp_15, dom_proj]), y_15, N_FOLDS, SEED)

    # ---- 7B (if cache present) ----
    if NPZ_7B.exists() and len(list(NPZ_7B.glob("problem_*.npz"))) >= 100:
        mlp_7, y_7, len_7 = load_mean_logprob(NPZ_7B)
        auroc_mean_7 = oof_1d_oriented(mlp_7, y_7, N_FOLDS, SEED)
        sum_7 = (mlp_7 * len_7).astype(np.float64)
        auroc_sum_7 = oof_1d_oriented(sum_7, y_7, N_FOLDS, SEED)
        n_7b = int(len(y_7))
    else:
        auroc_mean_7 = float("nan"); auroc_sum_7 = float("nan"); n_7b = 0

    F2_AUROC = 0.7731
    out = {
        "n_15b": int(len(y_15)),
        "n_7b": n_7b,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "auroc_mean_logp_15b": float(auroc_mean_15),
        "auroc_sum_logp_15b": float(auroc_sum_15),
        "joint_meanlogp_dom_auroc_15b": float(joint_auroc_15),
        "auroc_mean_logp_7b": float(auroc_mean_7),
        "auroc_sum_logp_7b": float(auroc_sum_7),
        "f2_auroc_anchor": F2_AUROC,
        "tokenprob_beats_dom_15b": bool(auroc_mean_15 > F2_AUROC),
        "scope_note": ("mean+sum aggregations only (cached scalar mean_logprob × seq_len). "
                       "min/product require fresh per-token logprobs, deferred to next H100 session.")
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    for k in ("auroc_mean_logp_15b", "auroc_sum_logp_15b", "joint_meanlogp_dom_auroc_15b",
              "auroc_mean_logp_7b", "auroc_sum_logp_7b"):
        v = out[k]
        if isinstance(v, float):
            print(f"{k}={v:.10f}")
        else:
            print(f"{k}={v}")
    print(f"tokenprob_beats_dom_15b={int(out['tokenprob_beats_dom_15b'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
