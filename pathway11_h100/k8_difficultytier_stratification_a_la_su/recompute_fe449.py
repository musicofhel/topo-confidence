"""P11-FE449 — K=8 difficulty-tier stratification (Su et al.) crossed with B/D-buckets.

Partitions MATH-500 into Easy (8/8 K=8 correct), Medium (mixed), Hard (0/8 correct)
using cached K=8 self-consistency outputs. Per tier, computes mean prefill tokens,
prefill-DoM AUROC against K=1 correctness, and K=1 positive rate. Then cross-tabs the
Hard tier against the ABCD bucket scheme (A: K1✗/K8✗, B: K1✗/K8✓, C: K1✓/K8✓,
D: K1✓/K8✗) to test Su et al.'s underthink-hard frame: if F-7's D-bucket is a
length-failure (underthink) signature rather than a geometric one, D-bucket problems
should pile into the Hard tier and carry shorter token budgets than C-bucket problems,
while remaining geometrically inseparable (low DoM AUROC) from C-bucket.
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
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/k8_difficulty_tiers/results.json"

N = 500


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def load_k8(idx: int) -> np.ndarray | None:
    """Return (8,) bool correctness for problem idx, or None if file missing."""
    for name in (f"problem_{idx:03d}.npz", f"problem_{idx}.npz"):
        path = K8_DIR / name
        if path.exists():
            return np.load(path)["correct"].astype(bool)
    return None


def tier_summary(mask: np.ndarray, dom: np.ndarray, k1: np.ndarray,
                 tokens: np.ndarray) -> dict:
    m = mask.astype(bool)
    n = int(m.sum())
    sub_k1 = k1[m]
    return {
        "n": n,
        "mean_tokens": float(tokens[m].mean()) if n else float("nan"),
        "median_tokens": float(np.median(tokens[m])) if n else float("nan"),
        "k1_positive_rate": float(sub_k1.mean()) if n else float("nan"),
        "prefill_dom_auroc": auroc(dom[m], sub_k1) if n else float("nan"),
    }


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2
    if not K8_DIR.exists():
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    k1 = cache["correct"].astype(bool)             # (500,) K=1 correctness
    tokens = cache["seq_len"].astype(np.float64)   # (500,) prefill seq lengths
    dom = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert k1.shape == (N,) and tokens.shape == (N,) and dom.shape == (N,)

    # K=8 per-problem correct counts (0..8); missing files flagged with -1.
    k8_count = np.full(N, -1, dtype=np.int64)
    missing = []
    for i in range(N):
        c = load_k8(i)
        if c is None:
            missing.append(i)
            continue
        k8_count[i] = int(c.sum())
    if missing:
        print(f"MISSING_REGEN_INPUT {len(missing)} k8 files (e.g. idx {missing[:5]})",
              file=sys.stderr)
        return 2

    valid = k8_count >= 0
    # Difficulty tiers (Su et al.) over the 8-sample correctness count.
    easy = valid & (k8_count == 8)
    hard = valid & (k8_count == 0)
    medium = valid & (k8_count > 0) & (k8_count < 8)

    # K=8 majority-vote correctness for ABCD bucketing (tie at 4/8 -> not correct).
    k8_majority = k8_count >= 5
    A = valid & (~k1) & (~k8_majority)   # K1 wrong, K8 wrong
    B = valid & (~k1) & (k8_majority)    # K1 wrong, K8 right
    C = valid & (k1) & (k8_majority)     # K1 right, K8 right
    D = valid & (k1) & (~k8_majority)    # K1 right, K8 wrong  (F-7 D-bucket / underthink)

    tiers = {
        "easy_8of8": tier_summary(easy, dom, k1, tokens),
        "medium_mixed": tier_summary(medium, dom, k1, tokens),
        "hard_0of8": tier_summary(hard, dom, k1, tokens),
    }

    buckets = {
        "A_k1wrong_k8wrong": int(A.sum()),
        "B_k1wrong_k8right": int(B.sum()),
        "C_k1right_k8right": int(C.sum()),
        "D_k1right_k8wrong": int(D.sum()),
    }

    # Cross-tab: where does the D-bucket live across difficulty tiers?
    d_in_hard = int((D & hard).sum())
    d_in_medium = int((D & medium).sum())
    d_in_easy = int((D & easy).sum())
    d_total = int(D.sum())

    # Underthink-vs-geometry test on the K=1-right subpopulation (C vs D):
    #   length signature  -> C and D differ in tokens
    #   geometric signature -> DoM separates C from D (high AUROC)
    cd_mask = C | D
    c_or_d_tokens_C = tokens[C]
    c_or_d_tokens_D = tokens[D]
    if cd_mask.sum() and C.sum() and D.sum():
        # AUROC of DoM distinguishing C (label 1) from D (label 0).
        cvd_auroc = auroc(dom[cd_mask], C[cd_mask].astype(bool))
        # AUROC of token length distinguishing C from D (the length-failure baseline).
        cvd_tokens_auroc = auroc(tokens[cd_mask], C[cd_mask].astype(bool))
    else:
        cvd_auroc = float("nan")
        cvd_tokens_auroc = float("nan")

    cross_tab = {
        "d_bucket_total": d_total,
        "d_in_hard_tier": d_in_hard,
        "d_in_medium_tier": d_in_medium,
        "d_in_easy_tier": d_in_easy,
        "frac_d_in_hard": (d_in_hard / d_total) if d_total else float("nan"),
        "hard_tier_d_fraction": (d_in_hard / int(hard.sum())) if hard.sum() else float("nan"),
        "c_vs_d_dom_auroc": cvd_auroc,
        "c_vs_d_tokens_auroc": cvd_tokens_auroc,
        "c_bucket_mean_tokens": float(c_or_d_tokens_C.mean()) if C.sum() else float("nan"),
        "d_bucket_mean_tokens": float(c_or_d_tokens_D.mean()) if D.sum() else float("nan"),
    }

    out = {
        "experiment": "P11-FE449",
        "n_problems": N,
        "n_valid_k8": int(valid.sum()),
        "tiers": tiers,
        "abcd_buckets": buckets,
        "hard_d_cross_tab": cross_tab,
        "interpretation_note": (
            "If frac_d_in_hard is high AND c_vs_d_tokens_auroc >> c_vs_d_dom_auroc, "
            "F-7's D-bucket reads as an underthink/length-failure signature (short "
            "reasoning on hard problems) rather than a geometric one."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())