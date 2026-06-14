"""P11-FE157 — Null distribution for pairwise cos(probe_a, probe_b) of L19 DoM directions.

F-3 reads cos(prefill_DoM, final_DoM)=0.046 as a low, *meaningful* number that
licenses an "asymmetric circuit" interpretation. Paper 2401.13558 predicts that
two arbitrary linear probes in a ReLU-family network are generically near-orthogonal
in a high-dimensional residual stream. This script builds the empirical null for
cos between L19 difference-of-means (DoM) directions probed on a battery of mutually
unrelated labels, and asks whether the observed real-label pairwise cosines fall
inside that null band.

Construction:
  * A DoM direction for a binary label L is unit-normalize(mean(X[L]) - mean(X[~L])).
  * Real labels are built from the only signals cached locally at L19:
      - correctness            (cache["correct"])
      - length median split    (seq_len > median)
      - length quartile one-hots (4 directions, each quartile vs rest)
      - dom-high split          (dom_score > median; an arbitrary geometric label)
      - k8 majority-correct     (>=4/8 generations correct, if the K=8 cache exists)
    The text-derived labels named in the FE plan (topic-one-hot, has-equation,
    has-fraction, CoT-pivot-token) require raw problem text that is NOT in the
    cached NPZs, so they are represented in the null by random balanced labels
    rather than fabricated. This is the conservative substitution: the null is
    built from *arbitrary* label assignments, which is exactly the claim under test.
  * The null is N_NULL random balanced binary labels; all pairwise cosines among
    their DoM directions form the null distribution. Real-label pairwise cosines
    are then z-scored against it.
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
OUT_JSON = ROOT / "pathway11_h100/probe_orthogonality_null/results.json"

SEED = 9999
N_NULL = 300


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def dom_direction(X: np.ndarray, label: np.ndarray) -> np.ndarray:
    """Unit-normalized difference-of-means direction for a binary label."""
    label = label.astype(bool)
    if label.sum() == 0 or (~label).sum() == 0:
        return np.zeros(X.shape[1], dtype=np.float64)
    d = X[label].mean(axis=0) - X[~label].mean(axis=0)
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.zeros_like(d)
    return d / n


def load_k8_majority(n: int) -> np.ndarray | None:
    """Majority-correct label from the K=8 self-consistency cache, or None if absent."""
    if not K8_DIR.exists():
        return None
    maj = np.zeros(n, dtype=bool)
    found = 0
    for i in range(n):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            return None
        c = np.load(f)["correct"].astype(bool)
        maj[i] = c.sum() >= (len(c) / 2.0)
        found += 1
    return maj if found == n else None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    seq_len = blob["seq_len"].astype(np.float64)
    assert X.shape == (500, 1536) and correct.shape == (500,)
    n, d = X.shape

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    # ---- Real, locally-derivable orthogonal labels --------------------------
    real_labels: dict[str, np.ndarray] = {}
    real_labels["correctness"] = correct
    real_labels["length_median"] = seq_len > np.median(seq_len)

    q = np.quantile(seq_len, [0.25, 0.5, 0.75])
    bins = np.digitize(seq_len, q)  # 0..3
    for qi in range(4):
        real_labels[f"length_q{qi+1}"] = bins == qi

    real_labels["dom_high"] = dom_score > np.median(dom_score)

    k8_maj = load_k8_majority(n)
    if k8_maj is not None:
        real_labels["k8_majority_correct"] = k8_maj

    real_names = list(real_labels.keys())
    real_dirs = np.stack([dom_direction(X, real_labels[k]) for k in real_names])

    # ---- Null distribution: pairwise cos among arbitrary balanced labels ----
    rng = np.random.default_rng(SEED)
    null_dirs = np.zeros((N_NULL, d), dtype=np.float64)
    for i in range(N_NULL):
        while True:
            lbl = rng.random(n) < 0.5
            if 0 < lbl.sum() < n:
                break
        null_dirs[i] = dom_direction(X, lbl)

    null_gram = null_dirs @ null_dirs.T
    iu = np.triu_indices(N_NULL, k=1)
    null_cos = null_gram[iu]
    null_abs = np.abs(null_cos)
    null_mean = float(null_cos.mean())
    null_std = float(null_cos.std(ddof=1))
    null_abs_mean = float(null_abs.mean())
    null_abs_p50 = float(np.percentile(null_abs, 50))
    null_abs_p95 = float(np.percentile(null_abs, 95))
    null_abs_p99 = float(np.percentile(null_abs, 99))

    # ---- Real-label pairwise cosines, z-scored against the null -------------
    real_gram = real_dirs @ real_dirs.T
    real_pairs = {}
    abs_real = []
    for i in range(len(real_names)):
        for j in range(i + 1, len(real_names)):
            c = float(real_gram[i, j])
            z = float((c - null_mean) / null_std) if null_std > 0 else float("nan")
            real_pairs[f"{real_names[i]}__{real_names[j]}"] = {
                "cos": c,
                "abs_cos": abs(c),
                "z_vs_null": z,
                "exceeds_null_p95": bool(abs(c) > null_abs_p95),
            }
            abs_real.append(abs(c))

    # F-3 anchor: cos(prefill_DoM, final_DoM) = 0.046 is the published number.
    # We cannot recompute it here (no final-token cache), but we can locate it
    # in the null: is 0.046 inside the typical band of arbitrary-label cosines?
    f3_value = 0.046
    f3_pct_in_null = float((null_abs <= f3_value).mean())

    out = {
        "experiment": "P11-FE157",
        "n": n,
        "dim": d,
        "n_null": N_NULL,
        "n_real_labels": len(real_names),
        "real_labels": real_names,
        "auroc_correctness_dom": float(auroc(dom_score, correct)),
        "null_cos_mean": null_mean,
        "null_cos_std": null_std,
        "null_abs_cos_mean": null_abs_mean,
        "null_abs_cos_p50": null_abs_p50,
        "null_abs_cos_p95": null_abs_p95,
        "null_abs_cos_p99": null_abs_p99,
        "real_pairs": real_pairs,
        "real_abs_cos_mean": float(np.mean(abs_real)) if abs_real else float("nan"),
        "real_abs_cos_max": float(np.max(abs_real)) if abs_real else float("nan"),
        "n_real_pairs_exceeding_null_p95": int(sum(p["exceeds_null_p95"] for p in real_pairs.values())),
        "f3_published_cos": f3_value,
        "f3_fraction_of_null_below": f3_pct_in_null,
        "f3_within_null_band": bool(f3_value <= null_abs_p95),
        "interpretation": (
            "If real-label pairwise |cos| cluster within the null band (<=p95) and "
            "0.046 sits inside it, F-3's near-orthogonality is generic and the "
            "'asymmetric circuit' reading is null-distribution; reframe."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())