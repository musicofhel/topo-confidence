"""P11-FE335 — Re-attribute F-8 selective prediction to label-free CoE-C gating.

F-8's headline (71.6% answered accuracy at coverage 0.5, K=2.5 avg) currently
rests on the *supervised* L19 prefill DoM direction. This script swaps the
gating signal for a closed-form, label-free "CoE-C" trajectory-geometry score
and asks whether the selective-prediction protocol still hits 71.6%.

Caveat on CoE-C: the canonical Chain-of-Embedding score needs the full
across-layer hidden trajectory, but only the single L19 prefill state is cached
(prefill_inversion/cache/m15b_prefill.npz). We therefore reconstruct CoE-C as
the label-free top covariance eigenvector (PC1) projection — the documented
label-free analog of DoM (cos(DoM, PC1) = 0.9216, FE291) — computed OOF per
fold, with only the eigenvector *sign* (one bit) oriented against train labels.
Everything else is unsupervised geometry on the same hidden states.

Selective-prediction protocol (identical for CoE-C and the DoM control):
  1. Rank the 500 problems by confidence (descending).
  2. Answer the top 50% (coverage 0.5 -> 250 problems).
  3. Allocate an integer per-problem self-consistency budget K_i in [1, 8] that
     averages K=2.5 over the answered set, giving the least-confident answered
     problems more samples (gentle monotone schedule).
  4. Answered accuracy = expected majority-vote correctness under the K8
     self-consistency cache (hypergeometric over the 8 cached samples).

If CoE-C-gated acc@coverage-0.5 >= 0.716, F-8 is re-attributable to label-free
trajectory geometry rather than to supervised DoM.
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
from scipy.stats import hypergeom

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
F8_RESULTS = ROOT / "pathway11_h100/prefill_gated_compute/results.json"
K8_DIR = ROOT / "pathway11_h100/data/k8_selfconsistency"
OUT_JSON = ROOT / "pathway11_h100/coe_c_selective/results.json"

N_FOLDS = 5
SEED = 9999
COVERAGE = 0.5
K_TARGET = 2.5
K8_N = 8
BASELINE_F8 = 0.716  # default; overridden from F8_RESULTS if available


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


def coe_c_oof(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Label-free OOF PC1 projection (CoE-C surrogate). Only sign uses labels."""
    n = len(y)
    coe = np.zeros(n, dtype=np.float64)
    folds = stratified_kfold(y, k, seed)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr = X[train_mask]
        ytr = y[train_mask]
        mu = Xtr.mean(axis=0)
        Xc = Xtr - mu
        cov = Xc.T @ Xc / max(len(Xtr) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, -1]  # top eigenvector (largest eigenvalue)
        # orient sign so higher projection => more likely correct (one bit of supervision)
        proj_tr = Xc @ v
        if np.corrcoef(proj_tr, ytr.astype(np.float64))[0, 1] < 0:
            v = -v
        coe[test_idx] = (X[test_idx] - mu) @ v
    return coe


def maj_vote_prob(c: int, k_alloc: int, n_total: int = K8_N) -> float:
    """Expected majority-vote correctness drawing k_alloc of n_total cached
    samples without replacement, c of them correct. Even-K ties split 0.5."""
    if k_alloc <= 0:
        return 0.0
    js = np.arange(k_alloc + 1)
    pmf = hypergeom.pmf(js, n_total, c, k_alloc)
    p = 0.0
    for j, ph in zip(js, pmf):
        if 2 * j > k_alloc:
            p += ph
        elif 2 * j == k_alloc:
            p += 0.5 * ph
    return float(p)


def allocate_k(conf_answered: np.ndarray, k_target: float) -> np.ndarray:
    """Integer budget in [1, 8] averaging k_target, least-confident get more."""
    n = len(conf_answered)
    total = int(round(k_target * n))
    K = np.ones(n, dtype=int)
    budget = total - n
    order = np.argsort(conf_answered)  # ascending: least confident first
    i = 0
    guard = 0
    while budget > 0 and guard < n * K8_N:
        idx = order[i % n]
        if K[idx] < K8_N:
            K[idx] += 1
            budget -= 1
        i += 1
        guard += 1
    return K


def selective_eval(score: np.ndarray, c_counts: np.ndarray, correct: np.ndarray,
                   coverage: float, k_target: float) -> dict:
    n = len(score)
    n_answer = int(round(coverage * n))
    order = np.argsort(score)[::-1]  # most confident first
    answered = order[:n_answer]
    conf_answered = score[answered]
    K = allocate_k(conf_answered, k_target)
    maj_acc = float(np.mean([maj_vote_prob(int(c_counts[a]), int(k))
                             for a, k in zip(answered, K)]))
    return {
        "n_answered": int(n_answer),
        "coverage": float(coverage),
        "mean_k": float(K.mean()),
        "acc_at_coverage_maj_vote": maj_acc,
        "acc_at_coverage_k1_greedy": float(correct[answered].mean()),
    }


def main() -> int:
    for path in (CACHE, DOM_NPZ):
        if not path.exists():
            print("MISSING_REGEN_INPUT", path, file=sys.stderr); return 2
    if not K8_DIR.is_dir():
        print("MISSING_REGEN_INPUT", K8_DIR, file=sys.stderr); return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    correct = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and correct.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (500,)
    if auroc(dom_score, correct) < 0.5:  # ensure higher => more correct
        dom_score = -dom_score

    # Per-problem correct-count from the K=8 self-consistency cache.
    c_counts = np.full(500, -1, dtype=np.int64)
    for i in range(500):
        f = K8_DIR / f"problem_{i:03d}.npz"
        if not f.exists():
            print("MISSING_REGEN_INPUT", f, file=sys.stderr); return 2
        c_counts[i] = int(np.load(f)["correct"].astype(bool).sum())

    coe_score = coe_c_oof(X, correct, N_FOLDS, SEED)

    baseline = BASELINE_F8
    if F8_RESULTS.exists():
        try:
            j = json.loads(F8_RESULTS.read_text())
            for key in ("selective_acc_at_cov50", "acc_at_coverage_0.5",
                        "answered_acc_cov50", "selective_prediction_acc"):
                if key in j:
                    baseline = float(j[key]); break
        except Exception:
            pass

    coe_eval = selective_eval(coe_score, c_counts, correct, COVERAGE, K_TARGET)
    dom_eval = selective_eval(dom_score, c_counts, correct, COVERAGE, K_TARGET)

    coe_acc = coe_eval["acc_at_coverage_maj_vote"]
    out = {
        "experiment": "P11-FE335",
        "description": "CoE-C (label-free PC1, OOF) vs DoM gating in F-8 selective prediction",
        "coverage": COVERAGE,
        "k_target": K_TARGET,
        "f8_baseline_acc": float(baseline),
        "coe_c": {
            "auroc": auroc(coe_score, correct),
            **coe_eval,
        },
        "dom_control": {
            "auroc": auroc(dom_score, correct),
            **dom_eval,
        },
        "cos_coe_dom": float(np.corrcoef(coe_score, dom_score)[0, 1]),
        "coe_acc_at_coverage": coe_acc,
        "matches_or_beats_f8": bool(coe_acc >= baseline),
        "verdict": ("REATTRIBUTABLE_LABELFREE" if coe_acc >= baseline
                    else "DOM_STILL_REQUIRED"),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())