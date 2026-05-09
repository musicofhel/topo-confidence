"""FE168 — PDS proxy (pairwise correctness agreement) vs prefill L19 DoM.

PDS (2402.10528) requires SummaC-NLI inference (torch/transformers), violating
recompute-script constraints. This computes two PDS proxies from the K=8
self-consistency cache (subsampled to K=5):

  1. Pairwise agreement rate: fraction of (i,j) pairs with same correctness.
     Symmetric (m=1 and m=4 yield identical scores), so a weak discriminator.
  2. Self-consistency fraction: fraction of K=5 rollouts that are correct.
     Directional and strictly more informative for correctness prediction.

Compares both proxies and DoM on AUROC, AUC-PR, best F1, and selective
prediction at coverage=0.5 against F-8 headline (71.6% acc, 50% cov, K=2.5).
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
OUT_JSON = ROOT / "pathway11_h100/pds_proxy/results.json"

N_PROBLEMS = 500
K_TOTAL = 8
K_USE = 5
SEED = 9999
COVERAGE = 0.5


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def auc_pr(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    y = labels[order].astype(np.float64)
    tp = np.cumsum(y)
    n_pred = np.arange(1.0, len(y) + 1.0)
    precision = tp / n_pred
    total_pos = max(y.sum(), 1.0)
    recall = tp / total_pos
    dr = np.diff(recall, prepend=0.0)
    return float(np.sum(precision * dr))


def best_f1(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(scores)
    best, best_t = 0.0, float(thresholds[0])
    for t in thresholds:
        pred = scores >= t
        tp = (pred & labels).sum()
        fp = (pred & ~labels).sum()
        fn = (~pred & labels).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f = 2 * prec * rec / max(prec + rec, 1e-12)
        if f > best:
            best, best_t = float(f), float(t)
    return best, best_t


def selective_accuracy(
    scores: np.ndarray, correct: np.ndarray, coverage: float,
) -> dict:
    n = len(scores)
    n_answer = max(1, int(round(n * coverage)))
    order = np.argsort(-scores)
    answered = order[:n_answer]
    acc = float(correct[answered].mean())
    return {
        "coverage": round(n_answer / n, 4),
        "n_answered": int(n_answer),
        "accuracy_on_answered": round(acc, 4),
    }


def risk_coverage_curve(
    scores: np.ndarray, correct: np.ndarray, steps: int = 10,
) -> list[dict]:
    points = []
    for i in range(1, steps + 1):
        cov = i / steps
        pt = selective_accuracy(scores, correct, cov)
        points.append(pt)
    return points


def pairwise_agreement(c: np.ndarray) -> float:
    k = len(c)
    m = int(c.sum())
    agree = m * (m - 1) // 2 + (k - m) * (k - m - 1) // 2
    total = k * (k - 1) // 2
    return float(agree / total) if total > 0 else 0.0


def main() -> int:
    for path, name in [(CACHE, "prefill cache"), (DOM_NPZ, "DoM NPZ")]:
        if not path.exists():
            print(f"MISSING_REGEN_INPUT {path}", file=sys.stderr)
            return 2
    if not K8_DIR.exists():
        print(f"MISSING_REGEN_INPUT {K8_DIR}", file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    correct_k1 = blob["correct"].astype(bool)
    assert correct_k1.shape == (N_PROBLEMS,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert dom_score.shape == (N_PROBLEMS,)

    rng = np.random.default_rng(SEED)
    pds_agree = np.zeros(N_PROBLEMS, dtype=np.float64)
    sc_frac = np.zeros(N_PROBLEMS, dtype=np.float64)
    majvote = np.zeros(N_PROBLEMS, dtype=bool)
    missing = []

    for i in range(N_PROBLEMS):
        p = K8_DIR / f"problem_{i:03d}.npz"
        if not p.exists():
            missing.append(i)
            continue
        c8 = np.load(p)["correct"].astype(bool)
        assert c8.shape == (K_TOTAL,), f"problem_{i:03d}: shape {c8.shape}"
        idx = rng.choice(K_TOTAL, size=K_USE, replace=False)
        c5 = c8[idx]
        pds_agree[i] = pairwise_agreement(c5)
        sc_frac[i] = c5.mean()
        majvote[i] = c5.sum() >= (K_USE + 1) // 2

    if missing:
        print(
            f"MISSING_REGEN_INPUT {len(missing)} K=8 files missing "
            f"(first: problem_{missing[0]:03d}.npz)",
            file=sys.stderr,
        )
        return 2

    k1_acc = round(float(correct_k1.mean()), 4)
    mv_acc = round(float(majvote.mean()), 4)

    def metrics_block(scores, label, correct, name):
        a = auroc(scores, correct)
        ap = auc_pr(scores, correct)
        f, ft = best_f1(scores, correct)
        sel = selective_accuracy(scores, correct, COVERAGE)
        rc = risk_coverage_curve(scores, correct)
        return {
            "auroc": round(a, 4),
            "auc_pr": round(ap, 4),
            "best_f1": round(f, 4),
            "best_f1_threshold": round(ft, 4),
            "selective_at_0.5": sel,
            "risk_coverage_curve": rc,
        }

    pds_metrics = metrics_block(pds_agree, "pds_agree", correct_k1, "PDS-agree")
    sc_metrics = metrics_block(sc_frac, "sc_frac", correct_k1, "SC-frac")
    dom_metrics = metrics_block(dom_score, "dom", correct_k1, "DoM")

    sel_mv = selective_accuracy(pds_agree, majvote, COVERAGE)
    sc_sel_mv = selective_accuracy(sc_frac, majvote, COVERAGE)

    pds_agree_vals, pds_agree_counts = np.unique(pds_agree, return_counts=True)

    out = {
        "experiment": "FE168",
        "description": (
            "PDS proxy (pairwise correctness agreement K=5) vs prefill L19 DoM. "
            "True PDS needs SummaC-NLI (torch); proxy uses K=8 cache subsampled to K=5."
        ),
        "seed": SEED,
        "K_used": K_USE,
        "K_total": K_TOTAL,
        "n_problems": N_PROBLEMS,
        "k1_accuracy": k1_acc,
        "majority_vote_k5_accuracy": mv_acc,
        "pds_proxy_agree_distribution": {
            str(round(v, 2)): int(c)
            for v, c in zip(pds_agree_vals, pds_agree_counts)
        },
        "pds_proxy_pairwise_agreement": pds_metrics,
        "self_consistency_k5_fraction": sc_metrics,
        "dom_prefill_l19": dom_metrics,
        "pds_proxy_selective_majvote_at_0.5": sel_mv,
        "sc_selective_majvote_at_0.5": sc_sel_mv,
        "f8_comparison": {
            "f8_selective_acc_at_0.5_coverage": 0.716,
            "f8_avg_K": 2.5,
            "pds_agree_sel_k1": pds_metrics["selective_at_0.5"]["accuracy_on_answered"],
            "sc_frac_sel_k1": sc_metrics["selective_at_0.5"]["accuracy_on_answered"],
            "sc_frac_sel_majvote": sc_sel_mv["accuracy_on_answered"],
            "dom_sel_k1": dom_metrics["selective_at_0.5"]["accuracy_on_answered"],
            "verdict": (
                "DoM_wins"
                if dom_metrics["auroc"] > max(pds_metrics["auroc"], sc_metrics["auroc"])
                else "PDS_proxy_wins"
            ),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Written: {OUT_JSON}")
    print(f"PDS-agree AUROC: {pds_metrics['auroc']:.4f}  (only {len(pds_agree_vals)} distinct values)")
    print(f"SC-frac   AUROC: {sc_metrics['auroc']:.4f}")
    print(f"DoM       AUROC: {dom_metrics['auroc']:.4f}")
    print(f"Sel@0.5 — PDS-agree(K=1): {pds_metrics['selective_at_0.5']['accuracy_on_answered']:.4f}"
          f"  SC-frac(K=1): {sc_metrics['selective_at_0.5']['accuracy_on_answered']:.4f}"
          f"  DoM: {dom_metrics['selective_at_0.5']['accuracy_on_answered']:.4f}")
    print(f"Sel@0.5 — SC-frac(majvote): {sc_sel_mv['accuracy_on_answered']:.4f}  (F-8 headline: 0.716)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())