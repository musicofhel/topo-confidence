#!/usr/bin/env python3
"""Phase 1B.1 — probe-reranking on the K=8 cache (executes H-40 / P11-FE41).

Fits a per-sample correctness probe on the K=8 final-token L19 states
(L19_samples is verified to be the final generated token), OOF with
PROBLEM-GROUPED folds (a problem's 8 samples never split across train/test).
Then three selection arms at IDENTICAL compute (K=8):

  (a) probe-argmax            answer of the highest-probe sample
  (b) probe-weighted majority votes weighted by probe score
  (c) confidence-fallback     keep the K=1 greedy answer if its probe score
                              clears a train-chosen threshold, else K=8 majority
                              (D-bucket protection — F-7)

Comparators at identical compute: K=8 plain majority (0.550), K=1 greedy
(0.486), oracle best-of-8 ceiling (0.704). Adjudication: paired McNemar vs
plain majority, stratified by the F-7 bucket (K1 x K8maj). WIN = an arm beats
plain majority significantly WITHOUT a D-bucket (K1-right -> wrong) regression.

This is SELECTION at fixed compute, not routing — the FE19 hull critique cannot
apply.

Output: results/1b1_reranking.json
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)

import activation_loader as AL
import k8_lib as K8
from metrics import DomProbe, mcnemar_test, SEED


def problem_grouped_oof(L19, correct, seed=SEED, n_folds=5):
    """OOF per-sample probe scores (n,8), folds over PROBLEMS. Probe = DoM on the
    pooled samples of the training problems predicting per-sample correct."""
    n = L19.shape[0]
    scores = np.zeros((n, 8), dtype=np.float64)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_p, te_p in kf.split(np.arange(n)):
        Xtr = L19[tr_p].reshape(-1, L19.shape[2])
        ytr = correct[tr_p].reshape(-1)
        probe = DomProbe().fit(Xtr, ytr)
        for i in te_p:
            scores[i] = probe.score(L19[i])
    return scores


def greedy_probe_scores(L19, correct, m15, seed=SEED, n_folds=5):
    """Score the K=1 GREEDY final-token state with the same OOF probe (for the
    confidence-fallback arm). Greedy states come from the main cache X_last[L19]."""
    n = L19.shape[0]
    Xg = m15.X("last")              # (500, 1536) greedy final-token L19
    sg = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr_p, te_p in kf.split(np.arange(n)):
        Xtr = L19[tr_p].reshape(-1, L19.shape[2])
        ytr = correct[tr_p].reshape(-1)
        probe = DomProbe().fit(Xtr, ytr)
        sg[te_p] = probe.score(Xg[te_p])
    return sg


def fallback_policy(greedy_scores, greedy_correct, maj_correct, y_train_mask,
                    tau_grid=None):
    """Confidence-fallback: keep greedy answer if greedy_score>=tau else majority.
    tau chosen on the train mask to maximise accuracy; applied to all. Returns
    per-problem correctness + chosen tau."""
    if tau_grid is None:
        tau_grid = np.quantile(greedy_scores, np.linspace(0.0, 0.95, 40))
    best_tau, best_acc = tau_grid[0], -1
    for tau in tau_grid:
        use_greedy = greedy_scores >= tau
        outcome = np.where(use_greedy, greedy_correct, maj_correct)
        acc = outcome[y_train_mask].mean()
        if acc > best_acc:
            best_acc, best_tau = acc, tau
    use_greedy = greedy_scores >= best_tau
    return np.where(use_greedy, greedy_correct, maj_correct), float(best_tau)


def buckets(k1, k8maj):
    """F-7 buckets by (K=1 x K=8 majority)."""
    return {
        "A_both_right": k1 & k8maj,
        "B_recovered_k1wrong_k8right": ~k1 & k8maj,
        "D_regressed_k1right_k8wrong": k1 & ~k8maj,
        "E_both_wrong": ~k1 & ~k8maj,
    }


def main():
    k8 = K8.load_all_k8()
    m15 = AL.load_cell("qwen1.5b", "math")
    n = k8["n"]
    L19, correct, answers = k8["L19"], k8["correct"], k8["answers"]

    # per-sample probe quality
    sc = problem_grouped_oof(L19, correct)
    auroc_persample = roc_auc_score(correct.reshape(-1), sc.reshape(-1))

    # comparators (per-problem correctness vectors)
    k1 = m15.y                                  # K=1 greedy
    maj = np.array([K8.plain_majority_correct(answers[i], correct[i])
                    for i in range(n)])
    oracle = np.array([correct[i].any() for i in range(n)])

    # arms
    arm_argmax = np.array([K8.probe_argmax_correct(sc[i], correct[i])
                           for i in range(n)])
    arm_wmaj = np.array([K8.probe_weighted_majority_correct(answers[i], correct[i],
                                                            sc[i]) for i in range(n)])
    # weighted-majority with tie-break by probe also reduces to mode when probe flat
    sg = greedy_probe_scores(L19, correct, m15)
    # fallback tau chosen OOF: choose on each train fold, apply to test fold
    arm_fb = np.zeros(n, dtype=bool)
    kf = KFold(5, shuffle=True, random_state=SEED)
    for tr_p, te_p in kf.split(np.arange(n)):
        mask_tr = np.zeros(n, dtype=bool); mask_tr[tr_p] = True
        out, _ = fallback_policy(sg, k1, maj, mask_tr)
        arm_fb[te_p] = out[te_p]

    bk = buckets(k1, maj)

    def summarize(name, outcome):
        mc = mcnemar_test(outcome, maj)         # vs plain majority
        # D-bucket regression: among A_both_right? no — D-bucket = where majority
        # already regressed K1. The arm's job: not introduce NEW regressions vs K1.
        new_reg = int((k1 & ~outcome).sum())    # K1-right that this arm gets wrong
        return {
            "accuracy": float(outcome.mean()),
            "vs_majority": mc,
            "delta_vs_majority_pp": float(100 * (outcome.mean() - maj.mean())),
            "k1_right_now_wrong": new_reg,
            "k1_right_now_wrong_majority": int((k1 & ~maj).sum()),
            "by_bucket_acc": {b: float(outcome[m].mean()) if m.any() else None
                              for b, m in bk.items()},
        }

    out = {
        "probe_persample_auroc": float(auroc_persample),
        "comparators": {
            "k1_greedy": float(k1.mean()),
            "k8_plain_majority": float(maj.mean()),
            "oracle_best_of_8": float(oracle.mean()),
        },
        "bucket_sizes": {b: int(m.sum()) for b, m in bk.items()},
        "arms": {
            "probe_argmax": summarize("probe_argmax", arm_argmax),
            "probe_weighted_majority": summarize("probe_weighted_majority", arm_wmaj),
            "confidence_fallback": summarize("confidence_fallback", arm_fb),
        },
    }
    (RESULTS / "1b1_reranking.json").write_text(json.dumps(out, indent=2))

    print(f"per-sample probe AUROC (final-token L19, problem-grouped OOF) = "
          f"{auroc_persample:.4f}")
    c = out["comparators"]
    print(f"\nComparators: K1={c['k1_greedy']:.4f}  K8maj={c['k8_plain_majority']:.4f}"
          f"  oracle={c['oracle_best_of_8']:.4f}")
    print(f"Buckets: {out['bucket_sizes']}")
    print("\nArms (vs plain majority 0.554):")
    for name, a in out["arms"].items():
        mc = a["vs_majority"]
        sig = "SIG" if mc["p"] < 0.05 else "n.s."
        print(f"  {name:24s} acc={a['accuracy']:.4f} "
              f"({a['delta_vs_majority_pp']:+.2f}pp, McNemar p={mc['p']:.3f} {sig}; "
              f"discord {mc['b_a_right_b_wrong']}/{mc['b_a_wrong_b_right']}) "
              f"new-regress K1right->wrong={a['k1_right_now_wrong']} "
              f"(maj={a['k1_right_now_wrong_majority']})")
    print(f"\n→ {RESULTS / '1b1_reranking.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
