#!/usr/bin/env python3
"""Experiment 3 / Phase 2: cross-scale analysis.

(a) Leave-one-out PR contribution on 7B: for each problem in the correct group,
    remove it and recompute PR. Delta = PR_correct_without_i - PR_correct_with_all.
    Identify the 10 most impactful problems driving the inversion.

(b) Three-way split on MATH-500:
      bothA   = 7B ✓ AND 1.5B ✓
      only7B  = 7B ✓ AND 1.5B ✗   (capability breadth candidates)
      bothC   = 7B ✗ AND 1.5B ✗
      only15B = 7B ✗ AND 1.5B ✓   (7B pathology candidates)
    Compute prefill PR + final-token PR per group, per model.
    If only7B has the highest 7B prefill PR, the inversion is tied to
    "7B-solvable problems that 1.5B can't do" — capability breadth story.

(c) BBH check on each subset:
      tracking_shuffled_objects_seven_objects  (acc 12.4%, imbalanced)
      logical_deduction_seven_objects           (acc 6.0%, imbalanced)
      web_of_lies                               (acc 54.0%, BALANCED)
    Compute prefill PR(correct) vs PR(incorrect) on each. Report ratios.
    web_of_lies is the key test — does 1.5B show a prefill inversion on
    a balanced non-math task?

(d) D-bucket analysis (added per user request):
    Load Exp 2's bucket assignments:
      A = greedy K=1 ✓ AND K=8 majority ✓  (207)
      B = greedy K=1 ✗ AND K=8 majority ✓  (68) — recoverable
      C = greedy K=1 ✗ AND K=8 majority ✗  (189) — hopeless
      D = greedy K=1 ✓ AND K=8 majority ✗  (36) — pathological
    For 1.5B (since buckets are defined on 1.5B):
      * prefill PR per bucket
      * final-token PR per bucket
      * prefill DoM score distribution per bucket (from Exp 2 phase2 OOF scores)
      * seq_len distribution per bucket
    Answer: do D problems have a distinctive signature? short-and-correct-at-K=1?
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BBH_SUBSETS, OUT_DIR, load_bbh_subset, participation_ratio)

CACHE_DIR = OUT_DIR / "cache"


def load_cached(name: str):
    fp = CACHE_DIR / f"{name}.npz"
    d = np.load(fp)
    return d["prefill"].astype(np.float32), d["final_tok"].astype(np.float32), \
           d["correct"].astype(bool), d["seq_len"]


def loo_contribution(H: np.ndarray, y: np.ndarray, sort_top: int = 10) -> dict:
    """Leave-one-out PR contribution for CORRECT group.

    Returns an array (n_problems,) of delta-PR values (NaN for indices not in correct
    group), the top-k most impactful problems, and similar for incorrect group.
    """
    n = len(y)
    correct_idxs = np.where(y)[0]
    incorrect_idxs = np.where(~y)[0]

    base_c = participation_ratio(H[y])
    base_i = participation_ratio(H[~y])

    delta_c = np.full(n, np.nan, dtype=np.float64)
    for idx in correct_idxs:
        mask = y.copy(); mask[idx] = False
        pr_loo = participation_ratio(H[mask])
        delta_c[idx] = pr_loo - base_c  # negative = this problem was pushing PR up

    delta_i = np.full(n, np.nan, dtype=np.float64)
    for idx in incorrect_idxs:
        mask = (~y).copy(); mask[idx] = False
        pr_loo = participation_ratio(H[mask])
        delta_i[idx] = pr_loo - base_i

    # Sort: most negative delta = biggest contributor to the PR (removing it drops PR most)
    top_correct_pushers = np.argsort(delta_c)[:sort_top].tolist()   # most negative
    top_incorrect_pushers = np.argsort(delta_i)[:sort_top].tolist()

    return dict(
        base_pr_correct=float(base_c), base_pr_incorrect=float(base_i),
        delta_correct=delta_c, delta_incorrect=delta_i,
        top_correct_pushers=[int(i) for i in top_correct_pushers],
        top_incorrect_pushers=[int(i) for i in top_incorrect_pushers],
    )


def three_way_groups(y7: np.ndarray, y15: np.ndarray) -> dict:
    both_right = y7 & y15
    only_7b = y7 & ~y15
    only_15b = ~y7 & y15
    both_wrong = ~y7 & ~y15
    return dict(both_right=both_right, only_7b=only_7b,
                only_15b=only_15b, both_wrong=both_wrong)


def group_pr(H: np.ndarray, mask: np.ndarray) -> float | None:
    n = int(mask.sum())
    if n < 5:
        return None
    return float(participation_ratio(H[mask]))


def analyze_three_way(pf7, ft7, pf15, ft15, y7, y15):
    groups = three_way_groups(y7, y15)
    results = {}
    for name, mask in groups.items():
        results[name] = dict(
            n=int(mask.sum()),
            pr_7b_prefill=group_pr(pf7, mask),
            pr_7b_finaltok=group_pr(ft7, mask),
            pr_15b_prefill=group_pr(pf15, mask),
            pr_15b_finaltok=group_pr(ft15, mask),
        )
    return results


def bbh_analysis():
    results = {}
    for subset in BBH_SUBSETS:
        pf, ft, c, sl, miss = load_bbh_subset(subset)
        n_c = int(c.sum()); n_i = int((~c).sum())
        results[subset] = dict(n=int(len(c)), n_correct=n_c, n_incorrect=n_i, acc=float(c.mean()))
        if n_c >= 5 and n_i >= 5:
            pr_c = participation_ratio(pf[c])
            pr_i = participation_ratio(pf[~c])
            pr_c_ft = participation_ratio(ft[c])
            pr_i_ft = participation_ratio(ft[~c])
            results[subset].update(
                pr_prefill_correct=float(pr_c),
                pr_prefill_incorrect=float(pr_i),
                pr_prefill_ratio=float(pr_c / pr_i),
                pr_finaltok_correct=float(pr_c_ft),
                pr_finaltok_incorrect=float(pr_i_ft),
                pr_finaltok_ratio=float(pr_c_ft / pr_i_ft),
            )
        else:
            results[subset]["note"] = "too few in one group for reliable PR"
    return results


def d_bucket_analysis(pf15, ft15, c15, sl15):
    """D-bucket analysis — per-user request.

    Loads Exp 2's K=8 majority correctness to define buckets on the 1.5B 500 problems:
      A = greedy K=1 ✓ AND K=8 majority ✓
      B = greedy K=1 ✗ AND K=8 majority ✓  (recoverable)
      C = greedy K=1 ✗ AND K=8 majority ✗  (hopeless)
      D = greedy K=1 ✓ AND K=8 majority ✗  (pathological)
    """
    ph1 = np.load(OUT_DIR.parent / "prefill_gated_compute" / "phase1_majority_vote.npz",
                  allow_pickle=True)
    ph2 = np.load(OUT_DIR.parent / "prefill_gated_compute" / "phase2_prefill_dom.npz",
                  allow_pickle=True)

    # Align by problem index (all 500).
    idx1 = ph1["problem_indices"]
    idx2 = ph2["problem_indices"]
    assert np.array_equal(idx1, np.arange(500)), "phase1 indices mismatch"
    assert np.array_equal(idx2, np.arange(500)), "phase2 indices mismatch"
    assert len(c15) == 500, "1.5B cache must have 500 entries"

    k1_greedy = ph2["correct_k1"].astype(bool)   # from Stage 2 — should match c15 exactly
    k8_maj = np.asarray(ph1["k8_majority_correct"]).astype(bool)
    prefill_score = ph2["prefill_score"].astype(float)
    seq_len_from_ph2 = ph2["seq_len"]

    # Sanity: c15 and k1_greedy should agree
    if not np.array_equal(c15, k1_greedy):
        print(f"  WARNING: c15 ({c15.sum()} correct) != k1_greedy ({k1_greedy.sum()} correct)")
        agreement = (c15 == k1_greedy).mean()
        print(f"  agreement: {agreement:.4f}")

    A = k1_greedy & k8_maj
    B = ~k1_greedy & k8_maj
    C = ~k1_greedy & ~k8_maj
    D = k1_greedy & ~k8_maj

    buckets = {"A_always_right": A, "B_recoverable": B,
               "C_never_right": C, "D_pathological": D}

    per_bucket = {}
    for name, mask in buckets.items():
        n = int(mask.sum())
        entry = dict(n=n)
        if n >= 5:
            entry["pr_prefill"] = float(participation_ratio(pf15[mask]))
            entry["pr_finaltok"] = float(participation_ratio(ft15[mask]))
            sc = prefill_score[mask]
            entry["prefill_score_mean"] = float(sc.mean())
            entry["prefill_score_std"] = float(sc.std())
            entry["prefill_score_median"] = float(np.median(sc))
            entry["prefill_score_quartiles"] = [float(np.percentile(sc, q)) for q in [25, 50, 75]]
            sl = sl15[mask]
            entry["seq_len_mean"] = float(sl.mean())
            entry["seq_len_std"] = float(sl.std())
            entry["seq_len_median"] = float(np.median(sl))
        else:
            entry["note"] = "bucket too small"
        per_bucket[name] = entry

    # Stat tests: is D distinctive?
    comparisons = {}
    if D.sum() >= 5:
        for other_name, other_mask in [("A", A), ("B", B), ("C", C)]:
            if other_mask.sum() < 5:
                continue
            # Prefill score
            u_score, p_score = stats.mannwhitneyu(prefill_score[D], prefill_score[other_mask],
                                                  alternative="two-sided")
            # Seq len
            u_len, p_len = stats.mannwhitneyu(sl15[D], sl15[other_mask], alternative="two-sided")
            # Cohen's d on seq_len
            d_len = (sl15[D].mean() - sl15[other_mask].mean()) / \
                (np.std(np.concatenate([sl15[D], sl15[other_mask]])) + 1e-9)
            comparisons[f"D_vs_{other_name}"] = dict(
                mwu_prefill_score=dict(u=float(u_score), p=float(p_score)),
                mwu_seq_len=dict(u=float(u_len), p=float(p_len)),
                cohen_d_seq_len=float(d_len),
                D_mean_score=float(prefill_score[D].mean()),
                other_mean_score=float(prefill_score[other_mask].mean()),
                D_mean_len=float(sl15[D].mean()),
                other_mean_len=float(sl15[other_mask].mean()),
            )

    return dict(
        bucket_sizes={k: int(v.sum()) for k, v in buckets.items()},
        per_bucket=per_bucket,
        D_vs_others=comparisons,
    )


def main():
    print("=" * 70)
    print("Exp 3 / Phase 2: cross-scale analysis")
    print("=" * 70)
    pf7, ft7, c7, sl7 = load_cached("m7b_prefill")
    pf15, ft15, c15, sl15 = load_cached("m15b_prefill")

    # (a) LOO contribution on 7B
    print("\n(a) LOO PR contribution on 7B ...")
    t0 = time.time()
    loo7 = loo_contribution(pf7, c7, sort_top=10)
    print(f"  base PR_correct={loo7['base_pr_correct']:.3f}, PR_incorrect={loo7['base_pr_incorrect']:.3f}")
    print(f"  top-10 correct-group PR pushers: {loo7['top_correct_pushers']}")
    print(f"  top-10 incorrect-group PR pushers: {loo7['top_incorrect_pushers']}")
    print(f"  {time.time()-t0:.1f}s")

    # Also LOO on 1.5B (as sanity)
    print("(a') LOO PR contribution on 1.5B ...")
    t0 = time.time()
    loo15 = loo_contribution(pf15, c15, sort_top=10)
    print(f"  base PR_correct={loo15['base_pr_correct']:.3f}, PR_incorrect={loo15['base_pr_incorrect']:.3f}")
    print(f"  {time.time()-t0:.1f}s")

    # (b) Three-way split
    print("\n(b) Three-way split ...")
    tw = analyze_three_way(pf7, ft7, pf15, ft15, c7, c15)
    for name, e in tw.items():
        print(f"  {name}: n={e['n']}")
        print(f"    7B  prefill PR={e['pr_7b_prefill']}  final-tok PR={e['pr_7b_finaltok']}")
        print(f"    1.5B prefill PR={e['pr_15b_prefill']}  final-tok PR={e['pr_15b_finaltok']}")

    # (c) BBH
    print("\n(c) BBH per-subset prefill PR ...")
    bbh = bbh_analysis()
    for subset, e in bbh.items():
        print(f"  {subset}: n={e['n']} acc={e['acc']:.3f} "
              f"n_correct={e['n_correct']} n_incorrect={e['n_incorrect']}")
        if "pr_prefill_ratio" in e:
            print(f"    prefill:   PR_c={e['pr_prefill_correct']:.3f} "
                  f"PR_i={e['pr_prefill_incorrect']:.3f} ratio={e['pr_prefill_ratio']:.3f}")
            print(f"    finaltok:  PR_c={e['pr_finaltok_correct']:.3f} "
                  f"PR_i={e['pr_finaltok_incorrect']:.3f} ratio={e['pr_finaltok_ratio']:.3f}")

    # (d) D-bucket
    print("\n(d) D-bucket analysis ...")
    d_out = d_bucket_analysis(pf15, ft15, c15, sl15)
    print(f"  bucket sizes: {d_out['bucket_sizes']}")
    print(f"{'bucket':<18} {'n':>4} {'pr_prefill':>11} {'pr_final':>10} {'score_mean':>11} {'len_mean':>9}")
    for name, e in d_out["per_bucket"].items():
        n = e["n"]
        if n >= 5:
            print(f"  {name:<18} {n:>4} {e['pr_prefill']:>11.3f} {e['pr_finaltok']:>10.3f} "
                  f"{e['prefill_score_mean']:>11.3f} {e['seq_len_mean']:>9.1f}")
    print("\n  D vs other buckets (Mann-Whitney U, two-sided):")
    for cmp_name, e in d_out["D_vs_others"].items():
        print(f"  {cmp_name}: score p={e['mwu_prefill_score']['p']:.2e} "
              f"(D_mean_score={e['D_mean_score']:.3f} vs {e['other_mean_score']:.3f}); "
              f"len p={e['mwu_seq_len']['p']:.2e} "
              f"(D_mean_len={e['D_mean_len']:.1f} vs {e['other_mean_len']:.1f}, d={e['cohen_d_seq_len']:.2f})")

    # Save (excluding large arrays; save them separately)
    loo7_save = {k: v for k, v in loo7.items() if not isinstance(v, np.ndarray)}
    loo15_save = {k: v for k, v in loo15.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(OUT_DIR / "phase2_loo_deltas.npz",
                        delta_correct_7b=loo7["delta_correct"],
                        delta_incorrect_7b=loo7["delta_incorrect"],
                        delta_correct_15b=loo15["delta_correct"],
                        delta_incorrect_15b=loo15["delta_incorrect"])
    out = dict(
        loo_7b=loo7_save,
        loo_15b=loo15_save,
        three_way=tw,
        bbh=bbh,
        d_bucket=d_out,
    )
    with open(OUT_DIR / "phase2_cross_scale.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nSaved: {OUT_DIR/'phase2_cross_scale.json'}")


if __name__ == "__main__":
    main()
