#!/usr/bin/env python3
"""v7_phase1a.py — SPEC v7 Phase 1a (EXP-84): zero-cost arms, local analysis.

Inputs: v7_rescore_local.py sidecars (MATH + BBH), the K=8 self-consistency
cache (1.5B MATH), the arm-2A prompt-cloud eigenspectrum cache.

Per arm: standalone OOF AUROC (frozen folds, LR protocol identical to
probes.fit_score_oof) + DeLong incremental gate of [arm + free] vs [free].
Gate G1: promote to pod time iff (incremental Δ>0, p<0.05) OR
(standalone ≥ free gate − 0.01) on in-domain dev.

H-K (spectral-α closure row): prefill-α adds over prompt-length in-domain
(paired Δ>0, p<0.05). Refuted → α CLOSED permanently.

Perplexity is documented-and-skipped: exp(−mean_logprob) is a monotone
transform of mean_logprob, AUROC-identical by rank invariance.

Output: results/v7_phase1a.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import activation_loader as AL                       # noqa: E402
from metrics import frozen_folds, auroc_sym, delong_paired_test  # noqa: E402
import k8_lib                                        # noqa: E402
from pathway8_layerwise.extract_math500 import _extract_boxed  # noqa: E402

RESULTS = HERE / "results"


def oof_scores(X: np.ndarray, y: np.ndarray, folds) -> np.ndarray:
    """probes.fit_score_oof protocol for plain feature matrices."""
    sc = np.zeros(len(y), dtype=np.float64)
    for tr, te in folds:
        s = StandardScaler().fit(X[tr])
        est = LogisticRegression(max_iter=2000, C=1.0).fit(
            s.transform(X[tr]), y[tr])
        sc[te] = est.predict_proba(s.transform(X[te]))[:, 1]
    return sc


def impute_median(col: np.ndarray) -> tuple[np.ndarray, float]:
    """NaN -> column median; returns (imputed, finite_fraction)."""
    out = col.copy().astype(np.float64)
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = np.median(out[~bad])
    return out, float(1.0 - bad.mean())


def arm_row(name, x_arm, base_X, y, folds, free_oof, free_auroc):
    """Standalone + incremental-over-free for one arm column (or matrix)."""
    X_arm = x_arm if x_arm.ndim == 2 else x_arm.reshape(-1, 1)
    s_alone = oof_scores(X_arm, y, folds)
    s_comb = oof_scores(np.column_stack([X_arm, base_X]), y, folds)
    dl = delong_paired_test(y, s_comb, free_oof)
    a_alone = auroc_sym(y, s_alone)
    a_comb = auroc_sym(y, s_comb)
    delta = a_comb - free_auroc
    g1 = bool((delta > 0 and dl["p"] < 0.05) or (a_alone >= free_auroc - 0.01))
    return {"arm": name, "standalone_auroc": a_alone,
            "combined_auroc": a_comb, "free_auroc": free_auroc,
            "incremental_delta": delta, "delong_z": dl["z"], "delong_p": dl["p"],
            "G1_promote": g1}


def run_cell(cellname: str) -> dict:
    cell = AL.load_cell("qwen1.5b", cellname)
    side = np.load(RESULTS / f"v7_rescore_qwen1.5b_{cellname}.npz")
    assert bool((side["y"] == cell.y).all()), "sidecar/cell label mismatch"
    r_align = float(side["alignment_corr"])
    y = cell.y
    folds = frozen_folds(y)
    base_X = np.column_stack([cell.n_gen_tokens, cell.mean_logprob])
    free_oof = oof_scores(base_X, y, folds)
    free_auroc = auroc_sym(y, free_oof)

    rows = []
    arms = [("min_token_logprob", side["min_lp"]),
            ("bottom_decile_logprob", side["p10_lp"]),
            ("mean_token_entropy", side["mean_entropy"]),
            ("mean_top2_logit_margin", side["mean_top2_margin"]),
            ("answer_span_logprob", side["ans_span_lp"])]
    coverage = {}
    for name, col in arms:
        x, cov = impute_median(col)
        coverage[name] = cov
        rows.append(arm_row(name, x, base_X, y, folds, free_oof, free_auroc))
    # all-rescore-features joint arm (the candidate-gate upper bound)
    X_all = np.column_stack([impute_median(c)[0] for _, c in arms])
    rows.append(arm_row("rescore_family_joint", X_all, base_X, y, folds,
                        free_oof, free_auroc))
    return {"alignment_corr": r_align, "free_gate_oof_auroc": free_auroc,
            "answer_span_coverage": coverage, "arms": rows}


def run_k_consistency() -> dict:
    """K-sample agreement/cluster-entropy arms from the existing K=8 cache.

    Signal view: predict the GREEDY answer's correctness (cell.y).
    System view: predict K=8 mode-vote correctness (k8_lib.plain_majority).
    Exact-match clustering = semantic entropy for math (2406.15927 triaged,
    no novelty claim).
    """
    cell = AL.load_cell("qwen1.5b", "math")
    y = cell.y
    folds = frozen_folds(y)
    base_X = np.column_stack([cell.n_gen_tokens, cell.mean_logprob])
    free_oof = oof_scores(base_X, y, folds)
    free_auroc = auroc_sym(y, free_oof)

    n = AL.n_k8_problems()
    assert n == len(y), (n, len(y))
    greedy_ans = []
    for i in range(n):
        d = np.load(AL.TRAJ_1P5B / f"problem_{i:03d}.npz", allow_pickle=True)
        greedy_ans.append(k8_lib.extract_answers([str(d["text"])])[0])

    feats = {f"k{K}_agreement": np.zeros(n) for K in (2, 4, 8)}
    feats.update({f"k{K}_cluster_entropy": np.zeros(n) for K in (2, 4, 8)})
    y_majority = np.zeros(n, dtype=bool)
    for i in range(n):
        k8 = AL.load_k8(i)
        answers = k8_lib.extract_answers(k8["texts"])
        y_majority[i] = k8_lib.plain_majority_correct(answers, k8["correct"])
        for K in (2, 4, 8):
            sub = answers[:K]
            feats[f"k{K}_agreement"][i] = float(
                np.mean([a == greedy_ans[i] and a != "" for a in sub]))
            _, counts = np.unique(sub, return_counts=True)
            p = counts / K
            feats[f"k{K}_cluster_entropy"][i] = float(-(p * np.log(p)).sum())

    rows = []
    for name, col in feats.items():
        rows.append(arm_row(name, col, base_X, y, folds, free_oof, free_auroc))

    # system-view AUROCs: same features predicting mode-vote correctness
    folds_m = frozen_folds(y_majority)
    base_m = oof_scores(base_X, y_majority, folds_m)
    sys_rows = []
    for name, col in feats.items():
        s = oof_scores(col.reshape(-1, 1), y_majority, folds_m)
        sys_rows.append({"arm": name,
                         "auroc_vs_majority_label": auroc_sym(y_majority, s)})
    return {"free_gate_oof_auroc": free_auroc,
            "acc_k8_majority": float(y_majority.mean()),
            "signal_view": rows,
            "system_view_note": ("cost adjudication (H-J vs cascade) happens "
                                 "in Phase 1b; these are label-semantics-2 "
                                 "AUROCs only"),
            "system_view": sys_rows,
            "free_gate_auroc_vs_majority_label": auroc_sym(
                y_majority, base_m)}


def run_alpha_closure() -> dict:
    """H-K: prefill spectral-α over prompt-length (arm-2A eigenspectrum cache)."""
    cell = AL.load_cell("qwen1.5b", "math")
    y = cell.y
    folds = frozen_folds(y)
    d = np.load(HERE / "cache" / "arm2a_prompt_cloud_qwen1.5b_math.npz")
    ev = d["eigvals"]                       # (500, 20)
    plen = d["prompt_len"].astype(np.float64)
    logr = np.log(np.arange(1, ev.shape[1] + 1))
    alpha = np.zeros(len(y))
    for i in range(len(y)):
        le = np.log(np.maximum(ev[i], 1e-12))
        alpha[i] = np.polyfit(logr, le, 1)[0]

    X_plen = plen.reshape(-1, 1)
    s_plen = oof_scores(X_plen, y, folds)
    a_plen = auroc_sym(y, s_plen)
    s_alpha = oof_scores(alpha.reshape(-1, 1), y, folds)
    s_comb = oof_scores(np.column_stack([alpha, plen]), y, folds)
    dl = delong_paired_test(y, s_comb, s_plen)
    a_comb = auroc_sym(y, s_comb)
    delta = a_comb - a_plen
    survives = bool(delta > 0 and dl["p"] < 0.05)
    return {"alpha_standalone_auroc": auroc_sym(y, s_alpha),
            "prompt_length_auroc": a_plen,
            "alpha_plus_length_auroc": a_comb,
            "incremental_delta": delta, "delong_z": dl["z"], "delong_p": dl["p"],
            "H_K_survives": survives,
            "verdict": ("alpha adds over prompt-length" if survives else
                        "H-K REFUTED — spectral-alpha CLOSED permanently, "
                        "zero further compute, no T5 pass")}


def main():
    out = {"spec": "v7 Phase 1a (EXP-84)",
           "perplexity_note": ("perplexity = exp(-mean_logprob) is a monotone "
                               "transform of mean_logprob; AUROC identical by "
                               "rank invariance — documented and skipped"),
           "math": run_cell("math"),
           "bbh": run_cell("bbh"),
           "k_consistency_math": run_k_consistency(),
           "alpha_closure_HK": run_alpha_closure()}

    # G1 roll-up
    promoted = sorted({r["arm"]
                       for c in ("math", "bbh")
                       for r in out[c]["arms"] if r["G1_promote"]}
                      | {r["arm"] for r in
                         out["k_consistency_math"]["signal_view"]
                         if r["G1_promote"]})
    out["G1_promoted_arms"] = promoted

    f = RESULTS / "v7_phase1a.json"
    f.write_text(json.dumps(out, indent=2, default=float))
    print(f"-> {f}")
    print(f"\nfree gate OOF: math={out['math']['free_gate_oof_auroc']:.4f} "
          f"bbh={out['bbh']['free_gate_oof_auroc']:.4f}")
    print(f"H-K survives: {out['alpha_closure_HK']['H_K_survives']} "
          f"(delta={out['alpha_closure_HK']['incremental_delta']:+.4f}, "
          f"p={out['alpha_closure_HK']['delong_p']:.3f})")
    print(f"G1 promoted arms: {promoted or 'NONE'}")
    for c in ("math", "bbh"):
        print(f"\n{c} arms:")
        for r in out[c]["arms"]:
            print(f"  {r['arm']:26s} alone={r['standalone_auroc']:.4f} "
                  f"comb={r['combined_auroc']:.4f} "
                  f"d={r['incremental_delta']:+.4f} p={r['delong_p']:.3f} "
                  f"{'PROMOTE' if r['G1_promote'] else '-'}")
    print("\nk-consistency (signal view, MATH):")
    for r in out["k_consistency_math"]["signal_view"]:
        print(f"  {r['arm']:26s} alone={r['standalone_auroc']:.4f} "
              f"comb={r['combined_auroc']:.4f} "
              f"d={r['incremental_delta']:+.4f} p={r['delong_p']:.3f} "
              f"{'PROMOTE' if r['G1_promote'] else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
