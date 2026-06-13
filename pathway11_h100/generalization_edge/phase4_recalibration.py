#!/usr/bin/env python3
"""Phase 4 (SPEC v6, V5-5) — per-domain recalibration recipe (first-class deliverable).

1B.3 CONFIRMED H-H: a conformal/LTT guarantee is valid in-domain but collapses to
0 coverage under the MATH->BBH shift. So a deployable cross-domain guarantee REQUIRES
target-domain recalibration. This phase measures: how little target-domain labelled
data re-fits a deployable F-8 gate?

Recipe: freeze the source-fit extractor, on the target cell re-fit ONLY a light head
on k in {0,8,16,32,64} labelled target examples, score the held-out remainder.
  - free_baseline (length+logprob): head = StandardScaler+Logistic (RE-WEIGHTS the two
    free scalars -> AUROC can move; this is the pinned V5-5 readout).
  - prefill_dom (frozen MATH DoM direction): a 1-D projected score; a monotone head
    cannot change AUROC, so what k buys here is CALIBRATION (valid threshold / conformal
    coverage) -- exactly the H-H collapse restored.

Three metrics per k, averaged over R stratified resamples of the support set, scoring
on the remainder (scoring n stated): (a) AUROC, (b) selective accuracy @ 50% coverage,
(c) conformal coverage + validity at (eps=0.2, delta=0.1) with tau chosen on the k
labels. k=0 = zero-shot transfer (source head applied directly). Ceiling = target
in-domain OOF. Power (V2-3): large cells only (BBH n=750, Qwen-7B MATH n=500).

Output: results/phase4_recalibration.json. Local CPU only.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)

import activation_loader as AL
import probes as PR
from metrics import DomProbe, frozen_folds, auroc_sym, SEED
from exp_1b3_conformal import choose_tau, cp_upper  # reuse the CP-upper LTT machinery

K_GRID = [0, 8, 16, 32, 64]
EPS = 0.2          # primary deployable target
EPS_LIST = [0.2, 0.3]  # also report a looser cert (V5-5 cross-domain crux)
DELTA = 0.1
R = 200  # resamples of the support set per k


def _free_X(cell):
    return np.column_stack([cell.n_gen_tokens, cell.mean_logprob])


def _stratified_support(y, k, rng):
    """Pick k indices, balanced across classes; return (support_idx, remainder_idx)."""
    pos = np.where(y)[0]; neg = np.where(~y)[0]
    kp = min(len(pos), k // 2); kn = min(len(neg), k - kp)
    sup = np.concatenate([rng.choice(pos, kp, replace=False),
                          rng.choice(neg, kn, replace=False)])
    mask = np.zeros(len(y), bool); mask[sup] = True
    return sup, np.where(~mask)[0]


def _selective_acc_at_cov(scores, y, cov=0.5):
    """Answer the top-cov fraction by score; accuracy among answered."""
    n_ans = max(1, int(round(cov * len(scores))))
    order = np.argsort(-scores)[:n_ans]
    return float(y[order].mean())


def _conformal_acc(acc):
    """Aggregate per-resample (feasible, coverage, valid) lists into a summary dict."""
    out = {}
    for eps, (feas, covs, valids) in acc.items():
        out[f"eps{eps}"] = {
            "feasible_frac": (sum(feas) / len(feas)) if feas else None,
            "mean_coverage": float(np.mean(covs)) if covs else 0.0,
            "validity": float(np.mean(valids)) if valids else None,
        }
    return out


def recalibrate_free(src, tgt):
    """free_baseline: refit the 2-D logistic head on k target labels (re-weighting)."""
    Xs, ys = _free_X(src), src.y
    Xt, yt = _free_X(tgt), tgt.y
    sc0 = StandardScaler().fit(Xs)
    est0 = LogisticRegression(max_iter=2000).fit(sc0.transform(Xs), ys)
    rng = np.random.default_rng(SEED)
    rows = {}
    for k in K_GRID:
        aurocs, selacc, scoring_ns = [], [], []
        conf = {e: ([], [], []) for e in EPS_LIST}   # eps -> (feasible, covs, valids)
        for _ in range(R if k > 0 else 1):
            if k == 0:
                rem = np.arange(len(yt))
                s = est0.predict_proba(sc0.transform(Xt))[:, 1]
                s_sup, y_sup = est0.predict_proba(sc0.transform(Xs))[:, 1], ys  # source cal
            else:
                sup, rem = _stratified_support(yt, k, rng)
                if len(np.unique(yt[sup])) < 2:
                    continue
                sc = StandardScaler().fit(Xt[sup])
                est = LogisticRegression(max_iter=2000).fit(sc.transform(Xt[sup]), yt[sup])
                s = est.predict_proba(sc.transform(Xt[rem]))[:, 1]
                s_sup, y_sup = est.predict_proba(sc.transform(Xt[sup]))[:, 1], yt[sup]
            yr = yt[rem]; scoring_ns.append(len(rem))
            aurocs.append(auroc_sym(yr, s)); selacc.append(_selective_acc_at_cov(s, yr))
            for eps in EPS_LIST:
                tau = choose_tau(s_sup, y_sup, eps, DELTA)
                f, c, v = conf[eps]
                if tau is None:
                    f.append(0); c.append(0.0)
                else:
                    f.append(1); ans = s >= tau
                    if ans.sum() > 0:
                        c.append(float(ans.mean())); v.append((1 - yr[ans].mean()) <= eps)
                    else:
                        c.append(0.0)
        rows[f"k{k}"] = {
            "scoring_n": int(np.mean(scoring_ns)) if scoring_ns else len(yt),
            "auroc": float(np.mean(aurocs)), "auroc_sd": float(np.std(aurocs)),
            "sel_acc@0.5cov": float(np.mean(selacc)),
            "conformal": _conformal_acc(conf),
        }
    return rows


def recalibrate_dom(src, tgt):
    """prefill_dom: frozen MATH DoM direction; 1-D score. Monotone head => AUROC fixed;
    k buys a valid threshold / conformal coverage only (the H-H restoration)."""
    direction = DomProbe().fit(src.X("prefill"), src.y)
    s_all = direction.score(tgt.X("prefill"))             # frozen projected score
    if auroc_sym(tgt.y, s_all) > 0.5 and np.corrcoef(s_all, tgt.y)[0, 1] < 0:
        s_all = -s_all
    yt = tgt.y
    auroc_frozen = auroc_sym(yt, s_all)                   # invariant to k (monotone head)
    s_src = direction.score(src.X("prefill"))
    rng = np.random.default_rng(SEED + 1)
    rows = {}
    for k in K_GRID:
        selacc, scoring_ns = [], []
        conf = {e: ([], [], []) for e in EPS_LIST}
        for _ in range(R if k > 0 else 1):
            if k == 0:
                rem = np.arange(len(yt)); s_cal, y_cal = s_src, src.y   # source-cal (H-H)
            else:
                sup, rem = _stratified_support(yt, k, rng)
                if len(np.unique(yt[sup])) < 2:
                    continue
                s_cal, y_cal = s_all[sup], yt[sup]
            yr = yt[rem]; scoring_ns.append(len(rem))
            selacc.append(_selective_acc_at_cov(s_all[rem], yr))
            for eps in EPS_LIST:
                tau = choose_tau(s_cal, y_cal, eps, DELTA)
                f, c, v = conf[eps]
                if tau is None:
                    f.append(0); c.append(0.0)
                else:
                    f.append(1); ans = s_all[rem] >= tau
                    if ans.sum() > 0:
                        c.append(float(ans.mean())); v.append((1 - yr[ans].mean()) <= eps)
                    else:
                        c.append(0.0)
        rows[f"k{k}"] = {
            "scoring_n": int(np.mean(scoring_ns)) if scoring_ns else len(yt),
            "auroc_frozen": float(auroc_frozen),
            "sel_acc@0.5cov": float(np.mean(selacc)),
            "conformal": _conformal_acc(conf),
        }
    return rows


def target_ceiling(tgt, readout):
    """In-domain OOF AUROC on the target cell = the recovery ceiling."""
    folds = frozen_folds(tgt.y)
    if readout == "free_baseline":
        s = PR.fit_score_oof(PR.REGISTRY["free_baseline"], tgt, folds)
    else:
        from metrics import oof_dom_scores
        s = oof_dom_scores(tgt.X("prefill"), tgt.y)
    return float(auroc_sym(tgt.y, s))


def main():
    math = AL.load_cell("qwen1.5b", "math")
    bbh = AL.load_cell("qwen1.5b", "bbh")
    m7 = AL.load_cell("qwen7b", "math")

    out = {"eps": EPS, "delta": DELTA, "R": R, "k_grid": K_GRID,
           "note": "freeze source extractor, refit light head on k target labels; "
                   "large cells only (V2-3 power)."}

    # --- Cross-domain MATH->BBH (the H-H collapse case; n=750) ---
    print("=== MATH -> BBH (cross-domain; H-H collapse) ===")
    ceil_free_bbh = target_ceiling(bbh, "free_baseline")
    ceil_dom_bbh = target_ceiling(bbh, "prefill_dom")
    rec_free = recalibrate_free(math, bbh)
    rec_dom = recalibrate_dom(math, bbh)
    out["math_to_bbh"] = {
        "target_n": int(bbh.n),
        "free_baseline": {"in_domain_ceiling": ceil_free_bbh, "by_k": rec_free},
        "prefill_dom": {"in_domain_ceiling": ceil_dom_bbh, "by_k": rec_dom},
    }
    for nm, ceil, rec in [("free_baseline", ceil_free_bbh, rec_free),
                          ("prefill_dom", ceil_dom_bbh, rec_dom)]:
        z = rec["k0"].get("auroc", rec["k0"].get("auroc_frozen"))
        print(f"  {nm}: ceiling={ceil:.3f}  zero-shot(k0)~{z:.3f}")
        for k in K_GRID:
            r = rec[f"k{k}"]; a = r.get("auroc", r.get("auroc_frozen"))
            c2 = r["conformal"]["eps0.2"]; c3 = r["conformal"]["eps0.3"]
            print(f"    k={k:2d} n={r['scoring_n']:3d} auroc={a:.3f} "
                  f"selacc@.5={r['sel_acc@0.5cov']:.3f} | "
                  f"e=.2 cov={c2['mean_coverage']:.2f} valid={c2['validity']} | "
                  f"e=.3 cov={c3['mean_coverage']:.2f} valid={c3['validity']}")

    # --- Cross-scale 1.5B->7B (free baseline only; n=500) ---
    print("\n=== MATH 1.5B -> 7B (cross-scale; free baseline) ===")
    ceil_free_7b = target_ceiling(m7, "free_baseline")
    rec_free_7b = recalibrate_free(math, m7)
    out["math1.5b_to_7b"] = {
        "target_n": int(m7.n),
        "free_baseline": {"in_domain_ceiling": ceil_free_7b, "by_k": rec_free_7b},
    }
    print(f"  free_baseline: ceiling={ceil_free_7b:.3f}")
    for k in K_GRID:
        r = rec_free_7b[f"k{k}"]; c2 = r["conformal"]["eps0.2"]; c3 = r["conformal"]["eps0.3"]
        print(f"    k={k:2d} n={r['scoring_n']:3d} auroc={r['auroc']:.3f} "
              f"selacc@.5={r['sel_acc@0.5cov']:.3f} | "
              f"e=.2 cov={c2['mean_coverage']:.2f} valid={c2['validity']} | "
              f"e=.3 cov={c3['mean_coverage']:.2f} valid={c3['validity']}")

    (RESULTS / "phase4_recalibration.json").write_text(json.dumps(out, indent=2))
    print(f"\n→ {RESULTS / 'phase4_recalibration.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
