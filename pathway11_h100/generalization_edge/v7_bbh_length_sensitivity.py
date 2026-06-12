#!/usr/bin/env python3
"""v7_bbh_length_sensitivity.py — SPEC v7 deviation D-3 sensitivity analysis.

Discovery (2026-06-12, during Phase 2): the consolidated BBH cache
(nocompute/cache/bbh_1p5b.npz) carries n_gen_tokens ALL-ZERO — a v6
cache-construction artifact. Every v6/v7 BBH "free gate (length+logprob)"
number was therefore logprob-only de facto, including the pinned MATH->BBH
0.785 transfer anchor and Phase 1a/1b's BBH free gate 0.7823.

The pod rescore sidecar provides real BBH generation lengths. This script
quantifies the correction WITHOUT mutating any pinned artifact:
  - honest in-domain BBH free gate (real length + logprob), vs pinned;
  - real length alone;
  - the Phase-1b H-I "winner" (mean_token_entropy) re-adjudicated against
    the honest gate.

Output: results/v7_bbh_length_sensitivity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import activation_loader as AL                                   # noqa: E402
from metrics import frozen_folds, auroc_sym, delong_paired_test  # noqa: E402
from v7_phase1a import oof_scores, impute_median, arm_row        # noqa: E402

RESULTS = HERE / "results"


def main():
    bbh = AL.load_cell("qwen1.5b", "bbh")
    side = np.load(RESULTS / "v7_rescore_qwen1.5b_bbh.npz")
    y = bbh.y
    folds = frozen_folds(y)
    ng_real = side["n_gen"].astype(float)
    assert float(bbh.n_gen_tokens.std()) == 0.0, \
        "cache n_gen_tokens no longer degenerate — rescope this analysis"

    X_pin = np.column_stack([bbh.n_gen_tokens, bbh.mean_logprob])
    s_pin = oof_scores(X_pin, y, folds)
    a_pin = auroc_sym(y, s_pin)
    X_hon = np.column_stack([ng_real, bbh.mean_logprob])
    s_hon = oof_scores(X_hon, y, folds)
    a_hon = auroc_sym(y, s_hon)
    dl = delong_paired_test(y, s_hon, s_pin)

    ent, _ = impute_median(side["mean_entropy"].astype(float))
    r_ent = arm_row("mean_token_entropy_vs_honest", ent, X_hon, y, folds,
                    s_hon, a_hon)
    r_len = arm_row("real_length_vs_pinned", ng_real, X_pin, y, folds,
                    s_pin, a_pin)

    # honest cross-domain transfer + honest H-L2 re-check (Phase-2 ripple:
    # under the degenerate gate H-L2's no-harm was vacuous — any 2-feature
    # refit collapses to the same logprob-only ranking)
    import gzip
    from collections import Counter
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from v7_phase2_pseudolabel import payload_rows, modal
    from v7_runpod_arms import bbh_extract_answer
    math = AL.load_cell("qwen1.5b", "math")
    Xs = np.column_stack([math.n_gen_tokens, math.mean_logprob])
    sc = StandardScaler().fit(Xs)
    est = LogisticRegression(max_iter=2000).fit(sc.transform(Xs), math.y)
    s_t = est.predict_proba(sc.transform(X_hon))[:, 1]
    honest_transfer = auroc_sym(y, s_t)
    with gzip.open(HERE / "pod_results" / "v7_k8bbh_qwen1.5b.json.gz",
                   "rt") as fh:
        k8 = json.load(fh)
    rows = sorted(k8["rows"], key=lambda r: r["idx"])
    payload = payload_rows("qwen1.5b_bbh")
    greedy = [bbh_extract_answer(p["text"]) for p in payload]
    subsets = np.array([p["subset"] for p in payload])
    yhat8 = np.array([greedy[i] == modal(r["answers"][:8]) and greedy[i] != ""
                      for i, r in enumerate(rows)])
    dev_m = ~(subsets == "web_of_lies")
    conf_m = subsets == "web_of_lies"
    sc2 = StandardScaler().fit(X_hon[dev_m])
    est2 = LogisticRegression(max_iter=2000).fit(
        sc2.transform(X_hon[dev_m]), yhat8[dev_m])
    s_refit = est2.predict_proba(sc2.transform(X_hon[conf_m]))[:, 1]
    hl2_honest = {
        "honest_frozen_true_auroc_web_of_lies":
            auroc_sym(y[conf_m], s_t[conf_m]),
        "honest_pseudo_refit_true_auroc_web_of_lies":
            auroc_sym(y[conf_m], s_refit)}
    hl2_honest["no_harm"] = bool(
        hl2_honest["honest_pseudo_refit_true_auroc_web_of_lies"]
        >= hl2_honest["honest_frozen_true_auroc_web_of_lies"] - 0.01)

    out = {"spec": "v7 deviation D-3 (BBH n_gen_tokens all-zero in cache)",
           "honest_math_to_bbh_transfer_auroc": honest_transfer,
           "pinned_transfer_anchor_de_facto_logprob_only": 0.785,
           "H_L2_honest_recheck": hl2_honest,
           "pinned_free_gate_auroc_logprob_only_de_facto": a_pin,
           "honest_free_gate_auroc_real_length": a_hon,
           "honest_vs_pinned": {"delta": a_hon - a_pin, "delong_z": dl["z"],
                                "delong_p": dl["p"]},
           "real_length_arm": r_len,
           "entropy_vs_honest_gate": r_ent,
           "verdict": ("H-I refutation in the BBH cell does NOT survive the "
                       "honest gate: entropy's +0.0151 over the degenerate "
                       "gate becomes -0.0004 over length+logprob — entropy "
                       "was proxying the missing length feature. Under the "
                       "honest gate H-I HOLDS in all dev cells and the "
                       "free-gate thesis strengthens (BBH in-domain 0.806 "
                       "> 0.782).")}
    f = RESULTS / "v7_bbh_length_sensitivity.json"
    f.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {f}")
    print(f"pinned {a_pin:.4f} -> honest {a_hon:.4f} "
          f"(d={a_hon-a_pin:+.4f}, p={dl['p']:.4f}); "
          f"length alone {r_len['standalone_auroc']:.4f}; "
          f"entropy vs honest d={r_ent['incremental_delta']:+.4f} "
          f"p={r_ent['delong_p']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
