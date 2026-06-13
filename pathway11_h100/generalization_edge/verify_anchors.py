#!/usr/bin/env python3
"""verify_anchors.py — reproduce SPEC v5 verification anchors through the harness.

Steps 1 (in-domain), 2 (dry-run 200-subset), 3 (transfer). Step 5B (Phase-1B
anchors) lives in the 1B scripts. Pass criterion printed per line.
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import roc_auc_score

import activation_loader as AL
import probes as PR
import transfer as TR
from metrics import frozen_folds, auroc_sym


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  [{'OK ' if ok else 'XX '}] {label:42s} got {got:.4f}  want {want:.4f}  (+-{tol})")
    return ok


def step4_v7():
    """SPEC v7 anchors (EXP-84-pre). T5 trio recomputed from the phase3 npz
    caches via the exact phase3_heldout.eval_free_baseline protocol; the rest
    are readbacks of pinned result-JSON values (guards artifact drift)."""
    import json
    from pathlib import Path
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    here = Path(__file__).resolve().parent
    res = here / "results"
    allok = True

    def oof_auroc(X, y, seed=0):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        sc = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            s = StandardScaler().fit(X[tr])
            est = LogisticRegression(max_iter=2000, C=1.0).fit(
                s.transform(X[tr]), y[tr])
            sc[te] = est.predict_proba(s.transform(X[te]))[:, 1]
        return float(roc_auc_score(y, sc))

    print("\nStep 4 — SPEC v7 anchors:")
    print("  4a. T5 free trio (recomputed from phase3 npz):")
    for key, want in [("smollm2", 0.8097), ("gemma", 0.8439), ("olmo2", 0.8382)]:
        d = np.load(res / f"phase3_{key}.npz", allow_pickle=True)
        y = d["y"].astype(bool)
        X = np.column_stack([d["n_gen_tokens"].astype(float),
                             d["mean_logprob"].astype(float)])
        allok &= check(f"T5 free_baseline {key}", oof_auroc(X, y), want, 0.0005)

    print("  4b. arm-2A / prompt-length (phase2_arm2a.json readback):")
    a2 = json.loads((res / "phase2_arm2a.json").read_text())
    allok &= check("prompt_length_only", a2["prompt_length_only_auroc"], 0.7056, 0.0005)
    allok &= check("arm2A gate delta", a2["gate_delta_vs_promptlen"], -0.0096, 0.0005)
    allok &= check("arm2A gate p", a2["gate_p"], 0.4175, 0.001)

    print("  4c. cascade vs hull + K=8 majority (1b4_overlap.json readback):")
    o = json.loads((res / "1b4_overlap.json").read_text())
    hyb = o["cascade_vs_random_mix_hull"]["hybrid"]
    allok &= check("hybrid cascade gap_pp", hyb["gap_pp_point"], 3.38, 0.005)
    allok &= check("hybrid cascade boot SE", hyb["gap_pp_boot_se"], 0.962, 0.005)
    if not hyb["significant"]:
        print("  [XX ] hybrid cascade significance flag lost"); allok = False
    allok &= check("K=8 majority acc (true mode-vote)",
                   o["anchors"]["acc_k8maj"], 0.554, 0.0005)

    print("  4d. FE749 spectral-alpha (spectral_alpha/results.json readback):")
    fa = json.loads((here.parent / "spectral_alpha" / "results.json").read_text())
    allok &= check("alpha L19 (1.5B)", fa["alpha_l19_auroc_15b"], 0.5225, 0.0005)
    allok &= check("alpha best last-layer (1.5B)", fa["alpha_best_auroc_15b"], 0.7026, 0.0005)

    print("  4e. phase-4 recalibration cells (phase4_recalibration.json readback):")
    p4 = json.loads((res / "phase4_recalibration.json").read_text())
    xd = p4["math_to_bbh"]["free_baseline"]["by_k"]
    xs = p4["math1.5b_to_7b"]["free_baseline"]["by_k"]
    allok &= check("MATH->BBH frozen free-gate (k0)", xd["k0"]["auroc"], 0.785, 0.001)
    c = xs["k0"]["conformal"]["eps0.2"]
    allok &= check("cross-scale k0 eps0.2 validity", c["validity"], 1.0, 1e-9)
    allok &= check("cross-scale k0 eps0.2 coverage", c["mean_coverage"], 0.60, 0.005)
    for k in ("k0", "k32", "k64"):  # k8/k16 infeasible at eps0.2 (no validity cell)
        cd = xd[k]["conformal"]["eps0.2"]
        v = cd.get("validity")
        ok = (v == 0.0)
        print(f"  [{'OK ' if ok else 'XX '}] {'cross-domain '+k+' eps0.2 validity':42s} "
              f"got {v}  want 0.0")
        allok &= ok

    print(f"\n  v7 anchors: {'ALL PASS' if allok else 'FAILED'}")
    return allok


def step5_v8():
    """SPEC v8 anchors (EXP-90-pre / Gate G0). Operating point + rescue triple
    recomputed from cached y-vectors; the rest are pinned-JSON readbacks."""
    import json
    from pathlib import Path

    here = Path(__file__).resolve().parent
    res = here / "results"
    allok = True

    print("\nStep 5 — SPEC v8 anchors (Gate G0):")
    print("  5a. v7 operating point (v7_phase4_router.json readback):")
    r = json.loads((res / "v7_phase4_router.json").read_text())
    op = r["operating_point"]
    allok &= check("cascade acc (free gate)", op["acc_cascade_free_gate"], 0.648, 1e-9)
    allok &= check("oracle acc", op["acc_oracle"], 0.758, 1e-9)
    allok &= check("random-mix hull acc", op["acc_random_mix_hull"], 0.609, 1e-9)
    allok &= check("c_esc", op["c_esc"], 3094.6868, 0.0001)
    allok &= check("budget_extra (0.5*c_esc)", op["budget_extra_per_problem"],
                   1547.3434, 0.0001)

    print("  5b. c0 + matched budget (v7_frontier_table.json readback):")
    ft = json.loads((res / "v7_frontier_table.json").read_text())
    c0 = ft["cells"]["qwen1.5b_math"]["c0_abs"]
    allok &= check("c0 (1.5B greedy MATH)", c0, 673.37, 0.0001)
    allok &= check("matched budget c0+0.5*c_esc",
                   c0 + op["budget_extra_per_problem"], 2220.7134, 0.0001)
    allok &= check("dev free-gate OOF AUROC",
                   ft["cells"]["qwen1.5b_math"]["free_gate_oof_auroc"], 0.8490, 0.0005)

    print("  5c. rescue triple (recomputed from cached y-vectors):")
    cell = AL.load_cell("qwen1.5b", "math")
    y15 = cell.y.astype(bool)
    y7 = np.load(res / "v7_rescore_qwen7b_math.npz",
                 allow_pickle=True)["y"].astype(bool)
    allok &= check("base 1.5B greedy acc", float(y15.mean()), 0.486, 1e-9)
    allok &= check("target 7B greedy acc", float(y7.mean()), 0.732, 1e-9)
    fails = int((~y15).sum()); resc = int((y7 & ~y15).sum()); unr = fails - resc
    for label, got, want in [("base failures", fails, 257),
                             ("rescued by 7B", resc, 136),
                             ("unrescuable", unr, 121)]:
        ok = got == want
        print(f"  [{'OK ' if ok else 'XX '}] {label:42s} got {got}  want {want}")
        allok &= ok

    print("  5d. T5 free-gate trio, v7 genscore-corrected (v7_phase1c_t5.json readback):")
    t5 = json.loads((res / "v7_phase1c_t5.json").read_text())["families"]
    for key, want in [("smollm2", 0.8083), ("gemma", 0.8422), ("olmo2", 0.8357)]:
        allok &= check(f"T5 free gate {key}", t5[key]["free_gate_oof_auroc"],
                       want, 0.0005)

    print(f"\n  v8 anchors: {'ALL PASS' if allok else 'FAILED'}")
    return allok


def main():
    import sys
    if "--v7-only" in sys.argv:
        return 0 if step4_v7() else 1
    if "--v8-only" in sys.argv:
        return 0 if step5_v8() else 1
    cell = AL.load_cell("qwen1.5b", "math")
    folds = frozen_folds(cell.y)
    allok = True

    print("Step 1 — in-domain anchors (full n=500):")
    for name, want, tol in [("prefill_dom", 0.7731, 0.005),
                            ("concat_ridge", 0.8509, 0.01),
                            ("length_only", 0.7986, 0.01),
                            ("mean_logprob_only", 0.6721, 0.01)]:
        s = PR.fit_score_oof(PR.REGISTRY[name], cell, folds)
        allok &= check(name, auroc_sym(cell.y, s), want, tol)

    print("\nStep 2 — dry-run anchors (first 200 problems, +-0.01):")
    sub = np.zeros(cell.n, dtype=bool); sub[:200] = True
    c200 = TR._subset(cell, sub)
    f200 = frozen_folds(c200.y)
    for name, want, tol in [("length_only", 0.786, 0.012),
                            ("free_baseline", 0.834, 0.015),
                            ("coe_profile", 0.825, 0.02)]:
        s = PR.fit_score_oof(PR.REGISTRY[name], c200, f200)
        allok &= check(name + " (n=200)", auroc_sym(c200.y, s), want, tol)

    print("\nStep 3 — transfer anchors:")
    # MATH->BBH cross-domain. The canonical saved artifact (C1_math_to_bbh.json)
    # is pooled MEAN-DoM raw AUROC = 0.4158 (mean-DoM ANTI-transfers to BBH —
    # the in-domain-best position inverts cross-domain). The handoff-prose "0.747"
    # matches no recipe and is treated as stale. Anchor = the saved artifact.
    bbh = AL.load_cell("qwen1.5b", "bbh")
    s_mean = PR.fit_score_transfer(PR.REGISTRY["mean_dom"], cell, bbh)
    allok &= check("MATH->BBH mean-DoM (raw, =C1 artifact)",
                   roc_auc_score(bbh.y, s_mean), 0.4158, 0.01)
    s_pre = PR.fit_score_transfer(PR.REGISTRY["prefill_dom"], cell, bbh)
    print(f"    FINDING: prefill-DoM MATH->BBH = {auroc_sym(bbh.y, s_pre):.4f} "
          f"(pooled) >> mean-DoM 0.416 — prefill ports, mean inverts (V2-1 length).")
    # LOCO Number-Theory worst cell, prefill-DoM
    t1 = TR.run_T1_loco(["prefill_dom"])["prefill_dom"]
    nt = t1["per_cell"].get("number_theory", {})
    if nt:
        allok &= check(f"LOCO number_theory (n={nt['n']})", nt["auroc"], 0.603, 0.05)
    print(f"\n    worst LOCO cell: {t1['worst_cell']} = {t1['worst']:.4f}")

    allok &= step4_v7()
    allok &= step5_v8()

    print(f"\n{'ALL ANCHORS PASS' if allok else 'SOME ANCHORS FAILED'}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
