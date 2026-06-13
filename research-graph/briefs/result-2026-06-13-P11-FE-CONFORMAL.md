# Result brief — Conformal sweep (link-forge directed): adaptive / online / group-conditional CP vs the cross-domain wall

A CPU-only, zero-GPU conformal-prediction sweep (C1/C2/C3) on already-cached
scalars (`results/v7_rescore_qwen1.5b_{math,bbh}.npz`: generation length, mean
logprob, true labels). Directed by a link-forge research-DB sweep for the one
**open, novel, unrefuted** conformal target: confgate's documented cross-domain
certificate refusal. v6 Phase 4 (P11-FE-EDGEGEN) declared a valid MATH→BBH
selective-risk certificate infeasible, but only ever tested **static**
split-conformal LTT with light-head k-recalibration. The entire distribution-shift
CP toolkit (weighted / non-exchangeable / online-adaptive / group-conditional) was
never tried against it. This sweep tries all of it.

**Honest framing up front:** conformal is a **guarantees / deployment-quality**
lever, NOT an accuracy lever (v8's axis is swap base/target). It does not raise
matched-cost MATH accuracy. Worth running because it closes confgate's single
documented limitation either way — and the answer is decisive.

## Headline

**The cross-domain certificate wall is FUNDAMENTAL — the model's task accuracy,
not the calibration method.** No adaptive, weighted, online, or group-conditional
conformal method restores a valid MATH→BBH selective-risk certificate where static
CP gives validity 0.0. The binding constraint is BBH base accuracy 0.241:
certifying an ≥80%-accurate answered set (ε=0.2) requires isolating an
ultra-confident sliver the free gate cannot separate (Barber non-exchangeable TV
coverage-gap 0.470 > ε ⇒ a distribution-free cross-domain cert is *formally
impossible* from MATH alone). **confgate 0.1.1 ships NO cross-domain feature**
(no winner); `certify_cross_domain()` stays refused, now hardened with positive
evidence that the whole shift-CP toolkit cannot save it. The one genuinely positive
deliverable is **in-domain group-conditional (Mondrian) certificates** for
high-accuracy categories — valid but low-coverage.

## What was run

Free gate = the pinned length+logprob logistic (`confgate.FreeGate`, recipe
`StandardScaler→LogisticRegression(max_iter=2000, C=1.0)`), fit on MATH; LTT parity
with phase4 via `exp_1b3_conformal.choose_tau/cp_upper`. δ=0.1, R=200, true labels
for all adjudication.

### C1 — adaptive/weighted CP vs the cross-domain wall (`c1_cross_domain_adaptive_cp.py`)
Five arms, MATH=calib / BBH=deploy, validity = frac of R calib draws with deployed
answered-error ≤ ε; "win" = validity ≥ 0.9 with coverage > 0 at ε=0.2.

| arm | validity @ε=0.2 | note |
|---|---|---|
| static (baseline) | **0.0** | wall reproduced (covs .25/.57/.67 at ε=.1/.2/.3) |
| weighted (Tibshirani covariate-shift) | **0.0** | domain-clf AUC 0.773, Kish eff n 183/500 — not pure covariate shift |
| online ACI (Gibbs&Candès/Zaffran) | **0.0** | coverage collapses (.01/.04/.13), long-run err 0.385 |
| Mondrian (per-BBH-task, k labels) | **0.0** @ε≤.2 | only 0.64 @ε=.3 — still < 0.9 target |
| Barber non-exchangeable | — | TV coverage-gap **0.470 > ε** ⇒ formally impossible |

**Mechanism:** BBH base acc 0.241; marginal feasibility needs ε≥0.76. The wall is
accuracy, not calibration. `cross_domain_cert_feasible = False`.

### C2 — online Conformal Risk Control under drift (`c2_online_crc_drift.py`)
Frozen τ vs online ACI/CRC τ (score-space, warm-started identical), streamed over a
**drift** stream (MATH block→BBH block) and a **shuffled stationary control**;
headline = post-drift trailing-window violation rate + fail-open/fail-safe.

- **No ε restores validity under drift** (cliff blocks it, consistent with C1).
- Frozen **FAILS OPEN** at ε∈{.2,.3,.4}: keeps answering BBH (cov .56/.67/.75) at
  err .61/.67/.70 — silently over budget (violation rate ~0.99).
- Online **FAILS SAFE** at ε∈{.2,.3,.4}: clamps coverage (.08/.18/.40) at no-worse
  error — it detects the drift and abstains.
- **Decisive control:** the frozen violation is **NOT drift-specific** — it is
  equally high in the shuffled stationary stream (0.99 ≈ 0.99). So the failure is
  *distributional hardness*, not non-stationarity; online "adaptation" is not
  tracking drift, it is just continually re-clamping coverage on hard BBH points.
  Online's only value here is fail-safe abstention.

### C3 — group-conditional (Mondrian) certificates (`c3_group_conditional_cp.py`)
Marginal vs per-group conditional CP, in-domain, honest scores (MATH 5-fold OOF;
BBH cross-domain gate). "Delivers" = ≥1 group reaches conditional validity ≥0.9
with cov>0 at some k AND marginal leaves a real gap.

- **MATH (base acc 0.486) DELIVERS** at ε≥0.2: marginal CP hides a cross-category
  coverage gap (0.38 @ε=.2, 0.46 @ε=.3) with 2–3/7 categories silently over budget;
  per-group Mondrian restores **valid** certs for the high-accuracy categories
  (algebra base .67, prealgebra base .62) — validity 1.0 — but at **low coverage**
  (algebra 0.5%, prealgebra 5.4% @ε=.2; prealgebra reaches a modest 11% @ε=.3) and
  a per-group label budget of k=16–32 true labels.
- **BBH (base acc 0.241) does NOT deliver** at ε≤0.2 — the accuracy cliff is
  per-group too (only web_of_lies @ε=.3 squeaks valid at 0.5% coverage).

## Net

The conformal whitespace is closed. **Cross-domain certificates are infeasible by
the model's accuracy, not the calibration method** — confirmed by exhausting the
shift-CP toolkit (C1), and online conformal converts frozen's silent fail-open into
a valid fail-safe abstention but cannot manufacture validity past the cliff, and the
underlying failure isn't even drift (C2). The deployable positive: **in-domain
group-conditional certificates** for high-accuracy categories, valid at a modest
per-group label budget but low coverage (C3). Deployment recipe unchanged and
reinforced: certify in-domain on ≥32 true labels, and the in-domain achievable
accuracy must exceed 1−ε; there is no cross-domain shortcut. confgate 0.1.1 ships no
cross-domain certificate.

## FE

```yaml
fe_id: P11-FE-CONFORMAL
status: COMPLETED
outcome: "Conformal sweep (C1/C2/C3, CPU-only on cached scalars, true-label adjudication) closes the cross-domain-certificate whitespace. C1: NO adaptive/weighted/online/group-conditional CP restores a valid MATH->BBH selective-risk cert where static gives validity 0.0 @eps=0.2 — weighted (Tibshirani) 0.0 (domain-clf AUC 0.773, Kish n 183/500), online ACI 0.0 (coverage collapses .04, long-run err 0.385), Mondrian 0.0 @eps<=.2 (0.64 @eps=.3 < 0.9 target), and Barber non-exchangeable TV coverage-gap 0.470 > eps => a distribution-free cross-domain cert is FORMALLY IMPOSSIBLE from MATH alone. Mechanism: BBH base acc 0.241; marginal feasibility needs eps>=0.76 — the wall is task accuracy, not calibration. C2: online conformal under a MATH->BBH drift stream gives NO eps that restores validity; frozen tau FAILS OPEN (keeps answering at err .61/.67/.70, cov .56/.67/.75, violation ~0.99) while online tau FAILS SAFE (clamps coverage .08/.18/.40 at no-worse error) — but the frozen violation is NOT drift-specific (equal 0.99 in a shuffled stationary control), so the failure is distributional hardness, not non-stationarity; online's only value is fail-safe abstention. C3: in-domain group-conditional (Mondrian) certs DELIVER for high-accuracy MATH categories (algebra base .67 / prealgebra base .62 reach validity 1.0 at k=16-32 labels) where marginal CP hides a 0.38-0.46 cross-category coverage gap with 2-3/7 categories over budget — but at LOW coverage (prealgebra best, 11% @eps=.3); BBH does not deliver (accuracy cliff is per-group too). Verdict: cross-domain certificates infeasible by model accuracy not calibration method; confgate 0.1.1 ships NO cross-domain feature, certify_cross_domain() stays refused, hardened with positive evidence the whole shift-CP toolkit cannot save it. Deployment recipe reinforced: certify in-domain on >=32 true labels and in-domain achievable accuracy must exceed 1-eps."
result_json: pathway11_h100/generalization_edge/results/c1_cross_domain_adaptive_cp.json
result_json_all:
  - pathway11_h100/generalization_edge/results/c1_cross_domain_adaptive_cp.json
  - pathway11_h100/generalization_edge/results/c2_online_crc_drift.json
  - pathway11_h100/generalization_edge/results/c3_group_conditional_cp.json
triggered_by: ["2402.10978"]
refutes_premise: ["conformal-cross-domain"]
would_update: ["F-8"]
```

## EXPERIMENT_LOG entry

## EXP-86: Conformal sweep (C1/C2/C3) — cross-domain cert wall is fundamental; group-conditional in-domain delivers

**Date:** 2026-06-13. **Compute:** local (CPU only, zero GPU), Qwen-2.5-1.5B, MATH-500 + BBH-750, cached scalars, no generation. **Harness:** `pathway11_h100/generalization_edge/c{1,2,3}_*.py` (13/13 conformal anchors reproduce via `verify_anchors.py --conformal-only`).

**C1 (adaptive/weighted CP):** no arm rescues the MATH→BBH wall — static/weighted/online-ACI/Mondrian all validity 0.0 @ε=0.2; Barber TV coverage-gap 0.470 > ε ⇒ distribution-free cross-domain cert formally impossible. Mechanism: BBH base acc 0.241 (marginal feasibility needs ε≥0.76). **Strengthens (re-refutes) the `conformal-cross-domain` premise** beyond k-recalibration to the full shift-CP toolkit.

**C2 (online CRC under drift):** no ε restores validity; frozen FAILS OPEN, online FAILS SAFE, but the frozen violation is NOT drift-specific (equal in shuffled control) → distributional hardness, not non-stationarity. Online value = fail-safe abstention only.

**C3 (group-conditional):** in-domain Mondrian certs DELIVER for high-accuracy MATH categories (algebra/prealgebra validity 1.0 at k=16–32) where marginal hides a 0.38–0.46 coverage gap; valid but low coverage (≤11%); BBH does not deliver. Claims +15 (`edge-c1/c2/c3-*`), internal invariant 264→279 PASS.

## STATE.md last-experiment update

**EXP-86 (Conformal sweep C1/C2/C3) — the cross-domain certificate wall is FUNDAMENTAL (accuracy-bound, not calibration), and the shift-CP toolkit cannot rescue it.** Local, CPU-only, Qwen-1.5B MATH+BBH, cached scalars. C1: static/weighted(Tibshirani)/online-ACI/Mondrian all validity 0.0 @ε=0.2; Barber TV gap 0.470>ε ⇒ formally impossible; mechanism BBH base acc 0.241. C2: online conformal = fail-safe abstention only (frozen fails open, online fails safe) and the failure is distributional hardness, NOT drift (shuffled control matches). C3: in-domain group-conditional certs DELIVER for high-acc MATH categories (algebra/prealgebra validity 1.0, k=16–32) at low coverage (≤11%); BBH cliff is per-group too. confgate 0.1.1 ships NO cross-domain feature — `certify_cross_domain()` stays refused, hardened. Strengthens the `conformal-cross-domain` premise (refuted). Claims +15 (edge-c1/c2/c3-*), 279/279 internal PASS; anchors +13 (`verify_anchors.py --conformal-only`).
