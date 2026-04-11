---
creator: agent-2
created: 2026-04-11T18:00:00
---
# Eval #176: DROP-ADD cycle #2 — yield decaying, +0.00067

**Score**: 0.95157 local = 0.9516 grader (12/12 consecutive grader-exact matches)
**Delta over #175 (0.9509)**: +0.00067
**Features**: 50 (drop cols 17, 29, 42; add cos(L1,L8), cos(L1,L16), accel(L6)×H1ent_exp)

## Concrete change history

| Eval | Δ       | Cumulative | Strategy                                  |
|------|---------|------------|-------------------------------------------|
| #172 | +0.00638 | 0.94666    | 5-feat forward: accel+cos stack on 41-base |
| #173 | +0.00253 | 0.94919    | 3-feat cosine triple (sub-additive 0.64)  |
| #174 | ≈saturation | — (scan) | dynamical + layer-PH scans: 1/429 positives |
| #175 | +0.00170 | 0.95089    | DROP-ADD #1: drop f29,f42 → add c(L7,L11)  |
| #176 | +0.00067 | 0.95157    | DROP-ADD #2: drop f17 → add c(L1,L8)+c(L1,L16)+a6×H1 |

## Yield is decaying 2.5x per DROP-ADD cycle

- #172 forward: +0.00638
- #173 forward: +0.00253 (-60%)
- #174 scan: 0 (saturation)
- #175 DROP-ADD: +0.00170 (-33% from #173)
- #176 DROP-ADD: +0.00067 (-61% from #175)

Half-life is ~1 cycle. At this rate the next DROP-ADD (#177) would yield ~+0.00025-0.00040 — barely above grader noise. Time to pivot.

## Univariate AUROC of new adds (eval #176 feedback)

| Feature                    | Uni AUROC | Comment                   |
|----------------------------|-----------|---------------------------|
| accel_l6_x_h1ent_exp_raw   | 0.687     | Strong — 2nd-highest of new adds ever |
| cos_l1_l16                 | 0.536     | Weak per weak-feature-trap, but part of the stack |
| cos_l1_l8                  | 0.513     | Very weak — interacts super-additively with accel_l6 via LR |

The dominant signal came from the accel×PH cross (0.687 uni). The two L1 cosines are stacking essentially noise for the LR. Cosine family is fully saturated and all forward cosine adds now rely on interaction with stronger existing features.

## Key surprise: accel_l6 × H1ent_exp is the highest-solo new feature all session

The dynamical × topological cross-products are extremely potent. The top-uni features in the current 50-feat set include:
- c4_6_x_h0ent35: 0.730
- accel_l4_x_h0ent35_bin: 0.728
- H0_entropy_thresh35: 0.729
- H0_tight_fine_product: 0.725
- H0_entropy_thresh50: 0.722
- bc7_26_x_h0tightfine: 0.716
- c27_28_x_h0tot: 0.693
- H0_total_persistence: 0.692
- accel_l6_x_h1ent_exp_raw: 0.687 ← NEW, from eval #176
- H1_persistence_entropy: 0.684
- accel_l15_x_h1tot_raw: 0.652
- H1_total_persistence: 0.656
- H0_max_lifetime: 0.642
- bc25_27_x_h1ent: 0.636

**9 of the top-14 features are cross-products between a topological (PH) feature and a geometric/dynamical feature.** This is the unifying pattern: TOPOLOGY × GEOMETRY beats either alone. Pure cosines uni < 0.55 without this cross.

## Mechanism for why #176 was marginally positive

Dropping f17 (layer_angle_min_rankbin) was a RISKY drop — it feeds the cross at idx 19 (`layer_angle_min × H1_total_persistence`). The cross is retained (idx 19 is computed BEFORE drop). The drop removed only the solo copy, not the interaction. Effective net information loss: minimal. That's why drop f17 was a +0.00024 "free move."

The triple add was super-additive because accel_l6 × H1ent_exp is a genuinely new cross (0.687 uni), and the two early-L1 cosines interact weakly with it to boost classifier boundary in the "early-layer-alignment × H1-topology" subspace.

## What failed in the scan

- **2-drop scans on 48-base**: 0 super-additive pairs beyond tied at 0.95113 (same as solo drop). Unlike eval #175 where f29+f42 combined for +0.00083, no such pair exists in the 48-feat base. This confirms one-time-use of the super-additive drop mechanism.
- **3-drop scans**: all negative.
- **Pair-adds on reduced base**: c1_8+c1_16 delta_r=+0.00020 (eff 0.36 — heavily redundant since both are L1 early-layer cosines).
- **Quads**: c1_8+c1_16+a6H1+c5_28b dropped back to +0.00036 — the 4th feature actively hurts.

## Confidence assessment

- **High confidence** that cosine-family adds are saturated (3 cycles of scan → all marginal).
- **Moderate confidence** that DROP-ADD cycles are still available but at 2-3x smaller yield.
- **Low confidence** that forward selection from existing families will yield anything worth an eval slot.
- **Genuinely untested directions**:
  1. **Persistence landscapes** on the 29 layer points (NOT on per-token — already saturated there)
  2. **Wasserstein distance** between early-layer PH diagram and late-layer PH diagram (intra-problem inter-layer topology shift)
  3. **Bottleneck distance** between token-PH and layer-PH (cross-modal topological alignment)
  4. **Attention concentration** features — requires loading attention patterns from saved data (may not be available)
  5. **Hessian-ish 3rd derivatives** — jerk(L_k) × existing features (2nd-deriv accel already very potent)
  6. **Per-token curvature statistics along the layer axis** — currently unexplored

## Next experiment plan

Try a **2-level persistence landscape** feature on the 29 layer states + **jerk × H features cross**. Two concrete candidates:

1. `landscape_lvl1_integral = integral of the top persistence diagram's λ_1(t) function` on H0 of layer states
2. `jerk(L_k) × H_ent_exp` for k in {5, 6, 7, 10, 15, 20}

Expected: +0.00030-0.00080 if any of these open a new axis. If all fail → pivot to attention features or Wasserstein distances.

**Decision rule**: If next 2 evals yield < +0.0003 cumulative, stop pushing 0.95 and explore fundamentally different axes (Wasserstein, attention, bottleneck).
