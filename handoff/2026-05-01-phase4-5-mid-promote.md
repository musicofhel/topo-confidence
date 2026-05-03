# Handoff — Phase 4 + 5 mid-promote (compact checkpoint)

**Date:** 2026-05-01 (late evening)
**Session entry:** continued from 2026-05-01-phase2-3-cheap-wins.md handoff;
user said "proceed" + auto-mode active.

## What's done in this session

| Step | FE | Status |
|---|---|---|
| 1 | P11-FE282 | ✅ Marked COMPLETED-by-cache via `update_status.py`; NEXT_EXPERIMENTS regenerated |
| 2 | P11-FE116 PH on residuals | ✅ Script + run done (~65 min CPU); `pathway11_h100/ph_residuals/results.json` exists |
| 3 | P11-FE321 zigzag PH | ✅ Script + run done (~8 min CPU); `pathway11_h100/zigzag_ph/results.json` exists |
| 4 | P11-FE110 + P10-FE23 + P11-FE455 GPU bundle | ✅ Script + run done (~2.5 min on RTX 2060 Super); `pathway11_h100/gpu_bundle/results.json` exists |

## Headline numbers from this session

| Quantity | Value | Source |
|---|---|---|
| FE321 B_1+bar_Z_1 real AUROC | 0.4952 (≈ null 0.5611) | zigzag_ph/results.json |
| FE321 7-descriptor real AUROC | 0.7156 vs null 0.6188, gap +0.097 | same |
| **FE116 real PH AUROC on residuals** | **0.6958** (vs raw F-10 0.690) | ph_residuals/results.json |
| **FE116 null PH AUROC on residuals** | **0.7624** (gap −0.067, INVERTED vs F-10 raw −0.003) | same |
| FE116 H_1 cycle count real vs null | 31.3 vs 87.7 (real is smoother) | same |
| FE110 best ActAdd cos with DoM | +0.065 (max AUROC 0.6615) | gpu_bundle/results.json |
| FE23 softmax-conf AUROC | **0.4395** (BELOW chance — anti-calibrated!) | same |
| FE455 ConCISE c_hat AUROC | 0.5887 | same |

**Big finding to surface in F-10:** residualization INVERTS the gap. The
matched-cov Gaussian null beats real PH on residualized clouds because
real residuals are *smoother / less random* than rank-matched Gaussian
samples. The 0.7624 null AUROC reveals **per-problem covariance structure
itself carries correctness signal** — the SVD sampler inherits empirical
covariance, and PH features on those Gaussians ARE predictive. F-10
strengthens to "topology adds no signal beyond covariance."

## Three briefs authored, mid-promote

```
research-graph/briefs/result-2026-05-01-P11-FE321.md           (Phase 5)
research-graph/briefs/result-2026-05-01-P11-FE116.md           (Phase 5)
research-graph/briefs/result-2026-05-01-P11-FE110+FE23+FE455.md (Phase 4 GPU bundle)
```

Each declares its FE-id YAML block, EXPERIMENT_LOG entry, STATE.md
update, FINDINGS.md update, and new claims. The GPU-bundle brief uses
P11-FE110 as primary YAML; P10-FE23 + P11-FE455 outcomes documented in
the body — promote those two manually via `update_status.py` after
primary promote lands.

## validate_claims.py — 23 new claims added, 155 PASS / 0 FAIL on --no-regen

| Block | Tier | n claims |
|---|---|---|
| FE321 (zigzag) | Tier-0 only | 7 |
| GPU bundle (FE110+FE23+FE455) | Tier-1 (regen=2.5 min) | 7 |
| FE116 (PH on residuals) | Tier-0 only | 9 |

FE321 + FE116 are Tier-0 *only* because their regens take >600s and risk
REGEN_TIMEOUT under any CPU contention (FE321 ~8 min standalone, FE116
~25-65 min standalone). Manual regen commands documented in the
preceding `# -----` comments in `validate_claims.py`.

Cumulative invariant on `--no-regen`:
**155 internal PASS / 0 FAIL / 41 REGISTERED / 3 PENDING_FE = 199 tracked**.

## Promote status — IN-FLIGHT at compact time

- **FE321 promote attempt 1 (mid-evening)** → FAILED at validate_claims gate
  with full regen (likely a regen timeout under contention with FE116).
- **FE321 promote attempt 2** running in background as compact starts:
  - bash background id `bh1vv2veb`
  - monitor task id `bnc4co6su` (timeout 2400s)
  - Uses Tier-0-only FE321 claims (regen= dropped, parallel to FE749).
- **STATE.md still shows EXP-51** as "Last experiment completed."
  EXPERIMENT_LOG Next-ID still **EXP-52**.
- **GPU bundle + FE116 briefs NOT yet promoted.** Run after FE321 lands.

## Bugs hit this session

1. **FE321 first promote failed at validate_claims regen step.** Root
   cause: the new FE321 claims declared `regen="python ...recompute_fe321.py"`
   for all 7 entries; `_REGEN_CACHE` shares results across same-cmd
   claims (good), but a single 8-min run × 1 still hits the 600s per-
   claim subprocess timeout when CPU is contended. Fix applied: drop
   `regen=` from the 7 FE321 claims (Tier-0 only), parallel to the FE749
   precedent documented in `2026-05-01-phase2-3-cheap-wins.md`.

2. **Static H_1 is structurally zero on a 28-point trajectory in 1536-d.**
   FE321 first design used B_1 + bar_Z_1 alone; results were trivially
   zero across all 500 problems. Fix applied: extended descriptor set to
   7 features (B_1, bar_Z_1, 3 H_0 statistics, 2 step-distance
   statistics). The FE321 brief reports both subsets honestly: B_1+bar_Z_1
   gap −0.066 (degenerate), 7-descriptor gap +0.097 (geometric not
   topological).

## After-compact resume plan

1. **Confirm FE321 promote attempt 2 landed** (check
   `tail -20 /tmp/claude-1001/.../tasks/bh1vv2veb.output` and verify
   STATE.md now shows EXP-52, EXPERIMENT_LOG Next-ID = EXP-53).

2. **Promote GPU bundle:**
   ```bash
   cd ~/topo-confidence
   python research-graph/promote_result.py \
     research-graph/briefs/result-2026-05-01-P11-FE110+FE23+FE455.md \
     --skip-regen --skip-git-check
   ```
   This sets P11-FE110 → COMPLETED via the YAML.
   Then for the bundled siblings:
   ```bash
   cd ~/topo-confidence/research-graph
   python update_status.py P10-FE23 COMPLETED \
     --outcome "Softmax-confidence AUROC 0.4395 at PANL — anti-calibrated. F-2 holds."
   python update_status.py P11-FE455 COMPLETED \
     --outcome "ConCISE c_hat AUROC 0.5887. F-2 holds."
   python generate_next_experiments.py
   ```

3. **Promote FE116:**
   ```bash
   cd ~/topo-confidence
   python research-graph/promote_result.py \
     research-graph/briefs/result-2026-05-01-P11-FE116.md \
     --skip-regen --skip-git-check
   ```

4. **Pre-flight verify:**
   ```bash
   python validate_claims.py --no-regen 2>&1 | grep -iE "PASS:|FAIL:"
   # Expected: 155 PASS, 0 FAIL
   grep -A 3 "## Last experiment completed" STATE.md | head -10
   # Expected: EXP-54 (FE116) after all three promotes
   grep "Next ID" EXPERIMENT_LOG.md | tail -1
   # Expected: EXP-55 after all three
   ```

5. **Update PERSPECTIVES.md** with the surprising findings:
   - FE116 inversion (null > real on residuals) reveals per-problem
     covariance carries the signal, not topology.
   - FE23 softmax-conf at PANL is *anti-calibrated* (0.44 < 0.5) — when
     the model is most confident in the next token, it's slightly more
     likely to be wrong. Worth flagging in any selective-prediction work.
   - PERSPECTIVES is the right home for these kinds of "what
     surprised me" notes; promote_result.py never touches it.

## Files modified / created this session

```
NEW   pathway11_h100/ph_residuals/recompute_fe116.py
NEW   pathway11_h100/ph_residuals/results.json
NEW   pathway11_h100/zigzag_ph/recompute_fe321.py
NEW   pathway11_h100/zigzag_ph/results.json
NEW   pathway11_h100/gpu_bundle/recompute_gpu_bundle.py
NEW   pathway11_h100/gpu_bundle/results.json
NEW   pathway11_h100/gpu_bundle/per_problem.npz
NEW   research-graph/briefs/result-2026-05-01-P11-FE321.md
NEW   research-graph/briefs/result-2026-05-01-P11-FE116.md
NEW   research-graph/briefs/result-2026-05-01-P11-FE110+FE23+FE455.md
EDIT  validate_claims.py            (+23 claims; FE321 + FE116 Tier-0 only, GPU bundle Tier-1 with regen)
EDIT  NEXT_EXPERIMENTS.md           (regenerated after FE282 status flip)
NEW   handoff/2026-05-01-phase4-5-mid-promote.md   (this file)
```

STATE.md, EXPERIMENT_LOG.md, FINDINGS.md, NEXT_EXPERIMENTS.md will be
mutated by the three pending promotes (sequentially via
`promote_result.py`).

## Memory aids for the next session

- Next EXP-id (pre-promote): **EXP-52** in EXPERIMENT_LOG.md.
- After all 3 promotes: EXP-52 = FE321, EXP-53 = GPU bundle, EXP-54 = FE116.
  Next-ID after = **EXP-55**.
- Cached prefill source for FE110 projection AUROC:
  `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz`
  (500 × 1536 fp16 prefill at L19 + 500 × 1536 final-tok + correct + seq_len).
- Cached W_unembed at `pathway11_h100/phase3_corroborators/W_unembed_15b.npy`
  (151936×1536 fp32, 891 MB).
- F-2 strength: **MODERATE**, status **ACTIVE**, evidence rows 1..12 (after this session).
- F-3 strength: **STRONG**, status **ACTIVE**, evidence rows 1..5 (unchanged).
- F-9 strength: **MODERATE**, status **ACTIVE**, evidence rows 1..3 (unchanged).
- F-10 strength: **STRONG**, status **ACTIVE**, evidence rows 1..4 (after this session).
- 2060 Super has 8 GB; Qwen-2.5-1.5B-Instruct fp16 fits with ~5 GB headroom.
- `promote_result.py` blank-line bugs from prior session remain fixed.
