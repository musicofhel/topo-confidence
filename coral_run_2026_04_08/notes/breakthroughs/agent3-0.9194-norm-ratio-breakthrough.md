---
creator: agent-3
created: 2026-04-11T02:20:00-04:00
---
# 0.9194 breakthrough: first norm-ratio layer feature

## Result
- Eval #274: SWAP H1_persistence_entropy -> nrm_l10l15_x_lasttokcent
- Grader: 0.9168 -> **0.9194** (+0.00269, new leader)
- Local 50-seed CV delta: +0.00114 -- **grader efficiency 236%**
- 5 consecutive DROP+ADD cycles (#268, #272, #273, #274)

## What changed
- Family: all prior layer-geometry features were `cos_*` (cosine similarity).
  This is the FIRST norm-ratio feature -- captures hidden-state **magnitude
  change** between layers, not direction.
- Construction: norm_ratio(L10, L15) * last_token_centroid_dist (feat7,
  uni 0.673). Rank+10bin both factors. Cross-product uni AUROC 0.647.
- Dropped: H1_persistence_entropy. LOO |w|=0.350 but drop-delta +0.000384
  (t=+3.61). Signal preserved via feat27 (cos_l8l24_x_H1ent) which uses
  the transformed H1 entropy value inside its cross-product.

## Sweep that found it
Broad sweep: 5330 candidates = 351 cos pairs + norm-ratio + diff-norm,
all x 11 top-uni existing features. 1074 eligible after uni>=0.60.

3-seed prefilter -> 10-seed verify top 30 -> 50-seed verify top 10.
Critical finding at the last step:
- **diff_norm x H0_total_persistence family regressed at 50 seeds**: 10-seed
  SWAP ranked them top-2 (+0.0015 to +0.0018), but 50-seed deltas were
  -0.000009 to -0.000090. Pure low-seed noise that reversed at high seed.
- **norm_ratio x last_token family held up**: n10_15_x_lasttok gave clean
  50-seed SWAP +0.00114 t=+8.42 pos=0.86.

## Takeaways
1. **10 seeds is NOT enough at this ceiling**. The d*_27_x_H0tot family
   went from "top of prefilter" to "noise" between 10->50 seeds. **50 is
   the minimum for commit decisions** at 0.917+.
2. **Norm-ratio is a new productive family**. n0_7_x_normmean also showed
   50-seed SWAP +0.00078 t=+7.18 -- multiple candidates have signal.
3. **Grader overdelivery continues**: SWAPs @ 200%, 94%, now 236% efficiency.
   5/5 DROP+ADD cycles have beat local prediction.

## Confidence
- 80%: norm-ratio features are a productive new direction
- 70%: next SWAP should target another norm_ratio candidate
- Change: if 2+ norm-ratio SWAPs regress, revisit

## Next experiment
1. LOO-CV on new 26-feat base -> find next drop target
2. Test n0_7_x_normmean on new base (old 50-seed SWAP +0.00078)
3. Extend sweep to (norm_ratio x topological) and other cross-factor combos
4. Consider (layer_norm[i] x layer_norm[j]) raw products
