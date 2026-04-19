# 2026-04-19: Full Documentation Update

## What changed

All project documentation was rewritten to reflect corrected, deconfounded results from Phase 6.5.

### Files modified
- **README.md** — Complete rewrite with corrected numbers, honest cross-benchmark assessment, updated feature descriptions (44 tiers, not 7)
- **docs/index.html** — Complete rewrite of GitHub Pages site. New sections: feature tiers, tau sweep chart, deconfounded cross-benchmark table, transparency correction banner
- **CLAUDE.md** — New file for future session context

### Key corrections
| What | Old (wrong) | New (corrected) |
|------|------------|-----------------|
| MATH-500 AUROC | 0.948 | 0.796 |
| Baseline accuracy | 57/500 (11.4%) | 104/500 (20.8%) |
| Features described | 7 | 44 (A+B+C tiers) |
| Cross-benchmark claim | "Works across benchmarks" | "Works on MATH, fails on GSM8K" |
| GSM8K AUROC | 0.676 / 0.731 | 0.615 (baseline wins at 0.741) |
| 7B MATH AUROC | 0.682 | 0.739 (genuine, +0.102) |

### Source data
All numbers sourced from:
- `pathway6_rebuild/phase1_prompt_model/holdout_metrics_v2.json`
- `pathway6_rebuild/phase1_prompt_model/experiment9_v2.json`
- `pathway6_rebuild/phase2_completion/experiment1_v2.json`
- `pathway6_rebuild/phase6_5/FINAL_SUMMARY.md`
- `pathway6_rebuild/phase6_5/gsm8k/summary.json`
- `pathway6_rebuild/phase6_5/math7b/summary.json`
