# Changelog

A short log of conceptual shifts. For per-experiment provenance see
[PROJECT_RECORD.md §1a chronology](PROJECT_RECORD.md) and
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

## Unreleased

### Added
- Auto-generated [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md) priority queue rendered from `:FutureExperiment` Neo4j nodes.
- Neo4j research graph (`research-graph/`) linking F-1…F-14 to ~40 papers via typed edges (CORROBORATED_BY, CONTRADICTED_BY, EXTENDED_BY, METHOD_DIFFERS, EXPLAINS). Separate from link-forge; see [RESEARCH_GRAPH.md](RESEARCH_GRAPH.md).
- `.env.example` documenting NEO4J_*, LINK_FORGE_*, MODEL_NAME, MATH_500_PATH, CACHE_DIR, RUNPOD_API_KEY.
- This changelog.

## Pathway 11 — H100 re-extraction (April 2026)

### Changed (the big one)
- Re-extracted Qwen-2.5-1.5B and Qwen-2.5-7B activations on MATH-500 at `max_new_tokens=1024`. **All headline numbers shifted.** New canonical figures: 1.5B accuracy 48.6% (was 20.8%), prefill L19 DoM AUROC 0.7731 (was 0.796 on the ABC-44 pipeline), 7B accuracy 73.2%. The truncation confound from Phase 6.5 propagated through every cross-scale and CoE comparison; those are now graveyard entries.
- Selective-prediction headline established: 71.6% accuracy at 50% coverage on answered fraction, +22 pp over unconditional, gating by prefill DoM.

### Added
- Prefill / final-token DoM orthogonality (cos = 0.046) — F-3.
- Dimensional breathing universality across Qwen-1.5B/7B, Phi-3-mini, Llama-3.2-1B (F-1, STRONG).
- Gibberish / random-token control: PR is flat for non-content tokens, breathing is content-dependent (F-5).
- No-CoT control: inconclusive as framed (median 2 tokens with "answer only" prompt) — see [topo-confidence-nocot-control.md](../.claude/projects/-home-musicofhel/memory/topo-confidence-nocot-control.md).

## Pathway 9 — CoE pivot + PH audit (April 2026)

### Changed
- Persistent homology on trained residual streams shown indistinguishable from rank-matched Gaussian null (0.690 vs 0.693). The original "topology predicts correctness" framing is overturned — F-10 (STRONG). PH measures covariance structure, not topology.
- CoE-60 trajectory features established at AUROC 0.811 with cross-domain transfer at 0.716 (MATH↔BBH). Later (Pathway 11) shown to be redundant with single-layer L19 DoM at 1024-tok labels — F-9.

### Added
- Length-deconfounding controls (`pathway9/exp1_length_deconfound.py`).
- Token-shuffle and count-control nulls (`exp3b`, `exp3c`).
- XGBoost vs logistic-regression comparison (`exp4_xgboost_oinfo*`).
- Cross-domain MATH↔BBH transfer (`exp5_cross_domain.py`).

## Pathway 8 — Layer-wise PH (April 2026)

### Added
- Per-layer PH analysis across all 28 transformer layers; modest lift from layer-aggregation, fully absorbed by Gaussian null in Pathway 9.

## Pathway 7 — Non-Euclidean PH (April 2026)

### Decided
- NO-GO: hyperbolic / spherical Vietoris-Rips features at 0.774, below the 0.796 Euclidean baseline. Logged for completeness.

## Pathway 6 rebuild — Corrected labels (April 2026)

### Fixed
- Phase 6.5 deconfounding identified `max_new_tokens=256` as the source of the 20.8% baseline. Re-running at 1024 tokens later (Pathway 11) bumped 1.5B to 48.6% and exposed the truncation confound in every prior cross-scale claim.

## Pathway 5 — Cross-benchmark / cross-model (April 2026)

### Added
- Transfer experiments across MATH-500 / GSM8K / BBH and Qwen-1.5B / 7B. "7B is a stronger verifier" originally concluded at this stage; superseded by Pathway 11 1024-tok results showing no asymmetry.

## Pathway 4 — Topo-guided selection (April 2026)

### Added
- Track A: spherical-steering selection achieved +11 net gain on holdout (`PUBLICATION_READY` at the time, before truncation correction).
- Track B (learned steering): null result.

## Pathway 3 — Complexity expansion (April 2026)

### Decided
- Adding complexity-tier features did not improve over Tier A. PIVOT decision logged.

## Pathway 2 — Spherical steering (April 2026)

### Added
- Track A spherical steering implementation; Phase 0–2 results.
- Pathway 10 v1 later showed the fitted steering vector was geometrically unrelated to the current DoM direction (direction rotation), refuting fixed-vector steering.

## Pathway 1 — CORAL validation (April 2026)

### Added
- Initial validation of 44-feature ABC pipeline on Qwen-2.5-1.5B-Instruct. Holdout AUROC 0.948 (against the now-superseded 256-tok labels). GO decision logged.

## v0.2.0 — Package release (April 2026)

### Added
- `topo_confidence/` pip-installable package: `TopoConfidence`, `CombinedConfidence`, CLI, bridge feature.
- Tests, README rewrite, HuggingFace Spaces Gradio demo, GitHub Pages site.
- LangChain evaluator + bridge health monitor.

## v0.1.0 — Initial commit (April 2026)

### Added
- Topological uncertainty estimation prototype (CORAL run, AUROC 0.699 with 7 features).
- 13 topological features and null hypothesis testing.
