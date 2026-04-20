# 2026-04-19: Pathway 7 — Non-Euclidean PH Pipeline Implementation

## What was built

16 new files in `pathway7/` (2565 lines) implementing the non-Euclidean PH pipeline upgrade based on a research playbook the user provided.

### Core modules
- **`distance_metrics.py`** (426 lines) — 4 non-Euclidean PH backends:
  - `effective_resistance_ph()`: kNN graph → normalized Laplacian → eigenvector embedding → ripser (Damrich et al. NeurIPS 2024)
  - `cosine_ph()`: pdist(cosine) → ripser(distance_matrix=True)
  - `diffusion_ph()`: diffusion-distance embedding (Damrich runner-up)
  - `dtm_ph()`: GUDHI DTM-Rips (optional, outlier-robust)
  - `compute_all_noneuclid_features()`: convenience function returning 18 features per point cloud
  - All return `dict[int, np.ndarray]` matching existing `compute_ph()` interface

- **`feature_extractor_v2.py`** (174 lines) — Composes frozen 44 ABC-tier features + 18 new non-Euclidean features. Two modes: pre-computed base features or from-scratch.

- **`zigzag_features.py`** (235 lines) — Gardinazzi et al. ICML 2025 recipe: per-layer kNN(k=4), intersection inclusions, Dionysus2 zigzag persistence. Requires Dionysus2 (source build in Docker).

- **`extract_all_layers.py`** (156 lines) — GPU script for all-layer hidden-state extraction (needed for zigzag). Saves (29, n_sub, hidden_dim) per problem.

### Phase 7.1 experiment
- **`phase71_noneuclid_math500.py`** (354 lines) — The key experiment. Reuses existing trajectories from `data/experiment1_v2/trajectories.npz`. Computes eff-res + cosine features, fits LR with multiple C values, runs DeLong comparison vs 0.796 baseline. **CPU only, ~1 hour, $0.**

### Validation suite
- `validation/synthetic_shapes.py` — PH correctness on circle/torus/sphere via all 3 backends. **All PASS.**
- `validation/metric_divergence.py` — Cosine vs Euclidean divergence. **Confirmed:** cosine H1 2.7× Euclidean in 50D; completely different scales on real 1536-D MATH trajectories.
- `validation/delong_comparison.py` — DeLong AUROC comparison (MLstatkit with bootstrap fallback).

### Benchmark pipelines
- `benchmarks/humaneval_pipeline.py` — 164 HumanEval problems, evalplus evaluation
- `benchmarks/bbh_pipeline.py` — 3 BBH subsets × 250 problems, 3-shot CoT

### Infrastructure
- **`Dockerfile`** — Pinned RunPod image: torch 2.4, ripser 0.6.14, gudhi 3.12.0, dionysus 2.0.10 (source build), persim, giotto-tda, tadasets, MLstatkit, evalplus
- **`requirements_pathway7.txt`** — All pinned versions
- **`runpod_pathway7.sh`** — 7-step orchestrator with `--from N` resume, go/no-go gate after Phase 7.1
- **`PLAYBOOK.md`** — Full research playbook (reference document, ~600 lines)

### 18 new features
8 from eff-res PH (H0+H1 × total_persistence, max_lifetime, entropy, n_features), 8 from cosine PH (same stats), 2 from graph spectral (spectral_gap, algebraic_connectivity).

## Key design decisions
1. **k=30 for eff-res** (not 100 from paper) — existing subsample=100 makes k=100 degenerate
2. **PCA-reduced space** for non-Euclidean PH — matching existing Euclidean pipeline for fair comparison
3. **Train-only PCA** — reuses `StratifiedShuffleSplit(random_state=9999)` from pathway6_rebuild
4. **Compose, don't modify** — winning_features.py is frozen; new features concatenated horizontally

## Updated files
- `CLAUDE.md` — Added pathway7 to code structure, running instructions
- Memory: `topo-confidence-experiments.md` — Added Pathway 7 phase schedule
- Memory: `pathway7-playbook.md` — New reference pointer
- Memory: `MEMORY.md` — Updated topo-confidence entry

## Phase 7.1 Results (2026-04-19, session 2)

**Go/no-go: NO-GO.** Non-Euclidean features do not beat 0.796 on MATH-500.

| Model | Holdout AUROC | CV AUROC | Features |
|-------|--------------|----------|----------|
| ABC-44 (baseline) | **0.796** | 0.695 | 44 |
| Combined (C=1.0) | 0.774 | 0.691 | 62 |
| Combined (C=0.1) | 0.757 | 0.711 | 62 |
| NonEuclid-only | 0.716 | 0.737 | 18 |
| EffRes-only | 0.718 | 0.741 | 8 |
| Cosine-only | 0.716 | **0.746** | 8 |

DeLong: combined vs baseline = -0.022, p=0.57 (ns).

### Key observation
Non-Euclidean features achieve **higher CV AUROC** (0.746 cosine, 0.741 eff-res) than ABC baseline CV (0.695), indicating more stable cross-fold performance. But ABC captures holdout-specific structure that yields higher holdout AUROC.

### Bug fixed in this session
`phase71_noneuclid_math500.py` had a label version mismatch: it used v2 labels (104 correct) to reconstruct the SSS split, but features were saved with old labels (57 correct). Different stratification → different split → feature-label misalignment → ABC baseline showed 0.529 (chance). Fixed by using old labels for SSS reconstruction and new labels for actual training (matching pathway6_rebuild pattern).

## What's next
1. **HumanEval + BBH benchmarks** on RunPod GPU (~$6 total) — per plan, proceed to new benchmarks for breadth even though non-Euclidean didn't beat MATH-500
2. **Zigzag persistence** — highest risk/cost, defer unless benchmarks show promise
3. Consider whether higher CV AUROC of non-Euclidean features suggests value in ensemble approaches

## Not committed
This session's work has NOT been committed to git. The user should review and commit when ready.
