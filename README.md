# topo-confidence

**Know when your LLM is wrong — from a single forward pass.**

topo-confidence uses persistent homology on hidden-state geometry to predict whether an LLM's output is correct. It extracts 44 topological features from the token-level point cloud at the final transformer layer and trains a logistic regression classifier to estimate P(correct).

## Headline result

On MATH-500 with Qwen2.5-1.5B-Instruct:

| Metric | Value |
|--------|-------|
| Topo AUROC (holdout) | **0.796** [0.671, 0.907] |
| Best baseline (vote_margin, 32-pass) | 0.767 [0.632, 0.878] |
| Gap over best baseline | **+0.057** |
| Greedy accuracy | 104/500 (20.8%) |
| Ungated majority vote | +12 net gain |
| Gated MV (tau=0.3) | +4 net gain, **0 R->W** |

The topo-confidence score comes from a single forward pass on the prompt (no generation required). It captures a fundamentally different signal from output-based methods — the geometry of the model's internal computation, not the uncertainty of its output distribution.

## Cross-benchmark results (deconfounded)

| Config | Accuracy | Topo AUROC | Best Baseline | Gap |
|--------|----------|------------|---------------|-----|
| Qwen2.5-1.5B x MATH-500 | 104/500 (20.8%) | **0.796** | vote_margin 0.767 | **+0.057** |
| Qwen2.5-1.5B x GSM8K | 871/1319 (66.0%) | 0.615 | neg_entropy 0.741 | -0.126 |
| Qwen2.5-7B x MATH-500 | 348/500 (69.6%) | **0.739** | first_token 0.637 | **+0.102** |

**Takeaway**: Topo features show genuine signal on MATH-500 across model scales (1.5B and 7B), but do not generalize to GSM8K where logprob baselines dominate. Cross-benchmark transfer is near chance (AUROC 0.504). The signal appears specific to hard mathematical reasoning where output probabilities are poorly calibrated.

## Install

```bash
pip install -e .
```

## Usage

### Score prompts

```python
from topo_confidence import TopoConfidence

tc = TopoConfidence("Qwen/Qwen2.5-1.5B-Instruct")
tc.calibrate(calibration_prompts, calibration_labels)
confidences = tc.predict_confidence(["What is 2+2?", "Prove the Riemann Hypothesis"])
```

### Selective prediction — only answer when confident

```python
results = tc.selective_predict(problems, threshold=0.7)
# results["answers"]: list of str | None (None = skipped)
# results["confidences"]: array of P(correct)
# results["answered_fraction"]: what fraction was answered
```

### CLI

```bash
# Score prompts
topo-confidence score "What is 2+2?" "What is the integral of x^3?"

# Calibrate on labeled data
topo-confidence calibrate data.jsonl -o calibrated.pkl

# Selective prediction
topo-confidence selective -c calibrated.pkl -f problems.txt -t 0.7

# Explain a prediction
topo-confidence explain -c calibrated.pkl "What is 2+2?"
```

### Combine with output entropy

```python
from topo_confidence import CombinedConfidence

cc = CombinedConfidence("Qwen/Qwen2.5-1.5B-Instruct")
cc.calibrate(prompts, labels)  # uses topo features + output entropy + max token prob
```

### Save and load calibrated models

```python
tc.save("math_calibrated.pkl")

tc2 = TopoConfidence("Qwen/Qwen2.5-1.5B-Instruct")
tc2.load("math_calibrated.pkl")
```

## How it works

Each prompt's hidden states at the final transformer layer form a point cloud in R^d (one point per token). The CORAL feature pipeline extracts 44 topological and geometric features organized into three tiers:

### Feature tiers

| Tier | Count | Description | Standalone AUROC |
|------|-------|-------------|-----------------|
| A | 9 | Persistent homology + geometry (H0/H1 entropy, lifetimes, centroid distances) | 0.704 |
| B | 30 | Layer dynamics (inter-layer cosines, PCA spectrum, SVD ratios) | 0.761 |
| C | 5 | Depth-2 products (cross-tier interactions) | 0.732 |
| **A+B+C** | **44** | **Full model** | **0.796** |

### Top features by leave-one-out importance

| Feature | LOO Impact | Tier |
|---------|-----------|------|
| cos_l23_l26 | -0.041 | B |
| last5_centroid_dist | -0.035 | A |
| H1_persistence_entropy | -0.030 | A |
| cos_l9_l28 | -0.028 | B |
| cos_l17_l27 | -0.028 | B |

The signal is distributed — no single feature is critical. Tier A (persistent homology) provides the irreplaceable anchor, while Tier B (layer dynamics) provides the largest AUROC boost (+0.057 from A to A+B).

## Baseline comparison

On the MATH-500 holdout (n=100):

| Method | Passes | AUROC |
|--------|--------|-------|
| **Topo-confidence (44 features)** | 1 | **0.796** |
| Vote margin | 32 | 0.767 |
| P(majority) | 32 | 0.708 |
| Neg agreement entropy | 32 | 0.617 |
| Inv answer diversity | 32 | 0.593 |
| Mean max token prob | 1 | 0.514 |
| Neg mean entropy | 1 | 0.508 |
| First token prob | 1 | 0.472 |

Topo-confidence is the only single-pass method that competes with 32-pass self-consistency baselines.

## Selection strategies

### Gated majority vote (tau sweep, holdout)

| Threshold | Answered | Net Gain | W->R | R->W |
|-----------|----------|----------|------|------|
| 0.1 | 8 | 0 | 0 | 0 |
| 0.3 | 37 | **+4** | 4 | **0** |
| 0.5 | 68 | +8 | 9 | 1 |
| 0.7 | 88 | +11 | 13 | 2 |
| Ungated | 100 | +12 | 14 | 2 |

At tau=0.3, gating achieves zero regressions (0 R->W) — the model never makes a correct answer worse. The tradeoff is coverage: only 37% of problems are answered.

### Adaptive sampling

Topo-confidence enables 3-tier routing (easy/medium/hard) that matches uniform 32-pass majority vote accuracy with 77-89% fewer samples.

## Research status

This is an active research project. Key findings as of April 2026:

**Established:**
- Topo AUROC 0.796 on MATH-500 x 1.5B (corrected from initial 0.948 after fixing answer extraction and PCA leakage bugs)
- Topo AUROC 0.739 on MATH-500 x 7B (genuine cross-model signal, +0.102 over baseline)
- Zero-regression gating at tau=0.3 (0 R->W)
- Adaptive sampling: 77-89% sample savings

**Refuted:**
- Cross-benchmark generalization to GSM8K (AUROC 0.615, baseline wins at 0.741)
- Cross-benchmark transfer (MATH -> GSM8K = 0.504, chance level)

**Open questions:**
- Why does the signal appear specific to MATH-500? (Hypothesis: hard reasoning where logprobs are poorly calibrated)
- Can per-completion features improve gated majority vote beyond +4?
- Comparison with SEP (Semantic Entropy Probes, Kossen et al.)

See `pathway6_rebuild/phase6_5/FINAL_SUMMARY.md` for the full deconfounded analysis.

## Project structure

```
topo_confidence/           # Python package (pip installable)
  confidence.py            # TopoConfidence class
  combined.py              # CombinedConfidence (topo + entropy fusion)
  extractor.py             # HiddenStateExtractor
  features.py              # TopologicalFeatureExtractor
  baselines.py             # Output entropy, max token prob
  cli.py                   # Command-line interface

pathway1/                  # CORAL feature extraction + baseline model
pathway2/                  # Steering experiments (Track A/B)
pathway3/                  # Generalization gap experiments
pathway4/                  # Selection strategies + GRPO steering
pathway5/                  # Cross-benchmark exploration
pathway6_rebuild/          # Bug-fix rebuild + Phase 6.5 deconfounding
  phase0_relabel/          # Answer extraction fix (+47 problems)
  phase1_prompt_model/     # Corrected AUROC 0.796
  phase2_completion/       # Selection strategies (MV, gating)
  phase3_cross_benchmark/  # GSM8K + 7B (confounded)
  phase4_report/           # Corrected vs old comparison
  phase6_5/                # Deconfounded results (max_tokens 1024)
```

## Citation

If you use topo-confidence in your research, please cite:

```bibtex
@software{topo_confidence,
  title={topo-confidence: Topological Uncertainty Estimation for LLM Reasoning},
  author={musicofhel},
  year={2026},
  url={https://github.com/musicofhel/topo-confidence}
}
```

## License

MIT
