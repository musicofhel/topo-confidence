# topo-confidence

**Predict whether an LLM will answer correctly — from a single forward pass.**

topo-confidence computes persistent homology on a language model's hidden states and uses topological features to estimate the probability that the model's output is correct. No multiple sampling. No output parsing. No fine-tuning.

## Quickstart

```bash
pip install -e .
```

```python
from topo_confidence import TopoConfidence

tc = TopoConfidence("Qwen/Qwen2.5-1.5B-Instruct")

# Calibrate on labeled examples (one-time cost)
tc.calibrate(calibration_prompts, calibration_labels)

# Predict confidence on new problems
confidences = tc.predict_confidence(["What is 2+2?", "Prove Fermat's Last Theorem"])
# [0.92, 0.03]

# Only answer when confident
results = tc.selective_predict(problems, threshold=0.7)
```

## How it works

1. Run the model's forward pass (you're doing this anyway to get the answer)
2. Extract the token-by-token hidden states at the final layer
3. PCA-reduce and compute persistent homology (H0 and H1)
4. Extract 6 scalar topological features
5. Predict P(correct) via pre-calibrated logistic regression

The key insight: **H0 persistence entropy** of the hidden-state token trajectory correlates with correctness at AUROC=0.824 as a single feature. Problems the model will solve correctly have geometrically simpler representations.

## Results (from ATT Phase 5)

| Benchmark | Model | AUROC | Top Feature |
|-----------|-------|-------|-------------|
| MATH-500  | Qwen2.5-1.5B-Instruct | 0.787 | H0_persistence_entropy |
| HumanEval | Qwen2.5-1.5B-Instruct | 0.772 | H1_n_features |

6-feature logistic regression, 50-fold CV: 0.783 ± 0.054 (95% CI [0.695, 0.889]).

## Why not just use...

| Method | Requires | Captures |
|--------|----------|----------|
| Output entropy | Logits | Output-level uncertainty |
| Self-consistency | 5-40 samples | Agreement across samples |
| P(true) | Extra prompt | Self-reported confidence |
| Semantic entropy | Multiple samples + clustering | Semantic diversity |
| **topo-confidence** | **1 forward pass** | **Geometry of internal representation** |

topo-confidence is complementary to all of the above. It measures a fundamentally different signal — the topological structure of the model's hidden states — at zero additional inference cost.

## Experiments

### Experiment 1: Reproduce MATH-500 result

```bash
python scripts/experiment1_math500.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda
```

Target: AUROC ≥ 0.75 on MATH-500 correctness prediction.

### Benchmark against baselines

```bash
python scripts/benchmark.py data/test.jsonl --model Qwen/Qwen2.5-1.5B-Instruct
```

### Interactive demo

```bash
python scripts/demo.py --model Qwen/Qwen2.5-1.5B-Instruct
```

## The 6 features

| Feature | Description | Interpretation |
|---------|-------------|----------------|
| H0_persistence_entropy | Shannon entropy of H0 lifetimes | High = complex connected components = uncertain |
| H1_max_lifetime | Longest-lived 1-cycle | Large loop = representational instability |
| H0_total_persistence | Sum of H0 lifetimes | Total connected-component complexity |
| H0_n_features | Number of H0 features | More components = more fragmented |
| H1_persistence_entropy | Shannon entropy of H1 lifetimes | High = many loops = uncertain |
| H1_n_features | Number of 1-cycles | More loops = more topological noise |

## Dependencies

- `torch` + `transformers` (model inference)
- `ripser` (persistent homology)
- `persim` (persistence utilities)
- `scikit-learn` (PCA, logistic regression)
- `numpy`

No GUDHI. No giotto-tda. Minimal stack.

## License

MIT
