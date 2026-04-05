# topo-confidence

**Know when your LLM is wrong — from a single forward pass.**

topo-confidence uses persistent homology on hidden-state geometry to predict whether an LLM's output is correct. It captures a fundamentally different signal from output entropy (correlation r=0.062), adds 0.007s overhead, and requires no extra generation.

## The headline result

On MATH-500 with Qwen2.5-1.5B-Instruct, selecting the top 10% of outputs by topo-confidence yields **30% accuracy vs 11.4% baseline** — a 2.6x lift. This works because topological features measure the *geometry of the model's internal computation*, not the *uncertainty of its output distribution*.

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

Each prompt's hidden states at the final transformer layer form a point cloud in R^d (one point per token). Persistent homology extracts 7 topological features from this cloud:

| Feature | What it measures |
|---------|-----------------|
| H0_persistence_entropy | How fragmented the token representations are |
| H0_total_persistence | Total spread of connected components |
| H0_n_features | Number of distinct clusters |
| H1_max_lifetime | Strength of the dominant loop |
| H1_persistence_entropy | Complexity of loop structure |
| H1_n_features | Number of loops |
| bridge_silhouette | Position-0 token's boundary score in k=2 clustering |

The bridge feature comes from a discovery that transformer hidden states universally organize into two clusters, with position 0 serving as the sole computational bridge between them (see [ATT research](https://github.com/musicofhel/att-docs)).

## Why it works

Correct answers have simpler hidden-state geometry:
- Lower H0 entropy (fewer fragments — model "knows what it's doing")
- The bridge token (position 0) sits cleanly between clusters for correct answers

This signal is **orthogonal to output entropy** (r=0.062) — topo-confidence catches failures that output-based methods miss, and vice versa.

## Performance

| Metric | Value |
|--------|-------|
| Selective prediction lift (top 10%) | 2.6x |
| AUROC (MATH-500, 7 features) | 0.699 |
| Best single feature AUROC | 0.824 (H0_persistence_entropy, from ATT Phase 5) |
| Feature computation overhead | 0.007s per problem |
| Correlation with output entropy | r=0.062 (orthogonal) |

## Applications

- **Model routing**: Use as a gate for cheap-to-expensive model cascading
- **Agentic verification**: Per-step confidence without multi-sampling
- **Training data filtering**: Select high-confidence synthetic data
- **Real-time monitoring**: Track hidden-state geometry during generation

## Project structure

```
topo_confidence/
├── confidence.py     # TopoConfidence class (calibrate, predict, explain)
├── combined.py       # CombinedConfidence (topo + output entropy fusion)
├── extractor.py      # HiddenStateExtractor (model inference)
├── features.py       # TopologicalFeatureExtractor (PH computation)
├── baselines.py      # Output entropy, max token prob baselines
├── cli.py            # Command-line interface
└── utils.py          # Persistence entropy, subsampling
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
