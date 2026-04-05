---
title: topo-confidence
emoji: 🔮
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: spaces/app.py
pinned: false
license: mit
---

# topo-confidence

Predict whether an LLM will answer correctly — from the shape of its hidden states.

Uses persistent homology to extract topological features from token representations. Position 0 serves as a computational bridge between two universal clusters. Correct answers have simpler geometry.

**2.6x selective prediction lift** at top-10% confidence threshold.
