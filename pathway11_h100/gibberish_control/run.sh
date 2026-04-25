#!/usr/bin/env bash
# Test 3 — gibberish control for PR breathing, on Qwen2.5-1.5B at L19.
#
# Runs on RunPod H100 pod after repo + pathway11_h100/ are in place.
# Outputs land in pathway11_h100/gibberish_control/data/{random,stream,math500}/
# and (after CPU analysis) pr_curves.json + gibberish_vs_math500.png.
#
# The analysis steps (compute_pr.py + plot_curves.py) can also be run locally
# after scp'ing the data/ back — they are CPU-only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
EXTRACT="$REPO_ROOT/pathway11_h100/exp1_cross_model/extract.py"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=============================================================="
echo "Step 1/5: generate prompt JSONs"
echo "=============================================================="
python "$HERE/gen_prompts.py" \
  --tokenizer Qwen/Qwen2.5-1.5B-Instruct \
  --n 20 \
  --tokens-per-prompt 60 \
  --seed 0

echo
echo "=============================================================="
echo "Step 2/5: extract random-token condition (n=20, no chat template)"
echo "=============================================================="
python "$EXTRACT" \
  --model qwen25-1.5b \
  --prompts-file "$HERE/prompts_random.json" \
  --skip-chat-template \
  --output-dir "$HERE/data/random" \
  --n-problems 20 \
  --max-new-tokens 256

echo
echo "=============================================================="
echo "Step 3/5: extract stream-of-consciousness condition (n=20)"
echo "=============================================================="
python "$EXTRACT" \
  --model qwen25-1.5b \
  --prompts-file "$HERE/prompts_stream.json" \
  --output-dir "$HERE/data/stream" \
  --n-problems 20 \
  --max-new-tokens 256

echo
echo "=============================================================="
echo "Step 4/5: extract MATH-500 baseline (n=50, for methodology match)"
echo "=============================================================="
python "$EXTRACT" \
  --model qwen25-1.5b \
  --output-dir "$HERE/data/math500" \
  --n-problems 50 \
  --max-new-tokens 256

echo
echo "=============================================================="
echo "Step 5/5: CPU analysis + plot"
echo "=============================================================="
python "$HERE/compute_pr.py"
python "$HERE/plot_curves.py"

echo
echo "Done. Results:"
echo "  $HERE/pr_curves.json"
echo "  $HERE/gibberish_vs_math500.png"
