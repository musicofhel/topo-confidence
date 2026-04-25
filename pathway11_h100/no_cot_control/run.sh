#!/usr/bin/env bash
# Test 2 — No-CoT control for PR breathing on Qwen2.5-1.5B at L19.
#
# Same MATH-500 first-50 subset, same extraction pipeline as the gibberish
# control MATH-500 baseline, but with system prompt swapped to suppress CoT:
# "Answer with just the number, no explanation."
#
# Compares against the existing gibberish_control/data/math500/ baseline
# (identical params except SYSTEM_PROMPT).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
EXTRACT="$REPO_ROOT/pathway11_h100/exp1_cross_model/extract.py"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "=============================================================="
echo "Step 1/3: extract no-CoT condition (n=50, MATH-500 first 50)"
echo "=============================================================="
python "$EXTRACT" \
  --model qwen25-1.5b \
  --system-prompt "Answer with just the number, no explanation." \
  --output-dir "$HERE/data/nocot" \
  --n-problems 50 \
  --max-new-tokens 256

echo
echo "=============================================================="
echo "Step 2/3: compute PR curves (no-CoT vs CoT baseline)"
echo "=============================================================="
python "$HERE/compute_pr.py"

echo
echo "=============================================================="
echo "Step 3/3: plot overlay"
echo "=============================================================="
python "$HERE/plot_curves.py"

echo
echo "Done. Results:"
echo "  $HERE/pr_curves.json"
echo "  $HERE/nocot_vs_cot.png"
