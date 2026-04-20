#!/usr/bin/env bash
# Pathway 7: Non-Euclidean PH Pipeline — Master execution script for RunPod H100
#
# Usage: bash pathway7/runpod_pathway7.sh [--from STEP]
#   --from 0  Start from validation (default)
#   --from 1  Start from Phase 7.1 non-Euclidean on MATH-500 (CPU)
#   --from 2  Start from HumanEval benchmark (GPU)
#   --from 3  Start from BBH benchmark (GPU)
#   --from 4  Start from all-layer extraction for zigzag (GPU)
#   --from 5  Start from zigzag features (CPU)
#   --from 6  Start from combined analysis (CPU)
set -euo pipefail

PATHWAY7_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$PATHWAY7_DIR")"
cd "$REPO_DIR"

# Parse --from argument
START_STEP=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --from) START_STEP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

run_step() {
    local step_num="$1"
    local label="$2"
    local script="$3"

    if [ "$step_num" -lt "$START_STEP" ]; then
        echo "[SKIP] Step $step_num: $label"
        return
    fi

    echo ""
    echo "================================================================"
    echo "Step $step_num/6: $label"
    echo "================================================================"
    local start_time=$(date +%s)

    python "$script" 2>&1 | tee "pathway7/logs/step${step_num}_$(date +%Y%m%d_%H%M%S).log"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo ""
    echo "[DONE] Step $step_num: ${mins}m ${secs}s"
    echo ""
}

# Create logs directory
mkdir -p pathway7/logs

echo "=== Pathway 7: Non-Euclidean PH Pipeline ==="
echo "Starting from step $START_STEP"
echo "Time: $(date)"
if nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "(No GPU detected — CPU-only steps will work)"
fi
echo ""

TOTAL_START=$(date +%s)

# Step 0: Validation — synthetic shape PH correctness (CPU, ~5 min)
run_step 0 "Validation: synthetic shapes + metric divergence (CPU)" \
    "pathway7/validation/synthetic_shapes.py"

# Also run metric divergence if step 0 is active
if [ "$START_STEP" -le 0 ]; then
    echo "--- Metric divergence validation ---"
    python pathway7/validation/metric_divergence.py 2>&1 | tee "pathway7/logs/step0b_$(date +%Y%m%d_%H%M%S).log"
fi

# Step 1: Phase 7.1 — non-Euclidean PH on existing MATH-500 (CPU, ~1 hr)
run_step 1 "Phase 7.1: eff-res + cosine PH on MATH-500 (CPU)" \
    "pathway7/phase71_noneuclid_math500.py"

# GO/NO-GO GATE: Check Phase 7.1 results
if [ "$START_STEP" -le 1 ] && [ -f "pathway7/results_phase71/phase71_results.json" ]; then
    echo ""
    echo "--- Phase 7.1 Results ---"
    python -c "
import json
r = json.load(open('pathway7/results_phase71/phase71_results.json'))
print(f'  Baseline AUROC:      {r[\"baseline_auroc\"]:.4f}')
print(f'  Best combined AUROC: {r[\"best_combined_auroc\"]:.4f}')
print(f'  Improvement:         {r[\"auroc_improvement\"]:+.4f}')
print(f'  DeLong p-value:      {r[\"delong\"][\"p_value\"]:.4f} ({r[\"delong\"][\"significance\"]})')
"
    echo ""
fi

# Step 2: HumanEval benchmark (GPU, ~30 min)
run_step 2 "HumanEval benchmark (GPU)" \
    "pathway7/benchmarks/humaneval_pipeline.py"

# Step 3: BBH benchmark (GPU, ~30 min)
run_step 3 "BBH benchmark: 3 subsets × 250 (GPU)" \
    "pathway7/benchmarks/bbh_pipeline.py"

# Steps 4-6: Zigzag + combined analysis (SKIPPED by default)
# Zigzag ($8, 5hrs) deferred unless HumanEval/BBH results justify it.
# Uncomment to re-enable:
#
# run_step 4 "All-layer hidden state extraction for zigzag (GPU)" \
#     "pathway7/extract_all_layers.py"
# run_step 5 "Zigzag features (CPU)" \
#     "pathway7/zigzag_features.py"
# run_step 6 "Combined analysis (CPU)" \
#     "pathway7/combined_analysis.py"

echo ""
echo "Steps 4-6 (zigzag) skipped. Re-enable in runpod_pathway7.sh if needed."

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "================================================================"
echo "ALL DONE — Total: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "================================================================"
echo ""
echo "Key outputs to check:"
echo "  pathway7/results_phase71/phase71_results.json   (non-Euclidean AUROC)"
echo "  pathway7/results_humaneval/summary.json          (HumanEval AUROC)"
echo "  pathway7/results_bbh/summary.json                (BBH AUROC)"
echo "  pathway7/logs/                                   (full logs)"
