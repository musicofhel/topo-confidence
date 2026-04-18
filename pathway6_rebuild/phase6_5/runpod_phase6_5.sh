#!/usr/bin/env bash
# Phase 6.5: Truncation Fix — RunPod H100 execution script
#
# Reruns GSM8K × 1.5B and MATH-500 × 7B with max_new_tokens=1024
# (was 256, causing 64%/90% truncation and confounded AUROCs).
#
# Usage: bash pathway6_rebuild/phase6_5/runpod_phase6_5.sh [--from STEP]
#   --from 1  GSM8K × 1.5B (default)
#   --from 2  MATH-500 × 7B
#   --from 3  Cross-analysis (CPU)
#   --from 4  Summary (CPU)
#
# Prerequisites: Run runpod_setup.sh first (installs deps, downloads models).
# Estimated total: 4-6 hours on H100, ~$15-25.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REBUILD_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REBUILD_DIR"

# Parse --from argument
START_STEP=1
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
    echo "Step $step_num/4: $label"
    echo "================================================================"
    local start_time=$(date +%s)

    python "$script" 2>&1 | tee "phase6_5/logs/step${step_num}_$(date +%Y%m%d_%H%M%S).log"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo ""
    echo "[DONE] Step $step_num: ${mins}m ${secs}s"
    echo ""
}

# Create logs directory
mkdir -p phase6_5/logs

echo "=== Phase 6.5: Truncation Fix (max_new_tokens=1024) ==="
echo "Starting from step $START_STEP"
echo "Time: $(date)"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
fi
echo ""

TOTAL_START=$(date +%s)

# Step 1: GSM8K × 1.5B with max_new_tokens=1024 (~3-4h)
run_step 1 "GSM8K × 1.5B (max_new_tokens=1024)" \
    "phase6_5/step1_gsm8k_1024.py"

# Step 2: MATH-500 × 7B with max_new_tokens=1024 (~2-3h)
run_step 2 "MATH-500 × 7B (max_new_tokens=1024)" \
    "phase6_5/step2_math7b_1024.py"

# Step 3: Cross-analysis (CPU, ~5 min)
run_step 3 "Cross-benchmark analysis (deconfounded)" \
    "phase6_5/step3_cross_analysis.py"

# Step 4: Summary (CPU, ~1 min)
run_step 4 "Generate FINAL_SUMMARY.md" \
    "phase6_5/step4_summary.py"

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
echo "  phase6_5/gsm8k/summary.json"
echo "  phase6_5/math7b/summary.json"
echo "  phase6_5/cross_analysis.json"
echo "  phase6_5/FINAL_SUMMARY.md"
echo ""
echo "Copy results back:"
echo "  rsync -avz phase6_5/ ~/topo-confidence-results/phase6_5/"
