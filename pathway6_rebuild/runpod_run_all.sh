#!/usr/bin/env bash
# Pathway 6 Rebuild — Master execution script for RunPod H100
# Usage: bash runpod_run_all.sh [--from STEP]
#   --from 1  Start from Phase 2 step1 (default)
#   --from 2  Start from Phase 2 step2 (skip GPU extraction)
#   --from 3  Start from Phase 3 GSM8K
#   --from 4  Start from Phase 3 MATH-7B
#   --from 5  Start from Phase 3 cross-analysis
#   --from 6  Start from Phase 4 report only
set -euo pipefail

REBUILD_DIR="$(cd "$(dirname "$0")" && pwd)"
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
    echo "Step $step_num/6: $label"
    echo "================================================================"
    local start_time=$(date +%s)

    python "$script" 2>&1 | tee "logs/step${step_num}_$(date +%Y%m%d_%H%M%S).log"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo ""
    echo "[DONE] Step $step_num: ${mins}m ${secs}s"
    echo ""
}

# Create logs directory
mkdir -p logs

echo "=== Pathway 6 Rebuild: Full Pipeline ==="
echo "Starting from step $START_STEP"
echo "Time: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

TOTAL_START=$(date +%s)

# Step 1: Phase 2 GPU — per-completion trajectory extraction (~30-60 min)
run_step 1 "Phase 2 GPU: per-completion trajectories (1.5B)" \
    "phase2_completion/step1_extract_trajectories.py"

# Step 2: Phase 2 CPU — re-run with real per-completion scores (~10 min)
run_step 2 "Phase 2 CPU: features + experiments (with real scores)" \
    "phase2_completion/step2_features_and_experiments.py"

# Step 3: Phase 3 GSM8K — full pipeline with chat template (~2-3h)
run_step 3 "Phase 3: GSM8K × 1.5B full pipeline" \
    "phase3_cross_benchmark/step1_gsm8k_full.py"

# Step 4: Phase 3 MATH-7B — full pipeline with chat template (~2-3h)
run_step 4 "Phase 3: MATH-500 × 7B full pipeline" \
    "phase3_cross_benchmark/step2_math7b_full.py"

# Step 5: Phase 3 cross-analysis — CPU only (~5 min)
run_step 5 "Phase 3: cross-benchmark analysis (CPU)" \
    "phase3_cross_benchmark/step3_cross_analysis.py"

# Step 6: Phase 4 report — regenerate with all results (~1 min)
run_step 6 "Phase 4: regenerate final report" \
    "phase4_report/generate_report.py"

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
echo "  phase2_completion/train_completion_features_v2.npz"
echo "  phase2_completion/experiment5_v2.json"
echo "  phase3_cross_benchmark/gsm8k/summary.json"
echo "  phase3_cross_benchmark/math7b/summary.json"
echo "  phase3_cross_benchmark/cross_analysis.json"
echo "  phase4_report/FINAL_REPORT.json"
