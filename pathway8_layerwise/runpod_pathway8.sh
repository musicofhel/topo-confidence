#!/usr/bin/env bash
# Pathway 8: Layer-wise PH Pipeline — Master execution script for RunPod H100
#
# Usage: bash pathway8_layerwise/runpod_pathway8.sh [--from STEP]
#   --from 0  Start from validation (default)
#   --from 1  Start from MATH-500 extraction (GPU)
#   --from 2  Start from Experiment 1 (CPU)
#   --from 3  Start from Experiment 2 (CPU)
#   --from 4  Start from GO/NO-GO gate
#   --from 5  Start from cross-domain extraction (GPU)
#   --from 6  Start from Experiment 3 (CPU)
#   --from 7  Start from Experiment 4 (CPU)
#   --from 8  Start from Experiment 5 (CPU)
#   --from 9  Start from summary report
set -euo pipefail

PATHWAY8_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$PATHWAY8_DIR")"
cd "$REPO_DIR"

# Parse --from argument
START_STEP=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --from) START_STEP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Create output directories
mkdir -p pathway8_layerwise/{data,results,logs}

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
    echo "Step $step_num: $label"
    echo "================================================================"
    local start_time=$(date +%s)

    python "$script" 2>&1 | tee "pathway8_layerwise/logs/step${step_num}_$(date +%Y%m%d_%H%M%S).log"

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo ""
    echo "[DONE] Step $step_num: ${mins}m ${secs}s"
    echo ""
}

echo "=== Pathway 8: Layer-wise PH Experiments ==="
echo "Starting from step $START_STEP"
echo "Time: $(date)"
echo ""

# GPU info
if nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "(No GPU detected — GPU steps will fail)"
fi
echo ""

TOTAL_START=$(date +%s)

# ---- Step 0: Validation (CPU, ~2 min) ----
if [ "$START_STEP" -le 0 ]; then
    echo ""
    echo "================================================================"
    echo "Step 0: PH Pipeline Validation"
    echo "================================================================"
    python -c "
import sys
sys.path.insert(0, '.')
from pathway8_layerwise.diagnostics import validate_ph_pipeline
validate_ph_pipeline()
print('[PASS] Step 0 complete')
" 2>&1 | tee "pathway8_layerwise/logs/step0_$(date +%Y%m%d_%H%M%S).log"
fi

# ---- Step 1: MATH-500 extraction (GPU, ~25 min) ----
run_step 1 "MATH-500 all-layer extraction (GPU)" \
    "pathway8_layerwise/extract_math500.py"

# ---- Step 2: Experiment 1 — Layer-wise PH (CPU, ~2 hr) ----
run_step 2 "Experiment 1: Layer-wise PH on MATH-500 (CPU)" \
    "pathway8_layerwise/exp1_layerwise_ph.py"

# ---- Step 3: Experiment 2 — Method comparison (CPU, ~1 hr) ----
run_step 3 "Experiment 2: PH vs CoE vs D2HScore (CPU)" \
    "pathway8_layerwise/exp2_method_comparison.py"

# ---- Step 4: GO/NO-GO gate ----
if [ "$START_STEP" -le 4 ]; then
    echo ""
    echo "================================================================"
    echo "Step 4: GO/NO-GO Gate"
    echo "================================================================"
    python -c "
import json, sys
exp1_path = 'pathway8_layerwise/results/exp1_results.json'
exp2_path = 'pathway8_layerwise/results/exp2_results.json'
try:
    exp1 = json.load(open(exp1_path))
    exp2 = json.load(open(exp2_path))
    print(f'  Exp 1 AUROC: {exp1[\"auroc_holdout\"]}')
    print(f'  Exp 2 best:  {exp2[\"best_model\"]} AUROC={exp2[\"best_auroc\"]}')
    print(f'  Baseline:    {exp1[\"baseline_auroc\"]}')
    best = max(exp1['auroc_holdout'], exp2['best_auroc'])
    if best > 0.75:
        print(f'  [GO] Best AUROC {best:.4f} > 0.75 — proceeding to cross-domain')
    else:
        print(f'  [CAUTION] Best AUROC {best:.4f} < 0.75 — proceeding anyway for completeness')
except Exception as e:
    print(f'  [WARN] Could not read results: {e}')
    print(f'  Proceeding to cross-domain extraction anyway')
" 2>&1 | tee "pathway8_layerwise/logs/step4_$(date +%Y%m%d_%H%M%S).log"
fi

# ---- Step 5: Cross-domain extraction (GPU, ~25 min) ----
if [ "$START_STEP" -le 5 ]; then
    echo ""
    echo "================================================================"
    echo "Step 5a: HumanEval extraction (GPU)"
    echo "================================================================"
    python pathway8_layerwise/extract_humaneval.py 2>&1 | tee "pathway8_layerwise/logs/step5a_$(date +%Y%m%d_%H%M%S).log"

    echo ""
    echo "================================================================"
    echo "Step 5b: BBH extraction (GPU)"
    echo "================================================================"
    python pathway8_layerwise/extract_bbh.py 2>&1 | tee "pathway8_layerwise/logs/step5b_$(date +%Y%m%d_%H%M%S).log"
fi

# ---- Step 6: Experiment 3 — Cross-domain (CPU, ~3 hr) ----
run_step 6 "Experiment 3: Cross-domain generalization (CPU)" \
    "pathway8_layerwise/exp3_cross_domain.py"

# ---- Step 7: Experiment 4 — TwoNN ID (CPU, ~30 min) ----
run_step 7 "Experiment 4: TwoNN intrinsic dimension (CPU)" \
    "pathway8_layerwise/exp4_twonn_id.py"

# ---- Step 8: Experiment 5 — Cross-layer trajectory (CPU, ~2 hr) ----
run_step 8 "Experiment 5: Cross-layer trajectory PH (CPU)" \
    "pathway8_layerwise/exp5_crosslayer_trajectory.py"

# ---- Step 9: Summary report ----
if [ "$START_STEP" -le 9 ]; then
    echo ""
    echo "================================================================"
    echo "Step 9: Summary Report"
    echo "================================================================"
    python -c "
import json, glob
print('=== Final Results ===')
for f in sorted(glob.glob('pathway8_layerwise/results/exp*_results.json')):
    try:
        data = json.load(open(f))
        exp = data.get('experiment', f)
        if 'auroc_holdout' in data:
            print(f'  {exp}: AUROC={data[\"auroc_holdout\"]}')
        elif 'best_auroc' in data:
            print(f'  {exp}: best={data[\"best_model\"]} AUROC={data[\"best_auroc\"]}')
        elif 'benchmarks' in data:
            for name, res in data['benchmarks'].items():
                if isinstance(res, dict) and 'methods' in res:
                    best = max(res['methods'].values(), key=lambda x: x['auroc_holdout'])
                    print(f'  {exp}/{name}: AUROC={best[\"auroc_holdout\"]}')
        elif 'models' in data:
            for m in data['models']:
                print(f'  {exp}/{m[\"name\"]}: AUROC={m[\"auroc_holdout\"]}')
    except Exception as e:
        print(f'  {f}: error reading ({e})')
print()
print('Done markers:')
import os
for f in sorted(glob.glob('pathway8_layerwise/results/*.done')):
    print(f'  {os.path.basename(f)}')
" 2>&1 | tee "pathway8_layerwise/logs/step9_$(date +%Y%m%d_%H%M%S).log"
fi

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$(( TOTAL_END - TOTAL_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "================================================================"
echo "ALL DONE — Total: ${TOTAL_MINS}m ${TOTAL_SECS}s"
echo "================================================================"
echo ""
echo "Key outputs:"
echo "  pathway8_layerwise/results/exp1_results.json    (layer-wise PH AUROC)"
echo "  pathway8_layerwise/results/exp2_results.json    (method comparison)"
echo "  pathway8_layerwise/results/exp3_results.json    (cross-domain)"
echo "  pathway8_layerwise/results/exp4_results.json    (TwoNN ID)"
echo "  pathway8_layerwise/results/exp5_results.json    (cross-layer trajectory)"
echo "  pathway8_layerwise/logs/                        (full logs)"
