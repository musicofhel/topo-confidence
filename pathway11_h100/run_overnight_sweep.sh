#!/usr/bin/env bash
set -euo pipefail

VENV="/home/musicofhel/topo-confidence/.venv/bin/python"
LOGDIR="/home/musicofhel/topo-confidence/pathway11_h100/logs/sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
FAILED=0

run_batch() {
    local batch_name=$1; shift
    local threads=$1; shift
    local pids=()
    local names=()

    echo ""
    echo "=== $batch_name (threads=$threads) === [$(date +%H:%M:%S)]"
    for script in "$@"; do
        local name=$(basename "$script" .py)
        echo "  Starting $name..."
        OMP_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads MKL_NUM_THREADS=$threads \
            "$VENV" "$script" > "$LOGDIR/${name}.log" 2>&1 &
        pids+=($!)
        names+=("$name")
    done

    for i in "${!pids[@]}"; do
        if ! wait "${pids[$i]}"; then
            echo "  FAILED: ${names[$i]} (exit $?, see $LOGDIR/${names[$i]}.log)"
            FAILED=1
        else
            echo "  OK: ${names[$i]} [$(date +%H:%M:%S)]"
        fi
    done
    echo "  Batch done [$(date +%H:%M:%S)]"
}

BASE="/home/musicofhel/topo-confidence/pathway11_h100"

echo "Overnight sweep started at $(date)"
echo "Logs: $LOGDIR"
echo ""

# Batch 1: Pure activation math (~10 min, 3 parallel)
run_batch "Batch 1: Activation Math" 9 \
    "$BASE/gram_eigenspectrum/recompute_fe01172B.py" \
    "$BASE/mcca_prefill_final/recompute_fe903.py" \
    "$BASE/subspace_angles/recompute_fe899.py"

# Batch 2: Model weights (~30 min, 4 parallel)
run_batch "Batch 2: Model Weights" 7 \
    "$BASE/rope_projection/recompute_fe930.py" \
    "$BASE/setol_ecs/recompute_fe901.py" \
    "$BASE/logit_lens_entropy/recompute_fe26841a.py" \
    "$BASE/procrustes_rotation/recompute_fe889.py"

# Batch 3: Compute-intensive (~45 min, 3 parallel)
run_batch "Batch 3: Compute-Intensive" 9 \
    "$BASE/alignment_profile/recompute_fe909.py" \
    "$BASE/hyvarinen_score/recompute_fe925.py" \
    "$BASE/diffusion_maps/recompute_fe919.py"

echo ""
echo "=== SWEEP COMPLETE === [$(date)]"
echo "Logs: $LOGDIR"

RESULT_COUNT=$(ls "$BASE/results/"fe*.json 2>/dev/null | wc -l)
echo "Result JSONs: $RESULT_COUNT/10"

if [ $FAILED -eq 0 ]; then
    echo "ALL PASSED"
else
    echo "SOME FAILED — check logs"
fi
exit $FAILED
