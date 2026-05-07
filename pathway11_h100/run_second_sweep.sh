#!/usr/bin/env bash
set -euo pipefail

VENV="/home/musicofhel/topo-confidence/.venv/bin/python"
LOGDIR="/home/musicofhel/topo-confidence/pathway11_h100/logs/sweep2_$(date +%Y%m%d_%H%M%S)"
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

echo "Second sweep started at $(date)"
echo "Logs: $LOGDIR"
echo ""

# Batch 1: Convenience cache only (~30 min, 5 parallel, 5 threads each)
run_batch "Batch 1: Convenience Cache" 5 \
    "$BASE/length_baseline/recompute_fe447.py" \
    "$BASE/lid_mle/recompute_fe188.py" \
    "$BASE/regularized_concat/recompute_fe421.py" \
    "$BASE/length_band_pr/recompute_fe15.py" \
    "$BASE/mp_bias_pr/recompute_fe16.py"

# Batch 2: Per-problem NPZs (~30 min, 3 parallel, 9 threads each)
run_batch "Batch 2: Per-Problem NPZs" 9 \
    "$BASE/l0_embedding_dom/recompute_fe428.py" \
    "$BASE/prefinal_token_dom/recompute_fe416.py" \
    "$BASE/layer_sweep_cos/recompute_fe119.py"

# Batch 3: Specialized data (~1h, 2 parallel, 14 threads each)
run_batch "Batch 3: Specialized Data" 14 \
    "$BASE/adaptive_bestofk/recompute_fe308.py" \
    "$BASE/cross_model_dom/recompute_fe459.py"

echo ""
echo "=== SWEEP COMPLETE === [$(date)]"
echo "Logs: $LOGDIR"

RESULT_COUNT=$(ls "$BASE/results/"fe*.json 2>/dev/null | wc -l)
echo "Result JSONs in results/: $RESULT_COUNT"

if [ $FAILED -eq 0 ]; then
    echo "ALL PASSED"
else
    echo "SOME FAILED — check logs"
fi
exit $FAILED
