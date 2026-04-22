#!/bin/bash
# Pathway 9 RunPod runner — executes the 4 remaining experiments with resume.
#
# Run order:
#   1. exp3a_v2 (CPU, ~30-60 min) — full-covariance null rerun [audit fix D1+D2]
#   2. exp2     (CPU, ~2 min)     — PH vs CoE orthogonality
#   3. exp5     (CPU, ~15 min)    — MATH↔BBH cross-domain transfer
#   4. exp3b    (GPU, ~15 min)    — token-shuffle null
#
# Resume: each script is idempotent via .done markers in pathway9/results/.
# Run this again after interruption and skipped stages will be noted.
#
# Usage: bash pathway9/run_runpod.sh [--skip-exp3b]  # skip GPU step if CPU-only pod

set -e
cd "$(dirname "$0")/.."

RESULTS="pathway9/results"
mkdir -p "$RESULTS"

SKIP_EXP3B=0
for arg in "$@"; do
    [[ "$arg" == "--skip-exp3b" ]] && SKIP_EXP3B=1
done

run_if_needed() {
    local name="$1" script="$2"
    if [ -f "$RESULTS/${name}.done" ]; then
        echo "[SKIP] $name already done ($RESULTS/${name}.done exists)"
        return 0
    fi
    echo ""
    echo "=========================================="
    echo "RUNNING $name: $script"
    echo "=========================================="
    python3 "$script"
    if [ ! -f "$RESULTS/${name}.done" ]; then
        echo "[WARN] $script did not write $RESULTS/${name}.done"
        return 1
    fi
}

# Order: cheap CPU tasks first, so GPU pod time is only used for exp3b
run_if_needed "exp3a_v2" "pathway9/exp3a_full_cov_null.py"
run_if_needed "exp2"     "pathway9/exp2_orthogonality.py"
run_if_needed "exp5"     "pathway9/exp5_cross_domain_transfer.py"

if [ "$SKIP_EXP3B" = "1" ]; then
    echo "[SKIP] exp3b (GPU) — --skip-exp3b flag set"
else
    run_if_needed "exp3b" "pathway9/exp3b_token_shuffle.py"
fi

echo ""
echo "=========================================="
echo "ALL DONE"
echo "=========================================="
ls -la "$RESULTS/"*.done
