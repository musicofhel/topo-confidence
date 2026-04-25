#!/usr/bin/env bash
# Exp 1 runner — extracts Phi-3-mini + Llama-3.2-1B sequentially on GPU,
# then runs analysis (CPU). Per-model per-problem checkpoints make this
# idempotent on restart.
#
# Usage:
#   bash pathway11_h100/exp1_cross_model/run.sh                 # both models + analyze
#   bash pathway11_h100/exp1_cross_model/run.sh phi3mini        # one model
#   bash pathway11_h100/exp1_cross_model/run.sh --analyze-only  # no extraction
set -euo pipefail

EXP1_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$EXP1_DIR/../.." && pwd)"
cd "$REPO_DIR"

LOGS_DIR="$EXP1_DIR/logs"
mkdir -p "$LOGS_DIR"

ANALYZE_ONLY=0
MODELS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --analyze-only) ANALYZE_ONLY=1; shift ;;
        phi3mini|llama32-1b) MODELS+=("$1"); shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("phi3mini" "llama32-1b")
fi

if [[ $ANALYZE_ONLY -eq 0 ]]; then
    for m in "${MODELS[@]}"; do
        ts=$(date +%Y%m%d_%H%M%S)
        log="$LOGS_DIR/extract_${m}_${ts}.log"
        echo "=== extract: $m === (log: $log)"
        python pathway11_h100/exp1_cross_model/extract.py --model "$m" 2>&1 | tee "$log"
    done
fi

ts=$(date +%Y%m%d_%H%M%S)
log="$LOGS_DIR/analyze_${ts}.log"
echo "=== analyze === (log: $log)"
python pathway11_h100/exp1_cross_model/analyze.py 2>&1 | tee "$log"

echo "Done."
echo "  data/          $EXP1_DIR/data/"
echo "  results.json   $EXP1_DIR/results.json"
echo "  temporal PR    $EXP1_DIR/temporal_pr_curve.png"
echo "  depth PR       $EXP1_DIR/depth_pr_prefill_vs_final.png"
