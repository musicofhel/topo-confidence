#!/usr/bin/env bash
# Pathway 11: H100 pod data run — master orchestrator
#
# Stages:
#   0  Preflight (GPU detect, disk, imports)
#   1  7B all-layer per-token extraction on MATH-500, 1024 tok (GPU, ~2-3 hr)
#   2  1.5B all-layer per-token extraction on MATH-500, 1024 tok (GPU, ~25 min)
#   3  K=8 self-consistency sampling × 1.5B at T=0.7, L19 per sample (GPU, ~45 min)
#   4a BBH extraction, 3 subsets × 250 (GPU, ~15 min)
#   4b BBH per-subset diagnostics (CPU, ~10 min)
#   5  L19 DoM cross-benchmark transfer (CPU, ~10 min)
#
# Usage:
#   bash pathway11_h100/runpod_pathway11.sh                  # resume-from-last-.done
#   bash pathway11_h100/runpod_pathway11.sh --from 3         # manual start at stage 3
#   bash pathway11_h100/runpod_pathway11.sh --dry-run        # print plan, no-op
#   STAGE3_N_PROBLEMS=5 bash runpod_pathway11.sh --from 3 --to 3   # smoke-test stage 3
#
# Checkpointing:
#   - Each stage writes pathway11_h100/results/stageN.done on success.
#   - On rerun, completed stages are auto-skipped (unless --from forces them).
#   - Within a stage, per-problem .npz checkpoints make individual reruns idempotent.
set -euo pipefail

P11_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$P11_DIR")"
cd "$REPO_DIR"

RESULTS_DIR="$P11_DIR/results"
LOGS_DIR="$P11_DIR/logs"
mkdir -p "$P11_DIR/data" "$RESULTS_DIR" "$LOGS_DIR"

# ---- args -----------------------------------------------------------------
FROM_STAGE=""
TO_STAGE=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --from) FROM_STAGE="$2"; shift 2 ;;
        --to)   TO_STAGE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            grep -E '^#( |$)' "$0" | sed 's/^# //; s/^#$//'
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Stage keys (stage name for .done file) and human labels, in execution order.
STAGE_KEYS=("stage0" "stage1" "stage2" "stage3" "stage4a" "stage4b" "stage5")
STAGE_LABELS=(
    "Preflight"
    "7B × MATH-500 all-layer extraction (GPU)"
    "1.5B × MATH-500 all-layer extraction (GPU)"
    "K=8 self-consistency × 1.5B (GPU)"
    "BBH extraction (GPU)"
    "BBH per-subset diagnostics (CPU)"
    "L19 DoM cross-benchmark transfer (CPU)"
)

# ---- helpers --------------------------------------------------------------
is_done() {
    # $1: stage key. Returns 0 if done marker exists.
    [[ -f "$RESULTS_DIR/$1.done" ]]
}

mark_done_bash() {
    # $1: stage key.
    touch "$RESULTS_DIR/$1.done"
}

run_cmd() {
    # $1: stage key for log file name
    # $2..: command + args to run
    local key="$1"; shift
    local log_file="$LOGS_DIR/${key}_$(date +%Y%m%d_%H%M%S).log"
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] $key: would run: $*"
        echo "[DRY-RUN] $key: log would go to $log_file"
        return 0
    fi
    "$@" 2>&1 | tee "$log_file"
}

stage_header() {
    local idx="$1"; local label="$2"
    echo ""
    echo "================================================================"
    echo "Stage $idx: $label"
    echo "================================================================"
}

stage_footer() {
    local elapsed="$1"
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo "[DONE] ${mins}m ${secs}s"
    echo ""
}

# Decide whether to run a given stage index based on --from/--to/.done markers.
should_run() {
    local idx="$1"; local key="$2"
    if [[ -n "$FROM_STAGE" && "$idx" -lt "$FROM_STAGE" ]]; then
        echo "[SKIP] Stage $idx ($key): before --from=$FROM_STAGE"
        return 1
    fi
    if [[ -n "$TO_STAGE" && "$idx" -gt "$TO_STAGE" ]]; then
        echo "[SKIP] Stage $idx ($key): past --to=$TO_STAGE"
        return 1
    fi
    if [[ -z "$FROM_STAGE" ]] && is_done "$key"; then
        echo "[SKIP] Stage $idx ($key): .done marker present"
        return 1
    fi
    return 0
}

# ---- banner + GPU/disk info ----------------------------------------------
echo "=== Pathway 11: H100 pod data run ==="
echo "Time: $(date)"
echo "From stage: ${FROM_STAGE:-auto (skip done)}, to stage: ${TO_STAGE:-end}"
if [[ $DRY_RUN -eq 1 ]]; then echo "(DRY RUN — no commands will execute)"; fi
echo ""
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true
else
    echo "(no nvidia-smi — GPU stages will fail on CPU-only machines)"
fi
echo "Disk in repo dir:"
df -h "$REPO_DIR" | tail -n 1
echo ""

TOTAL_START=$(date +%s)

# ============================================================================
# Stage 0: preflight
# ============================================================================
if should_run 0 "stage0"; then
    stage_header 0 "${STAGE_LABELS[0]}"
    t0=$(date +%s)
    run_cmd stage0 python -c "
import sys, shutil
from pathlib import Path

# Import sanity
sys.path.insert(0, '.')
from pathway11_h100 import config as cfg
from pathway8_layerwise.extraction_utils import load_model, extract_all_layers_and_attention  # noqa
from pathway8_layerwise.coe_features import compute_coe_batch  # noqa
print('[OK] imports')

# math_verify
try:
    from math_verify import parse, verify  # noqa
    print('[OK] math_verify importable')
except ImportError as e:
    print(f'[FAIL] math_verify missing: {e}')
    sys.exit(1)

# Disk check
du = shutil.disk_usage(str(cfg.P11_DIR))
free_gb = du.free / (1024**3)
print(f'Free disk at pathway11_h100/: {free_gb:.1f} GB')
if free_gb < 30:
    print(f'[FAIL] need >=30 GB free; have {free_gb:.1f}')
    sys.exit(1)
print('[OK] disk >=30 GB')

# GPU check (non-fatal — CPU stages still work)
try:
    import torch
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        free, total = torch.cuda.mem_get_info(0)
        print(f'[OK] GPU: {n} device(s), free={free/1e9:.1f} GB, total={total/1e9:.1f} GB')
        if free < 60e9:
            print(f'[WARN] GPU free<60 GB: 7B at bf16 needs ~15 GB weights + activation tensors')
    else:
        print('[WARN] no CUDA — GPU stages will fail; CPU-only stages (4b, 5) will still run')
except Exception as e:
    print(f'[WARN] torch import: {e}')
print('[PASS] preflight')
"
    elapsed=$(( $(date +%s) - t0 ))
    [[ $DRY_RUN -eq 0 ]] && mark_done_bash "stage0"
    stage_footer "$elapsed"
fi

# ============================================================================
# Stage 1: 7B MATH-500 extraction (GPU)
# ============================================================================
if should_run 1 "stage1"; then
    stage_header 1 "${STAGE_LABELS[1]}"
    t0=$(date +%s)
    # Script calls mark_done("stage1") on its own once all 500 complete.
    run_cmd stage1 python pathway11_h100/stage1_extract_math500_7b.py
    elapsed=$(( $(date +%s) - t0 ))
    stage_footer "$elapsed"
fi

# ============================================================================
# Stage 2: 1.5B MATH-500 extraction (reuses pathway8 script)
# ============================================================================
if should_run 2 "stage2"; then
    stage_header 2 "${STAGE_LABELS[2]}"
    t0=$(date +%s)
    run_cmd stage2 python pathway8_layerwise/extract_math500.py
    # The pathway8 script does not touch pathway11 markers — wrap it.
    if [[ $DRY_RUN -eq 0 ]]; then
        n_files=$(ls "$REPO_DIR/pathway8_layerwise/data/math500/"problem_*.npz 2>/dev/null | wc -l)
        if [[ "$n_files" -ge 500 ]]; then
            mark_done_bash "stage2"
            echo "[OK] 500/500 math500 files present, stage2 marker written"
        else
            echo "[FAIL] only $n_files/500 math500 files; not marking stage2 done"
            exit 1
        fi
    fi
    elapsed=$(( $(date +%s) - t0 ))
    stage_footer "$elapsed"
fi

# ============================================================================
# Stage 3: K=8 self-consistency (GPU)
# ============================================================================
if should_run 3 "stage3"; then
    stage_header 3 "${STAGE_LABELS[3]}"
    t0=$(date +%s)
    run_cmd stage3 python pathway11_h100/stage3_k8_selfconsistency.py
    elapsed=$(( $(date +%s) - t0 ))
    stage_footer "$elapsed"
fi

# ============================================================================
# Stage 4a: BBH extraction (reuses pathway8 script)
# ============================================================================
if should_run 4 "stage4a"; then
    stage_header 4a "${STAGE_LABELS[4]}"
    t0=$(date +%s)
    run_cmd stage4a python pathway8_layerwise/extract_bbh.py
    if [[ $DRY_RUN -eq 0 ]]; then
        # Expect at least 3 subsets × some files and a manifest
        manifest="$REPO_DIR/pathway8_layerwise/data/bbh/manifest.json"
        if [[ -f "$manifest" ]]; then
            mark_done_bash "stage4a"
            echo "[OK] BBH manifest written; stage4a marker set"
        else
            echo "[FAIL] BBH manifest missing at $manifest"
            exit 1
        fi
    fi
    elapsed=$(( $(date +%s) - t0 ))
    stage_footer "$elapsed"
fi

# ============================================================================
# Stage 4b: BBH per-subset diagnostics (CPU)
# ============================================================================
if should_run 5 "stage4b"; then
    stage_header 4b "${STAGE_LABELS[5]}"
    t0=$(date +%s)
    run_cmd stage4b python pathway11_h100/stage4b_bbh_per_subset.py
    elapsed=$(( $(date +%s) - t0 ))
    stage_footer "$elapsed"
fi

# ============================================================================
# Stage 5: L19 DoM cross-benchmark transfer (CPU)
# ============================================================================
if should_run 6 "stage5"; then
    stage_header 5 "${STAGE_LABELS[6]}"
    t0=$(date +%s)
    run_cmd stage5 python pathway11_h100/stage5_bbh_dom_transfer.py
    elapsed=$(( $(date +%s) - t0 ))
    stage_footer "$elapsed"
fi

# ---- summary --------------------------------------------------------------
TOTAL_END=$(date +%s)
TOTAL=$(( TOTAL_END - TOTAL_START ))
TOTAL_M=$(( TOTAL / 60 ))
TOTAL_S=$(( TOTAL % 60 ))

echo ""
echo "================================================================"
echo "ALL DONE — total ${TOTAL_M}m ${TOTAL_S}s"
echo "================================================================"
echo "Markers:"
ls "$RESULTS_DIR"/*.done 2>/dev/null || echo "  (none)"
echo ""
echo "Key outputs:"
echo "  pathway11_h100/data/math500_7b/          (stage 1: 7B per-token per-layer npz)"
echo "  pathway8_layerwise/data/math500/         (stage 2: 1.5B per-token per-layer npz)"
echo "  pathway11_h100/data/k8_selfconsistency/  (stage 3: K=8 + L19 per sample)"
echo "  pathway8_layerwise/data/bbh/{subset}/    (stage 4a: BBH per-token per-layer npz)"
echo "  pathway11_h100/results/                  (stages 4b, 5: JSON diagnostics)"
echo "  pathway11_h100/logs/                     (per-stage timestamped logs)"
