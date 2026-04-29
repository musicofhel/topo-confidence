#!/bin/bash
# RunPod launch script for Pathway 4 experiments
# Recommended: RTX 4090 (24GB), PyTorch 2.x template
#
# SETUP (on your local machine first):
#   1. Start a RunPod pod with PyTorch template + RTX 4090
#   2. SCP the data tarball:
#        scp -P <PORT> /tmp/topo-data-artifacts.tar.gz root@<RUNPOD_IP>:/workspace/
#   3. SSH in and run:
#        cd /workspace && bash runpod_launch.sh
#
# Total runtime on 4090: ~5-6 hours for all phases

set -euo pipefail

WORKSPACE=/workspace
REPO_DIR=$WORKSPACE/topo-confidence
DATA_TAR=$WORKSPACE/topo-data-artifacts.tar.gz
LOG=$WORKSPACE/pathway4_run.log

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ---- Step 0: Clone repo + install deps ----
log "=== Step 0: Setup ==="

if [ ! -d "$REPO_DIR" ]; then
    cd "$WORKSPACE"
    git clone https://github.com/musicofhel/topo-confidence.git
fi
cd "$REPO_DIR"

pip install -q ripser cripser scikit-learn datasets transformers accelerate scipy 2>&1 | tail -5

# ---- Step 1: Unpack data artifacts ----
log "=== Step 1: Unpacking data artifacts ==="
if [ -f "$DATA_TAR" ]; then
    tar xzf "$DATA_TAR" -C "$REPO_DIR"
    log "Data artifacts unpacked"
else
    log "ERROR: $DATA_TAR not found. SCP it first:"
    log "  scp -P <PORT> /tmp/topo-data-artifacts.tar.gz root@<IP>:/workspace/"
    exit 1
fi

# Verify critical files
for f in data/experiment1_v2/trajectories.npz \
         pathway2/track_a/phase0/baseline_correct.npy \
         pathway3/phase1/temperature_generations.json; do
    if [ ! -f "$REPO_DIR/$f" ]; then
        log "ERROR: Missing $f"
        exit 1
    fi
done
log "All data artifacts verified"

# ---- Step 2: Create output dirs ----
mkdir -p pathway4/track_a/{phase1,phase2,phase3}
mkdir -p pathway4/track_b/{pilot,checkpoints,evaluation}

# ---- TRACK A ----

# Phase A1: Score temperature completions
log "=== Track A Phase A1: Scoring completions ==="
cd "$REPO_DIR/pathway4/track_a"
python3 phase1_score.py 2>&1 | tee -a "$LOG"
log "Phase A1 complete"

# Phase A2: 5-fold CV
log "=== Track A Phase A2: CV evaluation ==="
python3 phase2_cv.py 2>&1 | tee -a "$LOG"
log "Phase A2 complete"

# Check go/no-go for A3
MEAN_GAIN=$(python3 -c "
import json
with open('phase2/locked_strategy.json') as f:
    d = json.load(f)
print(d.get('mean_net_gain', 0))
" 2>/dev/null || echo "0")
log "Phase A2 mean_net_gain: $MEAN_GAIN"

if python3 -c "exit(0 if float('$MEAN_GAIN') > 3 else 1)" 2>/dev/null; then
    log "=== Track A Phase A3: Holdout evaluation ==="
    python3 phase3_holdout.py 2>&1 | tee -a "$LOG"
    log "Phase A3 complete"
else
    log "SKIP Phase A3: mean_net_gain ($MEAN_GAIN) <= 3"
fi

# ---- TRACK B ----

# Phase B0: Pilot validation
log "=== Track B Phase B0: Pilot validation ==="
cd "$REPO_DIR/pathway4/track_b"
python3 phase0_pilot.py 2>&1 | tee -a "$LOG"
log "Phase B0 complete"

# Phase B1: Full GRPO training
log "=== Track B Phase B1: GRPO training ==="
python3 phase1_train.py 2>&1 | tee -a "$LOG"
log "Phase B1 complete"

# Phase B2: Evaluate
log "=== Track B Phase B2: Evaluation ==="
python3 phase2_evaluate.py 2>&1 | tee -a "$LOG"
log "Phase B2 complete"

# ---- Summary ----
log ""
log "=========================================="
log "=== ALL PHASES COMPLETE ==="
log "=========================================="
log ""
log "Results:"
log "  Track A Phase 1: pathway4/track_a/phase1/phase1_summary.json"
log "  Track A Phase 2: pathway4/track_a/phase2/locked_strategy.json"
log "  Track A Phase 3: pathway4/track_a/phase3/holdout_results.json"
log "  Track B Phase 0: pathway4/track_b/pilot/pilot_report.json"
log "  Track B Phase 1: pathway4/track_b/trained_bias.pt"
log "  Track B Phase 2: pathway4/track_b/evaluation/comparison.json"
log ""
log "To pull results back locally:"
log "  scp -P <PORT> -r root@<IP>:/workspace/topo-confidence/pathway4/ ~/topo-confidence/pathway4/"
log "  scp -P <PORT> root@<IP>:/workspace/pathway4_run.log ~/topo-confidence/"
