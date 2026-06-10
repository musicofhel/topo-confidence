#!/usr/bin/env bash
# FE19 staged orchestrator (local 2060/CPU). Gate-first, .done markers, exception-safe.
# Re-running skips completed stages.
#   bash run_fe19.sh                 # all stages
#   STAGES="S0 S1" bash run_fe19.sh  # subset
#   BATCH=8 bash run_fe19.sh         # smaller batch if 8GB OOM
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
RES="$HERE/results"; mkdir -p "$RES"
DONE="$HERE/.done"; mkdir -p "$DONE"
PY="${PY:-python}"
BATCH="${BATCH:-16}"
LOG="$RES/run.log"
ALL_STAGES="S0 S1 S2 S3 S4 S5 S6"
STAGES="${STAGES:-$ALL_STAGES}"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
have(){ [[ -f "$DONE/$1" ]]; }
mark(){ touch "$DONE/$1"; }
want(){ [[ " $STAGES " == *" $1 "* ]]; }

stage_run(){  # stage_run <name> <cmd...>
  local name="$1"; shift
  if have "$name"; then log "SKIP $name (done)"; return 0; fi
  log "START $name : $*"
  if "$@" >>"$LOG" 2>&1; then mark "$name"; log "OK   $name"; else
    log "FAIL $name (see $LOG) — continuing so earlier results survive"; return 1; fi
}

# ---- S0 preflight ----
if want S0 && ! have S0; then
  log "START S0 preflight"
  $PY - <<'PY' >>"$LOG" 2>&1
import torch, transformers, numpy as np, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
sys.path.insert(0, "../causal_dom")
import common  # noqa
from pathlib import Path
k8 = list(Path("../data/k8_selfconsistency").glob("problem_*.npz"))
ph1 = Path("../prefill_gated_compute/phase1_majority_vote.npz")
ph2 = Path("../prefill_gated_compute/phase2_prefill_dom.npz")
s2 = list(Path("../../pathway8_layerwise/data/math500").glob("problem_*.npz"))
assert len(s2) == 500, f"stage2 files {len(s2)}"
assert ph1.exists() and ph2.exists(), "missing phase1/phase2 cache"
print(f"caches OK: stage2={len(s2)} k8={len(k8)} phase1={ph1.exists()} phase2={ph2.exists()}")
PY
  if [[ $? -eq 0 ]]; then mark S0; log "OK   S0"; else log "FAIL S0"; exit 1; fi
fi

# ---- S1 K=1 greedy GATE (CPU cache-load; must reproduce 48.6%) ----
if want S1; then
  stage_run S1 $PY load_k1.py
  if ! have S1; then log "GATE FAILED at S1 — halting (plumbing bug)."; exit 3; fi
fi

# ---- S2 verification pass + PANL capture (GPU) ----
want S2 && stage_run S2 $PY verify_pass.py --batch "$BATCH"

# ---- S3 verify-then-correct (GPU) ----
want S3 && stage_run S3 $PY correct_pass.py --batch "$BATCH"

# ---- S4 PANL probe (CPU) ----
want S4 && stage_run S4 $PY fit_panl_probe.py

# ---- S5 routing simulation (CPU) ----
want S5 && stage_run S5 $PY route_sim.py

# ---- S6 decision (CPU) ----
want S6 && stage_run S6 $PY decision.py

log "DONE (stages: $STAGES)"
