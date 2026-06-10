#!/usr/bin/env bash
# FE269 staged orchestrator (cheapest-decisive-first, .done markers, gate-first).
# Mirrors runpod_pathway11.sh. Re-running skips completed stages. Each stage writes
# results/<name>.json; a hang still leaves earlier stages' results on disk.
#
#   bash run_fe269.sh            # run all stages
#   STAGES="S0 S1" bash run_fe269.sh   # run a subset
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
RES="$HERE/results"; mkdir -p "$RES"
DONE="$HERE/.done"; mkdir -p "$DONE"
PY="${PY:-python}"
BATCH="${BATCH:-32}"
LOG="$RES/run.log"
ALL_STAGES="S0 S1 S2 S3 S4 S5"
STAGES="${STAGES:-$ALL_STAGES}"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
have(){ [[ -f "$DONE/$1" ]]; }
mark(){ touch "$DONE/$1"; }

stage_run(){  # stage_run <name> <cmd...>
  local name="$1"; shift
  if have "$name"; then log "SKIP $name (done)"; return 0; fi
  log "START $name : $*"
  if "$@" >>"$LOG" 2>&1; then mark "$name"; log "OK   $name"; else
    log "FAIL $name (see $LOG) — continuing so earlier results survive"; return 1; fi
}

want(){ [[ " $STAGES " == *" $1 "* ]]; }

# ---- S0 preflight: GPU, imports, r_hat present, version pin, hook self-test ----
if want S0 && ! have S0; then
  log "START S0 preflight"
  $PY - <<'PY' >>"$LOG" 2>&1
import torch, transformers, numpy as np
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
assert transformers.__version__ == "4.57.6", "transformers must be 4.57.6 (bare-Tensor hook)"
assert torch.cuda.is_available(), "no CUDA"
r = np.load("r_hat.npy")
assert r.shape == (1536,), r.shape
print("r_hat ok", r.shape, float(np.linalg.norm(r)))
PY
  s0a=$?
  $PY intervention.py >>"$LOG" 2>&1; s0b=$?
  if [[ $s0a -eq 0 && $s0b -eq 0 ]]; then mark S0; log "OK   S0"; else log "FAIL S0"; fi
fi

# ---- S1 baseline GATE: MATH off/all must reproduce 48.6% ----
if want S1; then
  stage_run S1 $PY eval_math500.py --mode off --layers all --batch "$BATCH" --out "$RES/math_off.json"
  if have S1; then
    $PY - <<'PY' >>"$LOG" 2>&1
import json; a=json.load(open("results/math_off.json"))["accuracy"]*100
print(f"GATE math_off acc={a:.2f}%  (expect 47.0–50.2)")
import sys; sys.exit(0 if 47.0<=a<=50.2 else 3)
PY
    if [[ $? -ne 0 ]]; then
      log "GATE FAILED: baseline outside [47.0,50.2]. Halting interventions (plumbing bug)."
      log "If batched-numerics suspected, re-run with BATCH=8: STAGES='S1' BATCH=8 bash run_fe269.sh"
      exit 3
    fi
    log "GATE PASSED"
  fi
fi

# ---- S2 L19-only ablation (the specific-signal arm) + GSM8K off & L19-ablate ----
if want S2; then
  stage_run S2_math  $PY eval_math500.py --mode ablate --layers 18  --batch "$BATCH" --out "$RES/math_ablate_L19.json"
  stage_run S2_gsmoff $PY eval_gsm8k.py  --mode off    --layers all --batch "$BATCH" --out "$RES/gsm8k_off.json"
  stage_run S2_gsmab  $PY eval_gsm8k.py  --mode ablate --layers 18  --batch "$BATCH" --out "$RES/gsm8k_ablate_L19.json"
  have S2_math && have S2_gsmoff && have S2_gsmab && mark S2
fi

# ---- S3 all-layer ablation (aggressive) + MMLU off, L19-ablate, all-ablate ----
if want S3; then
  stage_run S3_math    $PY eval_math500.py --mode ablate --layers all --batch "$BATCH" --out "$RES/math_ablate_all.json"
  stage_run S3_gsmall  $PY eval_gsm8k.py   --mode ablate --layers all --batch "$BATCH" --out "$RES/gsm8k_ablate_all.json"
  stage_run S3_mmoff   $PY eval_mmlu.py    --mode off    --layers all --batch "$BATCH" --out "$RES/mmlu_off.json"
  stage_run S3_mmL19   $PY eval_mmlu.py    --mode ablate --layers 18  --batch "$BATCH" --out "$RES/mmlu_ablate_L19.json"
  stage_run S3_mmall   $PY eval_mmlu.py    --mode ablate --layers all --batch "$BATCH" --out "$RES/mmlu_ablate_all.json"
  have S3_math && have S3_mmoff && mark S3
fi

# ---- S4 addition α-sweep at L19 on FULL 500 (flip on incorrect + retention on correct) ----
if want S4; then
  for A in 0.5 1 2 4; do
    tag="${A/./p}"
    stage_run "S4_add_$tag" $PY eval_math500.py --mode add --layers 18 --alpha-mult "$A" \
      --batch "$BATCH" --out "$RES/math_add_a${tag}.json"
  done
  mark S4
fi

# ---- S5 verdict ----
if want S5; then
  log "START S5 decision"
  $PY decision.py --results "$RES" --out "$RES/verdict.json" >>"$LOG" 2>&1 && mark S5 && log "OK   S5"
  $PY - <<'PY'
import json
try:
    v=json.load(open("results/verdict.json"))
    print("\n================ FE269 VERDICT:", v["verdict"], "================")
    for c in v.get("ablation_configs",[]):
        print(f"  [{c['config']}] ΔMATH={c['delta_math']:.1f}pp  "
              f"gsm_drop={c['gsm8k_drop']}  mmlu_drop={c['mmlu_drop']}  -> {c['necessity_class']}")
    ba=v.get("best_addition")
    if ba: print(f"  best add: α×{ba['alpha_mult']} flip={ba['flip_rate']:.3f} retain={ba['retention']:.3f}")
except FileNotFoundError:
    print("no verdict.json yet")
PY
fi

log "run_fe269 done. Results in $RES (commit the *.json; npz stays ignored)."
