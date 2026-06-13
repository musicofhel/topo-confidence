#!/bin/bash
# v8_p2_train.sh — Phase 2 step 2: per-arm LoRA SFT then single-shot eval
# (interleaved so each arm's adapter + eval survive an interruption), then the
# un-adapted base reference (forgetting baseline + 0.486 re-confirm).
# Prereq: pod_payload/v8_traces_{a,b,c,d}.jsonl.gz already pushed.
cd /workspace
export HF_HUB_ENABLE_HF_TRANSFER=1
for ARM in a b c d; do
  if [ ! -f "pod_payload/v8_traces_${ARM}.jsonl.gz" ]; then
    echo "MISSING pod_payload/v8_traces_${ARM}.jsonl.gz — aborting"; touch pod_results/FAIL_p2; exit 1
  fi
  echo "=== SFT arm $ARM START $(date -u +%H:%M:%S) ==="
  python3 v8_runpod_distill.py --task sft --arm "$ARM" \
    > "pod_results/v8_sft_${ARM}.log" 2>&1
  rc=$?; echo "=== SFT $ARM EXIT rc=$rc ==="
  if [ $rc -ne 0 ]; then touch "pod_results/FAIL_sft_${ARM}"; continue; fi
  echo "=== EVAL arm $ARM START $(date -u +%H:%M:%S) ==="
  python3 v8_runpod_distill.py --task evalsft --arm "$ARM" \
    > "pod_results/v8_evalsft_${ARM}.log" 2>&1
  rc=$?; echo "=== EVAL $ARM EXIT rc=$rc ==="
  if [ $rc -eq 0 ]; then touch "pod_results/DONE_arm_${ARM}"; else touch "pod_results/FAIL_eval_${ARM}"; fi
done
echo "=== EVALBASE START $(date -u +%H:%M:%S) ==="
python3 v8_runpod_distill.py --task evalbase \
  > "pod_results/v8_evalbase.log" 2>&1
rc=$?; echo "=== EVALBASE EXIT rc=$rc ==="
if [ $rc -eq 0 ]; then touch "pod_results/DONE_base"; else touch "pod_results/FAIL_base"; fi
touch pod_results/ALL_P2_DONE
