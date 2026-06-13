#!/bin/bash
# v8_p2_teacher.sh TEACHER_KEY — Phase 2 step 1: generate teacher traces on
# the n=4000 MATH-train pool. Teacher frozen at Gate G1 (default math7b).
TEACHER="${1:-math7b}"
cd /workspace
export HF_HUB_ENABLE_HF_TRANSFER=1
echo "=== TEACHER $TEACHER START $(date -u +%H:%M:%S) ==="
python3 v8_runpod_distill.py --task teacher --model "$TEACHER" \
  > "pod_results/v8_teacher_${TEACHER}.log" 2>&1
rc=$?
echo "=== TEACHER EXIT rc=$rc $(date -u +%H:%M:%S) ==="
if [ $rc -eq 0 ]; then touch "pod_results/DONE_teacher"; else touch "pod_results/FAIL_teacher"; fi
