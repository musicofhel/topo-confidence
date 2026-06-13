#!/bin/bash
# v8_run_task.sh MODEL [EXTRA_ARGS...] — launch one frontier task in the
# background with a log; guard against double-launch (pgrep bracket trick).
MODEL="$1"; shift
if pgrep -f "[v]8_runpod_frontier" > /dev/null; then
  echo "BUSY: a frontier task is already running"; exit 1
fi
cd /workspace
export HF_HUB_ENABLE_HF_TRANSFER=1
nohup python3 v8_runpod_frontier.py --model "$MODEL" "$@" \
  > "pod_results/v8_${MODEL}.log" 2>&1 &
echo "LAUNCHED $MODEL pid $!"
