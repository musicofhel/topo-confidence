# Pathway 11 pipeline — live handoff (2026-04-24)

Pipeline running autonomously on RunPod H100. Compact-safe.

## Pod
- **ID**: `y687b9z2dgukcj`
- **SSH**: `ssh -i /home/musicofhel/.runpod/ssh/RunPod-Key-Go root@103.207.149.173 -p 15277`
- **Cost**: $2.99/hr H100 SXM secure cloud
- **Started**: 2026-04-23 16:56 UTC
- **Workspace**: `/workspace/topo-confidence/`
- **Run PID** (on pod): `cat /workspace/runlogs/pipeline.pid` (was 2142)
- **Run log**: `/workspace/runlogs/current.log`

## Final status (2026-04-24, pipeline complete, pod STOPPED)

Total runtime: 1016 min (~17 hr). All 7 markers present.

| Stage | Status | Output |
|---|---|---|
| 0 preflight | ✅ done | marker |
| 1 7B MATH-500 all-layer | ✅ 73.2% (366/500) | `~/topo-confidence/pathway11_h100/data/math500_7b/` (42 GB) |
| 2 1.5B MATH-500 all-layer | ✅ 48.6% (243/500) | `~/topo-confidence/pathway8_layerwise/data/math500/` (19 GB) |
| 3 K=8 self-consistency 1.5B | ✅ 500/500 | `~/topo-confidence/pathway11_h100/data/k8_selfconsistency/` (13 MB) |
| 4a BBH 3×250 | ✅ acc 12.4/6.0/54.0% | `~/topo-confidence/pathway8_layerwise/data/bbh/` (18 GB, 750 files) |
| 4b BBH per-subset diagnostics | ✅ done | `results/stage4b_bbh_per_subset.json` |
| 5 L19 DoM transfer | ✅ symmetric 0.720 vs CoE 0.716 | `results/stage5_bbh_dom_transfer.json` |

**Pod status**: EXITED (stopped, disk preserved at storage-only rate for tomorrow). Resume with `runpodctl pod start y687b9z2dgukcj`.

## Remaining work

All pipeline work complete. Pod is stopped with disk preserved. To resume tomorrow:
```bash
runpodctl pod start y687b9z2dgukcj
# IP/port may reassign — check with: runpodctl pod list
```
SSH key path: `/home/musicofhel/.runpod/ssh/RunPod-Key-Go`. Workspace at `/workspace/topo-confidence/`.

## Progress check snippet

```bash
POD_KEY=/home/musicofhel/.runpod/ssh/RunPod-Key-Go; POD_IP=103.207.149.173; POD_PORT=15277
ssh -i $POD_KEY -o StrictHostKeyChecking=no -p $POD_PORT root@$POD_IP '
  PID=$(cat /workspace/runlogs/pipeline.pid)
  ps -p $PID -o pid,etime,stat 2>&1 | head -3
  for d in pathway11_h100/data/math500_7b pathway8_layerwise/data/math500 pathway11_h100/data/k8_selfconsistency pathway8_layerwise/data/bbh; do
    n=$(ls /workspace/topo-confidence/$d/problem_*.npz 2>/dev/null | wc -l)
    echo "$d: $n"
  done
  echo "markers: $(ls /workspace/topo-confidence/pathway11_h100/results/*.done 2>/dev/null | xargs -n1 basename | tr "\n" " ")"
  grep -E "^Stage [0-9]|\[DONE\]|\[FAIL\]|Majority-vote" /workspace/runlogs/current.log 2>/dev/null | tail -10
'
```

## Design notes
- Stage 1 has `capture_attention=False` for 7B → `d2h_attn_entropy` array is all zeros (by design, GQA pass is skipped). Stage 2 has real attention entropy.
- Smoke test confirmed stage 3 K=8 + L19 capture code path (5 problems, 5m23s).
- Stage accuracy 73.2% vs phase6_5's 69.6%: +18 problems within bf16 non-determinism.
- max_new_tokens=1024; 7.8% of 7B and 14.0% of 1.5B problems hit the cap.

## Active background tasks in Claude session
- `bry33xx0k` (persistent Monitor): streams pod log progress lines + errors
- Scheduled wakeups (self-firing): every ~60 min to re-check progress

The wakeup prompts are self-contained — they include the pod ID and SSH details.
