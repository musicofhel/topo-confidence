# Test 3 — Gibberish control for PR breathing

**Status:** complete. Pod `lsuoka6bo8io7m` ran 2026-04-24 21:54–22:10 UTC,
stopped (not removed). Wall-clock ~15 min; GPU-cost ~$0.75.

**Headline:** PR breathing is **content-dependent**, not an autoregressive
artifact.

- MATH-500 (n=50): classic breathing — PR rises 13.7 → 30.5 at pos 50 → 22.2 at final.
- **Random tokens (n=20): flat at PR ≈ 10** across all positions (9.0–11.7). No breathing.
- Stream-of-consciousness (n=20): muted, differently-shaped — 4.6 → 15.7 @ pos 10 → 9.2 final. Early peak then monotonic decline, distinct from MATH.

Null (PR breathes on any generation) **rejected**: random-token input
collapses the entire curve to a flat plateau. Fluent-but-non-reasoning
produces some PR dynamics but with lower absolute values and different
timing than reasoning. Breathing tracks reasoning-task structure.

Figure: [`gibberish_vs_math500.png`](./gibberish_vs_math500.png).
Numbers: [`pr_curves.json`](./pr_curves.json).

## Question asked
Is the temporal PR "breathing" pattern a reasoning signal, or a
decoding-mechanical artifact that happens on any autoregressive generation?

## What exists

### Code (all local, committed)
- `gen_prompts.py` — produces the two prompt JSONs on the pod (Qwen tokenizer required).
- `prompts_random.json` — 20 random-token-decoded strings (pre-generated locally, seed=0).
- `prompts_stream.json` — 20 stream-of-consciousness prompts (hand-written).
- `compute_pr.py` — CPU analysis; 7-position PR per condition. Reuses
  `pathway11_h100/exp1_cross_model/analyze.participation_ratio`.
- `plot_curves.py` — overlays the 3 conditions on one figure.
- `run.sh` — orchestrator: gen_prompts → extract×3 → compute_pr → plot.

### Modified
- `pathway11_h100/exp1_cross_model/extract.py`:
  - New model: `qwen25-1.5b` (Qwen2.5-1.5B-Instruct, 28 layers, L19 = 2/3-depth).
  - New flags: `--prompts-file`, `--skip-chat-template`, `--output-dir`.
  - Gibberish path skips `check_correct` and the `mark_done` marker (benchmark tag reflects custom source).

## Three conditions

| Condition | Prompts | Chat template | n | Purpose |
|---|---|---|---|---|
| `random_tokens` | decoded random token IDs (60/prompt) | skip | 20 | null: any token sequence |
| `stream_of_consciousness` | hand-written SoC prompts | apply | 20 | null: fluent non-reasoning |
| `math500_baseline` | MATH-500 (first 50) | apply | 50 | methodology match for PR comparison |

All three use Qwen2.5-1.5B at L19, greedy T=0, max 256 new tokens, positions
`{1, 10, 25, 50, 100, 200, final}`.

## How to reproduce

The pod from this session (`lsuoka6bo8io7m`, H100 SXM, volumeInGb=50) is
**stopped not removed** — disk preserved. Fastest path is to resume it; the
from-scratch recipe is in step 1b below.

### 1a. Resume the existing pod

```bash
runpodctl pod start lsuoka6bo8io7m
runpodctl pod get lsuoka6bo8io7m   # read new ssh ip/port
```

Everything is already on disk: `/workspace/topo-confidence/` with the repo
cloned, pip deps installed, pathway11_h100/{exp1_cross_model,gibberish_control,
config.py} scp'd, the Qwen model cached, and results in gibberish_control/.

### 1b. Create a fresh H100 SXM pod

The previously documented `--imageName`/`--gpuType`/`--volumeInGb` flag
names are **out of date**. Current runpodctl flags:

```bash
runpodctl pod create \
  --image runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04 \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --volume-in-gb 50 \
  --container-disk-in-gb 50 \
  --name gibberish-control \
  --ports "22/tcp,8888/http" \
  --ssh
```

The `ssh_command` in the returned JSON uses the key at
`/home/musicofhel/.runpod/ssh/RunPod-Key-Go`.

### 2. Setup on pod (only for a fresh pod)

```bash
cd /workspace && git clone https://github.com/musicofhel/topo-confidence.git
cd topo-confidence
pip install -q -r pathway8_layerwise/requirements_pathway8.txt
pip install -q accelerate
```

### 3. SCP pathway11 and gibberish_control (not in git remote)

From local. **Gotcha from this session:** scp'ing `exp1_cross_model/` as a
directory also copies its `data/` subtree (≈230 MB of prior-run npz), which
is slow enough that the backgrounded scp command returned while
`exp1_cross_model/extract.py` was still in flight — the first attempt
launched `run.sh` against a missing extract.py. Copy `extract.py` and
`analyze.py` explicitly to side-step the issue:

```bash
POD_KEY=/home/musicofhel/.runpod/ssh/RunPod-Key-Go
POD_IP=<from runpodctl pod get>
POD_PORT=<from runpodctl pod get>

scp -i $POD_KEY -P $POD_PORT \
  pathway11_h100/exp1_cross_model/extract.py \
  pathway11_h100/exp1_cross_model/analyze.py \
  root@$POD_IP:/workspace/topo-confidence/pathway11_h100/exp1_cross_model/

scp -i $POD_KEY -P $POD_PORT \
  pathway11_h100/config.py \
  root@$POD_IP:/workspace/topo-confidence/pathway11_h100/config.py

scp -i $POD_KEY -P $POD_PORT -r \
  pathway11_h100/gibberish_control \
  root@$POD_IP:/workspace/topo-confidence/pathway11_h100/
```

### 4. Run

On the pod:
```bash
cd /workspace/topo-confidence
nohup bash pathway11_h100/gibberish_control/run.sh \
  > pathway11_h100/gibberish_control/run.log 2>&1 &
```

Expected wall-clock: ~10–15 min (first-time Qwen model download ~1 min, 90
total generations at 256 tokens ~ 7–10 min, CPU analysis <1 min).

Progress check:
```bash
ssh -i $POD_KEY -p $POD_PORT root@$POD_IP '
  for d in random stream math500; do
    n=$(ls /workspace/topo-confidence/pathway11_h100/gibberish_control/data/$d/problem_*.npz 2>/dev/null | wc -l)
    echo "$d: $n"
  done
  tail -15 /workspace/topo-confidence/pathway11_h100/gibberish_control/run.log'
```

### 5. Pull results back to local

```bash
scp -i $POD_KEY -P $POD_PORT -r \
  root@$POD_IP:/workspace/topo-confidence/pathway11_h100/gibberish_control/data \
  root@$POD_IP:/workspace/topo-confidence/pathway11_h100/gibberish_control/pr_curves.json \
  root@$POD_IP:/workspace/topo-confidence/pathway11_h100/gibberish_control/gibberish_vs_math500.png \
  root@$POD_IP:/workspace/topo-confidence/pathway11_h100/gibberish_control/run.log \
  pathway11_h100/gibberish_control/
```

### 6. Stop (not remove) the pod

```bash
runpodctl pod stop lsuoka6bo8io7m
```

volumeInGb=50 preserves disk at storage-only rate.

## Results

PR at L19 (2/3-depth), Qwen2.5-1.5B, greedy T=0, max 256 new tokens:

| Position | MATH-500 (n=50) | Stream-of-consc. | Random tokens |
|---|---:|---:|---:|
| 1     | 13.74 | 4.60  | 11.32 |
| 10    | 23.28 | 15.66 | 10.57 |
| 25    | 24.56 | 13.59 | 11.72 |
| 50    | **30.55** | 11.35 | 10.13 |
| 100   | 29.94 | 9.93  | 10.36 |
| 200   | 27.88 | 7.41  | 9.44  |
| final | 22.19 | 9.23  | 8.96  |

MATH-500 accuracy on this 50-problem subset: 10% (5/50). The earlier
48.6% figure came from max_new_tokens=1024 — here we truncate at 256
for speed. This is fine for the PR-shape question (the PR curve does
not depend on final answer correctness, and pos 1–200 numbers track
prior work at L19).

Interpretation by condition:

- **Random tokens — flat (9–12 across all positions).** No rise.
  Feeding the model genuinely meaningless input bypasses whatever
  mechanism produces the rise-then-collapse. This is the cleanest
  null-rejection.
- **Stream-of-consciousness — early peak, low absolute.** Pos 1 starts
  very low (4.6), peaks at pos 10 (15.7), then monotonically decays to
  ~9. No expansion into the 25–30 range MATH hits. Fluent generation
  produces some hidden-state dynamics but with different geometry.
- **MATH-500 — classic breathing.** Peak at pos 50–100 well above
  either gibberish condition, clear collapse by final.

Sample-count caveat for stream: the model terminates stream-of-consc
generations early, so `n` drops (pos 100 n=12, pos 200 n=9). The
absolute PR values at later positions have wider variance than
suggested by a 20-prompt condition. The random-token flat finding
stands independently: random sample counts stay at 20 throughout.

## Expected MATH-500 sanity range (for reruns)

pos 1 PR ≈ 13–25, peak around pos 50–100 in the 25–35 range, final
≈ 20–25. Anything outside this range indicates extraction drift.

## Sanity checks built in

- `compute_pr.py` prints the 7-position PR vector per condition at the end.
- MATH-500 baseline's `n_gen_tokens` distribution is printed — median should
  approach 256 (the cap). If most are far below, Qwen is terminating early —
  compare to prior MATH-500 Qwen runs.
- Random-token prompts are in `prompts_random.json` with meta showing
  tokenizer/n/seed for reproducibility.

## When not to trust the result

- N=20 per gibberish condition is small. A "flat" curve with large pos-to-pos
  noise isn't distinguishable from a muted breathing. If this happens, bump
  N to 50 per condition (re-run the two extract steps — they're resumable
  via the existing done-set logic in extract.py).
- If stream-of-consciousness all produces similar output (20 copies of the
  same completion), the PR will look artificially low everywhere. Check
  `data/stream/problem_000.npz` text fields for diversity.

## Possible follow-ups (not done in this session)

- **Tighten stream-of-consciousness N.** With N=20, sample counts drop to
  n=9 by pos 200 because Qwen terminates stream generations early. Bumping
  to N=50 would lift later-position counts enough for a cleaner comparison.
- **Re-run at 1024-token cap** for a direct MATH-500 apples-to-apples with
  the prior Qwen1.5B work (accuracy would rise from 10% to ~48%; PR curve
  should extend cleanly past pos 200). Current 256-cap was a speed choice.
- **Test 3 cross-model**: repeat on Phi-3-mini (L21) and Llama-3.2-1B (L11)
  to see if the content-dependence result is universal or Qwen-specific.
  Use the same `--prompts-file`/`--skip-chat-template` path in the same
  extract.py.
- **Prefill-inversion under gibberish.** Does prefill PR_correct > PR_incorrect
  hold on stream-of-consc? There's no "correct" label so "correct" would need
  a proxy (e.g., generation length, fluency score). This is speculative; only
  worth it if the main reasoning-specific story needs another supporting leg.
