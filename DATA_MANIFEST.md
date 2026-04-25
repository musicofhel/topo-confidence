# DATA_MANIFEST.md — cached activation inventory

Every cached activation file and its schema. Use this to know what's on disk
and what would need re-extraction on a fresh machine.

**Total on-disk footprint:** ~79 GB of cached activation NPZ files + ~4 GB of
result JSONs + auxiliary caches. All NPZ binaries are gitignored (`.gitignore`
includes `*.npz`, `*.npy`, `*.pkl`, `*.pt`).

**How to regenerate.** Every cached file is produced by a committed Python
script (see the "Produced by" column). All GPU extractions run on RunPod H100
SXM at ~$2.99/hr.

---

## Canonical all-layer per-token caches (Pathway 11)

These are the master caches that every downstream analysis reads from.

### 1.5B MATH-500 — `pathway8_layerwise/data/math500/`

| Field | Value |
|---|---|
| Size on disk | **19 GB** |
| File count | 500 × `problem_NNN.npz` + 1 × `manifest.json` |
| Model | Qwen2.5-1.5B-Instruct |
| Benchmark | MATH-500 |
| Generation config | **T=0 greedy, max_new_tokens=1024** |
| Label scheme | **1024tok** — 243/500 correct |
| Extracted by | `pathway8_layerwise/extract_math500.py` (reused as Pathway 11 Stage 2) |
| Pod | `y687b9z2dgukcj` (stopped) |
| Date extracted | 2026-04-23 → 2026-04-24 |

**Per-file NPZ schema** (example problem_000.npz):
- `states`: `(29, 399, 1536)` float16 — 29 layers (embedding + 28 transformer) × per-generation-token × hidden dim 1536. 399 here is the generation length including prefill.
- `d2h_attn_entropy`: `(28,)` float64 — attention entropy per layer.
- `text`: str — generated text.
- `correct`: bool — label at 1024-tok answer extraction.
- `mean_logprob`: float64.

### 7B MATH-500 — `pathway11_h100/data/math500_7b/`

| Field | Value |
|---|---|
| Size on disk | **42 GB** |
| File count | 500 × `problem_NNN.npz` + manifest |
| Model | Qwen2.5-7B-Instruct |
| Benchmark | MATH-500 |
| Generation config | **T=0 greedy, max_new_tokens=1024** |
| Label scheme | **1024tok** — 366/500 correct |
| Extracted by | `pathway11_h100/stage1_extract_math500_7b.py` |
| Critical caveat | `d2h_attn_entropy` is **all-zeros** because `capture_attention=False` was passed (7B GQA + eager fallback is fragile). Re-run with `capture_attention=True` if attention entropy is needed. |
| Pod | `y687b9z2dgukcj` |

**Per-file NPZ schema:**
- `states`: `(29, T, 3584)` float16 — T varies per problem.
- `d2h_attn_entropy`: `(28,)` float64 — **all zeros** (see caveat).
- `text`, `correct`, `mean_logprob`.

### BBH 3×250 — `pathway8_layerwise/data/bbh/`

| Field | Value |
|---|---|
| Size on disk | **18 GB** |
| File count | 750 × `problem_NNN.npz` split into 3 subdirs + manifest |
| Model | Qwen2.5-1.5B-Instruct |
| Benchmarks | `tracking_shuffled_objects_seven_objects` (250), `logical_deduction_seven_objects` (250), `web_of_lies` (250) |
| Generation config | T=0, max_new_tokens=1024 |
| Label scheme | BBH own labels — per-subset accuracy: 12.4%, 6.0%, 54.0% |
| Extracted by | `pathway8_layerwise/extract_bbh.py` (reused as Pathway 11 Stage 4a) |
| Pod | `y687b9z2dgukcj` |

**Subdirectories:**
```
pathway8_layerwise/data/bbh/
├── manifest.json
├── tracking_shuffled_objects_seven_objects/problem_NNN.npz  (n=250)
├── logical_deduction_seven_objects/problem_NNN.npz          (n=250)
└── web_of_lies/problem_NNN.npz                              (n=250)
```

Schema identical to 1.5B MATH-500 (states, d2h_attn_entropy, text, correct, mean_logprob).

### K=8 self-consistency — `pathway11_h100/data/k8_selfconsistency/`

| Field | Value |
|---|---|
| Size on disk | **13 MB** |
| File count | 500 × `problem_NNN.npz` + manifest |
| Model | Qwen2.5-1.5B-Instruct |
| Benchmark | MATH-500 |
| Generation config | **T=0.7, K=8 samples, max_new_tokens=1024** |
| Extracted by | `pathway11_h100/stage3_k8_selfconsistency.py` |
| Why small | Only L19 captured per-sample, not all layers — deliberate for size |

**Per-file NPZ schema:**
- `L19_samples`: `(8, 1536)` float16 — L19 activation per sample (pooled? per-token?) — check stage3 code.
- `texts`: `(8,)` object — 8 sample texts.
- `correct`: `(8,)` bool — 8 per-sample labels.

---

## Cross-architecture caches (Exp 1, Pathway 11)

### Phi-3-mini-4k — `pathway11_h100/exp1_cross_model/data/phi3mini/`

| Field | Value |
|---|---|
| Size on disk | **166 MB** |
| File count | 500 × `problem_NNN.npz` + manifest |
| Model | Phi-3-mini-4k-instruct (3.8B, 32 layers, 2/3-depth = L21) |
| Generation config | T=0 greedy, max_new_tokens=1024, `attn_implementation='eager'` |
| Labels | 224/500 correct (44.8%) |
| Extracted by | `pathway11_h100/exp1_cross_model/extract.py --model phi3mini` |
| Pod | `y687b9z2dgukcj` |

**Per-file NPZ schema** (reduced-capture version — no full per-token all-layer):
- `prefill_all_layers`: `(33, 3072)` float16 — prefill-end state at every layer (embedding + 32 transformer).
- `final_all_layers`: `(33, 3072)` float16 — final-token state at every layer.
- `twothirds_positions`: `(7, 3072)` float16 — L21 activation at 7 generation positions.
- `positions_actual`: `(7,)` int32 — generation token indices sampled (e.g., 1, 10, 25, 50, 100, 200, last).
- `twothirds_layer_idx`: `21` (int).
- `n_gen_tokens`: int — total generated.
- `text`, `correct`, `mean_logprob`.

### Llama-3.2-1B — `pathway11_h100/exp1_cross_model/data/llama32_1b/`

| Field | Value |
|---|---|
| Size on disk | **65 MB** |
| File count | 500 × `problem_NNN.npz` + manifest |
| Model | `unsloth/Llama-3.2-1B-Instruct` (1B, 16 layers, 2/3-depth = L11), SDPA attention |
| Generation config | T=0, max=1024 |
| Labels | 126/500 correct (25.2%) |
| Extracted by | `pathway11_h100/exp1_cross_model/extract.py --model llama32-1b` |

**Per-file NPZ schema** (same structure as Phi-3, different dims):
- `prefill_all_layers`: `(17, 2048)` float16.
- `final_all_layers`: `(17, 2048)` float16.
- `twothirds_positions`: `(7, 2048)` float16.
- `twothirds_layer_idx`: `11`.
- Same `positions_actual`, `n_gen_tokens`, `text`, `correct`, `mean_logprob`.

---

## Breathing-control caches (Tests 2 & 3, Pathway 11)

### Gibberish (Test 3) — `pathway11_h100/gibberish_control/data/`

| Subdir | Size | n | Purpose |
|---|---|---|---|
| `random/` | 3.1 MB | 20 | Random-token prompts (seed=0) |
| `stream/` | 3.0 MB | 20 | Stream-of-consciousness hand-written prompts |
| `math500/` | 7.7 MB | 50 | MATH-500 first-50 baseline for comparison |

- Model: Qwen2.5-1.5B (28 layers, L19 = 2/3-depth).
- Generation config: **T=0, max_new_tokens=256** (note: shorter than canonical Pathway 11 — speed choice for the controls).
- Extracted by: `pathway11_h100/exp1_cross_model/extract.py` + `--prompts-file` and `--skip-chat-template` flags.
- Pod: `lsuoka6bo8io7m` (stopped, volumeInGb=50).

Schema identical to Phi-3 / Llama caches (prefill_all_layers 29×1536,
final_all_layers 29×1536, twothirds_positions 7×1536, etc.), at 1.5B
dimensions.

### No-CoT (Test 2) — `pathway11_h100/no_cot_control/data/nocot/`

| Field | Value |
|---|---|
| Size | 7.0 MB |
| File count | 50 × `problem_NNN.npz` + manifest |
| Generation config | T=0, max=256, **"Answer with just the number, no explanation" system prompt** |
| Accuracy | 4/50 (8%) — note CoT baseline is 5/50 at 256-tok cap |
| Gen length | median **2 tokens** (min 1, max 10) — the model terminates near-immediately |

**Per-file NPZ schema:**
- `prefill_all_layers`: `(29, 1536)` — same as gibberish.
- `final_all_layers`: `(29, 1536)`.
- `twothirds_positions`: **`(2, 1536)`** — only 2 generation positions available because median gen length is 2 tokens. Hence the "test was inconclusive as framed" caveat in the no-CoT HANDOFF.

### `m7b_prefill.npz` / `m15b_prefill.npz` — `pathway11_h100/prefill_inversion/cache/`

Convenience caches for fast replay of Exp 3 analyses without re-walking 42 GB.

| Field | Value |
|---|---|
| Size | 7.6 MB total |
| Purpose | fp16 flat arrays of prefill + final for all 500 problems × 2 scales |

Schema:
- `prefill`: `(500, 3584)` float16 — 7B prefill-end activation at L19 (for 7B file; 1.5B file is `(500, 1536)`).
- `final_tok`: `(500, 3584)` float16 — final-token activation at L19.
- `correct`: `(500,)` bool — labels.
- `seq_len`: `(500,)` int64 — generation token count.

Built by `pathway11_h100/prefill_inversion/common.py` from the master caches.
Can be rebuilt in ~2 min on CPU from the full per-token NPZs.

---

## Historical pathway caches (pre-Pathway 11, 256-tok labels)

These are still on disk but correspond to the superseded 256-tok label
scheme. Use only for reproducing historical claims.

- `pathway5/track_a/phase_a1/gsm8k_trajectories.npz` — GSM8K trajectories
  (Pathway 5 Track A). ~1 MB.
- `pathway5/track_a/phase_a1/trajectory_checkpoint.npz` — intermediate
  checkpoint.
- `pathway5/track_a/phase_a3/completion_trajectories.npz` — T=0.7 completions.
- `data/experiment1_v2/trajectories.npz` — Pathway-1 era raw trajectories.

All gitignored.

---

## What would need re-extraction on a fresh machine

Ordering by cost if you need a total rebuild from scratch:

| Step | Cost | Wall time | Notes |
|---|---|---|---|
| 1. Stage 1 — 7B all-layer MATH-500 @ 1024 | ~$22 | ~7 h H100 | `pathway11_h100/stage1_extract_math500_7b.py` |
| 2. Stage 2 — 1.5B all-layer MATH-500 @ 1024 | ~$6 | ~2 h H100 | `pathway8_layerwise/extract_math500.py` |
| 3. Stage 4a — BBH 3×250 @ 1024 | ~$10 | ~3 h H100 | `pathway8_layerwise/extract_bbh.py` |
| 4. Stage 3 — K=8 self-consistency | ~$2 | ~40 min H100 | `pathway11_h100/stage3_k8_selfconsistency.py` |
| 5. Exp 1 Phi-3 extraction | ~$3 | ~40 min H100 | `exp1_cross_model/extract.py --model phi3mini` |
| 6. Exp 1 Llama-3.2 extraction | ~$2 | ~30 min H100 | `exp1_cross_model/extract.py --model llama32-1b` |
| 7. Test 3 gibberish | ~$0.75 | ~15 min H100 | `gibberish_control/run.sh` |
| 8. Test 2 no-CoT | ~$0.15 | ~4 min H100 | `no_cot_control/run.sh` |
| **Total** | **~$46** | **~14 h** | all 1024-tok extractions |

The post-hoc analyses (Exp 2, 2b, 3, cross-model analysis, gibberish analysis)
are all local CPU — ~15-30 min each, no GPU or dollar cost.

---

## Data hygiene checklist

- ✅ All binary artifacts gitignored.
- ✅ All numerical results committed as JSON.
- ✅ Each cache has a `manifest.json` alongside problem NPZs.
- ⚠️  Pathway 11 is **not in the git remote** at github.com/musicofhel/topo-confidence — the `pathway11_h100/` directory is local-only. Distribute via scp or a private branch.
- ⚠️  Two pods (`y687b9z2dgukcj` with volume=0, `lsuoka6bo8io7m` with volume=50) may have been stopped; check `runpodctl pod list` before assuming disk is preserved. Volume-0 pods lose disk on stop.
