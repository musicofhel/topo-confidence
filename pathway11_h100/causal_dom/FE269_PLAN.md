# FE269 — Causal-intervention test of the L19 prefill DoM (fast, full 3-arm)

> **Status:** APPROVED implementation plan (2026-06-09). Saved in-repo so it survives session/context
> handoffs. Mirror of `~/.claude/plans/reactive-waddling-toast.md` (which may be overwritten by future
> plan-mode sessions). This is the durable copy — execute from here.

## Context

**Why this experiment.** topo-confidence has established that a *prefill* L19 direction-of-means
(DoM) on Qwen-2.5-1.5B-Instruct predicts MATH-500 K=1 correctness (AUROC 0.7731; baseline acc
48.6%, 243/500), and that this generalizes cross-architecture (Phi-3 0.81, Llama-3.2 0.75). The
whole program now hinges on ONE open question that decides the finish line: **is that direction a
causal _lever_ (steer it → change correctness) or just a correlational _diagnostic_ (read it →
route/abstain)?** Per Arditi et al. 2406.11717, "a direction encodes X" requires *ablate→removes-X*
AND *add→induces-X*, with general capability preserved. FE269 runs exactly that test.
- CAUSAL → pursue steering (H-1) as a model-improvement lever.
- DIAGNOSTIC → ship FE19 verification-routing / selective prediction (71.6% answered-acc @ 50%
  coverage), now backed by cross-arch generalization.

**Fresh-eyes corrections to the prior handoff (verified during planning):**
1. The branch `max-depth-retriage-2026-04-28` is **already pushed to GitHub and up-to-date** — all
   git-tracked code/results are safe. The handoff's "push branch (local-only)" was wrong.
2. The only unbacked data is the **79GB gitignored NPZ caches**, which are *reproducible* and *not
   used by FE269* (FE269 re-generates fresh; the 42GB bulk is the 7B cache it never touches). User
   chose to rsync them to `/mnt/c` anyway as background cost-insurance — non-blocking.
3. Cost: handoff said "~$2/1hr"; FE269's own spec says 4h. **Batched generation** (safe here — see
   below) compresses the full 3-arm run to ~1 pod-hour ≈ $2.5.
4. **Correctness landmine (verified from installed source):** in `transformers 4.57.6`,
   `Qwen2DecoderLayer.forward` returns a **bare Tensor, not a tuple** — the usual `output[0]` hook
   pattern silently breaks. Hook must be shape-defensive AND the pod pins `transformers==4.57.6`.
5. **Off-by-one (verified from source):** `hidden_states[19]` = **output of `model.model.layers[18]`**
   (since `hidden_states[0]` = embeddings). "Ablate at L19" → hook `layers[18]`; "every layer" →
   hook `layers[0..27]` with the same `r̂` (Arditi recipe).

## Decisions taken
- **Scope:** full 3 arms (ablation + addition sweep + capability), compressed via batching.
- **Backup:** commit/push small untracked files + background rsync of 79GB to `/mnt/c`.
- **Compute:** RunPod **H100 SXM** (honor user memory over CLAUDE.md "local-only"; user explicitly
  chose RunPod for the causal test). User's own $20-capped key, to be burned post-session.

---

## Step 0 — Backup (do first, mostly background; ~2 min foreground)

1. **Commit + push the small at-risk untracked files** on `max-depth-retriage-2026-04-28`:
   `HANDOFF_2026-06-09.md`, the 2 triage briefs (2026-06-09), and
   `pathway11_h100/exp1_cross_model/{compute_dom_auroc.py,dom_auroc_results.json}`. (npz stays
   gitignored.) Then `git push`.
2. **Background rsync** the 79GB caches to the Windows drive (NTFS-friendly flags):
   ```
   mkdir -p /mnt/c/topo-confidence-backup
   rsync -a --no-perms --no-owner --no-group --inplace --size-only --info=progress2 \
     /home/musicofhel/topo-confidence/pathway11_h100/ \
     /home/musicofhel/topo-confidence/pathway8_layerwise/ \
     /mnt/c/topo-confidence-backup/   # run in background; does NOT gate the GPU work
   ```

## Step 1 — Compute the direction locally (CPU, instant)

Create `pathway11_h100/causal_dom/compute_rhat.py`:
- Load `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz` → `prefill (500,1536) f16`,
  `correct (500,) bool`. **Upcast to float32** before arithmetic (f16 mean of 500 vecs loses precision).
- `r = mean(P[correct]) − mean(P[~correct])`; `r_hat = r/‖r‖` (FULL-data, NOT the 5-fold OOF — this
  is a fixed intervention hypothesis, not a generalization estimate; cf. `phase2_prefill_dom.py`).
- Assert `correct.sum()==243`; save `r_hat.npy` (float32, 1536, ~6KB) + provenance JSON (`‖r‖`,
  n_correct, cos(full-data, fold-0) > 0.9 sanity).

## Step 2 — Build the intervention harness (new dir `pathway11_h100/causal_dom/`)

Reuse, don't reinvent: `pathway8_layerwise/extraction_utils.py::load_model()` (bf16, sdpa,
pad=151643, left-pad — **never fp16**), and `extract_math500.py`'s `check_correct` / `_extract_boxed`
/ prompt formatting (system prompt: "Please reason step by step, and put your final answer within
\boxed{}.", chat template, greedy do_sample=False, max_new_tokens=1024, eos=[eos, <|im_end|>]).

**`intervention.py`** — `DirectionIntervention(r_hat, mode, alpha)` with a shape-defensive
forward-hook + context manager:
```
def _edit(h):                       # h: (batch, seq, 1536) bf16
    r = self.r.to(h.dtype)
    if mode=="ablate": return h - (h @ r)[..., None] * r   # project DoM OUT, per-position
    if mode=="add":    return h + alpha * r
    return h                                                 # "off"
def _hook(module, inp, out):
    if isinstance(out, tuple): return (_edit(out[0]),) + out[1:]
    return _edit(out)                                        # bare Tensor (4.57.6)
register on model.model.layers[k] for k in layer_set; remove() via context manager (always).
```
KV-cache consistency: hooking *layer outputs* means the edited residual stream feeds the next
layer's K/V on every forward (prefill + each decode step), so the cache reflects edited activations
— no manual cache surgery. (Record the layer-0/embedding edge as a pre-registered design note.)

**Batching = the speed lever (safe here).** The edit is per-position-independent and padded
positions are attention-masked, so batched generation is correct. Use **batch 32** (1.5B on 80GB
H100 is tiny). This turns a 500-problem MATH pass from ~25 min → ~3–4 min. Keep left-padding +
attention_mask. The batched mode=off gate (Step 4 S1) must still reproduce 48.6% ±~8 problems; if
not, drop batch and retry (padding-numerics divergence).

**Eval drivers:**
- `eval_math500.py` — modes off/ablate/add, configurable layer-set, batched.
- `eval_gsm8k.py` — GSM8K test (1319), K=1 greedy max_new_tokens=512, reuse `check_correct` +
  a `_last_number()` fallback; batched.
- `eval_mmlu.py` — `cais/mmlu` all/test, **1000-Q fixed-seed subsample**, **answer-letter logprob**
  (single forward/Q, hook still fires) — ~1–2 min batched. No lm-eval-harness (won't compose with hooks).
- `decision.py` — loads result JSONs, applies the pre-registered thresholds below, emits verdict.

## Step 3 — Pod bootstrap

H100 SXM, base `runpod/pytorch:2.4.0`. `git clone` + `checkout max-depth-retriage-2026-04-28`.
`pip install -r pathway11_h100/causal_dom/requirements_fe269.txt` (new file: torch>=2.4,
**transformers==4.57.6**, math-verify, datasets, numpy<2.0, scikit-learn). Ship `r_hat.npy` (~6KB)
to the pod (runpodctl send / base64). **No 79GB caches needed** — weights download from HF.

## Step 4 — Execute (single pod, batched, gate-first; target ≤1 pod-hour ≈ $2.5)

`run_fe269.sh`, stage-based with `.done` markers (mirrors `runpod_pathway11.sh`), hard wall-clock
guard so a hang still leaves S1/S2 results:

| Stage | What | Mode / layers | ~Runtime (batch 32) |
|---|---|---|---|
| S0 | preflight (GPU, imports, r̂ present, version pin) | — | ~10 min (incl. model dl) |
| S1 | **baseline-repro GATE**: MATH-500 K=1 | off / all | ~4 min — MUST hit 48.6% |
| S2 | ablation MATH (L19-only) + GSM8K off+ablate baselines | ablate / {18} | ~10 min |
| S3 | ablation MATH (all-layer) + MMLU off+ablate | ablate / all | ~8 min |
| S4 | addition α-sweep on 257 incorrect, α∈{0.5,1,2,4}×‖r‖ | add / {18} | ~12 min |
| S5 | `decision.py` → verdict JSON | — | <1 min |

Pull back only the small `causal_dom/results/*.json`; commit them (npz stays ignored).

## Pre-registered decision criteria (commit before running; encoded in `decision.py`)

Baselines measured at mode=off (MATH must = 48.6% gate; G0=GSM8K, M0=MMLU). ΔMATH = 48.6 − ablated.
- **LEVER:** ΔMATH ≥ 15 pts AND GSM8K drop ≤ 5 AND MMLU drop ≤ 5 (necessity+specificity), AND
  addition flips ≥15% of the 257 incorrect at some α with originally-correct retention ≥85% (sufficiency).
- **DIAGNOSTIC:** MATH survives ≥45% under ablation (ΔMATH ≤ ~4) → reads out, doesn't cause.
- **GENERAL DAMAGE:** ΔMATH ≥ 15 BUT GSM8K and/or MMLU also drop ≥15 → broke the model broadly,
  not correctness-specific. (Most likely failure mode of all-layer ablation — why S2 L19-only runs first.)
- **INCONCLUSIVE:** ΔMATH 5–15 with ≤5 capability drop = "weak lever, needs replication."

## After the verdict (next-next, not this session unless time permits)
- LEVER → H-1 per-position DoM steering. DIAGNOSTIC → ship FE19 routing/selective-prediction.
- Either way: bookkeeping — `research-graph/update_status.py` for FE269 + the cross-arch FE; mark
  F-2 "cross-architecture prefill test" control SATISFIED in FINDINGS.md; add validate_claims
  entries (Phi-3 0.8089, Llama 0.7458); regenerate NEXT_EXPERIMENTS.md.
- **Verify the P10-v2 steering discrepancy** before trusting the addition arm's novelty: memory says
  "direction-rotation refuted fixed-vector steering" but `pathway10_v2.md` is marked NEVER EXECUTED.

## Files to create (all under `pathway11_h100/causal_dom/`)
`compute_rhat.py`, `intervention.py`, `eval_math500.py`, `eval_gsm8k.py`, `eval_mmlu.py`,
`decision.py`, `run_fe269.sh`, `requirements_fe269.txt`.

## Verification
1. **Local:** `compute_rhat.py` asserts `correct.sum()==243`, `‖r‖>0`, cos(full,fold0)>0.9.
2. **Gate (S1):** batched mode=off MATH ∈ [47.0%, 50.2%]. If outside → plumbing bug, halt.
3. **Hook sanity:** with `mode="ablate"`, assert post-hook `|h_edited · r̂| ≈ 0` on a captured
   activation (projection actually removed); with `mode="add"`, assert `(h+αr̂)·r̂ − h·r̂ ≈ α`.
4. **End-to-end:** `decision.py` emits a single LEVER/DIAGNOSTIC/GENERAL_DAMAGE/INCONCLUSIVE verdict
   from the result JSONs against the pre-registered thresholds.
5. **Backup:** `ls -la /mnt/c/topo-confidence-backup/` shows both pathway dirs; `git status` clean
   on the small files + `git push` confirmed on the remote branch.

## Cost / risk
- ~1 pod-hour H100 SXM ≈ **$2.5** of the $20 cap → ample headroom for FE283/FE19 follow-ups.
- Risk: batched gate fails to reproduce 48.6% → fall back to batch 8/4 (still ~2–3× faster than
  batch 1). Risk: all-layer ablation = general damage → S2 L19-only already isolates the specific signal.

## Key reused code (paths)
- `pathway8_layerwise/extraction_utils.py::load_model()` — model loader (bf16/sdpa/pad/left-pad).
- `pathway8_layerwise/extract_math500.py` — `check_correct`, `_extract_boxed`, prompt formatting, decode config.
- `pathway11_h100/prefill_gated_compute/phase2_prefill_dom.py` — DoM reference (STEERING_LAYER=19, OOF protocol to contrast against).
- `pathway11_h100/prefill_inversion/cache/m15b_prefill.npz` — source for r̂ (prefill 500×1536 f16 + correct bool).
- `pathway11_h100/runpod_pathway11.sh` — stage/.done orchestrator pattern to mirror in `run_fe269.sh`.
