# Test 2 — No-CoT control for PR breathing

**Status:** complete. Pod `lsuoka6bo8io7m` ran 2026-04-24 22:36–22:40 UTC,
stopped (not removed). Wall-clock ~4 min (no-CoT gens are 1–10 tokens).

**Headline:** the no-CoT system prompt works — the model produces a median of
**2 generated tokens** vs 255 for CoT — so the expected PR trajectory is
physically absent. This is **inconclusive in the way the test was framed**
("breathing over fewer tokens") because there are not enough tokens to
observe inflation-and-collapse. It is suggestive that breathing is bound to
trajectory length, not compressed into the first 1–2 tokens.

## Numbers

| Position | CoT baseline (n=50, median 255 tok) | No-CoT (n=50, median 2 tok) |
|---|---:|---:|
| 1     | 13.74 | **15.78** |
| 10    | 23.28 | — (n=2) |
| 25    | 24.56 | — (n=0) |
| 50    | 30.55 | — (n=0) |
| 100   | 29.94 | — (n=0) |
| 200   | 27.88 | — (n=0) |
| final | 22.19 | **19.80** |

No-CoT generation-length distribution: min=1, p25=1, median=2, p75=3, max=10.
Accuracy: no-CoT 4/50 (8%), CoT 5/50 (10%) — CoT baseline is capped at 256
tokens so both are underperforming a 1024-token Qwen1.5B run.

## How to read this

The framing — "does PR inflate/collapse over fewer tokens, or stay flat?" —
assumed the model would still generate a multi-token trajectory, just shorter.
Instead, it collapsed to 1–3 tokens: "answer only" prompts degenerate to
emitting a boxed number (or failing to). With 2 positions of PR per prompt,
there is no curve to inspect.

What we can still compare:
- **Pos 1**: no-CoT (15.78) is slightly **higher** than CoT (13.74). Consistent
  with "when the model isn't planning a long reasoning trajectory, its first-
  token hidden state is closer to its settled final-answer geometry."
- **Final**: no-CoT (19.80) ≈ CoT-pos-10 (23.28). The end of a 2-token no-CoT
  generation sits near where CoT is still in its rising phase.
- **The peak (pos 50–100, PR ≈ 30) is reachable only when the model is doing
  CoT.** No-CoT provides no token positions at that range.

So: the **breathing peak is specifically a CoT phenomenon** — not because
"CoT is special content" (the gibberish control already established content-
dependence on reasoning-like structure) but because breathing requires a
trajectory and CoT provides one. Take away the trajectory and you take away
the curve.

## Question asked

"Does breathing happen without chain-of-thought? If PR still inflates and
collapses but over fewer tokens, the breathing tracks generation length. If
it stays flat, the breathing is specifically about the CoT reasoning process."

## Answer

**The two options as stated don't quite apply** — the model does not produce
enough tokens to show either behavior. What the data does show:

1. Pos-1 PR under no-CoT is ~2 PR units higher than pos-1 under CoT. The
   first-token state is already more "spread out" when the model is in
   answer-only mode.
2. There's no breathing because there's no trajectory.

**Practical conclusion:** breathing is CoT-driven *at the trajectory level* —
turning off CoT turns off the whole curve, not compresses it. To cleanly
distinguish "tracks length" from "tracks CoT-qua-reasoning", the right
follow-up prompt is a *short-but-still-multi-token* instruction like
"Explain in one sentence and give the answer" which should produce ~20–40
tokens.

## What exists

- `run.sh` — orchestrator: extract → compute_pr → plot.
- `compute_pr.py` — 7-position PR for no-CoT vs CoT baseline at
  `../gibberish_control/data/math500/`.
- `plot_curves.py` — overlay of the two curves.
- `data/nocot/problem_*.npz` — per-prompt extractions.
- `pr_curves.json` — numeric PR values.
- `nocot_vs_cot.png` — figure.

## Modified

- `pathway11_h100/exp1_cross_model/extract.py`: added `--system-prompt`
  flag; `format_prompt` takes an optional `system_prompt` kwarg (defaults
  to `SYSTEM_PROMPT`). No change when the flag is omitted.

## How to reproduce

Pod `lsuoka6bo8io7m` is stopped (volumeInGb=50 preserves disk). Gotcha:
pip packages are on the container disk, not the volume — after a stop/start
cycle `transformers` needs reinstalling:

```bash
runpodctl pod start lsuoka6bo8io7m
# scp the updated extract.py and the new no_cot_control/ dir
ssh <pod> '
  cd /workspace/topo-confidence
  pip install -q -r pathway8_layerwise/requirements_pathway8.txt
  pip install -q accelerate
  nohup bash pathway11_h100/no_cot_control/run.sh \
    > pathway11_h100/no_cot_control/run.log 2>&1 &'
```

The no-CoT run is ~3 min of GPU work (1–10 tokens × 50 prompts) once the
Qwen model is already cached (it is).

## Follow-ups (not done in this session)

- **Short-CoT condition**: prompt "Solve this in one sentence, then give the
  answer in \boxed{}." Expected: 20–40 tokens. If PR breathes over that
  span (peak at pos 10–20, collapse at final), breathing tracks length;
  if PR stays ~15 throughout, breathing specifically tracks extended-CoT
  depth.
- **1024-token no-CoT rerun**: would not change the finding — the model is
  terminating at EOS after 2 tokens, not hitting the cap. Same data.
- **Cross-model no-CoT**: on Phi-3-mini and Llama-3.2-1B to confirm the
  "no-CoT → 1-3 tokens" behavior is universal and not Qwen-specific.
