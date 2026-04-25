# PERSPECTIVES.md — reflective notes

Not a paper. Not a summary. A thinking-out-loud document about what the
three weeks of topo-confidence experiments actually told us, what we thought
we saw that wasn't there, and where the shape of the problem seems to be
pointing.

---

## What surprised us most

### The truncation artifact

Every quantitative claim about cross-benchmark transfer, cross-scale
asymmetry, and the "20.8% baseline" was load-bearing on a setting we
never questioned: `max_new_tokens=256`. For MATH-500 the 1.5B needs more
tokens to complete a chain-of-thought that ends in a boxed answer. At 256
tokens, two-thirds of the correct trajectories got cut off before
`\boxed{...}` was emitted, and the answer extractor dutifully marked them
wrong. The true accuracy is 48.6%, not 20.8% — more than 2× higher.

What's striking isn't that this was a bug — bugs happen. It's that for a
full two weeks of experimentation the number 20.8% anchored every design
decision. "A +3 pp accuracy lift" meant a *meaningful intervention at 20.8%
baseline* and *noise at 48.6% baseline*. Entire gating strategies were
calibrated against imaginary headroom. Whole sections of the Pathway 9
SUMMARY talking about the 0.050 "length confound" turned out to be
describing the behavior of a hand-engineered feature pipeline on a
truncated label distribution — the raw residual-stream direction doesn't
have that confound at all.

The lesson I keep coming back to: the most dangerous numbers in a research
program are the ones everyone stops looking at. The unquestioned baseline
is the unquestioned ground.

### Prefill knows best

Before extracting the 1024-tok data, I would have guessed — mostly on
priors from Chain-of-Embedding, from the whole "late-layer final-token
probe" literature — that the correctness signal would be *strongest at the
end*: after the model has done its reasoning, the final hidden state
should know whether the answer came out right.

The opposite is true. On 1.5B, **prefill L19 DoM AUROC = 0.7731**,
**final-token DoM AUROC = 0.7186**. On 7B the gap widens to ~0.876 vs
~0.77. The prompt alone — before a single generation token — is a sharper
predictor of correctness than the completed reasoning trace.

Even more surprising: **cos(prefill, final) ≈ 0.046**. These aren't just
different magnitudes of the same signal. They're orthogonal directions.
Whatever "can I solve this?" is, it has nothing to do with "did I solve
this?" as a geometric quantity in residual-stream space.

This is the finding I'd most want to test on more models / benchmarks. If
it holds generally, it reframes how you'd do confidence estimation:
decide before you compute, not after.

### Dimensional breathing universality

When the first 1.5B temporal PR curve came in — low at prefill (~20),
peaking at ~67 mid-generation, collapsing to ~8 at final — I thought it
was Qwen-specific. Then it reproduced on the 7B at a higher absolute
scale (~88 peak, ~6 final). Then it reproduced on Phi-3-mini (peak ~100,
final ~18) and Llama-3.2-1B (peak ~115, final ~8). Different
architectures, different scales, same three-phase shape.

And then the gibberish control: random tokens gave a *flat* PR ≈ 10. The
expand-then-collapse isn't a property of autoregressive decoding — it
needs content structure. Reasoning-like content. Stream-of-consciousness
narrative produces a muted, different-shaped curve.

I don't know what this pattern *is* — below — but it's clearly an
inference-time structural phenomenon common to transformer reasoning.
It's the closest thing to a finding in this project that feels like it
might connect to something deeper than "this trick works."

---

## What we were wrong about

### The 7B verifier story

The Pathway 10 v1 plan had Direction A: use the 7B's 0.818 residual-stream
correctness signal as a verifier for 1.5B outputs. It motivated E4
distillation. The whole "use big model's self-knowledge to help small
model" framing.

It was all the truncation artifact. At 1024 tokens, 7B → 1.5B transfer is
**0.717**, 1.5B → self is **0.719**. No advantage. The 7B's sharper
self-prediction (0.782) doesn't transfer when the 7B is used to judge a
1.5B trajectory — because what the 7B knows about *its own* problems is
not the same thing it can say about *the 1.5B's* problems.

This one stung because I'd spent real thought designing a distillation
experiment around an asymmetry that didn't exist.

### The PH story

"Topological confidence" as a project framing had been alive since ATT
Phase 5 showed last-layer PH entropy predicts LLM correctness. The repo
is named topo-confidence. The initial commits in April were literally
"Add null hypothesis testing and expand to 13 topological features."

Pathway 9 Exp 3a v2 quietly killed the topology framing: 5 raw PH
features hit AUROC 0.690 vs a rank-matched Gaussian null at 0.693.
*Matched covariance structure fully explains the signal.* There is no
topology-specific predictive content.

What I originally called "topological structure in the hidden-state
cloud" is "covariance eigenvalues + their spacings." The information
content is linear. The CORAL pipeline's headline 0.7961 AUROC was
entirely geometry (cosines, norms, velocities). The 9 PH/geometry Tier A
features work *because of their geometry inputs*, not because PH is
doing topological work.

This connects to something predictable from Tuci et al.'s EoS /
Sharpness-Dimension framework: a trained model's residual stream is
approximately Gaussian in its top directions. PH summaries of such
clouds will match a Gaussian null.

### The E1 steering plan

v2 of Pathway 10 was organized around E1: fit a probe, use the weight
vector as a steering direction. A clean, tested (ITI) recipe.

It would have produced nothing. The DoM direction rotates continuously
through generation — cos with final-token DoM stays in [0.00, 0.19] at
every intermediate position, only climbing to 0.37 by position 200. An
injection of `α · w_final` at position 15 is energy in approximately the
wrong subspace.

The pre-cached Pathway-2 steering vector that was supposed to be our
shortcut has cos ≈ 0.05 with the current L19 DoM. Geometrically
unrelated.

I would not have known this from a "run the probe and steer" workflow.
The direction-rotation property has to be *measured* before steering, or
steering is a random perturbation.

---

## What does dimensional breathing actually mean?

I don't know. But here's what I think I see.

**It's a commitment dynamic.** At prefill, the model holds a cloud of
possible "approaches" to the problem — the activations span a ~20-dim
effective subspace. As generation proceeds, the model explores through
these alternatives, and its covariance *expands* into a higher-dim space
(peak ~67 for 1.5B, ~88 for 7B). Then at the point of answer commitment,
the covariance collapses sharply (to ~8 / ~6) — as if the model pulls
itself onto a low-dim answer manifold.

The correct-vs-incorrect asymmetry at final is suggestive: correct
trajectories collapse *harder* (1.5B: 4.3 vs 8.5; 7B: 2.6 vs 6.8). When
the model is right, it commits tightly to a rank-2 answer geometry. When
it's wrong, the hidden state stays more diffuse — as if multiple
candidate answers are still "active" and no single commitment happened.

This story has testable predictions:
- Tighter final-PR should predict confidence within correct answers.
- "Pathological" D-bucket problems (K=1 right, K=8 wrong) have final PR
  that looks like A-bucket (4.94 vs A's 4.06) — consistent with the
  greedy path committing correctly, and T=0.7 sampling variance breaking
  the commitment. Supported by the data.
- If breathing is commitment dynamics, short-CoT generation should show
  a compressed-shape version; no-CoT should just skip straight to
  commitment. The no-CoT control was inconclusive because the model
  didn't generate a trajectory, not because breathing was absent. A
  short-CoT prompt (target 20–40 tokens) is the next test.

**Alternative reading.** It's not about commitment — it's about
*exploration noise in the residual stream*. Mid-generation, the model
holds multiple partial derivations and its hidden state accumulates
"orthogonal noise" from all of them. By the final token, the relevant
content is localized to whichever few tokens encode the answer, and the
covariance collapses because only a few dimensions carry information.

I can't distinguish these two stories on current data. A probe at
different depths during generation — "is the answer readable off this
hidden state?" — would discriminate them.

---

## Why are prefill and final-token directions orthogonal?

cos(prefill_DoM, final_DoM) ≈ 0.046 is the number that bothers me most.

Plausible stories:

1. **Different circuits.** The model has a "will I solve this?" circuit
   (reads the prompt, predicts difficulty) and a "did I solve this?"
   circuit (reads the completed derivation, predicts correctness). No
   reason they share an axis in residual-stream space — different
   attention heads route different information.
2. **Representation rotation through generation.** Most representation
   engineering assumes directions are stable across tokens. This finding
   is evidence they're not — the relevant correctness axis drifts as
   content accumulates, and by the final token it's in a different
   location.
3. **Accumulation vs transformation.** The prefill state encodes "how
   many reasoning steps will fit in the budget" + "have I seen this
   problem type before"; the final-token state encodes "what did the
   reasoning produce." Both predict correctness but for entirely
   different reasons.

If story 1 is right, there are literally two circuits and we could try
to localize them via attribution (e.g., direct logit attribution on the
probe weights). If story 2 is right, steering requires per-position
direction banks. If story 3 is right, ensembling prefill + final should
give a disproportionate lift — and the two-feature LR at ~0.794 (vs
0.7731 prefill alone) suggests the lift exists, but is modest.

A simple experiment that would distinguish: **on the training subset of
the probe fit, ablate each attention head and watch which heads' removal
hurts prefill_DoM vs final_DoM.** If the hurt-head sets are disjoint,
it's different circuits.

---

## The D-bucket phenomenon

36/500 = 7.2% of MATH-500 problems: greedy K=1 gets them right, K=8
majority gets them wrong. These aren't just "unlucky T=0.7 draws." Over
8 samples at T=0.7 they consistently fail to majority-vote correct,
which means there's a structural reason the distribution at T=0.7 is not
centered on the correct answer.

What they share: **lowest prefill PR of any bucket (14.49)**, intermediate
generation length (527 tokens), final-token PR that "looks correct"
(4.94, close to A's 4.06).

The geometric story: D problems live in a small, concentrated subspace
of prefill activation space. Their greedy-decoded reasoning path is the
"easy ramp" through a compressed region of state space — deterministic
T=0 decoding falls onto it reliably, but any variance knocks samples off.

What this tells us about sampling-based inference: **sampling hurts
when the correct answer requires a specific reasoning path that multiple
alternative paths lead away from.** It's not random noise; there's a
structural attractor that only the greedy trajectory lands on.

This is testable. If you hold the T=0 greedy path fixed and perturb the
model slightly (e.g., a small activation perturbation at L19), you
should see the greedy-correct trajectory flip to wrong for D problems
much more readily than for A problems. I didn't run this.

The other thing D tells us: **local k-NN features won't find it.**
Exp 2b showed per-problem local PR (k=20 NN in prefill space) is flat
across all buckets (A=12.66, B=12.80, C=12.97, D=12.84). The 14.49
D-bucket group PR is a *collective* property — when you take the PR of
the whole D cluster together, it's low; but each D point's 20 nearest
neighbors are mostly non-D, so the locality measure averages out. This
is a genuinely interesting failure of k-NN methods to detect
cluster-scale structure — an argument for density-aware features (LOF,
isolation forest trained on D specifically) over geometric ones.

---

## Where does this connect to the broader field?

Four active threads I see.

**1. Test-time compute scaling.** The Wang 2022 / Brumm 2510.07364 line on
self-consistency compute allocation. Our finding: the Wang 3× assumption
(K=10 ≈ K=1 + 3× lift) is reasonable on MATH-500 × 1.5B (K=8 gives +6.4pp
at 8× compute). But monotonic prefill-gating doesn't beat uniform K; the
useful thing is selective prediction (refuse-and-spend). This bridges to
the refusal literature (Knowing When to Quit, 2604.18419).

**2. Representation engineering.** ITI (2306.03341), ReDeEP (2410.11414),
ALS (2509.18116). Our direction-rotation finding is a negative result for
the fixed-vector paradigm but a motivator for per-position-direction
methods. "The probe direction at test time != the probe direction at
train time" is a latent assumption everywhere; ours is the first number
I've seen measuring that rotation on a reasoning task.

**3. Prefill / prompt-only signals.** "The LLM Already Knows" (2509.12886)
argues prompt-only features predict output quality. Our 0.7731 prefill
AUROC on 1.5B and ~0.876 on 7B are consistent with that framing — but
crucially, we show the prefill signal is *orthogonal* to the final-token
signal, not just an earlier snapshot of it. If this replicates, it
reshapes what "post-hoc calibration" and "pre-generation routing" should
be architecturally.

**4. EoS / Sharpness Dimension.** Tuci et al. (2604.19740) describes
training-time Hessian dynamics that expand-then-collapse around edge of
stability. Our inference-time covariance PR is an analog: during CoT it
expands, at answer it collapses. Whether these share a formal
underlying structure or are merely analogous by shape is a real open
question (see H-4 in HYPOTHESES.md). It'd be the closest thing to a
theoretical contribution the project has produced.

---

## What's the strongest criticism of the breathing finding?

The null-rejection story rests on a specific counterfactual: "breathing
could be AR-mechanics." The gibberish control rejected that — random
tokens give flat PR.

But the sharper criticism is: **"you're measuring covariance of the
token-wise hidden states across a 500-problem cohort — of course
covariance grows as content diversifies across problems."** That is,
position-0 has all 500 problems look like "prompt end + first reasoning
token," low covariance. By position 50, the 500 problems have diverged
into 500 different reasoning states, covariance is naturally higher.
At final-token they've re-converged into the space of 500 *answer
tokens*, which come from a small vocabulary — covariance low again.

Is the final-PR collapse driven by "the answer tokens are literally from
a small vocabulary"? We haven't cleanly ruled it out. Tests that would
address it:

- Compute PR over the *penultimate* token rather than the last emitted
  one. If the collapse is still there, it's not purely answer-token
  vocabulary.
- Compute PR over the residual stream *minus the answer-token
  embedding*. The unembedding dimension is much less than the residual
  stream dimension, but it's not orthogonal.
- Compare PR collapse on problems where the answer is a large number
  (many unique tokens) vs small number (few unique tokens). If large-answer
  problems have less collapse, it's vocabulary-dominated.

None of these ran. The breathing finding needs this second null rejection
before a skeptical reader accepts it.

---

## If I had unlimited compute, what's the one experiment?

**The per-position DoM bank + steering sweep on 1.5B MATH-500.**

Not because it's the most elegant — but because it resolves the single
most-ambiguous point in the program:

- If per-position direction-matched steering moves accuracy, then the
  hidden-state signal is *actionable at inference*, and the program's
  "improve small models via hidden-state understanding" goal has a
  concrete recipe.
- If per-position steering doesn't move accuracy, then the
  0.77 prefill AUROC is a *diagnostic* signal (useful for refusal and
  compute gating) but not a *lever*. The program closes at "good
  selective predictor, not a steerable target."

Either answer reshapes what the project is about. I can imagine the paper
from either outcome. The one open question I can't write around is this
one.

Cost estimate: ~1 H100-day for the DoM bank extraction (N=200 problems ×
28 token positions × all 28 layers), ~2 H100-days for the steering
sweep. ~$200. It's a week of calendar time.

I wish we'd run it.
