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

---

## Late additions (2026-05-01): two more surprises

### Per-problem covariance carries the signal F-10 was supposed to refute

EXP-54 (P11-FE116) ran PH on Song-Zhong residualized clouds — strip out
the global mean, the per-position drift, and the per-context centroid,
then ask whether the *deviations* have homological structure that beats
matched-covariance Gaussian noise. The expectation was the same null we
keep finding: real PH ≈ Gaussian PH. Instead the gap *inverted*. Real PH
gives AUROC 0.696; the matched-cov Gaussian null gives 0.762. The
Gaussian beat the real residuals by 7 points.

The mechanism is now obvious in hindsight. The matched-cov sampler
inherits each problem's empirical 1536-d covariance via SVD. PH features
on those Gaussians are predictive because *the covariance matrix
itself* is structured by problem-difficulty in a way the unconditional
1024-tok hidden-state distribution isn't. Real residuals are *smoother*
than rank-matched Gaussians (H_1 cycle counts: 31.3 real vs 87.7 null —
random-ish topology in the null is doing the prediction, not the
geometry). So the takeaway hardens: F-10 isn't "PH falls to a Gaussian
null" anymore. F-10 is "topology adds no signal beyond covariance."
Anything you can compute from the covariance — DoM directions
included — already eats the lunch the topology was supposed to eat.

This shifts where I'd look for upside next. PCA on the per-problem
covariance is now the natural ablation: if a top-k covariance projection
matches DoM AUROC, then the L19 direction is a *covariance principal
component*, and the whole "directions" framing factors through second-
order statistics. If it doesn't match, then DoM is recovering a higher-
order moment we haven't named. Either way, the experimental cost is
small (the cache is already loaded) and the answer is structural.

### Softmax confidence is anti-calibrated on 1.5B at PANL

EXP-53 (the FE110 GPU bundle) included a ConCISE/softmax-confidence
side-test. At the position right before `\boxed{` in the model's own
generation — call it PANL — the top-1 softmax probability is supposed
to read out the model's certainty about the next token. AUROC for
predicting *correctness* from softmax-conf came out 0.4395.

Below 0.5. **Below chance**. When 1.5B-Instruct is *most* confident in
the boxed-answer token, it's slightly *more* likely to be wrong.

A few hypotheses for the mechanism, none confirmed: (1) PANL isn't a
single position — different problems put their `\boxed{` at different
distances from the actual answer-token, so the "confidence" we're
reading is a confounded mixture of formatting confidence and answer
confidence. (2) The instruct tuning teaches the model to commit
syntactically to whatever boxed-answer pattern came out, regardless of
whether the contents match the work above — confidence in the *form*
inversely correlates with correctness on hard problems. (3) The
hardest problems are the ones where the model doubles down (mode-
collapse onto a high-probability wrong answer) while easier problems
have flatter distributions over the right neighborhood.

Whichever it is, the practical implication is sharp: any selective-
prediction or refusal system that keys on softmax confidence at PANL
on this stack is **strictly worse than coin-flipping** on this regime.
The DoM probe at L19 (AUROC 0.7731) is doing real work that softmax
confidence cannot reach. Worth flagging when reviewers ask "why not
just use the model's own probabilities?" — because on this problem,
the model's own probabilities are an inverse signal.

ConCISE c_hat (the weighted "confident/sure/pretty" composite) lands
at 0.589 — weakly positive but well below DoM. So lexical hedging
words are slightly informative; raw softmax peaks are anti-informative.
The two coexist in the same logits.

---

## Late additions (2026-05-02): I ran the PCA ablation

### DoM is the dominant variance direction at L19 prefill

Yesterday's note ended with the obvious next move: PCA on per-problem
covariance, see whether a top-k projection matches DoM. Today the
ablation actually ran (EXP-55, P11-FE291), and the answer is sharper
than I expected.

The unsupervised top eigenvector of the L19 prefill covariance gets
single-feature 5-fold OOF AUROC = **0.7458**. The supervised DoM under
matched single-feature protocol gets **0.7679**. The two directions
have cosine **0.922**. **85% of unit-DoM energy lies in PC1 alone**;
97.6% in the top-10 PCs. PC1 itself carries 14.7% of total L19 prefill
variance — a plurality, by a wide margin over PC2's 11.5% and PC3's
6.8%.

CAST's class-mean-of-means recipe (Lee et al. 2409.05907) — the
labeled alternative to plain unsupervised PCA — gives PC1 AUROC
**0.7458**, cosine **0.922** with DoM, cosine ≈ 1.000 with the
unsupervised PC1. Class-mean centering makes no difference here because
classes are roughly balanced (243 correct / 257 incorrect). Whatever
gains CAST reports in their paper come from the layer-and-threshold
grid search, not from the PCA-PC1 step.

What this does to the F-2 framing. F-2 was a supervised probe finding
("we trained a 1536-d logistic regression on prefill activations and it
predicts correctness at AUROC 0.7731"). That framing implied the probe
discovered hidden structure. It did not. The structure is the dominant
variance direction. *Anyone running PCA on Qwen-1.5B prefills would
find it without labels.* F-2 is simultaneously stronger (the
correctness signal occupies a plurality of L19 variance — one direction
in 1536) and narrower (no covert supervision, no circuit-finding
mystery — it's the most prominent direction at the layer).

What this does to the F-10 framing. Yesterday I wrote that F-10 should
read as "topology adds no signal beyond covariance." Today that becomes
literal: the covariance pathway is *one direction*, PC1, and PH
features layered on the cloud cannot improve on a single mass-mean
projection along it. F-10 is now a tighter, more falsifiable claim:
PH features computed on the [PC2..PC1536] subspace (with PC1 projected
out) should still be null-bound. If they are, F-10's strongest form
holds. If not, there's residual orthogonal-to-PC1 topology I missed.

### The PC9 hint

Most PCs other than PC1 are at chance. PC2..PC8 give single-feature
AUROCs in the 0.48–0.56 band. Then PC9 spikes to **0.658** at variance
share 2.5%. It's the only mid-rank component with real correctness
signal. DoM's coefficient on PC9 is 0.225 (energy 5.0%) — the second-
largest term in the basis decomposition, well above the noise floor.

I don't know what PC9 is. The variance share is too small for it to be
a topic axis (we have ~20 topic distinctions on MATH-500). It could be
a difficulty axis — a two-axis decomposition where PC1 = correctness
mass-direction and PC9 = problem-class easy/hard — or a length axis
aligning with FE448 (prefill seq_len alone gets AUROC 0.799). The
cheap follow-up: two-feature [PC1, PC9] OOF logistic. If it matches
DoM 0.7679, PC9 is exactly the orthogonal component the supervised
probe is recovering. ~10 lines of Python, no new compute.

### Two-feature [PC1, PC9] beats DoM (post-script update, 2026-05-02)

Ran the cheap follow-up. Two-feature [PC1, PC9] OOF logistic AUROC =
**0.7856**. That's *higher* than supervised DoM at 0.7679, by 1.8
percentage points. I had predicted 0.76 ± 0.005 — match-DoM was the
"PC9 closes the gap cleanly" outcome. Overshooting by this much means
something different is going on.

The mechanism: DoM = μ⁺ − μ⁻ is the maximum-mean-difference projection,
not the maximum-AUROC projection. DoM in PC basis weighs PC1 at 0.922
and PC9 at 0.225 — the *natural* weights for separating class means. A
2-feature OOF logistic in (PC1, PC9) is free to re-weight, and what it
learns is a higher relative weight on PC9. PC9 has 2.5% variance share
but 0.658 single-feature AUROC — it's signal-dense per unit variance
(0.658 / 2.5% ≫ 0.7457 / 14.7%). DoM under-uses it because DoM is
measuring mean separation, not classification.

So the basis-decomposition story is *almost* clean but not perfectly
clean. DoM ≈ 0.92·PC1 + 0.22·PC9 + small noise *geometrically*, but
DoM's projection AUROC is sub-optimal — a re-weighted (PC1, PC9) probe
beats it. The supervised DoM is leaving correctness signal on the
table. The really interesting question is whether a fully supervised
1536-d logistic (i.e. the unconstrained F-2 probe at 0.7731) is closer
to the 2-feature 0.7856 than to plain DoM at 0.7679. If yes, the
supervised probe was effectively recovering the (PC1, PC9) re-weighting
all along — which means F-2's "supervised structure" is fully captured
by two named, unsupervised-identifiable directions. That would be
strong support for "everything F-2 does, an unsupervised PCA + small
labeled top-up could do."

### The thing I'd actually want to do with this

Now that PC1 is named, every "DoM" experiment in the project can be
recast as a "PC1" experiment for free. The three pending causal tests
(FE214 noising, FE269 ablation, FE283 layer-zero ablation) become
sharper: do you ablate PC1 or do you ablate "DoM" (which is 92% PC1)?
If PC1 ablation degrades MATH-500 K=1 to chance, the dominant-variance
direction is causal. If not, F-2 stays correlational and we're no
closer to "the model uses this direction" than we were before. The
ablation budget hasn't changed — but the framing of what we're
ablating has.

### F-10 strongest form: PH adds *negative* signal after PC1 removal

Ran the second cheap follow-up: PH features on PC1-residualized L19
trajectory clouds (project FE291 PC1 out of each token, then 5-PH
pipeline, matched-cov Gaussian null). I expected one of two outcomes:
real ≈ null (F-10 holds) or real > null (F-10 narrows). Got neither.

Real PH features OOF AUROC = **0.688**. Matched-cov Gaussian null
AUROC = **0.763**. The Gaussian null *beats* real by 7.4 percentage
points. F-10 sharpens further: not just "PH adds nothing beyond
covariance" — it's "PH adds **negative** signal beyond covariance,
once you remove the dominant direction." The actual residualized
clouds are *topologically simpler* than their cov-matched Gaussians
(27% fewer H1 loops; lower H0/H1 persistence entropy). Real
trajectories live on a low-d submanifold of the residual covariance
ellipsoid; PH features on this submanifold are less correctness-
discriminative than PH on a Gaussian sample with matching cov spectrum.

The null AUROC also *jumped* — from 0.693 (F-10 raw L19 trajectories)
to 0.763 (PC1-residualized). PC1 was apparently absorbing per-problem
cov-spectrum variation roughly correctness-uniformly; removing it
exposed the PC2..PC1536 spectrum whose per-problem eigenvalue pattern
correlates with correctness. The signal worth chasing — if any —
isn't homological, it's spectral. A direct per-problem-cov-eigenvalue
extractor (no Gaussian sampling, no PH) might be the cheapest way to
test "does the residual cov spectrum predict correctness on its own."

This third-branch outcome was not on the design tree at all. The
follow-up I'd actually want is: per-problem cov-spectrum probe (top-k
eigenvalues of each per-problem L19 trajectory cov, after PC1
residualization) → 5-fold OOF logistic. If that beats 0.688 and
matches or exceeds 0.763, the "F-2 is in covariance, not in topology"
story has a third concrete piece.

### Per-problem cov spectrum is itself the strongest probe so far

Ran that follow-up. Top-K log-eigvals of each PC1-residualized per-
problem L19 trajectory covariance, fed to a 5-fold StratifiedKFold OOF
logistic, sweep K∈{5, 10, 20, 50, 100}:

| K | OOF AUROC | CV5 mean ± std |
|---|---|---|
| 5 | 0.7632 | 0.7627 ± 0.024 |
| 10 | 0.7925 | 0.7935 ± 0.012 |
| **20** | **0.7928** | **0.7920 ± 0.027** |
| 50 | 0.7925 | 0.7936 ± 0.029 |
| 100 | 0.7886 | 0.7914 ± 0.021 |

K=20 is the headline (0.7928), but the K=10 number is essentially
identical (0.7925) and the CV5 std is tighter (0.012 vs 0.027), so
I'd treat K=10 as the more honest report. Either way: **the per-
problem cov spectrum on PC1-residualized clouds beats supervised DoM
0.7679 by ~2.5 pp, and beats the 2-feat (PC1, PC9) probe 0.7856 by
~0.7 pp.**

This closes the loop opened in the previous subsection. The 0.763
matched-cov Gaussian null wasn't an exotic property of PH features;
it was the cov spectrum leaking through. A direct extractor not only
recovers it, it goes further. Of the three probes that operate after
PC1 is projected out, the *covariance spectrum* is the strongest:

| Probe (PC1-residualized) | OOF AUROC |
|---|---|
| 5 PH features (real) | 0.6884 |
| Matched-cov Gaussian null on 5 PH | 0.7628 |
| Top-20 log-eigvals (this probe) | **0.7928** |

The single-feature baselines are also informative: the *largest*
eigvalue alone is essentially uninformative (AUROC 0.5223) — that's
mostly cloud size T_i (the n−1 SVD scaling). The third eigval alone
hits 0.7488; eigvals 2..20 collectively carry the signal. This
suggests the discriminative pattern is in the *shape* of the spectrum
tail (rate of eigenvalue decay, effective rank), not the absolute
scale.

What this changes about F-2's story. F-2 was originally "a single
direction (DoM) at L19 prefill predicts correctness." FE291 narrowed
that to "DoM ≈ PC1 of the prefill covariance, plus PC9 trim." Now we
have a third piece: **after removing PC1 from each token, the per-
problem residual covariance still carries correctness signal that no
single direction can capture.** F-2 is at least three things —
(a) the PC1 mean shift, (b) PC9 trim, (c) per-problem second-order
structure orthogonal to PC1. The 0.7928 ≥ 0.7856 ≥ 0.7679 ordering
implies (c) is doing real work beyond (a)+(b).

What this does *not* mean. The cov-spectrum probe is supervised on
prefill labels, same as DoM. Beating DoM doesn't mean we have a new
*unsupervised* signal. It means the PC1+PC9 directional summary is
incomplete: there's exploitable structure in PC2..PC1536's per-
problem eigenvalue magnitudes that a directional probe can't see.
The natural causal companion is now a 2D ablation along the (PC1,
PC9) plane plus a "ablate the residual second-order structure"
intervention (less obvious how to do — perhaps Mahalanobis-whitening
the residualized cloud per-problem before continuation).

For the FE-portfolio narrative: this is now the strongest known L19
probe, beating the 1536-d supervised logistic anchor (0.7731) that
F-2 is built around by ~2 pp. If we accept the 1024-tok 1536-d
number as the supervised ceiling, this 0.7928 is *above* it — which
means either the K=20 spectrum is overfitting at p=20/n=500 (the
CV5±0.027 leaves room) or the supervised 1536-d has untapped
re-weighting room. The two-feature decomposition triangle move from
the prior handoff is now urgent: a properly-regularized full 1536-d
logistic (C=0.01 or 0.1) is the right tiebreaker.

### The triangle resolves: F-2's extra signal is genuinely second-order

Ran the tiebreaker. Full 1536-d L2-reg logistic OOF AUROC, swept over
C ∈ {0.001, 0.01, 0.1, 1.0}:

| C | Full 1536-d OOF AUROC | Notes |
|---|---|---|
| **0.001** | **0.7847** | regularization sweet spot |
| 0.01 | 0.7585 | already overfitting |
| 0.1 | 0.7325 | |
| 1.0 | 0.7211 | severe overfit at p=1536/n=500 |

Best full-1536-d = **0.7847** at C=0.001, which is essentially equal
to the 2-feat (PC1, PC9) 0.7856 within fold noise (CV5 std ≈ 0.027).
The two-feature directional probe is the directional ceiling.

The full triangle:

| Probe | OOF AUROC | Δ vs DoM 0.7679 |
|---|---|---|
| 1-d DoM | 0.7679 | — |
| 2-feat (PC1, PC9) | 0.7856 | +1.77 pp |
| Full 1536-d L2-reg, best C | 0.7847 | +1.68 pp |
| **Top-20 log-eigvals (cov spectrum)** | **0.7928** | **+2.49 pp** |

Reading. The full 1536-d directional probe **does not close the gap**
to the cov-spectrum 0.7928 even with optimal L2 — it tops out at the
2-feat level. So the +0.7 pp the cov-spectrum gets over the 2-feat
is *genuinely second-order*, not just an under-regularized re-weight
of the 1536-d directional signal that a properly regularized linear
probe would also recover.

What this nails down. F-2 decomposes into three additive pieces and
the third is *not* a hidden directional one:

1. **PC1 mean shift** (1-d DoM ≈ 0.92·PC1; AUROC 0.7458 / 0.7679 OOF).
2. **PC9 trim** (2-d (PC1, PC9) ≈ directional ceiling; AUROC 0.7856).
3. **Per-problem residual second-order structure** orthogonal to PC1
   (cov-spectrum top-20 log-eigvals; AUROC 0.7928). This is the part
   that no linear directional probe — single-direction or full
   regularized 1536-d — can recover. The discriminative pattern lives
   in the *shape* of the per-problem covariance spectrum tail, after
   PC1 has been projected out, and a 5-fold OOF logistic on the top
   eigvals is the cheapest way to extract it.

This was the thing I most wanted the triangle to disambiguate, and
it landed in the cleanest possible position — directional ceiling at
2-feat, spectral lift above it. F-2 is now factually a layered claim:
"L19 prefill geometry predicts correctness" splits into a directional
component (PC1+PC9, ≈0.79) and a per-problem second-order component
(residual cov spectrum, +~0.7 pp on top). The natural causal companion
is now a **2D ablation along the (PC1, PC9) plane** for the directional
component, and a **rank-truncate-the-PC1-residualized-covariance**
intervention for the spectral component (zero out everything but the
top-K eigval directions per token before continuation; if that doesn't
disrupt correctness, the spectral signal is observational-only, not
load-bearing).

What this does *not* mean. The cov-spectrum probe is supervised.
Beating the directional ceiling supervised-vs-supervised says
"there's information you can't capture with linear directions on raw
activations." It does not say the model itself uses this information,
or that we have a new unsupervised signal. The causal companion is
the right next test.

What this *does* mean for the validate_claims invariant: 169 PASS
internal once the new pca-full-lr-best-auroc claim is wired (was 168
after the cov-spectrum claim). 213 total tracked.



## Late additions (2026-05-03): the synthesis cascade

Three docs landed: SYNTHESIS, NOVELTY_AUDIT, APPLICATIONS. Three weeks of
experiments compressed into 1,231 lines that an outside ML engineer could
read in 30 minutes and walk away with a reasonable picture of what's true,
what's new, and what's deployable.

What surprised in writing them.

The novelty audit was the hardest. Walking each F-N through
`research-graph/query.py novelty` with the 220-paper graph already loaded
forced an honest accounting: F-2 is a **parallel discovery** — Zhu's
"The LLM Already Knows" (2509.12886) found the same prefill direction on
Qwen-VL-7B at AUROC 0.85, ours at 0.7731 on Qwen-2.5-1.5B. We didn't cite
it as parallel; we cited it as motivating. The paper had been sitting in
the graph for weeks. The synthesis exposed it. Lesson: novelty audits
don't *discover* parallelism — they *force you to admit* it. The graph
was already telling us; we'd just been reading it as supportive instead
of as parallel discovery.

What the synthesis makes possible. The decomposition triangle table —
DoM 0.7679 → 2-feat 0.7856 → cov-spectrum 0.7928 → full-1536-d 0.7847 —
is the cleanest result of the program. It belongs *first* in any
external presentation, not buried in an evidence section. Putting it in
the README headline (after this session's intent-layer pass) was the
right move; visitors landing on the repo see the factorization before
they see the framing.

What the synthesis closes. Three open questions that the FE291 cascade
resolved and that the synthesis surfaces explicitly: (a) the cov-spectrum
lift is genuinely second-order (FE882 closed it); (b) F-10 strengthens
on residualization (FE880 closed it); (c) CAST PC1 ≈ supervised DoM
(FE291 closed it, now framed as "the supervised correctness direction
is unsupervised-identifiable"). Each one was implicit in the result JSONs;
the synthesis makes them explicit.

What the synthesis doesn't close. Eight open questions, four of which
were filed as new H-N entries in this pass (H-698 cross-arch, H-699
penultimate-token PR, H-700 PC9 mechanism, H-701 rank-truncate-cov-
spectrum causal companion). The other four — answer-vocab null, 7B PR
inversion mechanism, short-CoT breathing, the 12 untriaged Discord
papers — are queued in HYPOTHESES + handoffs. None block the synthesis
from being a faithful current-state snapshot.

What this means for the program's narrative arc. Three weeks ago the
project was named topo-confidence and the headline was "persistent
homology of token clouds predicts correctness at AUROC 0.796." Today
the headline is "L19 prefill direction (DoM ≈ PC1) predicts at 0.7731
+ a cov-spectrum lift to 0.7928, and PH adds *negative* signal beyond
covariance." The name no longer describes the result. That's not an
embarrassment — that's the program working. The synthesis is the artifact
that makes the renaming honest.
