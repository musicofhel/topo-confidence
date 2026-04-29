"""One-shot backfill of the 2026-04-28 Claude Desktop triage session.

Captures 8 experiments (H-15 through H-22, FE nodes P11-FE15 through P11-FE22)
and 6 source arxiv papers into the research graph and the narrative docs.

Modeled on seed_future_experiments.py: single Neo4j session, idempotent via MERGE.
Diff-then-apply for HYPOTHESES.md / PAPER_INDEX.md so the user can review before
mutating narrative state.

Usage:
    python backfill_2026-04-28_session.py             # dry-run: prints diffs
    python backfill_2026-04-28_session.py --apply     # writes graph + docs

After --apply: run `python generate_next_experiments.py` to regenerate
NEXT_EXPERIMENTS.md (the script does this for you).

Format-discovery sanity: parses one existing H-1 entry from HYPOTHESES.md to
verify the template still matches before any write.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")
TODAY = date.today().isoformat()


# ---------------------------------------------------------------------------
# Source papers (Paper stubs — blogs are recorded as text in FE.trigger only)
# ---------------------------------------------------------------------------

PAPERS: list[dict[str, Any]] = [
    {
        "arxiv_id": "2604.18805",
        "title": "AI scientists produce results without reasoning scientifically",
        "year": 2026,
        "relevance": (
            "Large-scale eval (25k+ runs) — LLM agents ignore evidence in 68% "
            "of traces, rarely revise from refutation. Motivates length-band PR "
            "control as a self-applied refutation test for our breathing finding."
        ),
        "tags": ["agent-evaluation", "scientific-reasoning", "refutation"],
        "status": "graphed",
    },
    {
        "arxiv_id": "2509.26560",
        "title": "Estimating Dimensionality of Neural Representations from Finite Samples",
        "year": 2025,
        "relevance": (
            "Bias-corrected participation-ratio estimator (Marchenko-Pastur). "
            "Direct methodological threat: our cross-model PR comparisons span "
            "P/Q ratios 0.14-0.33, exactly the bias-significant regime."
        ),
        "tags": ["dimensionality", "participation-ratio", "methodology"],
        "status": "graphed",
    },
    {
        "arxiv_id": "2604.22271",
        "title": "How LLMs Detect and Correct Their Own Errors: Internal Confidence Signals",
        "year": 2026,
        "relevance": (
            "Identifies post-answer-newline (PANL) token as a second-order "
            "confidence signal orthogonal to logprobs (cos=0.007). Our prefill "
            "DoM is the pre-hoc analog (cos=-0.06 with final-token DoM). "
            "Motivates PANL-equivalent + C_exact-routing experiments."
        ),
        "tags": ["confidence-estimation", "error-correction", "prefill"],
        "status": "graphed",
    },
    {
        "arxiv_id": "2405.07987",
        "title": "The Platonic Representation Hypothesis",
        "year": 2024,
        "relevance": (
            "Argues representations across vision and language models converge "
            "to a shared statistical model of reality. Our cross-architecture "
            "breathing universality (F-1) is evidence applied to inference-time "
            "dynamics. Motivates the kernel alignment fluctuation experiment."
        ),
        "tags": ["representation-learning", "convergence", "platonic"],
        "status": "graphed",
    },
    {
        "arxiv_id": "2604.24712",
        "title": "When Prompt Under-Specification Improves Code Correctness",
        "year": 2026,
        "relevance": (
            "Prompt over-specification triggers memorized-but-wrong solutions. "
            "Sampling analog of our D-bucket (K=1 right, K=8 wrong, F-7). Their "
            "Table 8 taxonomy provides categories for D-bucket pathology types."
        ),
        "tags": ["prompt-engineering", "code-generation", "d-bucket"],
        "status": "graphed",
    },
    {
        "arxiv_id": "2604.22709",
        "title": "Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought",
        "year": 2026,
        "relevance": (
            "Discrete latent CoT — 11.6x token compression at comparable accuracy. "
            "Tests whether breathing (F-1, F-5) is a property of verbalized "
            "reasoning or of reasoning itself."
        ),
        "tags": ["reasoning-efficiency", "abstract-cot", "breathing"],
        "status": "graphed",
    },
]


# ---------------------------------------------------------------------------
# FutureExperiments
# ---------------------------------------------------------------------------

FUTURE_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "id": "P11-FE15",
        "pathway_id": "P11",
        "description": (
            "Length-band PR control on MATH-500. Split 500 problems into three "
            "generation-length bands (<25th, 25-75th, >75th percentile). Compute "
            "final-token PR per band, separately for correct vs incorrect. "
            "Refutation test for whether the asymmetric-collapse finding "
            "(F-4) survives within length-matched subsets."
        ),
        "rationale": (
            "Ríos-García 2604.18805 anti-pattern: 68% of agent runs ignore "
            "disconfirming evidence. Self-applied refutation test for F-4. "
            "If correct/incorrect PR ratio approaches 1.0 within bands, the "
            "collapse is a length-mixing artifact, not a real signal."
        ),
        "trigger": "Ríos-García 2604.18805 motivates self-applied refutation tests for headline findings.",
        "status": "READY",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "20min CPU on cached Stage 2 NPZs",
        "roi_score": 9,
        "depends_on_findings": ["F-1", "F-4"],
        "would_update_findings": ["F-1", "F-4"],
        "triggered_by_papers": ["2604.18805"],
    },
    {
        "id": "P11-FE16",
        "pathway_id": "P11",
        "description": (
            "Implement Marchenko-Pastur bias correction for participation ratio. "
            "Recompute (a) temporal PR curve for Qwen 1.5B + 7B, (b) correct vs "
            "incorrect PR at prefill + final token, (c) cross-model peak-PR "
            "comparisons. Critical question: does asymmetric collapse (F-4) and "
            "cross-model breathing magnitude (F-1) survive correction?"
        ),
        "rationale": (
            "Chun et al. 2509.26560 show naive PR is biased by P/Q ratio. Our "
            "P/Q values: Qwen 1.5B 0.33, Qwen 7B 0.14, Llama 0.24, Phi-3 0.16 — "
            "all in the bias-significant regime. Cross-model PR comparisons at "
            "fixed n=500 are exactly the scenario the paper warns against."
        ),
        "trigger": "Chun et al. 2509.26560 published bias-corrected estimator with code.",
        "status": "READY",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "1h CPU on cached Stage 2 NPZs",
        "roi_score": 9,
        "depends_on_findings": ["F-1", "F-4"],
        "would_update_findings": ["F-1", "F-4"],
        "triggered_by_papers": ["2509.26560"],
    },
    {
        "id": "P11-FE17",
        "pathway_id": "P11",
        "description": (
            "RoPE de-rotation of the DoM direction. For 1.5B per-token L19 "
            "activations, undo the RoPE rotation at each token position "
            "(deterministic from model config), recompute the temporal DoM "
            "AUROC curve and cosine-to-final table. If cos(DoM_t, DoM_final) "
            "rises above 0.8 after de-rotation, the rotation is mechanical "
            "and E1 fixed-vector steering becomes viable again."
        ),
        "rationale": (
            "Puranik (Jane Street) shows all valid positional encodings are "
            "matrix groups exp(M·tau). RoPE applies constant-frequency rotations "
            "to dimension pairs. F-3 (orthogonal prefill/final DoM) might be "
            "purely a RoPE consequence rather than a computational orthogonality. "
            "L19 is post-attention/MLP so partial explanation (cos 0.2 to 0.6) "
            "is still informative."
        ),
        "trigger": (
            "Jane Street blog (Puranik) on group theory and positional "
            "encodings frames RoPE as a deterministic per-position rotation."
        ),
        "status": "READY",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "1h CPU on cached per-token activations",
        "roi_score": 8,
        "depends_on_findings": ["F-3"],
        "would_update_findings": ["F-3"],
        "triggered_by_papers": [],
    },
    {
        "id": "P11-FE18",
        "pathway_id": "P11",
        "description": (
            "PANL-equivalent correctability gate. In K=1 greedy generations, "
            "locate the post-answer-newline token per problem, extract L19 "
            "activations, compute DoM AUROC predicting (a) K=1 correctness, "
            "(b) K=8 majority correctness, (c) B-bucket membership (K=1 wrong, "
            "K=8 right — recoverable), (d) D-bucket membership (K=1 right, K=8 "
            "wrong — pathological, F-7). Compare against prefill (F-2) and "
            "final-token AUROC for the same four targets."
        ),
        "rationale": (
            "Kumaran et al. 2604.22271 PANL signal predicts error correction "
            "(AUROC 0.986 Gemma, 0.961 Qwen 7B) — orthogonal to verification "
            "logprobs (cos=0.007). Our prefill DoM is the pre-hoc analog "
            "(cos=-0.06 with final DoM). The B/D-bucket question prefill "
            "couldn't answer (Exp 2 monotonic gating fails because B lives at "
            "mid-confidence) might be answerable via PANL-equivalent."
        ),
        "trigger": "Kumaran et al. 2604.22271 introduces the PANL signal.",
        "status": "READY",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "30min H100 (2060 sufficient)",
        "roi_score": 9,
        "depends_on_findings": ["F-2", "F-7"],
        "would_update_findings": ["F-7"],
        "triggered_by_papers": ["2604.22271"],
    },
    {
        "id": "P11-FE19",
        "pathway_id": "P11",
        "description": (
            "C_exact verification routing — Desktop's highest-leverage proposal. "
            "For 500 MATH-500 problems: take K=1 greedy answer, prompt model to "
            "verify its own answer, extract PANL-equivalent activation, use as "
            "routing signal. Route verification-failed problems to K=8 sampling, "
            "keep verification-passed at K=1. Compare against (a) pure C_infer "
            "(uniform K=8), (b) pure C_exact (verify-then-correct, no sampling) "
            "at matched compute. Total avg K = 3-4."
        ),
        "rationale": (
            "Combines Rybin compute-allocation framework (C_train / C_infer / "
            "C_exact decomposition) with Kumaran PANL signal. E2 prefill-gated "
            "compute allocation failed because B-bucket lives at mid-confidence "
            "and sits within C_infer. Verification routing is C_exact, where "
            "the framework predicts the win. If C_exact > C_infer at matched "
            "compute, the entire selective-prediction story (F-8) shifts from "
            "predict-difficulty-at-prefill to verify-and-route-failures."
        ),
        "trigger": (
            "Rybin compute-allocation blog provides C_infer / C_exact framework. "
            "Kumaran 2604.22271 provides the PANL routing signal."
        ),
        "status": "READY",
        "blocked_by": None,
        "priority": "CRITICAL",
        "estimated_cost": "30min H100 + verification pass per problem",
        "roi_score": 10,
        "depends_on_findings": ["F-2", "F-7", "F-8"],
        "would_update_findings": ["F-8"],
        "triggered_by_papers": ["2604.22271"],
    },
    {
        "id": "P11-FE20",
        "pathway_id": "P11",
        "description": (
            "Cross-model kernel alignment fluctuation during inference. Compute "
            "mutual k-NN alignment between all 6 model pairs (Qwen 1.5B, "
            "Qwen 7B, Phi-3, Llama) at each of 7 temporal positions using "
            "cached 2/3-depth activations. Predicts: alignment peaks at "
            "low-PR moments (prefill, final token), reaches min at "
            "mid-generation peak PR. Extends Platonic Representation Hypothesis "
            "from static convergence (Huh et al.) to dynamic convergence."
        ),
        "rationale": (
            "Huh et al. 2405.07987 measure representational convergence via "
            "mutual k-NN alignment of kernel matrices. Our F-1 cross-arch "
            "breathing universality is evidence for the Platonic Hypothesis "
            "applied to inference-time dynamics — not just static "
            "representations but their temporal evolution converges."
        ),
        "trigger": "Huh et al. 2405.07987 Platonic Representation Hypothesis (existing — re-saved 2026-04-28).",
        "status": "READY",
        "blocked_by": None,
        "priority": "MEDIUM",
        "estimated_cost": "2h CPU, all data cached",
        "roi_score": 7,
        "depends_on_findings": ["F-1"],
        "would_update_findings": ["F-1"],
        "triggered_by_papers": ["2405.07987"],
    },
    {
        "id": "P11-FE21",
        "pathway_id": "P11",
        "description": (
            "D-bucket attention entropy at prefill. Extract attention weights at "
            "prefill for the 36 D-bucket and 207 A-bucket problems, compute "
            "entropy of attention across prompt regions, compare. Mechanism "
            "test: D-bucket (F-7) might rely on a single reasoning pathway "
            "triggered by a specific prompt feature, making it fragile to "
            "sampling perturbation."
        ),
        "rationale": (
            "Akli et al. 2604.24712 Figure 3 shows that on HumanEval "
            "(structurally simple) 60-86% of attention concentrates on the "
            "description, while LiveCodeBench (structurally rich) distributes "
            "attention across description, I/O format, and sample I/O. D-bucket "
            "fragility may be the same single-specification fragility expressed "
            "at sampling time rather than prompt time."
        ),
        "trigger": "Akli et al. 2604.24712 attention-distribution analysis.",
        "status": "READY",
        "blocked_by": None,
        "priority": "MEDIUM",
        "estimated_cost": "small GPU (~30 min), depends on whether Stage 2 saved attention",
        "roi_score": 6,
        "depends_on_findings": ["F-7"],
        "would_update_findings": ["F-7"],
        "triggered_by_papers": ["2604.24712"],
    },
    {
        "id": "P11-FE22",
        "pathway_id": "P11",
        "description": (
            "Abstract-CoT compressed breathing curve. Fine-tune Qwen2.5-1.5B "
            "with Abstract-CoT on MATH-500, extract per-token L19 activations "
            "during abstract reasoning, compute the temporal PR curve. Compare "
            "peak PR, inflation rate, and collapse asymmetry against natural-"
            "language CoT. If flat or monotonically decreasing PR, breathing "
            "(F-1, F-5) is specific to verbalized reasoning, not reasoning "
            "itself — i.e., dimensional inflation is the cost of verbalization."
        ),
        "rationale": (
            "Ramji et al. 2604.22709 achieve 11.6x token compression via "
            "discrete latent CoT. Cleanly separates reasoning-as-computation "
            "from reasoning-as-verbalization. Heaviest experiment in the "
            "queue (training pipeline + GPU week)."
        ),
        "trigger": "Ramji et al. 2604.22709 release of Abstract-CoT method.",
        "status": "BLOCKED",
        "blocked_by": "Awaiting Abstract-CoT training pipeline release from authors.",
        "priority": "MEDIUM",
        "estimated_cost": "~1 week H100 for fine-tune + extraction",
        "roi_score": 6,
        "depends_on_findings": ["F-1", "F-5"],
        "would_update_findings": ["F-1", "F-5"],
        "triggered_by_papers": ["2604.22709"],
    },
]


# ---------------------------------------------------------------------------
# HYPOTHESES.md — H-N markdown blocks (template matches H-1..H-14 exactly)
# ---------------------------------------------------------------------------

NEW_HYPOTHESES_MD = """\
### H-15: Length-band PR control validates asymmetric-collapse claim
**Priority:** HIGH (refutation test, user-prioritized 2026-04-28)
**Motivated by:** Ríos-García 2604.18805 (agents ignore disconfirming
evidence in 68% of traces). Self-applied refutation test for F-4.
**Test:** Split 500 MATH-500 problems by generation length (<25th,
25-75th, >75th percentile). Compute final-token PR per band, separately
for correct vs incorrect. Uses cached Stage 2 NPZs.
**Requires:** CPU only, ~20 min. Zero dollar.
**Would change:** If correct/incorrect PR ratio approaches 1.0 within
length-matched bands, asymmetric collapse is a length-mixing artifact and
F-4 retracts. If ratio holds within each band, F-4 is corroborated against
the strongest available within-data refutation.
**Blocks:** nothing. Cheap, decisive validity check.

### H-16: Marchenko-Pastur bias correction preserves breathing magnitude
**Priority:** HIGH (methodological, user-prioritized 2026-04-28)
**Motivated by:** Chun et al. 2509.26560. Our cross-model P/Q ratios
(Qwen 1.5B 0.33, Qwen 7B 0.14, Llama 0.24, Phi-3 0.16) are all in the
bias-significant regime; cross-model peak-PR comparisons at fixed n=500
are the scenario the paper warns against.
**Test:** Implement bias-corrected PR estimator (Section 4 of the paper
or their reference code). Recompute (a) temporal PR curve for Qwen 1.5B
and 7B, (b) correct vs incorrect PR at prefill and final token,
(c) cross-model peak-PR comparison. Report naive and corrected
side-by-side.
**Requires:** CPU only, ~1 h. Uses cached Stage 2 NPZs.
**Would change:** If correct-collapses-harder (F-4) and cross-arch
breathing magnitude (F-1) survive correction, the headline numbers are
robust. If correction shifts ratios materially, narrative claims need
re-stating with the corrected estimator.
**Blocks:** nothing.

### H-17: DoM rotation during generation is RoPE-mechanical, not computed
**Priority:** HIGH (could revive E1 steering)
**Motivated by:** Puranik (Jane Street) shows positional encodings are
matrix groups exp(M·tau); RoPE applies deterministic constant-frequency
rotations. F-3 (cos(prefill, final) ≈ 0.046) might be a RoPE consequence
rather than a computational orthogonality.
**Test:** For 1.5B per-token L19 activations, undo RoPE rotation at each
position (rotation angles deterministic from model config), recompute
temporal DoM AUROC curve and cosine-to-final table.
**Requires:** CPU only, ~1 h. Uses cached per-token L19 activations and
the model config.
**Would change:** If cos(DoM_t, DoM_final) rises above 0.8 after
de-rotation, F-3 becomes a positional-encoding artifact and the
direction-rotation argument against E1 steering collapses — H-1
per-position bank loses its motivation, and a single un-rotated DoM is
sufficient for steering. Partial explanation (cos 0.2 to 0.6) still
informative about how much of F-3 is mechanical.
**Blocks:** H-1 (per-position steering bank) becomes unnecessary if this
confirms; reframes if it partially confirms.

### H-18: Post-answer-newline activation predicts B/D-bucket membership
**Priority:** HIGH
**Motivated by:** Kumaran et al. 2604.22271 (PANL = orthogonal
second-order confidence signal, AUROC 0.986). Our prefill DoM is the
pre-hoc analog. The B/D-bucket question (F-7) — recoverable vs
pathological — sits at mid-confidence where prefill gating fails (Exp 2).
**Test:** In K=1 greedy generations, locate the post-answer-newline
token per problem, extract L19 activations at that position. Compute DoM
AUROC predicting (a) K=1 correctness, (b) K=8 majority correctness,
(c) B-bucket membership (K=1 wrong, K=8 right), (d) D-bucket membership
(K=1 right, K=8 wrong). Compare against prefill (F-2) and final-token
AUROC.
**Requires:** Model loaded (2060 sufficient), ~30 min. Uses cached
per-token trajectories from Stage 2.
**Would change:** If PANL-equivalent predicts B-bucket > prefill does,
post-hoc correctability gating becomes feasible — route only genuinely
recoverable problems to K=8. F-7 gains a mechanism.
**Blocks:** H-19 (verification routing builds on this signal).

### H-19: C_exact verification routing beats C_infer scaling at matched compute
**Priority:** CRITICAL (Desktop's highest-leverage call, user-prioritized 2026-04-28)
**Motivated by:** Rybin compute-allocation framework
(C_train / C_infer / C_exact decomposition) + Kumaran 2604.22271 PANL
signal. E2 prefill-gated compute (pure C_infer) failed because B-bucket
lives at mid-confidence; the right axis is C_infer ↔ C_exact.
**Test:** For 500 MATH-500 problems: take K=1 greedy answer, prompt the
model to verify its own answer (verify-then-correct paradigm), extract
PANL-equivalent activation, use as routing signal. Route
verification-failed problems to K=8 sampling, keep verification-passed at
K=1. Compare against (a) pure C_infer (uniform K=8), (b) pure C_exact
(verify-then-correct, no sampling) at matched compute. Total avg K ≈ 3-4.
**Requires:** Model loaded (2060 sufficient), ~30 min plus verification
pass per problem.
**Would change:** If C_exact routing beats C_infer routing at matched
compute, the selective-prediction story (F-8) shifts from predict-
difficulty-at-prefill to verify-after-generation-and-route-failures. This
is the result that converts the program from "good selective predictor"
to "actionable inference recipe."
**Blocks:** nothing downstream, but it's the highest-impact lever the
program has access to without GPU-week investment.

### H-20: Cross-model kernel alignment fluctuates during inference
**Priority:** MEDIUM
**Motivated by:** Huh et al. 2405.07987 (Platonic Representation
Hypothesis). Our F-1 cross-arch breathing universality is evidence for
the hypothesis applied to inference-time dynamics — not just trained
representations but their temporal evolution converges.
**Test:** Compute mutual k-NN alignment between all 6 model pairs (Qwen
1.5B, Qwen 7B, Phi-3, Llama) at each of the 7 temporal positions using
cached 2/3-depth activations. Predicts: alignment peaks at low-PR
moments (prefill, final token), reaches min at mid-generation peak PR.
**Requires:** CPU only, ~2 h, all data cached.
**Would change:** If alignment fluctuates as predicted, F-1 extends
from "all transformers breathe" to "all transformers explore differently
but compress to the same place" — a temporal version of the Platonic
hypothesis. If alignment is flat or anti-correlated with PR, F-1 is
universality of *shape* but not of *content* (each model breathes through
its own subspace).
**Blocks:** nothing.

### H-21: D-bucket has narrower prefill attention than A-bucket
**Priority:** MEDIUM
**Motivated by:** Akli et al. 2604.24712 single-specification fragility:
HumanEval prompts concentrate 60-86% attention on the description while
LiveCodeBench distributes across description, I/O format, sample I/O.
D-bucket fragility (F-7) may be the same effect at sampling time.
**Test:** Extract attention weights at prefill for the 36 D-bucket and
207 A-bucket problems, compute entropy of attention across prompt
regions, compare distributions.
**Requires:** Model loaded (small GPU, ~30 min), or approximate from
cached Stage 2 attention if saved.
**Would change:** If D-bucket prefill attention entropy is lower than
A-bucket, F-7 gains a mechanistic explanation: D-bucket relies on a
single prompt feature triggering a single reasoning pathway, fragile to
sampling perturbation. If equivalent, F-7 fragility lives elsewhere
(generation-time stochasticity, not prompt-side concentration).
**Blocks:** nothing.

### H-22: Abstract-CoT shows compressed or qualitatively different breathing
**Priority:** MEDIUM (heaviest experiment in the queue)
**Motivated by:** Ramji et al. 2604.22709 (Abstract Chain-of-Thought,
11.6x token compression at comparable accuracy). Cleanly separates
reasoning-as-computation from reasoning-as-verbalization.
**Test:** Fine-tune Qwen2.5-1.5B with Abstract-CoT on MATH-500. Extract
per-token L19 activations during abstract reasoning. Compute temporal PR
curve. Compare peak PR, inflation rate, and collapse asymmetry against
natural-language CoT.
**Requires:** GPU (training pipeline release pending from authors),
~1 week H100. ~$300.
**Would change:** If Abstract-CoT shows flat or monotonically decreasing
PR, breathing (F-1, F-5) is specific to verbalized reasoning, not
reasoning itself — dimensional inflation is the computational cost of
verbalization, not of computation. If breathing replicates with
compressed shape, breathing is a property of sequential autoregressive
generation regardless of vocabulary.
**Blocks:** nothing. Long-tail experiment.
"""


# ---------------------------------------------------------------------------
# HYPOTHESES.md — cost-summary-table rows (inserted into the existing table)
# ---------------------------------------------------------------------------

NEW_COST_TABLE_ROWS = [
    "| **Free / local** | H-15 | Length-band PR control (refutation) | 20 min CPU | $0 | Self-applied refutation test for F-4. Cached NPZs. |",
    "| **Free / local** | H-16 | Marchenko-Pastur PR bias correction | 1 h CPU | $0 | Methodology check on F-1, F-4. Cached NPZs. |",
    "| **Free / local** | H-17 | RoPE de-rotation of DoM | 1 h CPU | $0 | Could revive E1 steering and retire H-1 if F-3 is mechanical. |",
    "| **Cheap GPU** | H-18 | PANL-equivalent correctability gate | ~30 min H100 | ~$1 | Could explain F-7 (D-bucket) via second-order confidence. |",
    "| **Cheap GPU** | H-19 | C_exact verification routing ★ | ~30 min H100 | ~$1 | Highest-leverage; converts F-8 from predictor to recipe. |",
    "| **Free / local** | H-20 | Cross-model kernel alignment fluctuation | 2 h CPU | $0 | Extends F-1 to dynamic Platonic hypothesis. |",
    "| **Cheap GPU** | H-21 | D-bucket prefill attention entropy | ~30 min GPU | ~$1 | Mechanism for F-7 fragility. |",
    "| **Big swing** | H-22 | Abstract-CoT compressed breathing | ~1 week H100 | ~$300 | Tests whether breathing is verbalization or computation. |",
]


# ---------------------------------------------------------------------------
# PAPER_INDEX.md — paper sections (template matches existing entries)
# ---------------------------------------------------------------------------

NEW_PAPER_INDEX_MD = """\
## 2604.18805 — AI scientists produce results without reasoning scientifically (Ríos-García et al., 2026)

**Relevance:** Large-scale (25k+ run) eval of LLM-based scientific agents
showing 68% of traces ignore evidence and only 26% revise from refutation.
Directly applicable to our own workflow — motivates self-applied
refutation tests on the headline findings (F-1, F-4).

**Key claim we tested:** LLM agents systematically fit new evidence to
existing frames rather than treating it as disconfirming.

**Our result:** **CITED ONLY (motivates H-15 length-band control).** Their
finding triggered the strongest critique we have on the breathing claim —
that final-token PR collapse may be a length-mixing artifact across
problems with different generation lengths. H-15 is the experiment that
tests it.

**Related experiments:** H-15 (length-band PR control), all of P11
breathing analysis.

**Status:** CITED ONLY (motivated H-15).

---

## 2509.26560 — Estimating Dimensionality of Neural Representations from Finite Samples (Chun et al., 2025)

**Relevance:** Bias-corrected participation-ratio estimator using
Marchenko-Pastur correction. Direct methodological threat to our
cross-model PR comparisons.

**Key claim we tested:** Naive PR is systematically biased downward at
finite samples; the bias depends on the P/Q ratio and breaks cross-model
comparisons when models have different hidden dimensions.

**Our result:** **TO TEST.** Our cross-model P/Q values (Qwen 1.5B 0.33,
Qwen 7B 0.14, Llama 0.24, Phi-3 0.16) are all in the bias-significant
regime. Naive vs corrected estimator comparison is H-16. The within-model
breathing shape should survive (constant bias across positions); the
cross-model magnitude comparison is most vulnerable.

**Related experiments:** H-16 (bias-corrected PR), F-1 cross-arch
breathing (vulnerable), F-4 asymmetric collapse (within-model, less
vulnerable).

**Status:** TO TEST (H-16, HIGH priority).

---

## 2604.22271 — How LLMs Detect and Correct Their Own Errors: Internal Confidence Signals (Kumaran et al., 2026)

**Relevance:** Identifies the post-answer-newline (PANL) token as a
second-order confidence signal that predicts error detection at AUROC
0.986 (Gemma) and 0.961 (Qwen 7B). Orthogonal to verification logprobs
(cos = 0.007).

**Key claim we tested:** Confidence is a two-circuit architecture in
transformers: a generation signal (logprobs / final-token DoM) and an
independent evaluative signal (PANL / prefill DoM in our setup).

**Our result:** **TO TEST (sharpens F-3).** Our finding F-3 — prefill DoM
orthogonal to final-token DoM (cos = 0.046) — is the same architecture
viewed from the opposite temporal end. Their PANL is "did I answer
correctly" computed post-hoc; our prefill is "will I answer correctly"
computed pre-hoc. Both orthogonal to the generation-time signal. H-18
tests the PANL-equivalent in our data; H-19 routes compute on it.

**Delta from their setup:** They measure PANL as a static snapshot post-
generation; we have the full per-token trajectory and the pre-generation
prefill signal too — we already have temporal dynamics they don't address.

**Related experiments:** H-18 (PANL-equivalent gate), H-19 (C_exact
verification routing), F-2, F-3.

**Status:** TO TEST (H-18, H-19 — HIGH and CRITICAL priority).

---

## 2405.07987 — The Platonic Representation Hypothesis (Huh et al., 2024)

**Relevance:** Argues that representations across vision and language
models converge to a shared statistical model of reality — "all strong
models are alike." Our cross-architecture breathing universality (F-1) is
evidence for the hypothesis applied to inference-time dynamics, not just
trained representations.

**Key claim we tested:** Different models converge to similar
representations as they get more capable, measured via mutual k-NN
alignment of kernel matrices.

**Our result:** **EXTENDED (motivates H-20).** Static convergence is
their claim. Our F-1 finding — same inflate-then-collapse PR shape across
4 models, 3 architectures — extends the convergence claim from static
representations to temporal dynamics during inference. H-20 tests the
mutual-alignment fluctuation prediction directly: alignment should peak
at low-PR moments (prefill, final token) and reach minimum at peak PR
(mid-generation).

**Related experiments:** H-20 (kernel alignment fluctuation), F-1
cross-arch breathing.

**Status:** TO TEST (H-20, MEDIUM priority).

---

## 2604.24712 — When Prompt Under-Specification Improves Code Correctness (Akli et al., 2026)

**Relevance:** Shows prompt over-specification can mislead LLMs by
triggering memorized-but-wrong solution strategies. Sampling analog of our
D-bucket finding (F-7).

**Key claim we tested:** Over-specification triggers retrieval of
memorized-but-wrong patterns; structurally rich prompts distribute
attention; structurally simple prompts concentrate attention on a single
specification, making the model fragile to perturbation.

**Our result:** **CITED ONLY (motivates H-21).** Their attention-
concentration mechanism (Figure 3, HumanEval 60-86% attention on
description) is the prompt-time analog of our D-bucket fragility (F-7) at
sampling time. H-21 tests whether D-bucket problems also concentrate
prefill attention more than A-bucket problems.

**Delta from their setup:** Their unit of analysis is the prompt; ours is
the sampling distribution at fixed prompt. Same single-specification
fragility, different time axis.

**Related experiments:** H-21 (D-bucket attention entropy), F-7.

**Status:** CITED ONLY (motivates H-21, MEDIUM priority).

---

## 2604.22709 — Thinking Without Words: Efficient Latent Reasoning with Abstract Chain-of-Thought (Ramji et al., 2026)

**Relevance:** Discrete latent CoT achieving 11.6x token compression at
comparable accuracy. Cleanly separates reasoning-as-computation from
reasoning-as-verbalization — testable distinction for our breathing
finding (F-1, F-5).

**Key claim we tested:** Reasoning can be performed via short sequences
of discrete latent tokens from a reserved vocabulary, achieving
substantial compression with minimal accuracy loss.

**Our result:** **TO TEST.** If breathing is about reasoning, Abstract-
CoT should show a compressed inflate-collapse curve over fewer token
positions. If breathing is about verbalization (sequential autoregressive
generation through natural-language tokens), Abstract-CoT should show a
qualitatively different shape — flat or monotonic. H-22 tests the
prediction.

**Delta from their setup:** They focus on token-count efficiency; we'd
re-run their fine-tuning recipe to extract per-token L19 activations and
measure PR.

**Related experiments:** H-22 (Abstract-CoT breathing curve), F-1, F-5.

**Status:** TO TEST (H-22, MEDIUM priority — heaviest experiment).
"""


# ---------------------------------------------------------------------------
# Format-discovery sanity check
# ---------------------------------------------------------------------------

_H_HEADER_RE = re.compile(r"^### H-(\d+): (.+)$", re.MULTILINE)
_H_FIELD_RE = re.compile(r"^\*\*(\w[\w ]*):\*\*", re.MULTILINE)


def assert_template_matches() -> None:
    """Parse one existing H-N entry. Abort if the template has drifted."""
    text = (REPO / "HYPOTHESES.md").read_text()
    headers = _H_HEADER_RE.findall(text)
    fields = _H_FIELD_RE.findall(text)
    expected_fields = {"Priority", "Motivated by", "Test", "Requires", "Would change"}
    seen = set(fields)
    missing = expected_fields - seen
    if missing:
        raise SystemExit(
            f"Template drift: expected fields {expected_fields} not all present "
            f"in HYPOTHESES.md (missing: {missing}). "
            "Backfill aborted to avoid format-incompatible inserts."
        )
    if not headers:
        raise SystemExit("No H-N headers found in HYPOTHESES.md — abort.")
    next_id_line = re.search(r"Next ID: H-(\d+)", text)
    if not next_id_line:
        raise SystemExit('No "Next ID: H-N" line found in HYPOTHESES.md — abort.')
    next_n = int(next_id_line.group(1))
    if next_n != 15:
        raise SystemExit(
            f'Expected "Next ID: H-15" (next contiguous), got H-{next_n}. '
            "Refusing to clobber an unexpected state. Investigate before re-running."
        )
    print(f"  format ok: {len(headers)} existing H-N entries, all required fields present, next_id=H-15")


# ---------------------------------------------------------------------------
# Mutators
# ---------------------------------------------------------------------------

def write_graph(apply_changes: bool) -> None:
    if not apply_changes:
        print("\n--- DRY RUN: graph writes ---")
        for paper in PAPERS:
            print(f"  Would MERGE Paper {paper['arxiv_id']}: {paper['title']}")
        for fe in FUTURE_EXPERIMENTS:
            print(f"  Would MERGE FutureExperiment {fe['id']} ({fe['priority']}, ROI={fe['roi_score']})")
            for fnd in fe["depends_on_findings"]:
                print(f"    -[:DEPENDS_ON_FINDING]-> {fnd}")
            for fnd in fe["would_update_findings"]:
                print(f"    -[:WOULD_UPDATE]-> {fnd}")
            for arx in fe["triggered_by_papers"]:
                print(f"    -[:TRIGGERED_BY]-> {arx}")
        return

    print("\n--- APPLYING graph writes ---")
    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as s:
        for paper in PAPERS:
            s.run(
                """
                MERGE (p:Paper {arxiv_id: $arxiv_id})
                SET p.title = coalesce($title, p.title),
                    p.year = coalesce($year, p.year),
                    p.relevance_note = coalesce($rel, p.relevance_note),
                    p.status = $status
                """,
                arxiv_id=paper["arxiv_id"],
                title=paper.get("title"),
                year=paper.get("year"),
                rel=paper.get("relevance"),
                status=paper.get("status", "graphed"),
            )
            for tag in paper.get("tags", []):
                s.run("MERGE (:Tag {name: $t})", t=tag)
                s.run(
                    "MATCH (p:Paper {arxiv_id: $a}), (t:Tag {name: $tag}) MERGE (p)-[:TAGGED]->(t)",
                    a=paper["arxiv_id"], tag=tag,
                )
            print(f"  ok Paper {paper['arxiv_id']}")

        for fe in FUTURE_EXPERIMENTS:
            s.run(
                """
                MERGE (fe:FutureExperiment {id: $id})
                SET fe.pathway_id = $pathway_id,
                    fe.description = $description,
                    fe.rationale = $rationale,
                    fe.trigger = $trigger,
                    fe.status = $status,
                    fe.blocked_by = $blocked_by,
                    fe.priority = $priority,
                    fe.estimated_cost = $cost,
                    fe.roi_score = $roi,
                    fe.created_date = coalesce(fe.created_date, $today)
                """,
                id=fe["id"],
                pathway_id=fe["pathway_id"],
                description=fe["description"],
                rationale=fe["rationale"],
                trigger=fe["trigger"],
                status=fe["status"],
                blocked_by=fe["blocked_by"],
                priority=fe["priority"],
                cost=fe["estimated_cost"],
                roi=fe["roi_score"],
                today=TODAY,
            )
            s.run(
                """
                MATCH (p:Pathway {id: $pid}), (fe:FutureExperiment {id: $fid})
                MERGE (p)-[:HAS_FUTURE_EXPERIMENT]->(fe)
                """,
                pid=fe["pathway_id"], fid=fe["id"],
            )
            for fnd in fe["depends_on_findings"]:
                s.run(
                    """
                    MATCH (fe:FutureExperiment {id: $fid}), (f:Finding {id: $fnd})
                    MERGE (fe)-[:DEPENDS_ON_FINDING]->(f)
                    """,
                    fid=fe["id"], fnd=fnd,
                )
            for fnd in fe["would_update_findings"]:
                s.run(
                    """
                    MATCH (fe:FutureExperiment {id: $fid}), (f:Finding {id: $fnd})
                    MERGE (fe)-[:WOULD_UPDATE]->(f)
                    """,
                    fid=fe["id"], fnd=fnd,
                )
            for arxiv_id in fe["triggered_by_papers"]:
                s.run("MERGE (:Paper {arxiv_id: $a})", a=arxiv_id)
                s.run(
                    """
                    MATCH (fe:FutureExperiment {id: $fid}), (p:Paper {arxiv_id: $a})
                    MERGE (fe)-[:TRIGGERED_BY]->(p)
                    """,
                    fid=fe["id"], a=arxiv_id,
                )
            print(f"  ok FutureExperiment {fe['id']}")
    drv.close()


def update_hypotheses_md(apply_changes: bool) -> None:
    path = REPO / "HYPOTHESES.md"
    text = path.read_text()

    new_text = text.replace("Next ID: H-15.", "Next ID: H-23.")

    cost_table_anchor = "| **Blocked** | H-8 | Prefill direction across training checkpoints | ~3 H100-days | ~$200 | Same checkpoint blocker as H-4. |"
    if cost_table_anchor not in new_text:
        raise SystemExit("Cost-table anchor not found — abort to avoid mis-insertion.")
    cost_addition = "\n" + "\n".join(NEW_COST_TABLE_ROWS)
    new_text = new_text.replace(cost_table_anchor, cost_table_anchor + cost_addition)

    hist_anchor = "## Historical / abandoned"
    if hist_anchor not in new_text:
        raise SystemExit("Historical-section anchor not found — abort.")
    insertion = "\n" + NEW_HYPOTHESES_MD + "\n---\n\n"
    new_text = new_text.replace(hist_anchor, insertion + hist_anchor)

    if apply_changes:
        path.write_text(new_text)
        print(f"  ok HYPOTHESES.md updated (+{new_text.count(chr(10)) - text.count(chr(10))} lines)")
    else:
        print("\n--- DRY RUN: HYPOTHESES.md changes ---")
        print(f"  Next ID: H-15 -> H-23")
        print(f"  Cost table: +{len(NEW_COST_TABLE_ROWS)} rows after Blocked H-8")
        print(f"  H-15..H-22 inserted before 'Historical / abandoned' (~{NEW_HYPOTHESES_MD.count(chr(10))} lines)")


def update_paper_index_md(apply_changes: bool) -> None:
    path = REPO / "PAPER_INDEX.md"
    text = path.read_text()
    addition = "\n" + NEW_PAPER_INDEX_MD
    new_text = text.rstrip() + addition

    if apply_changes:
        path.write_text(new_text)
        print(f"  ok PAPER_INDEX.md updated (+{new_text.count(chr(10)) - text.count(chr(10))} lines)")
    else:
        print("\n--- DRY RUN: PAPER_INDEX.md changes ---")
        print(f"  Append {len([p for p in PAPERS])} paper sections (~{NEW_PAPER_INDEX_MD.count(chr(10))} lines)")


def regen_next_experiments(apply_changes: bool) -> None:
    if not apply_changes:
        print("\n--- DRY RUN: would call generate_next_experiments.py ---")
        return
    print("\n--- Regenerating NEXT_EXPERIMENTS.md ---")
    res = subprocess.run(
        [sys.executable, str(ROOT / "generate_next_experiments.py")],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("  generate_next_experiments.py FAILED:")
        print(res.stdout)
        print(res.stderr)
        raise SystemExit(res.returncode)
    print(res.stdout.strip() or "  ok")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true",
                   help="Write changes. Default is dry-run (prints diffs and graph plans).")
    args = p.parse_args()

    print("=== backfill_2026-04-28_session ===")
    print(f"mode: {'APPLY' if args.apply else 'DRY-RUN'}")
    print()
    print("step 0: format discovery")
    assert_template_matches()

    print()
    print("step 1: graph writes")
    write_graph(args.apply)

    print()
    print("step 2: HYPOTHESES.md")
    update_hypotheses_md(args.apply)

    print()
    print("step 3: PAPER_INDEX.md")
    update_paper_index_md(args.apply)

    print()
    print("step 4: regenerate NEXT_EXPERIMENTS.md")
    regen_next_experiments(args.apply)

    print()
    if args.apply:
        print("DONE. Verify with:")
        print("  cd ~/topo-confidence/research-graph && python query.py highest_roi")
        print("  python ~/topo-confidence/validate_claims.py  # should still be 91/91")
    else:
        print("Dry run complete. Re-run with --apply when satisfied.")


if __name__ == "__main__":
    main()
