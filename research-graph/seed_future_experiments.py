"""Seed FutureExperiment nodes for the topo-confidence research graph.

FutureExperiment nodes are forward-looking — what's NOT YET DONE. Completed
experiments live as :Experiment nodes (seeded by seed.py). When a future
experiment runs, set status=COMPLETED with an outcome; do not delete it.

Usage:
    python seed_future_experiments.py
    python seed_future_experiments.py --reset    # wipe FutureExperiment nodes only

Idempotent: re-running updates existing nodes via MERGE.
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")

TODAY = date.today().isoformat()


# ---------------------------------------------------------------------------
# Paper stubs that the future experiments reference (created if missing)
# ---------------------------------------------------------------------------

NEW_PAPER_STUBS: list[dict[str, Any]] = [
    {"arxiv_id": "2306.03819", "title": "LEACE: Linear Concept Erasure",
     "year": 2023, "repo_url": "https://github.com/EleutherAI/concept-erasure"},
    {"arxiv_id": "2103.07353", "title": "Fast Zigzag Persistence (Dey-Hou)",
     "year": 2021, "repo_url": "https://github.com/taohou01/fzz"},
    {"arxiv_id": "2504.10063", "title": "TOHA: Topological Divergence on Attention",
     "year": 2025, "repo_url": None},
    {"arxiv_id": "2601.01552", "title": "HalluZig: Zigzag PH on Attention",
     "year": 2026, "repo_url": None},
    {"arxiv_id": "2506.00653", "title": "Linear Representation Transferability (LRT)",
     "year": 2025, "repo_url": None},
    {"arxiv_id": "1207.6437", "title": "Persistence Landscapes (Bubenik)",
     "year": 2015, "repo_url": None},
    {"arxiv_id": "2410.04707", "title": "Learning How Hard to Think (Damani)",
     "year": 2024, "repo_url": None},
    {"arxiv_id": "2604.16217", "title": "Conformal Prediction via Internal Representations",
     "year": 2026, "repo_url": None},
    {"arxiv_id": "2501.17148", "title": "AxBench: Steering Benchmark",
     "year": 2025, "repo_url": None},
    {"arxiv_id": "2402.13212", "title": "Soft Self-Consistency",
     "year": 2024, "repo_url": "https://github.com/HanNight/soft_self_consistency"},
    {"arxiv_id": "2501.12948", "title": "DeepSeek-R1",
     "year": 2025, "repo_url": "https://github.com/deepseek-ai/DeepSeek-R1"},
    {"arxiv_id": "2412.01113", "title": "LLMs Faithfully Compute During CoT (Kudo)",
     "year": 2024, "repo_url": None},
    {"arxiv_id": "2404.15255", "title": "How to Use Activation Patching",
     "year": 2024, "repo_url": None},
    {"arxiv_id": "2510.18147", "title": "LLMs Encode Problem Difficulty (Lugoloobi)",
     "year": 2025, "repo_url": None},
]


# ---------------------------------------------------------------------------
# Future experiments
# ---------------------------------------------------------------------------

FUTURE_EXPERIMENTS: list[dict[str, Any]] = [
    # ---------- P1-P3 ----------
    {
        "id": "P1-FE1",
        "pathway_id": "P1",
        "description": "Revisit initial geometry observations with proper 1024-token labels and participation ratio instead of ad-hoc metrics. Re-extract on the same problems with current infrastructure.",
        "rationale": "P1 observations predate the truncation correction (F-13). Any early geometry finding may have been measuring truncation artifacts. A clean re-run with current methodology would either validate or retire P1 claims.",
        "trigger": "Already triggered — F-13 (truncation artifact) invalidated the label distribution P1 used.",
        "status": "READY",
        "blocked_by": None,
        "priority": "MEDIUM",
        "estimated_cost": "4h CPU on cached activations",
        "roi_score": 5,
        "depends_on_findings": ["F-13"],
        "would_update_findings": ["F-1"],
        "triggered_by_papers": [],
    },
    {
        "id": "P2-FE1",
        "pathway_id": "P2",
        "description": "Re-derive the steering vector from scratch using 1024-token labels and proper DoM at L19. Compare against the original P2 cached vector (cos=0.05 with current L19 DoM).",
        "rationale": "The P2 steering vector was derived with truncated labels. F-3 showed it's unusable (cos=0.05). A properly derived vector may still fail due to direction rotation (F-3), but at least it tests steering with a non-stale direction.",
        "trigger": "Already triggered — F-3 and F-13 together show the old vector was doubly wrong (wrong labels + wrong subspace).",
        "status": "READY",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "2h CPU + 4h H100 for generation with intervention",
        "roi_score": 7,
        "depends_on_findings": ["F-3", "F-13"],
        "would_update_findings": ["F-3"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2306.03341",
                "their_method": "ITI shifts activations across a small set of attention heads at inference, using directions learned from a few hundred labeled examples.",
                "their_result": "Alpaca's TruthfulQA score nearly doubles (32.5% to 65.1%) with minimal data and compute.",
                "our_method": "Re-derive the steering vector at L19 from 1024-token correctness labels and test it on MATH-500 generation, comparing cos against the project's stale P2 vector.",
                "same": "Same idea of a label-derived directional intervention on residual-stream geometry.",
                "differs": "Target is correctness on math (not truthfulness on QA); intervention site is residual stream not attention heads; the goal is to test whether re-derivation rescues a known-stale steering vector.",
            },
        ],
    },
    {
        "id": "P3-FE1",
        "pathway_id": "P3",
        "description": "Re-run P3's steering-for-accuracy using adaptive per-token steering (PID or STU-PID) instead of fixed-vector injection. Use the proper L19 DoM direction, apply PID controller across generation positions.",
        "rationale": "P3 failed because the steering direction rotates during generation (F-3). PID steering (2510.04309) and STU-PID (2506.18831) are designed to handle exactly this — they accumulate error and damp overshoot. STU-PID was tested on the project's exact model (DeepSeek-R1-Distill-Qwen-1.5B).",
        "trigger": "PID Steering and STU-PID papers published.",
        "status": "TRIGGERED",
        "blocked_by": None,
        "priority": "CRITICAL",
        "estimated_cost": "1 day H100",
        "roi_score": 9,
        "depends_on_findings": ["F-3", "F-4"],
        "would_update_findings": ["F-3"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2510.04309",
                "their_method": "Casts activation steering as a P controller, then proposes a full PID controller where I accumulates layer-wise error and D damps overshoot.",
                "their_result": "Closed-loop design with stability guarantees; consistently outperforms existing steering across multiple LLM families and benchmarks.",
                "our_method": "Apply a PID controller using the project's L19 prefill DoM as the reference signal during MATH-500 generation, instead of P3's failed fixed-vector injection.",
                "same": "Same closed-loop control framing; same idea of using a learned semantic direction as the feedback signal.",
                "differs": "Reference signal is a correctness direction (not generic semantic targets); benchmark is MATH-500; we evaluate accuracy and AUROC lift against the P3 baseline that failed due to direction rotation (F-3).",
            },
            {
                "arxiv_id": "2506.18831",
                "their_method": "Trains a chunk-level classifier for redundant reasoning patterns and uses a PID controller to adaptively modulate steering strength based on predicted redundancy.",
                "their_result": "On GSM8K: +6% accuracy and -32% tokens vs static-steering baselines, training-free at inference.",
                "our_method": "Plug STU-PID's controller into our pipeline using the L19 DoM correctness direction as the steering target on MATH-500.",
                "same": "Same per-chunk PID-controlled steering recipe targeting an LLM that overshoots without dynamic adjustment.",
                "differs": "Target signal is correctness (not redundancy); benchmark is MATH-500 not GSM8K; we layer it on an existing correctness probe rather than a fresh redundancy classifier.",
            },
        ],
    },
    # ---------- P4-P6 ----------
    {
        "id": "P4-FE1",
        "pathway_id": "P4",
        "description": "Rebuild the ABC-44 feature pipeline with LEACE-style concept erasure for the length direction instead of regression deconfounding. Compare LEACE-deconfounded AUROC against the regression-deconfounded 0.746.",
        "rationale": "P9-E1 used regression deconfounding which only removes linear length dependence. LEACE (2306.03819) provides closed-form perfect linear erasure — it's the gold standard for representation-level deconfounding.",
        "trigger": "LEACE paper and codebase available.",
        "status": "READY",
        "blocked_by": None,
        "priority": "MEDIUM",
        "estimated_cost": "2h CPU",
        "roi_score": 5,
        "depends_on_findings": ["F-9"],
        "would_update_findings": ["F-9"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2306.03819",
                "their_method": "Closed-form least-squares concept erasure that provably prevents any linear classifier from recovering a target concept while minimally perturbing the embedding.",
                "their_result": "Reduces gender bias in BERT and POS-information leakage across LLMs via 'concept scrubbing' applied at every layer.",
                "our_method": "Use LEACE to erase length from L19 activations, then refit the ABC-44 correctness probe and compare against the regression-deconfounded 0.746 AUROC.",
                "same": "Both deconfound a target representation against a nuisance direction before downstream prediction.",
                "differs": "Nuisance is generation length (not gender / POS); we use it as a baseline upgrade for an existing regression-deconfounded probe; we evaluate AUROC, not bias metrics.",
            },
        ],
    },
    {
        "id": "P4-FE2",
        "pathway_id": "P4",
        "description": "Replace TwoNN global ID with GeoMLE local intrinsic dimension estimation per-problem. Compute local LID at L19 for each MATH-500 problem's activation neighborhood (k=20). Test as correctness predictor.",
        "rationale": "F-10 showed global TwoNN has near-zero predictive power (AUROC 0.407). But Yin et al. (2402.18048) showed LOCAL LID predicts truthfulness with 5-8 AUROC points above baselines using GeoMLE. The discrepancy is likely global-vs-local — local LID may recover the signal TwoNN misses.",
        "trigger": "Already triggered — Yin et al. published with code.",
        "status": "TRIGGERED",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "4h CPU on cached activations",
        "roi_score": 8,
        "depends_on_findings": ["F-10"],
        "would_update_findings": ["F-10"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2402.18048",
                "their_method": "Estimates per-token local intrinsic dimension (LID) of LLM activations via GeoMLE on a k-NN neighborhood and uses it as a truthfulness score.",
                "their_result": "On four QA datasets, local LID outperforms entropy and verbalized-confidence baselines by 5-8 AUROC points.",
                "our_method": "Compute GeoMLE local LID at L19 per MATH-500 problem (k=20) on cached activations and test it as a correctness predictor.",
                "same": "Same LID-on-activations methodology and same GeoMLE estimator; same intent of recovering signal that global ID misses.",
                "differs": "Local instead of global TwoNN (the project's failed P9 setup); benchmark is MATH-500 correctness rather than QA truthfulness; layer is L19, not the paper's mid-layer choice.",
            },
        ],
    },
    {
        "id": "P5-FE1",
        "pathway_id": "P5",
        "description": "Run dimensional breathing analysis (temporal PR curve) on Gemma-2-2B and Mistral-7B on the same MATH-500 problems. Confirm breathing is architecture-universal beyond the Qwen/Phi/Llama families already tested.",
        "rationale": "F-1 replicated across Phi-3-mini and Llama-3.2-1B but all three are decoder-only with similar training recipes. Gemma-2 uses a different attention variant and Mistral uses sliding window attention — testing these extends the universality claim substantially.",
        "trigger": "Whenever H100 pod is next active for any other experiment.",
        "status": "READY",
        "blocked_by": "H100 pod time",
        "priority": "MEDIUM",
        "estimated_cost": "4h H100",
        "roi_score": 6,
        "depends_on_findings": ["F-1"],
        "would_update_findings": ["F-1"],
        "triggered_by_papers": [],
    },
    {
        "id": "P6-FE1",
        "pathway_id": "P6",
        "description": "Test cross-scale probe transfer using affine alignment (Bello et al. LRT, 2506.00653) instead of raw transfer. Train a lightweight affine map from 1.5B L19 activations to 7B L20 activations on a shared problem set, then test whether the 7B's DoM direction transfers to 1.5B through the map.",
        "rationale": "F-5 showed raw cross-scale transfer is flat (0.717 vs self 0.719). But LRT (2506.00653) showed affine maps between model sizes preserve steering-relevant structure. The question is whether an aligned 7B correctness direction transfers better than unaligned.",
        "trigger": "LRT paper published with theoretical framework.",
        "status": "TRIGGERED",
        "blocked_by": "H100 pod for 7B activations",
        "priority": "MEDIUM",
        "estimated_cost": "4h H100 + 2h CPU",
        "roi_score": 6,
        "depends_on_findings": ["F-5"],
        "would_update_findings": ["F-5"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2506.00653",
                "their_method": "Learns affine maps between hidden states of different-sized models on shared inputs, then transfers steering vectors through the map (the LRT hypothesis).",
                "their_result": "Steering vectors transferred small-to-large preserve their semantic effect, supporting linear representation transferability across scales.",
                "our_method": "Train an affine map from 1.5B L19 to 7B L20 activations, then test whether the 7B's correctness DoM transfers back to 1.5B and beats raw cross-scale transfer (0.717).",
                "same": "Same affine-alignment-then-transfer recipe; same hypothesis that representations across scales share linear structure.",
                "differs": "Direction is a correctness probe (not a generic steering target); we measure AUROC lift on MATH-500 vs the raw-transfer baseline; mapping direction is large-to-small (1.5B inheriting from 7B).",
            },
        ],
    },
    # ---------- P7 ----------
    {
        "id": "P7-FE1",
        "pathway_id": "P7",
        "description": "Run zigzag persistence on the cached pathway 8 layer-wise activations using the Dey-Hou fast zigzag implementation (github.com/taohou01/fzz) or the RitaSciencePark/topo_llm pipeline. Compare zigzag PH features against the standard Rips PH that hit the Gaussian null.",
        "rationale": "F-7 killed standard Rips PH. But zigzag PH captures transition dynamics between layers — the exact signal that CoE captures geometrically. If zigzag PH survives the Gaussian null where Rips didn't, it means PH *can* work on LLM activations but only when the filtration respects layer ordering.",
        "trigger": "Fast zigzag implementations now available. Dionysus2 install blocker removed.",
        "status": "TRIGGERED",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "3 days CPU",
        "roi_score": 7,
        "depends_on_findings": ["F-7", "F-8"],
        "would_update_findings": ["F-7", "F-8"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2103.07353",
                "their_method": "Near-linear-time algorithms for 0- and 1-dimensional zigzag persistence on graphs (O(m log^2 n) and O(m log^4 n)).",
                "their_result": "Practical zigzag computation on real-world graphs for the first time, removing the cubic-matrix-multiplication bottleneck.",
                "our_method": "Use Dey-Hou's fzz library to make P7-FE1 computationally feasible on 28-layer activations across 500 problems.",
                "same": "Same zigzag-persistence algorithm we depend on for the experiment to terminate.",
                "differs": "We are a downstream user, not a competitor; the project's contribution is whether the resulting features predict correctness, not the algorithm itself.",
            },
            {
                "arxiv_id": "2410.11042",
                "their_method": "Builds a zigzag-persistence filtration across LLM layers and extracts topological descriptors that track holes evolving through the network.",
                "their_result": "Descriptors are sensitive to model and dataset and enable layer pruning competitive with state-of-the-art while preserving a system-level view.",
                "our_method": "Run zigzag PH on the cached pathway-8 layer-wise activations and check whether zigzag features survive the Gaussian null where Rips-PH didn't (F-7).",
                "same": "Same zigzag-across-layers framing applied to LLM residual streams.",
                "differs": "We test for correctness signal (not pruning); we apply a Gaussian-null control because Rips-PH failed it; we compare against per-layer DoM rather than to layer-pruning baselines.",
            },
        ],
    },
    {
        "id": "P7-FE2",
        "pathway_id": "P7",
        "description": "Compute PH on attention graphs (token-token attention matrices per layer) instead of residual-stream point clouds. Use the TOHA/HalluZig approach: build a weighted graph from attention maps, compute PH of the graph filtration.",
        "rationale": "Every successful PH-on-LLM paper in the literature (Kushnareva 2021, Bazarova 2025) works on attention graphs, not residual-stream point clouds. The project tested only the latter. PH may be valid for LLM correctness detection — just on the wrong object.",
        "trigger": "TOHA (2504.10063) and HalluZig (2601.01552) published showing attention-graph PH works.",
        "status": "TRIGGERED",
        "blocked_by": None,
        "priority": "MEDIUM",
        "estimated_cost": "1 week CPU (attention extraction + PH computation)",
        "roi_score": 6,
        "depends_on_findings": ["F-7"],
        "would_update_findings": ["F-7"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2504.10063",
                "their_method": "Computes a topological-divergence metric between prompt-side and response-side attention subgraphs in the RAG setting.",
                "their_result": "Per-head divergence values correlate with hallucinations; achieves SOTA-or-competitive results on QA + summarization with minimal labeled data.",
                "our_method": "Use the same attention-graph object on MATH-500 and test correctness rather than RAG-grounding divergence.",
                "same": "Same conclusion that attention graphs (not residual streams) are the right object for PH-on-LLMs.",
                "differs": "We are not in the RAG setting (no prompt-vs-response divergence); we use a probe directly, not a divergence metric; benchmark is math correctness.",
            },
            {
                "arxiv_id": "2601.01552",
                "their_method": "Models per-layer attention matrices as a zigzag graph filtration and extracts a topological signature from the dynamic graph evolution.",
                "their_result": "Outperforms strong hallucination-detection baselines across benchmarks; signatures generalize across models and across partial network depth.",
                "our_method": "Apply the same zigzag-on-attention-graphs pipeline to MATH-500 correctness instead of factuality, on the project's cached attention matrices.",
                "same": "Same object (attention graphs) and same zigzag-on-attention filtration.",
                "differs": "Target is correctness (not factual hallucination); benchmark is math reasoning (not RAG QA); we compare against the residual-stream-PH null result (F-7) to test whether attention graphs are the right object.",
            },
        ],
    },
    # ---------- P8 ----------
    {
        "id": "P8-FE1",
        "pathway_id": "P8",
        "description": "Re-extract CoE-60 features using 1024-token generations instead of the original 256-token truncated runs. Refit the logistic regression classifier and report AUROC under the corrected label distribution (243/257 instead of 104/396).",
        "rationale": "F-6's headline 0.811 AUROC was measured on truncated labels. The corrected baseline accuracy is 48.6% not 20.8%, which means the class balance is ~50/50 instead of ~20/80. CoE AUROC may go up (more signal) or down (less separation) — we don't know until we test.",
        "trigger": "Already triggered — F-13 invalidated the labels.",
        "status": "READY",
        "blocked_by": "H100 pod for re-extraction",
        "priority": "CRITICAL",
        "estimated_cost": "2h H100 + 1h CPU",
        "roi_score": 10,
        "depends_on_findings": ["F-6", "F-13"],
        "would_update_findings": ["F-6", "F-8"],
        "triggered_by_papers": [],
    },
    {
        "id": "P8-FE2",
        "pathway_id": "P8",
        "description": "Extract CoE features at step/chunk boundaries during chain-of-thought (every 50 tokens) rather than only at the final token. Train a step-level correctness probe. Compare against Zhang et al.'s chunk-boundary probing (2504.05419) which achieved >0.9 AUROC on AIME.",
        "rationale": "Current CoE features are final-token only. Zhang et al. showed mid-trajectory probes can beat final-token probes substantially. Step-level CoE also feeds directly into E2 (confidence-gated compute) and E3 (refusal) — you need temporal resolution to gate during generation, not after.",
        "trigger": "Zhang et al. published with strong results.",
        "status": "TRIGGERED",
        "blocked_by": "H100 pod",
        "priority": "HIGH",
        "estimated_cost": "4h H100 + 2h CPU",
        "roi_score": 8,
        "depends_on_findings": ["F-6", "F-2"],
        "would_update_findings": ["F-6", "F-2"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2504.05419",
                "their_method": "Trains linear probes on hidden states at intermediate-answer positions during long-CoT reasoning to verify correctness mid-generation.",
                "their_result": "Probes verify intermediate answers with high accuracy and calibration; using the probe as an early-exit verifier cuts inference tokens by 24% with no quality drop.",
                "our_method": "Extract CoE features at chunk boundaries (every 50 tokens) on cached MATH-500 generations and train a step-level correctness probe.",
                "same": "Same step-level probing target (correctness encoded in hidden states) and same goal of beating final-token-only probes.",
                "differs": "We probe CoE-style geometry features (not raw hidden state); benchmark is MATH-500 (not AIME); the result feeds into selective-prediction (E2/E3), not just early exit.",
            },
        ],
    },
    {
        "id": "P8-FE3",
        "pathway_id": "P8",
        "description": "Compute representation dispersion (average pairwise cosine distance) per layer on the cached activations. Correlate with CoE magnitude features and with correctness. Test whether dispersion alone (1 feature per layer, 28 total) matches D2H-lite's 58-dim AUROC.",
        "rationale": "Li & Li (2506.24106) showed dispersion strongly predicts perplexity and downstream accuracy across Qwen family. This is exactly what D2H's intra-layer dispersion measures but in a simpler formulation. If 28-dim dispersion matches 58-dim D2H, the extra complexity is unnecessary.",
        "trigger": "Dispersion paper published with Qwen validation.",
        "status": "TRIGGERED",
        "blocked_by": None,
        "priority": "MEDIUM",
        "estimated_cost": "2h CPU on cached activations",
        "roi_score": 5,
        "depends_on_findings": ["F-6"],
        "would_update_findings": [],
        "triggered_by_papers": [
            {
                "arxiv_id": "2506.24106",
                "their_method": "Computes representation dispersion (mean pairwise cosine distance among hidden vectors) per layer and uses it as an unsupervised quality signal.",
                "their_result": "Dispersion correlates strongly and negatively with perplexity across LLaMA + Qwen on Wikipedia / news / scientific abstracts; helps rank examples by difficulty and select layers for kNN-LM.",
                "our_method": "Compute per-layer dispersion (1 number per layer, 28 total) on cached activations and test against D2H-lite's 58-dim AUROC for correctness.",
                "same": "Same metric (mean pairwise cosine distance) on the same kind of representations.",
                "differs": "Target is supervised correctness (not perplexity); benchmark is MATH-500; we compare against an existing 58-dim D2H baseline rather than to perplexity itself.",
            },
        ],
    },
    # ---------- P9 ----------
    {
        "id": "P9-FE1",
        "pathway_id": "P9",
        "description": "Run the full 5-experiment battery (length deconfound, orthogonality, Gaussian null, XGBoost-vs-LR, cross-domain transfer) using 1024-token corrected labels. Report which findings survive and which change.",
        "rationale": "Every P9 number was computed on the wrong labels. This is the single most important re-validation in the project.",
        "trigger": "Already triggered — F-13.",
        "status": "READY",
        "blocked_by": "H100 pod for fresh feature extraction",
        "priority": "CRITICAL",
        "estimated_cost": "2h H100 + 1 day CPU",
        "roi_score": 10,
        "depends_on_findings": ["F-13"],
        "would_update_findings": ["F-6", "F-7", "F-8", "F-9", "F-10"],
        "triggered_by_papers": [],
    },
    {
        "id": "P9-FE2",
        "pathway_id": "P9",
        "description": "Add persistence landscapes and persistence images as PH feature representations (instead of 5 summary statistics). Re-run the Gaussian null test. Persistence landscapes (Bubenik 2015) have statistical testing guarantees that raw summaries lack.",
        "rationale": "F-7 killed 5 raw PH summary stats. But the Gaussian null test was on a specific lossy representation. Persistence landscapes vectorize the full diagram and have known hypothesis testing properties. If landscapes also fail, PH is comprehensively dead for this use case.",
        "trigger": "Methodological — always applicable.",
        "status": "READY",
        "blocked_by": None,
        "priority": "LOW",
        "estimated_cost": "2 days CPU",
        "roi_score": 4,
        "depends_on_findings": ["F-7"],
        "would_update_findings": ["F-7"],
        "triggered_by_papers": [
            {
                "arxiv_id": "1207.6437",
                "their_method": "Defines persistence landscapes — a Banach-space vector representation of persistence diagrams that obeys a strong law of large numbers and a central limit theorem, enabling standard hypothesis tests.",
                "their_result": "Landscapes are stable, give bottleneck and Wasserstein lower bounds, and unlock parametric statistical inference on PH features.",
                "our_method": "Replace the 5-stat PH summary with persistence landscapes (and persistence images) and re-run the F-7 Gaussian null test.",
                "same": "Same Rips-PH on the same point clouds; same Gaussian-null control.",
                "differs": "Feature representation is a full vectorization (not 5 lossy stats); the comparison is whether a richer summary recovers signal lost to summarization, not whether PH itself works.",
            },
        ],
    },
    {
        "id": "P9-FE3",
        "pathway_id": "P9",
        "description": "Run cross-domain transfer experiment (P9-E5) on individual BBH subsets instead of pooled. Report per-subset AUROC for CoE, D2H, and DoM. Identify which subset drives the D2H asymmetry.",
        "rationale": "The pooled BBH transfer (0.720/0.712 for CoE) may be driven by one subset. web_of_lies has 54% accuracy (near-balanced), tracking_shuffled_objects has 12.4%, logical_deduction has 11.2%. The subset heterogeneity is extreme and pooling masks it.",
        "trigger": "Already identified as open question in pathway 9.",
        "status": "READY",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "2h CPU on cached features",
        "roi_score": 7,
        "depends_on_findings": ["F-6"],
        "would_update_findings": ["F-6"],
        "triggered_by_papers": [],
    },
    # ---------- P10 ----------
    {
        "id": "P10-FE1",
        "pathway_id": "P10",
        "description": "Run E1 steering experiment with three parallel arms: (a) probe-weight vector, (b) mass-mean (difference-of-means) vector, (c) trained per-layer bias (Sinii-style, 2505.18706). Sweep alpha, report accuracy on MATH-500 holdout for each. Use SEAL's thought-type vector (2504.07986) as a fourth comparison if feasible.",
        "rationale": "The literature strongly predicts mass-mean >= probe-weight, and RL-trained biases exceed both (AxBench ranking, AdaRAS, Sinii). Running all three arms on the same model and benchmark gives a definitive comparison that doesn't exist in the literature.",
        "trigger": "Sinii and SEAL published with results on same model family.",
        "status": "TRIGGERED",
        "blocked_by": "H100 pod",
        "priority": "CRITICAL",
        "estimated_cost": "1 day H100",
        "roi_score": 9,
        "depends_on_findings": ["F-3", "F-4"],
        "would_update_findings": ["F-3", "F-4"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2306.03341",
                "their_method": "ITI applies an activation-shift intervention to a small set of attention heads at inference, using directions derived from labeled probes.",
                "their_result": "Alpaca's TruthfulQA score nearly doubles (32.5% to 65.1%) with a few hundred examples and minimal compute.",
                "our_method": "Treat ITI's mass-mean direction as one arm of E1 (alongside probe-weight and trained-bias) and apply it at L19 on MATH-500 with sweeps over alpha.",
                "same": "Same idea of mass-mean / difference-in-means as a steering direction.",
                "differs": "Site is the residual stream at L19 (not selected attention heads); target is math correctness (not truthfulness); evaluation is MATH-500 accuracy across an alpha sweep.",
            },
            {
                "arxiv_id": "2505.18706",
                "their_method": "Trains a single d-dim per-layer steering bias with reinforcement learning while freezing all base weights.",
                "their_result": "On 8B models, +0.0016% extra params matches full RL-tuned reasoning accuracy on math benchmarks; reduces optimizer memory and inter-GPU communication.",
                "our_method": "Run an analogous RL-trained per-layer bias arm at L19 in E1, alongside ITI mass-mean and probe-weight, on MATH-500 holdout.",
                "same": "Same per-layer bias-only trainable adapter; same math-reasoning benchmark family.",
                "differs": "Backbone is Qwen-2.5-1.5B (not 8B); we ablate it as one arm of a three-arm comparison rather than as a standalone result.",
            },
            {
                "arxiv_id": "2504.07986",
                "their_method": "Categorizes CoT into execution / reflection / transition thoughts; offline-extracts a thought-type steering vector and applies it during generation.",
                "their_result": "+11% accuracy and 11.8-50.4% token reduction on Math500 + GSM8K + LiveCodeBench using DeepSeek-R1-Distill and QwQ-32B.",
                "our_method": "Run SEAL's thought-type vector as a fourth arm of E1 (when feasible).",
                "same": "Same offline-extraction-then-online-intervention recipe; same family of math benchmarks.",
                "differs": "Target is correctness (not reasoning-style efficiency); model is Qwen-2.5-1.5B base (not R1-Distill or QwQ-32B); the comparison is against probe-weight / mass-mean / bias arms.",
            },
            {
                "arxiv_id": "2501.17148",
                "their_method": "Large-scale benchmark on Gemma-2-2B/9B comparing prompting, finetuning, SAEs, supervised steering vectors, linear probes, and ReFT-r1.",
                "their_result": "Prompting > finetuning > all representation methods for steering; difference-in-means wins concept detection; SAEs are not competitive.",
                "our_method": "Use AxBench's ranking as the prior that orders our three E1 arms (probe-weight, mass-mean, trained bias) and predicts mass-mean >= probe-weight.",
                "same": "Same comparison axis (probe-weight vs difference-in-means vs trained methods).",
                "differs": "Single-model + math-benchmark setup vs a multi-method benchmark on Gemma; we test the literature ranking on a model and benchmark the original paper does not cover.",
            },
        ],
    },
    {
        "id": "P10-FE2",
        "pathway_id": "P10",
        "description": "Implement E3 refusal with conformal prediction wrapper (Mohri & Hashimoto, 2402.10978). Instead of a fixed threshold tau, use conformal calibration to provide distribution-free coverage guarantees. Report: at 90% coverage, what accuracy? At 95% precision on answered, what coverage?",
        "rationale": "F-11's 71.6%@50% coverage is strong but uses a fixed threshold. Conformal prediction gives the same operating point with a statistical guarantee that holds on future data. This is the publishable version of E3.",
        "trigger": "Conformal factuality paper published with MATH numbers.",
        "status": "TRIGGERED",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "4h CPU",
        "roi_score": 8,
        "depends_on_findings": ["F-11", "F-2"],
        "would_update_findings": ["F-11"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2402.10978",
                "their_method": "Conformal factuality: a back-off algorithm that progressively makes LM outputs less specific until conformal prediction guarantees high-probability correctness.",
                "their_result": "80-90% correctness guarantees on FActScore, NaturalQuestions, MATH while preserving most of the original output content.",
                "our_method": "Wrap the project's L19 DoM probe as a conformal nonconformity score and report coverage-vs-accuracy tradeoffs on MATH-500.",
                "same": "Same conformal-prediction wrapper around an LM-correctness signal; same MATH benchmark.",
                "differs": "Score is an internal-representation probe (not back-off entailment); we report a single operating point (e.g. 71.6%@50% coverage) with a guarantee, not a back-off cascade.",
            },
            {
                "arxiv_id": "2604.16217",
                "their_method": "Defines Layer-Wise Information (LI) scores from internal representations and uses them as conformal nonconformity scores in split conformal prediction.",
                "their_result": "LI scores beat token-probability and entropy baselines on the validity-efficiency tradeoff, especially under cross-domain shift.",
                "our_method": "Substitute the L19 DoM score for the LI score and report conformal coverage at fixed risk levels on MATH-500.",
                "same": "Same internal-rep nonconformity score; same conformal pipeline.",
                "differs": "Score is single-layer DoM (not layer-wise information); benchmark is math reasoning (not closed-book / open-domain QA); we have a published probe AUROC (0.7731) to compare against.",
            },
        ],
    },
    {
        "id": "P10-FE3",
        "pathway_id": "P10",
        "description": "Implement E2 (confidence-gated compute) using Damani et al.'s 'Learning How Hard to Think' framework (2410.04707). Train a lightweight predictor on prefill embeddings to estimate marginal reward of additional samples. Compare against the project's monotonic/non-monotonic gating baselines.",
        "rationale": "F-14 showed monotonic gating fails. Damani's framework is purpose-built for this — it trains a meta-predictor of when more compute helps, which is exactly the bucket-B recovery problem.",
        "trigger": "Damani et al. published.",
        "status": "TRIGGERED",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "1 day CPU + 4h H100 for sample generation",
        "roi_score": 8,
        "depends_on_findings": ["F-14", "F-2"],
        "would_update_findings": ["F-14"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2410.04707",
                "their_method": "Trains a meta-predictor of the reward distribution given an input + budget; uses it for adaptive best-of-k and decoder routing.",
                "their_result": "Up to -50% compute at fixed quality, or +10% quality at fixed compute, across programming + math + dialog suites.",
                "our_method": "Train a Damani-style predictor on prefill embeddings to estimate marginal reward of additional samples; gate K dynamically against the project's bucket-B problems.",
                "same": "Same meta-predictor framing; same input-adaptive compute-allocation goal.",
                "differs": "Predictor input is L19 prefill (not text features); gating policy targets the bucket-B recovery problem (F-14) rather than generic best-of-k; benchmark is MATH-500 with the project's K-vs-accuracy curve.",
            },
        ],
    },
    {
        "id": "P10-FE4",
        "pathway_id": "P10",
        "description": "Implement E4 distillation using TIP's Q3-token-focused loss (2604.14084). Fine-tune Qwen2.5-1.5B with auxiliary loss matching its L19 activations to 7B's L20 activations, but only at Q3 tokens (low student entropy, high teacher-student divergence). Compare against uniform distillation.",
        "rationale": "TIP showed Q3 tokens carry disproportionate corrective signal on the exact model pair (Qwen2.5-14B->1.5B). Combining with the project's finding that L19 carries the correctness signal gives a focused distillation target: match the correctness direction at positions where the student is overconfidently wrong.",
        "trigger": "TIP published with code and Qwen results.",
        "status": "TRIGGERED",
        "blocked_by": "Multi-day H100 allocation",
        "priority": "MEDIUM",
        "estimated_cost": "2+ weeks, multi-day H100",
        "roi_score": 6,
        "depends_on_findings": ["F-4", "F-5"],
        "would_update_findings": ["F-5"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2604.14084",
                "their_method": "Distillation token-importance taxonomy across student-entropy x teacher-student-divergence; Q3 = low-entropy + high-divergence (overconfident-wrong) tokens carry dense corrective signal.",
                "their_result": "Q3-only training on <20% of tokens surpasses full-token OPD on MATH-500 + AIME 2024/2025 + DeepPlanning across Qwen3 / Llama / Qwen2.5 teacher-student pairs.",
                "our_method": "Distill Qwen-2.5-1.5B from the 7B teacher with a Q3-token loss matching L19 (student) to L20 (teacher) only at correctness-relevant positions.",
                "same": "Same Q3 token-selection rule; same Qwen2.5 teacher-student family.",
                "differs": "Loss target is the L19 correctness direction (not the teacher's full distribution); we apply distillation to a single layer pair at the token positions where the student's correctness probe disagrees with the teacher's.",
            },
        ],
    },
    # ---------- P11 ----------
    {
        "id": "P11-FE1",
        "pathway_id": "P11",
        "description": "Run breathing analysis (temporal PR curve) on BBH subsets using Stage 4a per-token activations. Does breathing occur on non-math tasks? Does the correct/incorrect PR ratio hold on web_of_lies (54% accuracy, near-balanced)?",
        "rationale": "F-1 was measured on MATH-500 only. BBH data is already cached on the pod. If breathing occurs on BBH, it's a reasoning phenomenon. If not, it's math-specific.",
        "trigger": "Data already cached.",
        "status": "READY",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "2h CPU",
        "roi_score": 8,
        "depends_on_findings": ["F-1"],
        "would_update_findings": ["F-1"],
        "triggered_by_papers": [],
    },
    {
        "id": "P11-FE2",
        "pathway_id": "P11",
        "description": "Test whether the prefill direction encodes training-set familiarity or problem structure. Use MATH-500 difficulty level metadata (5 levels in the original dataset). Compute prefill DoM score vs difficulty level. If prefill tracks difficulty perfectly, it's familiarity. If orthogonal to difficulty but still predicts correctness, it's something more interesting.",
        "rationale": "F-2's strongest counterargument is that prefill encodes familiarity, not structure. This test directly addresses it. Lugoloobi & Russell (2510.18147) showed Qwen2.5-Math-1.5B encodes human difficulty with rho=0.88 — is our prefill DoM the same signal or a different one?",
        "trigger": "Lugoloobi & Russell published on exact model.",
        "status": "TRIGGERED",
        "blocked_by": None,
        "priority": "HIGH",
        "estimated_cost": "1h CPU",
        "roi_score": 8,
        "depends_on_findings": ["F-2"],
        "would_update_findings": ["F-2"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2510.18147",
                "their_method": "Trains linear probes across layers and token positions on 60 models and evaluates on Easy2HardBench math + coding subsets.",
                "their_result": "Human-difficulty rho approx 0.88 on AMC with clear model-size scaling; steering toward 'easier' reduces hallucination; GRPO on Qwen2.5-Math-1.5B amplifies the human-difficulty probe.",
                "our_method": "Compute the project's prefill DoM score against MATH-500 difficulty levels (1-5) on Qwen-2.5-1.5B and check whether the prefill direction is the same as Lugoloobi's difficulty direction or orthogonal to it.",
                "same": "Same model family (Qwen2.5-Math-1.5B), same kind of math benchmark, same probe-the-residual-stream method.",
                "differs": "We are testing whether prefill-DoM-correctness != difficulty (orthogonality); they showed difficulty itself is encoded — we want to know if our 'can I solve this' signal is a different direction.",
            },
        ],
    },
    {
        "id": "P11-FE3",
        "pathway_id": "P11",
        "description": "Compute the causal effect of prefill geometry on downstream correctness using activation patching. Swap prefill activations between a correct-predicted and incorrect-predicted problem pair at L19. If swapping flips the outcome, the prefill signal is causal. If not, it's correlational.",
        "rationale": "All findings so far are correlational probes. Kudo et al. (2412.01113) used causal interventions to confirm models compute sub-answers during CoT. The same methodology applied to prefill would determine whether the 'can I solve this' signal actually drives behavior or merely correlates with it.",
        "trigger": "Methodological — activation patching infra exists.",
        "status": "READY",
        "blocked_by": "H100 pod for generation with interventions",
        "priority": "HIGH",
        "estimated_cost": "4h H100",
        "roi_score": 9,
        "depends_on_findings": ["F-2"],
        "would_update_findings": ["F-2"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2412.01113",
                "their_method": "Multi-step arithmetic + activation patching to determine when LLMs commit to an answer relative to CoT generation.",
                "their_result": "LLMs do not pre-determine the answer; they iteratively compute sub-answers during CoT — CoT is a faithful reflection of internal computation.",
                "our_method": "Apply the same activation-patching methodology to swap prefill activations between correct- and incorrect-predicted MATH-500 problem pairs at L19.",
                "same": "Same causal-intervention recipe (activation patching across CoT positions).",
                "differs": "We test prefill (not mid-CoT) and correctness (not arithmetic-answer faithfulness); the swap is between problem pairs, not corrupted-vs-clean copies of the same problem.",
            },
            {
                "arxiv_id": "2404.15255",
                "their_method": "Survey + best-practice tutorial on activation patching: how to choose metrics, how to interpret causal effects, common pitfalls.",
                "their_result": "Methodology document with no headline number — provides the right-tool guidance for circuit-level interventions.",
                "our_method": "Follow the tutorial's metric and validation guidance when designing the prefill-swap intervention so the causal claim is interpretable.",
                "same": "Same activation-patching methodology and same concern about how to interpret the result.",
                "differs": "We are users, not contributors; the focus is whether prefill geometry is causal for correctness rather than circuit identification.",
            },
        ],
    },
    {
        "id": "P11-FE4",
        "pathway_id": "P11",
        "description": "Characterize D-bucket (F-12) problems by mid-generation features. Extract L19 activations at token positions 50, 100, 150 for all 500 MATH-500 problems. Train a classifier to distinguish bucket D (K=1 right, K=8 wrong) from bucket A (K=1 right, K=8 right). If mid-generation features detect D-bucket, the signal appears during reasoning, not before.",
        "rationale": "F-12 showed D-bucket is undetectable from prefill features. But D-bucket is about self-consistency failure during generation — the model starts right and talks itself into a wrong answer. The signal should appear when the reasoning trajectory diverges.",
        "trigger": "Soft Self-Consistency (2402.13212) documented SC failure modes on diverse answers — same phenomenon.",
        "status": "READY",
        "blocked_by": "Per-token activations from K=8 runs (may need pod)",
        "priority": "MEDIUM",
        "estimated_cost": "4h H100 + 2h CPU",
        "roi_score": 7,
        "depends_on_findings": ["F-12", "F-3"],
        "would_update_findings": ["F-12"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2402.13212",
                "their_method": "Replaces self-consistency majority voting with a continuous likelihood-based score, enabling selection on long-horizon agentic tasks where answers are sparsely distributed.",
                "their_result": "+1.3% (bash), +6.6% (WebShop), +4.7% (ALFWorld) absolute success-rate gains; matches SC quality with half the samples.",
                "our_method": "Train a classifier on mid-generation L19 features (token positions 50/100/150) to detect bucket-D problems where K=1 right but K=8 wrong (self-consistency failure).",
                "same": "Same target (self-consistency failure modes when answers diverge); same diagnosis that majority voting masks signal.",
                "differs": "Probe-on-hidden-states, not output-likelihood scoring; benchmark is MATH-500 K=8 sampling (not interactive agents); detection target is the bucket-D self-consistency-failure case (F-12).",
            },
        ],
    },
    {
        "id": "P11-FE5",
        "pathway_id": "P11",
        "description": "Use DeepSeek-R1-Distill-Qwen-1.5B (same architecture as project's base model, but distilled from R1 with long-CoT training) as a comparison model. Extract L19 activations on MATH-500. Compare: (a) Does breathing still occur? (b) Is the asymmetric collapse stronger? (c) Does the prefill signal improve (R1-Distill has 83.9% accuracy vs base 48.6%)? (d) Does the DoM direction transfer between base and R1-Distill?",
        "rationale": "R1-Distill is the same architecture with massively improved reasoning via SFT on R1 traces. Comparing base vs R1-Distill isolates the effect of reasoning training on the hidden-state geometry. If breathing/collapse are stronger in R1-Distill, they're tied to reasoning capability, not just architecture.",
        "trigger": "R1-Distill-Qwen-1.5B publicly available.",
        "status": "TRIGGERED",
        "blocked_by": "H100 pod",
        "priority": "HIGH",
        "estimated_cost": "4h H100 + 2h CPU",
        "roi_score": 9,
        "depends_on_findings": ["F-1", "F-2", "F-4"],
        "would_update_findings": ["F-1", "F-2", "F-4"],
        "triggered_by_papers": [
            {
                "arxiv_id": "2501.12948",
                "their_method": "Trains long-CoT reasoning capability via pure reinforcement learning without human-annotated reasoning trajectories; the distilled R1-Distill series transfers this to smaller backbones via SFT on R1 traces.",
                "their_result": "Emergent self-reflection and verification behaviors; R1-Distill-Qwen-1.5B reaches 83.9% on MATH-500 vs 48.6% for the base.",
                "our_method": "Run breathing / collapse / DoM analysis on R1-Distill-Qwen-1.5B and compare against the base Qwen-2.5-1.5B on MATH-500.",
                "same": "Same architecture; same MATH-500 benchmark; same residual-stream geometry analysis pipeline.",
                "differs": "Model is the SFT'd R1-Distill (not RL-on-base); we ask whether reasoning training amplifies F-1 / F-2 / F-4 or leaves them unchanged — a comparison the original R1 paper does not perform.",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Seed logic
# ---------------------------------------------------------------------------

def reset(session) -> None:
    """Wipe FutureExperiment nodes and their relationships only."""
    session.run("MATCH (fe:FutureExperiment) DETACH DELETE fe")


def seed_paper_stubs(session) -> int:
    """Create paper stubs that future experiments reference, if missing."""
    created = 0
    for p in NEW_PAPER_STUBS:
        result = session.run(
            """
            MERGE (paper:Paper {arxiv_id: $a})
            ON CREATE SET paper.title = $title, paper.year = $year,
                          paper.repo_url = $repo, paper._created_by = 'seed_future_experiments'
            ON MATCH  SET paper.title = coalesce(paper.title, $title),
                          paper.year = coalesce(paper.year, $year),
                          paper.repo_url = coalesce(paper.repo_url, $repo)
            RETURN paper._created_by AS creator
            """,
            a=p["arxiv_id"], title=p["title"], year=p["year"], repo=p["repo_url"],
        )
        rec = result.single()
        if rec and rec["creator"] == "seed_future_experiments":
            created += 1
    return created


def seed_future_experiments(session) -> None:
    for fe in FUTURE_EXPERIMENTS:
        session.run(
            """
            MERGE (fe:FutureExperiment {id: $id})
            SET fe.pathway_id = $pathway_id,
                fe.description = $description,
                fe.rationale = $rationale,
                fe.trigger = $trigger,
                fe.status = $status,
                fe.blocked_by = $blocked_by,
                fe.priority = $priority,
                fe.estimated_cost = $estimated_cost,
                fe.roi_score = $roi_score,
                fe.created_date = coalesce(fe.created_date, $today),
                fe.completed_date = coalesce(fe.completed_date, null),
                fe.outcome = coalesce(fe.outcome, null)
            """,
            id=fe["id"],
            pathway_id=fe["pathway_id"],
            description=fe["description"],
            rationale=fe["rationale"],
            trigger=fe["trigger"],
            status=fe["status"],
            blocked_by=fe["blocked_by"],
            priority=fe["priority"],
            estimated_cost=fe["estimated_cost"],
            roi_score=fe["roi_score"],
            today=TODAY,
        )

        # Pathway -> FutureExperiment
        session.run(
            """
            MATCH (p:Pathway {id: $pid}), (fe:FutureExperiment {id: $fid})
            MERGE (p)-[:HAS_FUTURE_EXPERIMENT]->(fe)
            """,
            pid=fe["pathway_id"], fid=fe["id"],
        )

        # FutureExperiment -[:DEPENDS_ON_FINDING]-> Finding
        for f_id in fe.get("depends_on_findings", []):
            session.run(
                """
                MATCH (fe:FutureExperiment {id: $fid}), (f:Finding {id: $f_id})
                MERGE (fe)-[r:DEPENDS_ON_FINDING]->(f)
                """,
                fid=fe["id"], f_id=f_id,
            )

        # FutureExperiment -[:WOULD_UPDATE]-> Finding
        for f_id in fe.get("would_update_findings", []):
            session.run(
                """
                MATCH (fe:FutureExperiment {id: $fid}), (f:Finding {id: $f_id})
                MERGE (fe)-[r:WOULD_UPDATE]->(f)
                """,
                fid=fe["id"], f_id=f_id,
            )

        # FutureExperiment -[:TRIGGERED_BY]-> Paper
        # Each entry is either a bare arxiv_id (legacy) or a dict carrying
        # comparison fields (their_method, their_result, our_method, same, differs).
        for entry in fe.get("triggered_by_papers", []):
            if isinstance(entry, str):
                arxiv_id, props = entry, {}
            else:
                arxiv_id = entry["arxiv_id"]
                props = {k: entry.get(k) for k in
                         ("their_method", "their_result", "our_method", "same", "differs")}
            session.run(
                """
                MATCH (fe:FutureExperiment {id: $fid}), (p:Paper {arxiv_id: $a})
                MERGE (fe)-[r:TRIGGERED_BY]->(p)
                SET r.their_method = $their_method,
                    r.their_result = $their_result,
                    r.our_method   = $our_method,
                    r.same         = $same,
                    r.differs      = $differs
                """,
                fid=fe["id"], a=arxiv_id,
                their_method=props.get("their_method"),
                their_result=props.get("their_result"),
                our_method=props.get("our_method"),
                same=props.get("same"),
                differs=props.get("differs"),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                        help="Wipe FutureExperiment nodes before seeding (does not touch Pathway/Finding/Paper nodes)")
    args = parser.parse_args()

    drv = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    with drv.session() as session:
        if args.reset:
            print("[reset] wiping FutureExperiment nodes...")
            reset(session)

        new_papers = seed_paper_stubs(session)
        print(f"[papers] {new_papers} new stubs created (the rest already existed)")

        seed_future_experiments(session)
        print(f"[future-experiments] {len(FUTURE_EXPERIMENTS)} nodes upserted")

        # quick verification
        result = session.run("MATCH (fe:FutureExperiment) RETURN count(fe) AS n").single()
        print(f"[verify] FutureExperiment count: {result['n']}")
        result = session.run(
            "MATCH (:Pathway)-[r:HAS_FUTURE_EXPERIMENT]->(:FutureExperiment) RETURN count(r) AS n"
        ).single()
        print(f"[verify] HAS_FUTURE_EXPERIMENT edges: {result['n']}")
        result = session.run(
            "MATCH (:FutureExperiment)-[r:TRIGGERED_BY]->(:Paper) RETURN count(r) AS n"
        ).single()
        print(f"[verify] TRIGGERED_BY edges: {result['n']}")
        result = session.run(
            "MATCH (:FutureExperiment)-[r:DEPENDS_ON_FINDING]->(:Finding) RETURN count(r) AS n"
        ).single()
        print(f"[verify] DEPENDS_ON_FINDING edges: {result['n']}")
        result = session.run(
            "MATCH (:FutureExperiment)-[r:WOULD_UPDATE]->(:Finding) RETURN count(r) AS n"
        ).single()
        print(f"[verify] WOULD_UPDATE edges: {result['n']}")
    drv.close()


if __name__ == "__main__":
    main()
