"""Seed the topo-confidence research graph.

Idempotent: re-running updates existing nodes via MERGE / property writes.
The hardcoded tables in this file are the authoritative source for the
graph; FINDINGS.md is a narrative companion that may drift.

Usage:
    python seed.py            # apply schema then seed
    python seed.py --reset    # also wipe nodes/edges first (does NOT touch link-forge)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

BOLT = os.environ.get("NEO4J_BOLT_URL", "bolt://localhost:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "topo_graph_dev")


# ---------------------------------------------------------------------------
# Pathways (P1..P11)
# ---------------------------------------------------------------------------

PATHWAYS: list[dict[str, Any]] = [
    {
        "id": "P1",
        "name": "Initial Exploration",
        "status": "CLOSED",
        "summary": "Early exploration of hidden-state geometry and topology as signals for model behavior.",
    },
    {
        "id": "P2",
        "name": "Steering Experiments (Early)",
        "status": "CLOSED",
        "summary": "First attempts at activation steering using cached vectors. Produced initial steering vector later shown to be unusable (cos=0.05 with correct direction).",
    },
    {
        "id": "P3",
        "name": "Steering for Accuracy",
        "status": "CLOSED",
        "summary": "Attempted to use steering vectors to improve accuracy. Mixed results; laid groundwork for understanding steering limitations.",
    },
    {
        "id": "P4",
        "name": "Feature Engineering",
        "status": "CLOSED",
        "summary": "Development of the ABC-44 feature pipeline combining geometric, topological, and trajectory features.",
    },
    {
        "id": "P5",
        "name": "Cross-Architecture",
        "status": "CLOSED",
        "summary": "Testing whether findings generalize across model architectures.",
    },
    {
        "id": "P6",
        "name": "Scaling Investigation",
        "status": "CLOSED",
        "summary": "Comparing 1.5B and 7B model behaviors and cross-scale transfer.",
    },
    {
        "id": "P7",
        "name": "Non-Euclidean Persistent Homology",
        "status": "CLOSED",
        "summary": "Applied PH with non-Euclidean metrics to hidden-state point clouds. NO-GO at 0.774 AUROC. Zigzag PH blocked on Dionysus2 install.",
    },
    {
        "id": "P8",
        "name": "Layer-wise Feature Extraction",
        "status": "CLOSED",
        "summary": "Extracted per-token, per-layer activations from Qwen2.5-1.5B on MATH-500 and BBH. Built CoE (60-dim), D2H (58-dim), layer-wise PH (168-dim), and TwoNN (28-dim) feature families.",
    },
    {
        "id": "P9",
        "name": "Comparative Experiment Battery",
        "status": "CLOSED",
        "summary": "5 experiments comparing feature families. CoE-60 won at 0.811 AUROC. PH collapsed to Gaussian null. CoE transfers cross-domain; PH and D2H do not (or asymmetrically). Length confound quantified at -0.050.",
        "handoff_doc": "pathway9/handoff.md",
    },
    {
        "id": "P10",
        "name": "Model Improvement Planning",
        "status": "ACTIVE",
        "summary": "Planning doc for using hidden-state signals to improve Qwen2.5-1.5B. E1 (steering), E2 (confidence-gated compute), E3 (refusal), E4 (distillation). v2 replaced v1.",
    },
    {
        "id": "P11",
        "name": "H100 Pod Session + Breakthroughs",
        "status": "ACTIVE",
        "summary": "Discovered truncation artifact (20.8% was wrong, actual 48.6%). Found dimensional breathing, prefill-knows-best, direction rotation, asymmetric collapse. Selective prediction landed at 71.6%@50% coverage.",
        "handoff_doc": "pathway11_h100/HANDOFF_compact.md",
    },
]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

EXPERIMENTS: list[dict[str, Any]] = [
    # Pathway 9 ---------------------------------------------------------
    {
        "id": "P9-E1",
        "pathway_id": "P9",
        "name": "Length Deconfound",
        "hypothesis": "ABC-44 features encode raw output length; deconfounding will lower AUROC if length is the dominant signal.",
        "result": "Deconfounding drops AUROC by 0.050 (0.796 -> 0.746). Several engineered features have r>0.6 with length.",
        "verdict": "SIGNIFICANT",
        "result_json": "pathway9/results/exp1_length_deconfound.json",
    },
    {
        "id": "P9-E2",
        "pathway_id": "P9",
        "name": "Orthogonality / Method Comparison",
        "hypothesis": "PH-168 carries information complementary to CoE-60.",
        "result": "Stacked LR ensemble adds +0.001 over CoE alone; correlation between probes r=0.388.",
        "verdict": "REDUNDANT",
        "result_json": "pathway9/results/exp2_orthogonality.json",
    },
    {
        "id": "P9-E3a",
        "pathway_id": "P9",
        "name": "Gaussian Null on PH Summaries",
        "hypothesis": "PH summary statistics carry topology beyond what an empirical-covariance Gaussian captures.",
        "result": "Real PH AUROC 0.690 vs Gaussian-null PH AUROC 0.693 — indistinguishable.",
        "verdict": "NULL",
        "result_json": "pathway9/results/exp3a_gaussian_null_v2.json",
    },
    {
        "id": "P9-E3b",
        "pathway_id": "P9",
        "name": "Token Shuffle Control",
        "hypothesis": "Shuffling tokens within a prompt destroys the topology PH measures.",
        "result": "60% of features become indistinguishable from shuffled baseline; supports F-7.",
        "verdict": "NULL",
        "result_json": "pathway9/results/exp3b_token_shuffle.json",
    },
    {
        "id": "P9-E3c",
        "pathway_id": "P9",
        "name": "Count Control",
        "hypothesis": "Per-problem token count alone could explain PH features.",
        "result": "Count-matched null partially explains the PH signal.",
        "verdict": "MIXED",
        "result_json": "pathway9/results/exp3c_count_control.json",
    },
    {
        "id": "P9-E4",
        "pathway_id": "P9",
        "name": "XGBoost vs LR + O-information",
        "hypothesis": "Nonlinear classifiers extract signal that linear probes miss.",
        "result": "XGBoost matches LR within noise — features are essentially linear.",
        "verdict": "NULL",
        "result_json": "pathway9/results/exp4_xgboost_oinfo_v2.json",
    },
    {
        "id": "P9-E5",
        "pathway_id": "P9",
        "name": "Cross-Domain Transfer (MATH<->BBH)",
        "hypothesis": "CoE-60 transfers across domains symmetrically.",
        "result": "MATH->BBH 0.720, BBH->MATH 0.712; CoE is the only symmetric domain-invariant signal.",
        "verdict": "SIGNIFICANT",
        "result_json": "pathway9/results/exp5_cross_domain.json",
    },

    # Pathway 11 --------------------------------------------------------
    {
        "id": "P11-E1",
        "pathway_id": "P11",
        "name": "Truncation Discovery (baseline re-evaluation)",
        "hypothesis": "The 20.8% MATH-500 baseline reflects model capability, not generation budget.",
        "result": "Re-evaluation at 1024 tokens yields 48.6% (1.5B) and 73.2% (7B). 256-token truncation cut off two-thirds of correct trajectories.",
        "verdict": "INVALIDATED",
        "result_json": "pathway11_h100/prefill_gated_compute/results.json",
    },
    {
        "id": "P11-E2",
        "pathway_id": "P11",
        "name": "Dimensional Breathing",
        "hypothesis": "Participation ratio is approximately constant during chain-of-thought generation.",
        "result": "PR rises from prefill ~20 to peak ~67-115, collapses to ~2-18 at the answer token. Correct collapses harder than incorrect.",
        "verdict": "SIGNIFICANT",
        "result_json": "pathway11_h100/exp1_cross_model/results.json",
    },
    {
        "id": "P11-E3",
        "pathway_id": "P11",
        "name": "Prefill Prediction",
        "hypothesis": "Final-token features predict correctness better than prefill features.",
        "result": "Prefill DoM AUROC 0.7731 (1.5B) and 0.876 (7B) — exceeds final-token DoM (0.7186, 1.5B). Pre-generation predicts post-generation.",
        "verdict": "SIGNIFICANT",
        "result_json": "pathway11_h100/prefill_gated_compute/results.json",
    },
    {
        "id": "P11-E4",
        "pathway_id": "P11",
        "name": "Direction Rotation",
        "hypothesis": "DoM direction is approximately fixed across positions (so cached steering vectors should work).",
        "result": "cos(intermediate_DoM, final_DoM) stays in [0.00, 0.20] across positions; reaches 0.37 only by position 200. Pathway-2 cached vector cos=0.046.",
        "verdict": "INVALIDATED",
        "result_json": "scratch/pathway10_temporal_and_verifier_results.json",
    },
    {
        "id": "P11-E5",
        "pathway_id": "P11",
        "name": "Raw DoM vs Engineered Features",
        "hypothesis": "ABC-44 engineered pipeline beats single-direction probes.",
        "result": "Two raw features (prefill DoM + final DoM) -> 0.794 AUROC; ABC-44 -> 0.796. Engineering adds nothing.",
        "verdict": "REDUNDANT",
        "result_json": "pathway11_h100/results/raw_dom_vs_abc44.json",
    },
    {
        "id": "P11-E6",
        "pathway_id": "P11",
        "name": "Cross-Scale Comparison (1024-tok labels)",
        "hypothesis": "7B activations predict 1.5B correctness better than 1.5B activations themselves (verifier hypothesis).",
        "result": "With 1024-tok labels the gap closes: 0.717 (7B->1.5B) vs 0.719 (1.5B->1.5B). Rank-correlation rho=0.864.",
        "verdict": "INVALIDATED",
        "result_json": "pathway11_h100/results/cross_scale_1024tok.json",
    },
    {
        "id": "P11-E7",
        "pathway_id": "P11",
        "name": "Compute Allocation / Self-Consistency (K=8)",
        "hypothesis": "Monotonic prefill-DoM gating allocates K best.",
        "result": "Monotonic fails: bucket-B recoverable problems live at mid-confidence, not low. Middle-heavy gating beats monotonic by +3.4pp. Neg-seq-len Pareto-dominates DoM.",
        "verdict": "MIXED",
        "result_json": "pathway11_h100/results/k8_allocation.json",
    },
    {
        "id": "P11-E8",
        "pathway_id": "P11",
        "name": "Selective Prediction (refuse-and-spend)",
        "hypothesis": "Prefill DoM enables better-than-random selective prediction.",
        "result": "Top 50% by prefill score answer at 71.6% accuracy (avg K=2.5) vs 49.4% random-refuse and 48.6% K=1 unconditional.",
        "verdict": "SIGNIFICANT",
        "result_json": "pathway11_h100/prefill_gated_compute/results.json",
    },
    {
        "id": "P11-E9",
        "pathway_id": "P11",
        "name": "D-Bucket Discovery",
        "hypothesis": "Pathological problems (K=1 right, K=8 majority wrong) form a per-problem detectable cluster.",
        "result": "36/500 problems (7.2%) are D-bucket. Group-level prefill PR = 14.49 (lowest of any bucket); per-problem local PR is indistinguishable. Cluster exists collectively, dissolves per-problem.",
        "verdict": "MIXED",
        "result_json": "pathway11_h100/prefill_inversion/phase2_cross_scale.json",
    },
]


# ---------------------------------------------------------------------------
# Findings (authoritative seed — supersedes FINDINGS.md when the two diverge)
# ---------------------------------------------------------------------------

FINDINGS: list[dict[str, Any]] = [
    {
        "id": "F-1",
        "claim": "During chain-of-thought, participation ratio inflates from ~20 to ~67-88, then collapses to ~8-6. Correct answers compress to PR≈4.3/2.6, incorrect to PR≈8.5/6.9. Collapse magnitude scales with model size.",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P11-E2"],
        "controls_passed": [
            "Cross-architecture replication (Phi-3-mini, Llama-3.2-1B)",
            "Gibberish null (no breathing)",
            "No-CoT null (no breathing)",
            "Stream-of-consciousness null (flat PR)",
        ],
        "controls_pending": ["BBH temporal PR curve"],
        "strongest_counterargument": "PR may be driven by token-count increase, not geometric structure",
        "would_be_overturned_by": "Showing breathing occurs equally for random token sequences of matched length",
    },
    {
        "id": "F-2",
        "claim": "Prefill hidden state (position 0) predicts correctness at 0.771 AUROC (1.5B) and 0.876 AUROC (7B), higher than final-token signals. The model can recognize from its initial representation whether it will get the answer wrong. Prefill direction is orthogonal to final-token direction (cos=-0.06).",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P11-E3"],
        "controls_passed": ["Both model scales show same pattern", "5-fold CV"],
        "controls_pending": ["Cross-architecture prefill test"],
        "strongest_counterargument": "Prefill may encode training-set familiarity, not problem structure",
        "would_be_overturned_by": "Showing prefill AUROC perfectly correlates with MATH difficulty level metadata (familiarity proxy)",
    },
    {
        "id": "F-3",
        "claim": "The DoM direction rotates continuously during generation — cosine between any intermediate-position DoM and final-token DoM stays below 0.2 at every measured position. This kills fixed-vector steering.",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P11-E4"],
        "controls_passed": [
            "Measured across 200+ positions",
            "Pre-cached pathway 2 vector confirmed unusable (cos=0.05)",
        ],
        "controls_pending": ["Temperature interaction"],
        "strongest_counterargument": "Rotation may be an artifact of mean-pooling over heterogeneous token types",
        "would_be_overturned_by": "Showing rotation disappears when DoM is computed on answer-tokens only",
    },
    {
        "id": "F-4",
        "claim": "A single DoM probe at L19 achieves 0.719 AUROC — hidden-state geometry predicts LLM correctness with minimal features. Two features (prefill DoM + final-token DoM) at 0.794 match the 44-feature ABC pipeline at 0.796. Length confound is absent in raw activations (cos≈-0.09).",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P11-E5"],
        "controls_passed": ["XGBoost ≈ LR confirms linearity", "Length-direction cosine measured"],
        "controls_pending": [],
        "strongest_counterargument": "0.794 vs 0.796 may both be at the ceiling for this label distribution",
        "would_be_overturned_by": "A nonlinear probe significantly exceeding 0.80 on the same data",
    },
    {
        "id": "F-5",
        "claim": "With proper 1024-token labels, 7B does NOT predict 1.5B correctness better than 1.5B predicts itself (0.717 vs 0.719). Both encode the same difficulty axis (ρ=0.864).",
        "strength": "MODERATE",
        "status": "ACTIVE",
        "evidence": ["P11-E6"],
        "controls_passed": ["Rank correlation measured"],
        "controls_pending": ["Cross-architecture (non-Qwen) test"],
        "strongest_counterargument": "May be specific to Qwen family sharing training data",
        "would_be_overturned_by": "A non-Qwen 7B predicting Qwen 1.5B correctness significantly above 0.72",
    },
    {
        "id": "F-6",
        "claim": "CoE-60 achieves 0.811 AUROC on MATH-500 correctness. It is the only symmetric domain-invariant signal: MATH->BBH 0.720, BBH->MATH 0.712.",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P9-E2", "P9-E5"],
        "controls_passed": ["5-fold CV", "Cross-domain both directions"],
        "controls_pending": [
            "Length deconfound on CoE-60 specifically",
            "Test with 1024-token labels",
        ],
        "strongest_counterargument": "0.811 was measured with 256-token truncated labels (104/396 split) — may change with proper labels",
        "would_be_overturned_by": "CoE-60 AUROC dropping below 0.75 with 1024-token labels",
    },
    {
        "id": "F-7",
        "claim": "5 raw PH summary features are indistinguishable from a full-covariance Gaussian null (real 0.690 vs null 0.693). PH on these activations measures covariance structure, not topology.",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P9-E3a", "P9-E3b"],
        "controls_passed": [
            "Full-covariance SVD null",
            "Token shuffle (60% indistinguishable)",
            "Multiple null draws (5 per problem)",
        ],
        "controls_pending": [],
        "strongest_counterargument": "Only tested 5 summary features; the full persistence diagram may carry topology",
        "would_be_overturned_by": "Persistence landscapes or persistence images beating the Gaussian null by >0.03 AUROC",
    },
    {
        "id": "F-8",
        "claim": "Layer-wise PH-168 is redundant with CoE-60. Stacked ensemble gives +0.001 lift. Prediction correlation r=0.388.",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P9-E2"],
        "controls_passed": ["Stacked LR ensemble", "Simple average ensemble"],
        "controls_pending": [],
        "strongest_counterargument": "Redundancy might break under domain shift",
        "would_be_overturned_by": "PH+CoE ensemble lifting >0.03 on BBH transfer",
    },
    {
        "id": "F-9",
        "claim": "Raw output length significantly confounds ABC-44 features: deconfounding drops AUROC by 0.050 (0.796 -> 0.746). Several engineered features have r>0.6 with length. Shorter=correct.",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P9-E1"],
        "controls_passed": [
            "Two independent length measures (traj_rows, raw_token_count)",
            "Regression deconfounding",
        ],
        "controls_pending": ["LEACE-style erasure comparison"],
        "strongest_counterargument": "Length correlation may be causal (shorter reasoning = less chance to err), not confound",
        "would_be_overturned_by": "Showing length-deconfounded features still match non-deconfounded on a length-balanced subset",
    },
    {
        "id": "F-10",
        "claim": "TwoNN intrinsic dimension has near-zero predictive power for correctness (AUROC 0.407) despite clear layer-wise structure (ID 1.7 -> 6.1 -> decline).",
        "strength": "MODERATE",
        "status": "ACTIVE",
        "evidence": ["P9-E4"],
        "controls_passed": ["Ensemble with PH (marginal +0.020 lift)"],
        "controls_pending": [
            "Local LID estimation (GeoMLE)",
            "Per-token ID rather than per-problem",
        ],
        "strongest_counterargument": "Global TwoNN averages over heterogeneous local geometry; local LID may work",
        "would_be_overturned_by": "Local LID achieving >0.65 AUROC",
    },
    {
        "id": "F-11",
        "claim": "Selective prediction using prefill DoM: the model recognizes when it will get the answer wrong and abstains. Top 50% by confidence achieves 71.6% accuracy on answered subset, +22pp over random at matched coverage.",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P11-E8"],
        "controls_passed": ["Coverage sweep", "Comparison against random baseline"],
        "controls_pending": [
            "Comparison against verbalized confidence",
            "Comparison against max-logit",
        ],
        "strongest_counterargument": "May not hold on harder benchmarks (AIME, competition math)",
        "would_be_overturned_by": "Verbalized confidence matching or exceeding probe-based refusal at same coverage",
    },
    {
        "id": "F-12",
        "claim": "7.2% of MATH-500 problems (36/500) are D-bucket: K=1 correct but K=8 majority vote wrong. Not individually detectable from prefill features.",
        "strength": "MODERATE",
        "status": "ACTIVE",
        "evidence": ["P11-E9"],
        "controls_passed": ["Prefill feature ablation (DoM, seq_len, local PR all fail)"],
        "controls_pending": [
            "Per-sample K=8 diversity analysis",
            "Final-token features on D-bucket",
        ],
        "strongest_counterargument": "36 problems may be too few for reliable characterization",
        "would_be_overturned_by": "A mid-generation probe detecting D-bucket membership at >0.70 AUROC",
    },
    {
        "id": "F-13",
        "claim": "The 20.8% MATH-500 baseline for Qwen2.5-1.5B was a truncation artifact. Actual accuracy at 1024 tokens is 48.6%. This invalidated the label distribution (104/396) underlying all pathway 9 experiments.",
        "strength": "STRONG",
        "status": "ACTIVE",
        "evidence": ["P11-E1"],
        "controls_passed": [
            "Multiple generation lengths tested",
            "7B also re-evaluated (73.2% at 1024 tokens)",
        ],
        "controls_pending": [],
        "strongest_counterargument": "None — this is a measurement correction",
        "would_be_overturned_by": "Nothing — it's a direct re-measurement",
    },
    {
        "id": "F-14",
        "claim": "Monotonic prefill gating for self-consistency fails because recoverable problems (bucket B) live at mid-confidence, not low confidence. Sequence length is Pareto-dominant over DoM for compute allocation.",
        "strength": "MODERATE",
        "status": "ACTIVE",
        "evidence": ["P11-E7"],
        "controls_passed": ["Non-monotonic (middle-heavy) gating beats monotonic by +3.4pp"],
        "controls_pending": ["Multi-signal gating with finaltok features"],
        "strongest_counterargument": "Mid-confidence location may be specific to MATH-500 difficulty distribution",
        "would_be_overturned_by": "Monotonic gating working on a different benchmark with more uniform difficulty",
    },
]


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------

PAPERS: list[dict[str, Any]] = [
    # CoE / trajectory
    {"arxiv_id": "2410.13640", "title": "Chain-of-Embedding (CoE)", "year": 2024, "repo_url": "https://github.com/Alsace08/Chain-of-Embedding",
     "relevance_note": "Original CoE paper — chain-of-embedding method for correctness prediction via closed-form trajectory score from per-layer hidden states. Source for our 60-dim feature family."},
    {"arxiv_id": "2604.05655", "title": "LLM Reasoning as Trajectories", "year": 2026, "repo_url": None,
     "relevance_note": "Step-specific trajectory geometry as a reasoning signal — methodological parallel to dimensional breathing."},
    {"arxiv_id": "2510.10494", "title": "Tracing the Traces", "year": 2025, "repo_url": None,
     "relevance_note": "Trajectory probing for reasoning failure modes."},
    {"arxiv_id": "2507.06087", "title": "CoRE: Metacognition via CoE", "year": 2025, "repo_url": None,
     "relevance_note": "Uses CoE for metacognitive prediction — overlaps with selective-prediction framing."},

    # Steering
    {"arxiv_id": "2306.03341", "title": "Inference-Time Intervention (ITI)", "year": 2023, "repo_url": None,
     "relevance_note": "Canonical activation steering method with mass-mean direction for truthfulness and honesty (TruthfulQA doubles). Key paper in this project's steering picture. Reference baseline our orthogonality finding builds against."},
    {"arxiv_id": "2510.04309", "title": "PID Steering", "year": 2025, "repo_url": None,
     "relevance_note": "Proportional-integral-derivative controller for steering — directly extends the DoM rotation finding by adapting to direction drift during generation."},
    {"arxiv_id": "2604.19018", "title": "LQR Steering", "year": 2026, "repo_url": None,
     "relevance_note": "Linear-quadratic regulator for activation control."},
    {"arxiv_id": "2505.18706", "title": "Bias-Only Steering (Sinii)", "year": 2025, "repo_url": None,
     "relevance_note": "Trained per-layer biases instead of cached vectors — alternative response to rotation."},
    {"arxiv_id": "2504.07986", "title": "SEAL: Steerable Reasoning Calibration", "year": 2025, "repo_url": "https://github.com/VITA-Group/SEAL",
     "relevance_note": "Reasoning calibration via steering."},
    {"arxiv_id": "2506.18831", "title": "STU-PID Steering", "year": 2025, "repo_url": "https://github.com/arambharadwaj/pid_steering",
     "relevance_note": "PID steering on R1-Distill-Qwen-1.5B — same model family as ours."},
    {"arxiv_id": "2407.12404", "title": "Generalization of Steering Vectors (Tan)", "year": 2024, "repo_url": None,
     "relevance_note": "Shows context-dependence of steerability — explains our cached-vector failure."},
    {"arxiv_id": "2509.06608", "title": "Small Vectors Big Effects", "year": 2025, "repo_url": "https://github.com/corl-team/steering-reasoning",
     "relevance_note": "Steering effects concentrate on first generated token — consistent with our cos<0.2 past pos 1."},

    # Truth / geometry
    {"arxiv_id": "2310.06824", "title": "Geometry of Truth (Marks & Tegmark)", "year": 2023, "repo_url": "https://github.com/saprmarks/geometry-of-truth",
     "relevance_note": "Mass-mean (DoM) probes match LR for truth — validates our raw-DoM=engineered-features finding."},
    {"arxiv_id": "2410.02707", "title": "LLMs Know More Than They Show", "year": 2024, "repo_url": "https://github.com/technion-cs-nlp/LLMsKnow",
     "relevance_note": "Multifaceted answer-token encoding of correctness. Methodological analogue to prefill probing."},
    {"arxiv_id": "2402.18048", "title": "Truthfulness via Local ID", "year": 2024, "repo_url": "https://github.com/fanyin3639/Truthfulness-LID",
     "relevance_note": "Local intrinsic dimension predicts truthfulness — alternative to TwoNN that may revive our F-10."},

    # PH / TDA
    {"arxiv_id": "2410.11042", "title": "Persistent Topological Features in LLMs", "year": 2024, "repo_url": "https://github.com/RitaSciencePark/topo_llm",
     "relevance_note": "Applies PH to LLM activations without a Gaussian null — our F-7 challenges their positive results."},
    {"arxiv_id": "2111.13171", "title": "ID, PH and Generalization (Birdal)", "year": 2021, "repo_url": None,
     "relevance_note": "PH bound on parameter-space SGD trajectory — different object from per-input activation PH."},

    # Dispersion / eigenvalue
    {"arxiv_id": "2402.03744", "title": "INSIDE / EigenScore", "year": 2024, "repo_url": None,
     "relevance_note": "Multi-sample covariance eigenvalues for hallucination — methodological cousin of CoE-60 single-pass features."},
    {"arxiv_id": "2509.11569", "title": "D2HScore", "year": 2025, "repo_url": None,
     "relevance_note": "Dispersion-based confidence score — overlaps with our D2H-58 feature family."},
    {"arxiv_id": "2506.24106", "title": "Representation Dispersion Predictive Power", "year": 2025, "repo_url": None,
     "relevance_note": "General theory of why dispersion predicts model behavior."},

    # Calibration / selective prediction
    {"arxiv_id": "2406.15927", "title": "Semantic Entropy Probes (SEPs)", "year": 2024, "repo_url": "https://github.com/OATML/semantic-entropy-probes",
     "relevance_note": "Single-pass approximation of semantic entropy for abstention — direct comparator for F-11 selective prediction."},
    {"arxiv_id": "2505.21772", "title": "CCPS Calibration", "year": 2025, "repo_url": None,
     "relevance_note": "Calibration via cross-prompt similarity."},
    {"arxiv_id": "2604.20614", "title": "Too Sharp Too Sure (CalMO)", "year": 2026, "repo_url": None,
     "relevance_note": "Multi-objective calibration — sharpness vs coverage tradeoff frame."},

    # Neural collapse / geometry
    {"arxiv_id": "2405.17767", "title": "Linguistic Collapse", "year": 2024, "repo_url": "https://github.com/rhubarbwu/linguistic-collapse",
     "relevance_note": "Neural collapse at final layer scales with model size — matches our asymmetric-collapse-with-scale finding."},
    {"arxiv_id": "2302.00294", "title": "Geometry of Hidden Representations", "year": 2023, "repo_url": "https://github.com/diegodoimo/geometry_representations",
     "relevance_note": "ID hump profile across layers — explains representation dimensionality changes during generation. Depth-axis analogue of breathing."},
    {"arxiv_id": "2405.15471", "title": "High-Dim Abstraction Phase", "year": 2024, "repo_url": "https://github.com/chengemily1/id-llm-abstraction",
     "relevance_note": "Expand-then-compress along depth in GPT-2/Pythia/LLaMA — direct mechanism for breathing."},

    # Self-consistency / compute
    {"arxiv_id": "2203.11171", "title": "Self-Consistency (Wang 2022)", "year": 2022, "repo_url": None,
     "relevance_note": "Foundational K-sample majority voting baseline."},
    {"arxiv_id": "2502.06703", "title": "Can 1B Surpass 405B", "year": 2025, "repo_url": None,
     "relevance_note": "Test-time compute scaling on small models — same regime as our K=8 experiments."},
    {"arxiv_id": "2501.04519", "title": "rStar-Math", "year": 2025, "repo_url": "https://github.com/microsoft/rStar",
     "relevance_note": "Process-reward MCTS on small models — alternative compute allocation."},

    # Distillation
    {"arxiv_id": "2604.14084", "title": "TIP: Token Importance in OPD", "year": 2026, "repo_url": "https://github.com/HJSang/OPSD_OnPolicyDistillation",
     "relevance_note": "Token-importance-weighted on-policy distillation."},

    # MCR / attention sinks
    {"arxiv_id": "2510.06477", "title": "Attention Sinks = Compression Valleys (MCR)", "year": 2025, "repo_url": None,
     "relevance_note": "Mid-layer compression mechanism for attention sinks — geometric kin of breathing collapse."},

    # Prefill prediction
    {"arxiv_id": "2509.12886", "title": "The LLM Already Knows (Zhu)", "year": 2025, "repo_url": None,
     "relevance_note": "Initial hidden state (prefill direction) predicts correctness on Qwen2.5-VL-7B — independent corroboration of F-2 prefill DoM finding."},
    {"arxiv_id": "2504.05419", "title": "Reasoning Models Know When They're Right", "year": 2025, "repo_url": "https://github.com/AngelaZZZ-611/reasoning_models_probing",
     "relevance_note": "Mid-trajectory probes at chunk boundaries get >0.9 AUROC on AIME — proposed extension to F-2."},
    {"arxiv_id": "2510.18147", "title": "LLMs Encode Problem Difficulty", "year": 2025, "repo_url": None,
     "relevance_note": "Same model (Qwen2.5-Math-1.5B), prefill encodes difficulty rho=0.88 — strong corroborator of F-2. Calibration and confidence in LLM predictions from hidden states."},

    # Length
    {"arxiv_id": "2310.03716", "title": "A Long Way to Go (length in RLHF)", "year": 2024, "repo_url": None,
     "relevance_note": "Length-only reward reproduces most RLHF gains — output verbosity biases reward model scores. Feature-level analogue of F-9."},
    {"arxiv_id": "2505.00127", "title": "Between Underthinking and Overthinking", "year": 2025, "repo_url": None,
     "relevance_note": "Underthinking and overthinking in LLM reasoning — incorrect responses systematically longer, corroborating the output length confound. Direct support for shorter=correct in F-9."},

    # Linear representation
    {"arxiv_id": "2311.03658", "title": "Linear Representation Hypothesis", "year": 2023, "repo_url": "https://github.com/KihoPark/linear_rep_geometry",
     "relevance_note": "Theory of linear concept directions and truth probing in representations — explains why raw DoM matches engineered features. Foundational for methods probing truth in LLM representations."},
    {"arxiv_id": "2303.08112", "title": "Tuned Lens", "year": 2023, "repo_url": "https://github.com/AlignmentResearch/tuned-lens",
     "relevance_note": "Per-layer linear decoders — methodological inspiration for layer-wise probing."},
    {"arxiv_id": "2406.19384", "title": "Stages of Inference (Lad, Gurnee, Tegmark)", "year": 2024, "repo_url": None,
     "relevance_note": "Layer-stage decomposition of transformer computation."},

    # Conformal / abstention
    {"arxiv_id": "2402.10978", "title": "Conformal Factuality (Mohri & Hashimoto)", "year": 2024, "repo_url": "https://github.com/tatsu-lab/conformal-factual-lm",
     "relevance_note": "Conformal prediction for LLM outputs — distribution-free wrapper for our DoM probe."},
]


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------

# (finding_id, rel_type, paper_arxiv_id, properties)
PAPER_EDGES: list[tuple[str, str, str, dict[str, Any]]] = [
    # F-1 dimensional breathing
    ("F-1", "CORROBORATED_BY", "2405.15471",
     {"their_model": "GPT-2/Pythia/LLaMA",
      "note": "Same expand-compress along depth axis; we show it along generation time"}),
    ("F-1", "CORROBORATED_BY", "2405.17767",
     {"their_model": "Various CLMs",
      "note": "Neural collapse at final layer scales with model size — matches our commitment-scales-with-size"}),
    ("F-1", "EXPLAINS", "2302.00294",
     {"mechanism": "ID hump profile across layers predicts semantic content peaks — our breathing is the temporal analogue"}),
    ("F-1", "METHOD_DIFFERS", "2604.05655",
     {"theirs": "Step-specific trajectory geometry",
      "ours": "Participation ratio temporal curve"}),

    # F-2 prefill knows best
    ("F-2", "CORROBORATED_BY", "2509.12886",
     {"their_model": "Qwen2.5-VL-7B",
      "note": "Same conclusion: initial hidden state predicts correctness. They lack the orthogonality measurement."}),
    ("F-2", "CORROBORATED_BY", "2510.18147",
     {"their_model": "Qwen2.5-Math-1.5B",
      "note": "Exact same model — prefill encodes difficulty. rho=0.88 with human labels."}),
    ("F-2", "EXTENDED_BY", "2504.05419",
     {"experiment_idea": "Mid-trajectory probes at chunk boundaries get >0.9 AUROC on AIME — we should test chunk-boundary probing",
      "actionable": True}),
    ("F-2", "METHOD_DIFFERS", "2410.02707",
     {"theirs": "Exact-answer-token probes, multifaceted encoding",
      "ours": "Prefill DoM, orthogonality between prefill and final-token"}),

    # F-3 direction rotation
    ("F-3", "EXPLAINS", "2407.12404",
     {"mechanism": "Steerability is context-dependent; cached vectors are unreliable across distributions"}),
    ("F-3", "EXTENDED_BY", "2510.04309",
     {"experiment_idea": "PID steering accumulates error across layers to handle rotation — directly addresses our failure mode",
      "actionable": True}),
    ("F-3", "EXTENDED_BY", "2506.18831",
     {"experiment_idea": "STU-PID on our exact model (R1-Distill-Qwen-1.5B), handles rotation with I and D terms",
      "actionable": True}),
    ("F-3", "CORROBORATED_BY", "2509.06608",
     {"note": "SV effects concentrate on first generated token — consistent with our cos<0.2 finding past position 1"}),

    # F-4 raw DoM matches engineered features
    ("F-4", "CORROBORATED_BY", "2310.06824",
     {"note": "Mass-mean (DoM) probes match or exceed logistic regression for truth — same conclusion"}),
    ("F-4", "EXPLAINS", "2311.03658",
     {"mechanism": "Linear Representation Hypothesis: LLMs encode high-level concepts as linear directions"}),

    # F-6 CoE-60
    ("F-6", "CORROBORATED_BY", "2410.13640",
     {"note": "Original CoE paper — our 0.811 is the trained-classifier extension of their closed-form score"}),
    ("F-6", "METHOD_DIFFERS", "2402.03744",
     {"theirs": "EigenScore on multi-sample covariance (K=5)",
      "ours": "CoE-60 single-pass trajectory features"}),
    ("F-6", "METHOD_DIFFERS", "2406.15927",
     {"theirs": "Semantic Entropy Probes approximating SE from single generation",
      "ours": "CoE-60 geometric trajectory features"}),

    # F-7 PH Gaussian null
    ("F-7", "CORROBORATED_BY", "2410.11042",
     {"note": "They use PH on LLMs but don't test Gaussian null — our finding challenges their positive results"}),
    ("F-7", "EXPLAINS", "2111.13171",
     {"mechanism": "Birdal's PH bound operates on parameter-space SGD trajectory, not per-input activations — different object"}),

    # F-9 length confound
    ("F-9", "CORROBORATED_BY", "2310.03716",
     {"note": "Length-only reward reproduces most RLHF gains — same phenomenon at feature level"}),
    ("F-9", "CORROBORATED_BY", "2505.00127",
     {"note": "Incorrect responses systematically longer — direct support for shorter=correct"}),

    # F-10 TwoNN failure
    ("F-10", "CONTRADICTED_BY", "2402.18048",
     {"why": "They show local ID predicts truthfulness with 5-8 AUROC points — but they use GeoMLE, not TwoNN"}),
    ("F-10", "EXTENDED_BY", "2402.18048",
     {"experiment_idea": "Replace TwoNN with local LID (GeoMLE) — may recover predictive signal",
      "actionable": True}),

    # F-11 selective prediction
    ("F-11", "METHOD_DIFFERS", "2406.15927",
     {"theirs": "SEP-based abstention",
      "ours": "Prefill DoM-based refusal"}),
    ("F-11", "EXTENDED_BY", "2402.10978",
     {"experiment_idea": "Wrap our DoM probe in conformal prediction for distribution-free coverage guarantees",
      "actionable": True}),
]

# (src_finding, rel_type, dst_finding, properties)
INTERNAL_FINDING_EDGES: list[tuple[str, str, str, dict[str, Any]]] = [
    ("F-6", "DEPENDS_ON", "F-13",
     {"reason": "CoE-60 AUROC measured with truncated labels — may change with proper labels"}),
    ("F-13", "INVALIDATED_BY", "F-6",
     {"reason": "0.811 AUROC was measured with truncated labels (104/396). Needs re-evaluation with proper labels (243/257)."}),
    ("F-8", "DEPENDS_ON", "F-6",
     {"reason": "Redundancy claim depends on CoE being the strong baseline"}),
    ("F-9", "DEPENDS_ON", "F-13",
     {"reason": "Length confound magnitude may differ with proper labels"}),
    ("F-14", "DEPENDS_ON", "F-2",
     {"reason": "Monotonic gating uses prefill DoM — gating failure means prefill signal has wrong shape for allocation"}),
    ("F-11", "DEPENDS_ON", "F-2",
     {"reason": "Selective prediction uses prefill DoM"}),
    ("F-4", "SUPERSEDED_BY", "F-6",
     {"reason": "Raw DoM at L19 (0.719) is simpler but lower than CoE-60 (0.811) — though CoE needs re-evaluation"}),
]


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

TAGS: list[str] = [
    "steering", "calibration", "PH", "CoE", "D2H", "breathing", "prefill",
    "length-confound", "selective-prediction", "self-consistency",
    "cross-domain", "cross-scale", "intrinsic-dimension", "neural-collapse",
    "linear-probing", "test-time-compute", "distillation",
    "direction-rotation", "truncation-artifact",
]

FINDING_TAGS: dict[str, list[str]] = {
    "F-1":  ["breathing", "neural-collapse", "intrinsic-dimension"],
    "F-2":  ["prefill", "linear-probing", "calibration"],
    "F-3":  ["direction-rotation", "steering"],
    "F-4":  ["linear-probing", "prefill", "length-confound"],
    "F-5":  ["cross-scale", "prefill"],
    "F-6":  ["CoE", "cross-domain"],
    "F-7":  ["PH", "intrinsic-dimension"],
    "F-8":  ["PH", "CoE"],
    "F-9":  ["length-confound"],
    "F-10": ["intrinsic-dimension"],
    "F-11": ["selective-prediction", "calibration", "prefill"],
    "F-12": ["self-consistency", "test-time-compute"],
    "F-13": ["truncation-artifact"],
    "F-14": ["self-consistency", "test-time-compute", "prefill"],
}

PAPER_TAGS: dict[str, list[str]] = {
    "2410.13640": ["CoE", "linear-probing"],
    "2604.05655": ["CoE", "breathing"],
    "2510.10494": ["CoE"],
    "2507.06087": ["CoE", "calibration"],
    "2306.03341": ["steering", "linear-probing", "activation-steering", "activation-editing", "truthfulness"],
    "2510.04309": ["steering", "direction-rotation"],
    "2604.19018": ["steering"],
    "2505.18706": ["steering"],
    "2504.07986": ["steering", "calibration"],
    "2506.18831": ["steering", "direction-rotation"],
    "2407.12404": ["steering", "direction-rotation"],
    "2509.06608": ["steering", "direction-rotation"],
    "2310.06824": ["linear-probing", "calibration"],
    "2410.02707": ["prefill", "linear-probing"],
    "2402.18048": ["intrinsic-dimension", "calibration"],
    "2410.11042": ["PH"],
    "2111.13171": ["PH", "intrinsic-dimension"],
    "2402.03744": ["D2H", "calibration"],
    "2509.11569": ["D2H", "calibration"],
    "2506.24106": ["D2H"],
    "2406.15927": ["selective-prediction", "calibration"],
    "2505.21772": ["calibration"],
    "2604.20614": ["calibration", "selective-prediction"],
    "2405.17767": ["neural-collapse", "cross-scale"],
    "2302.00294": ["intrinsic-dimension", "neural-collapse"],
    "2405.15471": ["intrinsic-dimension", "breathing"],
    "2203.11171": ["self-consistency", "test-time-compute"],
    "2502.06703": ["test-time-compute"],
    "2501.04519": ["test-time-compute"],
    "2604.14084": ["distillation"],
    "2510.06477": ["neural-collapse", "breathing"],
    "2509.12886": ["prefill"],
    "2504.05419": ["prefill", "selective-prediction"],
    "2510.18147": ["prefill", "calibration"],
    "2310.03716": ["length-confound"],
    "2505.00127": ["length-confound"],
    "2311.03658": ["linear-probing", "truthfulness"],
    "2303.08112": ["linear-probing"],
    "2406.19384": ["linear-probing"],
    "2402.10978": ["selective-prediction", "calibration"],
}


# ---------------------------------------------------------------------------
# Cypher seeders (idempotent)
# ---------------------------------------------------------------------------

def apply_schema(session) -> None:
    schema_path = ROOT / "schema.cypher"
    text = schema_path.read_text()
    # cypher-shell-style multi-statement file; split on `;` at end-of-line.
    for stmt in [s.strip() for s in text.split(";")]:
        # skip pure comment / empty fragments
        meaningful = "\n".join(
            line for line in stmt.splitlines() if line.strip() and not line.strip().startswith("//")
        )
        if not meaningful:
            continue
        session.run(stmt)


def reset(session) -> None:
    session.run("MATCH (n) DETACH DELETE n")


def seed_pathways(session) -> int:
    for p in PATHWAYS:
        session.run(
            """
            MERGE (n:Pathway {id: $id})
            SET n.name = $name,
                n.status = $status,
                n.summary = $summary,
                n.handoff_doc = $handoff_doc
            """,
            id=p["id"],
            name=p["name"],
            status=p["status"],
            summary=p["summary"],
            handoff_doc=p.get("handoff_doc"),
        )
    # Linear NEXT chain
    for prev, nxt in zip(PATHWAYS[:-1], PATHWAYS[1:]):
        session.run(
            """
            MATCH (a:Pathway {id: $a}), (b:Pathway {id: $b})
            MERGE (a)-[:NEXT]->(b)
            """,
            a=prev["id"],
            b=nxt["id"],
        )
    return len(PATHWAYS)


def seed_experiments(session) -> int:
    for e in EXPERIMENTS:
        session.run(
            """
            MERGE (x:Experiment {id: $id})
            SET x.pathway_id = $pathway_id,
                x.name = $name,
                x.hypothesis = $hypothesis,
                x.result = $result,
                x.verdict = $verdict,
                x.result_json = $result_json
            WITH x
            MATCH (p:Pathway {id: $pathway_id})
            MERGE (p)-[:HAS_EXPERIMENT]->(x)
            """,
            id=e["id"],
            pathway_id=e["pathway_id"],
            name=e["name"],
            hypothesis=e["hypothesis"],
            result=e["result"],
            verdict=e["verdict"],
            result_json=e.get("result_json"),
        )
        # Artifact link if a result_json is named
        if e.get("result_json"):
            session.run(
                """
                MERGE (a:Artifact {path: $path})
                SET a.type = 'result_json',
                    a.description = coalesce(a.description, $desc)
                WITH a
                MATCH (x:Experiment {id: $id})
                MERGE (x)-[:PRODUCED_ARTIFACT]->(a)
                """,
                path=e["result_json"],
                desc=f"Result JSON for {e['id']} ({e['name']})",
                id=e["id"],
            )
    return len(EXPERIMENTS)


def seed_findings(session) -> int:
    for f in FINDINGS:
        session.run(
            """
            MERGE (n:Finding {id: $id})
            SET n.claim = $claim,
                n.strength = $strength,
                n.status = $status,
                n.evidence = $evidence,
                n.controls_passed = $controls_passed,
                n.controls_pending = $controls_pending,
                n.strongest_counterargument = $counterarg,
                n.would_be_overturned_by = $overturn
            """,
            id=f["id"],
            claim=f["claim"],
            strength=f["strength"],
            status=f["status"],
            evidence=f["evidence"],
            controls_passed=f["controls_passed"],
            controls_pending=f["controls_pending"],
            counterarg=f["strongest_counterargument"],
            overturn=f["would_be_overturned_by"],
        )
        # link Experiment -> Finding via PRODUCED
        for exp_id in f["evidence"]:
            session.run(
                """
                MATCH (x:Experiment {id: $exp_id}), (f:Finding {id: $f_id})
                MERGE (x)-[:PRODUCED]->(f)
                """,
                exp_id=exp_id,
                f_id=f["id"],
            )
    return len(FINDINGS)


def seed_papers(session) -> int:
    for p in PAPERS:
        session.run(
            """
            MERGE (n:Paper {arxiv_id: $arxiv_id})
            SET n.title = $title,
                n.year = $year,
                n.repo_url = $repo_url,
                n.relevance_note = $relevance_note
            """,
            arxiv_id=p["arxiv_id"],
            title=p["title"],
            year=p["year"],
            repo_url=p.get("repo_url"),
            relevance_note=p.get("relevance_note"),
        )
    return len(PAPERS)


def seed_paper_edges(session) -> int:
    n = 0
    for f_id, rel, arxiv_id, props in PAPER_EDGES:
        cypher = f"""
            MATCH (f:Finding {{id: $f_id}}), (p:Paper {{arxiv_id: $arxiv_id}})
            MERGE (f)-[r:{rel}]->(p)
            SET r += $props
        """
        session.run(cypher, f_id=f_id, arxiv_id=arxiv_id, props=props)
        n += 1
    return n


def seed_internal_finding_edges(session) -> int:
    n = 0
    for src, rel, dst, props in INTERNAL_FINDING_EDGES:
        cypher = f"""
            MATCH (a:Finding {{id: $src}}), (b:Finding {{id: $dst}})
            MERGE (a)-[r:{rel}]->(b)
            SET r += $props
        """
        session.run(cypher, src=src, dst=dst, props=props)
        n += 1
    return n


def seed_tags(session) -> int:
    for t in TAGS:
        session.run("MERGE (:Tag {name: $name})", name=t)
    n = 0
    for f_id, tag_list in FINDING_TAGS.items():
        for t in tag_list:
            session.run(
                """
                MATCH (f:Finding {id: $f_id}), (t:Tag {name: $name})
                MERGE (f)-[:TAGGED]->(t)
                """,
                f_id=f_id,
                name=t,
            )
            n += 1
    for arxiv_id, tag_list in PAPER_TAGS.items():
        for t in tag_list:
            session.run(
                """
                MATCH (p:Paper {arxiv_id: $arxiv_id}), (t:Tag {name: $name})
                MERGE (p)-[:TAGGED]->(t)
                """,
                arxiv_id=arxiv_id,
                name=t,
            )
            n += 1
    return n


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the research graph DB before seeding (does not touch link-forge).",
    )
    args = parser.parse_args()

    driver = GraphDatabase.driver(BOLT, auth=(USER, PASSWORD))
    try:
        with driver.session() as session:
            print(f"Connected to {BOLT}")
            if args.reset:
                print("Resetting graph...")
                reset(session)
            print("Applying schema...")
            apply_schema(session)
            n_p = seed_pathways(session)
            print(f"  Created {n_p} pathways")
            n_e = seed_experiments(session)
            print(f"  Created {n_e} experiments")
            n_f = seed_findings(session)
            print(f"  Created {n_f} findings")
            n_pa = seed_papers(session)
            print(f"  Created {n_pa} papers")
            n_ed = seed_paper_edges(session)
            print(f"  Created {n_ed} finding<->paper edges")
            n_in = seed_internal_finding_edges(session)
            print(f"  Created {n_in} internal finding<->finding edges")
            n_tg = seed_tags(session)
            print(f"  Tagged {n_tg} (finding|paper) nodes")
        print("Done.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
