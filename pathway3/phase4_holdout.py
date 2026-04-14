#!/usr/bin/env python3
"""Phase 4: Holdout Evaluation & Mechanistic Analysis.

Apply the locked configuration to the original Pathway 1 holdout-100.
Compare against Track A results. Run mechanistic analysis if improved.

Input: pathway3/phase3/locked_config.json, all prior artifacts
Output: pathway3/phase4/ — holdout_results.json, per_problem_analysis.json,
        comparison_with_track_a.txt, statistical_tests.json, mechanistic/

Runtime: ~3 hours GPU.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    ADDITIONAL_LAYERS,
    BOOTSTRAP_SAMPLES,
    MULTI_LAYER_CONFIGS,
    N_PROBLEMS,
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE3_DIR,
    PHASE4_DIR,
    SEED,
    TRACK_A_BEST_T,
    TRACK_A_BEST_TAU,
    TRACK_A_LAYER,
    TRACK_A_PHASE0,
    TRACK_A_PHASE1,
    TRACK_A_PHASE3,
    bayesian_p_improvement,
    bootstrap_ci_net_gain,
    bootstrap_mean_diff_vector,
    check_correct,
    cluster_differences,
    compute_flip_metrics,
    compute_prototype_steering_vector,
    extract_answer,
    fit_confidence_pipeline,
    generate_one,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    load_model,
    load_trajectory_meta,
    logger,
    make_steering_hook,
    mcnemar_mid_p,
    pca_project_vector,
    predict_confidence,
    register_multi_layer_hooks,
    remove_all_hooks,
    soft_gate_strength,
    sparsify_vector,
)

EXPANDED_ACT_DIR = PHASE1_DIR / "expanded_activations"
ADDITIONAL_ACT_DIR = PHASE1_DIR / "additional_activations"
MECHANISTIC_DIR = PHASE4_DIR / "mechanistic"


def load_locked_config() -> dict:
    """Load the locked configuration from Phase 3."""
    with open(PHASE3_DIR / "locked_config.json") as f:
        return json.load(f)


def prepare_steering_from_full_train(
    locked: dict,
    train_idx: np.ndarray,
    all_activations: dict[int, np.ndarray],
    baseline_correct: np.ndarray,
    y_all: np.ndarray,
) -> dict:
    """Prepare steering vectors using ALL 400 train problems (not a CV fold).

    This gives the vectors maximum data for the final holdout evaluation.
    """
    config_name = locked["selected_config"]
    layers = locked.get("layers", [TRACK_A_LAYER])
    pca_k = locked.get("pca_k", 10)
    sparse_d = locked.get("sparse_d", 100)
    proto_K = locked.get("proto_K", None)

    # Load expanded correct activations for train problems
    correct_indices = np.load(EXPANDED_ACT_DIR / "correct_problem_indices.npy")
    train_set = set(train_idx.tolist())
    mask = np.array([idx in train_set for idx in correct_indices])

    incorrect_mask = ~baseline_correct[train_idx].astype(bool)

    if "config_1" in config_name:
        # Track A baseline: simple mean-diff at layer 24, no bootstrap
        acts_24 = all_activations[24][train_idx]
        correct_train = baseline_correct[train_idx].astype(bool)
        H_correct = acts_24[correct_train]
        H_incorrect = acts_24[~correct_train]

        rng = np.random.default_rng(SEED)
        n_correct = len(H_correct)
        sampled_idx = rng.choice(len(H_incorrect), size=min(n_correct, len(H_incorrect)), replace=False)
        v = H_correct.mean(axis=0) - H_incorrect[sampled_idx].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)

        return {"vectors": {24: v}, "use_prototypes": False}

    elif "config_6" in config_name or proto_K is not None:
        # Full stack with prototypes
        proto_temp = locked.get("proto_temp", 0.5)
        prototypes_per_layer = {}

        for layer in layers:
            expanded_path = EXPANDED_ACT_DIR / f"correct_activations_layer{layer}.npy"
            if expanded_path.exists():
                H_correct = np.load(expanded_path)[mask]
            else:
                H_correct = all_activations[layer][correct_indices[mask]]

            H_incorrect = all_activations[layer][train_idx][incorrect_mask]
            H_incorrect_mean = H_incorrect.mean(axis=0)

            if proto_K and len(H_correct) >= proto_K * 5:
                diff_matrix = H_correct - H_incorrect_mean[np.newaxis, :]
                centroids, _, quality = cluster_differences(diff_matrix, proto_K, seed=SEED)
                prototypes_per_layer[layer] = centroids
            else:
                v = bootstrap_mean_diff_vector(H_correct, H_incorrect_mean)
                diff_matrix = H_correct - H_incorrect_mean[np.newaxis, :]
                k = min(pca_k, *diff_matrix.shape)
                v = pca_project_vector(v, diff_matrix, k)
                v = sparsify_vector(v, sparse_d)
                prototypes_per_layer[layer] = v[np.newaxis, :]

        return {"prototypes": prototypes_per_layer, "use_prototypes": True, "proto_temp": proto_temp}

    else:
        # Configs 2, 4: denoised vectors without prototypes
        vectors = {}
        for layer in layers:
            expanded_path = EXPANDED_ACT_DIR / f"correct_activations_layer{layer}.npy"
            if expanded_path.exists():
                H_correct = np.load(expanded_path)[mask]
            else:
                H_correct = all_activations[layer][correct_indices[mask]]

            H_incorrect = all_activations[layer][train_idx][incorrect_mask]
            H_incorrect_mean = H_incorrect.mean(axis=0)

            v = bootstrap_mean_diff_vector(H_correct, H_incorrect_mean)
            diff_matrix = H_correct - H_incorrect_mean[np.newaxis, :]
            k = min(pca_k, *diff_matrix.shape)
            v = pca_project_vector(v, diff_matrix, k)
            v = sparsify_vector(v, sparse_d)
            vectors[layer] = v

        return {"vectors": vectors, "use_prototypes": False}


def run_holdout_evaluation(
    locked: dict,
    prepared: dict,
    model,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    holdout_idx: np.ndarray,
    all_activations: dict[int, np.ndarray],
    baseline_correct: np.ndarray,
    confidence_holdout: np.ndarray,
) -> dict:
    """Run the locked config on holdout-100.

    Returns detailed results dict.
    """
    config_name = locked["selected_config"]
    tau = locked.get("tau", TRACK_A_BEST_TAU)
    t_base = locked.get("t_base", TRACK_A_BEST_T)
    gating = "soft" if "config_6" in config_name else "hard"
    sharpness = locked.get("sigmoid_sharpness", 10.0)
    layers = locked.get("layers", [TRACK_A_LAYER])

    n_holdout = len(holdout_idx)
    steered_correct = np.zeros(n_holdout, dtype=int)
    steered_flags = np.zeros(n_holdout, dtype=int)
    steering_strengths = np.zeros(n_holdout, dtype=float)
    per_problem = []

    for i, idx in enumerate(holdout_idx):
        confidence = float(confidence_holdout[i])

        # Gating
        if gating == "soft":
            t_val = soft_gate_strength(confidence, t_base, tau, sharpness)
            should_steer = t_val > 0.01
        else:
            should_steer = confidence < tau
            t_val = t_base if should_steer else 0.0

        if should_steer:
            if prepared["use_prototypes"]:
                layer_vecs = {}
                for layer, protos in prepared["prototypes"].items():
                    h_input = all_activations[layer][idx]
                    if protos.shape[0] == 1:
                        layer_vecs[layer] = protos[0]
                    else:
                        layer_vecs[layer] = compute_prototype_steering_vector(
                            h_input, protos, prepared.get("proto_temp", 0.5)
                        )
            else:
                layer_vecs = prepared["vectors"]

            handles = register_multi_layer_hooks(model, layer_vecs, t_val)
            text = generate_one(model, tokenizer, prompts[idx])
            remove_all_hooks(handles)

            predicted = extract_answer(text)
            is_correct = check_correct(predicted, ground_truths[idx])
            steered_correct[i] = int(is_correct)
            steered_flags[i] = 1
            steering_strengths[i] = t_val
        else:
            steered_correct[i] = int(baseline_correct[idx])
            steered_flags[i] = 0

        per_problem.append({
            "holdout_pos": i,
            "global_idx": int(idx),
            "confidence": round(confidence, 4),
            "steered": int(steered_flags[i]),
            "steering_strength": round(float(steering_strengths[i]), 4),
            "correct": int(steered_correct[i]),
            "baseline_correct": int(baseline_correct[idx]),
            "ground_truth": ground_truths[idx],
        })

        if (i + 1) % 10 == 0:
            logger.info("  Holdout: %d/%d done", i + 1, n_holdout)

    # Metrics
    baseline_holdout = baseline_correct[holdout_idx]
    metrics = compute_flip_metrics(baseline_holdout, steered_correct)

    # Statistical tests
    b = metrics["wrong_to_right"]
    c = metrics["right_to_wrong"]
    metrics["mcnemar_mid_p"] = mcnemar_mid_p(c, b)
    metrics["bayesian_p_improvement"] = bayesian_p_improvement(b, c)
    metrics["bootstrap_ci_95"] = list(bootstrap_ci_net_gain(baseline_holdout, steered_correct))

    # Steered subset
    steered_mask = steered_flags.astype(bool)
    metrics["n_steered"] = int(steered_mask.sum())
    if steered_mask.any():
        metrics["steered_accuracy"] = float(steered_correct[steered_mask].mean())
        metrics["steered_baseline_accuracy"] = float(baseline_holdout[steered_mask].mean())
    else:
        metrics["steered_accuracy"] = 0.0
        metrics["steered_baseline_accuracy"] = 0.0

    return {
        "config": config_name,
        "metrics": metrics,
        "per_problem": per_problem,
        "gating": gating,
        "tau": tau,
        "t_base": t_base,
        "layers": layers,
    }


def load_track_a_holdout_results() -> dict | None:
    """Load Track A Phase 3 holdout results for comparison."""
    path = TRACK_A_PHASE3 / "holdout_results.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_comparison_table(p3_results: dict, track_a_results: dict | None) -> str:
    """Generate comparison table between Track A and Pathway 3."""
    lines = []
    lines.append("=" * 70)
    lines.append("Comparison: Track A vs Pathway 3 (Holdout-100)")
    lines.append("=" * 70)
    lines.append("")

    m = p3_results["metrics"]

    # Extract Track A metrics
    if track_a_results:
        # Track A's targeted condition is the one to compare
        ta_conditions = track_a_results.get("conditions", {})
        ta_targeted = ta_conditions.get("targeted", {}).get("metrics", {})
        ta_baseline = ta_conditions.get("baseline", {}).get("metrics", {})
    else:
        ta_targeted = {}
        ta_baseline = {}

    lines.append(f"{'Metric':<35} {'Track A':>15} {'Pathway 3':>15}")
    lines.append(f"{'-' * 65}")
    lines.append(f"{'Net gain':<35} {ta_targeted.get('net_gain', '?'):>15} {m['net_gain']:>15}")
    lines.append(f"{'Accuracy':<35} {ta_targeted.get('overall_accuracy', '?'):>15} {m['steered_accuracy']:>15.3f}")
    lines.append(f"{'Wrong→right':<35} {ta_targeted.get('newly_correct', '?'):>15} {m['wrong_to_right']:>15}")
    lines.append(f"{'Right→wrong':<35} {ta_targeted.get('newly_wrong', '?'):>15} {m['right_to_wrong']:>15}")
    lines.append(f"{'Problems steered':<35} {ta_targeted.get('n_steered', '?'):>15} {m['n_steered']:>15}")
    lines.append(f"{'Steered subset accuracy':<35} {ta_targeted.get('steered_subset_accuracy', '?'):>15} {m['steered_accuracy']:>15.3f}")
    lines.append(f"{'McNemar mid-p':<35} {'?':>15} {m['mcnemar_mid_p']:>15.4f}")
    lines.append(f"{'Bayesian P(improvement)':<35} {'?':>15} {m['bayesian_p_improvement']:>15.4f}")
    lines.append(f"{'Bootstrap 95% CI':<35} {'?':>15} {str(m['bootstrap_ci_95']):>15}")

    lines.append("")
    lines.append(f"Pathway 3 config: {p3_results['config']}")
    lines.append(f"Gating: {p3_results['gating']}")
    lines.append(f"Layers: {p3_results['layers']}")

    return "\n".join(lines)


def analyze_per_problem_flips(
    p3_results: dict,
    track_a_per_problem_path: Path,
) -> str:
    """Compare per-problem flips between Track A and Pathway 3."""
    lines = []
    lines.append("\nPer-Problem Flip Comparison")
    lines.append("=" * 70)

    # Load Track A per-problem analysis
    if track_a_per_problem_path.exists():
        with open(track_a_per_problem_path) as f:
            ta_per_problem = json.load(f)
        ta_lookup = {p["global_idx"]: p for p in ta_per_problem}
    else:
        ta_lookup = {}

    p3_lookup = {p["global_idx"]: p for p in p3_results["per_problem"]}

    # Find problems that flipped differently
    all_indices = sorted(set(ta_lookup.keys()) | set(p3_lookup.keys()))

    different_flips = []
    for idx in all_indices:
        ta = ta_lookup.get(idx, {})
        p3 = p3_lookup.get(idx, {})

        ta_baseline = ta.get("baseline_correct", ta.get("correct_baseline", None))
        ta_steered = ta.get("targeted_correct", ta.get("correct_targeted", None))
        p3_baseline = p3.get("baseline_correct", None)
        p3_steered = p3.get("correct", None)

        if ta_steered is not None and p3_steered is not None:
            if ta_steered != p3_steered:
                different_flips.append({
                    "idx": idx,
                    "baseline": p3_baseline,
                    "track_a": ta_steered,
                    "pathway_3": p3_steered,
                    "p3_steered": p3.get("steered", 0),
                    "p3_confidence": p3.get("confidence", None),
                })

    lines.append(f"\nProblems that flipped differently: {len(different_flips)}")
    for d in different_flips:
        direction_ta = "W→R" if d["baseline"] == 0 and d["track_a"] == 1 else (
            "R→W" if d["baseline"] == 1 and d["track_a"] == 0 else "NC"
        )
        direction_p3 = "W→R" if d["baseline"] == 0 and d["pathway_3"] == 1 else (
            "R→W" if d["baseline"] == 1 and d["pathway_3"] == 0 else "NC"
        )
        lines.append(
            f"  idx={d['idx']}: baseline={d['baseline']}, "
            f"Track A={d['track_a']} ({direction_ta}), "
            f"P3={d['pathway_3']} ({direction_p3}), "
            f"steered={d['p3_steered']}, conf={d['p3_confidence']}"
        )

    return "\n".join(lines)


def run_mechanistic_analysis(
    model,
    tokenizer,
    prepared: dict,
    locked: dict,
    holdout_idx: np.ndarray,
    all_activations: dict[int, np.ndarray],
    p3_results: dict,
):
    """Run mechanistic analysis if results improved over Track A.

    Includes:
    (a) Logit-lens decomposition
    (b) Prototype analysis (if K>1)
    (c) Multi-layer ablation (if multi-layer)
    """
    MECHANISTIC_DIR.mkdir(parents=True, exist_ok=True)
    analysis = {}

    layers = locked.get("layers", [TRACK_A_LAYER])

    # (b) Prototype analysis
    if prepared["use_prototypes"]:
        logger.info("Running prototype analysis...")
        proto_analysis = {}

        for layer, protos in prepared["prototypes"].items():
            if protos.shape[0] <= 1:
                continue

            K = protos.shape[0]
            # Which prototype is selected for each holdout problem?
            selections = []
            for i, idx in enumerate(holdout_idx):
                h_input = all_activations[layer][idx]
                projections = protos @ h_input
                selected = int(np.argmax(projections))
                selections.append(selected)

            # Tally selections
            from collections import Counter
            counts = Counter(selections)
            proto_analysis[str(layer)] = {
                "K": K,
                "selection_counts": dict(counts),
                "most_frequent": counts.most_common(1)[0][0],
            }

        analysis["prototype_analysis"] = proto_analysis
        with open(MECHANISTIC_DIR / "prototype_analysis.json", "w") as f:
            json.dump(proto_analysis, f, indent=2)

    # (c) Multi-layer ablation
    if len(layers) > 1:
        logger.info("Running multi-layer ablation...")
        ablation = {}

        # For each layer, run steering WITHOUT that layer
        for ablated_layer in layers:
            remaining_layers = [l for l in layers if l != ablated_layer]

            if prepared["use_prototypes"]:
                remaining_protos = {l: prepared["prototypes"][l] for l in remaining_layers if l in prepared["prototypes"]}
                ablated_prepared = {"prototypes": remaining_protos, "use_prototypes": True, "proto_temp": prepared.get("proto_temp", 0.5)}
            else:
                remaining_vecs = {l: prepared["vectors"][l] for l in remaining_layers if l in prepared["vectors"]}
                ablated_prepared = {"vectors": remaining_vecs, "use_prototypes": False}

            # Quick eval: just count net gain on holdout
            # (This requires re-running generation — only for problems that were steered)
            steered_problems = [p for p in p3_results["per_problem"] if p["steered"]]
            n_steered = len(steered_problems)

            if n_steered == 0:
                ablation[str(ablated_layer)] = {"skip": "no steered problems"}
                continue

            correct_count = 0
            baseline_count = 0
            for p in steered_problems:
                idx = p["global_idx"]
                t_val = p["steering_strength"]

                if ablated_prepared["use_prototypes"]:
                    layer_vecs = {}
                    for layer, protos in ablated_prepared["prototypes"].items():
                        h_input = all_activations[layer][idx]
                        if protos.shape[0] == 1:
                            layer_vecs[layer] = protos[0]
                        else:
                            layer_vecs[layer] = compute_prototype_steering_vector(
                                h_input, protos, ablated_prepared.get("proto_temp", 0.5)
                            )
                else:
                    layer_vecs = ablated_prepared["vectors"]

                if len(layer_vecs) == 0:
                    # All layers ablated — use baseline
                    correct_count += p["baseline_correct"]
                else:
                    handles = register_multi_layer_hooks(model, layer_vecs, t_val)
                    text = generate_one(model, tokenizer, prompts[idx])
                    remove_all_hooks(handles)
                    predicted = extract_answer(text)
                    is_correct = check_correct(predicted, ground_truths[idx])
                    correct_count += int(is_correct)

                baseline_count += p["baseline_correct"]

            ablation[str(ablated_layer)] = {
                "remaining_layers": remaining_layers,
                "steered_correct": correct_count,
                "steered_total": n_steered,
                "steered_baseline_correct": baseline_count,
            }
            logger.info("  Ablated layer %d: %d/%d correct (baseline: %d/%d)",
                        ablated_layer, correct_count, n_steered, baseline_count, n_steered)

        analysis["multi_layer_ablation"] = ablation
        with open(MECHANISTIC_DIR / "multi_layer_ablation.json", "w") as f:
            json.dump(ablation, f, indent=2)

    # Save combined analysis
    with open(MECHANISTIC_DIR / "analysis_summary.json", "w") as f:
        json.dump(analysis, f, indent=2)

    return analysis


def go_no_go_decision(cv_results: dict, holdout_metrics: dict) -> dict:
    """Determine go/no-go for Pathway 4.

    Returns decision dict.
    """
    cv_mean = cv_results.get("mean_net_gain", 0)
    holdout_gain = holdout_metrics["net_gain"]
    p_improvement = holdout_metrics.get("bayesian_p_improvement", 0)

    if cv_mean >= 5 and holdout_gain >= 5:
        decision = "FULL_GO"
        rationale = (
            f"CV mean net gain ({cv_mean}) >= 5 AND holdout net gain ({holdout_gain}) >= 5. "
            "Generalization gap closed. Steering direction can serve as privileged information for distillation."
        )
    elif holdout_gain >= 3 and holdout_gain > 3:  # > Track A's +3
        decision = "CONDITIONAL_GO"
        rationale = (
            f"Holdout net gain ({holdout_gain}) >= 3 and improved over Track A (+3). "
            "Gap narrowed. Proceed to Pathway 4 using topo-confidence routing signal."
        )
    elif cv_mean >= 3 and holdout_gain >= 3 and p_improvement > 0.90:
        decision = "PUBLICATION_READY_STOP"
        rationale = (
            f"CV mean ({cv_mean}) >= 3, holdout ({holdout_gain}) >= 3, "
            f"P(improvement) ({p_improvement:.3f}) > 0.90. System works. "
            "Write up Pathway 1-3 arc as paper."
        )
    else:
        decision = "PIVOT"
        rationale = (
            f"CV mean ({cv_mean}), holdout ({holdout_gain}). "
            "Consider: (a) PRA step-wise scoring, (b) Bias-Only Adaptation via RL, "
            "(c) Accept routing-only result from Pathway 1."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "cv_mean_net_gain": cv_mean,
        "holdout_net_gain": holdout_gain,
        "bayesian_p_improvement": round(p_improvement, 4),
    }


def main():
    print("=" * 70)
    print("Phase 4: Holdout Evaluation & Mechanistic Analysis")
    print("=" * 70)

    # ============================================
    # Load locked config
    # ============================================
    print("\n[1] Loading locked configuration...")
    locked = load_locked_config()
    print(f"  Config: {locked['selected_config']}")
    print(f"  CV mean net gain: {locked['mean_net_gain']}")

    # ============================================
    # Verify holdout mask
    # ============================================
    print("\n[2] Verifying holdout mask...")
    train_idx, holdout_idx = get_train_holdout_indices()
    assert len(holdout_idx) == 100
    print(f"  Holdout: {len(holdout_idx)} problems (verified)")

    # Load data
    print("\n[3] Loading data...")
    problems = load_math500()
    # Make prompts and ground_truths available in outer scope for mechanistic analysis
    global prompts, ground_truths
    prompts, ground_truths = get_prompts_and_ground_truths(problems)
    assert len(prompts) == N_PROBLEMS

    baseline_correct = np.load(TRACK_A_PHASE0 / "baseline_correct.npy")
    meta = load_trajectory_meta()
    y_all = np.array(meta["correct"], dtype=int)

    # Load activations
    print("\n[4] Loading activations...")
    all_activations = {}
    for layer in [20, 24]:
        all_activations[layer] = np.load(TRACK_A_PHASE0 / "activations" / f"layer_{layer}.npy")
    for layer in ADDITIONAL_LAYERS:
        path = PHASE1_DIR / "additional_activations" / f"layer_{layer}.npy"
        if path.exists():
            all_activations[layer] = np.load(path)
    print(f"  Loaded layers: {sorted(all_activations.keys())}")

    # Fit confidence on full train-400
    print("\n[5] Fitting confidence model on full train-400...")
    scaler, clf, abc_names, abc_indices = fit_confidence_pipeline(train_idx, y_all)
    confidence_holdout = predict_confidence(scaler, clf, abc_indices, holdout_idx)

    # Load model
    print("\n[6] Loading model...")
    model, tokenizer = load_model()

    # ============================================
    # Prepare steering vectors from full train
    # ============================================
    print("\n[7] Preparing steering vectors from full train-400...")
    prepared = prepare_steering_from_full_train(
        locked, train_idx, all_activations, baseline_correct, y_all
    )
    if prepared["use_prototypes"]:
        for layer, protos in prepared["prototypes"].items():
            print(f"  Layer {layer}: {protos.shape[0]} prototypes")
    else:
        for layer, vec in prepared["vectors"].items():
            print(f"  Layer {layer}: single vector ||v||={np.linalg.norm(vec):.4f}")

    # ============================================
    # Run holdout evaluation
    # ============================================
    print("\n[8] Running holdout evaluation...")
    t0 = time.time()
    results = run_holdout_evaluation(
        locked, prepared, model, tokenizer,
        prompts, ground_truths, holdout_idx,
        all_activations, baseline_correct, confidence_holdout,
    )
    eval_time = time.time() - t0
    print(f"  Holdout evaluation: {eval_time:.0f}s")

    m = results["metrics"]
    print(f"\n  Net gain: {m['net_gain']}")
    print(f"  Accuracy: {m['steered_accuracy']:.3f} (baseline: {m['baseline_accuracy']:.3f})")
    print(f"  Wrong→right: {m['wrong_to_right']}, Right→wrong: {m['right_to_wrong']}")
    print(f"  Steered: {m['n_steered']}/{len(holdout_idx)}")
    print(f"  McNemar mid-p: {m['mcnemar_mid_p']:.4f}")
    print(f"  Bayesian P(improvement): {m['bayesian_p_improvement']:.4f}")
    print(f"  Bootstrap 95% CI: {m['bootstrap_ci_95']}")

    # ============================================
    # Save holdout results
    # ============================================
    with open(PHASE4_DIR / "holdout_results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(PHASE4_DIR / "per_problem_analysis.json", "w") as f:
        json.dump(results["per_problem"], f, indent=2)

    # ============================================
    # Compare with Track A
    # ============================================
    print("\n[9] Comparing with Track A...")
    track_a_results = load_track_a_holdout_results()
    comparison = generate_comparison_table(results, track_a_results)
    print(comparison)

    with open(PHASE4_DIR / "comparison_with_track_a.txt", "w") as f:
        f.write(comparison)

    # Per-problem flip comparison
    ta_per_problem_path = TRACK_A_PHASE3 / "per_problem_analysis.json"
    flip_comparison = analyze_per_problem_flips(results, ta_per_problem_path)
    print(flip_comparison)

    with open(PHASE4_DIR / "per_problem_analysis.txt", "w") as f:
        f.write(comparison + "\n\n" + flip_comparison)

    # Statistical tests summary
    stat_tests = {
        "mcnemar_mid_p": m["mcnemar_mid_p"],
        "bayesian_p_improvement": m["bayesian_p_improvement"],
        "bootstrap_ci_95": m["bootstrap_ci_95"],
        "wrong_to_right": m["wrong_to_right"],
        "right_to_wrong": m["right_to_wrong"],
        "net_gain": m["net_gain"],
    }
    with open(PHASE4_DIR / "statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2)

    # ============================================
    # Mechanistic analysis (if improved)
    # ============================================
    track_a_holdout_gain = 3  # Track A's holdout net gain
    if m["net_gain"] > track_a_holdout_gain:
        print(f"\n[10] Net gain improved ({m['net_gain']} > {track_a_holdout_gain}). Running mechanistic analysis...")
        analysis = run_mechanistic_analysis(
            model, tokenizer, prepared, locked,
            holdout_idx, all_activations, results,
        )
    else:
        print(f"\n[10] Net gain ({m['net_gain']}) did not improve over Track A ({track_a_holdout_gain}). Skipping mechanistic analysis.")

    # ============================================
    # Go/No-Go decision
    # ============================================
    print(f"\n{'=' * 70}")
    print("Go/No-Go Decision")
    print(f"{'=' * 70}")

    cv_results = locked.get("results", {})
    decision = go_no_go_decision(cv_results, m)

    print(f"\n  Decision: {decision['decision']}")
    print(f"  {decision['rationale']}")

    with open(PHASE4_DIR / "go_no_go_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    # ============================================
    # Final summary
    # ============================================
    print(f"\n{'=' * 70}")
    print("Phase 4 Summary")
    print(f"{'=' * 70}")
    print(f"  Config: {locked['selected_config']}")
    print(f"  CV mean net gain: {locked['mean_net_gain']}")
    print(f"  Holdout net gain: {m['net_gain']}")
    print(f"  Decision: {decision['decision']}")
    print(f"  Artifacts saved to {PHASE4_DIR}/")
    print(f"\n  Pathway 3 complete.")


if __name__ == "__main__":
    main()
