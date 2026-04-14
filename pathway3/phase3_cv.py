#!/usr/bin/env python3
"""Phase 3: Cross-Validated Evaluation.

5-fold stratified CV replacing the underpowered single-holdout design.
Evaluates 4 priority configurations (1, 2, 4, 6) across all folds.
Selects and locks the best configuration before Phase 4.

Input: Phase 1 + Phase 2 artifacts, Track A artifacts
Output: pathway3/phase3/ — fold_assignments.json, cv_results.json,
        locked_config.json, per-fold checkpoints

Runtime: ~10+ hours GPU (4 configs × 5 folds × ~30 min each).
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from common import (
    ADDITIONAL_LAYERS,
    BOOTSTRAP_SAMPLES,
    CV_N_SPLITS,
    CV_RANDOM_STATE,
    MULTI_LAYER_CONFIGS,
    N_PROBLEMS,
    PHASE1_DIR,
    PHASE2_DIR,
    PHASE3_DIR,
    SEED,
    SIGMOID_SHARPNESS_SWEEP,
    TRACK_A_BEST_T,
    TRACK_A_BEST_TAU,
    TRACK_A_LAYER,
    TRACK_A_PHASE0,
    TRACK_A_PHASE1,
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

from sklearn.model_selection import StratifiedKFold

EXPANDED_ACT_DIR = PHASE1_DIR / "expanded_activations"
ADDITIONAL_ACT_DIR = PHASE1_DIR / "additional_activations"
DENOISED_DIR = PHASE2_DIR / "denoised_vectors"
PROTOTYPE_DIR = PHASE2_DIR / "prototypes"


# ============================================
# Configuration definitions
# ============================================


class SteeringConfig:
    """Represents a steering configuration to evaluate."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def prepare_fold(self, fold_train_idx, fold_test_idx, all_activations, baseline_correct, y_all):
        """Prepare steering vectors and confidence model for this fold.

        Returns dict with everything needed to steer test-fold problems.
        """
        raise NotImplementedError

    def should_steer(self, problem_idx, confidence, prepared):
        """Whether to steer this problem and at what strength."""
        raise NotImplementedError


class Config1_TrackABaseline(SteeringConfig):
    """Track A replication: single vector at layer 24, hard gate τ=0.40, t=0.15."""

    def __init__(self):
        super().__init__("config_1_track_a_baseline",
                         "Track A exact: layer 24, t=0.15, τ=0.40 hard gate")

    def prepare_fold(self, fold_train_idx, fold_test_idx, all_activations, baseline_correct, y_all):
        # Extract steering vector from fold's train set at layer 24
        acts_24 = all_activations[24][fold_train_idx]
        correct_train = baseline_correct[fold_train_idx].astype(bool)

        H_correct = acts_24[correct_train]
        H_incorrect = acts_24[~correct_train]

        if len(H_correct) == 0:
            return None

        # Mean-diff (Track A style, no bootstrap)
        rng = np.random.default_rng(SEED)
        n_correct = len(H_correct)
        sampled_idx = rng.choice(len(H_incorrect), size=min(n_correct, len(H_incorrect)), replace=False)
        v = H_correct.mean(axis=0) - H_incorrect[sampled_idx].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)

        # Fit confidence model
        scaler, clf, abc_names, abc_indices = fit_confidence_pipeline(fold_train_idx, y_all)
        conf_test = predict_confidence(scaler, clf, abc_indices, fold_test_idx)

        return {
            "vectors": {24: v},
            "t": TRACK_A_BEST_T,
            "tau": TRACK_A_BEST_TAU,
            "confidence_test": conf_test,
            "gating": "hard",
        }

    def should_steer(self, test_pos, confidence, prepared):
        if confidence < prepared["tau"]:
            return True, prepared["t"]
        return False, 0.0


class Config2_ExpandedData(SteeringConfig):
    """Expanded data: uses temperature-expanded correct set, layer 24, hard gate."""

    def __init__(self):
        super().__init__("config_2_expanded_data",
                         "Expanded contrastive set, layer 24, t=0.15, τ=0.40")

    def prepare_fold(self, fold_train_idx, fold_test_idx, all_activations, baseline_correct, y_all):
        # Use expanded correct activations for fold's train problems
        correct_indices = np.load(EXPANDED_ACT_DIR / "correct_problem_indices.npy")
        fold_train_set = set(fold_train_idx.tolist())

        # Filter correct activations to fold's train set
        mask = np.array([idx in fold_train_set for idx in correct_indices])
        if mask.sum() == 0:
            return None

        # Load expanded activations at layer 24
        expanded_path = EXPANDED_ACT_DIR / "correct_activations_layer24.npy"
        if expanded_path.exists():
            H_correct = np.load(expanded_path)[mask]
        else:
            # Fall back to Track A activations
            all_acts_24 = all_activations[24]
            correct_in_fold = correct_indices[mask]
            # Map global indices to positions in all_activations
            H_correct = all_acts_24[correct_in_fold]

        H_incorrect = all_activations[24][fold_train_idx][~baseline_correct[fold_train_idx].astype(bool)]

        if len(H_correct) == 0:
            return None

        # Bootstrap averaging with expanded data
        H_incorrect_mean = H_incorrect.mean(axis=0)
        v = bootstrap_mean_diff_vector(H_correct, H_incorrect_mean, n_bootstrap=BOOTSTRAP_SAMPLES)

        # Confidence
        scaler, clf, abc_names, abc_indices = fit_confidence_pipeline(fold_train_idx, y_all)
        conf_test = predict_confidence(scaler, clf, abc_indices, fold_test_idx)

        return {
            "vectors": {24: v},
            "t": TRACK_A_BEST_T,
            "tau": TRACK_A_BEST_TAU,
            "confidence_test": conf_test,
            "gating": "hard",
        }

    def should_steer(self, test_pos, confidence, prepared):
        if confidence < prepared["tau"]:
            return True, prepared["t"]
        return False, 0.0


class Config4_ExpandedDenoisedMultiLayer(SteeringConfig):
    """Expanded + denoised + best multi-layer pair."""

    def __init__(self, multi_layer_name: str = "non_adjacent_a", pca_k: int = 10, sparse_d: int = 100):
        self.multi_layer_name = multi_layer_name
        self.layers = MULTI_LAYER_CONFIGS[multi_layer_name]
        self.pca_k = pca_k
        self.sparse_d = sparse_d
        super().__init__(
            f"config_4_expanded_denoised_multilayer_{multi_layer_name}",
            f"Expanded + bootstrap + PCA k={pca_k} + sparse d={sparse_d} at layers {self.layers}",
        )

    def prepare_fold(self, fold_train_idx, fold_test_idx, all_activations, baseline_correct, y_all):
        correct_indices = np.load(EXPANDED_ACT_DIR / "correct_problem_indices.npy")
        fold_train_set = set(fold_train_idx.tolist())
        mask = np.array([idx in fold_train_set for idx in correct_indices])

        if mask.sum() == 0:
            return None

        vectors = {}
        for layer in self.layers:
            # Load correct activations at this layer
            expanded_path = EXPANDED_ACT_DIR / f"correct_activations_layer{layer}.npy"
            if expanded_path.exists():
                H_correct = np.load(expanded_path)[mask]
            else:
                H_correct = all_activations[layer][correct_indices[mask]]

            H_incorrect = all_activations[layer][fold_train_idx][~baseline_correct[fold_train_idx].astype(bool)]

            if len(H_correct) == 0:
                return None

            # Bootstrap → PCA → sparsify
            H_incorrect_mean = H_incorrect.mean(axis=0)
            v_boot = bootstrap_mean_diff_vector(H_correct, H_incorrect_mean)

            # PCA projection
            diff_matrix = H_correct - H_incorrect_mean[np.newaxis, :]
            k = min(self.pca_k, *diff_matrix.shape)
            v_pca = pca_project_vector(v_boot, diff_matrix, k)

            # Sparsify
            v_sparse = sparsify_vector(v_pca, self.sparse_d)
            vectors[layer] = v_sparse

        # Confidence
        scaler, clf, abc_names, abc_indices = fit_confidence_pipeline(fold_train_idx, y_all)
        conf_test = predict_confidence(scaler, clf, abc_indices, fold_test_idx)

        return {
            "vectors": vectors,
            "t": TRACK_A_BEST_T,
            "tau": TRACK_A_BEST_TAU,
            "confidence_test": conf_test,
            "gating": "hard",
        }

    def should_steer(self, test_pos, confidence, prepared):
        if confidence < prepared["tau"]:
            return True, prepared["t"]
        return False, 0.0


class Config6_FullStack(SteeringConfig):
    """Full stack: expanded + denoised + multi-layer + prototypes + soft gating."""

    def __init__(
        self,
        multi_layer_name: str = "non_adjacent_a",
        proto_K: int = 5,
        proto_temp: float = 0.5,
        sigmoid_sharpness: float = 10.0,
        pca_k: int = 10,
        sparse_d: int = 100,
    ):
        self.multi_layer_name = multi_layer_name
        self.layers = MULTI_LAYER_CONFIGS[multi_layer_name]
        self.proto_K = proto_K
        self.proto_temp = proto_temp
        self.sigmoid_sharpness = sigmoid_sharpness
        self.pca_k = pca_k
        self.sparse_d = sparse_d
        super().__init__(
            f"config_6_full_stack_{multi_layer_name}_K{proto_K}_sharp{sigmoid_sharpness}",
            f"Full stack: layers {self.layers}, K={proto_K}, soft gate sharpness={sigmoid_sharpness}",
        )

    def prepare_fold(self, fold_train_idx, fold_test_idx, all_activations, baseline_correct, y_all):
        correct_indices = np.load(EXPANDED_ACT_DIR / "correct_problem_indices.npy")
        fold_train_set = set(fold_train_idx.tolist())
        mask = np.array([idx in fold_train_set for idx in correct_indices])

        if mask.sum() == 0:
            return None

        prototypes_per_layer = {}
        for layer in self.layers:
            expanded_path = EXPANDED_ACT_DIR / f"correct_activations_layer{layer}.npy"
            if expanded_path.exists():
                H_correct = np.load(expanded_path)[mask]
            else:
                H_correct = all_activations[layer][correct_indices[mask]]

            H_incorrect = all_activations[layer][fold_train_idx][~baseline_correct[fold_train_idx].astype(bool)]

            if len(H_correct) < self.proto_K * 5:
                # Fall back to single vector
                H_incorrect_mean = H_incorrect.mean(axis=0)
                v = bootstrap_mean_diff_vector(H_correct, H_incorrect_mean)
                diff_matrix = H_correct - H_incorrect_mean[np.newaxis, :]
                k = min(self.pca_k, *diff_matrix.shape)
                v = pca_project_vector(v, diff_matrix, k)
                v = sparsify_vector(v, self.sparse_d)
                prototypes_per_layer[layer] = v[np.newaxis, :]  # (1, 1536)
            else:
                H_incorrect_mean = H_incorrect.mean(axis=0)
                diff_matrix = H_correct - H_incorrect_mean[np.newaxis, :]
                centroids, _, quality = cluster_differences(diff_matrix, self.proto_K, seed=SEED)
                prototypes_per_layer[layer] = centroids  # (K, 1536)

        # Confidence
        scaler, clf, abc_names, abc_indices = fit_confidence_pipeline(fold_train_idx, y_all)
        conf_test = predict_confidence(scaler, clf, abc_indices, fold_test_idx)

        return {
            "prototypes": prototypes_per_layer,
            "t_base": TRACK_A_BEST_T,
            "tau": TRACK_A_BEST_TAU,
            "confidence_test": conf_test,
            "gating": "soft",
            "sigmoid_sharpness": self.sigmoid_sharpness,
            "proto_temp": self.proto_temp,
        }

    def should_steer(self, test_pos, confidence, prepared):
        t = soft_gate_strength(
            confidence,
            prepared["t_base"],
            prepared["tau"],
            prepared["sigmoid_sharpness"],
        )
        if t < 0.01:  # Effectively zero
            return False, 0.0
        return True, t


# ============================================
# CV evaluation engine
# ============================================


def create_folds(y_all: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create 5 stratified folds.

    Returns list of (train_idx, test_idx) tuples.
    """
    skf = StratifiedKFold(n_splits=CV_N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    return list(skf.split(np.arange(N_PROBLEMS), y_all))


def load_all_activations_for_cv() -> dict[int, np.ndarray]:
    """Load activations at all layers needed for CV.

    Returns {layer: (500, 1536)} for layers {15, 20, 24, 25}.
    """
    activations = {}

    # Track A activations
    for layer in [20, 24]:
        path = TRACK_A_PHASE0 / "activations" / f"layer_{layer}.npy"
        activations[layer] = np.load(path)

    # Pathway 3 additional activations
    for layer in ADDITIONAL_LAYERS:
        path = ADDITIONAL_ACT_DIR / f"layer_{layer}.npy"
        if path.exists():
            activations[layer] = np.load(path)
        else:
            logger.warning("Missing additional layer %d activations", layer)

    return activations


def evaluate_fold(
    config: SteeringConfig,
    fold_idx: int,
    fold_train_idx: np.ndarray,
    fold_test_idx: np.ndarray,
    model,
    tokenizer,
    prompts: list[str],
    ground_truths: list[str],
    all_activations: dict[int, np.ndarray],
    baseline_correct: np.ndarray,
    baseline_answers: list[dict],
    y_all: np.ndarray,
) -> dict | None:
    """Evaluate a steering config on one CV fold.

    Returns fold results dict or None if preparation failed.
    """
    logger.info("  Preparing %s for fold %d...", config.name, fold_idx)

    # Prepare steering vectors and confidence for this fold
    prepared = config.prepare_fold(
        fold_train_idx, fold_test_idx, all_activations, baseline_correct, y_all
    )
    if prepared is None:
        logger.warning("  Config %s failed preparation for fold %d", config.name, fold_idx)
        return None

    n_test = len(fold_test_idx)
    steered_correct = np.zeros(n_test, dtype=int)
    steered_flags = np.zeros(n_test, dtype=int)
    steering_strengths = np.zeros(n_test, dtype=float)
    per_problem = []

    for i, idx in enumerate(fold_test_idx):
        confidence = float(prepared["confidence_test"][i])
        should_steer, t_val = config.should_steer(i, confidence, prepared)

        if should_steer and t_val > 0.01:
            # Determine steering vector(s)
            if "prototypes" in prepared:
                # Prototype-based: compute per-problem vector at each layer
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

            # Register hooks
            handles = register_multi_layer_hooks(model, layer_vecs, t_val)
            text = generate_one(model, tokenizer, prompts[idx])
            remove_all_hooks(handles)

            predicted = extract_answer(text)
            is_correct = check_correct(predicted, ground_truths[idx])
            steered_correct[i] = int(is_correct)
            steered_flags[i] = 1
            steering_strengths[i] = t_val
        else:
            # Reuse baseline
            steered_correct[i] = int(baseline_correct[idx])
            steered_flags[i] = 0

        per_problem.append({
            "test_pos": i,
            "global_idx": int(idx),
            "confidence": round(confidence, 4),
            "steered": int(steered_flags[i]),
            "steering_strength": round(float(steering_strengths[i]), 4),
            "correct": int(steered_correct[i]),
            "baseline_correct": int(baseline_correct[idx]),
        })

        if (i + 1) % 20 == 0:
            logger.info("    Fold %d: %d/%d done", fold_idx, i + 1, n_test)

    # Compute metrics
    baseline_fold = baseline_correct[fold_test_idx]
    metrics = compute_flip_metrics(baseline_fold, steered_correct)

    # Statistical tests
    b = metrics["wrong_to_right"]
    c = metrics["right_to_wrong"]
    metrics["mcnemar_mid_p"] = mcnemar_mid_p(c, b)
    metrics["bayesian_p_improvement"] = bayesian_p_improvement(b, c)
    metrics["bootstrap_ci_95"] = list(bootstrap_ci_net_gain(baseline_fold, steered_correct))

    # Steered subset metrics
    steered_mask = steered_flags.astype(bool)
    if steered_mask.any():
        metrics["n_steered"] = int(steered_mask.sum())
        metrics["steered_accuracy"] = float(steered_correct[steered_mask].mean())
        metrics["steered_baseline_accuracy"] = float(baseline_fold[steered_mask].mean())
    else:
        metrics["n_steered"] = 0
        metrics["steered_accuracy"] = 0.0
        metrics["steered_baseline_accuracy"] = 0.0

    return {
        "fold": fold_idx,
        "config": config.name,
        "metrics": metrics,
        "per_problem": per_problem,
    }


def aggregate_cv_results(fold_results: list[dict]) -> dict:
    """Aggregate results across 5 CV folds.

    Returns summary with mean, std, CI, and statistical tests.
    """
    net_gains = [r["metrics"]["net_gain"] for r in fold_results]
    total_w2r = sum(r["metrics"]["wrong_to_right"] for r in fold_results)
    total_r2w = sum(r["metrics"]["right_to_wrong"] for r in fold_results)

    # Pool all baseline/steered correctness arrays for bootstrap CI
    all_baseline = []
    all_steered = []
    for r in fold_results:
        for p in r["per_problem"]:
            all_baseline.append(p["baseline_correct"])
            all_steered.append(p["correct"])
    all_baseline = np.array(all_baseline)
    all_steered = np.array(all_steered)

    bca_ci = list(bootstrap_ci_net_gain(all_baseline, all_steered))

    return {
        "per_fold_net_gain": net_gains,
        "mean_net_gain": round(float(np.mean(net_gains)), 2),
        "std_net_gain": round(float(np.std(net_gains)), 2),
        "bca_ci_95": [round(x, 2) for x in bca_ci],
        "bayesian_p_improvement": round(bayesian_p_improvement(total_w2r, total_r2w), 4),
        "total_wrong_to_right": total_w2r,
        "total_right_to_wrong": total_r2w,
        "total_net_gain": total_w2r - total_r2w,
        "per_fold_accuracy": [
            round(r["metrics"]["steered_accuracy"], 4)
            for r in fold_results
            if r["metrics"].get("steered_accuracy") is not None
        ],
        "per_fold_mcnemar_p": [
            round(r["metrics"]["mcnemar_mid_p"], 4) for r in fold_results
        ],
    }


def sweep_sigmoid_sharpness(
    model, tokenizer, prompts, ground_truths,
    fold_train_idx, fold_test_idx,
    all_activations, baseline_correct, baseline_answers, y_all,
    multi_layer_name: str, proto_K: int, pca_k: int, sparse_d: int,
) -> float:
    """Sweep sigmoid sharpness on fold 1 to select best value.

    Returns best sharpness.
    """
    logger.info("  Sweeping sigmoid sharpness on fold 0...")
    best_gain = -999
    best_sharpness = 10.0

    for sharpness in SIGMOID_SHARPNESS_SWEEP:
        config = Config6_FullStack(
            multi_layer_name=multi_layer_name,
            proto_K=proto_K,
            proto_temp=0.5,
            sigmoid_sharpness=sharpness,
            pca_k=pca_k,
            sparse_d=sparse_d,
        )
        result = evaluate_fold(
            config, 0, fold_train_idx, fold_test_idx,
            model, tokenizer, prompts, ground_truths,
            all_activations, baseline_correct, baseline_answers, y_all,
        )
        if result is not None:
            gain = result["metrics"]["net_gain"]
            logger.info("    sharpness=%d: net_gain=%d", sharpness, gain)
            if gain > best_gain:
                best_gain = gain
                best_sharpness = sharpness

    logger.info("  Best sigmoid sharpness: %d (net_gain=%d)", best_sharpness, best_gain)
    return best_sharpness


def main():
    print("=" * 70)
    print("Phase 3: 5-Fold Cross-Validated Evaluation")
    print("=" * 70)

    # ============================================
    # Setup
    # ============================================
    print("\n[1] Loading data...")
    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)
    assert len(prompts) == N_PROBLEMS

    baseline_correct = np.load(TRACK_A_PHASE0 / "baseline_correct.npy")
    with open(TRACK_A_PHASE0 / "baseline_answers.json") as f:
        baseline_answers = json.load(f)

    meta = load_trajectory_meta()
    y_all = np.array(meta["correct"], dtype=int)

    print(f"  Baseline accuracy: {baseline_correct.sum()}/{N_PROBLEMS}")

    # Load activations
    print("\n[2] Loading activations...")
    all_activations = load_all_activations_for_cv()
    print(f"  Loaded layers: {sorted(all_activations.keys())}")

    # Create folds
    print(f"\n[3] Creating {CV_N_SPLITS}-fold stratified splits (seed={CV_RANDOM_STATE})...")
    folds = create_folds(y_all)

    fold_assignments = []
    for i, (train_idx, test_idx) in enumerate(folds):
        n_correct_train = y_all[train_idx].sum()
        n_correct_test = y_all[test_idx].sum()
        print(f"  Fold {i}: train={len(train_idx)} ({n_correct_train} correct), "
              f"test={len(test_idx)} ({n_correct_test} correct)")
        fold_assignments.append({
            "fold": i,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_correct_train": int(n_correct_train),
            "n_correct_test": int(n_correct_test),
        })

    with open(PHASE3_DIR / "fold_assignments.json", "w") as f:
        json.dump(fold_assignments, f, indent=2)

    # Load model
    print("\n[4] Loading model...")
    model, tokenizer = load_model()

    # ============================================
    # Define configurations
    # ============================================
    # Phase 2 summary tells us which vectors/prototypes are available
    phase2_summary_path = PHASE2_DIR / "phase2_summary.json"
    if phase2_summary_path.exists():
        with open(phase2_summary_path) as f:
            phase2_summary = json.load(f)
    else:
        phase2_summary = {}

    # Select best multi-layer config (default to non_adjacent_a = [15, 25])
    # and best prototype K (default to 5)
    multi_layer_name = "non_adjacent_a"
    proto_K = 5
    pca_k = 10
    sparse_d = 100

    # Priority configs: 1, 2, 4, 6
    configs = [
        Config1_TrackABaseline(),
        Config2_ExpandedData(),
        Config4_ExpandedDenoisedMultiLayer(multi_layer_name, pca_k, sparse_d),
    ]

    # For Config 6, first sweep sigmoid sharpness on fold 0
    print("\n[5] Sweeping sigmoid sharpness on fold 0...")
    best_sharpness = sweep_sigmoid_sharpness(
        model, tokenizer, prompts, ground_truths,
        folds[0][0], folds[0][1],
        all_activations, baseline_correct, baseline_answers, y_all,
        multi_layer_name, proto_K, pca_k, sparse_d,
    )

    configs.append(Config6_FullStack(
        multi_layer_name=multi_layer_name,
        proto_K=proto_K,
        proto_temp=0.5,
        sigmoid_sharpness=best_sharpness,
        pca_k=pca_k,
        sparse_d=sparse_d,
    ))

    # ============================================
    # Run all configs across all folds
    # ============================================
    all_cv_results = {}

    for config in configs:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {config.name}")
        print(f"  {config.description}")
        print(f"{'=' * 70}")

        fold_results = []
        t0_config = time.time()

        for fold_idx, (fold_train, fold_test) in enumerate(folds):
            # Check for checkpoint
            checkpoint_path = PHASE3_DIR / f"checkpoint_{config.name}_fold{fold_idx}.json"
            if checkpoint_path.exists():
                with open(checkpoint_path) as f:
                    result = json.load(f)
                fold_results.append(result)
                logger.info("  Fold %d: loaded from checkpoint (net_gain=%d)",
                            fold_idx, result["metrics"]["net_gain"])
                continue

            print(f"\n  --- Fold {fold_idx} ---")
            t0_fold = time.time()

            result = evaluate_fold(
                config, fold_idx, fold_train, fold_test,
                model, tokenizer, prompts, ground_truths,
                all_activations, baseline_correct, baseline_answers, y_all,
            )

            if result is None:
                logger.warning("  Fold %d: config failed, skipping", fold_idx)
                continue

            elapsed = time.time() - t0_fold
            print(f"  Fold {fold_idx}: net_gain={result['metrics']['net_gain']}, "
                  f"acc={result['metrics']['steered_accuracy']:.3f}, "
                  f"steered={result['metrics']['n_steered']}, "
                  f"time={elapsed:.0f}s")

            fold_results.append(result)

            # Save checkpoint (without per_problem for compactness)
            with open(checkpoint_path, "w") as f:
                json.dump(result, f)

        # Aggregate
        if len(fold_results) == CV_N_SPLITS:
            aggregated = aggregate_cv_results(fold_results)
            all_cv_results[config.name] = aggregated
            print(f"\n  {config.name} AGGREGATE:")
            print(f"    Mean net gain: {aggregated['mean_net_gain']} ± {aggregated['std_net_gain']}")
            print(f"    Per-fold: {aggregated['per_fold_net_gain']}")
            print(f"    BCa 95% CI: {aggregated['bca_ci_95']}")
            print(f"    Bayesian P(improvement): {aggregated['bayesian_p_improvement']}")
            print(f"    Total W→R: {aggregated['total_wrong_to_right']}, R→W: {aggregated['total_right_to_wrong']}")
        else:
            logger.warning("  Only %d/%d folds completed for %s",
                           len(fold_results), CV_N_SPLITS, config.name)
            all_cv_results[config.name] = {"incomplete": True, "n_folds": len(fold_results)}

        config_time = time.time() - t0_config
        print(f"  Config total time: {config_time:.0f}s ({config_time / 3600:.1f}h)")

    # ============================================
    # Select best configuration
    # ============================================
    print(f"\n{'=' * 70}")
    print("Configuration Selection")
    print(f"{'=' * 70}")

    best_config = None
    best_mean_gain = -999

    for name, results in all_cv_results.items():
        if results.get("incomplete"):
            continue
        mean_gain = results["mean_net_gain"]
        print(f"  {name}: mean_net_gain={mean_gain}")
        if mean_gain > best_mean_gain:
            best_mean_gain = mean_gain
            best_config = name
        elif mean_gain == best_mean_gain:
            # Tiebreaker: simpler config (fewer components = earlier in list)
            pass  # Keep the first one found (already simpler)

    if best_config is None:
        print("  *** ERROR: No complete configuration found ***")
        return

    print(f"\n  SELECTED: {best_config} (mean_net_gain={best_mean_gain})")

    # Lock configuration
    locked = {
        "selected_config": best_config,
        "mean_net_gain": best_mean_gain,
        "results": all_cv_results[best_config],
        "cv_n_splits": CV_N_SPLITS,
        "cv_random_state": CV_RANDOM_STATE,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Add config-specific parameters
    for config in configs:
        if config.name == best_config:
            if hasattr(config, "layers"):
                locked["layers"] = config.layers
            if hasattr(config, "pca_k"):
                locked["pca_k"] = config.pca_k
            if hasattr(config, "sparse_d"):
                locked["sparse_d"] = config.sparse_d
            if hasattr(config, "proto_K"):
                locked["proto_K"] = config.proto_K
            if hasattr(config, "sigmoid_sharpness"):
                locked["sigmoid_sharpness"] = config.sigmoid_sharpness
            if hasattr(config, "proto_temp"):
                locked["proto_temp"] = config.proto_temp
            if hasattr(config, "multi_layer_name"):
                locked["multi_layer_name"] = config.multi_layer_name
            locked["t_base"] = TRACK_A_BEST_T
            locked["tau"] = TRACK_A_BEST_TAU
            break

    with open(PHASE3_DIR / "locked_config.json", "w") as f:
        json.dump(locked, f, indent=2)

    # Save full CV results
    with open(PHASE3_DIR / "cv_results.json", "w") as f:
        json.dump(all_cv_results, f, indent=2)

    # Summary table
    print(f"\n{'=' * 70}")
    print("Phase 3 Summary")
    print(f"{'=' * 70}")
    print(f"\n  {'Config':<50} {'Mean Gain':>10} {'Std':>6} {'P(improv)':>10} {'CI 95%':>15}")
    print(f"  {'-'*91}")
    for name, results in all_cv_results.items():
        if results.get("incomplete"):
            print(f"  {name:<50} {'INCOMPLETE':>10}")
        else:
            ci = results["bca_ci_95"]
            print(f"  {name:<50} {results['mean_net_gain']:>10.1f} "
                  f"{results['std_net_gain']:>6.1f} "
                  f"{results['bayesian_p_improvement']:>10.3f} "
                  f"[{ci[0]:>5.1f}, {ci[1]:>5.1f}]")

    print(f"\n  LOCKED: {best_config}")
    print(f"  Artifacts saved to {PHASE3_DIR}/")
    print(f"\n  Next: Run phase4_holdout.py")


if __name__ == "__main__":
    main()
