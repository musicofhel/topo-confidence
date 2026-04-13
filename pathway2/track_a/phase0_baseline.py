#!/usr/bin/env python3
"""Phase 0: Baseline & Confidence Scores.

1. Verify Pathway 1 artifacts
2. Forward pass all 500 prompts: capture layer states + candidate activations
3. Generate all 500 MATH-500 answers (greedy, unsteered)
4. Assert baseline accuracy = 57/500 +/- 2
5. Replicate CORAL confidence scoring with train-only pipeline
6. Create 2x2 partition using Youden threshold
7. Save all artifacts

Runtime: ~1.5 hours GPU.
"""

from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (
    ARTIFACT_LOCKS_PATH,
    CANDIDATE_LAYERS,
    FEATURES_HOLDOUT_PATH,
    FEATURES_TRAIN_PATH,
    HOLDOUT_PREDICTIONS_PATH,
    LAYER_STATES_PATH,
    LR_PARAMS,
    MAX_LENGTH,
    MAX_NEW_TOKENS,
    N_PROBLEMS,
    PHASE0_DIR,
    SEED,
    TOPO_DIR,
    YOUDEN_THRESHOLD,
    check_correct,
    extract_answer,
    generate_one,
    get_abc_column_indices,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_cached_trajectories,
    load_math500,
    load_model,
    load_trajectory_meta,
    logger,
    verify_pathway1_artifacts,
)

PATHWAY1_PHASE0_DIR = TOPO_DIR / "pathway1" / "phase0"


def extract_all_layer_states_and_activations(
    model, tokenizer, prompts: list[str]
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Forward pass all prompts, capturing layer states and candidate activations.

    Returns:
        layer_states: (500, 29, 1536) — last-token hidden state at each layer
        candidate_activations: {layer: (500, 1536)} for each candidate layer
    """
    n = len(prompts)
    hidden_dim = model.config.hidden_size  # 1536
    n_layers = model.config.num_hidden_layers  # 28

    layer_states = np.zeros((n, n_layers + 1, hidden_dim), dtype=np.float32)
    candidate_activations = {
        layer: np.zeros((n, hidden_dim), dtype=np.float32)
        for layer in CANDIDATE_LAYERS
    }

    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            logger.info("Forward pass %d/%d...", i, n)

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors, each (1, seq_len, hidden_dim)
        # Index 0 = embeddings, 1..28 = layer 0..27 outputs
        # But Qwen2 has 28 layers (0-27), so hidden_states has 29 entries
        seq_len = inputs["attention_mask"].sum().item()
        last_pos = seq_len - 1  # last real token position

        for layer_idx in range(n_layers + 1):
            hs = outputs.hidden_states[layer_idx]  # (1, seq_len, hidden_dim)
            layer_states[i, layer_idx] = (
                hs[0, last_pos].cpu().float().numpy()
            )

        for layer in CANDIDATE_LAYERS:
            hs = outputs.hidden_states[layer + 1]  # +1 because index 0 = embeddings
            candidate_activations[layer][i] = (
                hs[0, last_pos].cpu().float().numpy()
            )

    return layer_states, candidate_activations


def generate_baseline_answers(
    model, tokenizer, prompts: list[str], ground_truths: list[str]
) -> tuple[list[dict], np.ndarray]:
    """Generate answers for all 500 problems. Return answer records and correctness array."""
    n = len(prompts)
    answers = []
    correct = np.zeros(n, dtype=int)

    for i, (prompt, gt) in enumerate(zip(prompts, ground_truths)):
        if i % 50 == 0:
            logger.info("Generating %d/%d...", i, n)

        text = generate_one(model, tokenizer, prompt)
        predicted = extract_answer(text)
        is_correct = check_correct(predicted, gt)
        correct[i] = int(is_correct)

        answers.append({
            "index": i,
            "generated_text": text[:1000],
            "predicted_answer": predicted,
            "ground_truth": gt,
            "correct": int(is_correct),
        })

    return answers, correct


def compute_confidence_scores_from_cached_features(
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
) -> tuple[np.ndarray, object, object, list[str]]:
    """Fallback: use cached features from Pathway 1 directly.

    Loads features_train400.npy and features_holdout100.npy,
    selects A+B+C columns, fits scaler+LR on train, predicts all 500.

    Returns:
        confidence_scores: (500,) P(correct)
        scaler: fitted StandardScaler
        classifier: fitted LogisticRegression
        abc_names: list of 44 feature names
    """
    logger.info("Using cached Pathway 1 features (fallback)...")

    # Load cached features
    features_train = np.load(FEATURES_TRAIN_PATH)  # (400, 78)
    features_holdout = np.load(FEATURES_HOLDOUT_PATH)  # (100, 78)
    logger.info("  Train features: %s, Holdout features: %s",
                features_train.shape, features_holdout.shape)

    # Load feature names from artifact_locks
    with open(ARTIFACT_LOCKS_PATH) as f:
        locks = json.load(f)
    feature_names = locks["feature_names"]

    # Select A+B+C columns
    abc_indices = get_abc_column_indices(feature_names)
    features_train_abc = features_train[:, abc_indices]
    features_holdout_abc = features_holdout[:, abc_indices]
    abc_names = [feature_names[i] for i in abc_indices]

    # Load labels
    meta = load_trajectory_meta()
    y = np.array(meta["correct"], dtype=int)
    y_train = y[train_idx]

    # Fit scaler on train
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_train_abc)

    # Fit LR on train
    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train, y_train)

    # Predict all 500
    confidence_scores = np.full(N_PROBLEMS, np.nan)

    # Train predictions
    confidence_scores[train_idx] = clf.predict_proba(X_train)[:, 1]

    # Holdout predictions
    X_holdout = scaler.transform(features_holdout_abc)
    confidence_scores[holdout_idx] = clf.predict_proba(X_holdout)[:, 1]

    # Verify holdout predictions match Pathway 1
    expected = np.load(HOLDOUT_PREDICTIONS_PATH)
    actual = confidence_scores[holdout_idx]
    max_diff = np.abs(actual - expected).max()
    logger.info("  Max holdout prediction difference: %.6f", max_diff)

    if max_diff > 0.01:
        logger.warning(
            "Holdout predictions differ by %.6f (> 0.01). "
            "This may indicate feature index mismatch.", max_diff
        )

    return confidence_scores, scaler, clf, abc_names


def compute_confidence_scores_coral(
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    layer_states: np.ndarray,
) -> tuple[np.ndarray, object, object, list[str]]:
    """Compute confidence scores using CORAL pipeline with train-only PCA.

    Follows three_number.py monkey-patching pattern:
    1. Call extract_features on train-400 only (PCA fit on train)
    2. Monkey-patch PCA to use train-fitted components
    3. Call extract_features on all 500 (PCA is pre-fitted)
    4. Select A+B+C features, fit scaler+LR on train, predict all

    Returns:
        confidence_scores: (500,) P(correct)
        scaler: fitted StandardScaler
        classifier: fitted LogisticRegression
        abc_names: list of 44 feature names
    """
    # Import winning_features from pathway1/phase0
    sys.path.insert(0, str(PATHWAY1_PHASE0_DIR))
    import winning_features as wf_module
    from winning_features import extract_features

    # Load trajectories
    trajectories = load_cached_trajectories()

    # Load labels
    meta = load_trajectory_meta()
    y = np.array(meta["correct"], dtype=int)
    y_train = y[train_idx]

    # layer_states shape: (500, 29, 1536) — index 0 is embeddings layer
    # winning_features expects layer_states[i] to be (29, 1536)
    # Verify shape
    assert layer_states.shape == (N_PROBLEMS, 29, 1536), (
        f"layer_states shape {layer_states.shape} != (500, 29, 1536)"
    )

    # Step 1: Extract train-400 features (PCA fit on train only)
    traj_train = [trajectories[i] for i in train_idx]
    ls_train = layer_states[train_idx]
    logger.info("Extracting train-400 features (train-only PCA)...")
    t0 = time.time()
    features_train, feature_names = extract_features(traj_train, ls_train)
    logger.info("  Done in %.1fs, shape: %s", time.time() - t0, features_train.shape)

    # Step 2: Get the PCA that was fit on train-400 trajectories
    from sklearn.decomposition import PCA as OrigPCA

    traj_train_concat = np.concatenate(traj_train, axis=0)
    n_components = min(45, traj_train_concat.shape[1], traj_train_concat.shape[0])
    train_pca = OrigPCA(n_components=n_components, svd_solver="full")
    train_pca.fit(traj_train_concat)

    # Step 3: Monkey-patch PCA to use train-fitted components
    class PreFittedPCA(OrigPCA):
        _prefitted = None

        def fit(self, X, y=None):
            self.components_ = PreFittedPCA._prefitted.components_.copy()
            self.mean_ = PreFittedPCA._prefitted.mean_.copy()
            self.explained_variance_ = PreFittedPCA._prefitted.explained_variance_.copy()
            self.explained_variance_ratio_ = PreFittedPCA._prefitted.explained_variance_ratio_.copy()
            self.singular_values_ = PreFittedPCA._prefitted.singular_values_.copy()
            self.n_components_ = PreFittedPCA._prefitted.n_components_
            self.n_samples_ = PreFittedPCA._prefitted.n_samples_
            self.n_features_in_ = PreFittedPCA._prefitted.n_features_in_
            self.noise_variance_ = PreFittedPCA._prefitted.noise_variance_
            return self

    PreFittedPCA._prefitted = train_pca
    original_pca = wf_module.PCA
    wf_module.PCA = PreFittedPCA

    try:
        logger.info("Extracting all-500 features (train-fitted PCA)...")
        t0 = time.time()
        features_all, _ = extract_features(trajectories, layer_states)
        logger.info("  Done in %.1fs, shape: %s", time.time() - t0, features_all.shape)
    finally:
        wf_module.PCA = original_pca

    # Step 4: Select A+B+C features
    abc_indices = get_abc_column_indices(feature_names)
    abc_names = [feature_names[i] for i in abc_indices]

    features_train_abc = features_train[:, abc_indices]
    features_all_abc = features_all[:, abc_indices]

    # Step 5: Fit scaler on train, transform all
    scaler = StandardScaler()
    X_train = scaler.fit_transform(features_train_abc)
    X_all = scaler.transform(features_all_abc)

    # Step 6: Fit LR on train
    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train, y_train)

    # Step 7: Predict all 500
    confidence_scores = clf.predict_proba(X_all)[:, 1]

    # Verify holdout predictions match Pathway 1
    expected = np.load(HOLDOUT_PREDICTIONS_PATH)
    actual = confidence_scores[holdout_idx]
    max_diff = np.abs(actual - expected).max()
    mean_diff = np.abs(actual - expected).mean()
    logger.info("  Holdout prediction diff: max=%.6f, mean=%.6f", max_diff, mean_diff)

    return confidence_scores, scaler, clf, abc_names


def create_partition(
    confidence_scores: np.ndarray,
    correct: np.ndarray,
    threshold: float = YOUDEN_THRESHOLD,
) -> dict:
    """Create 2x2 partition: predicted x actual correctness."""
    predicted_correct = confidence_scores >= threshold
    actually_correct = correct.astype(bool)

    partition = {
        "threshold": threshold,
        "predicted_correct_actually_correct": {
            "count": int((predicted_correct & actually_correct).sum()),
            "indices": np.where(predicted_correct & actually_correct)[0].tolist(),
        },
        "predicted_correct_actually_wrong": {
            "count": int((predicted_correct & ~actually_correct).sum()),
            "indices": np.where(predicted_correct & ~actually_correct)[0].tolist(),
        },
        "predicted_wrong_actually_correct": {
            "count": int((~predicted_correct & actually_correct).sum()),
            "indices": np.where(~predicted_correct & actually_correct)[0].tolist(),
        },
        "predicted_wrong_actually_wrong": {
            "count": int((~predicted_correct & ~actually_correct).sum()),
            "indices": np.where(~predicted_correct & ~actually_correct)[0].tolist(),
        },
    }
    return partition


def main():
    print("=" * 60)
    print("Phase 0: Baseline & Confidence Scores")
    print("=" * 60)

    # Step 1: Verify artifacts
    print("\n[1] Verifying Pathway 1 artifacts...")
    verify_pathway1_artifacts()

    # Step 2: Load problems and model
    print("\n[2] Loading MATH-500 problems...")
    problems = load_math500()
    prompts, ground_truths = get_prompts_and_ground_truths(problems)
    assert len(prompts) == N_PROBLEMS

    print("\n[3] Loading model...")
    model, tokenizer = load_model()

    # Step 3: Forward pass — extract layer states and candidate activations
    print("\n[4] Forward pass: extracting layer states and candidate activations...")
    t0 = time.time()
    layer_states, candidate_activations = extract_all_layer_states_and_activations(
        model, tokenizer, prompts
    )
    print(f"  Forward pass: {time.time() - t0:.1f}s")
    print(f"  Layer states shape: {layer_states.shape}")

    # Save candidate activations
    for layer, acts in candidate_activations.items():
        np.save(PHASE0_DIR / "activations" / f"layer_{layer}.npy", acts)
        print(f"  Saved layer_{layer}.npy: {acts.shape}")

    # Step 4: Generate all 500 baseline answers
    print("\n[5] Generating baseline answers (greedy, unsteered)...")
    t0 = time.time()
    answers, correct = generate_baseline_answers(
        model, tokenizer, prompts, ground_truths
    )
    gen_time = time.time() - t0
    n_correct = correct.sum()
    print(f"  Generation: {gen_time:.1f}s ({gen_time / N_PROBLEMS:.2f}s/problem)")
    print(f"  Baseline accuracy: {n_correct}/{N_PROBLEMS} ({n_correct / N_PROBLEMS:.1%})")

    # Assert baseline accuracy
    if abs(n_correct - 57) > 2:
        print(f"\n  *** HALT: Baseline accuracy {n_correct}/500 is outside "
              f"tolerance of 57 +/- 2 ***")
        # Save what we have for debugging
        np.save(PHASE0_DIR / "baseline_correct.npy", correct)
        with open(PHASE0_DIR / "baseline_answers.json", "w") as f:
            json.dump(answers, f, indent=2)
        sys.exit(1)
    else:
        print(f"  MATCH: {n_correct}/500 is within tolerance of 57 +/- 2")

    # Free GPU memory before CORAL pipeline
    del model
    torch.cuda.empty_cache()

    # Step 5: Train/holdout split
    train_idx, holdout_idx = get_train_holdout_indices()
    meta = load_trajectory_meta()
    y = np.array(meta["correct"], dtype=int)
    print(f"\n[6] Train: {len(train_idx)}, Holdout: {len(holdout_idx)}")
    print(f"  Train correct: {y[train_idx].sum()}/{len(train_idx)}")
    print(f"  Holdout correct: {y[holdout_idx].sum()}/{len(holdout_idx)}")

    # Step 6: CORAL confidence scoring
    # layer_states from our forward pass: (500, 29, 1536)
    # winning_features.py expects layer_states[i] = (29, 1536)
    # Our layer_states[:, 0] = embeddings, [:, 1:] = layers 0-27
    # But winning_features indexes ls_all = layer_states[i] and uses ls_all[0]..ls_all[28]
    # So it expects 29 entries where index k corresponds to layer k output
    # Our format: index 0 = embeddings, index 1 = layer 0, ..., index 28 = layer 27
    # MISMATCH: winning_features uses ls_all[9] for layer 9, but our index 9 = layer 8
    #
    # The original math500_hidden_states_aligned.npz had shape (500, 29, 1536) where
    # index 0 = layer 0 output (NOT embeddings). So we need to shift: use indices 1..28+1
    # Wait, the model has 28 layers (0-27). hidden_states from HF has 29 entries:
    # [embeddings, layer0_out, layer1_out, ..., layer27_out]
    # The original data had 29 entries where index 0..28 maps to layers 0..28
    # But there are only 28 layers (0-27)!
    # Looking at winning_features: ls_all[28] is used. That's the last layer output.
    # In our extraction: hidden_states[28] = layer 27 output (since index 0 = embeddings)
    # So our layer_states[i, 28] = layer 27 output, matching ls_all[28].
    # BUT: winning_features uses ls_all[0] for "layer 0" but our index 0 = embeddings.
    #
    # The original data from att-docs had layer_hidden_states with shape (500, 29, 1536).
    # In that data, index 0 was likely embeddings too (it came from output_hidden_states).
    # So our format DOES match. Let's verify by checking if the CORAL results match.

    print("\n[7] Computing CORAL confidence scores...")

    # Try CORAL pipeline first
    try:
        confidence_scores, scaler, clf, abc_names = compute_confidence_scores_coral(
            train_idx, holdout_idx, layer_states
        )

        # Check if holdout predictions are close enough
        expected = np.load(HOLDOUT_PREDICTIONS_PATH)
        actual = confidence_scores[holdout_idx]
        max_diff = np.abs(actual - expected).max()

        if max_diff > 0.05:
            logger.warning(
                "CORAL holdout predictions differ by %.4f. Trying fallback...", max_diff
            )
            raise ValueError("CORAL predictions too far from expected")

        logger.info("CORAL pipeline matched (max holdout diff: %.6f)", max_diff)
        used_fallback = False

    except Exception as e:
        logger.warning("CORAL pipeline failed: %s. Using cached features fallback.", e)
        confidence_scores, scaler, clf, abc_names = (
            compute_confidence_scores_from_cached_features(train_idx, holdout_idx)
        )
        used_fallback = True

    # Step 7: Create 2x2 partition
    print(f"\n[8] Creating 2x2 partition (threshold={YOUDEN_THRESHOLD:.4f})...")
    partition = create_partition(confidence_scores, correct, YOUDEN_THRESHOLD)
    print(f"  Predicted-correct & actually-correct: {partition['predicted_correct_actually_correct']['count']}")
    print(f"  Predicted-correct & actually-wrong:   {partition['predicted_correct_actually_wrong']['count']}")
    print(f"  Predicted-wrong & actually-correct:    {partition['predicted_wrong_actually_correct']['count']}")
    print(f"  Predicted-wrong & actually-wrong:      {partition['predicted_wrong_actually_wrong']['count']}")

    # Step 8: Save artifacts
    print("\n[9] Saving artifacts...")

    # Baseline answers
    with open(PHASE0_DIR / "baseline_answers.json", "w") as f:
        json.dump(answers, f, indent=2)

    # Correctness array
    np.save(PHASE0_DIR / "baseline_correct.npy", correct)

    # Accuracy summary
    with open(PHASE0_DIR / "baseline_accuracy.txt", "w") as f:
        f.write(f"{n_correct}/{N_PROBLEMS} ({n_correct / N_PROBLEMS:.1%})\n")

    # Layer states (for potential CORAL re-runs)
    np.save(PHASE0_DIR / "layer_states.npy", layer_states)

    # Confidence scores
    np.save(PHASE0_DIR / "confidence_scores.npy", confidence_scores)

    # Confidence model (scaler + classifier)
    with open(PHASE0_DIR / "confidence_model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "classifier": clf, "abc_names": abc_names}, f)

    # Partition
    with open(PHASE0_DIR / "partition.json", "w") as f:
        json.dump(partition, f, indent=2)

    # Summary
    print("\n" + "=" * 60)
    print("Phase 0 Summary")
    print("=" * 60)
    print(f"  Baseline accuracy: {n_correct}/{N_PROBLEMS}")
    print(f"  Confidence method: {'cached features (fallback)' if used_fallback else 'CORAL pipeline'}")
    print(f"  Youden threshold: {YOUDEN_THRESHOLD}")
    print(f"  Artifacts saved to {PHASE0_DIR}/")
    print(f"\n  Next: Run phase0p5_probe_sweep.py")


if __name__ == "__main__":
    main()
