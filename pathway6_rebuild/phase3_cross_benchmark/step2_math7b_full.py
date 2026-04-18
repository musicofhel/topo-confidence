#!/usr/bin/env python3
"""Phase 3 Step 2: MATH-500 × 7B Full Pipeline with Chat Template.

GPU REQUIRED. Run on RunPod H100.

Corrected pipeline for MATH-500 × Qwen2.5-7B-Instruct:
- Chat template formatting (fixes 20.6% → ~75-77% accuracy)
- Corrected answer extraction (extract_answer_v2)
- PCA on train-400 only (no leakage)
- Same 400/100 train/holdout split as 1.5B (seed 9999 mask)
- Corrected selection strategies (select_*_v2)

Combines pathway5/track_b phases B1-B4 into single checkpoint-capable script.
Estimated runtime: ~2-3 hours on H100.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from common import (
    COMPLETIONS_PER_PROBLEM,
    LR_PARAMS,
    MODEL_NAME_7B,
    TEMP_SAMPLING,
    bootstrap_auroc,
    check_correct_v2,
    extract_answer_v2,
    extract_completion_trajectory,
    extract_features_with_prefitted_pca,
    extract_trajectory,
    fit_greedy_pca_train_only,
    format_prompt_chat,
    generate_greedy_with_logprobs,
    generate_temperature,
    get_abc_column_indices,
    get_prompts_and_ground_truths,
    get_train_holdout_indices,
    load_math500,
    load_model_parameterized,
    logger,
    normalize_answer_v2,
    score_completions,
    select_confidence_weighted_vote_v2,
    select_majority_vote_v2,
    select_max_confidence_v2,
    select_random_v2,
)

OUTPUT_DIR = Path(__file__).parent / "math7b"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CKPT_DIR = OUTPUT_DIR / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SECTION 1: GREEDY GENERATION
# ============================================================================


def run_greedy(model, tokenizer, prompts, ground_truths):
    """Generate greedy answers + logprob features for all 500 problems."""
    ckpt_path = CKPT_DIR / "greedy.json"
    answers = {}

    if ckpt_path.exists():
        with open(ckpt_path) as f:
            answers = json.load(f)
        logger.info("Greedy: resumed (%d/500 done)", len(answers))

    n = len(prompts)
    t0 = time.time()

    for i in range(n):
        if str(i) in answers:
            continue

        text, entropy_feats = generate_greedy_with_logprobs(model, tokenizer, prompts[i])
        is_correct = check_correct_v2(text, ground_truths[i])

        answers[str(i)] = {
            "text": text[:1000],
            "correct": is_correct,
            "entropy": entropy_feats,
        }

        if (i + 1) % 50 == 0:
            with open(ckpt_path, "w") as f:
                json.dump(answers, f)
            logger.info("Greedy: %d/%d (%.1f min)", i + 1, n, (time.time() - t0) / 60)

    with open(OUTPUT_DIR / "greedy_answers.json", "w") as f:
        json.dump(answers, f)

    correct = np.array([answers[str(i)]["correct"] for i in range(n)])
    np.save(OUTPUT_DIR / "baseline_correct.npy", correct)

    logger.info("Greedy accuracy: %d/%d (%.1f%%)", correct.sum(), n, 100 * correct.mean())
    return answers, correct


# ============================================================================
# SECTION 2: TRAJECTORY EXTRACTION
# ============================================================================


def run_trajectories(model, tokenizer, prompts):
    """Extract greedy trajectories + layer states for all 500 problems."""
    ckpt_meta = CKPT_DIR / "traj_meta.json"
    ckpt_traj = CKPT_DIR / "traj_partial.npz"
    ckpt_ls = CKPT_DIR / "ls_partial.npy"

    n = len(prompts)
    trajectories = [None] * n
    layer_states_list = []
    start_i = 0

    if ckpt_meta.exists():
        with open(ckpt_meta) as f:
            start_i = json.load(f)["n_done"]
        if ckpt_traj.exists():
            data = np.load(ckpt_traj, allow_pickle=True)
            for k in data.files:
                idx = int(k.split("_")[1])
                if idx < n:
                    trajectories[idx] = data[k]
        if ckpt_ls.exists():
            partial = np.load(ckpt_ls)
            layer_states_list = list(partial)
        logger.info("Trajectories: resumed (%d/%d done)", start_i, n)

    t0 = time.time()
    for i in range(start_i, n):
        traj, ls = extract_trajectory(model, tokenizer, prompts[i])
        trajectories[i] = traj
        layer_states_list.append(ls)

        if (i + 1) % 100 == 0:
            traj_dict = {f"t_{j}": trajectories[j] for j in range(i + 1) if trajectories[j] is not None}
            np.savez_compressed(ckpt_traj, **traj_dict)
            np.save(ckpt_ls, np.stack(layer_states_list))
            with open(ckpt_meta, "w") as f:
                json.dump({"n_done": i + 1}, f)
            logger.info("Trajectories: %d/%d (%.1f min)", i + 1, n, (time.time() - t0) / 60)

    traj_dict = {f"t_{j}": trajectories[j] for j in range(n)}
    np.savez_compressed(OUTPUT_DIR / "trajectories.npz", **traj_dict)
    layer_states = np.stack(layer_states_list)
    np.save(OUTPUT_DIR / "layer_states.npy", layer_states)

    logger.info("Trajectories done: %d problems, hidden_dim=%d", n, trajectories[0].shape[1])
    return trajectories, layer_states


# ============================================================================
# SECTION 3: FEATURES + MODEL
# ============================================================================


def run_features_and_model(trajectories, layer_states, correct, entropy_answers):
    """Fit PCA on train-400, extract features, train model, compare baselines."""
    n = len(correct)
    train_idx, holdout_idx = get_train_holdout_indices()
    np.save(OUTPUT_DIR / "train_idx.npy", train_idx)
    np.save(OUTPUT_DIR / "holdout_idx.npy", holdout_idx)
    logger.info("Split: %d train, %d holdout (same as 1.5B)", len(train_idx), len(holdout_idx))

    # PCA on train-400 trajectories only
    pca = fit_greedy_pca_train_only(trajectories, train_idx)

    # Extract topo features
    features, feature_names = extract_features_with_prefitted_pca(
        trajectories, layer_states, pca
    )
    np.save(OUTPUT_DIR / "features.npy", features)
    with open(OUTPUT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)
    logger.info("Features: %s", features.shape)

    # Fit LR on train
    abc_idx = get_abc_column_indices(feature_names)
    X_train = features[train_idx][:, abc_idx]
    y_train = correct[train_idx].astype(int)
    X_hold = features[holdout_idx][:, abc_idx]
    y_hold = correct[holdout_idx].astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_hold_s = scaler.transform(X_hold)

    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_train_s, y_train)

    # Topo scores for all
    X_all_s = scaler.transform(features[:, abc_idx])
    topo_scores = clf.predict_proba(X_all_s)[:, 1]
    np.save(OUTPUT_DIR / "topo_scores.npy", topo_scores)

    # Holdout AUROC
    hold_scores = topo_scores[holdout_idx]
    auroc, ci_lo, ci_hi = bootstrap_auroc(y_hold, hold_scores)
    logger.info("Topo AUROC (holdout): %.4f [%.4f, %.4f]", auroc, ci_lo, ci_hi)

    # Baseline AUROCs
    baselines = {}
    for bname, sign in [("neg_mean_entropy", -1), ("mean_max_token_prob", 1), ("first_token_prob", 1)]:
        key_map = {"neg_mean_entropy": "entropy", "mean_max_token_prob": "max_token_prob", "first_token_prob": "first_token_prob"}
        try:
            bl_scores = np.array([
                sign * entropy_answers[str(i)]["entropy"][key_map[bname]]
                for i in range(n)
            ])
            bl_auroc = roc_auc_score(y_hold, bl_scores[holdout_idx])
            baselines[bname] = float(bl_auroc)
        except Exception as e:
            logger.warning("Baseline %s failed: %s", bname, e)
            baselines[bname] = None

    best_bl_name = max(baselines, key=lambda k: baselines[k] or 0)
    best_bl_auroc = baselines[best_bl_name] or 0

    results = {
        "topo_auroc": float(auroc),
        "topo_auroc_ci": [float(ci_lo), float(ci_hi)],
        "baselines": baselines,
        "best_baseline": best_bl_name,
        "best_baseline_auroc": best_bl_auroc,
        "topo_vs_best_gap": float(auroc - best_bl_auroc),
        "n_train": int(len(train_idx)),
        "n_holdout": int(len(holdout_idx)),
        "train_correct": int(correct[train_idx].sum()),
        "holdout_correct": int(correct[holdout_idx].sum()),
    }
    with open(OUTPUT_DIR / "model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    import pickle
    with open(OUTPUT_DIR / "model.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf, "abc_idx": abc_idx, "pca": pca}, f)

    logger.info("Best baseline: %s (%.4f), gap: %+.4f", best_bl_name, best_bl_auroc, auroc - best_bl_auroc)
    return train_idx, holdout_idx, pca, topo_scores, feature_names, results


# ============================================================================
# SECTION 4: TEMPERATURE SAMPLING
# ============================================================================


def run_temperature_sampling(model, tokenizer, prompts, ground_truths, correct, holdout_idx):
    """Generate 32 temperature completions for holdout wrong-greedy problems."""
    ckpt_path = CKPT_DIR / "temp_gens.json"
    generations = {}

    if ckpt_path.exists():
        with open(ckpt_path) as f:
            generations = json.load(f)
        logger.info("Temp sampling: resumed (%d problems done)", len(generations))

    wrong_holdout = [i for i in holdout_idx if not correct[i]]
    remaining = [i for i in wrong_holdout if str(i) not in generations]
    logger.info("Temp sampling: %d wrong-holdout, %d remaining", len(wrong_holdout), len(remaining))

    t0 = time.time()
    for count, idx in enumerate(remaining):
        completions = []
        correct_flags = []

        for c in range(COMPLETIONS_PER_PROBLEM):
            text = generate_temperature(
                model, tokenizer, prompts[idx],
                temperature=TEMP_SAMPLING["temperature"],
                top_p=TEMP_SAMPLING["top_p"],
            )
            is_correct = check_correct_v2(text, ground_truths[idx])
            completions.append(text[:500])
            correct_flags.append(is_correct)

        generations[str(idx)] = {
            "completions": completions,
            "correct": correct_flags,
            "n_correct": sum(correct_flags),
        }

        if (count + 1) % 20 == 0:
            with open(ckpt_path, "w") as f:
                json.dump(generations, f)
            logger.info("Temp sampling: %d/%d (%.1f min)", count + 1, len(remaining), (time.time() - t0) / 60)

    with open(OUTPUT_DIR / "temperature_generations.json", "w") as f:
        json.dump(generations, f)

    logger.info("Temp sampling done: %d problems", len(generations))
    return generations


# ============================================================================
# SECTION 5: PER-COMPLETION TRAJECTORIES + FEATURES
# ============================================================================


def run_completion_features(model, tokenizer, prompts, generations, pca, feature_names):
    """Extract per-completion trajectories and topo features."""
    ckpt_meta = CKPT_DIR / "comp_traj_meta.json"
    ckpt_traj = CKPT_DIR / "comp_traj_partial.npz"
    ckpt_ls = CKPT_DIR / "comp_ls_partial.npz"

    all_trajs = []
    all_ls = []
    comp_map = []
    all_correct = []
    processed = set()

    if ckpt_meta.exists():
        with open(ckpt_meta) as f:
            meta = json.load(f)
        processed = set(meta["processed"])
        comp_map = [tuple(x) for x in meta["comp_map"]]
        all_correct = meta["all_correct"]
        if ckpt_traj.exists():
            data = np.load(ckpt_traj, allow_pickle=True)
            all_trajs = [data[k] for k in sorted(data.files, key=lambda x: int(x.split("_")[1]))]
        if ckpt_ls.exists():
            data = np.load(ckpt_ls, allow_pickle=True)
            all_ls = [data[k] for k in sorted(data.files, key=lambda x: int(x.split("_")[1]))]
        logger.info("Completion trajectories: resumed (%d problems done)", len(processed))

    problem_indices = sorted(int(k) for k in generations.keys())
    remaining = [i for i in problem_indices if str(i) not in processed]
    logger.info("Completion trajectories: %d problems, %d remaining", len(problem_indices), len(remaining))

    t0 = time.time()
    for count, idx in enumerate(remaining):
        data = generations[str(idx)]
        completions = data["completions"]
        correct_flags = data["correct"]

        for ci, comp in enumerate(completions):
            try:
                traj, ls = extract_completion_trajectory(model, tokenizer, prompts[idx], comp)
            except Exception as e:
                logger.error("Problem %d comp %d: %s", idx, ci, e)
                hidden_dim = all_trajs[0].shape[1] if all_trajs else 3584
                traj = np.zeros((1, hidden_dim), dtype=np.float32)
                ls = np.zeros((29, hidden_dim), dtype=np.float32)

            all_trajs.append(traj)
            all_ls.append(ls)
            comp_map.append((idx, ci))
            all_correct.append(bool(correct_flags[ci]))

        processed.add(str(idx))

        if (count + 1) % 20 == 0:
            with open(ckpt_meta, "w") as f:
                json.dump({"processed": list(processed), "comp_map": comp_map, "all_correct": all_correct}, f)
            traj_dict = {f"t_{i}": t for i, t in enumerate(all_trajs)}
            np.savez_compressed(ckpt_traj, **traj_dict)
            ls_dict = {f"l_{i}": l for i, l in enumerate(all_ls)}
            np.savez_compressed(ckpt_ls, **ls_dict)
            logger.info("Comp trajectories: %d/%d (%.1f min)", count + 1, len(remaining), (time.time() - t0) / 60)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Extract topo features
    logger.info("Extracting per-completion topo features...")
    ls_array = np.stack(all_ls, axis=0)
    comp_features, _ = extract_features_with_prefitted_pca(all_trajs, ls_array, pca)
    comp_correct = np.array(all_correct)
    comp_map_arr = np.array(comp_map)

    np.savez_compressed(
        OUTPUT_DIR / "completion_features.npz",
        features=comp_features,
        correct=comp_correct,
        comp_map=comp_map_arr,
    )
    logger.info("Completion features: %s", comp_features.shape)

    return comp_features, comp_correct, comp_map_arr


# ============================================================================
# SECTION 6: SELECTION EVALUATION
# ============================================================================


def run_selection(
    correct, holdout_idx, generations, ground_truths,
    topo_scores, comp_features, comp_correct, comp_map, feature_names,
):
    """Evaluate all selection strategies on holdout."""
    n_greedy_correct = int(correct[holdout_idx].sum())
    n_holdout = len(holdout_idx)

    # Fit per-completion scorer
    abc_idx = get_abc_column_indices(feature_names)
    X_comp = comp_features[:, abc_idx]
    scaler_c = StandardScaler()
    X_comp_s = scaler_c.fit_transform(X_comp)
    clf_c = LogisticRegression(**LR_PARAMS)
    clf_c.fit(X_comp_s, comp_correct.astype(int))
    comp_scores = clf_c.predict_proba(X_comp_s)[:, 1]

    # Per-problem scores
    problem_comp_scores = {}
    for i, (pidx, cidx) in enumerate(comp_map):
        pidx = int(pidx)
        if pidx not in problem_comp_scores:
            problem_comp_scores[pidx] = []
        problem_comp_scores[pidx].append(float(comp_scores[i]))

    # Per-completion AUROC
    if len(np.unique(comp_correct)) >= 2:
        comp_auroc = roc_auc_score(comp_correct, comp_scores)
    else:
        comp_auroc = None

    within_aurocs = []
    for pidx in sorted(set(int(x[0]) for x in comp_map)):
        mask = comp_map[:, 0].astype(int) == pidx
        y = comp_correct[mask]
        s = comp_scores[mask]
        if len(np.unique(y)) >= 2:
            within_aurocs.append(float(roc_auc_score(y, s)))

    results = {
        "baseline": {"n_correct": n_greedy_correct, "n_holdout": n_holdout},
        "per_completion_auroc": comp_auroc,
        "within_problem_auroc_mean": float(np.mean(within_aurocs)) if within_aurocs else None,
        "within_problem_auroc_n": len(within_aurocs),
    }

    # Evaluate strategies
    for strategy in ["ungated_mv", "ungated_weighted", "random"]:
        rng = np.random.default_rng(42)
        random_trials = 100 if strategy == "random" else 1
        trial_results = []

        for trial in range(random_trials):
            tc = 0
            tw = 0
            tr = 0
            for gi in holdout_idx:
                greedy_ok = bool(correct[gi])
                if str(gi) not in generations:
                    if greedy_ok:
                        tc += 1
                    continue

                comps = generations[str(gi)]["completions"]
                gt = ground_truths[gi]

                if strategy == "ungated_mv":
                    _, sel_ok = select_majority_vote_v2(comps, gt)
                elif strategy == "ungated_weighted":
                    scores = np.array(problem_comp_scores.get(gi, np.ones(len(comps))))
                    _, sel_ok = select_confidence_weighted_vote_v2(scores, comps, gt)
                elif strategy == "random":
                    _, sel_ok = select_random_v2(np.ones(len(comps)), comps, gt, rng)

                if sel_ok:
                    tc += 1
                    if not greedy_ok:
                        tw += 1
                else:
                    if greedy_ok:
                        tr += 1
            trial_results.append((tc, tw, tr))

        if strategy == "random":
            avg_c = np.mean([t[0] for t in trial_results])
            results[strategy] = {"mean_correct": float(avg_c), "net_gain": float(avg_c - n_greedy_correct)}
        else:
            tc, tw, tr = trial_results[0]
            results[strategy] = {"n_correct": tc, "net_gain": tc - n_greedy_correct, "W_to_R": tw, "R_to_W": tr}

    # Gated strategies
    tau_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    gating_results = []
    for tau in tau_values:
        gc_count = 0
        w_to_r = 0
        r_to_w = 0
        gated = 0

        for gi in holdout_idx:
            greedy_ok = bool(correct[gi])
            score = float(topo_scores[gi])

            if score >= tau:
                gated += 1
                if greedy_ok:
                    gc_count += 1
                continue

            if str(gi) not in generations:
                if greedy_ok:
                    gc_count += 1
                continue

            comps = generations[str(gi)]["completions"]
            gt = ground_truths[gi]
            _, sel_ok = select_majority_vote_v2(comps, gt)

            if sel_ok:
                gc_count += 1
                if not greedy_ok:
                    w_to_r += 1
            else:
                if greedy_ok:
                    r_to_w += 1

        gating_results.append({
            "tau": tau, "n_correct": gc_count,
            "net_gain": gc_count - n_greedy_correct,
            "W_to_R": w_to_r, "R_to_W": r_to_w, "gated": gated,
        })
    results["gating"] = gating_results

    # Oracle ceiling
    oracle_correct = 0
    for gi in holdout_idx:
        if correct[gi]:
            oracle_correct += 1
        elif str(gi) in generations and any(generations[str(gi)]["correct"]):
            oracle_correct += 1
    results["oracle_ceiling"] = oracle_correct - n_greedy_correct

    with open(OUTPUT_DIR / "selection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Selection: MV %+d, Weighted %+d, Oracle +%d",
                results["ungated_mv"]["net_gain"],
                results["ungated_weighted"]["net_gain"],
                results["oracle_ceiling"])
    return results


# ============================================================================
# MAIN
# ============================================================================


def main():
    t_start = time.time()
    logger.info("=" * 70)
    logger.info("PHASE 3 STEP 2: MATH-500 × 7B FULL PIPELINE")
    logger.info("=" * 70)

    # Load data
    problems = load_math500()
    n = len(problems)
    _, ground_truths = get_prompts_and_ground_truths(problems)
    logger.info("Loaded %d MATH-500 problems", n)

    # Load 7B model
    model, tokenizer = load_model_parameterized(MODEL_NAME_7B)

    # Format prompts with CHAT TEMPLATE
    problem_texts = [p.get("problem", p.get("question", "")) for p in problems]
    prompts = [format_prompt_chat(t, tokenizer, template="math") for t in problem_texts]
    logger.info("Formatted %d prompts with 7B chat template", n)

    # Section 1: Greedy
    logger.info("\n--- Section 1: Greedy Generation ---")
    answers, correct = run_greedy(model, tokenizer, prompts, ground_truths)

    # Section 2: Trajectories
    logger.info("\n--- Section 2: Trajectory Extraction ---")
    trajectories, layer_states = run_trajectories(model, tokenizer, prompts)

    # Section 3: Features + model
    logger.info("\n--- Section 3: Features + Model ---")
    train_idx, holdout_idx, pca, topo_scores, feature_names, model_results = \
        run_features_and_model(trajectories, layer_states, correct, answers)

    # Section 4: Temperature sampling
    logger.info("\n--- Section 4: Temperature Sampling ---")
    generations = run_temperature_sampling(model, tokenizer, prompts, ground_truths, correct, holdout_idx)

    # Section 5: Per-completion features
    logger.info("\n--- Section 5: Per-Completion Features ---")
    comp_features, comp_correct, comp_map = run_completion_features(
        model, tokenizer, prompts, generations, pca, feature_names
    )

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Section 6: Selection
    logger.info("\n--- Section 6: Selection Evaluation ---")
    selection_results = run_selection(
        correct, holdout_idx, generations, ground_truths,
        topo_scores, comp_features, comp_correct, comp_map, feature_names,
    )

    # Summary
    elapsed = (time.time() - t_start) / 60
    summary = {
        "dataset": "MATH-500",
        "model": MODEL_NAME_7B,
        "n_problems": n,
        "greedy_accuracy": float(correct.mean()),
        "greedy_correct": int(correct.sum()),
        "topo_auroc": model_results["topo_auroc"],
        "topo_auroc_ci": model_results["topo_auroc_ci"],
        "best_baseline": model_results["best_baseline"],
        "best_baseline_auroc": model_results["best_baseline_auroc"],
        "mv_net_gain": selection_results["ungated_mv"]["net_gain"],
        "weighted_vote_net_gain": selection_results["ungated_weighted"]["net_gain"],
        "oracle_ceiling": selection_results["oracle_ceiling"],
        "elapsed_minutes": float(elapsed),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 70)
    logger.info("MATH-500 × 7B PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Accuracy: %d/%d (%.1f%%)", correct.sum(), n, 100 * correct.mean())
    logger.info("Topo AUROC: %.4f", model_results["topo_auroc"])
    logger.info("MV: %+d, Weighted: %+d", selection_results["ungated_mv"]["net_gain"], selection_results["ungated_weighted"]["net_gain"])
    logger.info("Elapsed: %.1f minutes", elapsed)


if __name__ == "__main__":
    main()
