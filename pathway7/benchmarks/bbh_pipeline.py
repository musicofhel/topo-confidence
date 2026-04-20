#!/usr/bin/env python3
"""BBH benchmark pipeline: 3 subsets x 250 problems.

Generates 3-shot CoT completions for three Big-Bench Hard subsets,
extracts hidden-state trajectories, computes Euclidean + non-Euclidean PH
features, evaluates correctness, and fits logistic regression AUROC models.

Reports 4 models: Euclidean PH (8), non-Euclidean PH (18), combined (26),
plus a mean-logprob baseline for comparison.

Subsets:
  - tracking_shuffled_objects_seven_objects (250, answer A-G)
  - logical_deduction_seven_objects (250, answer A-G)
  - web_of_lies (250, answer Yes/No)

Run on RunPod: python pathway7/benchmarks/bbh_pipeline.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from pathway7.distance_metrics import compute_all_ph_features

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results_bbh"
SUBSAMPLE = 100
N_PCA = 45
LR_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=42)
N_EUCLID = 8  # first 8 features are Euclidean PH

BBH_SUBSETS = [
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
]

# Pre-downloaded CoT prompts path (in Docker image)
COT_PROMPTS_DIR = Path("/opt/bbh-prompts")
# Fallback: download at runtime
COT_PROMPTS_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/{subset}.txt"


def subsample_points(points: np.ndarray, n: int = SUBSAMPLE) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n, dtype=int)
    return points[idx]


def load_cot_prompt(subset: str) -> str:
    """Load 3-shot CoT prompt for a BBH subset."""
    local_path = COT_PROMPTS_DIR / f"{subset}.txt"
    if local_path.exists():
        return local_path.read_text()

    # Fallback: download
    import urllib.request
    url = COT_PROMPTS_URL.format(subset=subset)
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.read().decode()
    except Exception as e:
        print(f"  WARNING: Could not load CoT prompt for {subset}: {e}")
        return ""


def format_bbh_prompt(question: str, cot_prompt: str, tokenizer) -> str:
    """Format BBH question with 3-shot CoT as chat prompt."""
    user_content = f"{cot_prompt}\n\nQ: {question}\nA: Let's think step by step."
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_answer_bbh(response: str) -> str:
    """Extract answer from BBH CoT response."""
    match = re.search(r"(?i)the answer is\s*\(?([A-Za-z]+)\)?", response)
    if match:
        return match.group(1).strip("()").upper()
    return ""


def generate_and_extract(
    model, tokenizer, prompt: str, max_new_tokens: int = 512,
) -> tuple[str, np.ndarray, float]:
    """Generate completion, extract trajectory + mean logprob."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    gen_ids = out.sequences[0][inputs.input_ids.shape[1]:]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

    traj = torch.stack([
        s[-1][0, -1, :].float().cpu()
        for s in out.hidden_states
    ]).numpy()

    # Mean token log-probability
    if len(out.scores) > 0 and len(gen_ids) > 0:
        logprobs = []
        for i, scores in enumerate(out.scores):
            if i >= len(gen_ids):
                break
            lp = torch.log_softmax(scores[0].float(), dim=-1)
            logprobs.append(lp[gen_ids[i]].item())
        mean_logprob = float(np.mean(logprobs))
    else:
        mean_logprob = -100.0

    return completion, traj, mean_logprob


def run_subset(
    model, tokenizer, subset: str, cot_prompt: str,
) -> dict:
    """Run a single BBH subset."""
    from datasets import load_dataset
    ds = load_dataset("lukaemon/bbh", subset, split="test")
    print(f"\n  {subset}: {len(ds)} problems")

    completions = []
    trajectories = []
    correct = []
    logprobs = []

    t0 = time.time()
    for i, item in enumerate(ds):
        prompt = format_bbh_prompt(item["input"], cot_prompt, tokenizer)
        completion, traj, mean_lp = generate_and_extract(model, tokenizer, prompt)

        pred = extract_answer_bbh(completion)
        target = item["target"].strip("()").upper()
        is_correct = pred == target

        completions.append({
            "input": item["input"], "response": completion,
            "pred": pred, "target": target, "correct": is_correct,
            "mean_logprob": mean_lp,
        })
        trajectories.append(traj)
        correct.append(is_correct)
        logprobs.append(mean_lp)

        if (i + 1) % 50 == 0:
            acc = sum(correct) / len(correct)
            print(f"    {i+1}/{len(ds)}: {acc:.1%} accuracy, {time.time()-t0:.0f}s")

    labels = np.array(correct)
    elapsed = time.time() - t0
    print(f"    Done: {labels.mean():.1%} accuracy ({labels.sum()}/{len(labels)}), {elapsed:.0f}s")

    return {
        "subset": subset,
        "labels": labels,
        "trajectories": trajectories,
        "completions": completions,
        "logprobs": logprobs,
    }


def evaluate_feature_set(
    X_train: np.ndarray,
    X_holdout: np.ndarray,
    y_train: np.ndarray,
    y_holdout: np.ndarray,
    name: str,
    C: float = 1.0,
) -> dict:
    """Fit LR on a feature set and return AUROC metrics."""
    if X_train.shape[1] == 0:
        return {"name": name, "auroc_holdout": 0.0, "auroc_cv": 0.0, "n_features": 0}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ho = scaler.transform(X_holdout)

    lr = LogisticRegression(C=C, **LR_PARAMS)
    lr.fit(X_tr, y_train)

    probs = lr.predict_proba(X_ho)[:, 1]
    auroc = roc_auc_score(y_holdout, probs) if len(np.unique(y_holdout)) > 1 else 0.0

    n_folds = min(10, max(2, int(y_train.sum())))
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    probs_cv = cross_val_predict(
        LogisticRegression(C=C, **LR_PARAMS),
        X_tr, y_train, cv=cv, method="predict_proba",
    )[:, 1]
    auroc_cv = roc_auc_score(y_train, probs_cv) if len(np.unique(y_train)) > 1 else 0.0

    print(f"  {name:30s} AUROC: holdout={auroc:.4f}, CV={auroc_cv:.4f}")
    return {
        "name": name,
        "auroc_holdout": round(auroc, 4),
        "auroc_cv": round(auroc_cv, 4),
        "n_features": X_train.shape[1],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BBH Pipeline: 3 Subsets x 250 Problems")
    print("=" * 70)

    # Load model
    print(f"\nLoading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Run all subsets
    all_results = []
    all_labels = []
    all_trajs = []
    all_logprobs = []

    for subset in BBH_SUBSETS:
        cot_prompt = load_cot_prompt(subset)
        result = run_subset(model, tokenizer, subset, cot_prompt)
        all_results.append(result)
        all_labels.extend(result["labels"])
        all_trajs.extend(result["trajectories"])
        all_logprobs.extend(result["logprobs"])

        # Save per-subset completions
        subset_dir = OUTPUT_DIR / subset
        subset_dir.mkdir(exist_ok=True)
        (subset_dir / "completions.json").write_text(
            json.dumps(result["completions"], indent=2, default=float)
        )

    labels = np.array(all_labels)
    trajectories = all_trajs

    # ---- Feature computation ----
    print(f"\nComputing features for {len(trajectories)} problems ...")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9999)
    train_idx, hold_idx = next(sss.split(np.zeros(len(labels)), labels.astype(int)))

    train_trajs = [trajectories[i] for i in train_idx]
    all_points = np.concatenate(train_trajs, axis=0)
    n_comp = min(N_PCA, all_points.shape[1], all_points.shape[0])
    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(all_points)
    print(f"  PCA: {n_comp} components, {100*pca.explained_variance_ratio_.sum():.1f}% variance")

    # Compute all PH features (26 per problem)
    all_features = []
    all_names = None
    for j, traj in enumerate(trajectories):
        reduced = pca.transform(traj)
        sub = subsample_points(reduced)
        feats, names = compute_all_ph_features(sub, k_effres=30)
        all_features.append(feats)
        if all_names is None:
            all_names = names
        if (j + 1) % 100 == 0:
            print(f"  Features: {j+1}/{len(trajectories)}")

    X_all = np.stack(all_features)
    X_train = X_all[train_idx]
    X_holdout = X_all[hold_idx]
    y_train = labels[train_idx]
    y_holdout = labels[hold_idx]

    print(f"  {X_all.shape[1]} features, train={len(train_idx)}, holdout={len(hold_idx)}")
    print(f"  Train: {y_train.sum()}/{len(y_train)} correct, Holdout: {y_holdout.sum()}/{len(y_holdout)} correct")

    # ---- Evaluate multiple feature sets (pooled across all subsets) ----
    print(f"\n{'='*70}")
    print("Model Comparison (pooled across subsets)")
    print(f"{'='*70}")

    results_models = []

    # Euclidean PH only (8 features)
    results_models.append(evaluate_feature_set(
        X_train[:, :N_EUCLID], X_holdout[:, :N_EUCLID],
        y_train, y_holdout, "Euclidean PH (8)"))

    # Non-Euclidean PH only (18 features)
    results_models.append(evaluate_feature_set(
        X_train[:, N_EUCLID:], X_holdout[:, N_EUCLID:],
        y_train, y_holdout, "Non-Euclidean PH (18)"))

    # Combined (26 features)
    results_models.append(evaluate_feature_set(
        X_train, X_holdout, y_train, y_holdout, "Combined PH (26)"))

    # Logprob baseline
    lp_arr = np.array(all_logprobs)
    lp_auroc_hold = roc_auc_score(y_holdout, lp_arr[hold_idx]) if len(np.unique(y_holdout)) > 1 else 0.0
    lp_auroc_train = roc_auc_score(y_train, lp_arr[train_idx]) if len(np.unique(y_train)) > 1 else 0.0
    print(f"  {'mean_logprob baseline':30s} AUROC: holdout={lp_auroc_hold:.4f}, train={lp_auroc_train:.4f}")
    results_models.append({
        "name": "mean_logprob baseline",
        "auroc_holdout": round(lp_auroc_hold, 4),
        "auroc_cv": round(lp_auroc_train, 4),
        "n_features": 1,
    })

    # Per-subset accuracy
    subset_metrics = []
    offset = 0
    for result in all_results:
        n = len(result["labels"])
        sub_labels = labels[offset:offset + n]
        acc = float(sub_labels.mean())
        subset_metrics.append({
            "subset": result["subset"],
            "n_problems": n,
            "accuracy": round(acc, 4),
            "n_correct": int(sub_labels.sum()),
        })
        offset += n

    # ---- Save results ----
    best_topo = max(results_models[:3], key=lambda r: r["auroc_holdout"])
    summary = {
        "benchmark": "bbh",
        "model": MODEL_NAME,
        "subsets": BBH_SUBSETS,
        "n_problems_total": len(labels),
        "accuracy_total": round(float(labels.mean()), 4),
        "n_train": len(train_idx),
        "n_holdout": len(hold_idx),
        "models": results_models,
        "best_topo": best_topo["name"],
        "best_topo_auroc": best_topo["auroc_holdout"],
        "logprob_auroc": round(float(lp_auroc_hold), 4),
        "topo_beats_logprob": bool(best_topo["auroc_holdout"] > lp_auroc_hold),
        "per_subset": subset_metrics,
        "feature_names": all_names,
        "n_features_total": X_all.shape[1],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    np.save(OUTPUT_DIR / "features_all.npy", X_all)
    np.save(OUTPUT_DIR / "labels.npy", labels)
    np.save(OUTPUT_DIR / "logprobs.npy", lp_arr)

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"  Total accuracy: {labels.mean():.1%} ({labels.sum()}/{len(labels)})")
    for sm in subset_metrics:
        print(f"    {sm['subset']}: {sm['accuracy']:.1%} ({sm['n_correct']}/{sm['n_problems']})")
    print(f"  Best topo: {best_topo['name']} AUROC={best_topo['auroc_holdout']:.4f}")
    print(f"  Logprob baseline AUROC={lp_auroc_hold:.4f}")
    print(f"  Topo beats logprob: {best_topo['auroc_holdout'] > lp_auroc_hold}")
    print(f"\n  Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
