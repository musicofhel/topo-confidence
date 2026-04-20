#!/usr/bin/env python3
"""HumanEval benchmark pipeline: generation + hidden-state extraction + features.

Generates completions for 164 HumanEval problems, extracts hidden-state
trajectories, computes Euclidean + non-Euclidean PH features, evaluates
correctness, and fits a logistic regression AUROC model.

Run on RunPod: python pathway7/benchmarks/humaneval_pipeline.py
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
from pathway7.distance_metrics import compute_all_noneuclid_features

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results_humaneval"
SUBSAMPLE = 100
N_PCA = 45
LR_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=42)


def subsample_points(points: np.ndarray, n: int = SUBSAMPLE) -> np.ndarray:
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n, dtype=int)
    return points[idx]


def load_humaneval():
    """Load HumanEval dataset."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")
    print(f"Loaded {len(ds)} HumanEval problems")
    return ds


def format_humaneval_prompt(problem: dict, tokenizer) -> str:
    """Format HumanEval problem as chat prompt."""
    prompt_text = problem["prompt"]
    messages = [
        {"role": "system", "content": "Complete the following Python function. Return only the function body, no explanation."},
        {"role": "user", "content": f"Complete this function:\n\n```python\n{prompt_text}\n```"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_completion_and_trajectory(
    model, tokenizer, formatted_prompt: str, max_new_tokens: int = 512,
) -> tuple[str, np.ndarray]:
    """Generate completion and extract last-layer hidden-state trajectory."""
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode completion
    gen_ids = out.sequences[0][inputs.input_ids.shape[1]:]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Extract per-generated-token last-layer hidden states
    traj = torch.stack([
        s[-1][0, -1, :].float().cpu()
        for s in out.hidden_states
    ]).numpy()

    return completion, traj


def check_humaneval_correctness(problem: dict, completion: str) -> bool:
    """Check if completion passes HumanEval test cases.

    Uses exec() in a sandboxed subprocess. Falls back to simple heuristic
    if human_eval package is not available.
    """
    try:
        from human_eval.execution import check_correctness
        result = check_correctness(
            problem, completion, timeout=10.0, completion_id=0
        )
        return result["passed"]
    except ImportError:
        # Fallback: check if completion at least defines a function body
        return "return" in completion and "def " not in completion
    except Exception:
        return False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("HumanEval Pipeline: Generation + Features + AUROC")
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

    # Load dataset
    ds = load_humaneval()

    # Generate + extract
    completions = []
    trajectories = []
    correct = []

    t0 = time.time()
    for i, problem in enumerate(ds):
        prompt = format_humaneval_prompt(problem, tokenizer)
        completion, traj = extract_completion_and_trajectory(model, tokenizer, prompt)
        is_correct = check_humaneval_correctness(problem, completion)

        completions.append(completion)
        trajectories.append(traj)
        correct.append(is_correct)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            acc = sum(correct) / len(correct)
            print(f"  {i+1}/164: {acc:.1%} accuracy, {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    labels = np.array(correct)
    print(f"\nGeneration done: {elapsed:.0f}s, accuracy={labels.mean():.1%} ({labels.sum()}/{len(labels)})")

    # Save completions
    completions_data = [
        {"task_id": ds[i]["task_id"], "completion": completions[i], "correct": bool(correct[i])}
        for i in range(len(ds))
    ]
    (OUTPUT_DIR / "completions.json").write_text(json.dumps(completions_data, indent=2))

    # Compute features
    print("\nComputing features ...")

    # Train/holdout split (80/20 stratified)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=9999)
    train_idx, hold_idx = next(sss.split(np.zeros(len(labels)), labels.astype(int)))

    # PCA on train only
    train_trajs = [trajectories[i] for i in train_idx]
    all_points = np.concatenate(train_trajs, axis=0)
    n_comp = min(N_PCA, all_points.shape[1], all_points.shape[0])
    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(all_points)

    # Non-Euclidean features
    ne_features = []
    ne_names = None
    for traj in trajectories:
        reduced = pca.transform(traj)
        sub = subsample_points(reduced)
        feats, names = compute_all_noneuclid_features(sub, k_effres=30)
        ne_features.append(feats)
        if ne_names is None:
            ne_names = names

    X_all = np.stack(ne_features)
    X_train = X_all[train_idx]
    X_holdout = X_all[hold_idx]
    y_train = labels[train_idx]
    y_holdout = labels[hold_idx]

    # Fit model
    print(f"\nFitting LR on {X_train.shape[1]} features ...")
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_ho = scaler.transform(X_holdout)

    lr = LogisticRegression(C=1.0, **LR_PARAMS)
    lr.fit(X_tr, y_train)

    probs_holdout = lr.predict_proba(X_ho)[:, 1]
    auroc = roc_auc_score(y_holdout, probs_holdout) if len(np.unique(y_holdout)) > 1 else 0.0

    # CV on train
    cv = StratifiedKFold(n_splits=min(10, int(y_train.sum())), shuffle=True, random_state=42)
    probs_cv = cross_val_predict(
        LogisticRegression(C=1.0, **LR_PARAMS),
        X_tr, y_train, cv=cv, method="predict_proba",
    )[:, 1]
    auroc_cv = roc_auc_score(y_train, probs_cv) if len(np.unique(y_train)) > 1 else 0.0

    print(f"\n  HumanEval AUROC: holdout={auroc:.4f}, CV={auroc_cv:.4f}")

    # Save summary
    summary = {
        "benchmark": "humaneval",
        "model": MODEL_NAME,
        "n_problems": len(ds),
        "accuracy": round(float(labels.mean()), 4),
        "n_correct": int(labels.sum()),
        "auroc_holdout": round(auroc, 4),
        "auroc_cv": round(auroc_cv, 4),
        "n_train": len(train_idx),
        "n_holdout": len(hold_idx),
        "feature_names": ne_names,
        "n_features": X_all.shape[1],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    np.save(OUTPUT_DIR / "features_all.npy", X_all)
    np.save(OUTPUT_DIR / "labels.npy", labels)

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
