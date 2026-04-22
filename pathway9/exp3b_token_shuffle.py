"""Exp 3b: Token-shuffle null (GPU).

Shuffle input token order, forward pass, compute PH on the last-layer
hidden states. Tests whether PH features capture order-dependent sequence
structure or just the embedding distribution (which is order-invariant).

If real vs shuffled PH features are barely distinguishable, PH is NOT
capturing attention-induced structure — just the token distribution.

50 random MATH-500 problems × 5 shuffles = 250 extra forward passes
(~15 min on H100).

Output: pathway9/results/exp3b_token_shuffle.json + exp3b.done
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from ripser import ripser
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
N_SAMPLE_PROBLEMS = 50
N_SHUFFLES = 5
SEED = 42


def compute_5_ph_features(cloud, subsample=100, pca_components=45):
    """Same 5-feature PH pipeline as exp3a v2."""
    if len(cloud) < 5:
        return np.zeros(5)
    n_comp = min(pca_components, cloud.shape[0] - 1, cloud.shape[1])
    pca = PCA(n_components=n_comp)
    cloud_pca = pca.fit_transform(cloud)
    if len(cloud_pca) > subsample:
        idx = np.linspace(0, len(cloud_pca) - 1, subsample, dtype=int)
        cloud_pca = cloud_pca[idx]
    dgms = ripser(cloud_pca, maxdim=1)["dgms"]
    feats = np.zeros(5)
    h0 = dgms[0][np.isfinite(dgms[0][:, 1])]
    if len(h0) > 0:
        lt = h0[:, 1] - h0[:, 0]
        feats[0] = lt.max()
        ltn = lt / lt.sum() if lt.sum() > 0 else lt
        ltn = ltn[ltn > 0]
        feats[1] = -np.sum(ltn * np.log(ltn)) if len(ltn) > 0 else 0
    if len(dgms) > 1 and len(dgms[1]) > 0:
        h1 = dgms[1][np.isfinite(dgms[1][:, 1])]
        if len(h1) > 0:
            lt1 = h1[:, 1] - h1[:, 0]
            lt1n = lt1 / lt1.sum() if lt1.sum() > 0 else lt1
            lt1n = lt1n[lt1n > 0]
            feats[2] = -np.sum(lt1n * np.log(lt1n)) if len(lt1n) > 0 else 0
            feats[3] = len(h1)
            feats[4] = lt1.max()
    return feats


def main():
    print("=" * 60)
    print("EXP 3B: Token-Shuffle Null (GPU)")
    print("=" * 60, flush=True)

    # Load 50 random MATH-500 problem prompts
    with open(REPO / "data/experiment1_v2/trajectory_meta.json") as f:
        meta = json.load(f)
    # Use generated texts as forward-pass inputs (proxy for prompt+completion).
    # NOTE: these are completions; for a true test we want prompts+completions,
    # but the existing trajectory captures completion-side hidden states.
    # Here we forward-pass the completion text alone as an ablation test.
    generated_texts = meta["generated_texts"]

    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(500, size=N_SAMPLE_PROBLEMS, replace=False))
    print(f"  Sampled problems: {sample_idx[:5]}... (n={N_SAMPLE_PROBLEMS})", flush=True)

    # Load model
    print(f"  Loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params", flush=True)

    real_ph = np.zeros((N_SAMPLE_PROBLEMS, 5))
    shuffled_ph = np.zeros((N_SAMPLE_PROBLEMS, N_SHUFFLES, 5))

    t0 = time.time()
    for i, pid in enumerate(sample_idx):
        text = generated_texts[pid]
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.cuda()

        if ids.shape[1] < 5:
            continue

        # Real forward pass
        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0].to(torch.float32).cpu().numpy()
        real_ph[i] = compute_5_ph_features(last_hidden)

        # Shuffled forward passes
        for k in range(N_SHUFFLES):
            g = torch.Generator(device="cuda").manual_seed(SEED + i * 100 + k)
            perm = torch.randperm(ids.shape[1], generator=g, device="cuda")
            shuf_ids = ids[:, perm]
            with torch.no_grad():
                out = model(shuf_ids, output_hidden_states=True)
            shuf_hidden = out.hidden_states[-1][0].to(torch.float32).cpu().numpy()
            shuffled_ph[i, k] = compute_5_ph_features(shuf_hidden)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_SAMPLE_PROBLEMS - i - 1)
            print(f"  {i+1}/{N_SAMPLE_PROBLEMS}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    print(f"  Done in {time.time()-t0:.0f}s", flush=True)

    # Metrics: feature-wise MAE between real and shuffled mean
    shuffled_mean = shuffled_ph.mean(axis=1)  # (n, 5)
    feature_mae = np.abs(real_ph - shuffled_mean).mean(axis=0)
    inter_problem_std = real_ph.std(axis=0)

    names = ["H0_max_lifetime", "H0_entropy", "H1_pers_entropy", "H1_n_features", "H1_max_lifetime"]
    feature_stats = {}
    print("\n  Per-feature comparison:")
    for j, name in enumerate(names):
        real_m, real_s = float(real_ph[:, j].mean()), float(real_ph[:, j].std())
        shuf_m, shuf_s = float(shuffled_mean[:, j].mean()), float(shuffled_mean[:, j].std())
        mae = float(feature_mae[j])
        ratio = mae / inter_problem_std[j] if inter_problem_std[j] > 0 else np.nan
        print(f"    {name:20s}  real={real_m:.2f}±{real_s:.2f}  shuf={shuf_m:.2f}±{shuf_s:.2f}  mae={mae:.2f}  mae/inter_std={ratio:.2f}", flush=True)
        feature_stats[name] = {
            "real_mean": real_m, "real_std": real_s,
            "shuffled_mean": shuf_m, "shuffled_std": shuf_s,
            "mae": mae, "mae_over_inter_problem_std": float(ratio),
        }

    # Aggregate: fraction of features where mae/inter_std < 1
    frac_indistinguishable = float(np.mean([s["mae_over_inter_problem_std"] < 1 for s in feature_stats.values()]))

    if frac_indistinguishable >= 0.8:
        verdict = (
            f"PH features are largely ORDER-INDEPENDENT: {frac_indistinguishable*100:.0f}% "
            f"of features differ from shuffled by less than 1 inter-problem std. "
            f"PH captures the embedding distribution, not attention-induced sequence structure."
        )
    elif frac_indistinguishable < 0.4:
        verdict = (
            f"PH features DO capture order: only {frac_indistinguishable*100:.0f}% indistinguishable. "
            f"Shuffled PH differs meaningfully from real."
        )
    else:
        verdict = f"MIXED — {frac_indistinguishable*100:.0f}% of features indistinguishable. Some order-sensitivity."

    print(f"\n  VERDICT: {verdict}", flush=True)

    results = {
        "experiment": "exp3b_token_shuffle",
        "n_problems_sampled": N_SAMPLE_PROBLEMS,
        "n_shuffles_per_problem": N_SHUFFLES,
        "sampled_problem_ids": sample_idx.tolist(),
        "feature_stats": feature_stats,
        "fraction_features_indistinguishable": frac_indistinguishable,
        "verdict": verdict,
    }
    (RESULTS / "exp3b_token_shuffle.json").write_text(json.dumps(results, indent=2))
    (RESULTS / "exp3b.done").touch()
    print("\n  Saved: pathway9/results/exp3b_token_shuffle.json")


if __name__ == "__main__":
    main()
