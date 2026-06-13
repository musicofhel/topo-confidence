#!/usr/bin/env python3
"""Phase 2 arm-2A (SPEC v6, V5-3) — prompt-token-cloud features, the ONLY untested
geometry and the sole remaining shot at a portable pre-flight (Tier-A) gate (H-E).

Logged NOT_IMPLEMENTED in phase1_bakeoff.py because Phase 1 ran on cached GENERATION
states only. arm-2A is geometry over the PROMPT tokens (pre-generation): a single
prefill forward per problem, capture L19 over all prompt positions -> (T_prompt, H)
cloud, summarise by its covariance spectrum (top-K log-eigvals; mirrors the cov_spectrum
recipe but over prompt tokens, so it is immune to the generation-length confound that
killed the V3-1 token-cloud). Its mandatory baseline is PROMPT-LENGTH (exogenous,
unmeasured anywhere). The decisive in-domain gate: does prompt-cloud add over
prompt-length on Qwen-1.5B MATH? If not, H-E is refuted and geometry is fully closed.

Local only (2060, 8GB). Qwen-1.5B fits (3GB); Qwen-7B (15GB bf16) does NOT -> the
cross-scale arm-2A cell is GPU-blocked locally and logged as such.

    python phase2_arm2a.py extract          # prefill pass -> cache prompt clouds
    python phase2_arm2a.py analyze          # in-domain gate vs prompt-length
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
CACHE = HERE / "cache"; CACHE.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "causal_dom"))

LAYER = 19
K_MAX = 20
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def cloud_eigvals(cloud: np.ndarray) -> np.ndarray:
    """Top eigvals (descending) of the cloud's sample covariance (mirrors
    cov_spectrum/recompute_pc1_resid_cov_spectrum.py::cloud_eigvals)."""
    n = cloud.shape[0]
    centered = cloud - cloud.mean(axis=0)
    s = np.linalg.svd(centered, compute_uv=False)
    eigvals = (s ** 2) / max(n - 1, 1)
    return np.sort(eigvals)[::-1]


def extract():
    """Prefill pass over the MATH-500 prompts on Qwen-1.5B; cache prompt-cloud
    eigvals + prompt-length. Prompt construction matches generation (format_chat
    + SYSTEM_PROMPT) so the L19 cloud is the exact pre-generation state."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from common import format_chat
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True).eval()
    print(f"  loaded {MODEL} on {next(model.parameters()).device}", flush=True)

    n = len(ds)
    eig = np.zeros((n, K_MAX), dtype=np.float64)
    plen = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for i in range(n):
        prompt = format_chat(ds[i]["problem"], tok)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        cloud = out.hidden_states[LAYER][0].float().cpu().numpy()  # (T_prompt, H)
        plen[i] = cloud.shape[0]
        e = cloud_eigvals(cloud)
        eig[i, :min(K_MAX, len(e))] = e[:K_MAX]
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{n}  ({(time.time()-t0)/60:.1f} min)", flush=True)
    out_f = CACHE / "arm2a_prompt_cloud_qwen1.5b_math.npz"
    np.savez(out_f, eigvals=eig, prompt_len=plen, layer=LAYER)
    print(f"  -> {out_f}  ({(time.time()-t0)/60:.1f} min)", flush=True)


def analyze():
    """In-domain gate: prompt-cloud (top-K log-eigvals) + prompt-length vs
    prompt-length-only, OOF + paired DeLong. Resolves H-E for the deployable case."""
    import json
    import activation_loader as AL
    from metrics import frozen_folds, auroc_sym, delong_paired_test
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    f = CACHE / "arm2a_prompt_cloud_qwen1.5b_math.npz"
    if not f.exists():
        print("run `extract` first."); return 1
    d = np.load(f)
    eig = d["eigvals"]; plen = d["prompt_len"].astype(float)
    log_eig = np.log(np.clip(eig, 1e-12, None))     # top-K log-eigvals (the feature)
    y = AL.load_cell("qwen1.5b", "math").y
    assert len(y) == len(plen), f"alignment {len(y)} vs {len(plen)}"
    folds = frozen_folds(y)

    def oof(X):
        s = np.zeros(len(y))
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            est = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
            s[te] = est.predict_proba(sc.transform(X[te]))[:, 1]
        return s

    s_plen = oof(plen.reshape(-1, 1))
    s_cloud = oof(log_eig)
    s_comb = oof(np.column_stack([log_eig, plen]))
    test = delong_paired_test(y, s_comb, s_plen)
    res = {
        "model": MODEL, "layer": LAYER, "K": K_MAX, "n": int(len(y)),
        "prompt_length_only_auroc": auroc_sym(y, s_plen),
        "prompt_cloud_only_auroc": auroc_sym(y, s_cloud),
        "cloud_plus_length_auroc": auroc_sym(y, s_comb),
        "gate_delta_vs_promptlen": test["delta"], "gate_p": test["p"], "gate_z": test["z"],
        "prompt_len_corr_with_cloud_pc1": float(
            np.corrcoef(plen, log_eig[:, 0])[0, 1]),
        "verdict_H_E": ("prompt-cloud ADDS over prompt-length"
                        if test["delta"] > 0 and test["p"] < 0.05
                        else "prompt-cloud does NOT add over prompt-length (H-E refuted)"),
        "note": ("cross-scale arm-2A (Qwen-7B prompt clouds) is GPU-blocked locally "
                 "(7B bf16 ~15GB > 2060 8GB); in-domain gate on 1.5B is decisive for H-E."),
    }
    (RESULTS / "phase2_arm2a.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"-> {RESULTS / 'phase2_arm2a.json'}")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "extract":
        extract()
    else:
        raise SystemExit(analyze())
