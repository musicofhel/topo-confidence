#!/usr/bin/env python3
"""Phase 3 (SPEC v6, V5-2) — the held-out T5 free-baseline test.

The v6 headline: the free length+mean_logprob readout is the most GENERALIZING
correctness signal (T1 0.845, T3 0.865) and needs NO activations. The live T5
question is whether it clears >=0.70 on two models that played ZERO role in any
selection — a near-family (SmolLM2-1.7B, Llama-arch) and a far-family holdout.
If yes, it is a shippable, model-agnostic, generate-then-abstain F-8 gate.

This is a GENERATION-ONLY pass (no hidden states): per problem we capture the
greedy completion -> correctness (math_verify), n_gen_tokens, mean_logprob.
mean_logprob is computed by a memory-bounded teacher-forcing scoring pass
(batch=1, logits gathered then freed) so it works for Gemma's 256k vocab on 8GB.

Recipe parity with how the Qwen caches were built
(pathway8_layerwise/extraction_utils.py): bf16, sdpa, greedy, SYSTEM_PROMPT,
mean of log_softmax at the realised next token over generated positions.

Usage:
    python phase3_heldout.py smollm2     # generate + save results/phase3_smollm2.npz
    python phase3_heldout.py gemma
    python phase3_heldout.py --eval       # compute free-baseline OOF AUROC for all saved
Local only (2060 Super). NOT RunPod (CLAUDE.md).
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "causal_dom"))
from common import _extract_boxed, check_correct  # noqa: E402

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

MODELS = {
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",   # near-family (Llama-arch)
    "gemma":   "google/gemma-2-2b-it",                   # far-family (gated)
    "olmo2":   "allenai/OLMo-2-0425-1B-Instruct",        # far-family alt (open)
    "falcon3": "tiiuae/Falcon3-1B-Instruct",             # far-family alt (open)
}
MAX_NEW = 1024


def load_math500():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return ds


def build_prompt(tok, problem: str) -> str:
    """Chat prompt; fold SYSTEM_PROMPT into the user turn if the template has no
    system role (Gemma)."""
    try:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": problem}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\n{problem}"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def eos_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id and i >= 0:
            ids.add(i)
    return sorted(i for i in ids if i is not None and i >= 0)


@torch.inference_mode()
def generate_and_score(model, tok, prompts, eos, gen_bs=8):
    """Two passes: (1) batched greedy generation -> sequences; (2) per-sample
    teacher-forcing forward -> mean_logprob (memory-bounded). Returns per-sample
    (text, n_gen_tokens, mean_logprob)."""
    dev = model.device
    texts, n_gens, logps = [], [], []
    n = len(prompts)
    eos_set = set(eos)
    for s in range(0, n, gen_bs):
        chunk = prompts[s:s + gen_bs]
        enc = tok(chunk, return_tensors="pt", padding=True).to(dev)
        plen = enc.input_ids.shape[1]
        gen = model.generate(
            **enc, max_new_tokens=MAX_NEW, do_sample=False,
            temperature=1.0, top_p=1.0,
            pad_token_id=tok.pad_token_id, eos_token_id=eos,
        )
        new = gen[:, plen:]  # (b, g) left-padded => prompt in first plen cols
        for b in range(new.shape[0]):
            ids = new[b].tolist()
            # truncate at first eos (inclusive of nothing after it)
            cut = len(ids)
            for j, t in enumerate(ids):
                if t in eos_set:
                    cut = j  # exclude the eos token itself from the count
                    break
            gen_ids = new[b, :cut]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            texts.append(text)
            n_gens.append(int(cut))
            # teacher-forcing logprob: forward [prompt(b) ++ gen_ids], gather
            full = torch.cat([enc.input_ids[b][enc.attention_mask[b].bool()],
                              gen_ids.to(dev)]).unsqueeze(0)
            if cut == 0:
                logps.append(-100.0)
                continue
            out = model(full)
            logits = out.logits[0].float()                 # (L, V)
            lp = torch.log_softmax(logits, dim=-1)
            # next-token logprob for the generated span:
            start = full.shape[1] - cut - 1
            chosen = full[0, start + 1: start + 1 + cut]
            tok_lp = lp[start:start + cut, :].gather(1, chosen.unsqueeze(1)).squeeze(1)
            logps.append(float(tok_lp.mean().cpu()))
            del out, logits, lp
        del gen, enc
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        done = min(s + gen_bs, n)
        print(f"    {done}/{n}", flush=True)
    return texts, np.array(n_gens), np.array(logps, dtype=np.float64)


def run_model(key: str, gen_bs=8, limit=None):
    name = MODELS[key]
    print(f"[phase3] {key} = {name}", flush=True)
    ds = load_math500()
    if limit:
        ds = ds.select(range(limit))
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa", trust_remote_code=True)
    model.eval()
    print(f"  loaded on {next(model.parameters()).device}", flush=True)
    eos = eos_ids(tok)
    prompts = [build_prompt(tok, ds[i]["problem"]) for i in range(len(ds))]
    t0 = time.time()
    texts, n_gen, logp = generate_and_score(model, tok, prompts, eos, gen_bs)
    y = np.array([check_correct(texts[i], ds[i]["answer"]) for i in range(len(ds))],
                 dtype=bool)
    subjects = np.array([ds[i]["subject"] for i in range(len(ds))], dtype=object)
    acc = float(y.mean())
    out = RESULTS / (f"phase3_{key}_smoke.npz" if limit else f"phase3_{key}.npz")
    np.savez(out, y=y, mean_logprob=logp, n_gen_tokens=n_gen.astype(np.float64),
             subjects=subjects, texts=np.array(texts, dtype=object), model=name)
    dt = time.time() - t0
    print(f"  acc={acc:.4f} ({y.sum()}/{len(y)})  mean_n_gen={n_gen.mean():.0f}  "
          f"mean_logprob={logp.mean():.3f}  {dt/60:.1f} min", flush=True)
    print(f"  -> {out}", flush=True)
    return out


def eval_free_baseline():
    """OOF 5-fold AUROC of free_baseline (length+logprob) per saved held-out model.
    T5 verdict: >=0.70 = shippable. Mirrors probes.fit_score_oof + frozen folds."""
    import json
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    def oof_auroc(X, y, seed=0):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        sc = np.zeros(len(y))
        for tr, te in skf.split(X, y):
            s = StandardScaler().fit(X[tr])
            est = LogisticRegression(max_iter=2000, C=1.0).fit(s.transform(X[tr]), y[tr])
            sc[te] = est.predict_proba(s.transform(X[te]))[:, 1]
        return float(roc_auc_score(y, sc))

    rows = {}
    for f in sorted(RESULTS.glob("phase3_*.npz")):
        key = f.stem.replace("phase3_", "")
        d = np.load(f, allow_pickle=True)
        y = d["y"].astype(bool)
        L = d["n_gen_tokens"].astype(float)
        lp = d["mean_logprob"].astype(float)
        if y.sum() < 5 or (~y).sum() < 5:
            rows[key] = {"acc": float(y.mean()), "note": "degenerate label balance"}
            continue
        free = oof_auroc(np.column_stack([L, lp]), y)
        leno = oof_auroc(L.reshape(-1, 1), y)
        lpo = oof_auroc(lp.reshape(-1, 1), y)
        rows[key] = {"model": str(d["model"]), "n": int(len(y)), "acc": float(y.mean()),
                     "free_baseline_auroc": free, "length_only": leno,
                     "logprob_only": lpo, "T5_pass_0.70": bool(free >= 0.70)}
        print(f"{key:10s} acc={y.mean():.3f}  free={free:.4f}  len={leno:.4f}  "
              f"logp={lpo:.4f}  T5>=0.70: {free>=0.70}", flush=True)
    (RESULTS / "phase3_t5_free_baseline.json").write_text(json.dumps(rows, indent=2))
    print(f"-> {RESULTS / 'phase3_t5_free_baseline.json'}", flush=True)
    return rows


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--eval" in args:
        eval_free_baseline()
    else:
        bs = 8
        limit = None
        for a in args:
            if a.startswith("--bs="):
                bs = int(a.split("=")[1])
            if a.startswith("--limit="):
                limit = int(a.split("=")[1])
        keys = [a for a in args if not a.startswith("--")]
        for k in keys:
            run_model(k, gen_bs=bs, limit=limit)
