#!/usr/bin/env python3
"""v7_runpod_arms.py — SPEC v7 Phases 1b/1c pod workloads (H100 SXM, bf16).

Self-contained: needs only torch/transformers/datasets/numpy + pod_payload/
(built by v7_build_pod_payload.py, rsync'd alongside this script). All outputs
are JSON(.gz) in pod_results/ — pull them before stopping the pod (pods
without a network volume lose data on stop).

Tasks
  rescore     --model M --cell C   teacher-forced per-token features from
                                   cached texts (deterministic, no generation)
  ptrue       --model M --cell C   P(True) second forward, TWO frozen formats
  verbalized  --model M --cell C   binary sure/unsure + 0-100, short decode
  k8bbh       --model qwen1.5b     K=8 T=0.7 sampling on BBH 750 (dual-use:
                                   1b consistency arm + Phase-2 pseudo-labels)
  prm         --model prm7b        Qwen2.5-Math-PRM-7B scores 1.5B MATH texts
                                   (optional; resolves P11-FE403)

Models: qwen1.5b, qwen7b, smollm2, gemma, olmo2 (cells: math, bbh, t5).
Pre-registered P(True) formats and the verbalized parse protocol are FROZEN
here (G2 freezes which propagate to T5). Every output carries token counts
for the token-FLOP cost axis (7B = 4.7x 1.5B per token).
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PAYLOAD = HERE / "pod_payload"
OUT = HERE / "pod_results"

SYSTEM_PROMPT = ("Please reason step by step, and put your final answer "
                 "within \\boxed{}.")
MODELS = {
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "smollm2": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "gemma": "google/gemma-2-2b-it",
    "olmo2": "allenai/OLMo-2-0425-1B-Instruct",
    "prm7b": "Qwen/Qwen2.5-Math-PRM-7B",
}
BBH_SUBSETS = [
    "tracking_shuffled_objects_seven_objects",
    "logical_deduction_seven_objects",
    "web_of_lies",
]

_BBH_ANSWER_PATS = [
    re.compile(r"(?:the\s+answer\s+is)\s*\(([A-Z])\)", re.I),
    re.compile(r"answer\s*(?:is|:)\s*\(([A-Z])\)", re.I),
    re.compile(r"(?:the\s+answer\s+is)\s*([A-Z])\b", re.I),
    re.compile(r"answer\s*(?:is|:)\s*([A-Z])\b", re.I),
    re.compile(r"\\boxed\{\(?([A-Z])\)?\}", re.I),
    re.compile(r"\(([A-Z])\)\s*\.?\s*$", re.M),
    re.compile(r"(?:the\s+answer\s+is)\s*(Yes|No)\b", re.I),
    re.compile(r"answer\s*(?:is|:)\s*(Yes|No)\b", re.I),
]


def extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        nums = re.findall(r"-?\d+(?:\.\d+)?", text)
        return nums[-1] if nums else text.strip()[-80:]
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start: i - 1]


def bbh_extract_answer(response: str) -> str:
    for pat in _BBH_ANSWER_PATS:
        m = pat.findall(response)
        if m:
            ans = m[-1].strip()
            if len(ans) == 1 and ans.isalpha() and ans.upper() != "N":
                return f"({ans.upper()})"
            return ans.capitalize() if ans.lower() in ("yes", "no") else ans
    return ""


def math_answer_span(text: str):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return (start, i - 1)


def bbh_answer_span(text: str):
    for pat in _BBH_ANSWER_PATS:
        matches = list(pat.finditer(text))
        if matches:
            return matches[-1].span(1)
    return None


def chat(tok, messages: list[dict]) -> str:
    """apply_chat_template; fold system into user when unsupported (Gemma)."""
    try:
        return tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        sys_txt = "\n\n".join(m["content"] for m in messages
                              if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        if sys_txt and rest and rest[0]["role"] == "user":
            rest = ([{"role": "user",
                      "content": f"{sys_txt}\n\n{rest[0]['content']}"}]
                    + rest[1:])
        return tok.apply_chat_template(rest, tokenize=False,
                                       add_generation_prompt=True)


def gen_prompt(tok, item: dict, cell: str) -> str:
    """The EXACT generation-time prompt for a payload item."""
    if cell in ("math", "t5"):
        return chat(tok, [{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": item["problem"]}])
    cot = (PAYLOAD / "bbh_cot_prompts" / f"{item['subset']}.txt").read_text()
    user = f"{cot}\n\nQ: {item['question']}\nA: Let's think step by step."
    return chat(tok, [{"role": "user", "content": user}])


def load_payload(model_key: str, cell: str) -> list[dict]:
    name = {"math": f"{model_key}_math", "bbh": f"{model_key}_bbh",
            "t5": f"t5_{model_key}"}[cell]
    rows = []
    with gzip.open(PAYLOAD / f"{name}.jsonl.gz", "rt") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_model(key: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda").eval()
    print(f"loaded {name} (bf16, sdpa)", flush=True)
    return model, tok


def eos_ids(tok):
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<end_of_turn>", "<|eot_id|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if i is not None and i != tok.unk_token_id and i >= 0:
            ids.add(i)
    return sorted(i for i in ids if i is not None and i >= 0)


def dump(obj: dict, name: str):
    OUT.mkdir(exist_ok=True)
    f = OUT / name
    if name.endswith(".gz"):
        with gzip.open(f, "wt") as fh:
            json.dump(obj, fh)
    else:
        f.write_text(json.dumps(obj, indent=1, default=float))
    print(f"-> {f}", flush=True)


# ---------------------------------------------------------------------------
# Task: rescore (full-forward — keeps Gemma-2 logit softcapping faithful)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def task_rescore(model_key: str, cell: str, limit=None):
    model, tok = load_model(model_key)
    items = load_payload(model_key, cell)[:limit]
    span_fn = math_answer_span if cell in ("math", "t5") else bbh_answer_span
    cols = {k: [] for k in ("min_lp", "p10_lp", "mean_entropy",
                            "mean_top2_margin", "ans_span_lp",
                            "mean_lp_rescored", "n_gen", "n_prompt")}
    t0 = time.time()
    for i, item in enumerate(items):
        prompt = gen_prompt(tok, item, cell)
        p_ids = tok(prompt, return_tensors="pt").input_ids[0]
        enc = tok(item["text"], add_special_tokens=False,
                  return_offsets_mapping=True)
        t_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        T, P = len(t_ids), len(p_ids)
        cols["n_prompt"].append(P)
        if T == 0:
            for k in ("min_lp", "p10_lp", "mean_entropy", "mean_top2_margin",
                      "ans_span_lp", "mean_lp_rescored"):
                cols[k].append(float("nan"))
            cols["n_gen"].append(0)
            continue
        ids = torch.cat([p_ids, t_ids]).unsqueeze(0).cuda()
        logits = model(ids).logits[0, P - 1: P + T - 1].float()  # (T, V)
        logp = torch.log_softmax(logits, dim=-1)
        tgt = t_ids.cuda()
        lps = logp.gather(1, tgt[:, None])[:, 0].cpu().numpy()
        prob = logp.exp()
        ents = -(prob * logp).sum(-1).cpu().numpy()
        top2 = logits.topk(2, dim=-1).values
        marg = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        del logits, logp, prob, top2

        ans_lp = float("nan")
        span = span_fn(item["text"])
        if span is not None:
            a, b = span
            ti = [j for j, (cs, ce) in enumerate(enc["offset_mapping"])
                  if cs < b and ce > a]
            if ti:
                ans_lp = float(lps[ti].mean())
        k10 = max(1, int(np.ceil(0.10 * T)))
        cols["min_lp"].append(float(lps.min()))
        cols["p10_lp"].append(float(np.sort(lps)[:k10].mean()))
        cols["mean_entropy"].append(float(ents.mean()))
        cols["mean_top2_margin"].append(float(marg.mean()))
        cols["ans_span_lp"].append(ans_lp)
        cols["mean_lp_rescored"].append(float(lps.mean()))
        cols["n_gen"].append(T)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(items)} ({el/(i+1):.2f}s/p)", flush=True)

    cached = [it["cached_mean_logprob"] for it in items]
    fin = np.isfinite(cols["mean_lp_rescored"])
    r = float(np.corrcoef(np.array(cols["mean_lp_rescored"])[fin],
                          np.array(cached)[fin])[0, 1])
    print(f"alignment corr = {r:.4f}", flush=True)
    dump({"task": "rescore", "model": model_key, "cell": cell,
          "alignment_corr": r, "y": [it["y"] for it in items],
          "cost_tokens_scored": int(np.sum(cols["n_gen"])),
          **cols}, f"v7_rescore_{model_key}_{cell}.json")


# ---------------------------------------------------------------------------
# Task: P(True) — two FROZEN formats (pre-registered; G2 picks one for T5)
# ---------------------------------------------------------------------------

def _first_token_variants(tok, words):
    ids = set()
    for w in words:
        for v in (w, " " + w):
            t = tok(v, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
    return sorted(ids)


@torch.inference_mode()
def task_ptrue(model_key: str, cell: str, limit=None):
    model, tok = load_model(model_key)
    items = load_payload(model_key, cell)[:limit]
    ex = extract_boxed if cell in ("math", "t5") else bbh_extract_answer

    A_yes = _first_token_variants(tok, ["A"])
    A_no = _first_token_variants(tok, ["B"])
    B_true = _first_token_variants(tok, ["True", "TRUE", "true"])
    B_false = _first_token_variants(tok, ["False", "FALSE", "false"])

    def ptrue(prompt, yes_ids, no_ids):
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        logits = model(ids).logits[0, -1].float()
        lp = torch.log_softmax(logits, dim=-1)
        py = float(torch.logsumexp(lp[yes_ids], 0).exp())
        pn = float(torch.logsumexp(lp[no_ids], 0).exp())
        return py / (py + pn + 1e-12), ids.shape[1]

    rows = {"p_true_A": [], "p_true_B": [], "n_prompt_A": [], "n_prompt_B": []}
    t0 = time.time()
    for i, item in enumerate(items):
        q = item.get("problem") or item.get("question")
        ans = ex(item["text"]) or "(no final answer given)"
        pa = chat(tok, [{"role": "user", "content":
                         f"Question: {q}\n\nProposed answer: {ans}\n\n"
                         f"Is the proposed answer correct? (A) Yes (B) No\n"
                         f"Reply with exactly one letter, A or B."}])
        pb = chat(tok, [{"role": "user", "content":
                         f"Question: {q}\n\nProposed answer: {ans}\n\n"
                         f"True or False: the proposed answer is correct.\n"
                         f"Reply with exactly one word, True or False."}])
        va, na = ptrue(pa, A_yes, A_no)
        vb, nb = ptrue(pb, B_true, B_false)
        rows["p_true_A"].append(va); rows["n_prompt_A"].append(na)
        rows["p_true_B"].append(vb); rows["n_prompt_B"].append(nb)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)} ({(time.time()-t0)/(i+1):.2f}s/p)",
                  flush=True)
    dump({"task": "ptrue", "model": model_key, "cell": cell,
          "y": [it["y"] for it in items],
          "cost_tokens_scored": int(np.sum(rows["n_prompt_A"]) +
                                    np.sum(rows["n_prompt_B"])),
          **rows}, f"v7_ptrue_{model_key}_{cell}.json")


# ---------------------------------------------------------------------------
# Task: verbalized confidence — binary + 0-100, short decode, frozen parse
# ---------------------------------------------------------------------------

@torch.inference_mode()
def task_verbalized(model_key: str, cell: str, limit=None):
    model, tok = load_model(model_key)
    items = load_payload(model_key, cell)[:limit]
    eos = eos_ids(tok)

    def follow_up(item, question):
        if cell in ("math", "t5"):
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": item["problem"]},
                    {"role": "assistant", "content": item["text"]},
                    {"role": "user", "content": question}]
        else:
            cot = (PAYLOAD / "bbh_cot_prompts" /
                   f"{item['subset']}.txt").read_text()
            msgs = [{"role": "user", "content":
                     f"{cot}\n\nQ: {item['question']}\n"
                     f"A: Let's think step by step."},
                    {"role": "assistant", "content": item["text"]},
                    {"role": "user", "content": question}]
        return chat(tok, msgs)

    def decode(prompt, max_new=8):
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos)
        return (tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True),
                ids.shape[1], int(out.shape[1] - ids.shape[1]))

    Q_BIN = ("Are you sure your final answer is correct? "
             "Reply with exactly one word: sure or unsure.")
    Q_NUM = ("On a scale of 0 to 100, how confident are you that your final "
             "answer is correct? Reply with just the number.")

    rows = {"verb_binary_raw": [], "verb_binary": [],
            "verb_score_raw": [], "verb_score": [], "cost": 0}
    t0 = time.time()
    for i, item in enumerate(items):
        tb, np_, ng = decode(follow_up(item, Q_BIN)); rows["cost"] += np_ + ng
        m = re.search(r"\b(sure|unsure)\b", tb, re.I)
        # parse protocol (pre-registered): unsure=0, sure=1, unparseable=NaN
        rows["verb_binary_raw"].append(tb)
        rows["verb_binary"].append(
            float("nan") if not m else (0.0 if m.group(1).lower() == "unsure"
                                        else 1.0))
        ts, np_, ng = decode(follow_up(item, Q_NUM)); rows["cost"] += np_ + ng
        m = re.search(r"\b(\d{1,3})\b", ts)
        v = float(m.group(1)) if m else float("nan")
        rows["verb_score_raw"].append(ts)
        rows["verb_score"].append(v if 0 <= v <= 100 else float("nan"))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(items)} ({(time.time()-t0)/(i+1):.2f}s/p)",
                  flush=True)
    nb = float(np.mean(~np.isfinite(rows["verb_binary"])))
    ns = float(np.mean(~np.isfinite(rows["verb_score"])))
    print(f"parse-failure: binary {nb:.3f}, score {ns:.3f}", flush=True)
    dump({"task": "verbalized", "model": model_key, "cell": cell,
          "y": [it["y"] for it in items],
          "parse_failure_binary": nb, "parse_failure_score": ns,
          "cost_tokens": rows.pop("cost"), **rows},
         f"v7_verbalized_{model_key}_{cell}.json")


# ---------------------------------------------------------------------------
# Task: K=8 T=0.7 sampling on BBH (1b consistency arm + Phase-2 pseudo-labels)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def task_k8bbh(model_key: str = "qwen1.5b", limit=None, K=8, temp=0.7,
               seed=9999, bs=16):
    torch.manual_seed(seed)
    model, tok = load_model(model_key)
    tok.padding_side = "left"
    items = load_payload(model_key, "bbh")[:limit]
    eos = eos_ids(tok)
    out_rows = []
    t0 = time.time()
    for s in range(0, len(items), bs):
        batch = items[s:s + bs]
        prompts = [gen_prompt(tok, it, "bbh") for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True).to("cuda")
        plen = enc.input_ids.shape[1]
        gen = model.generate(**enc, max_new_tokens=1024, do_sample=True,
                             temperature=temp, top_p=1.0,
                             num_return_sequences=K,
                             pad_token_id=tok.pad_token_id, eos_token_id=eos)
        new = gen[:, plen:].reshape(len(batch), K, -1)
        for b, it in enumerate(batch):
            texts = [tok.decode(new[b, k], skip_special_tokens=True)
                     for k in range(K)]
            answers = [bbh_extract_answer(t) for t in texts]
            corr = [a != "" and a == it["target"] for a in answers]
            out_rows.append({"idx": it["idx"], "subset": it["subset"],
                             "target": it["target"], "y_greedy": it["y"],
                             "texts": texts, "answers": answers,
                             "correct": corr,
                             "n_gen": [int((new[b, k] != tok.pad_token_id)
                                           .sum()) for k in range(K)]})
        done = min(s + bs, len(items))
        el = time.time() - t0
        print(f"  {done}/{len(items)} ({el/done:.1f}s/p, "
              f"ETA {(len(items)-done)*el/done/60:.0f} min)", flush=True)
    total = int(sum(sum(r["n_gen"]) for r in out_rows))
    dump({"task": "k8bbh", "model": model_key, "K": K, "temperature": temp,
          "seed": seed, "cost_tokens_generated": total, "rows": out_rows},
         f"v7_k8bbh_{model_key}.json.gz")


# ---------------------------------------------------------------------------
# Task: PRM (optional) — Qwen2.5-Math-PRM-7B on cached 1.5B MATH generations
# ---------------------------------------------------------------------------

@torch.inference_mode()
def task_prm(limit=None):
    from transformers import AutoModel, AutoTokenizer
    name = MODELS["prm7b"]
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModel.from_pretrained(name, dtype=torch.bfloat16,
                                      attn_implementation="sdpa",
                                      device_map="cuda",
                                      trust_remote_code=True).eval()
    print(f"loaded {name}", flush=True)
    items = load_payload("qwen1.5b", "math")[:limit]
    sep = "<extra_0>"
    sep_id = tok.encode(sep)[0]
    rows = {"prm_min": [], "prm_mean": [], "prm_last": [], "n_steps": []}
    cost = 0
    t0 = time.time()
    for i, item in enumerate(items):
        steps = [s for s in item["text"].split("\n\n") if s.strip()]
        if not steps:
            steps = [item["text"]]
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": item["problem"]},
                {"role": "assistant", "content": sep.join(steps) + sep}]
        conv = tok.apply_chat_template(msgs, tokenize=False)
        ids = tok(conv, return_tensors="pt").input_ids.cuda()
        cost += ids.shape[1]
        logits = model(input_ids=ids).logits          # (1, L, 2)
        probs = torch.softmax(logits.float(), dim=-1)[0]
        mask = (ids[0] == sep_id)
        step_p = probs[mask][:, 1].cpu().numpy()      # P(step correct)
        if len(step_p) == 0:
            step_p = np.array([float("nan")])
        rows["prm_min"].append(float(np.min(step_p)))
        rows["prm_mean"].append(float(np.mean(step_p)))
        rows["prm_last"].append(float(step_p[-1]))
        rows["n_steps"].append(int(len(step_p)))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(items)} ({(time.time()-t0)/(i+1):.2f}s/p)",
                  flush=True)
    dump({"task": "prm", "model": "prm7b", "cell": "math",
          "y": [it["y"] for it in items], "cost_tokens_scored": cost,
          "note": "honest cost comparator is 7B escalation (cascade)",
          **rows}, "v7_prm_qwen1.5b_math.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["rescore", "ptrue", "verbalized", "k8bbh", "prm"])
    ap.add_argument("--model", default="qwen1.5b")
    ap.add_argument("--cell", default="math", choices=["math", "bbh", "t5"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.task == "rescore":
        task_rescore(args.model, args.cell, args.limit)
    elif args.task == "ptrue":
        task_ptrue(args.model, args.cell, args.limit)
    elif args.task == "verbalized":
        task_verbalized(args.model, args.cell, args.limit)
    elif args.task == "k8bbh":
        task_k8bbh(args.model, args.limit)
    elif args.task == "prm":
        task_prm(args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
