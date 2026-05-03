"""GPU bundle: P11-FE110 (ActAdd vs supervised DoM) + P10-FE23 (softmax-conf
AUROC) + P11-FE455 (ConCISE 'So, I'm' detector).

Single Qwen-2.5-1.5B-Instruct fp16 load on RTX 2060 Super. Re-tokenizes each
of the 500 MATH-500 problems with the chat template + cached generated
solution from `pathway8_layerwise/data/math500/problem_*.npz` (which is the
same generation that produced the m15b_prefill cache used by F-2).

PANL = position right before `\\boxed{` in the generated text. We forward
[prompt + reasoning_up_to_PANL] for FE23, append ' So, I'm' for FE455.
FE110 runs 5 contrast prompt pairs through the same model and compares the
resulting L19 prefill direction to the supervised DoM r̂ from m15b_prefill.

Outputs:
  pathway11_h100/gpu_bundle/results.json
  pathway11_h100/gpu_bundle/per_problem.npz
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path("/home/musicofhel/topo-confidence")
sys.path.insert(0, str(ROOT))

from pathway8_layerwise.extract_math500 import load_math500  # noqa: E402

OUT_DIR = ROOT / "pathway11_h100/gpu_bundle"
OUT_JSON = OUT_DIR / "results.json"
OUT_NPZ = OUT_DIR / "per_problem.npz"
NPZ_DIR = ROOT / "pathway8_layerwise/data/math500"
M15B_PREFILL = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
N_PROBLEMS = 500
LAYER = 19           # 2/3-depth Qwen-2.5-1.5B (28 transformer layers)
N_FOLDS = 5
SEED = 42

CONTRAST_PAIRS = [
    ("correct math step", "incorrect math step"),
    ("solve carefully", "guess"),
    ("complete solution", "wrong solution"),
    ("right answer", "wrong answer"),
    ("certain", "uncertain"),
]

# Tokens to read for ConCISE c_hat. Use leading-space variants since
# 'So, I'm' is followed by a single space + word in natural text.
CONCISE_WORDS = [" confident", " sure", " pretty"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def oof_dom_auroc(X: np.ndarray, y: np.ndarray, k: int, seed: int) -> float:
    folds = stratified_kfold(y, k, seed)
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train = np.ones(n, dtype=bool); train[test_idx] = False
        d = X[train & y].mean(axis=0) - X[train & ~y].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d
    return auroc(scores, y)


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tok.pad_token = "<|endoftext|>"
    tok.pad_token_id = 151643
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def chat_prompt(tokenizer, problem_text: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.inference_mode()
def forward_last_token(model, tok, prompt: str, capture_layer: int | None = None
                        ) -> tuple[np.ndarray, np.ndarray | None]:
    """Run model on prompt; return (last-token logits as np.float32 1-d, optional capture-layer hidden state at last token)."""
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    if capture_layer is not None:
        out = model(**inp, output_hidden_states=True)
        last_hs = out.hidden_states[capture_layer][0, -1, :].float().cpu().numpy()
    else:
        out = model(**inp)
        last_hs = None
    logits = out.logits[0, -1, :].float().cpu().numpy()
    return logits, last_hs


@torch.inference_mode()
def forward_l19_prefill_only(model, tok, prompt: str) -> np.ndarray:
    """Forward only the prompt (no chat tail); return L19 hidden state at last prompt token."""
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[LAYER][0, -1, :].float().cpu().numpy()
    return h


def panl_prefix(generated_text: str) -> str:
    """Return generated text up to (but not including) the last `\\boxed{`."""
    idx = generated_text.rfind("\\boxed{")
    if idx < 0:
        return generated_text
    return generated_text[:idx]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} fp16 on cuda:0 ...")
    t_load = time.time()
    model, tok = load_model_and_tokenizer()
    print(f"  loaded in {time.time()-t_load:.1f}s; param sum {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # ---------------- FE110 contrast vectors ----------------
    print("\n[FE110] Computing ActAdd contrast vectors at L19 ...")
    contrast_vecs = []
    for pos_text, neg_text in CONTRAST_PAIRS:
        # Use raw text (no chat template) — ActAdd convention
        h_pos = forward_l19_prefill_only(model, tok, pos_text)
        h_neg = forward_l19_prefill_only(model, tok, neg_text)
        v = h_pos - h_neg
        contrast_vecs.append(v)
        print(f"  ({pos_text!r}, {neg_text!r}) ‖v‖={np.linalg.norm(v):.3f}")

    # ---------------- Load cached supervised DoM source ----------------
    cache = np.load(M15B_PREFILL, allow_pickle=True)
    prefill_cache = cache["prefill"].astype(np.float32)   # (500, 1536)
    correct_cache = cache["correct"].astype(bool)
    seq_len_cache = cache["seq_len"].astype(np.int64)
    print(f"\n[FE110] m15b_prefill cache: shape {prefill_cache.shape}, correct {correct_cache.sum()}/500")

    # Supervised DoM (full-sample, for cosine) and OOF AUROC (for sanity)
    supervised_dom = prefill_cache[correct_cache].mean(0) - prefill_cache[~correct_cache].mean(0)
    auroc_sup = oof_dom_auroc(prefill_cache, correct_cache, N_FOLDS, SEED)
    print(f"  supervised DoM AUROC (5-fold OOF) on cache = {auroc_sup:.4f}  (F-2 ≈ 0.7731)")

    fe110_per_pair = []
    for (pos_text, neg_text), v in zip(CONTRAST_PAIRS, contrast_vecs):
        c = cos(v, supervised_dom)
        proj_scores = prefill_cache @ v
        a = auroc(proj_scores, correct_cache)
        fe110_per_pair.append({
            "pair": [pos_text, neg_text],
            "cos_with_supervised_DoM": c,
            "projection_AUROC": a,
            "v_norm": float(np.linalg.norm(v)),
        })
        print(f"  cos={c:+.4f}  proj_AUROC={a:.4f}  ({pos_text!r} vs {neg_text!r})")

    fe110_max_cos = max(p["cos_with_supervised_DoM"] for p in fe110_per_pair)
    fe110_max_auroc = max(p["projection_AUROC"] for p in fe110_per_pair)

    # ---------------- FE23 + FE455: per-problem forward sweep ----------------
    print("\n[FE23+FE455] Loading MATH-500 problems and cached generations ...")
    ms = load_math500()
    cached = []
    for i in range(N_PROBLEMS):
        d = np.load(NPZ_DIR / f"problem_{i:03d}.npz", allow_pickle=True)
        cached.append({
            "text": str(d["text"]),
            "correct": bool(d["correct"]),
        })

    # Pre-resolve token IDs for ConCISE words
    concise_token_ids: list[int] = []
    for w in CONCISE_WORDS:
        ids = tok(w, add_special_tokens=False)["input_ids"]
        concise_token_ids.append(ids[0])
        if len(ids) > 1:
            print(f"  warn: {w!r} tokenizes to {ids} — using first id {ids[0]}")
    print(f"  ConCISE token ids: {dict(zip(CONCISE_WORDS, concise_token_ids))}")

    # Storage
    fe23_pred_prob = np.zeros(N_PROBLEMS, dtype=np.float64)   # max softmax = predicted-token prob
    fe23_top_token_id = np.zeros(N_PROBLEMS, dtype=np.int64)
    fe455_c_hat = np.zeros(N_PROBLEMS, dtype=np.float64)
    fe455_p_confident = np.zeros(N_PROBLEMS, dtype=np.float64)
    fe455_p_sure = np.zeros(N_PROBLEMS, dtype=np.float64)
    fe455_p_pretty = np.zeros(N_PROBLEMS, dtype=np.float64)

    correct = np.array([c["correct"] for c in cached], dtype=bool)
    n_no_panl = 0

    t0 = time.time()
    for i in range(N_PROBLEMS):
        question = ms[i]["problem"]
        gen_text = cached[i]["text"]
        prompt = chat_prompt(tok, question)
        panl_gen = panl_prefix(gen_text)
        if panl_gen == gen_text:
            n_no_panl += 1
        text_to_panl = prompt + panl_gen

        # FE23: forward up to PANL, read top-token softmax
        logits, _ = forward_last_token(model, tok, text_to_panl)
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        top_id = int(probs.argmax())
        fe23_pred_prob[i] = float(probs[top_id])
        fe23_top_token_id[i] = top_id

        # FE455: forward [PANL + " So, I'm"], read confident/sure/pretty probs
        text_so_im = text_to_panl + " So, I'm"
        logits2, _ = forward_last_token(model, tok, text_so_im)
        probs2 = np.exp(logits2 - logits2.max())
        probs2 = probs2 / probs2.sum()
        p_conf = float(probs2[concise_token_ids[0]])
        p_sure = float(probs2[concise_token_ids[1]])
        p_pretty = float(probs2[concise_token_ids[2]])
        fe455_p_confident[i] = p_conf
        fe455_p_sure[i] = p_sure
        fe455_p_pretty[i] = p_pretty

        # ConCISE c_hat = P(confident) + P(sure) + P(pretty) * (P(confident|pretty) + P(sure|pretty))
        # We approximate the conditional pair as P(confident|pretty)≈P(confident), etc, since computing
        # actual conditional requires another forward; the head-to-head AUROC captures the same signal.
        fe455_c_hat[i] = p_conf + p_sure + p_pretty * (p_conf + p_sure)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (N_PROBLEMS - i - 1)
            print(f"  {i+1}/{N_PROBLEMS}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)
    elapsed_total = time.time() - t0
    print(f"  forward sweep done in {elapsed_total:.0f}s, {n_no_panl} problems without `\\boxed{{`")

    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------- AUROCs ----------------
    fe23_signal = 1.0 - fe23_pred_prob   # high uncertainty = high signal of incorrectness
    auroc_fe23_uncert = auroc(fe23_signal, ~correct)        # AUROC for predicting incorrect
    auroc_fe23_conf = auroc(fe23_pred_prob, correct)        # AUROC for predicting correct
    print(f"\n[FE23] softmax-conf AUROC (predict correct via top-prob): {auroc_fe23_conf:.4f}")
    print(f"        softmax-uncert AUROC (predict incorrect via 1-top-prob): {auroc_fe23_uncert:.4f}")

    auroc_fe455 = auroc(fe455_c_hat, correct)
    auroc_fe455_just_conf = auroc(fe455_p_confident, correct)
    auroc_fe455_just_sure = auroc(fe455_p_sure, correct)
    print(f"[FE455] ConCISE c_hat AUROC: {auroc_fe455:.4f}")
    print(f"        P(' confident') alone:  {auroc_fe455_just_conf:.4f}")
    print(f"        P(' sure') alone:       {auroc_fe455_just_sure:.4f}")

    # Joint with cached prefill DoM (sanity) — does adding ConCISE help F-2?
    cache_correct_match = np.array_equal(correct, correct_cache)
    print(f"  pathway8 correctness vs m15b cache correctness match: {cache_correct_match}")

    F2_PREFILL_DOM_AUROC = 0.7731
    F2_FINAL_DOM_AUROC = 0.7186

    # ---------------- Verdicts ----------------
    fe110_verdict = (
        f"Best contrast pair achieves cos={fe110_max_cos:+.3f} with supervised DoM "
        f"and AUROC {fe110_max_auroc:.3f} on cached prefill scores. F-2 supervised "
        f"DoM is "
        f"{'recoverable from contrast pairs' if fe110_max_cos > 0.5 and fe110_max_auroc >= F2_PREFILL_DOM_AUROC - 0.02 else 'NOT recoverable from these contrast pairs (supervision adds signal beyond ActAdd)'}."
    )
    fe23_verdict = (
        f"Softmax-confidence AUROC {auroc_fe23_conf:.3f} vs prefill DoM {F2_PREFILL_DOM_AUROC} "
        f"({'≥' if auroc_fe23_conf >= F2_PREFILL_DOM_AUROC - 0.02 else '<'} F-2). "
        f"{'F-2 weakened — softmax matches.' if auroc_fe23_conf >= F2_PREFILL_DOM_AUROC - 0.02 else 'F-2 holds — softmax does not subsume DoM.'}"
    )
    fe455_verdict = (
        f"ConCISE c_hat AUROC {auroc_fe455:.3f} vs prefill DoM {F2_PREFILL_DOM_AUROC} "
        f"({'≥' if auroc_fe455 >= F2_PREFILL_DOM_AUROC - 0.02 else '<'} F-2). "
        f"{'F-2 weakened — ConCISE matches.' if auroc_fe455 >= F2_PREFILL_DOM_AUROC - 0.02 else 'F-2 holds — ConCISE does not subsume DoM.'}"
    )
    print(f"\n[FE110] {fe110_verdict}")
    print(f"[FE23]  {fe23_verdict}")
    print(f"[FE455] {fe455_verdict}")

    out = {
        "experiment": "P11-FE110+P10-FE23+P11-FE455",
        "depends_on": ["F-2", "F-3", "F-9"],
        "model": MODEL_NAME,
        "n_problems": N_PROBLEMS,
        "layer": LAYER,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "elapsed_seconds": float(elapsed_total),
        "n_problems_without_boxed": int(n_no_panl),
        "f2_prefill_dom_auroc": F2_PREFILL_DOM_AUROC,
        "f2_final_dom_auroc": F2_FINAL_DOM_AUROC,
        "fe110": {
            "supervised_dom_auroc_oof": auroc_sup,
            "per_pair": fe110_per_pair,
            "max_cos_with_dom": fe110_max_cos,
            "max_projection_auroc": fe110_max_auroc,
            "verdict": fe110_verdict,
        },
        "fe23": {
            "auroc_softmax_confidence": auroc_fe23_conf,
            "auroc_softmax_uncertainty": auroc_fe23_uncert,
            "verdict": fe23_verdict,
        },
        "fe455": {
            "auroc_c_hat": auroc_fe455,
            "auroc_p_confident": auroc_fe455_just_conf,
            "auroc_p_sure": auroc_fe455_just_sure,
            "concise_token_ids": dict(zip(CONCISE_WORDS, concise_token_ids)),
            "verdict": fe455_verdict,
        },
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    np.savez_compressed(
        OUT_NPZ,
        correct=correct,
        fe23_pred_prob=fe23_pred_prob,
        fe23_top_token_id=fe23_top_token_id,
        fe455_c_hat=fe455_c_hat,
        fe455_p_confident=fe455_p_confident,
        fe455_p_sure=fe455_p_sure,
        fe455_p_pretty=fe455_p_pretty,
        fe110_contrast_vecs=np.stack(contrast_vecs),
        supervised_dom=supervised_dom,
    )
    print(f"\nSaved {OUT_JSON.relative_to(ROOT)} and {OUT_NPZ.relative_to(ROOT)}")

    # Tier-1 regen-readback hooks for validate_claims.py (key=value parser).
    print(f"fe110.max_cos_with_dom={fe110_max_cos:.10f}")
    print(f"fe110.max_projection_auroc={fe110_max_auroc:.10f}")
    print(f"fe110.supervised_dom_auroc_oof={auroc_sup:.10f}")
    print(f"fe23.auroc_softmax_confidence={auroc_fe23_conf:.10f}")
    print(f"fe455.auroc_c_hat={auroc_fe455:.10f}")
    print(f"fe455.auroc_p_confident={auroc_fe455_just_conf:.10f}")
    print(f"fe455.auroc_p_sure={auroc_fe455_just_sure:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
