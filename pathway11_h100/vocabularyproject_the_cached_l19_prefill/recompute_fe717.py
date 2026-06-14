"""P11-FE717 — Vocabulary projection of the L19 prefill DoM direction.

Projects the supervised DoM direction (mean prefill activation of correct minus
incorrect problems, L19) through Qwen-2.5-1.5B's unembedding head W_LM to obtain
per-token logits, ranks the top-10 tokens the direction encodes in each polarity,
and categorizes each as content / punctuation / article / number following Taylor
(2603.14923) §6.1. Tests whether our supervised correctness direction aligns with
the syntactic (punctuation/article) axes that emerge unsupervised in routed
transformers — if it does, F-2's signal is partly a syntactic-pruning artifact
rather than a reasoning-content readout.

Requires a cached, CPU-friendly copy of the LM head (no torch / transformers):
  pathway11_h100/vocab_projection/qwen15b_lm_head.npz
    W      : (vocab, 1536) float32 — unembedding weight (rows = tokens)
    tokens : (vocab,) object/str — decoded token strings (byte-BPE, 'Ġ'=space)
Regenerate offline with:
  from transformers import AutoModelForCausalLM, AutoTokenizer
  m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B')
  tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B')
  W = m.lm_head.weight.detach().cpu().numpy()  # (vocab, 1536)
  toks = [tok.convert_ids_to_tokens(i) for i in range(W.shape[0])]
  np.savez('qwen15b_lm_head.npz', W=W, tokens=np.array(toks, dtype=object))
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
LM_HEAD_NPZ = ROOT / "pathway11_h100/vocab_projection/qwen15b_lm_head.npz"
OUT_JSON = ROOT / "pathway11_h100/vocab_projection/results.json"

TOP_K = 10
ARTICLES = {"a", "an", "the"}


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def clean_token(raw: str) -> str:
    """Strip byte-BPE whitespace markers to expose the lexical content."""
    if raw is None:
        return ""
    s = str(raw)
    # GPT-2/Qwen byte-level BPE uses 'Ġ' for a leading space, 'Ċ' for newline.
    s = s.replace("\u0120", " ").replace("\u010a", "\n").replace("\u2581", " ")
    return s.strip()


def categorize(raw: str) -> str:
    """content / punctuation / article / number, per Taylor 2603.14923 §6.1."""
    s = clean_token(raw)
    if s == "":
        return "punctuation"  # pure whitespace / control
    has_alpha = any(ch.isalpha() for ch in s)
    has_digit = any(ch.isdigit() for ch in s)
    if not has_alpha and not has_digit:
        return "punctuation"
    if has_digit and not has_alpha:
        return "number"
    if s.lower() in ARTICLES:
        return "article"
    return "content"


def rank_tokens(logits: np.ndarray, tokens: np.ndarray, k: int, descending: bool):
    order = np.argsort(logits)
    if descending:
        order = order[::-1]
    top = order[:k]
    return [
        {
            "rank": int(r),
            "token_id": int(idx),
            "token": str(tokens[idx]),
            "token_clean": clean_token(tokens[idx]),
            "logit": float(logits[idx]),
            "category": categorize(tokens[idx]),
        }
        for r, idx in enumerate(top)
    ]


def category_counts(entries) -> dict:
    counts = {"content": 0, "punctuation": 0, "article": 0, "number": 0}
    for e in entries:
        counts[e["category"]] += 1
    return counts


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not LM_HEAD_NPZ.exists():
        print("MISSING_REGEN_INPUT", LM_HEAD_NPZ, file=sys.stderr)
        return 2

    blob = np.load(CACHE)
    X = blob["prefill"].astype(np.float64)
    y = blob["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    # Supervised DoM direction at L19 (correct mean - incorrect mean).
    dom = X[y].mean(axis=0) - X[~y].mean(axis=0)
    dom_norm = float(np.linalg.norm(dom))
    if dom_norm > 0:
        dom_unit = dom / dom_norm
    else:
        dom_unit = dom

    # Sanity: the DoM should still discriminate correctness on the cache.
    dom_auroc = auroc(X @ dom, y)

    head = np.load(LM_HEAD_NPZ, allow_pickle=True)
    W = head["W"].astype(np.float64)  # (vocab, 1536)
    tokens = head["tokens"]
    if W.shape[1] != X.shape[1]:
        if W.shape[0] == X.shape[1]:
            W = W.T  # tolerate (1536, vocab) orientation
        else:
            print("MISSING_REGEN_INPUT", "W_LM dim mismatch", W.shape, file=sys.stderr)
            return 2
    assert W.shape[0] == len(tokens), (W.shape, len(tokens))

    # Vocabulary projection: logit each token assigns to the DoM direction.
    logits = W @ dom_unit  # (vocab,)

    top_pos = rank_tokens(logits, tokens, TOP_K, descending=True)
    top_neg = rank_tokens(logits, tokens, TOP_K, descending=False)

    out = {
        "experiment": "P11-FE717",
        "trigger_paper": "2603.14923",
        "vocab_size": int(len(tokens)),
        "hidden_dim": int(W.shape[1]),
        "dom_norm": dom_norm,
        "dom_auroc_on_cache": float(dom_auroc),
        "top_k": TOP_K,
        "top_positive_tokens": top_pos,
        "top_negative_tokens": top_neg,
        "category_counts_positive": category_counts(top_pos),
        "category_counts_negative": category_counts(top_neg),
        "content_fraction_positive": category_counts(top_pos)["content"] / TOP_K,
        "content_fraction_negative": category_counts(top_neg)["content"] / TOP_K,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())