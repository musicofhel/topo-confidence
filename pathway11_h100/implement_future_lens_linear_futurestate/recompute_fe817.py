"""P11-FE817 — Future Lens linear future-state probe as the competing null for F-2.

Future Lens (Pal et al.) fits a linear map W: h_T (an early/source hidden state)
-> h_{T+k} (a later "future" hidden state), then decodes the predicted future
state through the unembedding to read off look-ahead tokens. This script tests
whether such a *purely linear look-ahead probe* on the Qwen-2.5-1.5B L19 prefill
state can match the prefill-DoM AUROC of 0.7731 (F-2).

Pipeline (all CPU, post-extraction):
  1. Fit W_k (ridge least squares) on a Pile slice of (source -> future) hidden
     pairs at offsets k=1..K. Fitting is done per-k, source = prefill-position
     state, target = state k tokens ahead.
  2. Apply W_k to the 500 MATH-500 prefill L19 states (cached Stage 2 NPZ) to
     get predicted future states.
  3. Per-problem "decoded confidence": if an unembedding + true future-token
     cache is present, use max-softmax of the decoded token and report decoded-
     token accuracy; otherwise fall back to cosine(pred_k, true_future_k), the
     linear look-ahead fidelity.
  4. AUROC of decoded confidence vs the K=1 outcome, compared head-to-head with
     the prefill-DoM AUROC (F-2 = 0.7731).

Interpretation: if the future-state probe alone reaches ~0.77 AUROC, the prefill
DoM signal is a degenerate linear look-ahead (Refutation 1). If DoM beats it
substantially, F-2 survives its strongest available null.

The Pile-slice and future-state hidden states require a GPU/model extraction
step that is run elsewhere; this CPU script consumes the resulting NPZ. If that
cache is absent it prints MISSING_REGEN_INPUT and returns 2.
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
# GPU/model extraction product: Pile (source->future) pairs + MATH-500 future states.
FUTURE_NPZ = ROOT / "pathway11_h100/future_lens/future_states.npz"
OUT_JSON = ROOT / "pathway11_h100/future_lens/results.json"

F2_DOM_AUROC = 0.7731  # 1024-tok canonical, prefill L19 DoM OOF 5-fold
RIDGE_REL = 1e-2       # ridge alpha relative to per-dim trace
SEED = 9999


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def fit_ridge_map(src: np.ndarray, tgt: np.ndarray, rel: float) -> np.ndarray:
    """Ridge least squares W (d_src x d_tgt) s.t. src @ W ≈ tgt."""
    n, d = src.shape
    gram = src.T @ src
    trace = float(np.trace(gram))
    alpha = rel * trace / max(d, 1)
    A = gram + alpha * np.eye(d, dtype=src.dtype)
    B = src.T @ tgt
    return np.linalg.solve(A, B)


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not FUTURE_NPZ.exists():
        print("MISSING_REGEN_INPUT", FUTURE_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    prefill_cache = cache["prefill"].astype(np.float64)
    assert prefill_cache.shape == (500, 1536) and correct.shape == (500,)

    dom_auroc_local = float("nan")
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        dom_auroc_local = auroc(dom_score, correct)

    fut = np.load(FUTURE_NPZ)
    # Required: Pile source/target pairs and MATH-500 future targets.
    required = ["pile_src", "pile_tgt", "math_prefill", "math_tgt"]
    missing = [k for k in required if k not in fut.files]
    if missing:
        print("MISSING_REGEN_INPUT", FUTURE_NPZ, "keys:", missing, file=sys.stderr)
        return 2

    pile_src = fut["pile_src"].astype(np.float64)        # (P, 1536)
    pile_tgt = fut["pile_tgt"].astype(np.float64)        # (P, K, 1536)
    math_prefill = fut["math_prefill"].astype(np.float64)  # (500, 1536)
    math_tgt = fut["math_tgt"].astype(np.float64)        # (500, K, 1536)

    if pile_tgt.ndim != 3 or math_tgt.ndim != 3:
        print("MISSING_REGEN_INPUT", FUTURE_NPZ, "bad target rank", file=sys.stderr)
        return 2

    K = int(pile_tgt.shape[1])
    d = int(pile_src.shape[1])
    if math_tgt.shape[1] != K or math_prefill.shape != (500, d):
        print("MISSING_REGEN_INPUT", FUTURE_NPZ, "shape mismatch", file=sys.stderr)
        return 2

    # Sanity: cached MATH prefill in the future NPZ should match the Stage 2 cache.
    prefill_drift = float(np.linalg.norm(math_prefill - prefill_cache) /
                          (np.linalg.norm(prefill_cache) + 1e-12))

    # Optional true-token decode path.
    have_decode = ("unembed" in fut.files) and ("math_tgt_tokens" in fut.files)
    unembed = fut["unembed"].astype(np.float64) if have_decode else None        # (V, 1536)
    tgt_tokens = fut["math_tgt_tokens"].astype(np.int64) if have_decode else None  # (500, K)

    per_k = []
    # Confidence accumulators across k for the per-problem score.
    conf_cos = np.zeros(500, dtype=np.float64)
    conf_softmax = np.zeros(500, dtype=np.float64)

    for k in range(K):
        W = fit_ridge_map(pile_src, pile_tgt[:, k, :], RIDGE_REL)  # (d, d)
        pred = math_prefill @ W                                    # (500, d)
        true = math_tgt[:, k, :]                                   # (500, d)

        # Linear look-ahead fidelity: cosine between predicted and true future state.
        pn = np.linalg.norm(pred, axis=1) + 1e-12
        tn = np.linalg.norm(true, axis=1) + 1e-12
        cos_k = np.einsum("ij,ij->i", pred, true) / (pn * tn)
        conf_cos += cos_k

        entry = {"k": k + 1, "cos_fidelity_mean": float(cos_k.mean())}

        if have_decode:
            logits = pred @ unembed.T                              # (500, V)
            probs = softmax_rows(logits)
            argmax_tok = probs.argmax(axis=1)
            maxprob = probs.max(axis=1)
            conf_softmax += maxprob
            acc_k = float((argmax_tok == tgt_tokens[:, k]).mean())
            entry["decoded_token_acc"] = acc_k
            entry["decode_confidence_mean"] = float(maxprob.mean())
            entry["auroc_softmax_conf_vs_k1"] = auroc(maxprob, correct)

        entry["auroc_cos_conf_vs_k1"] = auroc(cos_k, correct)
        per_k.append(entry)

    conf_cos /= K
    conf_softmax /= K

    auroc_cos_pooled = auroc(conf_cos, correct)
    auroc_softmax_pooled = auroc(conf_softmax, correct) if have_decode else float("nan")

    # Best future-lens AUROC across the two confidence channels and pooled/per-k.
    candidates = [auroc_cos_pooled]
    if have_decode:
        candidates.append(auroc_softmax_pooled)
    for e in per_k:
        candidates.append(e["auroc_cos_conf_vs_k1"])
        if have_decode:
            candidates.append(e["auroc_softmax_conf_vs_k1"])
    candidates = [c for c in candidates if not np.isnan(c)]
    best_future_lens = float(max(candidates)) if candidates else float("nan")

    margin = F2_DOM_AUROC - best_future_lens if not np.isnan(best_future_lens) else float("nan")
    # Refutation 1 fires if the linear probe matches DoM within 0.01 AUROC.
    refutes_dom = bool(not np.isnan(margin) and margin <= 0.01)

    out = {
        "experiment": "P11-FE817",
        "description": "Future Lens linear future-state probe vs F-2 prefill DoM",
        "n_problems": 500,
        "k_max": K,
        "ridge_rel_alpha": RIDGE_REL,
        "have_decode_path": bool(have_decode),
        "prefill_drift_vs_stage2": prefill_drift,
        "f2_dom_auroc_reference": F2_DOM_AUROC,
        "dom_auroc_local_recompute": dom_auroc_local,
        "future_lens_auroc_cos_pooled": auroc_cos_pooled,
        "future_lens_auroc_softmax_pooled": auroc_softmax_pooled,
        "future_lens_best_auroc": best_future_lens,
        "dom_minus_future_lens_margin": margin,
        "refutes_f2_degenerate_lookahead": refutes_dom,
        "per_k": per_k,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())