"""FE477 — WAS Eq. 7 refusal direction vs. the L19 DoM correctness axis.

Computes the Welch/WAS refusal direction d_refuse = mean(W_U[refuse]) -
mean(W_U[answer]) for Qwen-2.5-1.5B directly from the unembedding matrix W_U
(rows live in the 1536-d residual-stream basis, so d_refuse is comparable to the
prefill/final L19 directions). Then:

  1. cos(d_refuse, prefill_L19_DoM) and cos(d_refuse, final_L19_DoM).
  2. Projects Stage-2 prefill activations onto d_refuse, fits a 1-feature
     logistic regression (5-fold OOF), and scores MATH-500 correctness AUROC.

Decision rule (from the brief):
  * d_refuse OOF AUROC >= 0.60  -> refusal axis carries non-trivial, label-free
    correctness signal (a DoM proxy for H-12).
  * |cos(d_refuse, prefill_DoM)| > 0.50 -> the project's L19 DoM is partially the
    refusal direction (task geometry == universal refusal geometry).

W_U is not in the activation caches; it must be exported to a local NPZ
(W_U + refuse_ids + answer_ids, derived from the WAS token lists + tokenizer)
on a GPU box and copied in. Missing -> MISSING_REGEN_INPUT / exit 2.
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
FINAL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
# W_U export: keys "W_U" (vocab, 1536), "refuse_ids" (int), "answer_ids" (int).
WU_NPZ = ROOT / "pathway11_h100/refusal_direction/wu_qwen15b.npz"
OUT_JSON = ROOT / "pathway11_h100/refusal_direction/results.json"

SEED = 9999
N_FOLDS = 5

# Provenance only (the actual ids come from the cache, since they require the
# tokenizer). Mirrors the WAS / ToxicChat refusal-recipe token sets.
REFUSE_TOKENS = ["I", "Sorry", "sorry", "cannot", "can't", "unable", "As",
                 "unfortunately", "Unfortunately", "apolog", "refuse", "won't"]
ANSWER_TOKENS = ["Sure", "Here", "here", "Certainly", "Of", "Yes", "The",
                 "To", "Step", "First", "We", "Let"]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2
    if not WU_NPZ.exists():
        print("MISSING_REGEN_INPUT", WU_NPZ, file=sys.stderr); return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr); return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert X.shape == (500, 1536) and y.shape == (500,)

    wu = np.load(WU_NPZ)
    for key in ("W_U", "refuse_ids", "answer_ids"):
        if key not in wu:
            print("MISSING_REGEN_INPUT", f"{WU_NPZ}:{key}", file=sys.stderr); return 2
    W_U = wu["W_U"].astype(np.float64)
    refuse_ids = wu["refuse_ids"].astype(np.int64)
    answer_ids = wu["answer_ids"].astype(np.int64)
    if W_U.ndim != 2 or W_U.shape[1] != X.shape[1]:
        print("MISSING_REGEN_INPUT", f"{WU_NPZ}:W_U_shape={W_U.shape}", file=sys.stderr); return 2
    if len(refuse_ids) == 0 or len(answer_ids) == 0:
        print("MISSING_REGEN_INPUT", f"{WU_NPZ}:empty_token_ids", file=sys.stderr); return 2

    # Eq. 7 refusal direction in residual-stream basis.
    d_refuse = unit(W_U[refuse_ids].mean(axis=0) - W_U[answer_ids].mean(axis=0))

    # Supervised DoM directions (full-data mean difference).
    prefill_dom = unit(X[y].mean(axis=0) - X[~y].mean(axis=0))

    final_dom = None
    if FINAL_CACHE.exists():
        fc = np.load(FINAL_CACHE)
        fkey = "final" if "final" in fc else ("prefill" if "prefill" in fc else None)
        if fkey is not None:
            Xf = fc[fkey].astype(np.float64)
            if Xf.shape == X.shape:
                final_dom = unit(Xf[y].mean(axis=0) - Xf[~y].mean(axis=0))

    cos_prefill = cos(d_refuse, prefill_dom)
    cos_final = cos(d_refuse, final_dom) if final_dom is not None else float("nan")

    # Project prefill activations onto d_refuse; raw directional AUROC.
    proj = X @ d_refuse
    auroc_raw = auroc(proj, y)

    # 1-feature logistic, 5-fold OOF (sign-invariant readout).
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=np.float64)
    feat = proj.reshape(-1, 1)
    for tr, te in skf.split(feat, y):
        mu = feat[tr].mean(); sd = feat[tr].std() + 1e-12
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit((feat[tr] - mu) / sd, y[tr])
        oof[te] = clf.predict_proba((feat[te] - mu) / sd)[:, 1]
    auroc_oof = auroc(oof, y)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)

    out = {
        "experiment": "P11-FE477",
        "n": int(len(y)),
        "n_refuse_tokens": int(len(refuse_ids)),
        "n_answer_tokens": int(len(answer_ids)),
        "refuse_tokens_ref": REFUSE_TOKENS,
        "answer_tokens_ref": ANSWER_TOKENS,
        "cos_d_refuse_prefill_dom": cos_prefill,
        "cos_d_refuse_final_dom": cos_final,
        "final_dom_available": final_dom is not None,
        "auroc_d_refuse_raw": float(auroc_raw),
        "auroc_d_refuse_oof_logistic": float(auroc_oof),
        "auroc_prefill_dom_score_ref": float(auroc(dom_score, y)),
        "decision_refusal_carries_signal": bool(
            not np.isnan(auroc_oof) and auroc_oof >= 0.60),
        "decision_dom_is_refusal": bool(
            not np.isnan(cos_prefill) and abs(cos_prefill) > 0.50),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())