"""P11-FE1079 — Token-frequency confound test for F-2 (DoM AUROC 0.7731).

Motivation: TIDE reports that token-identity persistence in residual streams is
frequency-dependent, so the L19 prefill DoM direction might be partially a
token-frequency proxy rather than a correctness signal. This script:

  1. Computes a per-problem mean log-unigram-probability feature `f` from the
     MATH-500 prompt text (unigram model estimated from the prompt corpus
     itself; tokenization is a builtin alnum-splitter proxy because the true
     model tokenizer would require `transformers`, which is banned here).
  2. Correlates `f` with the DoM score, with correctness, and with seq_len.
  3. Residualizes the DoM score on `f` out-of-fold (FE447-style) and re-scores.
  4. Partials the token-frequency feature out of the L19 activations themselves
     (OOF per-dimension linear regression on `f`), refits DoM on the residualized
     activations, and reports whether AUROC degrades.

If DoM is a frequency proxy, the partialled-out AUROC should collapse toward the
correlation-implied ceiling; if it survives, the frequency confound is ruled out.
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
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/token_frequency_confound/results.json"

# Candidate locations for the MATH-500 prompt text, in priority order. The first
# that exists and yields 500 problems is used.
PROMPT_CANDIDATES = [
    ROOT / "data/math500.jsonl",
    ROOT / "data/math500.json",
    ROOT / "data/math500_prompts.json",
    ROOT / "data/MATH-500.jsonl",
    ROOT / "pathway11_h100/data/math500_prompts.json",
    ROOT / "pathway11_h100/data/math500.jsonl",
]

SEED = 9999
N_FOLDS = 5


def auroc(scores, labels):
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def tokenize(text: str) -> list[str]:
    """Lowercase alnum-run tokenizer using only builtins (no `re`)."""
    out = []
    for ch in str(text).lower():
        out.append(ch if ch.isalnum() else " ")
    return "".join(out).split()


def load_prompts() -> list[str] | None:
    """Return a list of 500 prompt strings, or None if no source is available."""
    for path in PROMPT_CANDIDATES:
        if not path.exists():
            continue
        try:
            if path.suffix == ".jsonl":
                rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
            else:
                rows = json.loads(path.read_text())
        except (ValueError, OSError):
            continue
        if isinstance(rows, dict):
            rows = rows.get("problems") or rows.get("data") or list(rows.values())
        prompts = []
        for r in rows:
            if isinstance(r, str):
                prompts.append(r)
            elif isinstance(r, dict):
                for key in ("problem", "prompt", "question", "text"):
                    if key in r and isinstance(r[key], str):
                        prompts.append(r[key])
                        break
        if len(prompts) == 500:
            return prompts
    return None


def mean_log_unigram(prompts: list[str]) -> np.ndarray:
    """Per-problem mean log P(token) under a unigram model fit on the corpus."""
    tokenized = [tokenize(p) for p in prompts]
    counts: dict[str, int] = {}
    total = 0
    for toks in tokenized:
        for t in toks:
            counts[t] = counts.get(t, 0) + 1
            total += 1
    vocab = len(counts)
    # Laplace-smoothed unigram log-probabilities.
    logp = {t: float(np.log((c + 1.0) / (total + vocab))) for t, c in counts.items()}
    unk = float(np.log(1.0 / (total + vocab)))
    feat = np.empty(len(prompts), dtype=np.float64)
    for i, toks in enumerate(tokenized):
        if toks:
            feat[i] = float(np.mean([logp.get(t, unk) for t in toks]))
        else:
            feat[i] = unk
    return feat


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    prompts = load_prompts()
    if prompts is None:
        print("MISSING_REGEN_INPUT", "MATH-500 prompt text (tried: %s)"
              % ", ".join(str(p) for p in PROMPT_CANDIDATES), file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    X = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    seq_len = cache["seq_len"].astype(np.float64)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert X.shape == (500, 1536) and y.shape == (500,)

    f = mean_log_unigram(prompts)
    f_std = (f - f.mean()) / (f.std() + 1e-12)

    # --- correlations of the frequency feature ---
    def pearson(a, b):
        a = a - a.mean(); b = b - b.mean()
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / denom) if denom > 0 else float("nan")

    corr = {
        "pearson_freq_dom": pearson(f, dom_score),
        "spearman_freq_dom": float(spearmanr(f, dom_score).correlation),
        "pointbiserial_freq_correct": pearson(f, y.astype(np.float64)),
        "pearson_freq_seqlen": pearson(f, seq_len),
        "auroc_freq_alone": float(auroc(f, y)),
    }

    n = len(y)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # --- baseline OOF DoM (refit on raw train activations) ---
    raw_oof = np.zeros(n, dtype=np.float64)
    # --- DoM with frequency partialled out of activations ---
    resid_act_oof = np.zeros(n, dtype=np.float64)
    # --- DoM *score* residualized on frequency (FE447-style) ---
    resid_score_oof = np.zeros(n, dtype=np.float64)

    for tr, te in skf.split(X, y):
        ytr = y[tr]

        d_raw = X[tr][ytr].mean(0) - X[tr][~ytr].mean(0)
        raw_oof[te] = X[te] @ d_raw

        # Per-dimension OOF regression of activations on the frequency feature.
        A_tr = np.column_stack([f_std[tr], np.ones(len(tr))])
        B, _, _, _ = np.linalg.lstsq(A_tr, X[tr], rcond=None)  # (2, 1536)
        Xtr_res = X[tr] - A_tr @ B
        A_te = np.column_stack([f_std[te], np.ones(len(te))])
        Xte_res = X[te] - A_te @ B
        d_res = Xtr_res[ytr].mean(0) - Xtr_res[~ytr].mean(0)
        resid_act_oof[te] = Xte_res @ d_res

        # Residualize the DoM score itself on the frequency feature.
        coeffs, _, _, _ = np.linalg.lstsq(A_tr, dom_score[tr], rcond=None)
        resid_score_oof[te] = dom_score[te] - (A_te @ coeffs)

    auroc_raw = float(auroc(raw_oof, y))
    auroc_resid_act = float(auroc(resid_act_oof, y))
    auroc_resid_score = float(auroc(resid_score_oof, y))
    auroc_dom_published = float(auroc(dom_score, y))

    out = {
        "experiment": "P11-FE1079",
        "description": "Token-frequency confound test for F-2 DoM AUROC.",
        "n": n,
        "n_correct": int(y.sum()),
        "tokenizer": "builtin-alnum-splitter (transformers banned); unigram fit on prompt corpus, Laplace-smoothed",
        "freq_correlations": corr,
        "auroc_dom_published_inmem": auroc_dom_published,
        "auroc_dom_oof_refit_raw": auroc_raw,
        "auroc_dom_oof_freq_partialled_from_activations": auroc_resid_act,
        "auroc_dom_score_residualized_on_freq": auroc_resid_score,
        "delta_auroc_from_partialling_activations": auroc_resid_act - auroc_raw,
        "delta_auroc_from_residualizing_score": auroc_resid_score - auroc_raw,
        "verdict": (
            "FREQ_CONFOUND" if (auroc_raw - auroc_resid_act) > 0.02
            else "FREQ_NOT_CONFOUND"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())