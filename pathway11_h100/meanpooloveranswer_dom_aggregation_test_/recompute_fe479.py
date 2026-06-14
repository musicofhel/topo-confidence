"""P11-FE479 — Mean-pool-over-answer DoM aggregation test.

From cached P11 1024-tok per-token L19 activations, compute a mean-pooled
over-answer-span DoM direction and compare it against the prefill-only DoM
(main cache) and the final-token DoM (last valid per-token position) along two
axes: (i) cosine similarity of the DoM directions and (ii) MATH-500 correctness
AUROC (OOF 5-fold).

F-3 reports cos(prefill_DoM, final_DoM) = 0.046 — a near-orthogonality between
two *specific per-position summaries*. STA-style aggregation instead means-pools
activations over the whole answer span, implicitly treating within-answer
position as exchangeable. If the answer-mean DoM is simultaneously well-aligned
to BOTH the prefill and final-token DoMs (cos >= 0.5) while still predicting
correctness (AUROC >= 0.75), then F-3's orthogonality is an artifact of the
per-position summary choice rather than a structural property of L19 geometry.
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

# Per-token L19 activations. Supported layouts (first that resolves wins):
#   (a) single padded NPZ with a (500, T, 1536) hidden tensor + lengths
#   (b) a directory of problem_NNN.npz files, each with a (T, 1536) hidden array
PERTOKEN_NPZ = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_pertoken.npz"
PERTOKEN_DIR = ROOT / "pathway11_h100/data/pertoken_l19"

OUT_JSON = ROOT / "pathway11_h100/answer_mean_dom/results.json"

N = 500
DIM = 1536
N_FOLDS = 5
SEED = 9999

_HIDDEN_KEYS = ("hidden", "states", "acts", "l19", "hidden_states", "pertoken")
_LEN_KEYS = ("lengths", "seq_len", "length", "n_tokens", "lens")
_PREFILL_KEYS = ("prefill_len", "prompt_len", "n_prompt", "answer_start", "prompt_tokens")


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a)); nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(a @ b / (na * nb))


def global_dom(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def oof_dom_scores(X: np.ndarray, y: np.ndarray, folds: list[np.ndarray]) -> np.ndarray:
    n = len(y)
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        tr = np.ones(n, dtype=bool); tr[test_idx] = False
        pos = tr & y; neg = tr & ~y
        if not pos.any() or not neg.any():
            continue
        d = X[pos].mean(axis=0) - X[neg].mean(axis=0)
        scores[test_idx] = X[test_idx] @ d
    return scores


def _find(d, names):
    for n in names:
        if n in d:
            return d[n]
    return None


def _spans_from_padded(blob):
    """Return (answer_mean, final) (N,DIM) from a single padded NPZ, or None."""
    hidden = _find(blob, _HIDDEN_KEYS)
    lengths = _find(blob, _LEN_KEYS)
    prefill_len = _find(blob, _PREFILL_KEYS)
    if hidden is None or lengths is None or prefill_len is None:
        return None
    hidden = np.asarray(hidden)
    if hidden.ndim != 3 or hidden.shape[0] != N or hidden.shape[2] != DIM:
        return None
    lengths = np.asarray(lengths).astype(int).ravel()
    prefill_len = np.asarray(prefill_len).astype(int).ravel()
    if lengths.shape[0] != N or prefill_len.shape[0] != N:
        return None
    answer_mean = np.zeros((N, DIM), dtype=np.float64)
    final = np.zeros((N, DIM), dtype=np.float64)
    for i in range(N):
        L = int(lengths[i]); p = int(prefill_len[i])
        if L <= 0:
            return None
        a0 = min(max(p, 0), L - 1)  # answer span start, clamped into [0, L-1]
        seq = hidden[i, :L].astype(np.float64)
        answer_mean[i] = seq[a0:].mean(axis=0)
        final[i] = seq[L - 1]
    return answer_mean, final


def _spans_from_dir(directory):
    """Return (answer_mean, final) (N,DIM) from per-problem NPZ files, or None."""
    answer_mean = np.zeros((N, DIM), dtype=np.float64)
    final = np.zeros((N, DIM), dtype=np.float64)
    for i in range(N):
        fp = directory / f"problem_{i:03d}.npz"
        if not fp.exists():
            return None
        blob = np.load(fp)
        hidden = _find(blob, _HIDDEN_KEYS)
        if hidden is None:
            return None
        hidden = np.asarray(hidden).astype(np.float64)
        if hidden.ndim != 2 or hidden.shape[1] != DIM or hidden.shape[0] < 1:
            return None
        prefill_len = _find(blob, _PREFILL_KEYS)
        p = int(np.asarray(prefill_len).ravel()[0]) if prefill_len is not None else 0
        a0 = min(max(p, 0), hidden.shape[0] - 1)
        answer_mean[i] = hidden[a0:].mean(axis=0)
        final[i] = hidden[-1]
    return answer_mean, final


def load_answer_and_final():
    if PERTOKEN_NPZ.exists():
        try:
            res = _spans_from_padded(np.load(PERTOKEN_NPZ))
        except Exception:
            res = None
        if res is not None:
            return res
    if PERTOKEN_DIR.is_dir():
        res = _spans_from_dir(PERTOKEN_DIR)
        if res is not None:
            return res
    return None


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr); return 2

    cache = np.load(CACHE)
    prefill = cache["prefill"].astype(np.float64)
    y = cache["correct"].astype(bool)
    assert prefill.shape == (N, DIM) and y.shape == (N,)

    spans = load_answer_and_final()
    if spans is None:
        print("MISSING_REGEN_INPUT", PERTOKEN_NPZ, "or", PERTOKEN_DIR, file=sys.stderr)
        return 2
    answer_mean, final = spans

    folds = stratified_kfold(y, N_FOLDS, SEED)

    # Global DoM directions (for cosine comparison of the directions themselves).
    dom_prefill = global_dom(prefill, y)
    dom_final = global_dom(final, y)
    dom_answer = global_dom(answer_mean, y)

    cos_answer_prefill = cosine(dom_answer, dom_prefill)
    cos_answer_final = cosine(dom_answer, dom_final)
    cos_prefill_final = cosine(dom_prefill, dom_final)  # F-3 reference (~0.046)

    # OOF AUROC (DoM refit per train fold).
    auroc_prefill = auroc(oof_dom_scores(prefill, y, folds), y)
    auroc_final = auroc(oof_dom_scores(final, y, folds), y)
    auroc_answer = auroc(oof_dom_scores(answer_mean, y, folds), y)

    aligned_both = (cos_answer_prefill >= 0.5) and (cos_answer_final >= 0.5)
    predictive = auroc_answer >= 0.75
    f3_is_summary_artifact = bool(aligned_both and predictive)

    out = {
        "experiment": "P11-FE479",
        "label_scheme": "1024tok",
        "n": int(N),
        "n_correct": int(y.sum()),
        "cos_answer_prefill": cos_answer_prefill,
        "cos_answer_final": cos_answer_final,
        "cos_prefill_final": cos_prefill_final,
        "auroc_prefill_dom_oof": auroc_prefill,
        "auroc_final_dom_oof": auroc_final,
        "auroc_answer_mean_dom_oof": auroc_answer,
        "answer_aligned_to_both": bool(aligned_both),
        "answer_predictive": bool(predictive),
        "f3_orthogonality_is_per_position_summary_artifact": f3_is_summary_artifact,
        "verdict": (
            "F-3 0.046 orthogonality is a per-position summary artifact"
            if f3_is_summary_artifact
            else "F-3 orthogonality survives answer-mean aggregation (structural)"
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())