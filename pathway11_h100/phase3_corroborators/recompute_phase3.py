"""Phase 3 — causal corroborators bundle (P11-FE136, FE244, FE319, FE331, FE339, FE254).

Six pure-cache (or cache + tied-embedding) probes that test alternative
geometric framings of F-2 (prefill L19 DoM AUROC 0.7731) and F-3 (prefill/
final orthogonality cos = 0.046) without requiring fresh model forward
passes. P11-FE110 (ActAdd contrast pairs) is deferred to Phase 4 — it
needs short prompts run through the model.

References per FE:
  FE136 — Park-Choe-Veitch causal inner product (2311.03658)
  FE244 — NC3 / W_unembed alignment for orthogonality
  FE319 — quadratic probe on top-k SVD directions
  FE331 — Stolfo principled steering coefficient (gates H-1)
  FE339 — Linear-AcT / mean-AcT (variance-aware probe)
  FE254 — joint prefill+final concat probe (F-8 baseline check)

Inputs:
  pathway11_h100/prefill_inversion/cache/m15b_prefill.npz  (prefill, final_tok, correct, seq_len)
  pathway11_h100/phase3_corroborators/W_unembed_15b.npy    (tied embed_tokens for FE136 + FE244)
  pathway8_layerwise/data/math500/problem_*.npz            (text — for FE244 answer-token freq)

Outputs:
  pathway11_h100/phase3_corroborators/results.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
PREFILL_NPZ = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
W_UNEMBED_NPY = ROOT / "pathway11_h100/phase3_corroborators/W_unembed_15b.npy"
PROBLEM_NPZ_DIR = ROOT / "pathway8_layerwise/data/math500"
OUT_JSON = ROOT / "pathway11_h100/phase3_corroborators/results.json"

N_FOLDS = 5
SEED = 9999
RIDGE_REL = 1e-3
TOP_K_ANSWER_TOKENS = 50


# ---------- Probe helpers (shared with Phase 1/2) ----------

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


def oof_logreg_auroc(X: np.ndarray, y: np.ndarray, k: int, seed: int,
                     ridge_rel: float = RIDGE_REL) -> float:
    """OOF AUROC via Σ⁻¹(μ_pos - μ_neg) on whatever feature matrix X is."""
    folds = stratified_kfold(y, k, seed)
    n = len(y); d_ = X.shape[1]
    scores = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train = np.ones(n, dtype=bool); train[test_idx] = False
        Xtr = X[train]
        d = X[train & y].mean(0) - X[train & ~y].mean(0)
        Xc = Xtr - Xtr.mean(0)
        Sigma = Xc.T @ Xc / max(len(Xtr) - 1, 1)
        Sigma += ridge_rel * np.trace(Sigma) / d_ * np.eye(d_)
        w = np.linalg.solve(Sigma, d)
        scores[test_idx] = X[test_idx] @ w
    return auroc(scores, y)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float((a @ b) / (na * nb))


# ---------- FE136 — Park-Choe-Veitch causal inner product ----------

def fe136_causal_inner(prefill: np.ndarray, final_tok: np.ndarray,
                       correct: np.ndarray, W_unembed: np.ndarray) -> dict:
    """Whitened cosine via M = Cov(γ)^{-1} where γ = unembed rows.

    Park-Choe-Veitch 2311.03658: the causal-inner-product space is the one
    where unembedding directions are orthonormal. Compute cos in that space
    for prefill_DoM vs final_DoM and compare to raw cos = 0.046 (F-3 anchor).
    """
    # γ = W_unembed rows (each row = unembedding vector for one token)
    gamma = W_unembed.astype(np.float64)         # (V, H)
    # Cov(γ): center across vocabulary, compute H×H covariance
    g_c = gamma - gamma.mean(axis=0, keepdims=True)
    Sigma = (g_c.T @ g_c) / max(len(gamma) - 1, 1)   # (H, H)
    # Ridge for numerical stability
    Sigma += 1e-6 * np.trace(Sigma) / Sigma.shape[0] * np.eye(Sigma.shape[0])
    M = np.linalg.inv(Sigma)                         # whitening matrix

    d_prefill = prefill[correct].mean(0) - prefill[~correct].mean(0)
    d_final = final_tok[correct].mean(0) - final_tok[~correct].mean(0)

    # Raw cos (sanity)
    cos_raw = cos_sim(d_prefill, d_final)
    # Whitened cos: <a,b>_M = a^T M b; ||a||_M = sqrt(a^T M a)
    ip_M = float(d_prefill @ M @ d_final)
    n_p_M = float(np.sqrt(d_prefill @ M @ d_prefill))
    n_f_M = float(np.sqrt(d_final @ M @ d_final))
    cos_whitened = ip_M / (n_p_M * n_f_M) if n_p_M > 0 and n_f_M > 0 else 0.0

    return {
        "fe136_cos_raw": cos_raw,
        "fe136_cos_whitened": cos_whitened,
        "fe136_delta": cos_whitened - cos_raw,
    }


# ---------- FE244 — NC3 / W_unembed alignment ----------

def fe244_unembed_alignment(prefill: np.ndarray, final_tok: np.ndarray,
                            correct: np.ndarray, W_unembed: np.ndarray) -> dict:
    """cos(DoM, W_unembed[top-K answer tokens]).

    NC3 collapse predicts that DoM directions align with the unembedding
    rows of high-frequency output tokens. If final_DoM aligns (max cos
    > 0.3) but prefill_DoM does not, F-3 orthogonality is a generic NC3
    signature, not a circuit-level claim.

    "Top-K answer tokens": tokens that appear most often in the generated
    `text` field of the per-problem cache (acts as a simple proxy for
    answer-position tokens without re-tokenising).
    """
    # Identify top-K most-frequent tokens across all 500 generations (text).
    # The pathway8 cache stores `text` as bytes / str; we count unique token
    # IDs by re-tokenising via the same Qwen tokenizer.
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as e:
        return {"fe244_skipped": f"tokenizer load failed: {e}"}

    counter: Counter[int] = Counter()
    files = sorted(PROBLEM_NPZ_DIR.glob("problem_*.npz"))
    for f in files:
        d = np.load(f, allow_pickle=True)
        if "text" not in d.files:
            continue
        txt = str(d["text"])
        ids = tok.encode(txt, add_special_tokens=False)
        counter.update(ids)
    top_ids = [tid for tid, _ in counter.most_common(TOP_K_ANSWER_TOKENS)]
    if not top_ids:
        return {"fe244_skipped": "no answer tokens found"}

    d_prefill = prefill[correct].mean(0) - prefill[~correct].mean(0)
    d_final = final_tok[correct].mean(0) - final_tok[~correct].mean(0)

    rows = W_unembed[top_ids].astype(np.float64)   # (K, H)
    # cos(DoM, each unembed row)
    norm_p = np.linalg.norm(d_prefill); norm_f = np.linalg.norm(d_final)
    norm_rows = np.linalg.norm(rows, axis=1)
    eps = 1e-12
    cos_prefill_per = (rows @ d_prefill) / (norm_rows * max(norm_p, eps))
    cos_final_per = (rows @ d_final) / (norm_rows * max(norm_f, eps))

    return {
        "fe244_n_tokens": len(top_ids),
        "fe244_max_abs_cos_prefill_unembed": float(np.max(np.abs(cos_prefill_per))),
        "fe244_max_abs_cos_final_unembed": float(np.max(np.abs(cos_final_per))),
        "fe244_mean_abs_cos_prefill_unembed": float(np.mean(np.abs(cos_prefill_per))),
        "fe244_mean_abs_cos_final_unembed": float(np.mean(np.abs(cos_final_per))),
    }


# ---------- FE319 — quadratic probe on top-k SVD directions ----------

def fe319_quadratic_svd(prefill: np.ndarray, correct: np.ndarray) -> dict:
    """logistic(α + Σ β_i (v_i^T h)²) for k ∈ {1, 2, 5, 10, 20}.

    On 5-fold OOF: fold-internal SVD of train prefill matrix; score is
    Σ β_i (v_i^T h)² where β_i are fold-internal LDA-style coefficients
    on the squared projections.
    """
    folds = stratified_kfold(correct, N_FOLDS, SEED)
    out = {}
    for k in (1, 2, 5, 10, 20):
        scores = np.zeros(len(correct), dtype=np.float64)
        for test_idx in folds:
            train = np.ones(len(correct), dtype=bool); train[test_idx] = False
            Xtr = prefill[train]
            Xc = Xtr - Xtr.mean(0)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            Vk = Vt[:k]                                 # (k, H)
            feats_tr = (Xtr @ Vk.T) ** 2                # (n_tr, k)
            feats_te = (prefill[test_idx] @ Vk.T) ** 2  # (n_te, k)
            # Mean-difference probe in the squared-feature space
            d = (feats_tr[correct[train]].mean(0)
                 - feats_tr[~correct[train]].mean(0))
            scores[test_idx] = feats_te @ d
        out[f"fe319_quadratic_auroc_k{k}"] = float(auroc(scores, correct))
    return out


# ---------- FE331 — Stolfo principled steering coefficient ----------

def fe331_stolfo_c(prefill: np.ndarray, correct: np.ndarray) -> dict:
    """Dynamic-c steering coefficient for prefill_DoM (gates H-1).

    Stolfo Eq. 2: c = mean projection of correct activations onto the
    DoM direction (unit-normalised). H-1's arbitrary alpha-sweep is
    replaced by this principled magnitude.
    """
    d = prefill[correct].mean(0) - prefill[~correct].mean(0)
    u = d / np.linalg.norm(d)
    proj_correct = prefill[correct] @ u
    proj_incorrect = prefill[~correct] @ u
    return {
        "fe331_c_mean_correct_proj": float(proj_correct.mean()),
        "fe331_c_mean_incorrect_proj": float(proj_incorrect.mean()),
        "fe331_c_signed_diff": float(proj_correct.mean() - proj_incorrect.mean()),
        "fe331_c_recommended_alpha": float(proj_correct.mean() - proj_incorrect.mean()),
        "fe331_dom_norm": float(np.linalg.norm(d)),
    }


# ---------- FE339 — Linear-AcT / variance-aware probe ----------

def fe339_linear_act(prefill: np.ndarray, correct: np.ndarray) -> dict:
    """Per-coordinate (ω, β) closed-form Linear-AcT.

    For each hidden dim i: ω_i = σ_target_i / σ_source_i,
    β_i = μ_target_i - ω_i · μ_source_i.
    Score: signed-distance-to-corrected-source. We compute it OOF.

    Predicts: σ-aware probe beats mean-only DoM by ≥0.02 AUROC.
    """
    folds = stratified_kfold(correct, N_FOLDS, SEED)
    scores = np.zeros(len(correct), dtype=np.float64)
    for test_idx in folds:
        train = np.ones(len(correct), dtype=bool); train[test_idx] = False
        Xs = prefill[train & ~correct]            # source = incorrect
        Xt = prefill[train & correct]             # target = correct
        mu_s = Xs.mean(0); mu_t = Xt.mean(0)
        sigma_s = Xs.std(0) + 1e-6
        sigma_t = Xt.std(0) + 1e-6
        omega = sigma_t / sigma_s
        beta = mu_t - omega * mu_s
        # Apply Linear-AcT to test points: y = ω⊙x + β.
        # Score = || y - μ_t || - || x - μ_t ||  (smaller after-transform
        # means the point belongs to source → incorrect; we flip sign so
        # higher score = correct, then OOF threshold).
        Xte = prefill[test_idx]
        y_after = omega * Xte + beta
        d_before = np.linalg.norm(Xte - mu_t, axis=1)
        d_after = np.linalg.norm(y_after - mu_t, axis=1)
        # If a point is correct, d_before should already be small (no
        # transform needed). Use d_before - d_after: positive means
        # the transform helped (point was incorrect-shaped); flip.
        scores[test_idx] = -(d_before - d_after)
    return {
        "fe339_linear_act_auroc": float(auroc(scores, correct)),
    }


# ---------- FE254 — joint prefill+final concat probe ----------

def fe254_joint_concat(prefill: np.ndarray, final_tok: np.ndarray,
                       correct: np.ndarray) -> dict:
    """Concat [prefill, final_tok] → 5-fold OOF DoM-style probe.

    Uses the simple mean-diff projection (matching F-2's 0.7731 baseline
    methodology). LDA-style Σ⁻¹ projection is rank-deficient in this
    n<d regime (~400 train, 3072 dim) and produces misleading numbers
    even with light ridge — see initial run that returned 0.55 prefill-
    alone AUROC vs the established 0.7731.
    """
    Xj = np.concatenate([prefill, final_tok], axis=1).astype(np.float64)
    auroc_joint = oof_dom_auroc(Xj, correct, N_FOLDS, SEED)
    auroc_prefill_alone = oof_dom_auroc(prefill.astype(np.float64),
                                        correct, N_FOLDS, SEED)
    auroc_final_alone = oof_dom_auroc(final_tok.astype(np.float64),
                                      correct, N_FOLDS, SEED)
    return {
        "fe254_joint_auroc": auroc_joint,
        "fe254_prefill_alone_auroc": auroc_prefill_alone,
        "fe254_final_alone_auroc": auroc_final_alone,
        "fe254_lift_over_prefill": auroc_joint - auroc_prefill_alone,
    }


# ---------- Main ----------

def main() -> int:
    if not PREFILL_NPZ.exists():
        print(f"MISSING_REGEN_INPUT {PREFILL_NPZ}", file=sys.stderr); return 2
    if not W_UNEMBED_NPY.exists():
        print(f"MISSING_REGEN_INPUT {W_UNEMBED_NPY}", file=sys.stderr); return 2

    pf = np.load(PREFILL_NPZ)
    prefill = pf["prefill"].astype(np.float64)
    final_tok = pf["final_tok"].astype(np.float64)
    correct = pf["correct"].astype(bool)
    W_unembed = np.load(W_UNEMBED_NPY)
    print(f"loaded: prefill {prefill.shape}, final {final_tok.shape}, "
          f"W_unembed {W_unembed.shape}, correct {correct.sum()}/{len(correct)}")

    out: dict = {
        "n": int(len(correct)),
        "n_correct": int(correct.sum()),
        "n_folds": N_FOLDS,
        "seed": SEED,
        "ridge_rel": RIDGE_REL,
    }

    print("\n== FE136 (causal inner product) ==")
    out.update(fe136_causal_inner(prefill, final_tok, correct, W_unembed))
    for k, v in out.items():
        if k.startswith("fe136"): print(f"  {k}={v}")

    print("\n== FE244 (W_unembed alignment) ==")
    out.update(fe244_unembed_alignment(prefill, final_tok, correct, W_unembed))
    for k, v in out.items():
        if k.startswith("fe244"): print(f"  {k}={v}")

    print("\n== FE319 (quadratic SVD probe) ==")
    out.update(fe319_quadratic_svd(prefill, correct))
    for k, v in out.items():
        if k.startswith("fe319"): print(f"  {k}={v}")

    print("\n== FE331 (Stolfo c) ==")
    out.update(fe331_stolfo_c(prefill, correct))
    for k, v in out.items():
        if k.startswith("fe331"): print(f"  {k}={v}")

    print("\n== FE339 (Linear-AcT) ==")
    out.update(fe339_linear_act(prefill, correct))
    for k, v in out.items():
        if k.startswith("fe339"): print(f"  {k}={v}")

    print("\n== FE254 (joint concat) ==")
    out.update(fe254_joint_concat(prefill, final_tok, correct))
    for k, v in out.items():
        if k.startswith("fe254"): print(f"  {k}={v}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWROTE {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
