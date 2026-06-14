"""P11-FE485 — CCPS final-token feature-bundle replication on Qwen-2.5-1.5B MATH-500.

Replicates the CCPS (Confidence from Consistency under Perturbation) single-pass
confidence recipe and tests whether its perturbation feature bundle dominates the
single-direction prefill DoM (F-2) on Qwen-2.5-1.5B MATH-500.

CCPS, as published, derives its features from the model's LM head: it perturbs the
final-token hidden state along the gradient of the predicted-token log-prob and
records how the *predictive distribution* over the vocabulary responds (Jacobian
L2 norm, epsilon-to-flip, perturbation-effect index PEI, and KL/JS divergence of
the perturbed-vs-base distribution). That requires the (vocab x d) LM-head matrix
and a GPU forward/backward pass — neither is available CPU-only without torch and
neither is cached as an NPZ here.

CPU-only substitution (documented in the output JSON `note` field): we replace the
LM head with an UNSUPERVISED pseudo-vocabulary built from the top-K right singular
vectors of the (centered) hidden states. logits = H_c @ V gives a K-way predictive
distribution; the entire CCPS perturbation bundle (Jacobian L2, eps-to-flip, PEI,
KL/JS mean/max over S=5 steps to epsilon_max=20.0) is then computed exactly as in
the paper but against this geometry-derived distribution. The bundle is label-free,
so it is computed once over all 500 examples; only the downstream MLP correctness
classifier is fit OOF (5-fold). We report AUROC, ECE, Brier, and selective-prediction
accuracy at coverage 0.5, and contrast against F-2 prefill DoM 0.7731, final-token
DoM 0.7186, and F-8 selective accuracy 71.6%.
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

try:
    from sklearn.neural_network import MLPClassifier
except Exception as exc:  # pragma: no cover
    print("MISSING_REGEN_INPUT sklearn", exc, file=sys.stderr)
    raise

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
# Preferred final-token hidden-state caches; first existing wins, else fall back
# to the L19 prefill states (noted in output).
FINAL_CANDIDATES = [
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz",
    ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final_token.npz",
    ROOT / "pathway11_h100/final_token/cache/m15b_final.npz",
]
OUT_JSON = ROOT / "pathway11_h100/ccps_replication/results.json"

SEED = 9999
N_FOLDS = 5
K_PSEUDO = 32        # pseudo-vocabulary size (top-K singular directions)
S_STEPS = 5          # perturbation steps
EPS_MAX = 20.0       # epsilon_max in hidden-state space
EPS = 1e-12


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y); rng.shuffle(pos)
    neg = np.flatnonzero(~y); rng.shuffle(neg)
    pos_folds = np.array_split(pos, k)
    neg_folds = np.array_split(neg, k)
    return [np.concatenate([p, n]) for p, n in zip(pos_folds, neg_folds)]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def kl_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1.0)
    q = np.clip(q, EPS, 1.0)
    return (p * np.log(p / q)).sum(axis=1)


def js_div(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)


def ece(prob_pos: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    labels = labels.astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(prob_pos)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob_pos > lo) & (prob_pos <= hi) if hi < 1.0 else (prob_pos > lo) & (prob_pos <= hi + EPS)
        if not m.any():
            continue
        conf = prob_pos[m].mean()
        acc = labels[m].mean()
        out += (m.sum() / total) * abs(conf - acc)
    return float(out)


def load_hidden_states() -> tuple[np.ndarray, str]:
    """Return (final-token-ish hidden states, source-tag)."""
    for cand in FINAL_CANDIDATES:
        if cand.exists():
            blob = np.load(cand)
            for key in ("final", "final_token", "hidden", "prefill"):
                if key in blob.files:
                    arr = blob[key].astype(np.float64)
                    if arr.ndim == 2 and arr.shape[0] == 500:
                        return arr, f"{cand.name}:{key}"
    blob = np.load(CACHE)
    return blob["prefill"].astype(np.float64), "m15b_prefill.npz:prefill (final-token cache absent — prefill fallback)"


def ccps_bundle(H: np.ndarray) -> np.ndarray:
    """Unsupervised CCPS perturbation feature bundle.

    Builds a K-way pseudo-vocabulary from the top-K singular directions of the
    centered hidden states, then perturbs each state along the descent direction
    of its predicted pseudo-token's log-prob over S steps to EPS_MAX, recording
    the CCPS feature set. Returns (n, n_features).
    """
    n, d = H.shape
    mu = H.mean(axis=0)
    Hc = H - mu
    # Top-K right singular vectors (pseudo-vocabulary projection), unsupervised.
    _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
    V = Vt[:K_PSEUDO].T                      # (d, K)
    L0 = Hc @ V                              # (n, K) base pseudo-logits
    p0 = softmax(L0)
    top0 = L0.argmax(axis=1)                 # predicted pseudo-token

    # Base margin features.
    sorted_L0 = np.sort(L0, axis=1)
    margin = sorted_L0[:, -1] - sorted_L0[:, -2]
    top_prob = p0[np.arange(n), top0]

    # Jacobian of the top pseudo-token log-prob w.r.t. hidden state:
    #   d/dh log p_top = V[:, top] - V @ p0  (in hidden-state space).
    jac_h = V[:, top0].T - (V @ p0.T).T      # (n, d)
    jac_l2 = np.linalg.norm(jac_h, axis=1)
    unit = jac_h / (jac_l2[:, None] + EPS)   # descent direction (toward flip)
    U = unit @ V                             # (n, K): per-eps logit shift / eps

    eps_steps = np.linspace(EPS_MAX / S_STEPS, EPS_MAX, S_STEPS)
    kl_vals = np.zeros((n, S_STEPS))
    js_vals = np.zeros((n, S_STEPS))
    top_drop = np.zeros((n, S_STEPS))        # (p0_top - ps_top) per step
    flipped_at = np.full(n, 2.0 * EPS_MAX)   # sentinel: never flipped
    for s, eps in enumerate(eps_steps):
        Ls = L0 - eps * U
        ps = softmax(Ls)
        kl_vals[:, s] = kl_div(p0, ps)
        js_vals[:, s] = js_div(p0, ps)
        top_drop[:, s] = top_prob - ps[np.arange(n), top0]
        newly = (Ls.argmax(axis=1) != top0) & (flipped_at == 2.0 * EPS_MAX)
        flipped_at[newly] = eps

    pei = top_drop.mean(axis=1)              # perturbation-effect index
    feats = np.column_stack([
        jac_l2,
        flipped_at,                          # epsilon-to-flip
        pei,
        kl_vals.mean(axis=1), kl_vals.max(axis=1),
        js_vals.mean(axis=1), js_vals.max(axis=1),
        top_prob,
        margin,
    ])
    return feats


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    labels = np.load(CACHE)["correct"].astype(bool)
    if labels.shape != (500,):
        print("MISSING_REGEN_INPUT bad-label-shape", labels.shape, file=sys.stderr)
        return 2

    H, source = load_hidden_states()
    if H.shape[0] != 500:
        print("MISSING_REGEN_INPUT bad-hidden-shape", H.shape, file=sys.stderr)
        return 2

    X = ccps_bundle(H)
    n = len(labels)

    # 5-fold OOF MLP over the CCPS feature bundle.
    folds = stratified_kfold(labels, N_FOLDS, SEED)
    oof_prob = np.zeros(n, dtype=np.float64)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        Xtr, Xte = X[train_mask], X[test_idx]
        ytr = labels[train_mask]
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + EPS
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        clf = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1e-3,
            max_iter=500,
            random_state=SEED,
        )
        clf.fit(Xtr_s, ytr)
        classes = list(clf.classes_)
        pos_col = classes.index(True) if True in classes else classes.index(1)
        oof_prob[test_idx] = clf.predict_proba(Xte_s)[:, pos_col]

    bundle_auroc = auroc(oof_prob, labels)
    brier = float(np.mean((oof_prob - labels.astype(np.float64)) ** 2))
    ece_val = ece(oof_prob, labels)

    # Selective prediction: answer the top-coverage fraction by predicted
    # P(correct); report actual accuracy on the answered set.
    order = np.argsort(-oof_prob)
    k_answer = int(round(0.5 * n))
    answered = order[:k_answer]
    selective_acc = float(labels[answered].mean())

    # Per-feature univariate AUROC (which CCPS feature carries the signal).
    feat_names = [
        "jacobian_l2", "epsilon_to_flip", "pei",
        "kl_mean", "kl_max", "js_mean", "js_max",
        "top_prob", "margin",
    ]
    per_feature_auroc = {nm: float(auroc(X[:, j], labels)) for j, nm in enumerate(feat_names)}

    F2_PREFILL_DOM = 0.7731
    FINAL_DOM = 0.7186
    F8_SELECTIVE = 0.716

    out = {
        "experiment": "P11-FE485",
        "method": "CCPS feature-bundle replication (CPU pseudo-vocabulary surrogate)",
        "note": (
            "LM head unavailable CPU-only; pseudo-vocabulary = top-%d singular "
            "directions of centered hidden states. Perturbation bundle computed "
            "exactly as CCPS over S=%d steps to epsilon_max=%.1f." % (K_PSEUDO, S_STEPS, EPS_MAX)
        ),
        "hidden_state_source": source,
        "n_examples": n,
        "base_accuracy": float(labels.mean()),
        "k_pseudo_vocab": K_PSEUDO,
        "s_steps": S_STEPS,
        "epsilon_max": EPS_MAX,
        "ccps_bundle_auroc_oof": bundle_auroc,
        "ccps_bundle_ece": ece_val,
        "ccps_bundle_brier": brier,
        "ccps_bundle_selective_acc_at_coverage_0.5": selective_acc,
        "per_feature_univariate_auroc": per_feature_auroc,
        "baselines": {
            "F2_prefill_dom_auroc": F2_PREFILL_DOM,
            "final_token_dom_auroc": FINAL_DOM,
            "F8_selective_acc_at_coverage_0.5": F8_SELECTIVE,
        },
        "comparison": {
            "auroc_minus_prefill_dom": bundle_auroc - F2_PREFILL_DOM,
            "auroc_minus_final_dom": bundle_auroc - FINAL_DOM,
            "selective_minus_F8": selective_acc - F8_SELECTIVE,
            "ccps_bundle_dominates_prefill_dom": bool(bundle_auroc > F2_PREFILL_DOM),
            "ccps_bundle_dominates_final_dom": bool(bundle_auroc > FINAL_DOM),
        },
        "seed": SEED,
        "n_folds": N_FOLDS,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print("WROTE", OUT_JSON)
    print("ccps_bundle_auroc_oof=%.4f  selective@0.5=%.4f" % (bundle_auroc, selective_acc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())