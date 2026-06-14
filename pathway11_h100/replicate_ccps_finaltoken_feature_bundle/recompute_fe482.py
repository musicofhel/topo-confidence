"""P11-FE482 — CCPS final-token feature bundle vs prefill DoM on Qwen-2.5-1.5B MATH-500.

Replicates the CCPS (Confidence from Consistency under Perturbation of the
residual Stream) single-pass final-token confidence bundle on the cached P11
H100 final-token hidden states, then asks whether it dominates F-2's single
prefill DoM direction.

For each problem we take the final-token L19 hidden state h (1536-d), reconstruct
logits via the cached LM-head unembedding (logits = W_U @ h + b), and probe local
stability of the predicted token under S=5 perturbation steps (epsilon_max=20.0):

  - jac_l2     : ||d log p_{t*} / dh||_2  = ||W_U[t*] - E_p[W_U]||_2  (logit Jacobian
                 of the predicted token's log-prob, since logits are linear in h)
  - eps_to_flip: smallest epsilon along the margin-reducing direction that flips argmax
  - pei        : Perturbation Effect Index — mean drop in p[t*] over random perturbations
  - kl/js stats: mean/max KL(p||q) and JS(p,q) over S*R random unit perturbations
  - msp,ent,mgn: max-softmax-prob, entropy, top1-top2 logit margin (CCPS base features)

A 5-fold OOF MLP classifier is trained on the standardized bundle. We report AUROC,
ECE, Brier, and selective-prediction accuracy at coverage 0.5, and compare against
F-2 prefill DoM AUROC 0.7731, final-token DoM AUROC 0.7186, and F-8 selective acc 71.6%.

CPU/numpy only: the "Jacobian backward pass" is exact (linear head) and perturbed
logits are a single matmul per problem against the cached W_U.
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
# Final-token Stage-2 cache: hidden states + LM-head unembedding for logit reconstruction.
FINAL_NPZ = ROOT / "pathway11_h100/ccps_final_token/cache/m15b_final.npz"
LMHEAD_NPZ = ROOT / "pathway11_h100/ccps_final_token/cache/lm_head.npz"
# Labels fall back to the canonical prefill cache if the final cache lacks them.
MAIN_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/ccps_final_token/results.json"

SEED = 9999
N_FOLDS = 5
S_STEPS = 5
R_DIRS = 4
EPSILON_MAX = 20.0

# Literature / internal comparison anchors.
F2_PREFILL_DOM_AUROC = 0.7731
FINAL_DOM_AUROC = 0.7186
F8_SELECTIVE_ACC = 0.716

FEATURE_NAMES = [
    "jac_l2", "eps_to_flip", "pei",
    "kl_mean", "kl_max", "js_mean", "js_max",
    "msp", "entropy", "margin",
]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    probs = np.clip(probs, 0.0, 1.0)
    labels = labels.astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(probs)
    total = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (probs > lo) & (probs <= hi) if b > 0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        total += (mask.sum() / n) * abs(conf - acc)
    return float(total)


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels.astype(np.float64)) ** 2))


def selective_accuracy(conf: np.ndarray, correct: np.ndarray, coverage: float) -> float:
    """Accuracy on the most-confident `coverage` fraction of problems."""
    n = len(conf)
    k = max(1, int(round(coverage * n)))
    order = np.argsort(conf)[::-1][:k]
    return float(correct[order].astype(np.float64).mean())


def _get(blob, names):
    for n in names:
        if n in blob.files:
            return blob[n]
    return None


def log_softmax(logits: np.ndarray) -> np.ndarray:
    m = logits.max()
    shifted = logits - m
    return shifted - np.log(np.exp(shifted).sum())


def compute_ccps_features(h_all: np.ndarray, W_U: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Per-problem CCPS feature bundle. h_all: (N,1536), W_U: (V,1536)."""
    n, _ = h_all.shape
    W_U = np.ascontiguousarray(W_U.astype(np.float32))
    feats = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    eps_grid = EPSILON_MAX * np.arange(1, S_STEPS + 1, dtype=np.float64) / S_STEPS
    rng = np.random.default_rng(SEED)
    no_flip_sentinel = EPSILON_MAX * 2.0

    for i in range(n):
        h = h_all[i].astype(np.float32)
        base_logits = (W_U @ h).astype(np.float64) + bias
        log_p = log_softmax(base_logits)
        p = np.exp(log_p)
        t_star = int(np.argmax(base_logits))
        # second-best logit for margin / adversarial direction
        order = np.argpartition(base_logits, -2)[-2:]
        second = int(order[0]) if order[1] == t_star else int(order[1])

        # exact logit Jacobian of predicted log-prob (linear head): W_U[t*] - E_p[W_U]
        e_wu = (p @ W_U).astype(np.float64)
        grad_logp = W_U[t_star].astype(np.float64) - e_wu
        jac_l2 = float(np.linalg.norm(grad_logp))

        msp = float(p[t_star])
        entropy = float(-(p * log_p).sum())
        margin = float(base_logits[t_star] - base_logits[second])

        # margin-reducing direction: push h to lift `second` over `t_star`
        adir = (W_U[second].astype(np.float64) - W_U[t_star].astype(np.float64))
        an = np.linalg.norm(adir)
        adir = adir / an if an > 1e-12 else np.zeros_like(adir)

        # random unit perturbation directions
        g = rng.standard_normal((R_DIRS, h.shape[0]))
        g /= (np.linalg.norm(g, axis=1, keepdims=True) + 1e-12)

        # build perturbation set: S*R random, then S adversarial
        deltas = np.empty((S_STEPS * R_DIRS + S_STEPS, h.shape[0]), dtype=np.float32)
        k = 0
        for s in range(S_STEPS):
            for r in range(R_DIRS):
                deltas[k] = (eps_grid[s] * g[r]).astype(np.float32)
                k += 1
        adv_start = k
        for s in range(S_STEPS):
            deltas[k] = (eps_grid[s] * adir).astype(np.float32)
            k += 1

        pert_h = h[None, :] + deltas
        pert_logits = (pert_h @ W_U.T).astype(np.float64) + bias[None, :]

        # random-perturbation divergence + PEI stats
        kls, jss, drops = [], [], []
        for j in range(adv_start):
            log_q = log_softmax(pert_logits[j])
            q = np.exp(log_q)
            kl = float((p * (log_p - log_q)).sum())
            m = 0.5 * (p + q)
            log_m = np.log(m + 1e-30)
            js = float(0.5 * (p * (log_p - log_m)).sum() + 0.5 * (q * (log_q - log_m)).sum())
            kls.append(kl)
            jss.append(js)
            drops.append(msp - float(q[t_star]))
        kls = np.asarray(kls)
        jss = np.asarray(jss)

        # epsilon-to-flip along adversarial direction
        eps_to_flip = no_flip_sentinel
        for s in range(S_STEPS):
            if int(np.argmax(pert_logits[adv_start + s])) != t_star:
                eps_to_flip = float(eps_grid[s])
                break

        feats[i] = [
            jac_l2, eps_to_flip, float(np.mean(drops)),
            float(kls.mean()), float(kls.max()),
            float(jss.mean()), float(jss.max()),
            msp, entropy, margin,
        ]
    return feats


def main() -> int:
    # Load final-token hidden states + LM-head unembedding.
    if not FINAL_NPZ.exists():
        print("MISSING_REGEN_INPUT", FINAL_NPZ, file=sys.stderr)
        return 2
    fblob = np.load(FINAL_NPZ)
    h_all = _get(fblob, ["final", "final_hidden", "hidden", "final_token", "h_final"])
    if h_all is None:
        print("MISSING_REGEN_INPUT", f"{FINAL_NPZ}:final_hidden", file=sys.stderr)
        return 2
    h_all = h_all.astype(np.float32)

    W_U = _get(fblob, ["lm_head", "W_U", "unembed", "unembedding", "lm_head_weight"])
    bias = _get(fblob, ["lm_head_bias", "bias", "b"])
    if W_U is None:
        if not LMHEAD_NPZ.exists():
            print("MISSING_REGEN_INPUT", LMHEAD_NPZ, file=sys.stderr)
            return 2
        lblob = np.load(LMHEAD_NPZ)
        W_U = _get(lblob, ["lm_head", "W_U", "unembed", "unembedding", "lm_head_weight"])
        if bias is None:
            bias = _get(lblob, ["lm_head_bias", "bias", "b"])
        if W_U is None:
            print("MISSING_REGEN_INPUT", f"{LMHEAD_NPZ}:lm_head", file=sys.stderr)
            return 2

    W_U = W_U.astype(np.float32)
    if W_U.shape[1] != h_all.shape[1]:
        if W_U.shape[0] == h_all.shape[1]:
            W_U = W_U.T  # accept (hidden, vocab) layout
        else:
            print("MISSING_REGEN_INPUT", f"lm_head shape {W_U.shape} incompatible", file=sys.stderr)
            return 2
    vocab = W_U.shape[0]
    bias = np.zeros(vocab, dtype=np.float64) if bias is None else bias.astype(np.float64)

    # Labels.
    correct = _get(fblob, ["correct", "labels", "y"])
    if correct is None:
        if not MAIN_CACHE.exists():
            print("MISSING_REGEN_INPUT", MAIN_CACHE, file=sys.stderr)
            return 2
        correct = np.load(MAIN_CACHE)["correct"]
    correct = correct.astype(bool)

    n = len(correct)
    assert h_all.shape == (n, 1536), f"unexpected hidden shape {h_all.shape}"

    feats = compute_ccps_features(h_all, W_U, bias)

    # Univariate AUROC per feature (sign-flip so larger => more confident).
    univ = {}
    for j, name in enumerate(FEATURE_NAMES):
        a = auroc(feats[:, j], correct)
        univ[name] = max(a, 1.0 - a) if not np.isnan(a) else float("nan")

    # 5-fold OOF MLP on standardized bundle.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    oof_prob = np.zeros(n, dtype=np.float64)
    for train_idx, test_idx in skf.split(feats, correct):
        scaler = StandardScaler().fit(feats[train_idx])
        Xtr = scaler.transform(feats[train_idx])
        Xte = scaler.transform(feats[test_idx])
        clf = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=1e-3,
            max_iter=800,
            random_state=SEED,
        )
        clf.fit(Xtr, correct[train_idx].astype(int))
        cls = list(clf.classes_)
        pos_col = cls.index(1) if 1 in cls else 0
        oof_prob[test_idx] = clf.predict_proba(Xte)[:, pos_col]

    auroc_bundle = auroc(oof_prob, correct)
    auroc_msp = univ["msp"]
    sel_acc = selective_accuracy(oof_prob, correct, 0.5)

    out = {
        "experiment": "P11-FE482",
        "method": "CCPS final-token feature bundle vs prefill DoM",
        "config": {
            "n": int(n),
            "vocab": int(vocab),
            "hidden": int(h_all.shape[1]),
            "S_steps": S_STEPS,
            "R_dirs": R_DIRS,
            "epsilon_max": EPSILON_MAX,
            "n_folds": N_FOLDS,
            "seed": SEED,
            "features": FEATURE_NAMES,
        },
        "auroc_ccps_bundle_oof": float(auroc_bundle),
        "auroc_msp_only": float(auroc_msp),
        "univariate_auroc": univ,
        "ece_ccps_bundle": ece(oof_prob, correct),
        "brier_ccps_bundle": brier(oof_prob, correct),
        "selective_acc_coverage_0.5": float(sel_acc),
        "base_accuracy": float(correct.mean()),
        "comparison": {
            "f2_prefill_dom_auroc": F2_PREFILL_DOM_AUROC,
            "final_token_dom_auroc": FINAL_DOM_AUROC,
            "f8_selective_acc": F8_SELECTIVE_ACC,
            "ccps_beats_prefill_dom": bool(auroc_bundle > F2_PREFILL_DOM_AUROC),
            "ccps_beats_final_dom": bool(auroc_bundle > FINAL_DOM_AUROC),
            "ccps_beats_f8_selective": bool(sel_acc > F8_SELECTIVE_ACC),
        },
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"CCPS bundle OOF AUROC = {auroc_bundle:.4f} (prefill DoM {F2_PREFILL_DOM_AUROC}, "
          f"final DoM {FINAL_DOM_AUROC}); selective@0.5 = {sel_acc:.4f} (F-8 {F8_SELECTIVE_ACC})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())