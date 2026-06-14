"""P11-FE408 — fTRI-on-MLP-neurons sanity check at L19 prefill.

Class-averaged differential-activation analysis (the fTRI / MoTE-style sparse
mediator probe) applied to cached pathway11_h100 L19 prefill activations on
Qwen-2.5-1.5B. For each neuron we take the class-mean difference (correct vs
incorrect); the top-K |differential| neurons form a sparse, sign-aligned
read-out whose summed (z-scored) activation is scored by AUROC for
K in {1, 5, 10, 50}. Both in-sample (selection==scoring set) and honest 5-fold
OOF (select on train, score on test) numbers are reported and compared against
the F-2 L19 DoM AUROC of 0.7731.

If MoTE's "0.07% of routes mediate 52% of behavior" sparsity transfers to dense
Qwen-1.5B correctness, a handful of neurons should approach the continuous DoM;
if even K=50 lags far behind 0.7731, DoM is not a lossy projection of a sparse
mechanism (Refutation 1).

The 8960-d MLP-intermediate cache is not committed locally; when it is absent we
fall back to the 1536-d L19 residual-stream basis (the committed prefill cache)
as the neuron basis and flag `neuron_basis` in the output JSON. Either way the
methodology — class-averaged differential activation, top-K sparse read-out — is
identical.
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
MLP_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill_mlp.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/ftri_mlp_neurons/results.json"

F2_DOM_AUROC = 0.7731
TOP_K = [1, 5, 10, 50]
N_FOLDS = 5
SEED = 9999


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


def diff_act_score(Xz: np.ndarray, y_sel: np.ndarray, k: int):
    """Top-k differential-activation neurons selected on Xz[y_sel rows].

    Returns (neuron_idx, signs) for a sign-aligned summed read-out.
    Xz is already z-scored, so per-neuron class-mean difference is comparable.
    """
    diff = Xz[y_sel].mean(axis=0) - Xz[~y_sel].mean(axis=0)
    order = np.argsort(-np.abs(diff))[:k]
    signs = np.sign(diff[order])
    signs[signs == 0] = 1.0
    return order, signs


def main() -> int:
    # Pick the neuron basis: MLP-intermediate cache if present, else residual.
    if MLP_CACHE.exists():
        blob = np.load(MLP_CACHE)
        # tolerate a couple of plausible key names for the MLP activation matrix
        key = next((k for k in ("mlp_act", "mlp", "prefill_mlp", "act") if k in blob.files), None)
        if key is None:
            print("MISSING_REGEN_INPUT", MLP_CACHE, "(no mlp activation key)", file=sys.stderr)
            return 2
        X = blob[key].astype(np.float64)
        neuron_basis = f"mlp_intermediate:{key}"
        # correctness labels: prefer co-located, else main cache
        if "correct" in blob.files:
            y = blob["correct"].astype(bool)
        elif CACHE.exists():
            y = np.load(CACHE)["correct"].astype(bool)
        else:
            print("MISSING_REGEN_INPUT", CACHE, "(labels)", file=sys.stderr)
            return 2
    elif CACHE.exists():
        blob = np.load(CACHE)
        X = blob["prefill"].astype(np.float64)
        y = blob["correct"].astype(bool)
        neuron_basis = "l19_residual_stream_1536d"
    else:
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2

    n, d = X.shape
    if X.shape[0] != y.shape[0]:
        print("MISSING_REGEN_INPUT shape mismatch", X.shape, y.shape, file=sys.stderr)
        return 2

    # --- In-sample (whole-set selection == scoring) -------------------------
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xz_all = (X - mu) / sd
    insample = {}
    for k in TOP_K:
        order, signs = diff_act_score(Xz_all, y, k)
        score = Xz_all[:, order] @ signs
        insample[str(k)] = float(auroc(score, y))

    # --- Honest 5-fold OOF (select top-K on train, score held-out test) -----
    folds = stratified_kfold(y, N_FOLDS, SEED)
    oof_scores = {k: np.zeros(n, dtype=np.float64) for k in TOP_K}
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool); train_mask[test_idx] = False
        Xtr, ytr = X[train_mask], y[train_mask]
        mu_tr = Xtr.mean(axis=0)
        sd_tr = Xtr.std(axis=0); sd_tr[sd_tr < 1e-12] = 1.0
        Xtr_z = (Xtr - mu_tr) / sd_tr
        Xte_z = (X[test_idx] - mu_tr) / sd_tr
        for k in TOP_K:
            order, signs = diff_act_score(Xtr_z, ytr, k)
            oof_scores[k][test_idx] = Xte_z[:, order] @ signs
    oof = {str(k): float(auroc(oof_scores[k], y)) for k in TOP_K}

    # --- DoM reference (sanity: confirms the 0.7731 ballpark on this cache) --
    dom_auroc = None
    if DOM_NPZ.exists():
        dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
        if dom_score.shape[0] == n:
            dom_auroc = float(auroc(dom_score, y))

    best_oof_k = max(TOP_K, key=lambda k: oof[str(k)])
    best_oof = oof[str(best_oof_k)]

    out = {
        "experiment": "P11-FE408",
        "description": "fTRI/MoTE class-averaged differential-activation top-K neuron read-out vs L19 DoM",
        "neuron_basis": neuron_basis,
        "n_examples": int(n),
        "n_neurons": int(d),
        "n_correct": int(y.sum()),
        "top_k": TOP_K,
        "auroc_insample_by_k": insample,
        "auroc_oof_by_k": oof,
        "best_oof_k": int(best_oof_k),
        "best_oof_auroc": best_oof,
        "f2_dom_auroc_reference": F2_DOM_AUROC,
        "dom_auroc_recomputed_on_cache": dom_auroc,
        "oof_gap_to_dom": float(F2_DOM_AUROC - best_oof),
        "sparse_mechanism_supported": bool(best_oof >= F2_DOM_AUROC - 0.02),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())