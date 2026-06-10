#!/usr/bin/env python3
"""FE269 Step 1 — compute the L19 prefill direction-of-means (DoM) for intervention.

The causal test (FE269) needs ONE fixed direction r_hat to ablate/add. Unlike the
OOF-AUROC estimate (phase2_prefill_dom.py, which 5-fold cross-validates to estimate
generalization), here we want the BEST single estimate of the direction itself, so
we use the FULL dataset — this is a fixed intervention hypothesis, not a
generalization measurement.

Source: pathway11_h100/prefill_inversion/cache/m15b_prefill.npz
  prefill (500,1536) f16  = L19 last-prefill-token activation (= hidden_states[19]
                            = output of model.model.layers[18])
  correct (500,) bool     = K=1 greedy MATH-500 correctness (243 True / 48.6%)

r     = mean(P[correct]) - mean(P[~correct])     # float32 arithmetic (f16 mean of
r_hat = r / ||r||                                #  500 vecs loses precision)

Output:
  r_hat.npy            float32 (1536,)  ~6KB — the direction to ship to the pod
  rhat_provenance.json ||r||, n_correct, cos(full-data, fold-0 OOF dom) sanity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
NPZ = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_DIR = ROOT / "pathway11_h100/causal_dom"
SEED, N_FOLDS = 9999, 5


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(NPZ, allow_pickle=True) as d:
        P = d["prefill"].astype(np.float32)   # (500,1536) upcast from f16
        y = d["correct"].astype(bool)         # (500,)

    n, dim = P.shape
    n_correct = int(y.sum())
    print(f"loaded {n} prefill vecs, dim={dim}, correct={n_correct}/{n} ({n_correct/n:.4f})")
    assert n_correct == 243, f"expected 243 correct, got {n_correct}"
    assert dim == 1536, f"expected dim 1536, got {dim}"

    # Full-data DoM (the intervention direction)
    r = P[y].mean(axis=0) - P[~y].mean(axis=0)
    r_norm = float(np.linalg.norm(r))
    assert r_norm > 0, "zero-norm DoM"
    r_hat = (r / r_norm).astype(np.float32)
    print(f"||r|| = {r_norm:.6f}   ||r_hat|| = {np.linalg.norm(r_hat):.6f}")

    # Sanity: full-data direction should align with a fold-0 OOF DoM (cos > 0.9).
    # Mirror phase2's protocol: mean-center on train, DoM = mu_c - mu_i, unit-norm.
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    tr, _ = next(iter(skf.split(P, y)))
    mu = P[tr].mean(axis=0)
    Pc = P[tr] - mu
    dom0 = Pc[y[tr]].mean(axis=0) - Pc[~y[tr]].mean(axis=0)
    dom0 /= (np.linalg.norm(dom0) + 1e-30)
    cos_full_fold0 = float(r_hat @ dom0)
    print(f"cos(full-data r_hat, fold-0 OOF dom) = {cos_full_fold0:.4f}")
    assert cos_full_fold0 > 0.9, f"full vs fold-0 cos too low: {cos_full_fold0:.4f}"

    np.save(OUT_DIR / "r_hat.npy", r_hat)
    prov = {
        "source_npz": str(NPZ.relative_to(ROOT)),
        "layer_semantics": "hidden_states[19] = output of model.model.layers[18]; hook layers[18] for 'L19'",
        "n": n, "n_correct": n_correct, "accuracy": n_correct / n,
        "dim": dim, "r_norm": r_norm,
        "cos_full_vs_fold0_oof": cos_full_fold0,
        "method": "full-data DoM = mean(prefill[correct]) - mean(prefill[~correct]), unit-normalized, float32",
        "seed": SEED,
    }
    (OUT_DIR / "rhat_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"saved {OUT_DIR/'r_hat.npy'} (float32, {dim}) and rhat_provenance.json")


if __name__ == "__main__":
    main()
