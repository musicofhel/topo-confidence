"""FE448 — Length partial-correlation control on F-2.

Direct refutation: if the prefill L19 DoM AUROC (0.7731) is mostly explained
by length information leaking into the prefill activations, residualizing
prefill against predicted-length should collapse the AUROC toward 0.5.

Method:
  1. OLS regress seq_len on prefill (1536-dim) with small ridge for stability.
  2. Predicted length ŝ = X β; report R² (length-predictability from L19).
  3. Residualize each prefill column against ŝ (Frisch-Waugh per-feature).
  4. Recompute DoM AUROC on residualized prefill.
  5. Compare to raw in-sample DoM AUROC (apples-to-apples on the same data).

Inputs (cached, deterministic):
  - pathway11_h100/prefill_inversion/cache/m15b_prefill.npz

Output: pathway11_h100/length_partial/results.json
Stdout: key=value lines for Tier-1 regen.

Refutation threshold (per FE448 spec): residual AUROC ≤ 0.55 ⇒ F-2 is a
length artifact. Residual AUROC > 0.65 ⇒ F-2 survives length partialing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
OUT_JSON = ROOT / "pathway11_h100/length_partial/results.json"

RIDGE = 1e-3  # numerical stability for n=500, d=1536 (n < d)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression. Returns coefficient vector β (d,)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    # Use the n×n form when n < d (faster, numerically nicer):
    # β = Xᵀ (X Xᵀ + αI)⁻¹ y
    n = Xc.shape[0]
    K = Xc @ Xc.T + alpha * np.eye(n, dtype=Xc.dtype)
    a = np.linalg.solve(K, yc)
    beta = Xc.T @ a
    return beta


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    d = np.load(CACHE)
    X = d["prefill"].astype(np.float64)  # (500, 1536)
    y = d["correct"].astype(bool)
    s = d["seq_len"].astype(np.float64)
    assert X.shape == (500, 1536)
    assert y.shape == (500,)
    assert s.shape == (500,)

    # 1. Ridge regression seq_len ~ prefill
    beta = ridge_fit(X, s, RIDGE)

    # 2. Predicted length (uncentered: add the means back so ŝ has the right mean)
    s_hat = (X - X.mean(axis=0, keepdims=True)) @ beta + s.mean()

    # 3. R² of s vs s_hat
    ss_res = float(np.sum((s - s_hat) ** 2))
    ss_tot = float(np.sum((s - s.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    pearson_s_shat = float(np.corrcoef(s, s_hat)[0, 1])

    # 4. Residualize each prefill column against ŝ (Frisch-Waugh).
    # X_resid[:, j] = X[:, j] - alpha_j * ŝ where alpha_j minimizes ‖X[:,j] - alpha_j ŝ‖
    s_hat_centered = s_hat - s_hat.mean()
    denom = float(s_hat_centered @ s_hat_centered) + 1e-12
    Xc = X - X.mean(axis=0, keepdims=True)
    alpha = (Xc.T @ s_hat_centered) / denom  # (d,)
    X_resid = Xc - np.outer(s_hat_centered, alpha)
    X_resid = X_resid + X.mean(axis=0, keepdims=True)  # restore mean for DoM

    # 5. DoM AUROC, raw + residualized (in-sample, same n=500)
    def dom_auroc(features: np.ndarray) -> tuple[float, np.ndarray]:
        mu_pos = features[y].mean(axis=0)
        mu_neg = features[~y].mean(axis=0)
        dom = mu_pos - mu_neg
        scores = features @ dom
        return auroc(scores, y), dom

    auroc_raw, dom_raw = dom_auroc(X)
    auroc_resid, dom_resid = dom_auroc(X_resid)

    # Cosine between raw DoM and residualized DoM (sanity)
    cos_doms = float(
        (dom_raw @ dom_resid)
        / (np.linalg.norm(dom_raw) * np.linalg.norm(dom_resid) + 1e-12)
    )

    # Length AUROC for context (AUROC of seq_len alone vs correct, both signs)
    auroc_length = max(auroc(s, y), auroc(-s, y))

    # AUROC of predicted-length alone vs correct (does the length-predictive
    # subspace itself carry the correctness signal?)
    auroc_s_hat = max(auroc(s_hat, y), auroc(-s_hat, y))

    out = {
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "ridge_alpha": RIDGE,
        "length_r2_from_prefill": r2,
        "length_pearson_s_shat": pearson_s_shat,
        "auroc_raw_in_sample": auroc_raw,
        "auroc_residualized": auroc_resid,
        "auroc_drop": auroc_raw - auroc_resid,
        "cos_dom_raw_vs_resid": cos_doms,
        "auroc_length_alone": auroc_length,
        "auroc_predicted_length_alone": auroc_s_hat,
        "f2_length_artifact_threshold": 0.55,
        "f2_length_artifact": auroc_resid <= 0.55,
        "f2_survives_partialing_threshold": 0.65,
        "f2_survives_partialing": auroc_resid > 0.65,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print(f"length_r2_from_prefill={out['length_r2_from_prefill']:.10f}")
    print(f"length_pearson_s_shat={out['length_pearson_s_shat']:.10f}")
    print(f"auroc_raw_in_sample={out['auroc_raw_in_sample']:.10f}")
    print(f"auroc_residualized={out['auroc_residualized']:.10f}")
    print(f"auroc_drop={out['auroc_drop']:.10f}")
    print(f"cos_dom_raw_vs_resid={out['cos_dom_raw_vs_resid']:.10f}")
    print(f"auroc_length_alone={out['auroc_length_alone']:.10f}")
    print(f"auroc_predicted_length_alone={out['auroc_predicted_length_alone']:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
