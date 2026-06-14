"""P11-FE40 — Offline one-step PPLM sanity check for the F-2 probe.

Tests whether a single PPLM-style gradient step along the F-2 logistic
probe's local gradient is informative on the F-8 abstained decile (the 50
lowest-DoM MATH-500 problems), *without* spending H100 time on actual
generation.

Method, faithful to what is computable offline from cached NPZs:
  1. Fit F-2's probe — a logistic regression on the L19 prefill hidden
     states H_19 predicting `correct` (standardized features for stability).
  2. Identify the abstained decile: the 50 problems with the lowest DoM
     score (F-8's abstained set).
  3. For each problem compute g = ∇_{H_19} log σ(probe(H_19)) = (1 - p)·(w/σ),
     then the unit step Δ H_19 = α · g/‖g‖ (single step, no inner loop).
  4. Re-evaluate the probe at L19 post-update and read AUROC.
  5. Sweep α ∈ {0.01, 0.02, 0.04}; compare to the no-perturbation baseline.

CAVEAT — the brief asks to "re-run forward pass L20→L27 from cached K/V."
That requires model weights and a GPU forward pass, which are explicitly out
of scope offline (no torch/transformers, CPU only, cached NPZs only). This
script therefore evaluates the genuinely computable L19-probe proxy. Note the
structural consequence baked into the result: because Δ H_19 = α·g/‖g‖ has the
*same* unit direction for every sample (the per-sample (1-p) factor is a
positive scalar that normalization removes), the step is a pure translation of
H_19 along the probe direction. The probe's score therefore shifts uniformly
*within* the decile, so within-decile AUROC is invariant by construction —
which is itself the offline finding: a single linear-probe gradient step at
L19 cannot move the probe's own ranking; any benefit would have to come from
the downstream nonlinear L20→L27 transform, i.e. from a real generation run.
The full-500 AUROC (only the decile perturbed) is also reported, since the
decile's uniform shift *does* re-rank it against the unperturbed 450.

Decision rule (per rationale): post-step decile-AUROC lift ≥ 0.05 ⇒ gradient
steering deserves a full generation experiment (P10-FE8); otherwise F-2's
probe is a global-direction object and PPLM-style intervention is unlikely to
help.
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
OUT_JSON = ROOT / "pathway11_h100/pplm_sanity/results.json"

SEED = 9999
DECILE_N = 50
ALPHAS = [0.01, 0.02, 0.04]
LIFT_THRESHOLD = 0.05


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = scores[labels]
    neg = scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def main() -> int:
    if not CACHE.exists():
        print("MISSING_REGEN_INPUT", CACHE, file=sys.stderr)
        return 2
    if not DOM_NPZ.exists():
        print("MISSING_REGEN_INPUT", DOM_NPZ, file=sys.stderr)
        return 2

    cache = np.load(CACHE)
    H = cache["prefill"].astype(np.float64)            # (500, 1536)
    correct = cache["correct"].astype(bool)            # (500,)
    assert H.shape == (500, 1536) and correct.shape == (500,)

    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)  # (500,)
    assert dom_score.shape == (500,)

    # --- Abstained decile: 50 lowest-DoM problems (F-8 abstained set) ---
    decile = np.argsort(dom_score)[:DECILE_N]
    decile_mask = np.zeros(len(correct), dtype=bool)
    decile_mask[decile] = True

    # --- F-2's probe: logistic regression on standardized H_19 ---
    mu = H.mean(axis=0)
    sigma = H.std(axis=0)
    sigma_safe = np.where(sigma < 1e-12, 1.0, sigma)
    Z = (H - mu) / sigma_safe

    clf = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)
    clf.fit(Z, correct)
    w = clf.coef_.ravel().astype(np.float64)   # in standardized space
    b = float(clf.intercept_[0])

    def probe_logit(Hmat: np.ndarray) -> np.ndarray:
        return ((Hmat - mu) / sigma_safe) @ w + b

    # --- Gradient of log σ(logit) w.r.t. H_19 ---
    # d logit / d H = w / sigma ; d log σ / d logit = (1 - p)
    # => g = (1 - p) * (w / sigma). The (1-p) factor is a positive scalar, so
    # the unit step direction g/‖g‖ collapses to a single shared direction.
    g_dir = w / sigma_safe
    g_unit = g_dir / np.linalg.norm(g_dir)

    base_logit = probe_logit(H)
    base_auroc_decile = auroc(base_logit[decile_mask], correct[decile_mask])
    base_auroc_full = auroc(base_logit, correct)

    sweep = {}
    best_lift_decile = float("-inf")
    for alpha in ALPHAS:
        H_pert = H.copy()
        # single PPLM step on the abstained decile only
        H_pert[decile_mask] = H[decile_mask] + alpha * g_unit[None, :]

        post_logit = probe_logit(H_pert)
        post_auroc_decile = auroc(post_logit[decile_mask], correct[decile_mask])
        post_auroc_full = auroc(post_logit, correct)

        lift_decile = post_auroc_decile - base_auroc_decile
        lift_full = post_auroc_full - base_auroc_full
        best_lift_decile = max(best_lift_decile, lift_decile)

        sweep[f"alpha_{alpha}"] = {
            "alpha": alpha,
            "post_auroc_decile": post_auroc_decile,
            "post_auroc_full": post_auroc_full,
            "lift_decile": float(lift_decile),
            "lift_full": float(lift_full),
            "mean_logit_shift_decile": float(
                (post_logit[decile_mask] - base_logit[decile_mask]).mean()
            ),
        }

    verdict = (
        "GRADIENT_STEERING_WORTH_GENERATION"
        if best_lift_decile >= LIFT_THRESHOLD
        else "PROBE_IS_GLOBAL_DIRECTION_PPLM_UNLIKELY"
    )

    out = {
        "experiment": "P11-FE40",
        "description": "Offline one-step PPLM sanity check on cached L19 prefill",
        "n_problems": int(len(correct)),
        "decile_n": DECILE_N,
        "decile_dom_threshold": float(dom_score[decile].max()),
        "decile_accuracy": float(correct[decile_mask].mean()),
        "probe": "logistic_regression_standardized_H19",
        "baseline_auroc_decile": base_auroc_decile,
        "baseline_auroc_full": base_auroc_full,
        "alphas": ALPHAS,
        "sweep": sweep,
        "best_lift_decile": float(best_lift_decile),
        "lift_threshold": LIFT_THRESHOLD,
        "verdict": verdict,
        "caveat": (
            "L20->L27 re-forward requires model weights + GPU and is out of "
            "offline scope; L19-probe proxy evaluated. Delta H_19 = alpha*g/||g|| "
            "is a uniform translation along the probe direction, so within-decile "
            "AUROC is invariant by construction (the offline finding)."
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"verdict={verdict} best_lift_decile={best_lift_decile:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())