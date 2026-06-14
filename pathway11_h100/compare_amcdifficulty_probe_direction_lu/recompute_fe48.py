"""P10-FE48 — AMC-difficulty probe direction vs prefill / final correctness DoM.

Lugoloobi & Russell (2510.18147) report a continuous AMC-difficulty probe
direction (ρ≈0.88 with IRT difficulty). This experiment asks whether that
continuous-difficulty direction is the upstream representation our *binary*
correctness DoM is merely thresholding.

Procedure (all on cached L19 prefill activations, CPU only):
  1. prefill_DoM_L19  = mean(prefill | correct) - mean(prefill | wrong) on the
     500 MATH-500 problems (m15b_prefill.npz).
  2. final_DoM_L19    = same construction on final-token L19 states
     (m15b_final.npz).
  3. amc_dir          = ridge probe trained on Easy2HardBench AMC IRT difficulty
     labels using cached AMC prefill L19 activations (amc_prefill.npz +
     amc_irt_labels.json). OOF coefficient averaging for stability.
  4. Report cos(amc_dir, prefill_DoM_L19) and cos(amc_dir, final_DoM_L19).

Interpretation of F-3 (prefill/final orthogonality, cos≈0.046):
  - high |cos| with BOTH directions  -> F-3 is a labeling artifact (binary
    correctness vs continuous difficulty pick out the same axis).
  - high |cos| with only ONE          -> F-3 is robust; the orthogonality is
    not explained by a shared difficulty axis.

Every cached input is required; if any is absent (notably the AMC label/prefill
caches, which need a one-time Easy2HardBench download + extract) the script
prints MISSING_REGEN_INPUT and returns 2 — no network access is performed here.
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
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

ROOT = Path("/home/musicofhel/topo-confidence")

# --- inputs ---
MATH_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
FINAL_CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_final.npz"
AMC_PREFILL = ROOT / "pathway11_h100/amc_difficulty_probe/cache/amc_prefill.npz"
AMC_LABELS = ROOT / "pathway11_h100/amc_difficulty_probe/cache/amc_irt_labels.json"

# --- output ---
OUT_JSON = ROOT / "pathway11_h100/amc_difficulty_probe/results.json"

SEED = 9999
N_FOLDS = 5
RIDGE_ALPHAS = (1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)
HIGH_COS = 0.50  # |cos| threshold for "shares the axis"


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(unit(a) @ unit(b))


def dom_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return X[y].mean(axis=0) - X[~y].mean(axis=0)


def fit_amc_dir(X: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, float]:
    """OOF-averaged ridge coefficient direction + mean OOF Spearman-ish R."""
    n = X.shape[0]
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    dc = d - d.mean()

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    coefs = np.zeros(X.shape[1], dtype=np.float64)
    oof_pred = np.zeros(n, dtype=np.float64)
    for tr, te in kf.split(Xs):
        model = RidgeCV(alphas=RIDGE_ALPHAS)
        model.fit(Xs[tr], dc[tr])
        coefs += model.coef_
        oof_pred[te] = model.predict(Xs[te])
    coefs /= N_FOLDS
    # de-standardize the direction back into raw activation space so cosines are
    # comparable with the DoM directions (which live in raw space).
    raw_dir = coefs / sd
    if dc.std() > 0 and oof_pred.std() > 0:
        r = float(np.corrcoef(oof_pred, dc)[0, 1])
    else:
        r = float("nan")
    return raw_dir, r


def load_amc():
    blob = np.load(AMC_PREFILL, allow_pickle=True)
    if "prefill" not in blob:
        raise KeyError("amc_prefill.npz missing 'prefill' array")
    X = blob["prefill"].astype(np.float64)
    labels = json.loads(AMC_LABELS.read_text())
    # labels: {problem_id: irt_difficulty}; align via 'problem_id' if present,
    # else assume row order matches the label list order.
    if "problem_id" in blob:
        pids = [str(p) for p in blob["problem_id"]]
        try:
            d = np.array([float(labels[p]) for p in pids], dtype=np.float64)
        except KeyError as e:
            raise KeyError(f"AMC label missing for problem_id {e}") from e
    else:
        vals = list(labels.values()) if isinstance(labels, dict) else list(labels)
        if len(vals) != X.shape[0]:
            raise ValueError("AMC label count != prefill rows and no problem_id")
        d = np.array([float(v) for v in vals], dtype=np.float64)
    return X, d


def main() -> int:
    required = [MATH_CACHE, FINAL_CACHE, AMC_PREFILL, AMC_LABELS]
    missing = [p for p in required if not p.exists()]
    if missing:
        for p in missing:
            print("MISSING_REGEN_INPUT", p, file=sys.stderr)
        return 2

    # MATH-500 prefill DoM (L19)
    mblob = np.load(MATH_CACHE)
    Xp = mblob["prefill"].astype(np.float64)
    y = mblob["correct"].astype(bool)
    if Xp.shape != (500, 1536) or y.shape != (500,):
        print("BAD_SHAPE prefill", Xp.shape, y.shape, file=sys.stderr)
        return 3
    prefill_dom = dom_direction(Xp, y)

    # Final-token DoM (L19)
    fblob = np.load(FINAL_CACHE)
    fkey = "final" if "final" in fblob else ("prefill" if "prefill" in fblob else None)
    if fkey is None:
        print("MISSING_REGEN_INPUT", FINAL_CACHE, "(no 'final' array)", file=sys.stderr)
        return 2
    Xf = fblob[fkey].astype(np.float64)
    if Xf.shape != (500, 1536):
        print("BAD_SHAPE final", Xf.shape, file=sys.stderr)
        return 3
    final_dom = dom_direction(Xf, y)

    # AMC difficulty probe direction
    try:
        X_amc, d_amc = load_amc()
    except (KeyError, ValueError) as e:
        print("MISSING_REGEN_INPUT", AMC_PREFILL, str(e), file=sys.stderr)
        return 2
    if X_amc.shape[1] != 1536:
        print("BAD_SHAPE amc", X_amc.shape, file=sys.stderr)
        return 3
    amc_dir, amc_oof_r = fit_amc_dir(X_amc, d_amc)

    cos_amc_prefill = cos(amc_dir, prefill_dom)
    cos_amc_final = cos(amc_dir, final_dom)
    cos_prefill_final = cos(prefill_dom, final_dom)

    hi_prefill = abs(cos_amc_prefill) >= HIGH_COS
    hi_final = abs(cos_amc_final) >= HIGH_COS
    if hi_prefill and hi_final:
        verdict = "F3_LABELING_ARTIFACT"
    elif hi_prefill ^ hi_final:
        verdict = "F3_ROBUST"
    else:
        verdict = "INCONCLUSIVE_amc_dir_unrelated_to_both"

    out = {
        "experiment": "P10-FE48",
        "trigger_paper": "2510.18147",
        "n_amc": int(X_amc.shape[0]),
        "amc_probe_oof_pearson_r": amc_oof_r,
        "auroc_prefill_dom": auroc(Xp @ prefill_dom, y),
        "auroc_final_dom": auroc(Xf @ final_dom, y),
        "cos_amc_prefill_dom": cos_amc_prefill,
        "cos_amc_final_dom": cos_amc_final,
        "cos_prefill_final_dom": cos_prefill_final,
        "high_cos_threshold": HIGH_COS,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())