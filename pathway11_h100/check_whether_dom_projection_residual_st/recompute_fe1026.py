"""P11-FE1026 — Does DoM survive partialing out logit-lens calibration features?

Disambiguates the copy-suppression account of F-2: is the L19 prefill DoM
projection a genuine correctness-geometry signal, or merely a calibration proxy
for logit-lens entropy / top-token confidence?

Procedure (5-fold OOF, leak-free):
  1. Raw DoM AUROC (reference).
  2. AUROC of the calibration features (entropy, top-token confidence) alone.
  3. Residualize DoM on [entropy, top_token_conf, 1] with per-fold OLS fit on
     the train split, subtract the prediction on the held-out test split, score
     the OOF residual.
If the residual AUROC stays near the raw DoM AUROC, DoM carries information
beyond calibration. If it collapses to ~0.5, copy suppression fully explains F-2.

Logit-lens features require the LM-unembedding projection, which is not derivable
on local CPU from the cached L19 hidden states alone; they are expected as a
precomputed cache. If that cache is absent the script reports MISSING_REGEN_INPUT.
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

ROOT = Path("/home/musicofhel/topo-confidence")
CACHE = ROOT / "pathway11_h100/prefill_inversion/cache/m15b_prefill.npz"
DOM_NPZ = ROOT / "pathway11_h100/prefill_gated_compute/phase2_prefill_dom.npz"
LOGIT_LENS_NPZ = ROOT / "pathway11_h100/logit_lens/cache/m15b_logitlens.npz"
OUT_JSON = ROOT / "pathway11_h100/logit_lens_partial/results.json"

SEED = 9999
N_FOLDS = 5

ENTROPY_KEYS = ("entropy", "ll_entropy", "logit_lens_entropy", "prefill_entropy")
TOPCONF_KEYS = ("top_token_conf", "top_token_confidence", "ll_top_conf",
                "topconf", "max_prob", "top1_prob")


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels.astype(bool)]
    neg = scores[~labels.astype(bool)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (len(pos) * len(neg)))


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [test_idx for _, test_idx in skf.split(np.zeros(len(y)), y)]


def _first_key(blob, keys, label):
    for key in keys:
        if key in blob:
            return blob[key].astype(np.float64)
    print(f"MISSING_REGEN_INPUT {LOGIT_LENS_NPZ} (no {label} key in {list(blob.keys())})",
          file=sys.stderr)
    return None


def main() -> int:
    for path in (CACHE, DOM_NPZ, LOGIT_LENS_NPZ):
        if not path.exists():
            print("MISSING_REGEN_INPUT", path, file=sys.stderr)
            return 2

    cache = np.load(CACHE)
    correct = cache["correct"].astype(bool)
    dom_score = np.load(DOM_NPZ)["prefill_score"].astype(np.float64)
    assert correct.shape == (500,) and dom_score.shape == (500,)

    ll = np.load(LOGIT_LENS_NPZ)
    entropy = _first_key(ll, ENTROPY_KEYS, "entropy")
    top_conf = _first_key(ll, TOPCONF_KEYS, "top-token confidence")
    if entropy is None or top_conf is None:
        return 2
    if entropy.shape != (500,) or top_conf.shape != (500,):
        print("MISSING_REGEN_INPUT", LOGIT_LENS_NPZ,
              f"(shape {entropy.shape}/{top_conf.shape} != (500,))", file=sys.stderr)
        return 2

    n = len(correct)
    oof_residual = np.zeros(n, dtype=np.float64)
    folds = stratified_kfold(correct, N_FOLDS, SEED)
    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        feats_tr = np.column_stack([entropy[train_mask], top_conf[train_mask],
                                    np.ones(int(train_mask.sum()))])
        coeffs, _, _, _ = np.linalg.lstsq(feats_tr, dom_score[train_mask], rcond=None)
        feats_te = np.column_stack([entropy[test_idx], top_conf[test_idx],
                                    np.ones(len(test_idx))])
        oof_residual[test_idx] = dom_score[test_idx] - feats_te @ coeffs

    out = {
        "experiment": "P11-FE1026",
        "n": int(n),
        "auroc_dom_raw": float(auroc(dom_score, correct)),
        "auroc_entropy_alone": float(auroc(-entropy, correct)),
        "auroc_top_conf_alone": float(auroc(top_conf, correct)),
        "auroc_dom_residualized": float(auroc(oof_residual, correct)),
        "calibration_features": ["logit_lens_entropy", "top_token_confidence"],
        "interpretation": (
            "residual AUROC near raw DoM AUROC => DoM carries information beyond "
            "calibration; residual AUROC ~0.5 => copy-suppression calibration "
            "fully explains F-2"),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())