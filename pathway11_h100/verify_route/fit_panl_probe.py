#!/usr/bin/env python3
"""S4 — PANL activation probe (CPU).

5-fold OOF probe on the captured PANL activations (verify_acts.npz, N x 29 x 1536),
predicting (a) "K=1 is wrong" and (b) "B-bucket (recoverable) vs rest". Layer sweep
→ best OOF AUROC; also report L19. Two probe methods per layer:
  - DoM projection (matched to the prefill-DoM 0.7731 baseline's method)
  - L2 logistic regression on standardized features (the canonical "probe")

Compare against: verbalized-verdict accuracy (verify.json) and prefill-DoM AUROC (0.7731).
Kumaran predicts the PANL probe beats the verbalized verdict, especially at 1.5B.

Saves OOF routing scores (oriented higher = more likely WRONG) at the best "K=1-wrong"
layer and at L19, for use by route_sim.py.

Outputs: results/panl_probe.json, results/panl_oof.npz
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cache_utils as cu  # noqa: E402

SEED = 9999
N_FOLDS = 5


def oof_dom(H, y):
    """OOF DoM-projection scores oriented so higher = y==True. Returns (scores, auroc)."""
    scores = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(H, y):
        mu_pos = H[tr][y[tr]].mean(axis=0)
        mu_neg = H[tr][~y[tr]].mean(axis=0)
        v = mu_pos - mu_neg
        v /= (np.linalg.norm(v) + 1e-12)
        center = 0.5 * (mu_pos + mu_neg)
        scores[te] = (H[te] - center) @ v
    return scores, float(roc_auc_score(y, scores))


def oof_logreg(H, y):
    scores = np.zeros(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(H, y):
        sc = StandardScaler().fit(H[tr])
        clf = LogisticRegression(max_iter=2000, C=0.1).fit(sc.transform(H[tr]), y[tr])
        scores[te] = clf.predict_proba(sc.transform(H[te]))[:, 1]
    return scores, float(roc_auc_score(y, scores))


def main() -> int:
    d = np.load(HERE / "results" / "verify_acts.npz", allow_pickle=True)
    acts = d["acts"].astype(np.float32)  # (N, L+1, 1536)
    idx = d["idx"].astype(int)
    correct_k1 = d["correct_k1"].astype(bool)
    order = np.argsort(idx)
    acts, correct_k1 = acts[order], correct_k1[order]  # now in 0..499 order
    n, n_layer_slots, dim = acts.shape
    print(f"  acts {acts.shape} | K=1 acc {correct_k1.mean():.4f}")

    # targets
    y_wrong = ~correct_k1
    p1 = cu.load_phase1()
    _, greedy = cu.load_k1_greedy()
    assert np.array_equal(greedy, correct_k1), "k1_greedy.json vs verify_acts correctness mismatch"
    bmask = cu.buckets(greedy, p1["k8_majority_correct"])
    y_B = bmask["B_recoverable"]

    # verbalized verdict diagnostic
    vrows = json.loads((HERE / "results" / "verify.json").read_text())["rows"]
    vrows = sorted(vrows, key=lambda r: r["idx"])
    verdict_says_correct = np.array([r["verdict_bool"] for r in vrows], dtype=bool)
    # "predict wrong" = verdict says No (~says_correct); accuracy at catching K=1-wrong:
    verb_acc_vs_truth = float((verdict_says_correct == correct_k1).mean())
    # AUROC of the binary verdict as a wrong-detector
    verb_auroc_wrong = float(roc_auc_score(y_wrong, (~verdict_says_correct).astype(float))) \
        if y_wrong.any() and (~y_wrong).any() else float("nan")

    prefill_dom = cu.load_prefill_dom()  # higher = correct-like
    prefill_auroc_wrong = float(roc_auc_score(y_wrong, -prefill_dom))  # negate -> wrong-detector

    layers = []
    for L in range(n_layer_slots):
        H = acts[:, L, :]
        sd, ad = oof_dom(H, y_wrong)
        _, al = oof_logreg(H, y_wrong)
        # B-vs-rest (DoM only, small positive class)
        try:
            _, adB = oof_dom(H, y_B)
        except Exception:
            adB = float("nan")
        layers.append({"layer": L, "dom_auroc_wrong": ad, "logreg_auroc_wrong": al,
                       "dom_auroc_Bvsrest": adB})
        print(f"  L{L:2d}: DoM(wrong)={ad:.4f}  LR(wrong)={al:.4f}  DoM(B)={adB:.4f}", flush=True)

    best = max(layers, key=lambda r: r["dom_auroc_wrong"])
    bestL = best["layer"]
    # routing scores oriented higher = WRONG, at best layer and L19 (slot 19)
    s_best, _ = oof_dom(acts[:, bestL, :], y_wrong)
    s_l19, auroc_l19 = oof_dom(acts[:, 19, :], y_wrong)

    np.savez_compressed(
        HERE / "results" / "panl_oof.npz",
        panl_score_best=s_best.astype(np.float32),   # higher = more likely K=1-wrong
        panl_score_l19=s_l19.astype(np.float32),
        best_layer=bestL,
        correct_k1=correct_k1,
    )

    out = {
        "n": int(n), "n_layer_slots": int(n_layer_slots),
        "best_layer_for_wrong": int(bestL),
        "best_dom_auroc_wrong": best["dom_auroc_wrong"],
        "best_logreg_auroc_wrong": best["logreg_auroc_wrong"],
        "L19_dom_auroc_wrong": float(auroc_l19),
        "per_layer": layers,
        "comparators": {
            "prefill_DoM_auroc_wrong": prefill_auroc_wrong,
            "prefill_DoM_auroc_correct_published": 0.7731,
            "verbalized_verdict_accuracy_vs_truth": verb_acc_vs_truth,
            "verbalized_verdict_auroc_wrong": verb_auroc_wrong,
            "verbalized_says_correct_rate": float(verdict_says_correct.mean()),
        },
    }
    (HERE / "results" / "panl_probe.json").write_text(json.dumps(out, indent=1))
    print(f"\n  BEST PANL layer for K=1-wrong: L{bestL}  DoM AUROC={best['dom_auroc_wrong']:.4f} "
          f"(LR {best['logreg_auroc_wrong']:.4f})")
    print(f"  PANL L19 DoM AUROC(wrong) = {auroc_l19:.4f}")
    print(f"  comparator — prefill-DoM AUROC(wrong) = {prefill_auroc_wrong:.4f}; "
          f"verbalized verdict acc = {verb_acc_vs_truth:.4f}, AUROC(wrong) = {verb_auroc_wrong:.4f}")
    print("  wrote results/panl_probe.json + results/panl_oof.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
