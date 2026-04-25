"""
Pathway 10 v2 — E3 risk-coverage / refusal policy.

Q1: L19 DoM (1.5B) as refusal signal. Sweep tau from 10th to 90th percentile.
Q2: Baselines — random, sequence-length (refuse longest), max-token-prob.
Q3: 7B L15 DoM as refusal signal for 1.5B correctness.

Split: stratified 80/20 on NEW labels, seed 9999 (same as prior sessions).
Primary eval: holdout (100 problems); percentile grid from holdout-sorted scores.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr  # noqa: F401  (kept for parity with prior scratch)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path("/home/musicofhel/topo-confidence")
SCRATCH = ROOT / "scratch"
SCRATCH.mkdir(exist_ok=True)

# ---- Load ----------------------------------------------------------------
ls_15b = np.load(ROOT / "pathway2/track_a/phase0/layer_states.npy")  # (500, 29, 1536)
ls_7b = np.load(ROOT / "pathway6_rebuild/phase3_cross_benchmark/math7b/layer_states.npy")  # (500, 29, 3584)
y_new = np.load(ROOT / "pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy").astype(int)
y_7b = np.load(ROOT / "pathway6_rebuild/phase3_cross_benchmark/math7b/baseline_correct.npy").astype(int)
traj_npz = np.load(ROOT / "data/experiment1_v2/trajectories.npz")
with open(ROOT / "data/experiment1_v2/output_entropy_scores.json") as f:
    entropy_json = json.load(f)

n = len(y_new)
print(f"1.5B states: {ls_15b.shape}, NEW labels: {int(y_new.sum())}/{n}")
print(f"7B   states: {ls_7b.shape},   7B labels: {int(y_7b.sum())}/{n}")

# ---- Split ---------------------------------------------------------------
idx = np.arange(n)
i_tr, i_ho = train_test_split(idx, test_size=0.2, random_state=9999, stratify=y_new)
y_hold = y_new[i_ho]
hold_base_rate = y_hold.mean()
print(
    f"Split: train={len(i_tr)} pos={int(y_new[i_tr].sum())}, "
    f"hold={len(i_ho)} pos={int(y_hold.sum())}, hold base rate={hold_base_rate:.3f}"
)

# ---- Seq-length and max-token-prob arrays --------------------------------
seq_lens = np.array([traj_npz[f"traj_{i}"].shape[0] for i in range(n)], dtype=float)

scores_list = entropy_json["scores"]
assert len(scores_list) == n, "output_entropy_scores must have n_problems=500"
max_tok_prob = np.array([r["max_token_prob"] for r in scores_list], dtype=float)
entropy_vec = np.array([r["entropy"] for r in scores_list], dtype=float)
gen_len_et = np.array([r["gen_len"] for r in scores_list], dtype=float)
print(
    f"entropy_json: max_new_tokens={entropy_json['max_new_tokens']}, "
    f"mean gen_len={gen_len_et.mean():.1f}  "
    f"(trajectories seq_len: mean={seq_lens.mean():.1f}, median={np.median(seq_lens):.1f}, max={seq_lens.max():.0f})"
)


# ---- Helpers -------------------------------------------------------------
def dom(X_tr, y_tr):
    mu1 = X_tr[y_tr == 1].mean(axis=0)
    mu0 = X_tr[y_tr == 0].mean(axis=0)
    w = mu1 - mu0
    return w / (np.linalg.norm(w) + 1e-12)


def risk_coverage(scores_ho, y_ho, cov_grid):
    """Higher score = higher confidence. At coverage c, answer top c fraction."""
    order = np.argsort(-scores_ho)  # desc
    y_sorted = y_ho[order]
    rows = []
    n_ho = len(y_ho)
    for c in cov_grid:
        k = max(1, int(round(c * n_ho)))
        ans = y_sorted[:k]
        ref = y_sorted[k:]
        rows.append({
            "coverage": float(c),
            "k_answered": int(k),
            "k_refused": int(n_ho - k),
            "acc_answered": float(ans.mean()),
            "acc_refused": float(ref.mean()) if len(ref) else float("nan"),
        })
    return rows


def bootstrap_acc_at_cov(scores_ho, y_ho, cov, n_boot=1000, seed=9999):
    rng = np.random.default_rng(seed)
    n_ho = len(y_ho)
    k = max(1, int(round(cov * n_ho)))
    accs = np.empty(n_boot)
    for b in range(n_boot):
        bi = rng.integers(0, n_ho, size=n_ho)
        s = scores_ho[bi]
        y = y_ho[bi]
        order = np.argsort(-s)
        accs[b] = y[order][:k].mean()
    return float(np.percentile(accs, 2.5)), float(np.median(accs)), float(np.percentile(accs, 97.5))


def aurc(scores_ho, y_ho):
    """Area Under Risk-Coverage curve (risk=1-accuracy). Lower is better."""
    order = np.argsort(-scores_ho)
    y_sorted = y_ho[order]
    cum_correct = np.cumsum(y_sorted)
    ks = np.arange(1, len(y_ho) + 1)
    acc_at_k = cum_correct / ks
    risk_at_k = 1 - acc_at_k
    return float(risk_at_k.mean())


# ---- Random baseline with bootstrap expectation --------------------------
def random_curve(y_ho, cov_grid, n_boot=1000, seed=9999):
    rng = np.random.default_rng(seed)
    n_ho = len(y_ho)
    rows = []
    for c in cov_grid:
        k = max(1, int(round(c * n_ho)))
        accs = np.empty(n_boot)
        for b in range(n_boot):
            perm = rng.permutation(n_ho)
            accs[b] = y_ho[perm][:k].mean()
        rows.append({
            "coverage": float(c),
            "k_answered": int(k),
            "k_refused": int(n_ho - k),
            "acc_answered_mean": float(accs.mean()),
            "acc_answered_2_5": float(np.percentile(accs, 2.5)),
            "acc_answered_97_5": float(np.percentile(accs, 97.5)),
        })
    return rows


# ---- Coverage grid -------------------------------------------------------
# User: tau percentiles 10..90 step 5 => coverage 0.90..0.10 step 0.05 (17 pts).
cov_grid = np.round(np.arange(0.10, 0.91, 0.05), 2)[::-1]  # 0.90 down to 0.10
cov_grid = np.sort(cov_grid)

# ---- Q1: L19 DoM (1.5B) --------------------------------------------------
L15 = 19
X15 = ls_15b[:, L15, :]
w15 = dom(X15[i_tr], y_new[i_tr])
scores_15_all = X15 @ w15
scores_15_ho = scores_15_all[i_ho]
auroc_15 = roc_auc_score(y_hold, scores_15_ho)
aurc_15 = aurc(scores_15_ho, y_hold)
q1_curve = risk_coverage(scores_15_ho, y_hold, cov_grid)

# ---- Q2a Random ---------------------------------------------------------
q2a_curve = random_curve(y_hold, cov_grid, n_boot=1000, seed=9999)

# ---- Q2b Length (refuse longest => score = -seq_len) --------------------
scores_len_ho = -seq_lens[i_ho]
auroc_len = roc_auc_score(y_hold, scores_len_ho)
aurc_len = aurc(scores_len_ho, y_hold)
q2b_curve = risk_coverage(scores_len_ho, y_hold, cov_grid)

# ---- Q2c Max token prob (logprob surrogate, different run caveat) -------
scores_mtp_ho = max_tok_prob[i_ho]
auroc_mtp = roc_auc_score(y_hold, scores_mtp_ho)
aurc_mtp = aurc(scores_mtp_ho, y_hold)
q2c_curve = risk_coverage(scores_mtp_ho, y_hold, cov_grid)

# negative entropy as a second confidence signal
scores_negH_ho = -entropy_vec[i_ho]
auroc_negH = roc_auc_score(y_hold, scores_negH_ho)
aurc_negH = aurc(scores_negH_ho, y_hold)
q2d_curve = risk_coverage(scores_negH_ho, y_hold, cov_grid)

# ---- Q3: 7B L15 DoM trained on 7B labels -------------------------------
L7 = 15
X7 = ls_7b[:, L7, :]
w7 = dom(X7[i_tr], y_7b[i_tr])
scores_7_all = X7 @ w7
scores_7_ho = scores_7_all[i_ho]
auroc_7 = roc_auc_score(y_hold, scores_7_ho)
aurc_7 = aurc(scores_7_ho, y_hold)
q3_curve = risk_coverage(scores_7_ho, y_hold, cov_grid)


# ---- Print tables -------------------------------------------------------
def print_curve(name, curve, auroc=None, aurc_val=None):
    head = f"\n=== {name} ==="
    if auroc is not None:
        head += f"  (AUROC={auroc:.3f}, AURC={aurc_val:.3f})"
    print(head)
    print(f"{'cov':>5} {'k_ans':>6} {'k_ref':>6} {'acc_ans':>9} {'acc_ref':>9}")
    for r in curve:
        ar = r.get("acc_answered", r.get("acc_answered_mean"))
        aref = r.get("acc_refused", float("nan"))
        print(
            f"{r['coverage']:>5.2f} {r['k_answered']:>6d} {r['k_refused']:>6d} "
            f"{ar:>9.3f} {aref:>9.3f}"
        )


print_curve("Q1: 1.5B L19 DoM", q1_curve, auroc_15, aurc_15)
print_curve("Q2a: Random (bootstrap mean)", q2a_curve)
print_curve("Q2b: Length (refuse longest)", q2b_curve, auroc_len, aurc_len)
print_curve("Q2c: max_token_prob (256-tok run caveat)", q2c_curve, auroc_mtp, aurc_mtp)
print_curve("Q2d: -entropy (same caveat)", q2d_curve, auroc_negH, aurc_negH)
print_curve("Q3: 7B L15 DoM (trained on 7B labels)", q3_curve, auroc_7, aurc_7)


# ---- Bootstrap CIs at canonical coverage points -------------------------
print("\n=== Bootstrap 95% CIs on acc_answered ===")
print(f"{'method':>25} {'cov':>5} {'n_ans':>6} {'lo':>6} {'med':>6} {'hi':>6}")
for cov in [0.25, 0.50, 0.75]:
    k = max(1, int(round(cov * len(y_hold))))
    for name, s in [
        ("1.5B L19 DoM", scores_15_ho),
        ("7B L15 DoM", scores_7_ho),
        ("-length", scores_len_ho),
        ("max_token_prob", scores_mtp_ho),
        ("-entropy", scores_negH_ho),
    ]:
        lo, med, hi = bootstrap_acc_at_cov(s, y_hold, cov, n_boot=1000, seed=9999)
        print(f"{name:>25} {cov:>5.2f} {k:>6d} {lo:>6.3f} {med:>6.3f} {hi:>6.3f}")


# ---- Summary at operating points ----------------------------------------
def at_cov(curve, cov):
    for r in curve:
        if abs(r["coverage"] - cov) < 1e-6:
            return r.get("acc_answered", r.get("acc_answered_mean"))
    return None


print("\n=== Head-to-head accuracy on answered subset ===")
print(
    f"{'cov':>5} {'1.5B L19 DoM':>14} {'7B L15 DoM':>12} {'max_tok_p':>10} "
    f"{'-entropy':>10} {'-length':>10} {'random':>10}"
)
for cov in [0.25, 0.50, 0.75]:
    vals = [
        at_cov(q1_curve, cov), at_cov(q3_curve, cov),
        at_cov(q2c_curve, cov), at_cov(q2d_curve, cov),
        at_cov(q2b_curve, cov), at_cov(q2a_curve, cov),
    ]
    print(
        f"{cov:>5.2f} "
        + "  ".join(f"{v:>10.3f}" if v is not None else f"{'NA':>10}" for v in vals)
    )

print(f"\nHoldout unconditional accuracy = {hold_base_rate:.1%}")
print("v2 success targets: >=40% at 50% coverage, >=60% at 25% coverage")


# ---- Save ----------------------------------------------------------------
out = {
    "meta": {
        "seed": 9999,
        "test_size": 0.2,
        "n_hold": int(len(i_ho)),
        "n_hold_pos": int(y_hold.sum()),
        "hold_base_rate": float(hold_base_rate),
        "labels": "NEW (baseline_correct_v2)",
        "coverage_grid": [float(c) for c in cov_grid],
    },
    "Q1_1_5B_L19_DoM": {
        "layer": L15,
        "auroc_holdout": float(auroc_15),
        "aurc_holdout": float(aurc_15),
        "curve": q1_curve,
    },
    "Q2a_random": {"curve": q2a_curve},
    "Q2b_length": {
        "direction": "refuse_longest (score = -seq_len from trajectories.npz)",
        "auroc_holdout": float(auroc_len),
        "aurc_holdout": float(aurc_len),
        "seq_len_stats": {
            "mean": float(seq_lens.mean()),
            "median": float(np.median(seq_lens)),
            "min": float(seq_lens.min()),
            "max": float(seq_lens.max()),
        },
        "curve": q2b_curve,
    },
    "Q2c_max_token_prob": {
        "source": "data/experiment1_v2/output_entropy_scores.json (max_new_tokens=256)",
        "caveat": "This run's generations may differ from the activations/NEW-labels run; "
                  "use as a logprob-refusal surrogate on the same problems, not the same outputs.",
        "auroc_holdout": float(auroc_mtp),
        "aurc_holdout": float(aurc_mtp),
        "curve": q2c_curve,
    },
    "Q2d_neg_entropy": {
        "source": "same as Q2c (-entropy as confidence)",
        "auroc_holdout": float(auroc_negH),
        "aurc_holdout": float(aurc_negH),
        "curve": q2d_curve,
    },
    "Q3_7B_L15_DoM": {
        "layer": L7,
        "trained_on": "7B correctness labels (81/500, 256-tok run)",
        "applied_to": "1.5B NEW labels",
        "auroc_holdout": float(auroc_7),
        "aurc_holdout": float(aurc_7),
        "curve": q3_curve,
    },
}
out_path = SCRATCH / "pathway10_risk_coverage_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
