"""
Pathway 10 v2 E3 — quartile accuracy + compute-gated simulation.

Takes 5-fold OOF combined scores (L19 DoM + 7B L15 DoM, no length),
splits problems into quartiles by predicted difficulty, and simulates
confidence-gated compute allocation vs uniform K=1 / K=8.

Assumption (user-supplied, from Wang 2022 self-consistency): K=8 majority
vote triples a 20% base-rate problem's accuracy. Applied per-quartile as
acc_K8 = min(1.0, 3 * acc_K1). This yields the CEILING of gating without
actually generating samples.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
SCRATCH = ROOT / "scratch"

ls_15b = np.load(ROOT / "pathway2/track_a/phase0/layer_states.npy")
ls_7b = np.load(ROOT / "pathway6_rebuild/phase3_cross_benchmark/math7b/layer_states.npy")
y_new = np.load(ROOT / "pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy").astype(int)
y_7b = np.load(ROOT / "pathway6_rebuild/phase3_cross_benchmark/math7b/baseline_correct.npy").astype(int)
n = len(y_new)


def dom(X, y):
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-12)


# ---- Regenerate OOF combined_nolen scores -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)
oof_combined = np.empty(n)

for tr, te in skf.split(np.arange(n), y_new):
    w15 = dom(ls_15b[tr, 19, :], y_new[tr])
    w7 = dom(ls_7b[tr, 15, :], y_7b[tr])
    proj15_tr = ls_15b[tr, 19, :] @ w15
    proj15_te = ls_15b[te, 19, :] @ w15
    proj7_tr = ls_7b[tr, 15, :] @ w7
    proj7_te = ls_7b[te, 15, :] @ w7

    feat_tr = np.stack([proj15_tr, proj7_tr], axis=1)
    feat_te = np.stack([proj15_te, proj7_te], axis=1)
    mu = feat_tr.mean(0)
    sd = feat_tr.std(0) + 1e-12
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit((feat_tr - mu) / sd, y_new[tr])
    oof_combined[te] = lr.decision_function((feat_te - mu) / sd)

# ---- Quartile split -----------------------------------------------------
# Higher combined score = higher predicted correctness probability
# Q4 = easiest (top 25%), Q1 = hardest (bottom 25%)
order = np.argsort(-oof_combined)  # descending: easiest first
quartile_labels = np.empty(n, dtype=int)
quartile_labels[order[:125]] = 4   # top
quartile_labels[order[125:250]] = 3
quartile_labels[order[250:375]] = 2
quartile_labels[order[375:]] = 1   # bottom

print(f"n={n}, base rate={y_new.mean():.3f}")
print("\n=== Per-quartile accuracy (K=1 baseline) ===")
print(f"{'quartile':>10} {'n':>5} {'n_correct':>10} {'acc_K1':>8} {'score_range':>24}")
q_stats = {}
for q in [4, 3, 2, 1]:
    mask = quartile_labels == q
    nq = mask.sum()
    n_correct = int(y_new[mask].sum())
    acc = y_new[mask].mean()
    scores_q = oof_combined[mask]
    q_stats[q] = {
        "n": int(nq),
        "n_correct": n_correct,
        "acc_K1": float(acc),
        "score_min": float(scores_q.min()),
        "score_max": float(scores_q.max()),
    }
    tag = {4: "top (easy)", 3: "upper-mid", 2: "lower-mid", 1: "bot (hard)"}[q]
    print(f"  Q{q} {tag:>6} {nq:>5} {n_correct:>10} {acc:>8.3f}  "
          f"[{scores_q.min():+.3f}, {scores_q.max():+.3f}]")

# ---- Simulation ---------------------------------------------------------
# Assumption: K=8 majority vote ≈ 3× K=1 accuracy (per-quartile, capped at 1.0)
K8_MULT = 3.0

def k8_acc(acc_k1):
    return min(1.0, K8_MULT * acc_k1)

print("\n=== K=8 projected accuracy per quartile (3× multiplier, cap=1.0) ===")
for q in [4, 3, 2, 1]:
    a1 = q_stats[q]["acc_K1"]
    a8 = k8_acc(a1)
    q_stats[q]["acc_K8_projected"] = a8
    print(f"  Q{q}: acc_K1={a1:.3f} -> acc_K8={a8:.3f}")

# ---- Schemes -------------------------------------------------------------
# We evaluate five schemes, all answering ALL 500 problems (no refusal):
#   (a) uniform K=1
#   (b) uniform K=8
#   (c) gated-A: Q4 K=1, Q3/Q2 K=1, Q1 K=8  (spend extra on hardest only)
#   (d) gated-B: Q4 K=1, Q1 K=8, Q3/Q2 REFUSE  (user's literal reading)
#   (e) gated-budget-matched: as scheme (c) but adjust to match K=8 compute

def scheme_stats(per_q_K, per_q_refuse=None):
    """per_q_K: dict q -> int K (samples per problem in that quartile).
    per_q_refuse: dict q -> bool (if True, problems are refused, not answered).
    Returns (overall_acc_on_answered, coverage, total_compute, avg_K_on_answered)."""
    per_q_refuse = per_q_refuse or {q: False for q in [1, 2, 3, 4]}
    total_correct = 0.0
    total_answered = 0
    total_compute = 0
    weighted_K_sum = 0
    for q in [1, 2, 3, 4]:
        stats = q_stats[q]
        k = per_q_K[q]
        total_compute += stats["n"] * k
        if per_q_refuse[q]:
            continue
        acc = k8_acc(stats["acc_K1"]) if k >= 8 else stats["acc_K1"]
        total_correct += acc * stats["n"]
        total_answered += stats["n"]
        weighted_K_sum += stats["n"] * k
    coverage = total_answered / n
    acc_answered = total_correct / total_answered if total_answered > 0 else float("nan")
    acc_overall = total_correct / n  # treats refused as 0 correct (selective acc includes refused as incorrect)
    avg_K_ans = weighted_K_sum / total_answered if total_answered > 0 else 0
    return {
        "acc_answered": float(acc_answered),
        "acc_overall_if_refused_wrong": float(acc_overall),
        "coverage": float(coverage),
        "total_compute": int(total_compute),
        "avg_K_answered": float(avg_K_ans),
    }

schemes = {
    "uniform_K1":           {"K": {1: 1, 2: 1, 3: 1, 4: 1}, "refuse": None},
    "uniform_K8":           {"K": {1: 8, 2: 8, 3: 8, 4: 8}, "refuse": None},
    "gated_spend_on_hard":  {"K": {1: 8, 2: 1, 3: 1, 4: 1}, "refuse": None},
    "gated_top_K1_bot_K8_refuse_mid": {
        "K": {1: 8, 2: 0, 3: 0, 4: 1},
        "refuse": {1: False, 2: True, 3: True, 4: False},
    },
    "gated_hard_K8_top_refuse": {
        # Inverse: only answer the ones we're unsure about, with more compute
        "K": {1: 8, 2: 1, 3: 1, 4: 0},
        "refuse": {1: False, 2: False, 3: False, 4: True},
    },
    "gated_spend_on_middle": {
        # Q3 is where 3x hurts the least (already 21.6% -> 64.8% is realistic)
        "K": {1: 1, 2: 1, 3: 8, 4: 1},
        "refuse": None,
    },
    "gated_spend_on_upper_half": {
        # Q3 + Q4 get K=8: compute=125*8+125*8+125+125=2250
        "K": {1: 1, 2: 1, 3: 8, 4: 8},
        "refuse": None,
    },
    "gated_refuse_bot_half_K8_upper": {
        # Refuse the hopeless half (Q1+Q2, acc ~6%), K=8 on upper half
        "K": {1: 0, 2: 0, 3: 8, 4: 8},
        "refuse": {1: True, 2: True, 3: False, 4: False},
    },
    "gated_refuse_bot_half_K1_upper": {
        # Same refusal, K=1 on answered (compute=250)
        "K": {1: 0, 2: 0, 3: 1, 4: 1},
        "refuse": {1: True, 2: True, 3: False, 4: False},
    },
}

print("\n=== Compute-gated simulation ===")
print(f"{'scheme':>32} {'cov':>6} {'acc_ans':>9} {'acc_all':>9} {'compute':>9} {'avgK_ans':>9}")
sim_results = {}
for name, cfg in schemes.items():
    refuse = cfg["refuse"] if cfg["refuse"] else {q: False for q in [1, 2, 3, 4]}
    r = scheme_stats(cfg["K"], refuse)
    sim_results[name] = r
    print(f"  {name:>30} {r['coverage']:>6.2f} {r['acc_answered']:>9.3f} "
          f"{r['acc_overall_if_refused_wrong']:>9.3f} {r['total_compute']:>9d} {r['avg_K_answered']:>9.2f}")

# ---- Compute-matched comparison -----------------------------------------
# If we fix compute = 500 (uniform K=1), can gating beat K=1 overall acc?
# Gated-hard-Q1-K=8 compute = 125*8 + 375*1 = 1000+375 = 1375, 2.75x K=1.
# To match 500: need per-quartile Ks summing to 500. E.g. Q1 K=2, others K=0.67?
# Not integer, so we report iso-compute at 500 (uniform K=1) and scale up.

# The interesting comparison: what fraction of uniform-K=8 gain does gated capture
# at what fraction of its compute?
u1_acc = sim_results["uniform_K1"]["acc_answered"]
u8_acc = sim_results["uniform_K8"]["acc_answered"]
u8_compute = sim_results["uniform_K8"]["total_compute"]

print(f"\n=== Efficiency vs uniform K=8 (ceiling under 3x assumption) ===")
print(f"uniform_K1: acc={u1_acc:.3f}  compute=500 (baseline)")
print(f"uniform_K8: acc={u8_acc:.3f}  compute={u8_compute} (ceiling)")
print(f"{'scheme':>32} {'acc_gain_frac':>14} {'compute_frac':>14} {'efficiency':>12}")
for name in [
    "gated_spend_on_hard",
    "gated_top_K1_bot_K8_refuse_mid",
    "gated_hard_K8_top_refuse",
    "gated_spend_on_middle",
    "gated_spend_on_upper_half",
    "gated_refuse_bot_half_K8_upper",
    "gated_refuse_bot_half_K1_upper",
]:
    r = sim_results[name]
    # Use acc_overall for gated-refuse schemes (refused count as 0 against the 500-problem pool)
    acc_for_compare = r["acc_overall_if_refused_wrong"]
    gain = (acc_for_compare - u1_acc) / (u8_acc - u1_acc) if u8_acc > u1_acc else float("nan")
    compute_frac = r["total_compute"] / u8_compute
    eff = gain / compute_frac if compute_frac > 0 else float("nan")
    print(f"  {name:>30} {gain:>14.2%} {compute_frac:>14.2%} {eff:>12.2f}")

# ---- Save ---------------------------------------------------------------
out = {
    "meta": {
        "n": int(n),
        "base_rate": float(y_new.mean()),
        "K8_multiplier_assumption": K8_MULT,
        "score_source": "5-fold OOF LR(L19_DoM + 7B_L15_DoM), no length",
    },
    "quartiles": q_stats,
    "schemes": sim_results,
}
out_path = SCRATCH / "pathway10_quartile_compute_sim_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
