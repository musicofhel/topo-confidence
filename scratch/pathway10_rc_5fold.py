"""
Pathway 10 v2 E3 — 5-fold CV risk-coverage + combined selector.

Decision gate: does the CI around acc@25%-cov and acc@50%-cov include the v2
targets (>=60% and >=40%)? Each fold holds out ~100 problems; aggregating the
out-of-fold scores gives 500 evaluated problems, shrinking CIs from ~+/-22pp
to ~+/-8pp.

Combined selector: stack L19-DoM-proj + 7B-L15-DoM-proj + (-seq_len) into a
logistic regression *per fold*. Tests whether length is carrying all the
weight of the hidden-state signals or whether they're complementary.

Skipped: max-logprob baseline from a fresh run (decision was to wait).
Cached max_token_prob from the 256-tok run is included as a side reference only.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path("/home/musicofhel/topo-confidence")
SCRATCH = ROOT / "scratch"

# ---- Load ----------------------------------------------------------------
ls_15b = np.load(ROOT / "pathway2/track_a/phase0/layer_states.npy")
ls_7b = np.load(ROOT / "pathway6_rebuild/phase3_cross_benchmark/math7b/layer_states.npy")
y_new = np.load(ROOT / "pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy").astype(int)
y_7b = np.load(ROOT / "pathway6_rebuild/phase3_cross_benchmark/math7b/baseline_correct.npy").astype(int)
traj_npz = np.load(ROOT / "data/experiment1_v2/trajectories.npz")
with open(ROOT / "data/experiment1_v2/output_entropy_scores.json") as f:
    entropy_json = json.load(f)

n = len(y_new)
seq_lens = np.array([traj_npz[f"traj_{i}"].shape[0] for i in range(n)], dtype=float)
max_tok_prob = np.array([r["max_token_prob"] for r in entropy_json["scores"]], dtype=float)
entropy_vec = np.array([r["entropy"] for r in entropy_json["scores"]], dtype=float)

print(f"n={n}, NEW positives={int(y_new.sum())} ({y_new.mean():.1%}), 7B positives={int(y_7b.sum())}")

# ---- Helpers -------------------------------------------------------------
def dom(X, y):
    w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return w / (np.linalg.norm(w) + 1e-12)


def risk_coverage(scores, y, cov_grid):
    order = np.argsort(-scores)
    y_s = y[order]
    n_ = len(y)
    rows = []
    for c in cov_grid:
        k = max(1, int(round(c * n_)))
        ans = y_s[:k]
        ref = y_s[k:]
        rows.append({
            "coverage": float(c),
            "k_answered": int(k),
            "k_refused": int(n_ - k),
            "acc_answered": float(ans.mean()),
            "acc_refused": float(ref.mean()) if len(ref) else float("nan"),
        })
    return rows


def aurc(scores, y):
    order = np.argsort(-scores)
    y_s = y[order]
    cum = np.cumsum(y_s)
    ks = np.arange(1, len(y) + 1)
    return float((1 - cum / ks).mean())


def boot_acc_at_cov(scores, y, cov, n_boot=2000, seed=9999):
    rng = np.random.default_rng(seed)
    n_ = len(y)
    k = max(1, int(round(cov * n_)))
    accs = np.empty(n_boot)
    for b in range(n_boot):
        bi = rng.integers(0, n_, size=n_)
        s = scores[bi]
        yb = y[bi]
        accs[b] = yb[np.argsort(-s)][:k].mean()
    return (
        float(np.percentile(accs, 2.5)),
        float(np.median(accs)),
        float(np.percentile(accs, 97.5)),
    )


def boot_auroc(scores, y, n_boot=2000, seed=9999):
    rng = np.random.default_rng(seed)
    n_ = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        bi = rng.integers(0, n_, size=n_)
        yb = y[bi]
        if yb.sum() == 0 or yb.sum() == n_:
            vals[b] = np.nan
            continue
        vals[b] = roc_auc_score(yb, scores[bi])
    vals = vals[~np.isnan(vals)]
    return (
        float(np.percentile(vals, 2.5)),
        float(np.median(vals)),
        float(np.percentile(vals, 97.5)),
    )


# ---- 5-fold out-of-fold scores ------------------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9999)
oof_L19 = np.empty(n)
oof_7B = np.empty(n)
oof_len = -seq_lens.copy()  # no fitting needed
oof_combined = np.empty(n)
oof_combined_nolen = np.empty(n)  # DoM + 7B only (no length)

# Per-fold LR coefficients (to inspect what the combiner is leaning on)
lr_coefs_all = []

for fold, (tr, te) in enumerate(skf.split(np.arange(n), y_new)):
    # 1.5B L19 DoM
    w15 = dom(ls_15b[tr, 19, :], y_new[tr])
    proj15_tr = ls_15b[tr, 19, :] @ w15
    proj15_te = ls_15b[te, 19, :] @ w15
    oof_L19[te] = proj15_te

    # 7B L15 DoM (trained on 7B labels)
    w7 = dom(ls_7b[tr, 15, :], y_7b[tr])
    proj7_tr = ls_7b[tr, 15, :] @ w7
    proj7_te = ls_7b[te, 15, :] @ w7
    oof_7B[te] = proj7_te

    # Combined: standardize features from train, apply to test, LR on NEW labels
    feat_tr = np.stack([proj15_tr, proj7_tr, -seq_lens[tr]], axis=1)
    feat_te = np.stack([proj15_te, proj7_te, -seq_lens[te]], axis=1)
    mu = feat_tr.mean(0)
    sd = feat_tr.std(0) + 1e-12
    feat_tr_s = (feat_tr - mu) / sd
    feat_te_s = (feat_te - mu) / sd
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(feat_tr_s, y_new[tr])
    oof_combined[te] = lr.decision_function(feat_te_s)
    lr_coefs_all.append({
        "fold": fold,
        "coef": lr.coef_[0].tolist(),
        "intercept": float(lr.intercept_[0]),
        "feature_order": ["L19_DoM_proj", "7B_L15_DoM_proj", "neg_seq_len"],
    })

    # Combined no-length
    feat2_tr = np.stack([proj15_tr, proj7_tr], axis=1)
    feat2_te = np.stack([proj15_te, proj7_te], axis=1)
    mu2 = feat2_tr.mean(0)
    sd2 = feat2_tr.std(0) + 1e-12
    lr2 = LogisticRegression(max_iter=2000, C=1.0)
    lr2.fit((feat2_tr - mu2) / sd2, y_new[tr])
    oof_combined_nolen[te] = lr2.decision_function((feat2_te - mu2) / sd2)

# ---- Aggregate: AUROC / AURC / full curve over all 500 ------------------
cov_grid = np.round(np.arange(0.10, 0.91, 0.05), 2)

methods = {
    "1.5B_L19_DoM": oof_L19,
    "7B_L15_DoM": oof_7B,
    "neg_length": oof_len,
    "combined_DoM+7B+len": oof_combined,
    "combined_DoM+7B": oof_combined_nolen,
    "max_token_prob_cached_256tok": max_tok_prob,
    "neg_entropy_cached_256tok": entropy_vec * -1,
}

print("\n=== 5-fold out-of-fold AUROC (n=500) ===")
results = {}
for name, s in methods.items():
    auroc = roc_auc_score(y_new, s)
    au_lo, au_med, au_hi = boot_auroc(s, y_new, n_boot=2000)
    aurc_val = aurc(s, y_new)
    print(f"  {name:>30}  AUROC={auroc:.3f}  95%CI=[{au_lo:.3f}, {au_hi:.3f}]  AURC={aurc_val:.3f}")
    results[name] = {
        "auroc": float(auroc),
        "auroc_95ci": [au_lo, au_hi],
        "aurc": float(aurc_val),
        "curve": risk_coverage(s, y_new, cov_grid),
    }

# ---- Full curves ---------------------------------------------------------
print("\n=== Full risk-coverage curves (acc_answered, n=500) ===")
hdr = f"{'cov':>5} " + "".join(f"{m[:14]:>16}" for m in methods)
print(hdr)
for i, c in enumerate(cov_grid):
    line = f"{c:>5.2f} "
    for name in methods:
        acc = results[name]["curve"][i]["acc_answered"]
        line += f"{acc:>16.3f}"
    print(line)

# ---- Bootstrap CIs at operating points ----------------------------------
print("\n=== Bootstrap 95% CI on acc_answered at key coverages (n_boot=2000) ===")
print(f"{'method':>32} {'cov':>5} {'k_ans':>5} {'lo':>7} {'med':>7} {'hi':>7}")
op_points = {}
for cov in [0.25, 0.50, 0.75]:
    op_points[f"cov_{cov}"] = {}
    for name, s in methods.items():
        lo, med, hi = boot_acc_at_cov(s, y_new, cov, n_boot=2000)
        k = max(1, int(round(cov * n)))
        print(f"{name:>32} {cov:>5.2f} {k:>5d} {lo:>7.3f} {med:>7.3f} {hi:>7.3f}")
        op_points[f"cov_{cov}"][name] = {"lo": lo, "median": med, "hi": hi, "k_answered": k}

# ---- v2 success criteria check ------------------------------------------
print("\n=== v2 success criteria check ===")
print("Targets: >=60% acc at 25% cov, >=40% acc at 50% cov")
print(f"{'method':>32} {'acc@25%':>9} {'ci@25%':>18} {'acc@50%':>9} {'ci@50%':>18}")
for name in methods:
    a25 = results[name]["curve"][3]["acc_answered"]  # cov=0.25 is index 3
    a50 = results[name]["curve"][8]["acc_answered"]  # cov=0.50 is index 8
    ci25 = op_points["cov_0.25"][name]
    ci50 = op_points["cov_0.5"][name]
    hit25 = "HIT" if ci25["lo"] >= 0.60 else ("MAYBE" if ci25["hi"] >= 0.60 else "MISS")
    hit50 = "HIT" if ci50["lo"] >= 0.40 else ("MAYBE" if ci50["hi"] >= 0.40 else "MISS")
    print(
        f"{name:>32} {a25:>9.3f} [{ci25['lo']:.2f},{ci25['hi']:.2f}]={hit25:>5} "
        f"{a50:>9.3f} [{ci50['lo']:.2f},{ci50['hi']:.2f}]={hit50:>5}"
    )

# ---- Inspect combiner: which features did LR weight? --------------------
coefs_arr = np.array([c["coef"] for c in lr_coefs_all])
print("\n=== Combined LR coefficients (standardized features, 5 folds) ===")
print(f"{'feature':>20} {'mean':>8} {'std':>8}")
for i, feat in enumerate(["L19_DoM_proj", "7B_L15_DoM_proj", "neg_seq_len"]):
    print(f"{feat:>20} {coefs_arr[:, i].mean():>8.3f} {coefs_arr[:, i].std():>8.3f}")

# ---- Save ---------------------------------------------------------------
out = {
    "meta": {
        "n": int(n),
        "n_positive": int(y_new.sum()),
        "base_rate": float(y_new.mean()),
        "cv": "StratifiedKFold(5, shuffle=True, random_state=9999)",
        "coverage_grid": [float(c) for c in cov_grid],
    },
    "methods": results,
    "operating_points": op_points,
    "combined_lr_coefs": lr_coefs_all,
    "notes": {
        "max_token_prob_caveat": (
            "Cached from output_entropy_scores.json (max_new_tokens=256); "
            "different generation run than the activation cache. Shown for reference."
        ),
        "7B_labels_caveat": (
            "7B labels from a max_tokens=256 run (81/500 correct). Phase 6.5 had "
            "348/500 at 1024 tokens but those activations were not saved."
        ),
    },
}
out_path = SCRATCH / "pathway10_rc_5fold_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved: {out_path}")
