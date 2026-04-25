"""
Five follow-up questions on pathway 10 E1 preflight.

Q1: Length-residualize DoM at layers 2, 10, 19, 25. Report cos(raw, resid), AUROC.
Q2: First-generated-token DoM — is the signal in prompt encoding or built during gen?
Q3: Bootstrap layer-19 DoM. 1000 resamples. 95% CI on AUROC + direction stability.
Q4: Multi-layer DoM (concat L10+L19+L25). Beats single-layer?
Q5: 7B verifier 2x2: of 104 1.5B-correct, how many 7B-correct? And 7B-right + 1.5B-wrong bucket?
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/musicofhel/topo-confidence")
OUT = ROOT / "scratch" / "pathway10_five_questions_results.json"
OUT.parent.mkdir(exist_ok=True, parents=True)

SEED = 9999
rng = np.random.default_rng(SEED)


def auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def dom(X, y):
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


print("Loading data...", flush=True)
layer_states_15b = np.load(ROOT / "pathway2/track_a/phase0/layer_states.npy")
y_new = np.load(ROOT / "pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy").astype(int)
layer_states_7b = np.load(ROOT / "pathway6_rebuild/phase3_cross_benchmark/math7b/layer_states.npy")
y_7b = np.load(ROOT / "pathway6_rebuild/phase3_cross_benchmark/math7b/baseline_correct.npy").astype(int)
traj = np.load(ROOT / "data/experiment1_v2/trajectories.npz", allow_pickle=True)
print(f"  1.5B states {layer_states_15b.shape} labels {y_new.sum()}/{len(y_new)}", flush=True)
print(f"  7B   states {layer_states_7b.shape} labels {y_7b.sum()}/{len(y_7b)}", flush=True)

seq_lens = np.array([traj[f"traj_{i}"].shape[0] for i in range(500)])
print(f"  Trajectory seq_lens mean={seq_lens.mean():.1f}, range=({seq_lens.min()},{seq_lens.max()})", flush=True)

# Deterministic 80/20 split reused across questions
idx = np.arange(500)
itr, iho = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y_new)


def split_dom_auroc(X, y, itr, iho):
    """DoM on train, score holdout by dot product with direction."""
    d = dom(X[itr], y[itr])
    s_tr = X[itr] @ d
    s_ho = X[iho] @ d
    return auroc(y[itr], s_tr), auroc(y[iho], s_ho), d


def split_lr_auroc(X, y, itr, iho):
    sc = StandardScaler().fit(X[itr])
    lr = LogisticRegression(max_iter=200, solver="liblinear", C=1.0).fit(sc.transform(X[itr]), y[itr])
    w = lr.coef_.ravel()
    return auroc(y[iho], lr.decision_function(sc.transform(X[iho]))), w, sc


# =============================================================
# Q1: Length residualization
# =============================================================
print("\n=== Q1: Length residualization ===", flush=True)
q1 = {}
for L in [2, 10, 19, 25]:
    X = layer_states_15b[:, L, :].astype(np.float32)
    # Fit length regression on train only: X -> seq_len (continuous)
    sc = StandardScaler().fit(X[itr])
    Xtr = sc.transform(X[itr])
    Xho = sc.transform(X[iho])
    # Scalar length prediction from standardized activations
    ell_reg = LinearRegression().fit(Xtr, seq_lens[itr])
    ell = ell_reg.coef_.astype(np.float32)  # 1536-dim weight in standardized space
    ell_unit = ell / (np.linalg.norm(ell) + 1e-12)

    # DoM in standardized space (comparable basis)
    d_std = Xtr[y_new[itr] == 1].mean(0) - Xtr[y_new[itr] == 0].mean(0)
    d_unit = d_std / (np.linalg.norm(d_std) + 1e-12)

    # Cosines
    cos_raw_ell = float(d_unit @ ell_unit)
    # Residualize DoM: remove projection onto ell_unit
    d_resid = d_std - (d_std @ ell_unit) * ell_unit
    cos_raw_resid = float((d_std @ d_resid) / (np.linalg.norm(d_std) * np.linalg.norm(d_resid) + 1e-12))

    # Score train and holdout with raw and residualized directions (in standardized space)
    s_tr_raw = Xtr @ d_std
    s_ho_raw = Xho @ d_std
    s_tr_res = Xtr @ d_resid
    s_ho_res = Xho @ d_resid

    q1[f"L{L}"] = {
        "raw_train_auroc": auroc(y_new[itr], s_tr_raw),
        "raw_holdout_auroc": auroc(y_new[iho], s_ho_raw),
        "resid_train_auroc": auroc(y_new[itr], s_tr_res),
        "resid_holdout_auroc": auroc(y_new[iho], s_ho_res),
        "cos_DoM_length": cos_raw_ell,
        "cos_DoM_raw_vs_resid": cos_raw_resid,
        "len_reg_r2_train": float(ell_reg.score(Xtr, seq_lens[itr])),
        "len_reg_r2_holdout": float(ell_reg.score(Xho, seq_lens[iho])),
    }
    r = q1[f"L{L}"]
    print(
        f"L{L:>2}: raw_ho={r['raw_holdout_auroc']:.3f}  resid_ho={r['resid_holdout_auroc']:.3f}  "
        f"drop={r['raw_holdout_auroc']-r['resid_holdout_auroc']:+.3f}  "
        f"cos(DoM,len)={r['cos_DoM_length']:+.3f}  cos(raw,resid)={r['cos_DoM_raw_vs_resid']:.3f}  "
        f"len_r2_ho={r['len_reg_r2_holdout']:.2f}",
        flush=True,
    )

# =============================================================
# Q2: First-token vs last-token DoM from trajectories
# =============================================================
print("\n=== Q2: Prompt encoding vs reasoning trajectory (trajectories.npz) ===", flush=True)
first_tok = np.stack([traj[f"traj_{i}"][0] for i in range(500)])        # (500, 1536)
last_tok = np.stack([traj[f"traj_{i}"][-1] for i in range(500)])        # (500, 1536)
mid_tok = np.stack([traj[f"traj_{i}"][traj[f"traj_{i}"].shape[0] // 2] for i in range(500)])

# Identify which pathway2 layer matches the trajectory's last token (final-token activation)
# Normalized cosine between trajectory last_tok and each pathway2 layer's final-token state.
traj_layer_match = {}
for L in range(29):
    lt = layer_states_15b[:, L, :]
    # mean absolute difference of L2-normalized vectors — near 0 means match
    a = lt / (np.linalg.norm(lt, axis=1, keepdims=True) + 1e-8)
    b = last_tok / (np.linalg.norm(last_tok, axis=1, keepdims=True) + 1e-8)
    cos_per_row = (a * b).sum(1)
    traj_layer_match[L] = float(cos_per_row.mean())
best_L = max(traj_layer_match, key=traj_layer_match.get)
print(f"Trajectory layer best matches pathway2 layer {best_L} (mean cos={traj_layer_match[best_L]:.3f})", flush=True)

q2 = {"trajectory_layer_id_cos_by_layer": traj_layer_match, "best_match_layer": int(best_L)}

for name, X in [("first_token", first_tok), ("mid_token", mid_tok), ("last_token", last_tok)]:
    tr, ho, _ = split_dom_auroc(X, y_new, itr, iho)
    q2[name] = {"train_auroc": tr, "holdout_auroc": ho}
    print(f"  {name:>11}: train={tr:.3f}  holdout={ho:.3f}", flush=True)

# Is first_token's DoM just a rescaling of last_token's DoM?
d_first = dom(first_tok[itr], y_new[itr])
d_last = dom(last_tok[itr], y_new[itr])
q2["cos_DoM_first_vs_last"] = float(d_first @ d_last / (np.linalg.norm(d_first) * np.linalg.norm(d_last) + 1e-12))
print(f"  cos(DoM_first, DoM_last) = {q2['cos_DoM_first_vs_last']:+.3f}", flush=True)

# =============================================================
# Q3: Bootstrap layer-19 DoM
# =============================================================
print("\n=== Q3: Bootstrap layer-19 DoM (1000 resamples) ===", flush=True)
L = 19
X = layer_states_15b[:, L, :].astype(np.float32)
d_full = dom(X, y_new)
d_full_unit = d_full / (np.linalg.norm(d_full) + 1e-12)
auroc_full = auroc(y_new, X @ d_full)

B = 1000
aurocs = np.zeros(B)
cosines = np.zeros(B)
rng_q3 = np.random.default_rng(42)
for b in range(B):
    bidx = rng_q3.integers(0, 500, size=500)
    # If no positives/negatives in sample, skip
    yb = y_new[bidx]
    if yb.sum() == 0 or yb.sum() == len(yb):
        aurocs[b] = np.nan
        cosines[b] = np.nan
        continue
    Xb = X[bidx]
    d_b = dom(Xb, yb)
    # AUROC on the bootstrap sample
    aurocs[b] = auroc(yb, Xb @ d_b)
    # Cosine to the full-sample direction
    cosines[b] = d_b @ d_full_unit / (np.linalg.norm(d_b) + 1e-12)

valid = ~np.isnan(aurocs)
ci_lo, ci_hi = np.percentile(aurocs[valid], [2.5, 97.5])
cos_p5 = np.percentile(cosines[valid], 5)
cos_p50 = np.percentile(cosines[valid], 50)
q3 = {
    "full_sample_auroc": auroc_full,
    "bootstrap_auroc_mean": float(np.mean(aurocs[valid])),
    "bootstrap_auroc_95ci": [float(ci_lo), float(ci_hi)],
    "bootstrap_cos_with_full_median": float(cos_p50),
    "bootstrap_cos_with_full_5pct": float(cos_p5),
    "bootstrap_cos_with_full_mean": float(np.mean(cosines[valid])),
    "n_valid_bootstraps": int(valid.sum()),
}
print(f"  full auroc = {auroc_full:.3f}", flush=True)
print(f"  bootstrap AUROC 95% CI = [{ci_lo:.3f}, {ci_hi:.3f}]", flush=True)
print(f"  direction cos vs full: median={cos_p50:.3f}, 5th pct={cos_p5:.3f}, mean={np.mean(cosines[valid]):.3f}", flush=True)

# =============================================================
# Q4: Multi-layer DoM (L10+L19+L25 concat)
# =============================================================
print("\n=== Q4: Multi-layer DoM (L10+L19+L25 concat) ===", flush=True)
X_multi = np.concatenate([layer_states_15b[:, L, :] for L in [10, 19, 25]], axis=1).astype(np.float32)
tr, ho, d_multi = split_dom_auroc(X_multi, y_new, itr, iho)
q4 = {
    "layers_concat": [10, 19, 25],
    "feature_dim": int(X_multi.shape[1]),
    "train_auroc": tr,
    "holdout_auroc": ho,
}
# Compare to each single-layer at these indices
for L in [10, 19, 25]:
    Xl = layer_states_15b[:, L, :].astype(np.float32)
    tr_l, ho_l, _ = split_dom_auroc(Xl, y_new, itr, iho)
    q4[f"single_L{L}_holdout"] = ho_l
print(f"  concat {X_multi.shape[1]}d: train={tr:.3f}  holdout={ho:.3f}", flush=True)
print(f"  single L10 holdout = {q4['single_L10_holdout']:.3f}", flush=True)
print(f"  single L19 holdout = {q4['single_L19_holdout']:.3f}", flush=True)
print(f"  single L25 holdout = {q4['single_L25_holdout']:.3f}", flush=True)

# =============================================================
# Q5: 7B verifier 2x2 disaggregation
# =============================================================
print("\n=== Q5: 7B vs 1.5B 2x2 disaggregation ===", flush=True)
both = int(((y_new == 1) & (y_7b == 1)).sum())
only_15 = int(((y_new == 1) & (y_7b == 0)).sum())
only_7 = int(((y_new == 0) & (y_7b == 1)).sum())
neither = int(((y_new == 0) & (y_7b == 0)).sum())
agree = both + neither

q5 = {
    "both_correct": both,
    "only_1_5B_correct": only_15,
    "only_7B_correct_verifier_winnable": only_7,
    "neither_correct": neither,
    "agreement": agree,
    "n_1_5B_correct_total": int(y_new.sum()),
    "n_7B_correct_total": int(y_7b.sum()),
    "pct_1_5B_correct_that_7B_also": float(both / y_new.sum()),
    "7B_labels_note": "7B run was truncated (max_tokens=256); phase6_5 had 348/500 at 1024 but no activations saved",
}
print(f"  both={both}  only_1.5B={only_15}  only_7B={only_7}  neither={neither}", flush=True)
print(f"  of {y_new.sum()} 1.5B-correct, 7B also correct on {both} ({both/y_new.sum()*100:.1f}%)", flush=True)
print(f"  verifier-winnable bucket (7B_right & 1.5B_wrong): {only_7} problems", flush=True)

# =============================================================
# Dump
# =============================================================
results = {
    "meta": {"seed": SEED, "split_test_size": 0.2, "labels": "NEW", "n_problems": 500},
    "Q1_length_residualization": q1,
    "Q2_prompt_vs_trajectory": q2,
    "Q3_bootstrap_layer19": q3,
    "Q4_multi_layer_concat": q4,
    "Q5_7B_verifier_disagg": q5,
}
OUT.write_text(json.dumps(results, indent=2))
print(f"\nSaved: {OUT}", flush=True)
