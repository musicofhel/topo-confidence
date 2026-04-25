"""
Four new questions:
  T1. Temporal DoM curve in trajectories.npz (per-token).
  T2. Divergence point: when do correct/incorrect centroid paths diverge?
  T3. Feasibility of per-token steering (local GPU + model cache check).
  T4. 7B<->1.5B rank-correlation of DoM-projected scores.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

root = Path("/home/musicofhel/topo-confidence")
out: dict = {}


def auroc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def dom(X, y):
    return X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


# ----------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------
ls_15b = np.load(root / "pathway2/track_a/phase0/layer_states.npy")
y_new = np.load(
    root / "pathway6_rebuild/phase0_relabel/baseline_correct_v2.npy"
).astype(int)
ls_7b = np.load(
    root / "pathway6_rebuild/phase3_cross_benchmark/math7b/layer_states.npy"
)
y_7b = np.load(
    root / "pathway6_rebuild/phase3_cross_benchmark/math7b/baseline_correct.npy"
).astype(int)
traj_npz = np.load(root / "data/experiment1_v2/trajectories.npz")

# Deterministic 80/20 stratified split — reused so numbers are comparable
idx = np.arange(len(y_new))
i_tr, i_ho = train_test_split(idx, test_size=0.2, random_state=9999, stratify=y_new)
print(
    f"Split: train={len(i_tr)} (pos={int(y_new[i_tr].sum())}), "
    f"hold={len(i_ho)} (pos={int(y_new[i_ho].sum())})"
)

# ----------------------------------------------------------------------
# Which layer is trajectories.npz?  User said "layer 15" but Q2 showed L28.
# Re-identify by cos-matching each traj's last-token against each layer's
# final-token activation in layer_states_15b.
# ----------------------------------------------------------------------
last_tokens = np.stack(
    [traj_npz[f"traj_{i}"][-1] for i in range(len(y_new))], axis=0
)  # (500, 1536)
layer_match = {}
for L in range(ls_15b.shape[1]):
    a = last_tokens / (np.linalg.norm(last_tokens, axis=1, keepdims=True) + 1e-12)
    b = ls_15b[:, L, :] / (
        np.linalg.norm(ls_15b[:, L, :], axis=1, keepdims=True) + 1e-12
    )
    layer_match[L] = float((a * b).sum(axis=1).mean())
best_layer = max(layer_match, key=layer_match.get)
print(
    f"Trajectory layer identified: L={best_layer} "
    f"(mean cos={layer_match[best_layer]:.3f}). User said L15."
)

# ----------------------------------------------------------------------
# T1. Temporal DoM at every 10th token position up to median seq_len
# ----------------------------------------------------------------------
seq_lens = np.array(
    [traj_npz[f"traj_{i}"].shape[0] for i in range(len(y_new))]
)
median_len = int(np.median(seq_lens))
print(f"\nseq_len: median={median_len}, mean={seq_lens.mean():.1f}, "
      f"p25={int(np.percentile(seq_lens, 25))}, p75={int(np.percentile(seq_lens, 75))}")

# First compute the final-token DoM (reference direction for cos)
d_final_tr = dom(last_tokens[i_tr], y_new[i_tr])

positions = [0] + list(range(9, median_len + 1, 10))  # token 1, 10, 20, ..., ~median
temporal = []
for p in positions:
    # mask: problems whose seq_len > p (so token p exists)
    mask = seq_lens > p
    if mask.sum() < 50 or y_new[mask].sum() < 5 or (1 - y_new[mask]).sum() < 5:
        continue
    X_p = np.stack(
        [traj_npz[f"traj_{i}"][p] for i in range(len(y_new)) if mask[i]], axis=0
    )
    y_p = y_new[mask]
    # Train/holdout on the subset (respecting original split)
    tr_mask = np.isin(np.where(mask)[0], i_tr)
    ho_mask = ~tr_mask
    if y_p[tr_mask].sum() < 3 or (1 - y_p[tr_mask]).sum() < 3:
        continue
    d_p_tr = dom(X_p[tr_mask], y_p[tr_mask])
    a_tr = auroc(y_p[tr_mask], X_p[tr_mask] @ d_p_tr)
    a_ho = auroc(y_p[ho_mask], X_p[ho_mask] @ d_p_tr)
    c = cos(d_p_tr, d_final_tr)
    temporal.append(
        {
            "token_pos": int(p),
            "n_problems": int(mask.sum()),
            "n_pos": int(y_p.sum()),
            "dom_train_auroc": a_tr,
            "dom_holdout_auroc": a_ho,
            "cos_with_final_token_dom": c,
        }
    )
    print(
        f"  t={p:3d}  n={mask.sum():3d}  pos={int(y_p.sum()):3d}  "
        f"DoM_ho={a_ho:.3f}  cos(final)={c:+.3f}"
    )
out["T1_temporal_dom"] = {
    "trajectory_layer_empirical": int(best_layer),
    "median_seq_len": median_len,
    "positions": temporal,
}

# ----------------------------------------------------------------------
# T2. Divergence point — cos between correct-centroid and incorrect-centroid
#     path at each token position, first crossing of 0.95
# ----------------------------------------------------------------------
# Use the widest range we can — positions where >=20 correct AND >=20 incorrect
max_p = int(seq_lens.max())
div_points = []
first_below_95 = None
for p in range(0, max_p, 1):
    mask = seq_lens > p
    n_c = int(y_new[mask & (y_new == 1)].sum() + 0)  # num correct w/ token p
    n_i = int((~y_new.astype(bool) & mask).sum())
    if n_c < 20 or n_i < 20:
        continue
    X_c = np.stack(
        [traj_npz[f"traj_{i}"][p] for i in range(len(y_new)) if mask[i] and y_new[i] == 1],
        axis=0,
    )
    X_i = np.stack(
        [traj_npz[f"traj_{i}"][p] for i in range(len(y_new)) if mask[i] and y_new[i] == 0],
        axis=0,
    )
    cc = X_c.mean(axis=0)
    ci = X_i.mean(axis=0)
    c = cos(cc, ci)
    div_points.append({"token_pos": int(p), "n_c": n_c, "n_i": n_i, "cos": c})
    if first_below_95 is None and c < 0.95:
        first_below_95 = {"token_pos": int(p), "cos": c, "n_c": n_c, "n_i": n_i}

# Sample the curve for printout (every 5th)
print("\nDivergence curve (every 5th token):")
for r in div_points[::5][:30]:
    print(f"  t={r['token_pos']:3d}  cos(correct,incorrect)={r['cos']:.4f}  "
          f"n_c={r['n_c']}  n_i={r['n_i']}")
print(f"First token where cos < 0.95: {first_below_95}")
out["T2_divergence"] = {
    "samples_every_5th": div_points[::5],
    "first_below_0_95": first_below_95,
    "last_point": div_points[-1] if div_points else None,
}

# ----------------------------------------------------------------------
# T3. Feasibility — local GPU, model cache, α sweep cost estimate
# ----------------------------------------------------------------------
# Use nvidia-smi + hub cache + file sizes from bash
def cmd(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

gpu_info = cmd("nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader")
# Sizes in GB of the two Qwen model directories
size_15b = cmd("du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct 2>/dev/null | cut -f1")
size_7b = cmd("du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct 2>/dev/null | cut -f1")
params_15b = 1.54e9
params_7b = 7.62e9
fp16_gb = lambda p: p * 2 / 1024**3
# Sweep estimate: 7 α × 100 problems × ~200 tokens on a 1.5B-fp16 on RTX 2060:
# a 1.5B model gives ~40 tok/s on a 2060. 7 * 100 * 200 / 40 = 3500s ≈ 1h.
# On H100 (≈200 tok/s for 1.5B fp16 greedy): 7 * 100 * 200 / 200 = 700s ≈ 12 min.
out["T3_feasibility"] = {
    "gpu_info": gpu_info,
    "qwen_1_5b_cache_size": size_15b,
    "qwen_7b_cache_size": size_7b,
    "vram_fp16_gb_1_5b": fp16_gb(params_15b),
    "vram_fp16_gb_7b": fp16_gb(params_7b),
    "fits_on_local_2060": fp16_gb(params_15b) < 6.5,  # ~7GB usable of 8GB
    "sweep_alpha_count": 7,
    "sweep_problems": 100,
    "sweep_tokens_per_problem": 200,
    "sweep_estimate_rtx2060_hours": 7 * 100 * 200 / 40 / 3600,
    "sweep_estimate_h100_minutes": 7 * 100 * 200 / 200 / 60,
}
print("\nT3 feasibility:")
print(f"  GPU: {gpu_info}")
print(f"  1.5B cache: {size_15b}  |  7B cache: {size_7b}")
print(f"  1.5B fp16 VRAM: {fp16_gb(params_15b):.2f} GB (fits on 2060)")
print(f"  7B fp16 VRAM:   {fp16_gb(params_7b):.2f} GB (pod only)")
print(f"  α-sweep: 7×100×200 tok  →  ~{7*100*200/40/60:.0f} min on 2060  |  ~{7*100*200/200/60:.0f} min on H100")

# ----------------------------------------------------------------------
# T4. 7B vs 1.5B rank correlation of DoM-projected scores
# ----------------------------------------------------------------------
# 1.5B: L19 DoM (best single-layer). Train on i_tr NEW labels, project all 500.
# 7B:   L15 DoM (cross-scale probe sweet spot from prior analysis).
#       Train on i_tr 7B labels, project all 500.
L_15b = 19
L_7b = 15

X_15b = ls_15b[:, L_15b, :]
d_15b = dom(X_15b[i_tr], y_new[i_tr])
s_15b = X_15b @ d_15b  # all 500

X_7b = ls_7b[:, L_7b, :]
d_7b = dom(X_7b[i_tr], y_7b[i_tr])
s_7b = X_7b @ d_7b  # all 500

rho_all, p_all = spearmanr(s_15b, s_7b)
# Also on the 44 both-correct problems
both_right = (y_new == 1) & (y_7b == 1)
print(f"\n  both-correct (44): expected ~44, got {int(both_right.sum())}")
if both_right.sum() >= 10:
    rho_both, p_both = spearmanr(s_15b[both_right], s_7b[both_right])
else:
    rho_both, p_both = float("nan"), float("nan")

# And on just the (7B right & 1.5B wrong) bucket — the "verifier-winnable" set
bucket = (y_new == 0) & (y_7b == 1)
if bucket.sum() >= 10:
    rho_bucket, p_bucket = spearmanr(s_15b[bucket], s_7b[bucket])
else:
    rho_bucket, p_bucket = float("nan"), float("nan")

# Extra sanity: rank correlation between 1.5B score and 7B score on the
# 1.5B-wrong subset (where we'd actually want a verifier)
wrong_15b = y_new == 0
rho_wrong, p_wrong = spearmanr(s_15b[wrong_15b], s_7b[wrong_15b])

out["T4_rank_corr"] = {
    "layers": {"1.5B": L_15b, "7B": L_7b},
    "n_all": int(len(y_new)),
    "spearman_rho_all": float(rho_all),
    "spearman_p_all": float(p_all),
    "n_both_correct": int(both_right.sum()),
    "spearman_rho_both_correct": float(rho_both),
    "spearman_p_both_correct": float(p_both),
    "n_verifier_winnable": int(bucket.sum()),
    "spearman_rho_verifier_bucket": float(rho_bucket),
    "spearman_p_verifier_bucket": float(p_bucket),
    "n_1_5B_wrong": int(wrong_15b.sum()),
    "spearman_rho_1_5B_wrong": float(rho_wrong),
    "spearman_p_1_5B_wrong": float(p_wrong),
    "truncation_caveat": "7B scores use max_tokens=256 activations (81/500 labels). "
                        "phase6_5 has 348/500 at max_tokens=1024 but no activations cached.",
}
print(f"\nT4 Spearman (1.5B L{L_15b} DoM score  vs  7B L{L_7b} DoM score):")
print(f"  all 500:                 rho={rho_all:+.3f}  p={p_all:.2e}")
print(f"  both-right ({int(both_right.sum())}):          rho={rho_both:+.3f}  p={p_both:.2e}")
print(f"  verifier-winnable ({int(bucket.sum())}):    rho={rho_bucket:+.3f}  p={p_bucket:.2e}")
print(f"  1.5B-wrong ({int(wrong_15b.sum())}):         rho={rho_wrong:+.3f}  p={p_wrong:.2e}")

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
out_path = root / "scratch/pathway10_temporal_and_verifier_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nWrote {out_path}")
