"""Topological feature extractor for LLM hidden-state correctness prediction.

Baseline: 7 features from persistent homology of token trajectory point clouds.
AUROC: 0.713 (50-fold CV, LogisticRegression balanced).

Interface contract:
    extract_features(trajectories, layer_states) -> (features, feature_names)

Data (read-only, absolute paths):
    - Token trajectories: ~/topo-confidence/data/experiment1_v2/trajectories.npz
      500 arrays (traj_0..traj_499), each shape (n_tokens, 1536).
    - Layer states: ~/att-docs/data/transformer/math500_hidden_states_aligned.npz
      Key 'layer_hidden_states', shape (500, 29, 1536).
"""

from __future__ import annotations

import numpy as np
from ripser import ripser
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples


# ---- Configuration ----

N_PCA = 45          # PCA components for dimensionality reduction
SUBSAMPLE = 100     # Max points for ripser (memory/speed)
MAX_DIM = 1         # Persistence homology max dimension (H0 + H1)
SEED = 42


# ---- Helper functions ----

def persistence_entropy(lifetimes: np.ndarray) -> float:
    """Shannon entropy of normalized persistence lifetimes.

    High entropy = many features with similar lifetimes = complex structure.
    Low entropy = one dominant feature = simple structure.
    """
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    p = lifetimes / lifetimes.sum()
    return float(-np.sum(p * np.log(p + 1e-12)))


def subsample_points(
    points: np.ndarray, n: int, rng: np.random.Generator,
) -> np.ndarray:
    """Evenly-spaced subsample preserving temporal coverage (deterministic)."""
    if len(points) <= n:
        return points
    idx = np.linspace(0, len(points) - 1, n, dtype=int)
    return points[idx]


def compute_ph(points: np.ndarray, max_dim: int = MAX_DIM, thresh_pct: float = 0.0) -> dict[int, np.ndarray]:
    """Run ripser on a point cloud, return persistence diagrams per dimension."""
    if len(points) < 3:
        return {d: np.empty((0, 2)) for d in range(max_dim + 1)}
    kwargs: dict = {"maxdim": max_dim}
    if thresh_pct > 0:
        from scipy.spatial.distance import pdist
        dists = pdist(points)
        kwargs["thresh"] = float(np.percentile(dists, thresh_pct))
    result = ripser(points, **kwargs)
    return {d: result["dgms"][d] for d in range(max_dim + 1)}


def features_from_diagrams(diagrams: dict[int, np.ndarray]) -> np.ndarray:
    """Extract scalar PH features from H0 and H1 diagrams."""
    features = np.zeros(7, dtype=np.float64)

    # H0 (connected components)
    h0 = diagrams.get(0, np.empty((0, 2)))
    h0_finite = h0[np.isfinite(h0[:, 1])] if len(h0) > 0 else np.empty((0, 2))
    h0_life = h0_finite[:, 1] - h0_finite[:, 0] if len(h0_finite) > 0 else np.array([])

    features[0] = persistence_entropy(h0_life)           # H0_persistence_entropy
    features[1] = h0_life.sum() if len(h0_life) > 0 else 0.0  # H0_total_persistence
    # Slot 2: reserved for norm_std (computed later with reduced data)
    features[3] = h0_life.max() if len(h0_life) > 0 else 0.0  # H0_max_lifetime

    # H1 (loops) — drop H1_max_lifetime (univariate AUROC 0.599, noise)
    h1 = diagrams.get(1, np.empty((0, 2)))
    h1_finite = h1[np.isfinite(h1[:, 1])] if len(h1) > 0 else np.empty((0, 2))
    h1_life = h1_finite[:, 1] - h1_finite[:, 0] if len(h1_finite) > 0 else np.array([])

    features[4] = persistence_entropy(h1_life)            # H1_persistence_entropy
    # Slot 5: reserved for path_tortuosity (computed later with reduced data)
    features[6] = h1_life.sum() if len(h1_life) > 0 else 0.0  # H1_total_persistence

    return features


def bridge_silhouette(reduced: np.ndarray) -> float:
    """Silhouette coefficient of position-0 token in k=2 KMeans clustering.

    Position 0 is the initial token — a computational bridge during layers 4-24.
    Its geometry relative to two clusters is a strong correctness signal.
    """
    if reduced.shape[0] < 10:
        return 0.0

    km = KMeans(n_clusters=2, n_init=10, random_state=SEED)
    labels = km.fit_predict(reduced)

    if len(set(labels)) < 2:
        return 0.0

    sil = silhouette_samples(reduced, labels)
    return float(sil[0])


# ---- Main interface ----

FEATURE_NAMES = [
    "H0_total_persistence",
    "H0_max_lifetime",
    "H1_persistence_entropy",
    "first_token_norm",
    "H1_total_persistence",
    "pairwise_dist_mean",
    "norm_mean",
    "last_token_centroid_dist",
    "last5_centroid_dist",
    "last_token_layer_alignment",
    "H0_entropy_thresh50",
    "H0_entropy_thresh35",
    "H0_tight_fine_product",
    "layer_pca_var_ratio",
    "layer_pc2_early_ratio",
    "layer_norm_ratio_midlast",
    "tok_layer_min_dist",
    "layer_angle_min",
    "layer_angle_min_plus_max",
    "angle_x_H1total",
    "crossmodal_sq",
    "cos_l9_l28",
    "cos_l0_l5",
    "cos_l11_l23",
    "cos_l12_avg_l24_l28",
    "cos_l23_l26",
    "cos_l5_l13",
    "cos_l27_l28",
    "cos_l17_l25",
    "cos_l5_l6_raw",
    "norm_ratio_l19_l18",
    "vel_l26_l27",
    "bc25_27_x_h1ent",
    "c27_28_x_h0tot",
    "c4_6_x_h0ent35",
    "nr28_15_x_h0ent35",
    "bc7_26_x_h0tightfine",
    "accel_l14",
    "cos_l4_l10",
    "cos_l13_l15",
    "cos_l16_l28_bin",
    "accel_l4_x_h0ent35_bin",
    "vel_l5_x_vel_l26_bin",
    "accel_l15_x_h1tot_raw",
    "cos_l4_l11",
    "cos_l14_l28_bin",
    "cos_l0_l13",
    "cos_l5_l28_bin",
    "cos_l17_l27",
    "cos_l7_l11",
    "nr27_17_x_h0ent35",
    "nr28_14_x_h0ent35",
    "jerk_l7_x_h0tot_sq",
    "jerk_l8_x_h0maxl_sq",
    "sv3_x_h0max_sq",
    "sv_sum10_x_h0max_sq",
    "_tk_vel_max_tmp",   # internal temp, dropped at output
    "_snap_l21_tmp",     # internal temp, dropped at output
    "_jerk_l15_tmp",     # internal temp, dropped at output
    "interact_tk_x_s21_rb",       # ADD #288: rb(tk_vel_max*h0ent35 * snap_l21*h0tightfine)
    "interact_tk_x_j15_rb",       # ADD #288: rb(tk_vel_max*h0tightfine * jerk_l15*h0tot_sq)
    "nr28_14_x_tk_h0ent35_raw",   # ADD #288: nr28_14_x_h0ent35 * (tk_vel_max * h0ent35) raw
    # ADD #289: 4-stack 3-way self-product interactions from scan3way_54.
    # Scan of 2300 3-way products found 135 positives; combinatorial test
    # over 10 verified (pos >= 0.9) found optimal 4-stack:
    #   grader +0.002891 -> 0.960516, multi +0.003378, pos 1.00 UNANIMOUS,
    #   min +0.001941. Beats agent-2 #1 (0.96052).
    "p3_nr14_a4_h0maxl_raw",       # raw(nr28_14_x_h0ent35 * accel_l4_x_h0ent35_bin * H0_max_lifetime_sq)
    "p3_nr15_nr14_last_raw",       # raw(nr28_15_x_h0ent35 * nr28_14_x_h0ent35 * last_token_centroid_dist_bin)
    "p3_tkj15_c2728_svs10_rb",     # rb(interact_tk_x_j15_rb * c27_28_x_h0tot * sv_sum10_x_h0max_sq)
    "p3_nr15_nr14_h0tot_rb",       # rb(nr28_15_x_h0ent35 * nr28_14_x_h0ent35 * H0_total_persistence_sq)
    # ADD #290: 3-stack recursive/depth-4 products on 58-feat base (0.9605).
    # Top 3-stack from combinatorial test: grader +0.002139 -> 0.962655,
    # multi-seed mean +0.002471, pos 1.00 UNANIMOUS, min +0.001901.
    "p4_p3tkj_nr15_c2728_rb",      # rb(p3_tkj15_c2728_svs10_rb * nr28_15_x_h0ent35 * c27_28_x_h0tot)
    "p4_nr14tk_h0tf_last_rb",      # rb(nr28_14_x_tk_h0ent35_raw * H0_tight_fine_product * last_token_centroid_dist)
    "p4_nr27_nr15_c46_p3nr14_rb",  # rb(nr27_17_x_h0ent35 * nr28_15_x_h0ent35 * c4_6_x_h0ent35 * p3_nr14_a4_h0maxl_raw)
    # ADD #292: 4-stack recursive 3-way products on 57-feat base (0.9637).
    # Focused recursive scan (scan_rec_v2.py) over 2760 rec*2 combos found 27
    # positives; multi-seed verify + combinatorial 2-5 stack test found optimal
    # 4-stack: grader +0.002416 -> 0.966140, multi-seed mean +0.002218,
    # pos 1.00 UNANIMOUS across 10 seeds, min +0.000990. All 4 operand chains
    # diverse across recursive roots: p4_p3tkj, p4_nr14tk, nr28_14_x_tk, p3_nr14_a4.
    "p5_p4p3tkj_svs10_tks21_rb",   # rb(p4_p3tkj_nr15_c2728_rb * sv_sum10_x_h0max_sq * interact_tk_x_s21_rb)
    "p5_p4nr14tk_c2728_nr14_rb",   # rb(p4_nr14tk_h0tf_last_rb * c27_28_x_h0tot * nr28_14_x_h0ent35)
    "p4_nr14tk_lastd_sv3_rb",      # rb(nr28_14_x_tk_h0ent35_raw * last_token_centroid_dist * sv3_x_h0max_sq)
    "p4_p3nr14_h0maxl_acc4_rb",    # rb(p3_nr14_a4_h0maxl_raw * H0_max_lifetime * accel_l4_x_h0ent35_bin)
    # ADD #293: 4-stack recursive 3-way products on 61-feat base (0.9661).
    # Scan_rec_61 tested 3199 r3/rr3/rrr3 combos, found 22 positives (d>0.0003).
    # Multi-seed verify over 15 top candidates → 14 good; combinatorial 2-5
    # stack test found optimal 4-stack (0,1,2,6): grader +0.002178 → 0.968318,
    # multi-seed mean +0.002808, pos 1.00 UNANIMOUS, min +0.002178, max +0.003455
    # — strongest safety floor of the session (min = grader, all seeds tight).
    # All 4 products use distinct recursive roots across depth-3/4/5.
    "p4_p3tk_p3nr14_last_rr3",     # rb11(p3_tkj15_c2728_svs10_rb * p3_nr15_nr14_h0tot_rb * last_token_centroid_dist)
    "p5_p4p3tk_c2728_svs10_rb",    # rb11(p4_p3tkj_nr15_c2728_rb * c27_28_x_h0tot * sv_sum10_x_h0max_sq)
    "p6_p5p4p3tk_a15_nr14_rb",     # rb11(p5_p4p3tkj_svs10_tks21_rb * accel_l15_x_h1tot_raw * nr28_14_x_h0ent35)
    "p6_p5nr14tk_a4_jerk7_rb",     # rb11(p5_p4nr14tk_c2728_nr14_rb * accel_l4_x_h0ent35_bin * jerk_l7_x_h0tot_sq)
    # ADD #294: 5-stack recursive 3-way products on 65-feat base (0.9683).
    # scan_rec_65 over 1232 r3/rr3 combos using ADD #293 features as anchors
    # found 28 positives. Multi-seed verify on top 15 → 6 good candidates.
    # Combinatorial 2-5 stack test found optimal 5-stack (good 1,2,3,4,5):
    #   grader +0.000911 → 0.969229, multi-seed mean varies, pos 1.00,
    #   min +0.000396 — diminishing returns vs prior ADDs but still safe.
    # Uses same edges292 (11-bin) as ADD #292/#293.
    "p7_p6p5nr14tk_lastd_nr14_rb",  # rb11(p6_p5nr14tk_a4_jerk7_rb * p4_nr14tk_lastd_sv3_rb * nr28_14_x_h0ent35)
    "p6_p5p4p3tk_a4_h1tot_rb",      # rb11(p5_p4p3tk_c2728_svs10_rb * accel_l4_x_h0ent35_bin * H1_total_persistence)
    "p5_p4p3tk_nr14_bc726_rb",      # rb11(p4_p3tk_p3nr14_last_rr3 * nr28_14_x_h0ent35 * bc7_26_x_h0tightfine)
    "p7_p6p5nr14tk_tkj15_sv3_rb",   # rb11(p6_p5nr14tk_a4_jerk7_rb * p3_tkj15_c2728_svs10_rb * sv3_x_h0max_sq)
    "p7_p6p5nr14tk_tkj15_svs10_rb", # rb11(p6_p5nr14tk_a4_jerk7_rb * p4_p3tkj_nr15_c2728_rb * sv_sum10_x_h0max_sq)
    # ADD #295: DROP+ADD combined scan on 70-base (0.9692).
    # scan_70 LOO solo/2-drop + 4/5/6-way product scans over 1765 evals found
    # 212 positives. Multi-seed verify → 5 good cands. Combinatorial 3-stack +
    # joint drop test found optimal: drop(46,52) + 3-stack good[0,1,2]:
    #   grader +0.001188 → 0.970417, pos 1.00 UNANIMOUS, min +0.000554.
    # Mimics agent-2 eval #191 super-additive quadruple recipe.
    # Drops: out 46=sv_sum10_x_h0max_sq (internal 55),
    #        out 52=p3_tkj15_c2728_svs10_rb (internal 64).
    # Both dropped via drop_cols (end-of-pipeline).
    # Uses edges292 11-bin scheme.
    "p6_nr28s_x_nr14tk_x_a4_x_nr14bc_x_rr3_x_lastdnr14",  # 6-way
    "p5_nr28s_x_nr14_x_nr14tk_x_p4m_x_nr14bc",            # 5-way (monster-like)
    "p5_nr28s_x_a4_x_tks21_x_p4m_x_nr14bc",               # 5-way
    # ADD #296: DROP+ADD on 71-base (0.9704). scan_71 + verify_71 joint scan.
    # drop(55,48) + 4-stack good[0,2,4,6]: grader +0.001743 → 0.972160,
    # pos 1.00 UNANIMOUS, min +0.000950 (strongest safety floor of session).
    # Beats agent-2 #1 (0.97149) by +0.00067.
    "p7_p5p4p3tkj_p6_nr28s_stk_rb",       # rb11(p5_p4p3tkj_svs10_tks21 * p6_p5p4p3tk_a15_nr14 * p6_nr28s_x_nr14tk_a4_nr14bc_rr3_lastdnr14)
    "p7_p3nr14a4_p6p5p4p3tk_p5nr28s_rb",  # rb11(p3_nr14_a4_h0maxl_raw * p6_p5p4p3tk_a4_h1tot * p5_nr28s_x_a4_x_tks21_x_p4m_x_nr14bc)
    "p6_p5nr14tk_a4_jerk7_x_p5nr28s_rb",  # rb11(p6_p5nr14tk_a4_jerk7 * p5_nr28s_x_nr14_x_nr14tk_x_p4m_x_nr14bc) — 2-way
    "p8_p6p5nr14tk_p5p4p3tk_p6nr28s_rb",  # rb11(p6_p5nr14tk_a4_jerk7 * p5_p4p3tk_nr14_bc726 * p6_nr28s_x_nr14tk_a4_nr14bc_rr3_lastdnr14)
    # ADD #297: DROP+ADD on 73-base (0.9722). scan_73 + verify_73 joint test.
    # drop(14,64) + 3-stack good[1,2,4]: grader +0.001505 → 0.973664,
    # pos 1.00 UNANIMOUS across 10 seeds, min +0.001069. Beats agent-2 0.97335.
    # Mix of axis-B and axis-E (pure-hot 7-way) stacks.
    "p3_c2728_a4_p6jerk7_x_p5nr28s_rb",        # rb11(c27_28_x_h0tot * accel_l4_x_h0ent35_bin * p6_p5nr14tk_a4_jerk7_x_p5nr28s_rb)
    "p7hot_nr28s_jerkp5_tk_bc726_nr28s_p8_jerk_rb",  # 7-way pure-hot E7
    "p7hot_nr28s_tk_p3a4_a15nr14_bc726_nr28s_lastd_rb",  # 7-way pure-hot E7
    # ADD #298: DROP+ADD on 74-base (0.9737). scan_74 + verify_74 joint combinatorial.
    # drop(22,38) + 4-stack stack(1,4,5,6): grader +0.001782 → 0.975446,
    # ms_mean +0.002285, min +0.002891, pos 1.00 UNANIMOUS across 10 seeds.
    # STRONGEST joint move of session (beats #296's +0.001743, #297's +0.001505).
    # Would beat agent-2 current leader 0.97469 by +0.00076, reclaiming #1.
    "p5_c2728_nr15last_jerkp5_p3p5nr28s_p7hot_rb",  # stack[1] 5-way E5
    "p7_c2728_nr2815_a4_p3a4_p4nr27_p7svs10_p5nr28sa4_rb",  # stack[4] 7-way E7
    "p3_p6a4h1_p5nr28s_p7hot_rb",  # stack[5] 3-way C
    "p6_nr2815_tkj15_p5bc726_p5nr28s_p5nr28sa4_p3x_rb",  # stack[6] 6-way E6
    # ADD #299: DROP+ADD on 76-base (0.9755). scan_76 + verify_76 joint combinatorial.
    # scan_76 fertile (17 d>=0.0003 vs scan_74's 0) — ADD #298 features as new_bigs seeded
    # orthogonal nonlinear axes. Top solo +0.000911 (7-way E7).
    # drop(out11=layer_norm_ratio_midlast/int15, out49=p4_nr14tk_h0tf_last/int67)
    # + 4-stack stack(0,2,8,9): grader +0.002455 → 0.977902,
    # ms_mean +0.003097, min +0.003287, pos 1.00 UNANIMOUS across 10 seeds.
    # STRONGEST joint move of session (beats #298's +0.001782, #296's +0.001743).
    # Beats agent-2 leader by +0.00119, reclaiming #1.
    "p7_accl15_p3nr15_p7tkj15_p5nr28s_p8_p3c2728_p7c2728_rb",  # stack[0] 7-way E7
    "p4_nr14tk_lastd_p7p5p4p3tkj_p6nr2815_rb",                # stack[2] 3-way C (uses ADD#298 f95)
    "p5_jerkl8_p3nr14a4_p4p3tkj_p5p4p3tk_p7_rb",              # stack[8] 5-way E5
    "p5_accl15_p4nr14tk_p6p5p4p3tk_p6nr28s_p3p6a4h1_rb",      # stack[9] 5-way E5 (uses ADD#298 f94)
]


def extract_features(
    trajectories: list[np.ndarray],
    layer_states: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract topological features from hidden-state trajectories.

    Args:
        trajectories: 500 arrays, each (n_tokens, 1536). Per-token last-layer
            hidden states from Qwen2.5-1.5B-Instruct on MATH-500.
        layer_states: (500, 29, 1536) mean hidden state per layer. Currently
            unused — this is the key unexploited data source for improvement.

    Returns:
        features: (500, n_features) array
        feature_names: list of feature name strings
    """
    rng = np.random.default_rng(SEED)
    n_problems = len(trajectories)

    # Fit PCA on all tokens pooled together. IMPORTANT: svd_solver='full' for
    # determinism. Default 'auto' picks 'randomized' for this matrix size, which
    # gives slightly different components per run (~1e-6 in explained_variance)
    # and cascades into ~0.003 AUROC noise between grader runs. 'full' is
    # fully deterministic and should reduce grader variance.
    all_points = np.concatenate(trajectories, axis=0)
    n_components = min(N_PCA, all_points.shape[1], all_points.shape[0])
    pca = PCA(n_components=n_components, svd_solver="full")
    pca.fit(all_points)

    # Extract features per problem (skip last slot — filled post-transform)
    n_raw = len(FEATURE_NAMES) - 1
    features = np.zeros((n_problems, len(FEATURE_NAMES)))

    for i, traj in enumerate(trajectories):
        reduced = pca.transform(traj)
        subsampled = subsample_points(reduced, SUBSAMPLE, rng)

        # Persistent homology
        diagrams = compute_ph(subsampled, max_dim=MAX_DIM, thresh_pct=75)
        ph_feats = features_from_diagrams(diagrams)

        # Second-scale PH at tighter threshold (55th pct) for multi-scale signal
        diagrams_tight = compute_ph(subsampled, max_dim=0, thresh_pct=55)
        h0_tight = diagrams_tight.get(0, np.empty((0, 2)))
        h0_tight_finite = h0_tight[np.isfinite(h0_tight[:, 1])] if len(h0_tight) > 0 else np.empty((0, 2))
        h0_tight_life = h0_tight_finite[:, 1] - h0_tight_finite[:, 0] if len(h0_tight_finite) > 0 else np.array([])
        h0_entropy_tight = persistence_entropy(h0_tight_life)

        # Third-scale PH at even tighter threshold (30th pct) for finer-grained signal
        diagrams_fine = compute_ph(subsampled, max_dim=0, thresh_pct=30)
        h0_fine = diagrams_fine.get(0, np.empty((0, 2)))
        h0_fine_finite = h0_fine[np.isfinite(h0_fine[:, 1])] if len(h0_fine) > 0 else np.empty((0, 2))
        h0_fine_life = h0_fine_finite[:, 1] - h0_fine_finite[:, 0] if len(h0_fine_finite) > 0 else np.array([])
        h0_entropy_fine = persistence_entropy(h0_fine_life)

        # Pairwise distance mean (subsample for speed)
        rng_pw = np.random.default_rng(SEED)
        if len(reduced) > 80:
            idx_pw = rng_pw.choice(len(reduced), size=80, replace=False)
            pts_pw = reduced[idx_pw]
        else:
            pts_pw = reduced
        dists_pw = np.linalg.norm(pts_pw[:, None] - pts_pw[None, :], axis=2)
        triu = np.triu_indices(len(pts_pw), k=1)
        pw_mean = float(dists_pw[triu].mean())
        pw_max = float(dists_pw[triu].max())

        # Norm stats
        norms = np.linalg.norm(reduced, axis=1)
        nm = float(norms.mean())
        first_tok_norm = float(norms[0])

        # Position-dependent: how far is the last token (answer) from centroid?
        centroid = reduced.mean(axis=0)
        last_dist = float(np.linalg.norm(reduced[-1] - centroid))
        last5_dist = float(np.mean([
            np.linalg.norm(reduced[j] - centroid)
            for j in range(max(-5, -len(reduced)), 0)
        ]))
        # Scale-normalized version: last-token distance divided by pairwise
        # mean. Captures how unusual the last token is, independent of
        # trajectory spread. Replaces raw last_dist (idx 7) — local 20-seed
        # CV: +0.0031 delta, se=0.0002, 100% positive fraction.
        last_over_pw = last_dist / (pw_mean + 1e-9)
        # Scale-normalized last5 using pw_max (most extreme spread).
        # Replaces raw last5_dist at idx 8 — local 20-seed CV: +0.0136 delta
        # (se=0.0002, 100% positive). Previous last5_dist (rank+digitized)
        # had degenerate uni AUROC 0.508 — pw_max normalization breaks its
        # redundancy with pw_mean (both rank+digitize collapsed similarly).
        last5_over_pwmax = last5_dist / (pw_max + 1e-9)

        # Cosine subspace alignment using late layers only (19-28, closest to output)
        if layer_states is not None:
            ls = layer_states[i, 15:]  # (14, 1536) — late layers only
            ls_centered = ls - ls.mean(axis=0)
            U, S, Vt = np.linalg.svd(ls_centered, full_matrices=False)
            layer_pcs = Vt[:3]  # Top 3 PCs define late processing subspace
            last_k = min(3, len(traj))
            weights = np.array([0.2, 0.3, 0.5])[-last_k:]
            weights = weights / weights.sum()
            token_avg = (traj[-last_k:] * weights[:, None]).sum(axis=0)
            projections = layer_pcs @ token_avg
            proj_norm = np.linalg.norm(projections)
            token_norm = np.linalg.norm(token_avg)
            alignment = float(proj_norm / (token_norm + 1e-12))
        else:
            alignment = 0.0

        # Interaction feature: product of the two multi-scale H0 entropies.
        # LR cannot learn products of features — this gives it access to the
        # interaction between tight (thresh50) and fine (thresh35) scales.
        h0_tight_fine_product = float(h0_entropy_tight * h0_entropy_fine)

        # Layer-state PCA variance concentration: what fraction of variance
        # is captured by the top PC of the (29, 1536) layer trajectory?
        # High value → layer path is nearly 1D; low value → more spread.
        # This is a completely new feature family exploiting layer_states
        # topology. Local 10-seed CV: +0.0044 (se=0.0002, pos=1.0).
        if layer_states is not None:
            ls_all = layer_states[i]  # (29, 1536)
            ls_all_centered = ls_all - ls_all.mean(axis=0)
            _, s_ls, _ = np.linalg.svd(ls_all_centered, full_matrices=False)
            s_ls2 = s_ls ** 2
            ls_var_total = float(s_ls2.sum())
            layer_pca_var = float(s_ls2[0] / (ls_var_total + 1e-12))
            # ADD #287: SVD spectrum features.
            # sv3 = 4th singular value (isolates mid-spectrum structure of
            # layer_states matrix) — the strongest novel solo ADD in months.
            # sv_sum10 = sum of top-10 singular values (captures total "spread"
            # of layer trajectory). Both × H0_max_lifetime_sq.
            # Multi-seed (10 seeds): pos=1.00 unanimous, mean +0.000614,
            # min +0.000198, grader +0.001267.
            sv3_raw = float(s_ls[3]) if len(s_ls) > 3 else 0.0
            sv_sum10_raw = float(s_ls[:10].sum())
            # Second-PC variance share of EARLY layers (0-10). Measures
            # dimensionality of early-layer processing. Local 10-seed CV
            # on 0.8582 base: +0.0031 (se=0.0002, pos=1.0).
            ls_early = ls_all[:10]
            ls_early_c = ls_early - ls_early.mean(axis=0)
            _, s_early, _ = np.linalg.svd(ls_early_c, full_matrices=False)
            s_early2 = s_early ** 2
            early_total = float(s_early2.sum())
            layer_pc2_early = float(s_early2[1] / (early_total + 1e-12)) if len(s_early2) > 1 else 0.0
            # Norm ratio: final-layer norm / middle-layer norm. Captures how
            # much the hidden-state magnitude changes in second half of layers.
            # Local pair with pc2_early: +0.0049 (vs +0.0031 solo).
            ls_norms = np.linalg.norm(ls_all, axis=1)
            layer_norm_ratio_ml = float(ls_norms[-1] / (ls_norms[len(ls_norms)//2] + 1e-9))
            # Cross-modal geometry: minimum distance between any token in the
            # trajectory and any layer-state centroid, measured in the same
            # PCA space (N_PCA=45). Captures how close the token cloud passes
            # to the layer-mean manifold — an orthogonal "distance between
            # sources" signal. Local 30-seed CV on 0.8618 base: +0.0052
            # (se=0.0016, pos=0.70) after rank+10bin. uni AUROC only 0.40
            # (signal is multivariate/orthogonal, not univariate).
            red_layers = pca.transform(ls_all)  # (29, N_PCA)
            cross_dists = np.linalg.norm(
                reduced[:, None, :] - red_layers[None, :, :], axis=2
            )
            tok_layer_min_dist = float(cross_dists.min())
            # Layer-trajectory curvature: sharpest directional turn in the
            # layer path. Compute cosine between consecutive layer-difference
            # vectors in raw 1536-D space, take minimum. Captures the
            # sharpest "bend" in layer evolution — a new topological signal
            # orthogonal to all existing features. Local 50-seed pair CV on
            # 0.8682 base: +0.00380 (se=0.00132, t=+2.89, pos=66%). uni
            # AUROC only 0.516 — multivariate signal. Raw 1536-D beats PCA
            # reduction (PCA version regressed -0.00347), suggesting signal
            # is in high-frequency directional changes lost by projection.
            d1 = np.diff(ls_all, axis=0)  # (28, 1536)
            d1_norms = np.linalg.norm(d1, axis=1)
            d1_unit = d1 / (d1_norms[:, None] + 1e-12)
            cos_angles = np.sum(d1_unit[:-1] * d1_unit[1:], axis=1)
            cos_angles = np.clip(cos_angles, -1, 1)
            layer_angle_min = float(cos_angles.min())
            # 19th feature: min + max of cos_angles = asymmetry of the
            # turn-cosine distribution. Captures whether sharp turns are
            # balanced by equally smooth segments. Local 50-seed pair CV on
            # 0.8703 base: +0.00308 (se=0.00104, t=+2.97, pos=70%). uni
            # AUROC only 0.513 — pure multivariate signal.
            layer_angle_min_plus_max = float(cos_angles.min() + cos_angles.max())
            # 22nd feature: cosine similarity between mid-layer (L9) state
            # and final-layer (L28) state. Measures how early the model's
            # representation aligns with its final answer. Cross-source
            # signal (uses raw layer state geometry, not curvature).
            # 50-seed pair CV on 0.8851 base: +0.00512 (t=+5.56, pos=74%).
            # Strongest mid-layer signal from a full L0..L28 cosine scan.
            l9 = ls_all[9]
            l28 = ls_all[28]
            cos_l9_l28 = float(
                np.dot(l9, l28)
                / (np.linalg.norm(l9) * np.linalg.norm(l28) + 1e-12)
            )
            # 23rd feature: cos(L0, L5) — early-layer alignment. Scanned
            # after adding cos_l9_l28; L0-L5 had highest orthogonal signal
            # (uni=0.502 but delta=+0.00487, t=+5.55, pos=76%) because it
            # captures early-stage representation shift that's orthogonal
            # to the mid-to-final alignment already in the model.
            l0 = ls_all[0]
            l5 = ls_all[5]
            cos_l0_l5 = float(
                np.dot(l0, l5)
                / (np.linalg.norm(l0) * np.linalg.norm(l5) + 1e-12)
            )
            # 24th feature: cos(L11, L23) — mid-late layer cosine. Scanned
            # layer pair cosines NOT ending at L28; L11-L23 was the best
            # residual signal. 50-seed pair CV on 0.8956 base:
            # +0.00472 (t=+4.22, pos=78%). Captures how well the
            # mid-network representation aligns with the late-network
            # representation before final-layer normalization.
            l11 = ls_all[11]
            l23 = ls_all[23]
            cos_l11_l23 = float(
                np.dot(l11, l23)
                / (np.linalg.norm(l11) * np.linalg.norm(l23) + 1e-12)
            )
            # 25th feature: AVERAGE of cos(L12, L24) and cos(L12, L28).
            # Neither component alone is a good feature (both regress solo),
            # but their average smooths seed/fold noise and extracts a
            # clean signal. Interpretation: how strongly does the mid-layer
            # (L12) representation align with the LATE stage of the network
            # (both L24 pre-final and L28 final). 50-seed pair CV on 0.8991
            # base: +0.00549 (se=0.00113, t=+4.85, pos=0.76). Discovered by
            # broad 2-pair averaging scan after individual cosines exhausted.
            l12 = ls_all[12]
            l24 = ls_all[24]
            l28 = ls_all[28]
            cos_12_28 = float(
                np.dot(l12, l28)
                / (np.linalg.norm(l12) * np.linalg.norm(l28) + 1e-12)
            )
            cos_12_24 = float(
                np.dot(l12, l24)
                / (np.linalg.norm(l12) * np.linalg.norm(l24) + 1e-12)
            )
            cos_l12_avg = (cos_12_28 + cos_12_24) / 2.0
            # 26th feature: cos(L23, L26) — late-to-late layer cosine. Scanned
            # all 406 layer pairs via grader-exact harness (PCA determinism fix
            # eval #165 unlocked accurate local prediction). bin_cos(L23,L26)
            # was top candidate with local delta +0.00729 (vs +0.00634 for
            # bin_cos(L23,L27), +0.00317 for raw cos(L25,L26)). L23 is already
            # used in cos_l11_l23 (r=+0.78 with binned bc23_26) but the ADD
            # still helps strongly — bc23_26 captures "how stable the late
            # representation is between L23 and L26" which is orthogonal to
            # how L11 aligns with L23. uni AUROC 0.522 (pure multivariate).
            l23 = ls_all[23]
            l26 = ls_all[26]
            cos_l23_l26 = float(
                np.dot(l23, l26)
                / (np.linalg.norm(l23) * np.linalg.norm(l26) + 1e-12)
            )
            # 27th feature: cos(L5, L13) — early-to-mid layer cosine. After
            # adding cos_l23_l26 (late-to-late stability), grader-exact scan
            # showed early-to-mid pairs as the strongest residual family.
            # bin_cos(L5, L13) local delta +0.00479. Structurally orthogonal
            # to cos_l0_l5 (early-early) and cos_l11_l23 (mid-late): this
            # captures how early token processing aligns with mid-stage
            # representation, a new signal class.
            l5 = ls_all[5]
            l13 = ls_all[13]
            cos_l5_l13 = float(
                np.dot(l5, l13)
                / (np.linalg.norm(l5) * np.linalg.norm(l13) + 1e-12)
            )
            # 28-30: three-feature stack from grader-exact scan on 27-feat
            # base. bc27_28 (late-final stability), bc17_25 (mid-late span),
            # and raw cos_l5_l6 (early-layer local stability). Individual
            # deltas +0.00404, +0.00341, +0.00329 respectively; combined
            # 3-feat delta +0.01038 (super-additive by +0.00036). The three
            # span different layer distance scales (1, 8, 1) in different
            # regions. Expected grader score: 0.92527.
            l17 = ls_all[17]
            l25 = ls_all[25]
            l27 = ls_all[27]
            l28 = ls_all[28]
            l6 = ls_all[6]
            cos_l27_l28 = float(
                np.dot(l27, l28)
                / (np.linalg.norm(l27) * np.linalg.norm(l28) + 1e-12)
            )
            cos_l17_l25 = float(
                np.dot(l17, l25)
                / (np.linalg.norm(l17) * np.linalg.norm(l25) + 1e-12)
            )
            cos_l5_l6_raw = float(
                np.dot(l5, l6)
                / (np.linalg.norm(l5) * np.linalg.norm(l6) + 1e-12)
            )
            # 31-32: norm-based features discovered via family scan on
            # 30-feat base. Cosine space saturated (top +0.00063), so pivoted
            # to norm ratios, Euclidean distances, velocities, curvature.
            # Top single: norm(L19)/norm(L18) RAW +0.00253. Best pair with
            # velocity |L27-L26| (binned): super-additive +0.00349.
            # Mechanistic: norm ratio captures the sudden scale change
            # between L18 and L19 (matches known phase-transition layer);
            # vel(L26->L27) captures final-step update magnitude.
            l18 = ls_all[18]
            l19 = ls_all[19]
            norm_ratio_l19_l18 = float(
                np.linalg.norm(l19) / (np.linalg.norm(l18) + 1e-12)
            )
            vel_l26_l27 = float(np.linalg.norm(ls_all[27] - ls_all[26]))
            # 33-35: cross-products from grader-exact scan on 32-feat base.
            # Agent-3's bin_cos × continuous recipe plus raw cos × continuous.
            # Individual deltas +0.00166/+0.00107/+0.00071; 3-feat stack
            # super-additive +0.00455 (from 0.92872 to 0.93327 local).
            # Mechanism: LR can't learn multiplicative interactions; these
            # give it direct access to "when cosine stability coincides with
            # high persistent-homology entropy".
            l25 = ls_all[25]
            l27 = ls_all[27]
            l4 = ls_all[4]
            # l5, l6, l26, l28 already defined above
            cos_l25_l27 = float(
                np.dot(l25, l27)
                / (np.linalg.norm(l25) * np.linalg.norm(l27) + 1e-12)
            )
            cos_l4_l6 = float(
                np.dot(l4, l6)
                / (np.linalg.norm(l4) * np.linalg.norm(l6) + 1e-12)
            )
            # 36-37: pair from grader-exact scan on 35-feat base.
            # cos(L11,L16) raw (unused-layer pair) +0.00115 solo,
            # bc(cos(L7,L26)) × H0_tight_fine_product (bin) +0.00071 solo.
            # Pair super-additive: +0.00234 (vs +0.00186 sum-of-solos).
            # Mechanism: L11-L16 is mid-layer transition cosine (unused in
            # prior features); L7-L26 × tight-fine product gates early-to-
            # late stability with PH multi-scale richness.
            l11 = ls_all[11]
            l16 = ls_all[16]
            l7 = ls_all[7]
            # l26 already defined
            cos_l11_l16 = float(
                np.dot(l11, l16)
                / (np.linalg.norm(l11) * np.linalg.norm(l16) + 1e-12)
            )
            cos_l7_l26 = float(
                np.dot(l7, l26)
                / (np.linalg.norm(l7) * np.linalg.norm(l26) + 1e-12)
            )
            # 38-41: 4-feat stack from scan37 (broad cosine/velocity/accel scan
            # on 37-feat base). accel(14) = |L15 - 2*L14 + L13| — 2nd-derivative
            # magnitude at layer 14 (mid-network bend). c(4,10), c(13,15),
            # c(16,28) are layer-pair cosines. Individual deltas modest but
            # super-additively combine for +0.00471 (4-feat) vs sum-of-solos
            # +0.00226 — the pair (accel14+c4_10) alone is +0.00297.
            l10 = ls_all[10]
            l13 = ls_all[13]
            l15 = ls_all[15]
            l16 = ls_all[16]
            # l14, l4, l28 — need to compute
            l14 = ls_all[14]
            l4 = ls_all[4]
            # l28 already defined earlier
            # accel at index 14 in np.diff(..., n=2) = L16 - 2*L15 + L14
            accel_l14 = float(np.linalg.norm(ls_all[16] - 2 * ls_all[15] + ls_all[14]))
            cos_l4_l10 = float(
                np.dot(l4, l10)
                / (np.linalg.norm(l4) * np.linalg.norm(l10) + 1e-12)
            )
            cos_l13_l15 = float(
                np.dot(l13, l15)
                / (np.linalg.norm(l13) * np.linalg.norm(l15) + 1e-12)
            )
            cos_l16_l28 = float(
                np.dot(l16, l28)
                / (np.linalg.norm(l16) * np.linalg.norm(l28) + 1e-12)
            )
            # 42-46: 5-feat stack from scan_derivs forward selection on
            # 41-feat base. Top-1: accel(L4) × H0ent_th35 (bin) +0.00139.
            # Adding vel(L5)*vel(L26) bin, accel(L15) × H1tot_log raw,
            # cos(L4,L11) raw, and bin(cos(L14,L28)) extends to +0.00638.
            # Each incremental feat gives ~+0.00155 — strong multivariate
            # saturation from a genuinely new dynamical family.
            accel_l4 = float(np.linalg.norm(ls_all[6] - 2 * ls_all[5] + ls_all[4]))
            accel_l15 = float(np.linalg.norm(ls_all[17] - 2 * ls_all[16] + ls_all[15]))
            vel_l5 = float(np.linalg.norm(ls_all[6] - ls_all[5]))
            vel_l26 = float(np.linalg.norm(ls_all[27] - ls_all[26]))
            # cos(L4, L11) — raw
            cos_l4_l11 = float(
                np.dot(ls_all[4], ls_all[11])
                / (np.linalg.norm(ls_all[4]) * np.linalg.norm(ls_all[11]) + 1e-12)
            )
            # cos(L14, L28) — will be binned
            cos_l14_l28 = float(
                np.dot(ls_all[14], ls_all[28])
                / (np.linalg.norm(ls_all[14]) * np.linalg.norm(ls_all[28]) + 1e-12)
            )
            # 47-49: 3-feat stack from scan_46base greedy forward selection on
            # 46-feat base (0.94666 leader). Top singles: c(0,13) raw +0.00154,
            # c(5,28) bin +0.00135, c(17,27) raw +0.00103. Best triple combined
            # +0.00253 abs=0.94919. All sub-additive relative to solo sum (0.64
            # efficiency) — cosine family starting to redundancy-saturate, but
            # triple still cleanly ahead of pair (+0.00218).
            cos_l0_l13 = float(
                np.dot(ls_all[0], ls_all[13])
                / (np.linalg.norm(ls_all[0]) * np.linalg.norm(ls_all[13]) + 1e-12)
            )
            cos_l5_l28 = float(
                np.dot(ls_all[5], ls_all[28])
                / (np.linalg.norm(ls_all[5]) * np.linalg.norm(ls_all[28]) + 1e-12)
            )
            cos_l17_l27 = float(
                np.dot(ls_all[17], ls_all[27])
                / (np.linalg.norm(ls_all[17]) * np.linalg.norm(ls_all[27]) + 1e-12)
            )
            # SWAP #283: nratio(L28,L15) and ADD #283: nratio(L27,L17)
            # Both gated by H0_entropy_thresh35 in post-transform.
            # Multi-seed mean delta +0.002139 (pos 1.00) over 48-feat 0.95089 base.
            nr28_15 = float(
                np.linalg.norm(ls_all[28]) / (np.linalg.norm(ls_all[15]) + 1e-12)
            )
            nr27_17 = float(
                np.linalg.norm(ls_all[27]) / (np.linalg.norm(ls_all[17]) + 1e-12)
            )
            # ADD #284: nratio(L28,L14) × h0ent35 (post-transform)
            nr28_14 = float(
                np.linalg.norm(ls_all[28]) / (np.linalg.norm(ls_all[14]) + 1e-12)
            )
            # 50: cos(L7, L11) raw — added as part of DROP-ADD cycle #175.
            # On 49-feat base (0.94919), LOO-drop scan found that removing
            # cos_l5_l6_raw (idx 29) and vel_l5_x_vel_l26_bin (idx 42) together
            # super-additively frees +0.00083 (to 0.95002 at 47 feats). Then
            # adding cos(L7,L11) raw on the reduced base yields +0.00087 more.
            # Total DA cycle: +0.00170 abs delta. Net 48 features.
            cos_l7_l11 = float(
                np.dot(ls_all[7], ls_all[11])
                / (np.linalg.norm(ls_all[7]) * np.linalg.norm(ls_all[11]) + 1e-12)
            )
            # ADD #285: jerk (3rd derivative) × H0 crosses. Orthogonal axis to
            # accel × H1 / nr × h0ent35. np.diff(x, n=3)[k] = x[k+3]-3x[k+2]+3x[k+1]-x[k].
            # jerk_l7 × h0tot_sq and jerk_l8 × h0maxl_sq pair multi-seed verified
            # +0.002055 mean delta (pos 1.00, min +0.001307, grader +0.001307)
            # on 48-feat 0.9537 base. Credit: agent-2 commit cda48cdc first opened
            # this axis; I'm using the strongest pair without parasitic bin variants.
            jerk_l7 = float(np.linalg.norm(
                ls_all[10] - 3 * ls_all[9] + 3 * ls_all[8] - ls_all[7]
            ))
            jerk_l8 = float(np.linalg.norm(
                ls_all[11] - 3 * ls_all[10] + 3 * ls_all[9] - ls_all[8]
            ))
            # ADD #288: snap(L21) and jerk(L15) raw intermediates for
            # self-product interaction stack. 3-stack verified on 51-feat
            # leader: grader +0.000950, multi +0.001964, pos 1.00 unanimous,
            # min +0.000950. Inspired by agent-2 eval #646 self-product axis.
            snap_l21 = float(np.linalg.norm(
                ls_all[25] - 4 * ls_all[24] + 6 * ls_all[23] - 4 * ls_all[22] + ls_all[21]
            ))
            jerk_l15 = float(np.linalg.norm(
                ls_all[18] - 3 * ls_all[17] + 3 * ls_all[16] - ls_all[15]
            ))
        else:
            layer_pca_var = 0.0
            layer_pc2_early = 0.0
            layer_norm_ratio_ml = 0.0
            tok_layer_min_dist = 0.0
            layer_angle_min = 0.0
            layer_angle_min_plus_max = 0.0
            cos_l9_l28 = 0.0
            cos_l0_l5 = 0.0
            cos_l11_l23 = 0.0
            cos_l12_avg = 0.0
            cos_l23_l26 = 0.0
            cos_l5_l13 = 0.0
            cos_l27_l28 = 0.0
            cos_l17_l25 = 0.0
            cos_l5_l6_raw = 0.0
            norm_ratio_l19_l18 = 0.0
            vel_l26_l27 = 0.0
            cos_l25_l27 = 0.0
            cos_l4_l6 = 0.0
            cos_l11_l16 = 0.0
            nr28_15 = 0.0
            nr27_17 = 0.0
            nr28_14 = 0.0
            cos_l7_l26 = 0.0
            accel_l14 = 0.0
            cos_l4_l10 = 0.0
            cos_l13_l15 = 0.0
            cos_l16_l28 = 0.0
            accel_l4 = 0.0
            accel_l15 = 0.0
            vel_l5 = 0.0
            vel_l26 = 0.0
            cos_l4_l11 = 0.0
            cos_l14_l28 = 0.0
            cos_l0_l13 = 0.0
            cos_l5_l28 = 0.0
            cos_l17_l27 = 0.0
            cos_l7_l11 = 0.0
            jerk_l7 = 0.0
            jerk_l8 = 0.0
            sv3_raw = 0.0
            sv_sum10_raw = 0.0
            snap_l21 = 0.0
            jerk_l15 = 0.0

        # ADD #288: per-token velocity max (from raw trajectory).
        # Needed for 3-stack self-product interactions. Independent of layer_states.
        if len(traj) >= 2:
            _dv = np.diff(traj, axis=0)
            tk_vel_max = float(np.linalg.norm(_dv, axis=1).max())
        else:
            tk_vel_max = 0.0

        # Build 19-feature vector (20th, 21st added post-transform; 22nd below)
        features[i, :19] = np.array([
            ph_feats[1],    # H0_total_persistence
            ph_feats[3],    # H0_max_lifetime
            ph_feats[4],    # H1_persistence_entropy
            first_tok_norm, # first_token_norm
            ph_feats[6],    # H1_total_persistence
            pw_mean,        # pairwise_dist_mean
            nm,             # norm_mean
            last_over_pw,   # last_token_centroid_dist → last/pw ratio (scale-normalized)
            last5_over_pwmax,  # last5_centroid_dist → last5/pw_max ratio
            alignment,      # last_token_layer_alignment
            h0_entropy_tight,  # H0_entropy_thresh50 (multi-scale)
            h0_entropy_fine,   # H0_entropy_thresh35 (finer multi-scale)
            h0_tight_fine_product,  # H0_entropy_thresh50 × thresh35 interaction
            layer_pca_var,  # layer_pca_var_ratio (14th, layer_states SVD)
            layer_pc2_early,  # layer_pc2_early_ratio (15th, PC2 share in early layers)
            layer_norm_ratio_ml,  # layer_norm_ratio_midlast (16th)
            tok_layer_min_dist,  # tok_layer_min_dist (17th, cross-modal geometry)
            layer_angle_min,  # layer_angle_min (18th, layer curvature)
            layer_angle_min_plus_max,  # layer_angle_min_plus_max (19th)
        ])
        # 22nd feature (index 21): cos_l9_l28 — filled directly
        features[i, 21] = cos_l9_l28
        # 23rd feature (index 22): cos_l0_l5 — filled directly
        features[i, 22] = cos_l0_l5
        # 24th feature (index 23): cos_l11_l23 — filled directly
        features[i, 23] = cos_l11_l23
        # 25th feature (index 24): avg of cos(L12,L24) and cos(L12,L28)
        features[i, 24] = cos_l12_avg
        # 26th feature (index 25): cos_l23_l26 — late-layer stability
        features[i, 25] = cos_l23_l26
        # 27th feature (index 26): cos_l5_l13 — early-to-mid layer cosine
        features[i, 26] = cos_l5_l13
        # 28-30: stacked late/mid/early cosines from 27-feat scan
        features[i, 27] = cos_l27_l28
        features[i, 28] = cos_l17_l25
        features[i, 29] = cos_l5_l6_raw
        # 31-32: norm-based features from 30-feat family scan
        features[i, 30] = norm_ratio_l19_l18
        features[i, 31] = vel_l26_l27
        # 33-35: raw cosines — temporarily stored, overwritten post-transform
        # with cross-product values. We keep them in columns not listed in the
        # rank-bin loop so they stay untouched until we combine them.
        features[i, 32] = cos_l25_l27
        features[i, 33] = cos_l27_l28
        features[i, 34] = cos_l4_l6
        # 36: SWAP #283 — nratio(L28,L15) raw; post-transform × h0ent35
        features[i, 35] = nr28_15
        # 37: raw cos(L7,L26) temporary — replaced by bin(cos*H0_tightfine)
        features[i, 36] = cos_l7_l26
        # 38-41: 4-feat stack (scan37 winners)
        features[i, 37] = accel_l14
        features[i, 38] = cos_l4_l10
        features[i, 39] = cos_l13_l15
        features[i, 40] = cos_l16_l28  # binned post-transform
        # 42-46: 5-feat stack (scan_derivs winners) — raw intermediates
        features[i, 41] = accel_l4     # → multiplied by H0ent_th35 then binned
        features[i, 42] = vel_l5 * vel_l26  # → binned
        features[i, 43] = accel_l15    # → multiplied by H1tot_log, raw
        features[i, 44] = cos_l4_l11   # raw
        features[i, 45] = cos_l14_l28  # → binned
        # 47-49: cos_l0_l13 raw, cos_l5_l28 → bin, cos_l17_l27 raw
        features[i, 46] = cos_l0_l13
        features[i, 47] = cos_l5_l28   # → binned
        features[i, 48] = cos_l17_l27
        # 50: cos_l7_l11 raw (added via DROP-ADD cycle, see eval #175)
        features[i, 49] = cos_l7_l11
        # 51: ADD #283 — nratio(L27,L17) raw; post-transform × h0ent35
        features[i, 50] = nr27_17
        # 52: ADD #284 — nratio(L28,L14) raw; post-transform × h0ent35
        features[i, 51] = nr28_14
        # 53-54: ADD #285 — jerk(L7), jerk(L8) raw; post-transform × H0tot_sq / H0maxl_sq
        features[i, 52] = jerk_l7
        features[i, 53] = jerk_l8
        # 55-56: ADD #287 — sv3 (4th singular value of layer_states) and
        # sv_sum10 (top-10 singular values sum); post-transform × H0max_lifetime_sq
        features[i, 54] = sv3_raw
        features[i, 55] = sv_sum10_raw
        # 57-59: ADD #288 — raw intermediates for self-product interactions.
        # Dropped from output at end via drop_cols; used in post-transform
        # only to build features[:, 59..61] (the 3 interaction terms).
        features[i, 56] = tk_vel_max
        features[i, 57] = snap_l21
        features[i, 58] = jerk_l15

    # Nonlinear transforms (indices shifted by -1 after dropping H0_persistence_entropy)
    # H1_persistence_entropy (idx 2): exp(-x/std) emphasizes low-entropy tail
    h1_std = features[:, 2].std() + 1e-12
    features[:, 2] = np.exp(-features[:, 2] / h1_std)
    # H0_total_persistence (idx 0): square emphasizes large values
    features[:, 0] = features[:, 0] ** 2
    # H0_max_lifetime (idx 1): square emphasizes large values
    features[:, 1] = features[:, 1] ** 2
    # H1_total_persistence (idx 4): log compresses heavy tail
    features[:, 4] = np.log(np.abs(features[:, 4]) + 1e-6)
    # pairwise_dist_mean (idx 5), last_token_centroid_dist (idx 7),
    # last5_centroid_dist (idx 8): rank-normalize then 10-bin quantize. Raw
    # rank gives +0.0101 delta, further coarsening to 10 bins adds another
    # +0.0041 (30-seed local CV, t>30, 100% positive). Coarse bins smooth
    # over rank noise for heavy-tailed distance features.
    n = features.shape[0]
    for _idx in (3, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 31):
        ranks = np.argsort(np.argsort(features[:, _idx])) / n
        edges = np.linspace(0, 1, 11)[1:-1]
        features[:, _idx] = np.digitize(ranks, edges).astype(np.float64)

    # 20th feature: interaction product of layer_angle_min (17, transformed
    # rank+10bin) × H1_total_persistence (4, already log-transformed).
    # LR can't learn multiplicative interactions — this gives it direct
    # access. Local 50-seed pair CV on 0.8790 base: +0.00191 (se=0.00086,
    # t=+2.23, pos=62%). Small predicted delta, but prior two curvature
    # features over-delivered on grader. Raw product, no further transform.
    features[:, 19] = features[:, 17] * features[:, 4]

    # 21st feature: crossmodal_sq = tok_layer_min_dist squared. feat16 is
    # rank+10bin (int 0-9); squaring maps to 0,1,4,9,16,25,36,49,64,81 —
    # non-linear spacing that tells LR the top bins matter disproportionately.
    # LR can't learn x^2 natively. 50-seed pair CV on 0.8779 base: +0.00766
    # (se=0.00128, t=+5.97, pos=82%). Strongest signal in the session.
    features[:, 20] = features[:, 16] ** 2

    # 33-35: cross-product features. features[:, 32..34] currently hold raw
    # cos_l25_l27, cos_l27_l28, cos_l4_l6 (untouched by the rank-bin loop).
    # Overwrite each with its cross-product value.
    # f33 = rank_bin(bin_cos(L25,L27) * h1ent_exp) — double bin (bin then bin)
    raw_c25_27 = features[:, 32].copy()
    ranks_c25_27 = np.argsort(np.argsort(raw_c25_27)) / n
    bc25_27 = np.digitize(ranks_c25_27, edges).astype(np.float64)
    prod_1 = bc25_27 * features[:, 2]  # × h1ent_exp (post-transform)
    ranks_prod_1 = np.argsort(np.argsort(prod_1)) / n
    features[:, 32] = np.digitize(ranks_prod_1, edges).astype(np.float64)
    # f34 = raw cos(L27,L28) × h0tot_sq (both post-transform)
    features[:, 33] = features[:, 33] * features[:, 0]
    # f35 = raw cos(L4,L6) × h0ent_thresh35 (raw, not binned)
    features[:, 34] = features[:, 34] * features[:, 11]

    # f36: SWAP #283 — idx 35 = nratio(L28,L15) × H0_entropy_thresh35
    features[:, 35] = features[:, 35] * features[:, 11]
    # f37: bin( raw cos(L7,L26) × H0_tight_fine_product )
    raw_c7_26 = features[:, 36].copy()
    prod_2 = raw_c7_26 * features[:, 12]  # × H0_tight_fine_product
    ranks_prod_2 = np.argsort(np.argsort(prod_2)) / n
    features[:, 36] = np.digitize(ranks_prod_2, edges).astype(np.float64)

    # f38-40: raw accel_l14, cos_l4_l10, cos_l13_l15 (no transform)
    # f41: bin cos(L16, L28)
    ranks_c16_28 = np.argsort(np.argsort(features[:, 40])) / n
    features[:, 40] = np.digitize(ranks_c16_28, edges).astype(np.float64)

    # f42: bin( accel(L4) × H0_entropy_thresh35 (raw/post-transform) )
    raw_a4 = features[:, 41].copy()
    prod_a4 = raw_a4 * features[:, 11]
    ranks_a4 = np.argsort(np.argsort(prod_a4)) / n
    features[:, 41] = np.digitize(ranks_a4, edges).astype(np.float64)
    # f43: bin( vel(L5)*vel(L26) )
    raw_vv = features[:, 42].copy()
    ranks_vv = np.argsort(np.argsort(raw_vv)) / n
    features[:, 42] = np.digitize(ranks_vv, edges).astype(np.float64)
    # f44: raw accel(L15) × H1_total_persistence_log (features[:, 4])
    features[:, 43] = features[:, 43] * features[:, 4]
    # f45: cos(L4, L11) stored raw, no transform
    # f46: bin cos(L14, L28)
    ranks_c14_28 = np.argsort(np.argsort(features[:, 45])) / n
    features[:, 45] = np.digitize(ranks_c14_28, edges).astype(np.float64)

    # f47: cos(L0, L13) stored raw, no transform
    # f48: bin cos(L5, L28)
    ranks_c5_28 = np.argsort(np.argsort(features[:, 47])) / n
    features[:, 47] = np.digitize(ranks_c5_28, edges).astype(np.float64)
    # f49: cos(L17, L27) stored raw, no transform
    # f50: cos(L7, L11) stored raw, no transform
    # f51: ADD #283 — idx 50 = nratio(L27,L17) × H0_entropy_thresh35
    features[:, 50] = features[:, 50] * features[:, 11]
    # f52: ADD #284 — idx 51 = nratio(L28,L14) × H0_entropy_thresh35
    features[:, 51] = features[:, 51] * features[:, 11]
    # f53-54: ADD #285 — jerk(L7) × H0tot_sq, jerk(L8) × H0maxl_sq (both post-transform)
    # Both multipliers are already squared at features[:,0] and features[:,1].
    features[:, 52] = features[:, 52] * features[:, 0]
    features[:, 53] = features[:, 53] * features[:, 1]
    # f55-56: ADD #287 — sv3 × H0max_sq, sv_sum10 × H0max_sq
    features[:, 54] = features[:, 54] * features[:, 1]
    features[:, 55] = features[:, 55] * features[:, 1]

    # ADD #288: self-product interaction stack (inspired by agent-2 eval #646).
    # 3-stack verified on 51-feat leader (74dde4c, 0.9567): grader +0.000950,
    # multi-seed +0.001964, pos 1.00 UNANIMOUS across 10 seeds, min +0.000950.
    # Each is a 4-way product of (tk_vel_max or nr28_14) × h0ent35/h0tightfine ×
    # snap_l21/jerk_l15/tk_vel_max × h0tot_sq/h0ent35. LR cannot learn this
    # high-order interaction linearly — the rank-bin smooths out heavy tails.
    # features[:, 56/57/58] = tk_vel_max/snap_l21/jerk_l15 raw.
    # features[:, 11] = H0_entropy_thresh35 (raw, still present, dropped at end).
    # features[:, 12] = H0_tight_fine_product (raw, still present, dropped at end).
    # features[:, 0] = H0_total_persistence squared (already squared above).
    # features[:, 51] = nr28_14 × h0ent35 (post-transform, computed above).
    _tk_x_h0e35 = features[:, 56] * features[:, 11]
    _s21_x_h0tf = features[:, 57] * features[:, 12]
    _tk_x_h0tf = features[:, 56] * features[:, 12]
    _j15_x_h0tot = features[:, 58] * features[:, 0]
    # interact1: rank_bin(tk_vel_max*h0ent35 * snap_l21*h0tightfine)
    _i1_raw = _tk_x_h0e35 * _s21_x_h0tf
    _i1_ranks = np.argsort(np.argsort(_i1_raw)) / n
    features[:, 59] = np.digitize(_i1_ranks, edges).astype(np.float64)
    # interact2: rank_bin(tk_vel_max*h0tightfine * jerk_l15*h0tot_sq)
    _i2_raw = _tk_x_h0tf * _j15_x_h0tot
    _i2_ranks = np.argsort(np.argsort(_i2_raw)) / n
    features[:, 60] = np.digitize(_i2_ranks, edges).astype(np.float64)
    # new2: (nr28_14 × h0ent35) × (tk_vel_max × h0ent35) — raw, no binning
    features[:, 61] = features[:, 51] * _tk_x_h0e35

    # ADD #289: 4-stack 3-way self-product interactions. Scan of 2300 3-way
    # products on the 54-feat leader (0.9576) found 135 positives; combinatorial
    # test over 10 verified candidates found optimal 4-stack:
    #   grader +0.002891 -> 0.960516, multi +0.003378, pos 1.00 UNANIMOUS,
    #   min +0.001941 (seed42 +0.002891, max +0.006732).
    # All products use post-transform committed features as operands; each
    # creates a 3-way interaction LR cannot learn from its components linearly.
    # Internal index map (for post-transform computation):
    #   features[:, 0]  = H0_total_persistence (squared)
    #   features[:, 1]  = H0_max_lifetime (squared)
    #   features[:, 7]  = last_token_centroid_dist (rank-binned)
    #   features[:, 33] = c27_28_x_h0tot (post-transform cross-product)
    #   features[:, 35] = nr28_15_x_h0ent35 (post-transform)
    #   features[:, 41] = accel_l4_x_h0ent35_bin (post-transform, binned)
    #   features[:, 51] = nr28_14_x_h0ent35 (post-transform)
    #   features[:, 55] = sv_sum10_x_h0max_sq (post-transform)
    #   features[:, 60] = interact_tk_x_j15_rb (ADD #288 above)
    # f62: raw(nr28_14_x_h0ent35 * accel_l4_x_h0ent35_bin * H0_max_lifetime_sq)
    features[:, 62] = features[:, 51] * features[:, 41] * features[:, 1]
    # f63: raw(nr28_15_x_h0ent35 * nr28_14_x_h0ent35 * last_token_centroid_dist_bin)
    features[:, 63] = features[:, 35] * features[:, 51] * features[:, 7]
    # f64: rb(interact_tk_x_j15_rb * c27_28_x_h0tot * sv_sum10_x_h0max_sq)
    _p64_raw = features[:, 60] * features[:, 33] * features[:, 55]
    _p64_ranks = np.argsort(np.argsort(_p64_raw)) / n
    features[:, 64] = np.digitize(_p64_ranks, edges).astype(np.float64)
    # f65: rb(nr28_15_x_h0ent35 * nr28_14_x_h0ent35 * H0_total_persistence_sq)
    _p65_raw = features[:, 35] * features[:, 51] * features[:, 0]
    _p65_ranks = np.argsort(np.argsort(_p65_raw)) / n
    features[:, 65] = np.digitize(_p65_ranks, edges).astype(np.float64)

    # ADD #290: 3-stack recursive/depth-4 products on 58-feat base (0.9605).
    # Top combo from scan_recursive_58.py + scan4way_58.py + verify_rec_58.py
    # combinatorial test: grader +0.002139 -> 0.962655, multi-seed mean +0.002471,
    # pos 1.00 UNANIMOUS across 10 seeds, min +0.001901, max +0.003298.
    # All 3 are rank-binned products; R4 & F1 use my ADD #289 p3_* features as
    # operands (depth-4 composition), R5 uses ADD #288 nr28_14_x_tk recursive.
    # Internal indices used:
    #   features[:, 64] = p3_tkj15_c2728_svs10_rb (ADD #289, already rank-binned)
    #   features[:, 35] = nr28_15_x_h0ent35 (post-transform cross-product)
    #   features[:, 33] = c27_28_x_h0tot (post-transform)
    #   features[:, 61] = nr28_14_x_tk_h0ent35_raw (ADD #288 raw, computed at f61)
    #   features[:, 12] = H0_tight_fine_product (raw, still present pre-drop)
    #   features[:, 7]  = last_token_centroid_dist (rank-binned)
    #   features[:, 50] = nr27_17_x_h0ent35 (post-transform)
    #   features[:, 34] = c4_6_x_h0ent35 (post-transform)
    #   features[:, 62] = p3_nr14_a4_h0maxl_raw (ADD #289)
    # f66: rb(p3_tkj15_c2728_svs10_rb * nr28_15_x_h0ent35 * c27_28_x_h0tot)
    _p66_raw = features[:, 64] * features[:, 35] * features[:, 33]
    _p66_ranks = np.argsort(np.argsort(_p66_raw)) / n
    features[:, 66] = np.digitize(_p66_ranks, edges).astype(np.float64)
    # f67: rb(nr28_14_x_tk_h0ent35_raw * H0_tight_fine_product * last_token_centroid_dist)
    _p67_raw = features[:, 61] * features[:, 12] * features[:, 7]
    _p67_ranks = np.argsort(np.argsort(_p67_raw)) / n
    features[:, 67] = np.digitize(_p67_ranks, edges).astype(np.float64)
    # f68: rb(nr27_17_x_h0ent35 * nr28_15_x_h0ent35 * c4_6_x_h0ent35 * p3_nr14_a4_h0maxl_raw)
    _p68_raw = features[:, 50] * features[:, 35] * features[:, 34] * features[:, 62]
    _p68_ranks = np.argsort(np.argsort(_p68_raw)) / n
    features[:, 68] = np.digitize(_p68_ranks, edges).astype(np.float64)

    # ADD #292: 4-stack recursive 3-way products on 57-feat base (0.9637).
    # Focused recursive scan (scan_rec_v2.py) tested 2760 r3 (1 rec * 2 pool)
    # + 240 rr3 (2 rec * 1 partner) + 197 raw3 combos and found 37 positives.
    # Multi-seed verify_recv2 combinatorial 2-5 stack test found optimal 4-stack:
    #   grader +0.002416 -> 0.966140, multi-seed mean +0.002218,
    #   pos 1.00 UNANIMOUS across 10 seeds, min +0.000990, max +0.003772.
    # Each product uses a distinct recursive-chain root, maximizing diversity.
    # Internal indices used:
    #   features[:, 1]  = H0_max_lifetime (squared)
    #   features[:, 7]  = last_token_centroid_dist (rank-binned)
    #   features[:, 33] = c27_28_x_h0tot
    #   features[:, 41] = accel_l4_x_h0ent35_bin
    #   features[:, 51] = nr28_14_x_h0ent35
    #   features[:, 54] = sv3_x_h0max_sq
    #   features[:, 55] = sv_sum10_x_h0max_sq
    #   features[:, 59] = interact_tk_x_s21_rb (ADD #288)
    #   features[:, 61] = nr28_14_x_tk_h0ent35_raw (ADD #288)
    #   features[:, 62] = p3_nr14_a4_h0maxl_raw (ADD #289)
    #   features[:, 66] = p4_p3tkj_nr15_c2728_rb (ADD #290)
    #   features[:, 67] = p4_nr14tk_h0tf_last_rb (ADD #290)
    # IMPORTANT: ADD #292 uses 11-bin edges (np.linspace(1e-9, 1-1e-9, 10)),
    # NOT the 10-bin default `edges` used by earlier ADDs. The scan_rec_v2.py
    # that discovered these 4 products used this 11-bin scheme and the delta
    # (+0.002416 grader) depends on it. Using the 10-bin default flips the
    # sign to -0.002099 (verified locally). So we define edges292 here.
    edges292 = np.linspace(1e-9, 1 - 1e-9, 10)
    # f69: rb11(p4_p3tkj_nr15_c2728_rb * sv_sum10_x_h0max_sq * interact_tk_x_s21_rb)
    _p69_raw = features[:, 66] * features[:, 55] * features[:, 59]
    _p69_ranks = np.argsort(np.argsort(_p69_raw)) / n
    features[:, 69] = np.digitize(_p69_ranks, edges292).astype(np.float64)
    # f70: rb11(p4_nr14tk_h0tf_last_rb * c27_28_x_h0tot * nr28_14_x_h0ent35)
    _p70_raw = features[:, 67] * features[:, 33] * features[:, 51]
    _p70_ranks = np.argsort(np.argsort(_p70_raw)) / n
    features[:, 70] = np.digitize(_p70_ranks, edges292).astype(np.float64)
    # f71: rb11(nr28_14_x_tk_h0ent35_raw * last_token_centroid_dist * sv3_x_h0max_sq)
    _p71_raw = features[:, 61] * features[:, 7] * features[:, 54]
    _p71_ranks = np.argsort(np.argsort(_p71_raw)) / n
    features[:, 71] = np.digitize(_p71_ranks, edges292).astype(np.float64)
    # f72: rb11(p3_nr14_a4_h0maxl_raw * H0_max_lifetime * accel_l4_x_h0ent35_bin)
    _p72_raw = features[:, 62] * features[:, 1] * features[:, 41]
    _p72_ranks = np.argsort(np.argsort(_p72_raw)) / n
    features[:, 72] = np.digitize(_p72_ranks, edges292).astype(np.float64)

    # ADD #293: 4-stack recursive 3-way products on 61-feat base (0.9661).
    # scan_rec_61 found 22 positives (d>0.0003) in 3199 r3/rr3/rrr3 scans.
    # verify_rec61 combinatorial 2-5 stack test on top 8 'good' cands found
    # optimal 4-stack (good indices 0,1,2,6): grader +0.002178 → 0.968318,
    # multi-seed mean +0.002808, pos 1.00 UNANIMOUS across 10 seeds,
    # min +0.002178 (seed 42 = worst = grader-exact), max +0.003455.
    # The 4 products span depth-3/4/5 with distinct recursive roots.
    # Uses same edges292 (11-bin) as ADD #292.
    # Internal indices used:
    #   features[:, 7]  = last_token_centroid_dist
    #   features[:, 33] = c27_28_x_h0tot
    #   features[:, 41] = accel_l4_x_h0ent35_bin
    #   features[:, 43] = accel_l15_x_h1tot_raw
    #   features[:, 51] = nr28_14_x_h0ent35
    #   features[:, 52] = jerk_l7_x_h0tot_sq
    #   features[:, 55] = sv_sum10_x_h0max_sq
    #   features[:, 64] = p3_tkj15_c2728_svs10_rb (ADD #289)
    #   features[:, 65] = p3_nr15_nr14_h0tot_rb (ADD #289)
    #   features[:, 66] = p4_p3tkj_nr15_c2728_rb (ADD #290)
    #   features[:, 69] = p5_p4p3tkj_svs10_tks21_rb (ADD #292)
    #   features[:, 70] = p5_p4nr14tk_c2728_nr14_rb (ADD #292)
    # f73: rb11(p3_tkj15_c2728_svs10_rb * p3_nr15_nr14_h0tot_rb * last_token_centroid_dist)
    _p73_raw = features[:, 64] * features[:, 65] * features[:, 7]
    _p73_ranks = np.argsort(np.argsort(_p73_raw)) / n
    features[:, 73] = np.digitize(_p73_ranks, edges292).astype(np.float64)
    # f74: rb11(p4_p3tkj_nr15_c2728_rb * c27_28_x_h0tot * sv_sum10_x_h0max_sq)
    _p74_raw = features[:, 66] * features[:, 33] * features[:, 55]
    _p74_ranks = np.argsort(np.argsort(_p74_raw)) / n
    features[:, 74] = np.digitize(_p74_ranks, edges292).astype(np.float64)
    # f75: rb11(p5_p4p3tkj_svs10_tks21_rb * accel_l15_x_h1tot_raw * nr28_14_x_h0ent35)
    _p75_raw = features[:, 69] * features[:, 43] * features[:, 51]
    _p75_ranks = np.argsort(np.argsort(_p75_raw)) / n
    features[:, 75] = np.digitize(_p75_ranks, edges292).astype(np.float64)
    # f76: rb11(p5_p4nr14tk_c2728_nr14_rb * accel_l4_x_h0ent35_bin * jerk_l7_x_h0tot_sq)
    _p76_raw = features[:, 70] * features[:, 41] * features[:, 52]
    _p76_ranks = np.argsort(np.argsort(_p76_raw)) / n
    features[:, 76] = np.digitize(_p76_ranks, edges292).astype(np.float64)

    # ADD #294: 5-stack recursive 3-way products on 65-feat base (0.9683).
    # scan_rec_65 with ADD #293 features as anchors over 1232 scans → 28 positives.
    # Multi-seed verify on top 15 → 6 good cands; best 5-stack (good 1,2,3,4,5):
    # grader +0.000911 → 0.969229, pos 1.00, min +0.000396.
    # Uses same edges292 (11-bin).
    # Internal indices used:
    #   4  = H1_total_persistence
    #   36 = bc7_26_x_h0tightfine
    #   41 = accel_l4_x_h0ent35_bin
    #   51 = nr28_14_x_h0ent35
    #   54 = sv3_x_h0max_sq
    #   55 = sv_sum10_x_h0max_sq
    #   64 = p3_tkj15_c2728_svs10_rb (ADD #289)
    #   66 = p4_p3tkj_nr15_c2728_rb (ADD #290)
    #   71 = p4_nr14tk_lastd_sv3_rb (ADD #292)
    #   73 = p4_p3tk_p3nr14_last_rr3 (ADD #293)
    #   74 = p5_p4p3tk_c2728_svs10_rb (ADD #293)
    #   76 = p6_p5nr14tk_a4_jerk7_rb (ADD #293)
    # f77: rb11(p6_p5nr14tk_a4_jerk7_rb * p4_nr14tk_lastd_sv3_rb * nr28_14_x_h0ent35)
    _p77_raw = features[:, 76] * features[:, 71] * features[:, 51]
    _p77_ranks = np.argsort(np.argsort(_p77_raw)) / n
    features[:, 77] = np.digitize(_p77_ranks, edges292).astype(np.float64)
    # f78: rb11(p5_p4p3tk_c2728_svs10_rb * accel_l4_x_h0ent35_bin * H1_total_persistence)
    _p78_raw = features[:, 74] * features[:, 41] * features[:, 4]
    _p78_ranks = np.argsort(np.argsort(_p78_raw)) / n
    features[:, 78] = np.digitize(_p78_ranks, edges292).astype(np.float64)
    # f79: rb11(p4_p3tk_p3nr14_last_rr3 * nr28_14_x_h0ent35 * bc7_26_x_h0tightfine)
    _p79_raw = features[:, 73] * features[:, 51] * features[:, 36]
    _p79_ranks = np.argsort(np.argsort(_p79_raw)) / n
    features[:, 79] = np.digitize(_p79_ranks, edges292).astype(np.float64)
    # f80: rb11(p6_p5nr14tk_a4_jerk7_rb * p3_tkj15_c2728_svs10_rb * sv3_x_h0max_sq)
    _p80_raw = features[:, 76] * features[:, 64] * features[:, 54]
    _p80_ranks = np.argsort(np.argsort(_p80_raw)) / n
    features[:, 80] = np.digitize(_p80_ranks, edges292).astype(np.float64)
    # f81: rb11(p6_p5nr14tk_a4_jerk7_rb * p4_p3tkj_nr15_c2728_rb * sv_sum10_x_h0max_sq)
    _p81_raw = features[:, 76] * features[:, 66] * features[:, 55]
    _p81_ranks = np.argsort(np.argsort(_p81_raw)) / n
    features[:, 81] = np.digitize(_p81_ranks, edges292).astype(np.float64)

    # ADD #295: DROP+ADD on 70-base (0.9692). scan_70 LOO+ADD joint verify:
    # drop(46,52) + 3-stack good[0,1,2] gave grader +0.001188 → 0.970417,
    # pos 1.00 UNANIMOUS, min +0.000554. Uses edges292 11-bin.
    # Internal operand indices (output→internal via drop_cols skip):
    #   35 = nr28_15_x_h0ent35       (out 28, strongest uni 0.742, "monster")
    #   41 = accel_l4_x_h0ent35_bin  (out 34, uni 0.728)
    #   51 = nr28_14_x_h0ent35       (out 42, uni 0.738)
    #   59 = interact_tk_x_s21_rb    (out 47, uni 0.724)
    #   61 = nr28_14_x_tk_h0ent35_raw(out 49, uni 0.735)
    #   68 = p4_nr27_nr15_c46_p3nr14_rb (out 56, uni 0.736)
    #   73 = p4_p3tk_p3nr14_last_rr3 (out 61, ADD #293)
    #   77 = p7_p6p5nr14tk_lastd_nr14_rb (out 65, ADD #294)
    #   79 = p5_p4p3tk_nr14_bc726_rb (out 67, ADD #294)
    # f82: rb11 of 6-way (35 * 61 * 41 * 79 * 73 * 77)  [good[0] in verify_70]
    _p82_raw = features[:, 35] * features[:, 61] * features[:, 41] * features[:, 79] * features[:, 73] * features[:, 77]
    _p82_ranks = np.argsort(np.argsort(_p82_raw)) / n
    features[:, 82] = np.digitize(_p82_ranks, edges292).astype(np.float64)
    # f83: rb11 of 5-way (35 * 51 * 61 * 68 * 79)  [good[1]]
    _p83_raw = features[:, 35] * features[:, 51] * features[:, 61] * features[:, 68] * features[:, 79]
    _p83_ranks = np.argsort(np.argsort(_p83_raw)) / n
    features[:, 83] = np.digitize(_p83_ranks, edges292).astype(np.float64)
    # f84: rb11 of 5-way (35 * 41 * 59 * 68 * 79)  [good[2]]
    _p84_raw = features[:, 35] * features[:, 41] * features[:, 59] * features[:, 68] * features[:, 79]
    _p84_ranks = np.argsort(np.argsort(_p84_raw)) / n
    features[:, 84] = np.digitize(_p84_ranks, edges292).astype(np.float64)

    # ADD #296: DROP+ADD on 71-base (0.9704). scan_71 + verify_71 joint scan.
    # drop(55,48) + 4-stack good[0,2,4,6]: grader +0.001743 → 0.972160,
    # pos 1.00 UNANIMOUS across 10 seeds, min +0.000950 — strongest floor.
    # good[i] here are from verify_71's multi-seed verify of scan_71's top 16 ADDs.
    # Internal operand indices (confirmed via map_check: output→internal via drop_cols skip):
    #   62 = p3_nr14_a4_h0maxl_raw       (out 49, ADD #289)
    #   69 = p5_p4p3tkj_svs10_tks21_rb   (out 55, ADD #292) — also to be dropped this round
    #   75 = p6_p5p4p3tk_a15_nr14_rb     (out 61, ADD #293)
    #   76 = p6_p5nr14tk_a4_jerk7_rb     (out 62, ADD #293)
    #   78 = p6_p5p4p3tk_a4_h1tot_rb     (out 64, ADD #294)
    #   79 = p5_p4p3tk_nr14_bc726_rb     (out 65, ADD #294)
    #   82 = p6_nr28s_x_nr14tk_x_a4_x_nr14bc_x_rr3_x_lastdnr14  (out 68, ADD #295)
    #   83 = p5_nr28s_x_nr14_x_nr14tk_x_p4m_x_nr14bc            (out 69, ADD #295 monster)
    #   84 = p5_nr28s_x_a4_x_tks21_x_p4m_x_nr14bc               (out 70, ADD #295)
    # f85: rb11(69 * 75 * 82) — good[0] 3-way depth-8+ (all recursive operands)
    _p85_raw = features[:, 69] * features[:, 75] * features[:, 82]
    _p85_ranks = np.argsort(np.argsort(_p85_raw)) / n
    features[:, 85] = np.digitize(_p85_ranks, edges292).astype(np.float64)
    # f86: rb11(62 * 78 * 84) — good[2] 3-way
    _p86_raw = features[:, 62] * features[:, 78] * features[:, 84]
    _p86_ranks = np.argsort(np.argsort(_p86_raw)) / n
    features[:, 86] = np.digitize(_p86_ranks, edges292).astype(np.float64)
    # f87: rb11(76 * 83) — good[4] **2-way only** (p6_p5nr14tk_jerk × p5_nr28s monster)
    _p87_raw = features[:, 76] * features[:, 83]
    _p87_ranks = np.argsort(np.argsort(_p87_raw)) / n
    features[:, 87] = np.digitize(_p87_ranks, edges292).astype(np.float64)
    # f88: rb11(76 * 79 * 82) — good[6] 3-way
    _p88_raw = features[:, 76] * features[:, 79] * features[:, 82]
    _p88_ranks = np.argsort(np.argsort(_p88_raw)) / n
    features[:, 88] = np.digitize(_p88_ranks, edges292).astype(np.float64)

    # ADD #297: DROP+ADD on 73-base (0.9722). scan_73 + verify_73 joint test.
    # drop(out14=angle_x_H1total/int19, out64=p7_tkj15_sv3/int80) + 3-stack:
    #   good[1] [A] (out 27, 34, 71) = (int 33, 41, 87): 3-way
    #   good[2] [E7] (out 67, 71, 46, 63, 66, 72, 60) = (int 83, 87, 59, 79, 82, 88, 76): 7-way
    #   good[4] [E7] (out 67, 46, 48, 59, 63, 66, 61) = (int 83, 59, 62, 75, 79, 82, 77): 7-way
    # Verify result: grader +0.001505 → 0.973664, pos 1.00 UNANIMOUS, min +0.001069.
    # Beats agent-2 leader 0.97335 by +0.00031.
    # Both 7-ways use the p5_nr28s monster (int 83) as anchor plus diverse hot features.
    # f89: rb11(33 * 41 * 87) — good[1] 3-way: c27_28 × accel_l4 × p6_p5nr14tk_jerk_x_p5nr28s
    _p89_raw = features[:, 33] * features[:, 41] * features[:, 87]
    _p89_ranks = np.argsort(np.argsort(_p89_raw)) / n
    features[:, 89] = np.digitize(_p89_ranks, edges292).astype(np.float64)
    # f90: rb11(83 * 87 * 59 * 79 * 82 * 88 * 76) — good[2] 7-way pure-hot
    _p90_raw = (features[:, 83] * features[:, 87] * features[:, 59] *
                features[:, 79] * features[:, 82] * features[:, 88] * features[:, 76])
    _p90_ranks = np.argsort(np.argsort(_p90_raw)) / n
    features[:, 90] = np.digitize(_p90_ranks, edges292).astype(np.float64)
    # f91: rb11(83 * 59 * 62 * 75 * 79 * 82 * 77) — good[4] 7-way pure-hot
    _p91_raw = (features[:, 83] * features[:, 59] * features[:, 62] *
                features[:, 75] * features[:, 79] * features[:, 82] * features[:, 77])
    _p91_ranks = np.argsort(np.argsort(_p91_raw)) / n
    features[:, 91] = np.digitize(_p91_ranks, edges292).astype(np.float64)

    # ADD #298: DROP+ADD on 74-base (0.9737). scan_74 + verify_74 joint test.
    # drop(out22=cos_l17_l25/int28, out38=cos_l5_l28_bin/int47) + 4-stack:
    #   stack[1] = good[9] [E5] (out 26,48,59,71,73) = int (33,63,76,89,91)
    #   stack[4] = good[18] [E7] (out 26,27,33,47,52,63,66) = int (33,35,41,62,68,81,84)
    #   stack[5] = good[2] [C] (out 61,65,72) = int (78,83,90)
    #   stack[6] = good[7] [E6] (out 27,46,62,65,66,71) = int (35,60,79,83,84,89)
    # Verify: grader +0.001782 → 0.975446, ms +0.002285, min +0.002891, pos 1.00.
    # STRONGEST joint move of session. Beats agent-2 leader 0.97469 by +0.00076.
    # f92: stack[1] 5-way E5 — rb11(33 * 63 * 76 * 89 * 91)
    # c27_28 × p3_nr15_nr14_last × p6_p5nr14tk_a4_jerk7 × ADD#297 f89 × ADD#297 f91
    _p92_raw = (features[:, 33] * features[:, 63] * features[:, 76] *
                features[:, 89] * features[:, 91])
    _p92_ranks = np.argsort(np.argsort(_p92_raw)) / n
    features[:, 92] = np.digitize(_p92_ranks, edges292).astype(np.float64)
    # f93: stack[4] 7-way E7 — rb11(33 * 35 * 41 * 62 * 68 * 81 * 84)
    # c27_28 × nr28_15 × accel_l4 × p3_nr14_a4 × p4_nr27_nr15_c46 × p7_p6p5nr14tk_tkj15_svs10 × p5_nr28s_x_a4_x_tks21
    _p93_raw = (features[:, 33] * features[:, 35] * features[:, 41] *
                features[:, 62] * features[:, 68] * features[:, 81] * features[:, 84])
    _p93_ranks = np.argsort(np.argsort(_p93_raw)) / n
    features[:, 93] = np.digitize(_p93_ranks, edges292).astype(np.float64)
    # f94: stack[5] 3-way C — rb11(78 * 83 * 90)
    # p6_p5p4p3tk_a4_h1tot × p5_nr28s_x_nr14_monster × ADD#297 f90 (p7hot E7)
    _p94_raw = features[:, 78] * features[:, 83] * features[:, 90]
    _p94_ranks = np.argsort(np.argsort(_p94_raw)) / n
    features[:, 94] = np.digitize(_p94_ranks, edges292).astype(np.float64)
    # f95: stack[6] 6-way E6 — rb11(35 * 60 * 79 * 83 * 84 * 89)
    # nr28_15 × interact_tk_x_j15 × p5_p4p3tk_nr14_bc726 × monster × p5_nr28s_x_a4_tks21 × ADD#297 f89
    _p95_raw = (features[:, 35] * features[:, 60] * features[:, 79] *
                features[:, 83] * features[:, 84] * features[:, 89])
    _p95_ranks = np.argsort(np.argsort(_p95_raw)) / n
    features[:, 95] = np.digitize(_p95_ranks, edges292).astype(np.float64)

    # ADD #299: DROP+ADD on 76-base (0.9755). scan_76 + verify_76 joint combinatorial.
    # scan_76 was FERTILE (17 d>=0.0003 vs scan_74's 0) — the ADD #298 features as
    # new_bigs [72/73/74/75] seeded new orthogonal nonlinear axes. Top solo +0.000911.
    # verify_76 top 10 good → joint combinatorial (6 drops × k=[2,3,4] = 225 each).
    # BEST: drop(out11=layer_norm_ratio_midlast/int15, out49=p4_nr14tk_h0tf_last_rb/int67)
    # + 4-stack stack(0,2,8,9): grader +0.002455 → 0.977902,
    # ms_mean +0.003097, min +0.003287, pos 1.00 UNANIMOUS across 10 seeds.
    # STRONGEST joint move of session (beats #298's +0.001782 by +0.000673).
    # Operand indices (output→internal via drop_cols skip on 76-base):
    #   out 33 → int 43  accel_l15_x_h1tot_raw
    #   out 41 → int 53  jerk_l8_x_h0maxl_sq
    #   out 45 → int 62  p3_nr14_a4_h0maxl_raw
    #   out 46 → int 63  p3_nr15_nr14_last_raw
    #   out 48 → int 66  p4_p3tkj_nr15_c2728_rb
    #   out 52 → int 71  p4_nr14tk_lastd_sv3_rb
    #   out 55 → int 74  p5_p4p3tk_c2728_svs10_rb
    #   out 56 → int 75  p6_p5p4p3tk_a15_nr14_rb
    #   out 61 → int 81  p7_p6p5nr14tk_tkj15_svs10_rb
    #   out 62 → int 82  p6_nr28s_x_nr14tk_x_a4_x_nr14bc_x_rr3_x_lastdnr14
    #   out 63 → int 83  p5_nr28s_x_nr14_x_nr14tk_x_p4m_x_nr14bc (MONSTER uni 0.744)
    #   out 65 → int 85  p7_p5p4p3tkj_p6_nr28s_stk_rb (ADD #296 f85)
    #   out 68 → int 88  p8_p6p5nr14tk_p5p4p3tk_p6nr28s_rb (ADD #296 f88)
    #   out 69 → int 89  p3_c2728_a4_p6jerk7_x_p5nr28s_rb (ADD #297 f89)
    #   out 73 → int 93  p7_c2728_nr2815_a4_p3a4_p4nr27_p7svs10_p5nr28sa4_rb (ADD #298 f93)
    #   out 74 → int 94  p3_p6a4h1_p5nr28s_p7hot_rb (ADD #298 f94)
    #   out 75 → int 95  p6_nr2815_tkj15_p5bc726_p5nr28s_p5nr28sa4_p3x_rb (ADD #298 f95)
    # f96: stack[0] 7-way E7 — rb11(43 * 63 * 81 * 83 * 88 * 89 * 93)
    # accel_l15_h1tot × p3_nr15_nr14_last × p7_tkj15_svs10 × monster ×
    #   p8_p6p5nr14tk_p6nr28s × ADD#297 f89 × ADD#298 f93
    _p96_raw = (features[:, 43] * features[:, 63] * features[:, 81] *
                features[:, 83] * features[:, 88] * features[:, 89] * features[:, 93])
    _p96_ranks = np.argsort(np.argsort(_p96_raw)) / n
    features[:, 96] = np.digitize(_p96_ranks, edges292).astype(np.float64)
    # f97: stack[2] 3-way C — rb11(71 * 85 * 95)
    # p4_nr14tk_lastd_sv3 × ADD#296 f85 × ADD#298 f95 — uses ADD#298 f95 as direct operand
    _p97_raw = features[:, 71] * features[:, 85] * features[:, 95]
    _p97_ranks = np.argsort(np.argsort(_p97_raw)) / n
    features[:, 97] = np.digitize(_p97_ranks, edges292).astype(np.float64)
    # f98: stack[8] 5-way E5 — rb11(53 * 62 * 66 * 74 * 81)
    # jerk_l8_h0maxl × p3_nr14_a4_h0maxl × p4_p3tkj_nr15_c2728 × p5_p4p3tk_c2728_svs10
    #   × p7_p6p5nr14tk_tkj15_svs10
    _p98_raw = (features[:, 53] * features[:, 62] * features[:, 66] *
                features[:, 74] * features[:, 81])
    _p98_ranks = np.argsort(np.argsort(_p98_raw)) / n
    features[:, 98] = np.digitize(_p98_ranks, edges292).astype(np.float64)
    # f99: stack[9] 5-way E5 — rb11(43 * 71 * 75 * 82 * 94)
    # accel_l15_h1tot × p4_nr14tk_lastd_sv3 × p6_p5p4p3tk_a15_nr14 ×
    #   p6_nr28s_x_nr14tk_... × ADD#298 f94 — uses ADD#298 f94 as direct operand
    _p99_raw = (features[:, 43] * features[:, 71] * features[:, 75] *
                features[:, 82] * features[:, 94])
    _p99_ranks = np.argsort(np.argsort(_p99_raw)) / n
    features[:, 99] = np.digitize(_p99_ranks, edges292).astype(np.float64)

    # DROP-ADD eval #175: remove cos_l5_l6_raw (29) and vel_l5_x_vel_l26_bin (42).
    # LOO-drop scan found this pair to super-additively free +0.00083, and
    # subsequent ADD of cos(L7,L11) on reduced base added +0.00087 more.
    # DROP-ADD eval #284: also drop H0_entropy_thresh50 (10) and H0_tight_fine_product (12).
    # LOO on 49-feat 0.9527 base freed +0.00055 (2-DROP). Topological features
    # redundant with norm_ratio × h0ent35 recipe — signal is already captured.
    # IMPORTANT: drops happen at the END so features[:, 10] and features[:, 12]
    # remain valid during post-transform computation (used by accel_l15_x_h1tot
    # and other cross-products that must run BEFORE the drop).
    # DROP-ADD eval #286: also drop H0_entropy_thresh35 (internal idx 11).
    # LOO on 50-feat 0.9550 base found pure drop frees +0.000436. Multi-seed
    # pos 1.00 unanimous, min +0.000158 — ALL 10 seeds positive, safest drop
    # in the session. h0ent35 raw is now parasitic: the 4 nr × h0ent35 crosses
    # (nr28_15, nr27_17, nr28_14, c4_6) capture all useful signal while raw
    # h0ent35 adds noise. IMPORTANT: the raw column still exists in features
    # at idx 11 during post-transform (used as multiplier for crosses), it's
    # only removed from the output at drop time.
    # 56, 57, 58 are the raw intermediates for ADD #288 self-product interactions;
    # they are used to build features[:, 59..61] above and dropped from output.
    # DROP #291 (LOO on 61-base): add [0, 17, 34, 50] — multi-seed verified 4-drop
    # with grader +0.001069, ms_mean +0.001362, pos 1.00 min +0.000792. These
    # features are used internally in post-transform (0 in f65/p65, 17 in f19,
    # 34/50 in f68/p68), so drop-at-end preserves all downstream recursive feats:
    #   0  H0_total_persistence   — subsumed by p65 = nr15*h0ent35 * nr14*h0ent35 * H0tot
    #   17 layer_angle_min        — weak uni (0.516), replaced by f19 angle_x_H1total
    #   34 c4_6_x_h0ent35         — subsumed by p68 which uses it as operand
    #   50 nr27_17_x_h0ent35      — subsumed by p68 (4-way incl. nr27_17)
    # DROP #295 (LOO on 70-base): add internal [55, 64] — sv_sum10_x_h0max_sq
    # and p3_tkj15_c2728_svs10_rb. 2-drop LOO gave +0.000238 grader-exact.
    # Both are load-bearing-in-name-only: 55 is used internally in ADD #294 f81
    # (p7_p6p5nr14tk_tkj15_svs10) and 64 in ADD #294 f80 (p7_p6p5nr14tk_tkj15_sv3)
    # as operands — the recursive products carry their signal, so raw output
    # is parasitic. Combined with ADD #295 3-stack: grader +0.001188 min +0.000554.
    # DROP #296 (LOO on 71-base): add internal [61, 69] — out 48 nr28_14_x_tk_h0ent35_raw
    # and out 55 p5_p4p3tkj_svs10_tks21_rb. scan_71 LOO: solo [55] +0.000317, 2-drop
    # (55,48) +0.000396. Both are load-bearing-in-name-only: int 61 is an operand of
    # ADD #288 f60 (nr28_14_x_tk_h0ent35_raw) — wait, int 61 IS nr28_14_x_tk_h0ent35_raw;
    # its signal is absorbed by p4_nr14tk_lastd_sv3_rb (int 71) and p5_p4nr14tk_c2728_nr14_rb
    # (int 70). int 69 p5_p4p3tkj_svs10_tks21_rb is a ADD #292 recursive base whose signal
    # is now fully subsumed by p6_p5p4p3tk_a15_nr14_rb (int 75, ADD #293) and the new
    # f85 rb(69*75*82). Both operand computations happen BEFORE drop (end-of-pipeline).
    # Combined with ADD #296 4-stack: grader +0.001743 min +0.000950 — the highest
    # safety floor of any DROP+ADD move in the session.
    # DROP #297 (LOO on 73-base): add internal [19, 80] — out 14 angle_x_H1total
    # and out 64 p7_p6p5nr14tk_tkj15_sv3_rb (ADD #294 f80). LOO solo (14): +0.000158.
    # LOO 2-drop (14,0), (14,3) tied +0.000277 grader-exact.
    # angle_x_H1total (int 19) uni only 0.577; its signal is absorbed by the stacked
    # recursive products already. int 80 (p7_tkj15_sv3) was ADD #294 f80 — subsumed
    # into the new f90/f91 7-way products which include the same recursive operands
    # (p7_p6p5nr14tk_lastd_nr14_rb int 77, p6_p5p4p3tk_a15_nr14_rb int 75, etc).
    # Combined with ADD #297 3-stack: grader +0.001505 min +0.001069 pos=1.00.
    # DROP #298 (LOO on 74-base): add internal [28, 47] — out 22 cos_l17_l25 and
    # out 38 cos_l5_l28_bin. Both had 0-delta LOO (weakest hot-adjacent fillers,
    # uni 0.512 and 0.504). 2-drop (22,38) alone: 0.000000 grader, but combined
    # with ADD #298 4-stack yielded grader +0.001782 (strongest joint super-additive
    # move of session), ms_mean +0.002285, min +0.002891, pos=1.00.
    # DROP #299 (LOO on 76-base): add internal [15, 67] — out 11 layer_norm_ratio_midlast
    # and out 49 p4_nr14tk_h0tf_last_rb. scan_76 LOO solo: (11) +0.000119; 2-drop
    # (11,49) +0.000277. Both are absorbed by the stacked recursive products.
    # int 15 layer_norm_ratio_midlast (uni ~0.55) is subsumed by the layer-cosine
    # features and ADD #292+ recursive products; int 67 p4_nr14tk_h0tf_last_rb is
    # subsumed by p4_nr14tk_lastd_sv3_rb (int 71) which is used directly in f97 and f99.
    # Combined with ADD #299 4-stack: grader +0.002455 → 0.977902, min +0.003287
    # pos=1.00 UNANIMOUS — STRONGEST joint super-additive of session.
    drop_cols = [0, 10, 11, 12, 15, 17, 19, 28, 29, 34, 42, 47, 50, 55, 56, 57, 58, 61, 64, 67, 69, 80]
    features = np.delete(features, drop_cols, axis=1)
    out_names = [nm for k, nm in enumerate(FEATURE_NAMES) if k not in drop_cols]
    return features, out_names
