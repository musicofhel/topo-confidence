"""Phase 4: Routing Prototype.

Build a minimal inference-time router using topo-confidence scores.
Compare against random routing and output-entropy routing.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

PHASE4_DIR = Path(__file__).parent
PHASE1_DIR = PHASE4_DIR.parent / "phase1"
PHASE2_DIR = PHASE4_DIR.parent / "phase2"
PHASE0_DIR = PHASE4_DIR.parent / "phase0"
TOPO_DIR = PHASE4_DIR.parent.parent
TRAJ_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectories.npz"
META_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectory_meta.json"
LAYER_PATH = Path.home() / "att-docs" / "data" / "transformer" / "math500_hidden_states_aligned.npz"

sys.path.insert(0, str(PHASE0_DIR))
from winning_features import extract_features, PCA as OrigPCA

LR_PARAMS = dict(max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs")


def load_data():
    t = np.load(TRAJ_PATH)
    trajectories = [t[f"traj_{i}"] for i in range(500)]
    with open(META_PATH) as f:
        meta = json.load(f)
    y = np.array(meta["correct"], dtype=int)
    ls = np.load(LAYER_PATH)["layer_hidden_states"]
    return trajectories, y, ls


def make_prefitted_pca(traj_list):
    from sklearn.decomposition import PCA as SkPCA
    concat = np.concatenate(traj_list, axis=0)
    n_comp = min(45, concat.shape[1], concat.shape[0])
    pca = SkPCA(n_components=n_comp, svd_solver="full")
    pca.fit(concat)
    return pca


def extract_with_train_pca(trajectories, layer_states, train_pca):
    from sklearn.decomposition import PCA as SkPCA

    class PreFittedPCA(SkPCA):
        _prefitted = None
        def fit(self, X, y=None):
            for attr in ['components_', 'mean_', 'explained_variance_',
                         'explained_variance_ratio_', 'singular_values_',
                         'n_components_', 'n_samples_', 'n_features_in_',
                         'noise_variance_']:
                setattr(self, attr, getattr(PreFittedPCA._prefitted, attr))
            return self

    PreFittedPCA._prefitted = train_pca
    import winning_features as wf_module
    original = wf_module.PCA
    wf_module.PCA = PreFittedPCA
    try:
        features, names = extract_features(trajectories, layer_states)
    finally:
        wf_module.PCA = original
    return features, names


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() * abs(avg_conf - avg_acc)
    return ece / len(y_true)


def routing_metrics(y_true, confidence, threshold):
    """Compute routing metrics at a given confidence threshold."""
    trusted = confidence >= threshold
    n_trusted = trusted.sum()
    n_total = len(y_true)

    coverage = n_trusted / n_total

    if n_trusted == 0:
        return {"coverage": 0.0, "trusted_acc": None, "escalation_ppv": None, "oracle_gain": None}

    trusted_acc = y_true[trusted].mean() if n_trusted > 0 else None

    escalated = ~trusted
    n_escalated = escalated.sum()
    if n_escalated > 0:
        # Escalation PPV: fraction of escalated that model actually got wrong
        escalation_ppv = (1 - y_true[escalated]).mean()
    else:
        escalation_ppv = None

    # Oracle gain: assume big model gets 100% of escalated correct
    base_acc = y_true.mean()
    oracle_acc = (y_true[trusted].sum() + n_escalated) / n_total if n_total > 0 else 0
    oracle_gain = oracle_acc - base_acc

    return {
        "coverage": float(coverage),
        "trusted_acc": float(trusted_acc) if trusted_acc is not None else None,
        "escalation_ppv": float(escalation_ppv) if escalation_ppv is not None else None,
        "oracle_gain": float(oracle_gain),
    }


def bootstrap_routing(y_true, confidence, threshold, n_boot=1000, seed=42):
    """Bootstrap CI for coverage and trusted accuracy."""
    rng = np.random.default_rng(seed)
    coverages = []
    trusted_accs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        m = routing_metrics(y_true[idx], confidence[idx], threshold)
        coverages.append(m["coverage"])
        if m["trusted_acc"] is not None:
            trusted_accs.append(m["trusted_acc"])

    cov_ci = (np.percentile(coverages, 2.5), np.percentile(coverages, 97.5)) if coverages else (None, None)
    acc_ci = (np.percentile(trusted_accs, 2.5), np.percentile(trusted_accs, 97.5)) if trusted_accs else (None, None)
    return cov_ci, acc_ci


def main():
    print("=" * 60)
    print("Phase 4: Routing Prototype")
    print("=" * 60)

    # Check prerequisite
    holdout_metrics = json.load(open(PHASE1_DIR / "holdout_metrics.json"))
    ci_lower = holdout_metrics["ci_lower_bound"]
    print(f"\n  Prerequisite check: holdout CI lower bound = {ci_lower:.4f}")
    if ci_lower >= 0.78:
        print("  PASSED (>= 0.78)")
    elif ci_lower >= 0.75:
        print("  NARROW GO ([0.75, 0.78)) - proceed with caution")
    else:
        print("  FAILED (< 0.75) - should not proceed")
        return

    # Load best surviving tier
    best_tier_line = open(PHASE2_DIR / "best_surviving_tier.txt").readline().strip()
    print(f"  {best_tier_line}")

    # Load data
    print("\n[1] Loading data...")
    trajectories, y, layer_states = load_data()

    mask = np.load(PHASE1_DIR / "holdout_mask_seed9999.npy")
    train_idx = np.where(~mask)[0]
    hold_idx = np.where(mask)[0]
    y_train = y[train_idx]
    y_holdout = y[hold_idx]

    # Extract features
    print("[2] Extracting features...")
    traj_train = [trajectories[i] for i in train_idx]
    ls_train = layer_states[train_idx]

    t0 = time.time()
    features_train, _ = extract_features(traj_train, ls_train)
    print(f"  Train extraction: {time.time() - t0:.1f}s")

    train_pca = make_prefitted_pca(traj_train)
    t0 = time.time()
    features_all_trainpca, _ = extract_with_train_pca(trajectories, layer_states, train_pca)
    features_holdout = features_all_trainpca[hold_idx]
    print(f"  Holdout extraction: {time.time() - t0:.1f}s")

    # Use best surviving tier (A+B+C from Phase 2)
    # Load tier assignments to get indices
    tier_assignments = json.load(open(PHASE2_DIR / "tier_assignments.json"))
    all_names = list(tier_assignments.keys())

    # Best tier is A+B+C - use all features NOT in tier D
    best_cols = [i for i, nm in enumerate(all_names) if tier_assignments[nm] in ("A", "B", "C")]
    print(f"  Using best surviving tier (A+B+C): {len(best_cols)} features")

    X_tr = features_train[:, best_cols]
    X_ho = features_holdout[:, best_cols]

    # Filter degenerate
    good = np.array([X_tr[:, j].std() > 1e-10 for j in range(X_tr.shape[1])])
    X_tr = X_tr[:, good]
    X_ho = X_ho[:, good]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_ho_scaled = scaler.transform(X_ho)

    # ==============================
    # Step 1: Calibration
    # ==============================
    print("\n[3] Calibration check...")
    clf = LogisticRegression(**LR_PARAMS)
    clf.fit(X_tr_scaled, y_train)
    proba_holdout = clf.predict_proba(X_ho_scaled)[:, 1]

    # ECE and Brier score
    ece = compute_ece(y_holdout, proba_holdout)
    brier = brier_score_loss(y_holdout, proba_holdout)
    print(f"  ECE = {ece:.4f}")
    print(f"  Brier score = {brier:.4f}")

    calibrated = False
    if ece > 0.05:
        print(f"  ECE > 0.05 — applying Platt scaling via nested CV on train...")
        cal_clf = CalibratedClassifierCV(
            LogisticRegression(**LR_PARAMS),
            method="sigmoid",
            cv=5,
        )
        cal_clf.fit(X_tr_scaled, y_train)
        proba_holdout_cal = cal_clf.predict_proba(X_ho_scaled)[:, 1]
        ece_cal = compute_ece(y_holdout, proba_holdout_cal)
        brier_cal = brier_score_loss(y_holdout, proba_holdout_cal)
        print(f"  Calibrated ECE = {ece_cal:.4f}")
        print(f"  Calibrated Brier = {brier_cal:.4f}")
        if ece_cal < ece:
            print("  Using calibrated probabilities for routing.")
            proba_holdout = proba_holdout_cal
            calibrated = True
        else:
            print("  Calibration did not improve — using original probabilities.")

    cal_metrics = {
        "ece_uncalibrated": round(float(ece), 4),
        "brier_uncalibrated": round(float(brier), 4),
        "calibrated": calibrated,
    }
    if calibrated:
        cal_metrics["ece_calibrated"] = round(float(ece_cal), 4)
        cal_metrics["brier_calibrated"] = round(float(brier_cal), 4)

    with open(PHASE4_DIR / "calibration_metrics.json", "w") as f:
        json.dump(cal_metrics, f, indent=2)

    # ==============================
    # Step 2: Routing sweep
    # ==============================
    print("\n[4] Routing threshold sweep...")
    thresholds = np.arange(0.1, 0.91, 0.05)
    topo_routing = []

    for tau in thresholds:
        m = routing_metrics(y_holdout, proba_holdout, tau)
        cov_ci, acc_ci = bootstrap_routing(y_holdout, proba_holdout, tau)
        topo_routing.append({
            "tau": round(float(tau), 2),
            **{k: round(v, 4) if v is not None else None for k, v in m.items()},
            "coverage_ci": [round(float(x), 4) if x is not None else None for x in cov_ci],
            "trusted_acc_ci": [round(float(x), 4) if x is not None else None for x in acc_ci],
        })

    print(f"\n  {'tau':>5} {'coverage':>9} {'trusted_acc':>12} {'esc_ppv':>8} {'oracle_gain':>12}")
    for r in topo_routing:
        ta = f"{r['trusted_acc']:.4f}" if r['trusted_acc'] is not None else "N/A"
        ep = f"{r['escalation_ppv']:.4f}" if r['escalation_ppv'] is not None else "N/A"
        og = f"{r['oracle_gain']:.4f}" if r['oracle_gain'] is not None else "N/A"
        print(f"  {r['tau']:>5.2f} {r['coverage']:>9.4f} {ta:>12} {ep:>8} {og:>12}")

    # ==============================
    # Step 3: Random baseline
    # ==============================
    print("\n[5] Random routing baseline...")
    rng = np.random.default_rng(42)
    random_routing = []
    for r in topo_routing:
        target_escalation_frac = 1 - r["coverage"]
        accs = []
        ppvs = []
        for _ in range(100):
            rand_conf = rng.random(len(y_holdout))
            rand_threshold = np.quantile(rand_conf, target_escalation_frac) if target_escalation_frac > 0 else 0
            m = routing_metrics(y_holdout, rand_conf, rand_threshold)
            if m["trusted_acc"] is not None:
                accs.append(m["trusted_acc"])
            if m["escalation_ppv"] is not None:
                ppvs.append(m["escalation_ppv"])
        random_routing.append({
            "tau": r["tau"],
            "coverage": r["coverage"],  # matched
            "trusted_acc": float(np.mean(accs)) if accs else None,
            "escalation_ppv": float(np.mean(ppvs)) if ppvs else None,
        })

    # ==============================
    # Step 4: Output-entropy baseline
    # ==============================
    print("[6] Output-entropy baseline...")
    # Check if logits exist
    logits_path = TOPO_DIR / "data" / "experiment1_v2"
    logits_files = list(logits_path.glob("*logit*"))
    output_logits = None

    if not logits_files:
        print("  No cached logits found. Checking for alternative entropy source...")
        # Try to compute from trajectory data as proxy
        # Actually, we need the model's output logits. Check if extractor can help.
        try:
            from topo_confidence.baselines import output_entropy
            print("  baselines.output_entropy available but needs logits.")
        except ImportError:
            print("  baselines module not available.")

        # Check if there's a logits file anywhere in the repo
        logits_any = list(TOPO_DIR.rglob("*logit*"))
        if logits_any:
            print(f"  Found logits files: {logits_any}")
        else:
            print("  No logits available. Skipping output-entropy baseline.")
            print("  (Would need to re-extract with collect_logits=True, which requires GPU)")

    entropy_routing = None
    if output_logits is not None:
        pass  # Would compute entropy routing here

    # ==============================
    # Step 5: Prompt-length baseline (trivial)
    # ==============================
    print("\n[7] Prompt-length baseline (trivial)...")
    # Use trajectory length (number of tokens) as proxy for problem difficulty
    traj_data = np.load(TRAJ_PATH)
    prompt_lengths = np.array([traj_data[f"traj_{i}"].shape[0] for i in hold_idx])
    # Shorter = easier = trust. Use negative length as confidence.
    length_conf = -prompt_lengths.astype(float)
    # Normalize to [0, 1]
    length_conf = (length_conf - length_conf.min()) / (length_conf.max() - length_conf.min() + 1e-12)

    length_routing = []
    for r in topo_routing:
        target_coverage = r["coverage"]
        length_threshold = np.quantile(length_conf, 1 - target_coverage) if target_coverage < 1 else 0
        m = routing_metrics(y_holdout, length_conf, length_threshold)
        length_routing.append({
            "coverage": round(float(m["coverage"]), 4),
            "trusted_acc": round(float(m["trusted_acc"]), 4) if m["trusted_acc"] is not None else None,
        })

    # ==============================
    # Step 6: Find operating point
    # ==============================
    print("\n[8] Finding operating point...")
    operating_point = None
    for r in topo_routing:
        if r["coverage"] is not None and r["trusted_acc"] is not None:
            if r["coverage"] >= 0.90 and r["trusted_acc"] >= 0.95:
                operating_point = r
                break

    if operating_point:
        print(f"  Found: tau={operating_point['tau']}, coverage={operating_point['coverage']:.4f}, "
              f"trusted_acc={operating_point['trusted_acc']:.4f}")
    else:
        # Find coverage at which trusted accuracy first exceeds 95%
        for r in sorted(topo_routing, key=lambda x: -(x["coverage"] or 0)):
            if r["trusted_acc"] is not None and r["trusted_acc"] >= 0.95:
                print(f"  No point with coverage>=0.90 AND trusted_acc>=0.95.")
                print(f"  Best: tau={r['tau']}, coverage={r['coverage']:.4f}, "
                      f"trusted_acc={r['trusted_acc']:.4f}")
                operating_point = r
                break

    if operating_point is None:
        print("  No threshold achieves trusted_acc >= 0.95")

    with open(PHASE4_DIR / "operating_point.json", "w") as f:
        json.dump(operating_point, f, indent=2)

    # ==============================
    # Step 7: Comparison table
    # ==============================
    print("\n" + "=" * 60)
    print("Baseline Comparison")
    print("=" * 60)

    print(f"\n  {'Coverage':>9} {'Topo_acc':>9} {'Random_acc':>11} {'Length_acc':>11}")
    for i, r in enumerate(topo_routing):
        ta = f"{r['trusted_acc']:.4f}" if r['trusted_acc'] is not None else "N/A"
        ra = f"{random_routing[i]['trusted_acc']:.4f}" if random_routing[i]['trusted_acc'] is not None else "N/A"
        la = f"{length_routing[i]['trusted_acc']:.4f}" if length_routing[i]['trusted_acc'] is not None else "N/A"
        print(f"  {r['coverage']:>9.4f} {ta:>9} {ra:>11} {la:>11}")

    # Count operating points where topo beats baselines
    topo_wins_random = 0
    topo_wins_length = 0
    for i, r in enumerate(topo_routing):
        if r["trusted_acc"] is not None and random_routing[i]["trusted_acc"] is not None:
            if r["trusted_acc"] > random_routing[i]["trusted_acc"]:
                topo_wins_random += 1
        if r["trusted_acc"] is not None and length_routing[i]["trusted_acc"] is not None:
            if r["trusted_acc"] > length_routing[i]["trusted_acc"]:
                topo_wins_length += 1

    n_points = len(topo_routing)
    print(f"\n  Topo beats random: {topo_wins_random}/{n_points} operating points")
    print(f"  Topo beats length: {topo_wins_length}/{n_points} operating points")

    # Interpretation
    print("\n  Interpretation:")
    if not entropy_routing:
        print("  Output-entropy baseline not available (no cached logits, no GPU).")
        print("  Topo vs random and topo vs prompt-length comparisons available.")
    print(f"  Topo-confidence router beats random routing at {topo_wins_random}/{n_points} points.")
    if topo_wins_random >= n_points * 0.8:
        print("  Topo-confidence carries substantial routing value beyond random.")
    elif topo_wins_random >= n_points * 0.5:
        print("  Topo-confidence has moderate routing value.")
    else:
        print("  Topo-confidence shows limited routing value.")

    # Save all routing metrics
    all_routing = {
        "topo_confidence": topo_routing,
        "random": random_routing,
        "prompt_length": length_routing,
        "output_entropy": entropy_routing,
    }
    with open(PHASE4_DIR / "routing_metrics.json", "w") as f:
        json.dump(all_routing, f, indent=2)

    # Save comparison text
    comp_lines = ["Baseline Comparison (Phase 4)", "=" * 50, ""]
    comp_lines.append(f"Topo beats random: {topo_wins_random}/{n_points} operating points")
    comp_lines.append(f"Topo beats length: {topo_wins_length}/{n_points} operating points")
    comp_lines.append("")
    if entropy_routing:
        comp_lines.append("Output-entropy comparison: available")
    else:
        comp_lines.append("Output-entropy comparison: NOT available (no logits cached)")
        comp_lines.append("Re-extract with collect_logits=True on GPU to enable this comparison.")
    with open(PHASE4_DIR / "baseline_comparison.txt", "w") as f:
        f.write("\n".join(comp_lines) + "\n")

    # ==============================
    # Plot
    # ==============================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Calibration plot
        ax = axes[0]
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        observed = []
        predicted = []
        for i in range(n_bins):
            m = (proba_holdout >= bin_edges[i]) & (proba_holdout < bin_edges[i + 1])
            if m.sum() > 0:
                observed.append(y_holdout[m].mean())
                predicted.append(proba_holdout[m].mean())
            else:
                observed.append(np.nan)
                predicted.append(bin_centers[i])

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        ax.plot(predicted, observed, "bo-", label="Model")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed fraction correct")
        ax.set_title(f"Calibration (ECE={ece:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Coverage vs accuracy
        ax = axes[1]
        coverages = [r["coverage"] for r in topo_routing if r["trusted_acc"] is not None]
        topo_accs = [r["trusted_acc"] for r in topo_routing if r["trusted_acc"] is not None]
        random_accs_plot = [random_routing[i]["trusted_acc"] for i in range(len(topo_routing))
                          if topo_routing[i]["trusted_acc"] is not None and random_routing[i]["trusted_acc"] is not None]
        length_accs_plot = [length_routing[i]["trusted_acc"] for i in range(len(topo_routing))
                           if topo_routing[i]["trusted_acc"] is not None and length_routing[i]["trusted_acc"] is not None]
        coverages_matched = [r["coverage"] for i, r in enumerate(topo_routing)
                            if r["trusted_acc"] is not None and random_routing[i]["trusted_acc"] is not None]

        ax.plot(coverages, topo_accs, "bo-", label="Topo-confidence", markersize=6)
        if len(random_accs_plot) == len(coverages_matched):
            ax.plot(coverages_matched, random_accs_plot, "g^--", label="Random", markersize=5, alpha=0.7)
        if len(length_accs_plot) == len(coverages_matched):
            ax.plot(coverages_matched, length_accs_plot, "rv--", label="Prompt length", markersize=5, alpha=0.7)
        ax.axhline(0.95, color="orange", linestyle=":", alpha=0.5, label="95% accuracy")
        ax.set_xlabel("Coverage (fraction trusted)")
        ax.set_ylabel("Trusted accuracy")
        ax.set_title("Routing: Coverage vs Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0.5, 1.05)

        plt.tight_layout()
        plt.savefig(PHASE4_DIR / "routing_tradeoff_curve.png", dpi=150)
        plt.savefig(PHASE4_DIR / "calibration_plot.png", dpi=150)
        print(f"\n  Plots saved")
    except ImportError:
        print("\n  matplotlib not available, skipping plots")

    # ==============================
    # Go/No-Go Decision
    # ==============================
    print("\n" + "=" * 60)
    print("GO/NO-GO Decision")
    print("=" * 60)

    holdout_auroc = holdout_metrics["auroc"]
    print(f"\n  Holdout AUROC: {holdout_auroc:.4f}")
    print(f"  Bootstrap 95% CI: [{holdout_metrics['bootstrap_ci_95'][0]:.4f}, "
          f"{holdout_metrics['bootstrap_ci_95'][1]:.4f}]")
    print(f"  CI lower bound: {ci_lower:.4f}")

    if ci_lower >= 0.78 and topo_wins_random >= n_points * 0.5:
        if entropy_routing and topo_wins_random < n_points * 0.5:
            decision = "CONDITIONAL GO"
            reason = "CI >= 0.78 but topo ≈ entropy. Proceed but Pathway 4 may not add value."
        else:
            decision = "GO"
            reason = "CI lower bound >= 0.78 AND topo beats random at most operating points."
    elif ci_lower >= 0.75:
        decision = "NARROW GO"
        reason = "CI lower bound in [0.75, 0.78). Use Tier A+B features only."
    else:
        decision = "NO-GO"
        reason = "CI lower bound < 0.75."

    print(f"\n  Decision: {decision}")
    print(f"  Reason: {reason}")

    # Note about entropy baseline
    if not entropy_routing:
        print("\n  NOTE: Output-entropy baseline not computed (no logits available).")
        print("  The full GO vs CONDITIONAL GO distinction requires entropy comparison.")
        print(f"  Based on available evidence (CI lower={ci_lower:.4f}, topo beats random "
              f"{topo_wins_random}/{n_points}): {decision}")

    print(f"\n  All Phase 4 artifacts saved to {PHASE4_DIR}/")


if __name__ == "__main__":
    main()
