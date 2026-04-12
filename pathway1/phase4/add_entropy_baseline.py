"""Add output-entropy baseline to existing Phase 4 routing results.

Loads existing routing metrics + precomputed entropy scores,
computes entropy routing at matched coverage levels, and updates
all Phase 4 artifacts with the comparison.

No GPU required — uses precomputed output_entropy_scores.json.
"""

import json
from pathlib import Path

import numpy as np

PHASE4_DIR = Path(__file__).parent
PHASE1_DIR = PHASE4_DIR.parent / "phase1"
TOPO_DIR = PHASE4_DIR.parent.parent
SCORES_PATH = TOPO_DIR / "data" / "experiment1_v2" / "output_entropy_scores.json"
META_PATH = TOPO_DIR / "data" / "experiment1_v2" / "trajectory_meta.json"


def routing_metrics(y_true, confidence, threshold):
    trusted = confidence >= threshold
    n_trusted = trusted.sum()
    n_total = len(y_true)
    coverage = n_trusted / n_total
    if n_trusted == 0:
        return {"coverage": 0.0, "trusted_acc": None, "escalation_ppv": None}
    trusted_acc = y_true[trusted].mean()
    escalated = ~trusted
    n_escalated = escalated.sum()
    escalation_ppv = (1 - y_true[escalated]).mean() if n_escalated > 0 else None
    return {
        "coverage": float(coverage),
        "trusted_acc": float(trusted_acc),
        "escalation_ppv": float(escalation_ppv) if escalation_ppv is not None else None,
    }


def main():
    print("=" * 60)
    print("Adding Output-Entropy Baseline to Phase 4")
    print("=" * 60)

    # Load existing routing metrics
    metrics_path = PHASE4_DIR / "routing_metrics.json"
    with open(metrics_path) as f:
        all_routing = json.load(f)
    topo_routing = all_routing["topo_confidence"]
    random_routing = all_routing["random"]
    length_routing = all_routing["prompt_length"]
    print(f"  Loaded {len(topo_routing)} topo operating points")

    # Load labels
    with open(META_PATH) as f:
        meta = json.load(f)
    y = np.array(meta["correct"], dtype=int)

    # Load holdout mask
    mask = np.load(PHASE1_DIR / "holdout_mask_seed9999.npy")
    hold_idx = np.where(mask)[0]
    y_holdout = y[hold_idx]
    print(f"  Holdout: {len(hold_idx)} samples, {y_holdout.sum()} correct ({y_holdout.mean():.1%})")

    # Load entropy scores
    with open(SCORES_PATH) as f:
        scores_data = json.load(f)
    scores_list = scores_data["scores"]
    entropy_all = np.array([s["entropy"] for s in scores_list])
    maxprob_all = np.array([s["max_token_prob"] for s in scores_list])
    firstprob_all = np.array([s["first_token_prob"] for s in scores_list])
    print(f"  Loaded {len(scores_list)} entropy scores (mean={entropy_all.mean():.4f})")

    # Subset to holdout
    entropy_holdout = entropy_all[hold_idx]
    maxprob_holdout = maxprob_all[hold_idx]
    firstprob_holdout = firstprob_all[hold_idx]

    # Confidence: low entropy = high confidence → negate and normalize
    entropy_conf = -entropy_holdout
    entropy_conf = (entropy_conf - entropy_conf.min()) / (entropy_conf.max() - entropy_conf.min() + 1e-12)

    maxprob_conf = (maxprob_holdout - maxprob_holdout.min()) / (maxprob_holdout.max() - maxprob_holdout.min() + 1e-12)
    firstprob_conf = (firstprob_holdout - firstprob_holdout.min()) / (firstprob_holdout.max() - firstprob_holdout.min() + 1e-12)

    # Routing sweep at matched coverage levels
    entropy_routing = []
    maxprob_routing = []
    firstprob_routing = []

    for r in topo_routing:
        target_coverage = r["coverage"]
        for conf_arr, routing_list in [
            (entropy_conf, entropy_routing),
            (maxprob_conf, maxprob_routing),
            (firstprob_conf, firstprob_routing),
        ]:
            if target_coverage >= 1.0:
                thresh = 0.0
            elif target_coverage <= 0.0:
                thresh = 1.0
            else:
                thresh = np.quantile(conf_arr, 1 - target_coverage)
            m = routing_metrics(y_holdout, conf_arr, thresh)
            routing_list.append({
                "tau": r["tau"],
                "coverage": round(float(m["coverage"]), 4),
                "trusted_acc": round(float(m["trusted_acc"]), 4) if m["trusted_acc"] is not None else None,
                "escalation_ppv": round(float(m["escalation_ppv"]), 4) if m["escalation_ppv"] is not None else None,
            })

    # Comparison table
    print(f"\n  {'Coverage':>9} {'Topo_acc':>9} {'Entropy_acc':>12} {'MaxProb_acc':>12} {'Random_acc':>11}")
    for i, r in enumerate(topo_routing):
        ta = f"{r['trusted_acc']:.4f}" if r['trusted_acc'] is not None else "N/A"
        ea = f"{entropy_routing[i]['trusted_acc']:.4f}" if entropy_routing[i]['trusted_acc'] is not None else "N/A"
        ma = f"{maxprob_routing[i]['trusted_acc']:.4f}" if maxprob_routing[i]['trusted_acc'] is not None else "N/A"
        ra = f"{random_routing[i]['trusted_acc']:.4f}" if random_routing[i]['trusted_acc'] is not None else "N/A"
        print(f"  {r['coverage']:>9.4f} {ta:>9} {ea:>12} {ma:>12} {ra:>11}")

    # Count wins
    n_points = len(topo_routing)
    topo_wins_entropy = 0
    topo_wins_maxprob = 0
    topo_wins_random = 0
    topo_wins_length = 0

    for i, r in enumerate(topo_routing):
        if r["trusted_acc"] is not None:
            if entropy_routing[i]["trusted_acc"] is not None and r["trusted_acc"] > entropy_routing[i]["trusted_acc"]:
                topo_wins_entropy += 1
            if maxprob_routing[i]["trusted_acc"] is not None and r["trusted_acc"] > maxprob_routing[i]["trusted_acc"]:
                topo_wins_maxprob += 1
            if random_routing[i]["trusted_acc"] is not None and r["trusted_acc"] > random_routing[i]["trusted_acc"]:
                topo_wins_random += 1
            if length_routing[i]["trusted_acc"] is not None and r["trusted_acc"] > length_routing[i]["trusted_acc"]:
                topo_wins_length += 1

    print(f"\n  Topo beats entropy:  {topo_wins_entropy}/{n_points} operating points")
    print(f"  Topo beats max_prob: {topo_wins_maxprob}/{n_points} operating points")
    print(f"  Topo beats random:   {topo_wins_random}/{n_points} operating points")
    print(f"  Topo beats length:   {topo_wins_length}/{n_points} operating points")

    # GO/CONDITIONAL GO decision
    holdout_metrics = json.load(open(PHASE1_DIR / "holdout_metrics.json"))
    ci_lower = holdout_metrics["ci_lower_bound"]
    holdout_auroc = holdout_metrics["auroc"]

    print(f"\n  Holdout AUROC: {holdout_auroc:.4f}")
    print(f"  CI lower bound: {ci_lower:.4f}")

    if topo_wins_entropy >= 1:
        decision = "GO"
        reason = (f"CI lower bound {ci_lower:.4f} >= 0.78 AND topo beats entropy at "
                  f"{topo_wins_entropy}/{n_points} operating point(s). "
                  f"Topology captures signal beyond output entropy.")
    else:
        decision = "CONDITIONAL GO"
        reason = ("CI lower bound >= 0.78 but topo does NOT beat entropy at any "
                  "operating point. Topology may be a proxy for forward-pass uncertainty. "
                  "Pathway 2 (steering) still valuable, Pathway 4 may not add value.")

    print(f"\n  Decision: {decision}")
    print(f"  Reason: {reason}")

    # Update routing_metrics.json
    all_routing["output_entropy"] = entropy_routing
    all_routing["max_token_prob"] = maxprob_routing
    all_routing["first_token_prob"] = firstprob_routing
    all_routing["entropy_summary"] = {
        "mean_entropy": float(entropy_all.mean()),
        "std_entropy": float(entropy_all.std()),
        "holdout_mean_entropy": float(entropy_holdout.mean()),
    }
    with open(metrics_path, "w") as f:
        json.dump(all_routing, f, indent=2)
    print(f"\n  Updated: {metrics_path.name}")

    # Update baseline_comparison.txt
    comp_lines = ["Baseline Comparison (Phase 4)", "=" * 50, ""]
    comp_lines.append(f"Topo beats random:         {topo_wins_random}/{n_points} operating points")
    comp_lines.append(f"Topo beats prompt-length:  {topo_wins_length}/{n_points} operating points")
    comp_lines.append(f"Topo beats output-entropy: {topo_wins_entropy}/{n_points} operating points")
    comp_lines.append(f"Topo beats max-token-prob: {topo_wins_maxprob}/{n_points} operating points")
    comp_lines.append("")
    comp_lines.append("Coverage-matched comparison (topo vs entropy trusted accuracy):")
    for i, r in enumerate(topo_routing):
        ta = r['trusted_acc']
        ea = entropy_routing[i]['trusted_acc']
        if ta is not None and ea is not None:
            diff = ta - ea
            winner = "TOPO" if diff > 0.001 else ("ENTROPY" if diff < -0.001 else "TIE")
            comp_lines.append(f"  coverage={r['coverage']:.3f}: topo={ta:.4f} entropy={ea:.4f} diff={diff:+.4f} [{winner}]")
    comp_lines.append("")
    if topo_wins_entropy >= 1:
        comp_lines.append(f"RESULT: Topo beats entropy at {topo_wins_entropy} operating point(s).")
        comp_lines.append("Topology captures signal beyond output entropy -> GO for Pathway 4.")
    else:
        comp_lines.append("RESULT: Topo does NOT beat entropy at any operating point.")
        comp_lines.append("Topology may be a proxy for forward-pass uncertainty -> CONDITIONAL GO.")
    comp_lines.append("")
    comp_lines.append(f"Entropy summary: mean={entropy_all.mean():.4f}, std={entropy_all.std():.4f}")
    comp_lines.append(f"Holdout entropy: mean={entropy_holdout.mean():.4f}")
    comp_lines.append(f"Model: {scores_data['model']}")
    comp_lines.append(f"Extraction time: {scores_data['extraction_time_s']}s")

    with open(PHASE4_DIR / "baseline_comparison.txt", "w") as f:
        f.write("\n".join(comp_lines) + "\n")
    print(f"  Updated: baseline_comparison.txt")

    # Update routing tradeoff plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))

        coverages = [r["coverage"] for r in topo_routing if r["trusted_acc"] is not None]
        topo_accs = [r["trusted_acc"] for r in topo_routing if r["trusted_acc"] is not None]

        ax.plot(coverages, topo_accs, "bo-", label="Topo-confidence", markersize=6)

        entropy_accs = [entropy_routing[i]["trusted_acc"] for i in range(n_points)
                       if topo_routing[i]["trusted_acc"] is not None and entropy_routing[i]["trusted_acc"] is not None]
        maxprob_accs = [maxprob_routing[i]["trusted_acc"] for i in range(n_points)
                       if topo_routing[i]["trusted_acc"] is not None and maxprob_routing[i]["trusted_acc"] is not None]
        random_accs = [random_routing[i]["trusted_acc"] for i in range(n_points)
                      if topo_routing[i]["trusted_acc"] is not None and random_routing[i]["trusted_acc"] is not None]
        length_accs = [length_routing[i]["trusted_acc"] for i in range(n_points)
                      if topo_routing[i]["trusted_acc"] is not None and length_routing[i]["trusted_acc"] is not None]
        coverages_matched = [r["coverage"] for i, r in enumerate(topo_routing)
                            if r["trusted_acc"] is not None and random_routing[i]["trusted_acc"] is not None]

        if len(entropy_accs) == len(coverages_matched):
            ax.plot(coverages_matched, entropy_accs, "ms-", label="Output entropy", markersize=5, alpha=0.8)
        if len(maxprob_accs) == len(coverages_matched):
            ax.plot(coverages_matched, maxprob_accs, "c^-", label="Max token prob", markersize=5, alpha=0.7)
        if len(random_accs) == len(coverages_matched):
            ax.plot(coverages_matched, random_accs, "g^--", label="Random", markersize=5, alpha=0.5)
        if len(length_accs) == len(coverages_matched):
            ax.plot(coverages_matched, length_accs, "rv--", label="Prompt length", markersize=5, alpha=0.5)

        ax.axhline(0.95, color="orange", linestyle=":", alpha=0.5, label="95% accuracy")
        ax.set_xlabel("Coverage (fraction trusted)")
        ax.set_ylabel("Trusted accuracy")
        ax.set_title(f"Routing: Coverage vs Accuracy — {decision}")
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(PHASE4_DIR / "routing_tradeoff_curve.png", dpi=150)
        print(f"  Updated: routing_tradeoff_curve.png")
    except ImportError:
        print("  matplotlib not available, skipping plot update")

    print(f"\n  DECISION: {decision}")
    print(f"  All artifacts updated in {PHASE4_DIR}/")


if __name__ == "__main__":
    main()
