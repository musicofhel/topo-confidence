#!/usr/bin/env python3
"""Exp 1 analysis: temporal PR curve + correct/incorrect split, per model.

Reads per-problem npz files from exp1_cross_model/data/<subdir>/ and produces:
  - results.json       — temporal + depth PR curves with 95% bootstrap CIs
  - temporal_pr_curve.png
  - depth_pr_prefill_vs_final.png

CPU only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP1_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP1_DIR / "data"

MODELS = [
    ("phi3mini", "Phi-3-mini-4k-instruct"),
    ("llama32_1b", "Llama-3.2-1B-Instruct"),
]

N_BOOT = 50  # confirmatory CIs — pattern existence already known from Qwen
SEED = 9999
TARGET_POSITIONS = [1, 10, 25, 50, 100, 200]
MIN_N_PER_POS = 20


def participation_ratio(X: np.ndarray) -> float:
    """PR of rows of X (centered). Equal to (sum s_i^2)^2 / sum s_i^4."""
    if X.shape[0] < 2:
        return float("nan")
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    s2 = s.astype(np.float64) ** 2
    denom = (s2 ** 2).sum()
    if denom <= 0:
        return float("nan")
    return float(s2.sum() ** 2 / denom)


def bootstrap_pr(X: np.ndarray, n_boot: int, seed: int) -> tuple[float, float, float]:
    n = len(X)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = participation_ratio(X)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = participation_ratio(X[idx])
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def load_model_data(subdir: str) -> dict:
    """Load all per-problem arrays into aligned lists.

    Returns dict with:
      prefill_all:  (n_problems, n_layers_total, d) float32
      final_all:    (n_problems, n_layers_total, d) float32
      twothirds_by_pos: dict[int, dict] with keys
          "vecs":  (n_problems_reaching_pos, d) float32
          "problem_idx": (n_problems_reaching_pos,) int
      final_twothirds:
          "vecs":  (n_problems, d) — per-problem final-generated-token at 2/3 layer
          "problem_idx": (n_problems,) int (0..n-1)
      correct:      (n_problems,) bool
      n_gen_tokens: (n_problems,) int
      twothirds_layer_idx: int
    """
    d = DATA_DIR / subdir
    files = sorted(d.glob("problem_*.npz"))
    if not files:
        raise FileNotFoundError(f"No problem_*.npz files in {d}")

    prefills, finals = [], []
    corrects, n_gen = [], []
    twothirds_by_pos: dict[int, dict] = {}
    final_vecs, final_problem_idx = [], []
    twothirds_layer_idx = None

    for pi, f in enumerate(files):
        data = np.load(f, allow_pickle=True)
        prefills.append(data["prefill_all_layers"])
        finals.append(data["final_all_layers"])
        corrects.append(bool(data["correct"]))
        n_gen.append(int(data["n_gen_tokens"]))
        twothirds_layer_idx = int(data["twothirds_layer_idx"])

        positions = np.asarray(data["positions_actual"]).astype(int)
        tt = np.asarray(data["twothirds_positions"])
        for i, p in enumerate(positions):
            entry = twothirds_by_pos.setdefault(int(p), {"vecs": [], "problem_idx": []})
            entry["vecs"].append(tt[i])
            entry["problem_idx"].append(pi)

        # Per-problem final (last captured position = n_gen_tokens)
        if len(positions) > 0:
            final_vecs.append(tt[-1])
            final_problem_idx.append(pi)

    # Stack
    for p, entry in twothirds_by_pos.items():
        entry["vecs"] = np.stack(entry["vecs"], axis=0).astype(np.float32)
        entry["problem_idx"] = np.asarray(entry["problem_idx"], dtype=int)

    return {
        "prefill_all": np.stack(prefills, axis=0).astype(np.float32),
        "final_all": np.stack(finals, axis=0).astype(np.float32),
        "twothirds_by_pos": twothirds_by_pos,
        "final_twothirds": {
            "vecs": np.stack(final_vecs, axis=0).astype(np.float32),
            "problem_idx": np.asarray(final_problem_idx, dtype=int),
        },
        "correct": np.asarray(corrects, dtype=bool),
        "n_gen_tokens": np.asarray(n_gen, dtype=int),
        "twothirds_layer_idx": twothirds_layer_idx,
    }


def _pr_triple_split(X: np.ndarray, corrects: np.ndarray) -> dict:
    a = bootstrap_pr(X, N_BOOT, SEED)
    c = bootstrap_pr(X[corrects], N_BOOT, SEED)
    i = bootstrap_pr(X[~corrects], N_BOOT, SEED)
    return {
        "all": a,
        "correct": c,
        "incorrect": i,
        "n": int(X.shape[0]),
        "n_correct": int(corrects.sum()),
        "n_incorrect": int((~corrects).sum()),
    }


def temporal_pr(mdata: dict) -> dict:
    """Temporal PR curve: one row per kept position (+ 'final')."""
    corrects_all = mdata["correct"]
    rows = []

    for p in TARGET_POSITIONS:
        entry = mdata["twothirds_by_pos"].get(p)
        if entry is None or len(entry["vecs"]) < MIN_N_PER_POS:
            continue
        X = entry["vecs"]
        c = corrects_all[entry["problem_idx"]]
        r = _pr_triple_split(X, c)
        r["position"] = p
        rows.append(r)

    # Final position — one per problem (at n_gen_tokens_i)
    fe = mdata["final_twothirds"]
    X = fe["vecs"]
    c = corrects_all[fe["problem_idx"]]
    r = _pr_triple_split(X, c)
    r["position"] = "final"
    rows.append(r)

    return {"rows": rows, "twothirds_layer_idx": mdata["twothirds_layer_idx"]}


def depth_pr(H: np.ndarray, corrects: np.ndarray, label: str = "") -> list[dict]:
    """PR per layer. H is (n_problems, n_layers_total, d)."""
    import time
    out = []
    n_layers = H.shape[1]
    t0 = time.time()
    for l in range(n_layers):
        r = _pr_triple_split(H[:, l, :], corrects)
        r["layer"] = l
        out.append(r)
        elapsed = time.time() - t0
        rate = (l + 1) / elapsed if elapsed > 0 else 0
        eta = (n_layers - l - 1) / rate if rate > 0 else 0
        print(
            f"    [{label}] layer {l+1}/{n_layers}: "
            f"PR_all={r['all'][0]:.2f} PR_c={r['correct'][0]:.2f} "
            f"PR_i={r['incorrect'][0]:.2f} | {elapsed:.0f}s elapsed, ETA {eta:.0f}s",
            flush=True,
        )
    return out


def analyze_one(subdir: str, label: str) -> dict:
    print(f"\n=== {label} ({subdir}) ===")
    mdata = load_model_data(subdir)
    corrects = mdata["correct"]
    n_total = len(corrects)
    n_correct = int(corrects.sum())
    print(f"  n={n_total}, correct={n_correct} ({n_correct/n_total:.1%})")
    print(f"  2/3-depth layer idx: {mdata['twothirds_layer_idx']}")

    print("  temporal PR curve...", flush=True)
    temporal = temporal_pr(mdata)
    print("  depth PR @ prefill...", flush=True)
    depth_prefill = depth_pr(mdata["prefill_all"], corrects, label=f"{subdir} prefill")
    print("  depth PR @ final...", flush=True)
    depth_final = depth_pr(mdata["final_all"], corrects, label=f"{subdir} final")

    print(f"  {'pos':>6} {'all':>22} {'correct':>22} {'incorrect':>22} {'n/nC/nI':>16}")
    for r in temporal["rows"]:
        a, c, i = r["all"], r["correct"], r["incorrect"]
        nstr = f"{r['n']}/{r['n_correct']}/{r['n_incorrect']}"
        print(
            f"  {str(r['position']):>6} "
            f"{a[0]:7.2f} [{a[1]:6.2f},{a[2]:6.2f}] "
            f"{c[0]:7.2f} [{c[1]:6.2f},{c[2]:6.2f}] "
            f"{i[0]:7.2f} [{i[1]:6.2f},{i[2]:6.2f}] "
            f"{nstr:>16}"
        )

    return {
        "subdir": subdir,
        "label": label,
        "n_total": n_total,
        "n_correct": n_correct,
        "accuracy": n_correct / n_total,
        "twothirds_layer_idx": mdata["twothirds_layer_idx"],
        "temporal": temporal,
        "depth_prefill": depth_prefill,
        "depth_final": depth_final,
    }


def plot_temporal(results: list[dict], out_path: Path):
    fig, axes = plt.subplots(1, len(results), figsize=(6.5 * len(results), 4.5), sharey=False)
    if len(results) == 1:
        axes = [axes]
    for ax, r in zip(axes, results):
        rows = r["temporal"]["rows"]
        x = np.arange(len(rows))
        labels = [str(row["position"]) for row in rows]
        for key, color in [("all", "black"), ("correct", "tab:green"), ("incorrect", "tab:red")]:
            pts = np.array([row[key][0] for row in rows])
            lo = np.array([row[key][1] for row in rows])
            hi = np.array([row[key][2] for row in rows])
            ax.plot(x, pts, "-o", color=color, label=key, linewidth=2, markersize=5)
            ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Generation position")
        ax.set_ylabel("Participation Ratio")
        ax.set_title(
            f"{r['label']} — temporal PR @ L{r['twothirds_layer_idx']} (2/3 depth)"
        )
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_depth(results: list[dict], out_path: Path):
    fig, axes = plt.subplots(2, len(results), figsize=(6.5 * len(results), 8.5), sharex=False)
    if len(results) == 1:
        axes = axes.reshape(2, 1)
    for col, r in enumerate(results):
        for row, (key, title) in enumerate(
            [("depth_prefill", "prefill-end"), ("depth_final", "final token")]
        ):
            ax = axes[row, col]
            d = r[key]
            x = np.arange(len(d))
            for split, color in [("all", "black"), ("correct", "tab:green"), ("incorrect", "tab:red")]:
                pts = np.array([row_entry[split][0] for row_entry in d])
                lo = np.array([row_entry[split][1] for row_entry in d])
                hi = np.array([row_entry[split][2] for row_entry in d])
                ax.plot(x, pts, "-", color=color, label=split, linewidth=1.5)
                ax.fill_between(x, lo, hi, color=color, alpha=0.12)
            ax.axvline(r["twothirds_layer_idx"], color="tab:blue", linestyle=":",
                       alpha=0.6, label=f"2/3-depth L{r['twothirds_layer_idx']}")
            ax.set_xlabel("Layer index (0=embed)")
            ax.set_ylabel("Participation Ratio")
            ax.set_title(f"{r['label']} — depth PR @ {title}")
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("=" * 70)
    print("Exp 1 analysis: temporal + depth PR curves, cross-model")
    print("=" * 70)

    results = []
    for subdir, label in MODELS:
        d = DATA_DIR / subdir
        if not d.exists() or not list(d.glob("problem_*.npz")):
            print(f"  [skip] {subdir} — no data in {d}")
            continue
        results.append(analyze_one(subdir, label))

    if not results:
        print("No data to analyze. Run extract.py first.")
        return

    plot_temporal(results, EXP1_DIR / "temporal_pr_curve.png")
    plot_depth(results, EXP1_DIR / "depth_pr_prefill_vs_final.png")

    summary = {
        "models": [
            {
                "subdir": r["subdir"],
                "label": r["label"],
                "n_total": r["n_total"],
                "n_correct": r["n_correct"],
                "accuracy": r["accuracy"],
                "twothirds_layer_idx": r["twothirds_layer_idx"],
                "temporal": {
                    "rows": r["temporal"]["rows"],
                },
                "depth_prefill": r["depth_prefill"],
                "depth_final": r["depth_final"],
            }
            for r in results
        ],
    }
    (EXP1_DIR / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"  Saved: {EXP1_DIR/'results.json'}")


if __name__ == "__main__":
    main()
