#!/usr/bin/env python3
"""Phase 1B.4 — 1.5Bx7B overlap table + hull-honest cascade re-adjudication.

Builds the per-problem [correct_1.5B_K1, correct_1.5B_K8maj, correct_7B] table
(which does not exist anywhere — V4-5), reports which MATH categories 7B
actually rescues, and re-scores the pre-gen / hybrid cascade frontiers against
the RANDOM-MIX HULL (the FE19 comparator) with paired bootstrap SEs at matched
cost.

Cost model (from FU2): keep-on-1.5B = 1, escalate-to-7B = 5 (pre-gen decision,
no wasted draft). cost_rel = (5 - 4*pct_kept)/5 = 1 - 0.8*pct_kept relative to
all-7B. Random-mix hull = line from (cost_rel 0.2, acc 0.486) to (1.0, 0.732).

Outputs:
  results/1b4_overlap_table.csv
  results/1b4_overlap.json
"""
from __future__ import annotations
import warnings; warnings.filterwarnings("ignore")

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
ROOT = Path("/home/musicofhel/topo-confidence")

import activation_loader as AL
import k8_lib as K8
from metrics import oof_dom_scores, SEED


def _manifest(path):
    return json.loads((path / "manifest.json").read_text())


def build_table():
    m15 = AL.load_cell("qwen1.5b", "math")
    m7 = AL.load_cell("qwen7b", "math")
    # alignment check via manifests (unique_id order must match)
    man15 = _manifest(AL.ROOT / "pathway8_layerwise" / "data" / "math500")
    man7 = _manifest(AL.ROOT / "pathway11_h100" / "data" / "math500_7b")
    ids15 = [p["unique_id"] for p in man15["problems"]]
    ids7 = [p["unique_id"] for p in man7["problems"]]
    assert ids15 == ids7, "1.5B and 7B problem ordering differ!"

    k8 = K8.load_all_k8()
    assert k8["n"] == m15.n, f"k8 n={k8['n']} != math n={m15.n}"
    k8_maj = np.array([K8.plain_majority_correct(k8["answers"][i], k8["correct"][i])
                       for i in range(k8["n"])])

    subjects = m15.subjects
    table = {
        "unique_id": ids15,
        "subject": subjects.tolist(),
        "correct_1p5b_k1": m15.y.astype(int).tolist(),
        "correct_1p5b_k8maj": k8_maj.astype(int).tolist(),
        "correct_7b": m7.y.astype(int).tolist(),
    }
    return table, m15, m7, k8_maj


def category_rescues(table):
    """Per-category: 7B rescues of K=1 failures, K=8 rescues, where 7B fails too."""
    subj = np.array(table["subject"])
    k1 = np.array(table["correct_1p5b_k1"], dtype=bool)
    k8 = np.array(table["correct_1p5b_k8maj"], dtype=bool)
    b7 = np.array(table["correct_7b"], dtype=bool)
    rows = {}
    for s in sorted(set(subj.tolist())):
        m = subj == s
        n = int(m.sum())
        k1f = ~k1 & m            # 1.5B-K1 wrong
        rows[s] = {
            "n": n,
            "acc_k1": float(k1[m].mean()),
            "acc_k8maj": float(k8[m].mean()),
            "acc_7b": float(b7[m].mean()),
            "k1_fail": int(k1f.sum()),
            "rescued_by_7b": int((k1f & b7).sum()),
            "rescued_by_k8maj": int((k1f & k8).sum()),
            "k1_fail_and_7b_also_fails": int((k1f & ~b7).sum()),
        }
    # global
    k1f = ~k1
    rows["_ALL"] = {
        "n": len(k1),
        "acc_k1": float(k1.mean()), "acc_k8maj": float(k8.mean()),
        "acc_7b": float(b7.mean()),
        "k1_fail": int(k1f.sum()),
        "rescued_by_7b": int((k1f & b7).sum()),
        "rescued_by_k8maj": int((k1f & k8).sum()),
        "k8maj_recovers_pct_of_k1_fails": float((k1f & k8).sum() / k1f.sum()),
        "7b_recovers_pct_of_k1_fails": float((k1f & b7).sum() / k1f.sum()),
    }
    return rows


# ---------------------------------------------------------------------------
# Cascade gates + random-mix hull
# ---------------------------------------------------------------------------

def build_gates(m15):
    """Pre-gen prefill-DoM gate (0.7731) and hybrid gate (DoM+logprob+length)."""
    y = m15.y
    Xpf = m15.X("prefill")          # L19 prefill
    s_pre = oof_dom_scores(Xpf, y)  # higher = more likely correct
    # hybrid: per-fold logistic on [dom_score, logprob, n_tokens] (FU2 recipe)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    s_hyb = np.zeros(len(y))
    for tr, te in skf.split(Xpf, y):
        from metrics import DomProbe
        dp = DomProbe().fit(Xpf[tr], y[tr])
        feat_tr = np.column_stack([dp.score(Xpf[tr]), m15.mean_logprob[tr],
                                   m15.n_gen_tokens[tr]])
        feat_te = np.column_stack([dp.score(Xpf[te]), m15.mean_logprob[te],
                                   m15.n_gen_tokens[te]])
        lr = LogisticRegression(max_iter=2000).fit(feat_tr, y[tr])
        s_hyb[te] = lr.predict_proba(feat_te)[:, 1]
    return s_pre, s_hyb


def cascade_sweep(scores, y15, y7, n_tau=51):
    """Keep the highest-`scores` problems on 1.5B, escalate the rest to 7B.

    Returns list of (pct_kept, accuracy, cost_rel). cost_rel = 1 - 0.8*pct_kept.
    """
    n = len(y15)
    order = np.argsort(-scores)         # highest score (most confident) first
    rows = []
    for k in np.linspace(0, n, n_tau).astype(int):
        keep = order[:k]
        esc = order[k:]
        correct = y15[keep].sum() + y7[esc].sum()
        acc = correct / n
        pct_kept = k / n
        cost_rel = 1.0 - 0.8 * pct_kept
        rows.append({"pct_kept": float(pct_kept), "accuracy": float(acc),
                     "cost_rel": float(cost_rel), "n_kept": int(k)})
    return rows


def hull_acc_at_cost(cost_rel, acc15, acc7):
    """Random-mix line: pct_kept = (1-cost_rel)/0.8; acc linear in pct_kept."""
    pct_kept = (1.0 - cost_rel) / 0.8
    pct_kept = np.clip(pct_kept, 0.0, 1.0)
    return pct_kept * acc15 + (1.0 - pct_kept) * acc7


def adjudicate(scores, y15, y7, label, n_boot=2000, seed=SEED):
    """For each operating point, gap = cascade_acc - hull_acc at matched cost.
    Report the max gap and a paired bootstrap SE on the accuracy difference at
    the best operating point (paired over problems)."""
    rows = cascade_sweep(scores, y15, y7)
    acc15, acc7 = float(y15.mean()), float(y7.mean())
    best = None
    for r in rows:
        if r["pct_kept"] in (0.0, 1.0):
            continue
        hull = hull_acc_at_cost(r["cost_rel"], acc15, acc7)
        gap = r["accuracy"] - hull
        r["hull_acc"] = float(hull)
        r["gap_pp"] = float(100 * gap)
        if best is None or gap > best["gap_pp"] / 100:
            best = r
    # paired bootstrap SE at the best operating point: compare cascade outcome
    # vs a random-mix outcome with the SAME pct_kept, per-problem paired.
    rng = np.random.default_rng(seed)
    n = len(y15)
    k = best["n_kept"]
    order = np.argsort(-scores)
    keep_mask = np.zeros(n, dtype=bool); keep_mask[order[:k]] = True
    cascade_correct = np.where(keep_mask, y15, y7).astype(float)
    # random-mix correctness vector at same keep-fraction (expected per problem)
    p_keep = k / n
    randmix_expected = p_keep * y15.astype(float) + (1 - p_keep) * y7.astype(float)
    diffs = cascade_correct - randmix_expected
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return {"label": label, "best_point": best,
            "gap_pp_point": best["gap_pp"],
            "gap_pp_boot_mean": float(100 * boot.mean()),
            "gap_pp_boot_se": float(100 * boot.std()),
            "gap_pp_ci95": [float(100 * np.percentile(boot, 2.5)),
                            float(100 * np.percentile(boot, 97.5))],
            "significant": bool(np.percentile(boot, 2.5) > 0),
            "sweep": rows}


def main():
    table, m15, m7, k8_maj = build_table()
    cats = category_rescues(table)

    # write CSV
    csv = RESULTS / "1b4_overlap_table.csv"
    with open(csv, "w") as f:
        f.write("idx,unique_id,subject,correct_1p5b_k1,correct_1p5b_k8maj,correct_7b\n")
        for i in range(len(table["unique_id"])):
            f.write(f"{i},{table['unique_id'][i]},{table['subject'][i]},"
                    f"{table['correct_1p5b_k1'][i]},{table['correct_1p5b_k8maj'][i]},"
                    f"{table['correct_7b'][i]}\n")

    s_pre, s_hyb = build_gates(m15)
    adj_pre = adjudicate(s_pre, m15.y, m7.y, "pre-gen prefill-DoM (G1)")
    adj_hyb = adjudicate(s_hyb, m15.y, m7.y, "hybrid DoM+logprob+len (FU2)")

    out = {
        "anchors": {"acc_k1": float(m15.y.mean()), "acc_k8maj": float(k8_maj.mean()),
                    "acc_7b": float(m7.y.mean())},
        "category_rescues": cats,
        "cascade_vs_random_mix_hull": {
            "cost_model": "keep=1, escalate=5; cost_rel=1-0.8*pct_kept",
            "pregen": {k: v for k, v in adj_pre.items() if k != "sweep"},
            "hybrid": {k: v for k, v in adj_hyb.items() if k != "sweep"},
        },
    }
    save = RESULTS / "1b4_overlap.json"
    save.write_text(json.dumps(out, indent=2))

    # console summary
    a = out["anchors"]
    print(f"Anchors: K1={a['acc_k1']:.3f} K8maj={a['acc_k8maj']:.3f} 7B={a['acc_7b']:.3f}")
    g = cats["_ALL"]
    print(f"\nGlobal: {g['k1_fail']} K=1 failures; 7B rescues {g['rescued_by_7b']} "
          f"({g['7b_recovers_pct_of_k1_fails']:.1%}), K=8maj rescues "
          f"{g['rescued_by_k8maj']} ({g['k8maj_recovers_pct_of_k1_fails']:.1%})")
    print("\nPer-category 7B rescue of K=1 fails:")
    for s, r in cats.items():
        if s == "_ALL":
            continue
        print(f"  {s:22s} n={r['n']:3d} k1={r['acc_k1']:.2f} 7b={r['acc_7b']:.2f} "
              f"rescued={r['rescued_by_7b']:2d}/{r['k1_fail']:2d} "
              f"7b-also-fails={r['k1_fail_and_7b_also_fails']:2d}")
    print("\nCascade vs random-mix hull (FE19-honest):")
    for adj in (adj_pre, adj_hyb):
        b = adj["best_point"]
        print(f"  {adj['label']:34s} best @ keep={b['pct_kept']:.2f} "
              f"cost_rel={b['cost_rel']:.3f}: cascade={b['accuracy']:.3f} "
              f"hull={b['hull_acc']:.3f} gap={adj['gap_pp_point']:+.2f}pp "
              f"(boot {adj['gap_pp_boot_mean']:+.2f}+-{adj['gap_pp_boot_se']:.2f}pp, "
              f"{'SIG' if adj['significant'] else 'n.s.'})")
    print(f"\n→ {save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
