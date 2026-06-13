"""transfer.py — the transfer panel runner (Phase 1 adjudication engine).

run_panel(probe_names) computes, per transfer tier, like-for-like:
  T1 cross-category  : LOCO leave-one-MATH-subject-out on Qwen-1.5B   (all probes)
  T2 cross-domain    : Qwen-1.5B MATH <-> BBH                          (all probes)
  T3 cross-scale     : Qwen-1.5B MATH -> Qwen-7B MATH                  ("all" probes only)

Per tier reports: per-cell AUROC + n, worst cell, robust aggregate (mean
off-diagonal), in-domain OOF, overfit gap, and the incremental gate
(feature+free vs free) paired DeLong. The free baseline (length+logprob) runs
on every tier as a first-class candidate.

T4 (cross-arch) and T5 (held-out families) need Phase 2/3 generation and are not
cache-ready; run_panel skips them with a logged note.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

import activation_loader as AL
import probes as PR
from metrics import frozen_folds, auroc_sym, delong_paired_test, overfit_gap


# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

def _oof(probe, cell, folds):
    return PR.fit_score_oof(probe, cell, folds)


def _transfer(probe, train_cell, test_cell):
    return PR.fit_score_transfer(probe, train_cell, test_cell)


def _auroc(y, s):
    return auroc_sym(y, s)


def _augment_free(probe, cell):
    """Feature matrix augmented with the free baseline (length+logprob) for the
    incremental gate. Returns a synthetic logreg probe over [feat, free]."""
    free = np.column_stack([cell.n_gen_tokens, cell.mean_logprob])
    feat = probe.featurize(cell)
    return np.concatenate([feat, free], axis=1)


# ---------------------------------------------------------------------------
# Incremental gate (V3-2): (feature + free) vs (free) on the same OOF folds
# ---------------------------------------------------------------------------

def incremental_gate(probe, cell, folds):
    """Paired DeLong of (readout-score + free) vs free-only OOF scores.

    The correct V3-2 test: take the probe's own 1-D OOF readout score (NOT its
    raw high-dim features — that would overfit the gate), combine it with the two
    free scalars in a logistic, and test whether the combination beats free
    alone. Positive significant delta => the readout adds over the free baseline.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    free_probe = PR.REGISTRY["free_baseline"]
    s_free = _oof(free_probe, cell, folds)
    s_feat = _oof(probe, cell, folds)            # 1-D readout score
    y = cell.y

    # orient both "higher = correct"
    if roc_auc_score(y, s_free) < 0.5:
        s_free = -s_free
    if roc_auc_score(y, s_feat) < 0.5:
        s_feat = -s_feat

    # combined logreg on [readout_score, length, logprob], OOF on same folds
    Xc = np.column_stack([s_feat, cell.n_gen_tokens, cell.mean_logprob])
    s_comb = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(Xc[tr])
        est = LogisticRegression(max_iter=2000).fit(sc.transform(Xc[tr]), y[tr])
        s_comb[te] = est.predict_proba(sc.transform(Xc[te]))[:, 1]

    test = delong_paired_test(y, s_comb, s_free)
    return {"auroc_readout": float(_auroc(y, s_feat)),
            "auroc_feat_plus_free": float(max(roc_auc_score(y, s_comb),
                                              1 - roc_auc_score(y, s_comb))),
            "auroc_free": float(_auroc(y, s_free)),
            "delta": test["delta"], "p": test["p"], "z": test["z"]}


# ---------------------------------------------------------------------------
# Tier runners
# ---------------------------------------------------------------------------

def run_T1_loco(probe_names):
    """Leave-one-MATH-category-out on Qwen-1.5B (cross-category)."""
    cell = AL.load_cell("qwen1.5b", "math")
    masks = AL.loco_cells(cell)
    out = {}
    for name in probe_names:
        probe = PR.REGISTRY[name]
        per_cell = {}
        for subj, mask in masks.items():
            tr = ~mask
            te = mask
            # build sub-cells via index views
            train_cell = _subset(cell, tr)
            test_cell = _subset(cell, te)
            if probe.kind == "dom":
                from metrics import DomProbe
                s = DomProbe().fit(probe.featurize(train_cell), train_cell.y).score(
                    probe.featurize(test_cell))
            else:
                s = _transfer(probe, train_cell, test_cell)
            if len(np.unique(test_cell.y)) < 2:
                continue
            per_cell[subj] = {"auroc": _auroc(test_cell.y, s),
                              "n": int(test_cell.n),
                              "n_pos": int(test_cell.y.sum())}
        aurocs = [v["auroc"] for v in per_cell.values()]
        out[name] = {"per_cell": per_cell,
                     "worst": float(min(aurocs)),
                     "worst_cell": min(per_cell, key=lambda k: per_cell[k]["auroc"]),
                     "aggregate": float(np.mean(aurocs))}
    return out


def run_T2_crossdomain(probe_names):
    """Qwen-1.5B MATH <-> BBH (cross-domain, same model)."""
    math = AL.load_cell("qwen1.5b", "math")
    bbh = AL.load_cell("qwen1.5b", "bbh")
    out = {}
    for name in probe_names:
        probe = PR.REGISTRY[name]
        cells = {}
        for direction, (tr_c, te_c) in {"math->bbh": (math, bbh),
                                        "bbh->math": (bbh, math)}.items():
            s = _transfer(probe, tr_c, te_c)
            cells[direction] = {"auroc": _auroc(te_c.y, s), "n": int(te_c.n)}
        aurocs = [v["auroc"] for v in cells.values()]
        out[name] = {"per_cell": cells, "worst": float(min(aurocs)),
                     "aggregate": float(np.mean(aurocs))}
    return out


def run_T3_crossscale(probe_names):
    """Qwen-1.5B MATH -> Qwen-7B MATH. Only fixed-dim/scalar ('all') probes."""
    m15 = AL.load_cell("qwen1.5b", "math")
    m7 = AL.load_cell("qwen7b", "math")
    out = {}
    for name in probe_names:
        probe = PR.REGISTRY[name]
        if probe.tiers != "all":
            out[name] = {"skipped": "hidden-dim-bound; cannot cross scale"}
            continue
        cells = {}
        for direction, (tr_c, te_c) in {"1.5b->7b": (m15, m7),
                                        "7b->1.5b": (m7, m15)}.items():
            s = _transfer(probe, tr_c, te_c)
            cells[direction] = {"auroc": _auroc(te_c.y, s), "n": int(te_c.n)}
        aurocs = [v["auroc"] for v in cells.values()]
        out[name] = {"per_cell": cells, "worst": float(min(aurocs)),
                     "aggregate": float(np.mean(aurocs))}
    return out


def in_domain(probe_names):
    """In-domain OOF AUROC + incremental gate on Qwen-1.5B MATH (the alarm ref)."""
    cell = AL.load_cell("qwen1.5b", "math")
    folds = frozen_folds(cell.y)
    out = {}
    for name in probe_names:
        probe = PR.REGISTRY[name]
        s = _oof(probe, cell, folds)
        row = {"auroc": _auroc(cell.y, s)}
        if name not in ("free_baseline", "length_only", "mean_logprob_only"):
            row["gate"] = incremental_gate(probe, cell, folds)
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# Cell subsetting
# ---------------------------------------------------------------------------

def _subset(cell: AL.Cell, mask: np.ndarray) -> AL.Cell:
    return AL.Cell(
        model=cell.model, dataset=cell.dataset,
        y=cell.y[mask], mean_logprob=cell.mean_logprob[mask],
        n_gen_tokens=cell.n_gen_tokens[mask], subjects=cell.subjects[mask],
        _X_mean=cell._X_mean[mask], _X_prefill=cell._X_prefill[mask],
        _X_last=cell._X_last[mask],
        d2h_attn_entropy=None if cell.d2h_attn_entropy is None
        else cell.d2h_attn_entropy[mask],
    )


def run_panel(probe_names, tiers=("in_domain", "T1", "T2", "T3")):
    result = {}
    if "in_domain" in tiers:
        result["in_domain"] = in_domain(probe_names)
    if "T1" in tiers:
        result["T1_loco"] = run_T1_loco(probe_names)
    if "T2" in tiers:
        result["T2_crossdomain"] = run_T2_crossdomain(probe_names)
    if "T3" in tiers:
        result["T3_crossscale"] = run_T3_crossscale(probe_names)
    result["_note"] = ("T4 cross-arch and T5 held-out families need Phase 2/3 "
                       "generation; not cache-ready.")
    return result
