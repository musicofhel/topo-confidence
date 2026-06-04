"""Diagnostic: trace retrieval paths for failing synthetic eval cases."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from query import (
    _embed_query,
    _hyde_embed,
    _concept_expand,
    _path_paper_vec,
    _path_finding_vec,
    _path_finding_ft,
    _path_paper_ft,
    _path_tag_match,
    _path_dataset_match,
    _path_finding_neighborhood,
    rrf_merge,
    DEFAULT_WEIGHTS,
)


FAILING_CASES = [
    {
        "label": "activation-steering crowding (typical)",
        "query": "How does CAA compaire to HyperSteer in terms of activaton-steering methds for residual-stream interventions?",
        "missed": ("Paper", "2506.03292"),
        "hit": ("Paper", "2312.06681"),
    },
    {
        "label": "isolated paper (prefill tag, 4 neighbors)",
        "query": "Waht evidnce suports the idea that prefill direcion can predikt corectness in LLMs?",
        "missed": ("Paper", "2509.12886"),
        "hit": None,
    },
    {
        "label": "underscore tag (CalMO selective_prediction)",
        "query": "How do I use calibration techniques like Semantic Entropy Probes and CalMO together for selective prediction?",
        "missed": ("Paper", "2604.20614"),
        "hit": ("Paper", "2406.15927"),
    },
    {
        "label": "representation-engineering multi-hop",
        "query": "representation-engineering activation steering multi-layer vs single-layer intervention methods",
        "missed": ("Paper", "2602.04428"),
        "hit": ("Paper", "2403.05767"),
    },
    {
        "label": "confidence/calibration multi-hop",
        "query": "What evidence supports the use of linear probes in mechanistic-interpretability for steering model behavior?",
        "missed": ("Paper", "2603.25052"),
        "hit": ("Paper", "2602.15293"),
    },
]


def trace_case(case: dict):
    query = case["query"]
    missed_label, missed_id = case["missed"]
    missed_key = f"{missed_label}||{missed_id}"

    print(f"\n{'='*70}")
    print(f"  {case['label']}")
    print(f"  Query: {query[:100]}")
    print(f"  Missed: {missed_label}:{missed_id}")
    if case["hit"]:
        print(f"  Hit:    {case['hit'][0]}:{case['hit'][1]}")
    print(f"{'='*70}")

    query_vec = _embed_query(query)

    paths = {}
    paths["paper-vec"] = _path_paper_vec(query_vec)
    paths["finding-vec"] = _path_finding_vec(query_vec)

    hyde_vec = _hyde_embed(query)
    if hyde_vec:
        paths["paper-vec-hyde"] = _path_paper_vec(hyde_vec)
        paths["finding-vec-hyde"] = _path_finding_vec(hyde_vec)

    paths["finding-ft"] = _path_finding_ft(query)
    paths["paper-ft"] = _path_paper_ft(query)
    paths["tag-match"] = _path_tag_match(query)
    paths["dataset-match"] = _path_dataset_match(query)

    terms = _concept_expand(query)
    if terms:
        expanded = query + " " + " ".join(terms)
        paths["expand-tag"] = _path_tag_match(expanded, limit=8)
        paths["expand-ft-paper"] = _path_paper_ft(expanded, limit=10)
        paths["expand-ft-finding"] = _path_finding_ft(expanded, limit=5)

    finding_seed = paths["finding-vec"] + paths["finding-ft"]
    paths["finding-neighborhood"] = _path_finding_neighborhood(finding_seed)

    print("\n  Per-path results for MISSED paper:")
    found_in_any = False
    for path_id, results in sorted(paths.items()):
        for label, nid, rank in results:
            if label == missed_label and nid == missed_id:
                print(f"    {path_id:25s} → rank {rank}")
                found_in_any = True
                break
        else:
            total = len(results)
            print(f"    {path_id:25s} → NOT IN TOP {total}")

    # Show what DID appear in the top results for context
    print(f"\n  Top papers across all paths (unique):")
    all_papers = {}
    for path_id, results in paths.items():
        for label, nid, rank in results:
            if label == "Paper":
                key = nid
                if key not in all_papers:
                    all_papers[key] = []
                all_papers[key].append((path_id, rank))

    # Sort by number of paths that surface the paper
    by_coverage = sorted(all_papers.items(), key=lambda x: (-len(x[1]), min(r for _, r in x[1])))
    for nid, appearances in by_coverage[:15]:
        marker = " *** MISSED" if nid == missed_id else ""
        marker = " *** HIT" if case["hit"] and nid == case["hit"][1] else marker
        path_summary = ", ".join(f"{p}@{r}" for p, r in sorted(appearances, key=lambda x: x[1]))
        print(f"    {nid:16s} [{len(appearances)} paths] {path_summary}{marker}")

    # RRF merge
    weights = dict(DEFAULT_WEIGHTS)
    if hyde_vec:
        weights["paper-vec-hyde"] = 1.0
        weights["finding-vec-hyde"] = 1.0
    if terms:
        weights["expand-tag"] = 0.25
        weights["expand-ft-paper"] = 0.25
        weights["expand-ft-finding"] = 0.25

    merged = rrf_merge(paths, weights=weights, top_n=13)
    print(f"\n  RRF merge (top 13):")
    for i, r in enumerate(merged):
        marker = ""
        if r["label"] == missed_label and r["id"] == missed_id:
            marker = " *** MISSED"
        elif case["hit"] and r["label"] == case["hit"][0] and r["id"] == case["hit"][1]:
            marker = " *** HIT"
        print(f"    [{i+1:2d}] {r['label']:8s} {r['id']:16s} score={r['score']:.6f}{marker}")

    # Check if missed paper appears at all in expanded merge
    found_in_merge = any(
        r["label"] == missed_label and r["id"] == missed_id
        for r in rrf_merge(paths, weights=weights, top_n=50)
    )
    if not found_in_merge:
        print(f"    → {missed_id} NOT IN TOP 50 of RRF merge")
    elif not any(r["label"] == missed_label and r["id"] == missed_id for r in merged):
        for r in rrf_merge(paths, weights=weights, top_n=50):
            if r["label"] == missed_label and r["id"] == missed_id:
                print(f"    → {missed_id} at RRF rank ~{rrf_merge(paths, weights=weights, top_n=50).index(r)+1} (below top-10 cutoff)")
                break


if __name__ == "__main__":
    for case in FAILING_CASES:
        trace_case(case)
    print("\n\nDone.")
