"""Eval harness for research-graph RAG.

41 hand-written test cases across 9 categories + optional synthetic cases.

Modes:
  --mode fulltext-only    Baseline: per-type fulltext indexes only
  --mode vector+fulltext  Phase 2: 7-path retrieval + RRF merge
  --mode full             All enhancements: HyDE + concept-expand + reranking

Synthetic eval:
  --synthetic PATH        Load synthetic cases from JSON sidecar
  --validate-synthetic    Run freshness check before eval, skip stale cases
  --difficulty TIER       Filter to a specific difficulty tier

Usage:
    python eval_rag.py --mode full --out expanded
    python eval_rag.py --mode full --out synth-run --synthetic eval/synthetic-v1.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "eval"
EVAL_DIR.mkdir(exist_ok=True)

BOLT = "bolt://localhost:7688"
AUTH = ("neo4j", "topo_graph_dev")

LUCENE_RESERVED = set('+-&|!(){}[]^"~*?:\\/')


def _escape_lucene(s: str) -> str:
    out = []
    for ch in s:
        if ch in LUCENE_RESERVED:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return " ".join("".join(out).split())


# ── Test cases ──────────────────────────────────────────────────────────────

@dataclass
class EvalCase:
    id: int
    query: str
    expected: list[tuple[str, str]]  # (label, node_id)
    category: str  # finding | paper | cross-type | control
    control_cmd: str | None = None  # for control cases, the query.py subcommand


CASES: list[EvalCase] = [
    # ── Finding retrieval (6) ───────────────────────────────────────────────
    EvalCase(1, "What predicts LLM correctness from hidden states?",
             [("Finding", "F-2"), ("Finding", "F-4")], "finding"),
    EvalCase(2, "Does persistent homology add signal beyond covariance?",
             [("Finding", "F-7"), ("Finding", "F-10")], "finding"),
    EvalCase(3, "What happens to participation ratio during chain of thought?",
             [("Finding", "F-1")], "finding"),
    EvalCase(4, "How does the DoM direction change during generation?",
             [("Finding", "F-3")], "finding"),
    EvalCase(5, "What accuracy can selective prediction achieve?",
             [("Finding", "F-11")], "finding"),
    EvalCase(6, "Is output length a confound for correctness prediction?",
             [("Finding", "F-9")], "finding"),

    # ── Paper retrieval (8) ─────────────────────────────────────────────────
    EvalCase(7, "Papers about activation steering for truthfulness",
             [("Paper", "2306.03341")], "paper"),
    EvalCase(8, "Linear probes on LLM representations for truth",
             [("Paper", "2310.06824")], "paper"),
    EvalCase(9, "Chain of embedding for correctness",
             [("Paper", "2410.13640")], "paper"),
    EvalCase(10, "Papers showing LLMs encode problem difficulty",
             [("Paper", "2510.18147")], "paper"),
    EvalCase(11, "Persistent topological features in language models",
             [("Paper", "2410.11042")], "paper"),
    EvalCase(12, "Length bias in RLHF reward models",
             [("Paper", "2310.03716")], "paper"),
    EvalCase(13, "Intrinsic dimension and truthfulness",
             [("Paper", "2402.18048")], "paper"),
    EvalCase(14, "Small steering vectors with large effects",
             [("Paper", "2509.06608")], "paper"),

    # ── Cross-type semantic (6) ─────────────────────────────────────────────
    EvalCase(15, "What's the full picture on steering in this project?",
             [("Finding", "F-3"), ("Paper", "2306.03341")], "cross-type"),
    EvalCase(16, "Evidence for and against topological features",
             [("Finding", "F-7"), ("Paper", "2410.11042")], "cross-type"),
    EvalCase(17, "What corroborates the prefill direction finding?",
             [("Finding", "F-2"), ("Paper", "2509.12886")], "cross-type"),
    EvalCase(18, "Calibration and confidence in LLM predictions",
             [("Finding", "F-11"), ("Paper", "2510.18147")], "cross-type"),
    EvalCase(19, "Papers evaluated on MATH-500",
             [("Paper", "2510.18147"), ("Paper", "2601.06002"), ("Paper", "2602.02710")],
             "cross-type"),
    EvalCase(20, "Papers evaluated on TruthfulQA",
             [("Paper", "2306.03341"), ("Paper", "2312.06681"), ("Paper", "2405.20974")],
             "cross-type"),

    # ── Control cases (5) ───────────────────────────────────────────────────
    EvalCase(21, "What high-ROI experiments should we run?",
             [("Control", "highest-roi")], "control",
             control_cmd="highest-roi"),
    EvalCase(22, "What experiments are blocked?",
             [("Control", "blocked")], "control",
             control_cmd="blocked"),
    EvalCase(23, "What papers contradict F-10?",
             [("Paper", "2402.18048")], "control",
             control_cmd="contradictors F-10"),
    EvalCase(24, "FEs triggered by the ITI paper",
             [("Control", "triggered")], "control",
             control_cmd="watchlist"),
    EvalCase(25, "Full neighborhood of F-2",
             [("Control", "subgraph")], "control",
             control_cmd="subgraph F-2"),

    # ── Paraphrase (4) — vocabulary robustness ─────────────────────────────
    EvalCase(26, "How does representation dimensionality change during step-by-step reasoning?",
             [("Finding", "F-1")], "paraphrase"),
    EvalCase(27, "How well does the confident-subset prediction strategy perform?",
             [("Finding", "F-11")], "paraphrase"),
    EvalCase(28, "Which paper uses activation editing to improve model honesty?",
             [("Paper", "2306.03341")], "paraphrase"),
    EvalCase(29, "Papers showing output verbosity biases reward model scores",
             [("Paper", "2310.03716")], "paraphrase"),

    # ── Dataset-path (2) — dataset-match retrieval ─────────────────────────
    EvalCase(30, "Chain-of-embedding method evaluated on GSM8K",
             [("Paper", "2410.13640")], "dataset"),
    EvalCase(31, "Semantic entropy probes evaluated on TriviaQA",
             [("Paper", "2406.15927")], "dataset"),

    # ── Tag-path (2) — tag-match retrieval ─────────────────────────────────
    EvalCase(32, "Sparse autoencoder features for LLM uncertainty",
             [("Paper", "2604.19974")], "tag"),
    EvalCase(33, "Linear representation hypothesis in neural networks",
             [("Paper", "2311.03658")], "tag"),

    # ── Cross-type / neighborhood (4) ──────────────────────────────────────
    EvalCase(34, "CoE correctness prediction compared to semantic entropy probes",
             [("Finding", "F-6"), ("Paper", "2406.15927")], "cross-type"),
    EvalCase(35, "What work extends the DoM rotation finding?",
             [("Finding", "F-3"), ("Paper", "2510.04309")], "cross-type"),
    EvalCase(36, "What explains representation dimensionality changes during generation?",
             [("Finding", "F-1"), ("Paper", "2302.00294")], "cross-type"),
    EvalCase(37, "What corroborates the output length confound finding?",
             [("Finding", "F-9"), ("Paper", "2505.00127")], "cross-type"),

    # ── Broad / ambiguous (2) ──────────────────────────────────────────────
    EvalCase(38, "How can a model recognize when it will get the answer wrong?",
             [("Finding", "F-2"), ("Finding", "F-11")], "broad"),
    EvalCase(39, "Methods for probing truth in LLM representations",
             [("Paper", "2310.06824"), ("Paper", "2311.03658")], "broad"),

    # ── Long-tail (2) ──────────────────────────────────────────────────────
    EvalCase(40, "Underthinking and overthinking in LLM reasoning",
             [("Paper", "2505.00127")], "long-tail"),
    EvalCase(41, "Semantic entropy probes for uncertainty estimation",
             [("Paper", "2406.15927")], "long-tail"),
]


# ── Retrieval backends ──────────────────────────────────────────────────────

def _get_driver():
    return GraphDatabase.driver(BOLT, auth=AUTH, notifications_min_severity="OFF")


def retrieve_fulltext_only(
    query: str, case: EvalCase
) -> list[tuple[str, str, int]]:
    """Fulltext-only retrieval. Returns [(label, id, rank), ...]."""
    safe = _escape_lucene(query)
    results: list[tuple[str, str, float]] = []

    with _get_driver() as drv, drv.session() as s:
        # Finding fulltext
        rows = s.run(
            """
            CALL db.index.fulltext.queryNodes('finding_claims', $q)
            YIELD node, score
            RETURN 'Finding' AS label, node.id AS id, score
            ORDER BY score DESC LIMIT 10
            """,
            q=safe,
        )
        for r in rows:
            results.append((r["label"], r["id"], r["score"]))

        # Paper fulltext
        rows = s.run(
            """
            CALL db.index.fulltext.queryNodes('paper_relevance', $q)
            YIELD node, score
            RETURN 'Paper' AS label, node.arxiv_id AS id, score
            ORDER BY score DESC LIMIT 20
            """,
            q=safe,
        )
        for r in rows:
            results.append((r["label"], r["id"], r["score"]))

    # Sort by score descending, assign ranks
    results.sort(key=lambda x: -x[2])
    ranked = [(label, nid, rank) for rank, (label, nid, _) in enumerate(results)]
    return ranked[:10]


def retrieve_vector_fulltext(
    query: str, case: EvalCase,
    hyde: bool = False, expand: bool = False,
    rerank: bool = False, no_cache: bool = False,
) -> list[tuple[str, str, int]]:
    """7-path retrieval via query.py semantic_search."""
    sys.path.insert(0, str(ROOT))
    from query import semantic_search
    results = semantic_search(
        query, top_n=10,
        hyde=hyde, concept_expand=expand,
        rerank=rerank, no_cache=no_cache,
    )
    return [(r["label"], r["id"], i) for i, r in enumerate(results)]


def retrieve_control(case: EvalCase) -> list[tuple[str, str, int]]:
    """Control cases: run the existing query.py subcommand and check it works."""
    cmd = case.control_cmd
    if not cmd:
        return []

    if cmd == "contradictors F-10":
        with _get_driver() as drv, drv.session() as s:
            rows = s.run(
                """
                MATCH (f:Finding {id: 'F-10'})-[:CONTRADICTED_BY]->(p:Paper)
                RETURN 'Paper' AS label, p.arxiv_id AS id
                """,
            )
            return [(r["label"], r["id"], i) for i, r in enumerate(rows)]

    # For structural commands, just verify they execute successfully
    result = subprocess.run(
        [sys.executable, str(ROOT / "query.py")] + cmd.split(),
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0:
        return [("Control", cmd.split()[0], 0)]
    return []


# ── Metrics ─────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: int
    query: str
    category: str
    expected: list[tuple[str, str]]
    retrieved: list[tuple[str, str, int]]
    hits: list[tuple[str, str]]
    misses: list[tuple[str, str]]
    precision: float
    mrr: float
    difficulty: str = ""


def evaluate_case(
    case: EvalCase,
    retriever: str,
    **kwargs,
) -> CaseResult:
    """Run one eval case and compute metrics."""
    hyde = "hyde" in retriever or "full" in retriever
    expand = "expand" in retriever or "full" in retriever
    rerank = "rerank" in retriever or "full" in retriever
    no_cache = kwargs.get("no_cache", False)

    if case.category == "control":
        retrieved = retrieve_control(case)
    elif retriever == "fulltext-only":
        retrieved = retrieve_fulltext_only(case.query, case)
    elif retriever.startswith("vector"):
        retrieved = retrieve_vector_fulltext(case.query, case)
    elif retriever == "full":
        retrieved = retrieve_vector_fulltext(
            case.query, case,
            hyde=True, expand=True, rerank=True,
            no_cache=no_cache,
        )
    else:
        raise ValueError(f"Unknown retriever: {retriever}")

    retrieved_set = {(label, nid) for label, nid, _ in retrieved}

    hits = []
    misses = []
    for label, nid in case.expected:
        if label == "Control":
            # Control cases: check the command ran (result contains something)
            if retrieved:
                hits.append((label, nid))
            else:
                misses.append((label, nid))
        elif (label, nid) in retrieved_set:
            hits.append((label, nid))
        else:
            misses.append((label, nid))

    precision = len(hits) / len(case.expected) if case.expected else 1.0

    # MRR: 1 / (rank_of_first_hit + 1)
    mrr = 0.0
    for label, nid in case.expected:
        for rl, rid, rank in retrieved:
            if rl == label and rid == nid:
                candidate = 1.0 / (rank + 1)
                mrr = max(mrr, candidate)
                break

    return CaseResult(
        case_id=case.id,
        query=case.query,
        category=case.category,
        expected=case.expected,
        retrieved=[(l, i, r) for l, i, r in retrieved],
        hits=hits,
        misses=misses,
        precision=precision,
        mrr=mrr,
    )


# ── Report ──────────────────────────────────────────────────────────────────

@dataclass
class EvalReport:
    mode: str
    timestamp: str
    cases: list[CaseResult]
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    overall_precision: float = 0.0
    overall_mrr: float = 0.0

    def compute_aggregates(self):
        cats: dict[str, list[CaseResult]] = {}
        for c in self.cases:
            cats.setdefault(c.category, []).append(c)

        for cat, results in sorted(cats.items()):
            avg_prec = sum(r.precision for r in results) / len(results)
            avg_mrr = sum(r.mrr for r in results) / len(results)
            self.by_category[cat] = {
                "precision": round(avg_prec, 3),
                "mrr": round(avg_mrr, 3),
                "count": len(results),
            }

        all_prec = sum(r.precision for r in self.cases) / len(self.cases)
        all_mrr = sum(r.mrr for r in self.cases) / len(self.cases)
        self.overall_precision = round(all_prec, 3)
        self.overall_mrr = round(all_mrr, 3)


def print_report(report: EvalReport, synthetic_report: EvalReport | None = None):
    print(f"\n{'='*70}")
    print(f"  Research-Graph RAG Eval — mode: {report.mode}")
    print(f"  {report.timestamp}")
    print(f"{'='*70}\n")

    for c in report.cases:
        status = "PASS" if c.precision == 1.0 else "PARTIAL" if c.precision > 0 else "FAIL"
        print(f"  [{c.case_id:2d}] [{status:7s}] {c.query[:60]}")
        print(f"       precision: {c.precision:.3f}  MRR: {c.mrr:.3f}")
        if c.misses:
            miss_str = ", ".join(f"{l}:{i}" for l, i in c.misses)
            print(f"       MISSING: {miss_str}")
        if c.hits:
            hit_str = ", ".join(f"{l}:{i}" for l, i in c.hits)
            print(f"       HIT: {hit_str}")
        print()

    print(f"  {'Category':<14} {'Precision':>10} {'MRR':>10} {'Cases':>7}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*7}")
    for cat, stats in sorted(report.by_category.items()):
        print(f"  {cat:<14} {stats['precision']:>10.3f} {stats['mrr']:>10.3f} {stats['count']:>7}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*7}")
    print(f"  {'GOLDEN':<14} {report.overall_precision:>10.3f} {report.overall_mrr:>10.3f} {len(report.cases):>7}")
    print()

    if synthetic_report and synthetic_report.cases:
        print(f"  {'--- Synthetic ---':^46}")
        print()
        for c in synthetic_report.cases:
            status = "PASS" if c.precision == 1.0 else "PARTIAL" if c.precision > 0 else "FAIL"
            print(f"  [{c.case_id:4d}] [{status:7s}] {c.query[:56]}")
            print(f"         precision: {c.precision:.3f}  MRR: {c.mrr:.3f}")

        # Per-difficulty
        diff_buckets: dict[str, list[CaseResult]] = {}
        for c in synthetic_report.cases:
            tier = getattr(c, "difficulty", "unknown")
            diff_buckets.setdefault(tier, []).append(c)

        print()
        print(f"  {'Difficulty':<22} {'Precision':>10} {'MRR':>10} {'Cases':>7}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*7}")
        for tier, cases in sorted(diff_buckets.items()):
            avg_p = sum(c.precision for c in cases) / len(cases)
            avg_m = sum(c.mrr for c in cases) / len(cases)
            print(f"  {tier:<22} {avg_p:>10.3f} {avg_m:>10.3f} {len(cases):>7}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*7}")
        print(f"  {'SYNTHETIC OVERALL':<22} {synthetic_report.overall_precision:>10.3f} {synthetic_report.overall_mrr:>10.3f} {len(synthetic_report.cases):>7}")
        print()


def save_report(report: EvalReport, name: str):
    path = EVAL_DIR / f"{name}.json"

    def serialize(obj):
        if isinstance(obj, CaseResult):
            d = asdict(obj)
            d["expected"] = [list(t) for t in obj.expected]
            d["retrieved"] = [list(t) for t in obj.retrieved]
            d["hits"] = [list(t) for t in obj.hits]
            d["misses"] = [list(t) for t in obj.misses]
            return d
        if isinstance(obj, EvalReport):
            return {
                "mode": obj.mode,
                "timestamp": obj.timestamp,
                "overall_precision": obj.overall_precision,
                "overall_mrr": obj.overall_mrr,
                "by_category": obj.by_category,
                "cases": [serialize(c) for c in obj.cases],
            }
        return obj

    with open(path, "w") as f:
        json.dump(serialize(report), f, indent=2)
    print(f"  Saved: {path}")


# ── Synthetic case loading ─────────────────────────────────────────────────

def _load_synthetic_cases(
    path: Path,
    validate_freshness: bool = False,
    difficulty_filter: str | None = None,
) -> list[EvalCase]:
    """Load synthetic cases from JSON, optionally filtering by freshness/difficulty."""
    with open(path) as f:
        data = json.load(f)

    cases_data = data.get("cases", [])
    fresh_ids: set[int] | None = None

    if validate_freshness:
        try:
            from ragas_gen.freshness import check_freshness
            from ragas_gen.schemas import SyntheticEvalSet
            eval_set = SyntheticEvalSet(**data)
            results = check_freshness(eval_set)
            fresh_ids = {r.case_id for r in results if r.status == "FRESH"}
            stale = [r for r in results if r.status != "FRESH"]
            if stale:
                print(f"  Freshness: {len(fresh_ids)} fresh, {len(stale)} stale/missing (skipped)")
                for r in stale:
                    print(f"    [{r.status:7s}] case {r.case_id}: {r.details}")
        except Exception as e:
            print(f"  Warning: freshness check failed: {e}")

    eval_cases = []
    for c in cases_data:
        case_id = c["id"]
        if fresh_ids is not None and case_id not in fresh_ids:
            continue

        difficulty = c.get("difficulty", "")
        if difficulty_filter and difficulty != difficulty_filter:
            continue

        expected = [(e[0], e[1]) for e in c["expected"]]
        eval_cases.append(EvalCase(
            id=case_id,
            query=c["query"],
            expected=expected,
            category=c.get("category", "synthetic"),
        ))
    return eval_cases


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Research-graph RAG eval harness")
    parser.add_argument("--mode", required=True,
                        choices=["fulltext-only", "vector+fulltext", "full"],
                        help="Retrieval mode to evaluate")
    parser.add_argument("--out", required=True,
                        help="Output filename (without .json)")
    parser.add_argument("--cases", type=str, default=None,
                        help="Comma-separated case IDs to run (default: all)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Skip LLM cache reads and writes")
    parser.add_argument("--synthetic", type=str, default=None,
                        help="Path to synthetic eval set JSON")
    parser.add_argument("--validate-synthetic", action="store_true",
                        help="Run freshness check before eval, skip stale cases")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=["single-hop-exact", "single-hop-paraphrase", "multi-hop", "adversarial"],
                        help="Filter synthetic cases to a difficulty tier")
    args = parser.parse_args()

    case_ids = None
    if args.cases:
        case_ids = {int(x) for x in args.cases.split(",")}

    # Golden cases
    cases_to_run = [c for c in CASES if case_ids is None or c.id in case_ids]
    print(f"\n  Running {len(cases_to_run)} golden cases in mode: {args.mode}\n")

    results = []
    t0 = time.time()
    for case in cases_to_run:
        t_case = time.time()
        result = evaluate_case(case, args.mode, no_cache=args.no_cache)
        elapsed = time.time() - t_case
        status = "PASS" if result.precision == 1.0 else "FAIL"
        print(f"  [{case.id:2d}] {status} ({elapsed:.1f}s) {case.query[:55]}")
        results.append(result)

    # Synthetic cases
    synthetic_results = []
    if args.synthetic:
        synth_path = Path(args.synthetic)
        if not synth_path.is_absolute():
            synth_path = ROOT / synth_path
        if synth_path.exists():
            synth_cases = _load_synthetic_cases(
                synth_path,
                validate_freshness=args.validate_synthetic,
                difficulty_filter=args.difficulty,
            )
            print(f"\n  Running {len(synth_cases)} synthetic cases...\n")

            # Load difficulty info for reporting
            with open(synth_path) as f:
                synth_data = json.load(f)
            diff_by_id = {c["id"]: c.get("difficulty", "") for c in synth_data.get("cases", [])}

            for case in synth_cases:
                t_case = time.time()
                result = evaluate_case(case, args.mode, no_cache=args.no_cache)
                result.difficulty = diff_by_id.get(case.id, "")
                elapsed = time.time() - t_case
                status = "PASS" if result.precision == 1.0 else "FAIL"
                print(f"  [{case.id:4d}] {status} ({elapsed:.1f}s) {case.query[:51]}")
                synthetic_results.append(result)
        else:
            print(f"  Warning: synthetic file not found: {synth_path}")

    elapsed_total = time.time() - t0

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    report = EvalReport(mode=args.mode, timestamp=ts, cases=results)
    report.compute_aggregates()

    synth_report = None
    if synthetic_results:
        synth_report = EvalReport(mode=args.mode, timestamp=ts, cases=synthetic_results)
        synth_report.compute_aggregates()

    print(f"\n  Total time: {elapsed_total:.1f}s")
    print_report(report, synth_report)
    save_report(report, args.out)
    if synth_report:
        save_report(synth_report, f"{args.out}-synthetic")


if __name__ == "__main__":
    main()
