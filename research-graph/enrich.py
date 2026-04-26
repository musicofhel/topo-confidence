"""Pull title + forgeScore from link-forge for every Paper stub.

Thin wrapper over `bridge.enrich_all_papers()`. Safe to re-run.
"""
from __future__ import annotations

import argparse

from bridge import enrich_all_papers


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    summary = enrich_all_papers(verbose=not args.quiet)
    print(f"\nEnriched: {summary['enriched']} / Missing: {summary['missing']}")
    if summary["missing_ids"]:
        print("Missing arxiv IDs:")
        for a in summary["missing_ids"]:
            print(f"  - {a}")


if __name__ == "__main__":
    main()
