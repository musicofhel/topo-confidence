# archive/ — superseded files kept for provenance

Nothing here is current. Everything is referenced by historical sections of [PROJECT_RECORD.md](../PROJECT_RECORD.md) §1a and §1d. Don't run anything from here without checking whether a current pathway directory has replaced it.

## design_docs/

Per-pathway plan/handoff documents from the original chronological work. PROJECT_RECORD §1a summarizes each pathway in one paragraph and supersedes these.

| File | Pathway | Status |
|---|---|---|
| `pathway1_v2.md` | P1 CORAL | Superseded by P6 rebuild label correction |
| `pathway2_v2.md`, `pathway2_v3.md` | P2 spherical steering | Refuted by P10 direction-rotation analysis |
| `pathway3_v1.md` | P3 complexity expansion | NO-GO; complexity didn't help |
| `pathway10_v1.md`, `pathway10_v2.md` | P10 v1 five-directions, v2 steering | Findings rolled into PROJECT_RECORD |
| `pathway10_handoff_v1.md` | P10 handoff | Superseded by STATE.md |
| `pathway10-papers.md` | P10 literature dump | Curated subset is in PAPER_INDEX.md |

## legacy_runners/

Pre-rebuild experiment runners and stale logs.

| File | Replaced by |
|---|---|
| `scripts/experiment*.py`, `scripts/run_*.sh` | Per-pathway pipelines (`pathway6_rebuild/`, `pathway8_layerwise/`, `pathway11_h100/`) |
| `experiments.log` (Apr 11), `experiments_remaining.log` (Apr 11) | EXPERIMENT_LOG.md |
| `runpod_launch.sh` (Apr 15) | Per-pathway `runpod_*.sh` orchestrators |
