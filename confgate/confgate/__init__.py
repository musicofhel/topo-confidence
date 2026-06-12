"""confgate — the free confidence gate for LLM correctness.

Ships the pinned SPEC-v6 result of the topo-confidence project: a logistic
regression on two zero-cost scalars of a greedy generation —
(n_gen_tokens, mean_logprob) — is the most GENERALIZING correctness readout
tested (per-family deployment AUROC 0.81-0.89; held-out cross-architecture
OOF AUROC SmolLM2 0.810 / Gemma-2-2b 0.844 / OLMo-2-1B 0.838).

Components:
  gate      — FreeGate: fit/score, pinned per-family coefficients (5 families)
  route     — Cascade: keep-on-small / escalate-to-large routing + cost frontier
  certify   — split-conformal LTT certificates (cross-scale zero-shot only;
              cross-domain certificates are a documented product limit)
  preflight — prompt-length pre-generation signal (Tier-A, AUROC ~0.71)
  cli       — `confgate demo`, `confgate score`
"""
from __future__ import annotations

from .gate import FreeGate, pinned_families, pinned_meta
from .route import Cascade, KEEP_COST, ESCALATE_COST
from .certify import certify, certify_cross_domain, cp_upper, choose_tau
from .preflight import PreflightGate

__version__ = "0.1.0"

__all__ = [
    "FreeGate", "pinned_families", "pinned_meta",
    "Cascade", "KEEP_COST", "ESCALATE_COST",
    "certify", "certify_cross_domain", "cp_upper", "choose_tau",
    "PreflightGate",
    "__version__",
]
