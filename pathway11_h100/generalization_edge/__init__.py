"""Generalization-First Edge Program — SPEC v5 harness.

Shared infrastructure for the transfer bake-off (Phase 1) and the applied
quick-wins (Phase 1B). All compute is LOCAL/CPU on cached activations.

Builds on the verified nocompute/lib.py substrate (DomProbe, oof_dom_scores,
DeLong, bootstrap, ECE, risk-coverage, AURC) rather than re-implementing it.
"""
