"""
validate_claims.py — ground-truth every quantitative claim in PROJECT_RECORD.md
and RESEARCH_SUMMARY.md against the source JSONs on disk.

Each Claim entry carries:
  - a short ID and description (mirrors PROJECT_RECORD §1b)
  - source file path (relative to repo root)
  - jq-style key path into the JSON
  - expected value as stated in the narrative docs
  - tolerance (abs) for float comparison
  - label_scheme tag — 256tok_NEW / 256tok_manifest / 1024tok — so we can
    flag anything computed against the superseded truncation-artifact labels.

Run:  python validate_claims.py > validation_report.txt
"""
from __future__ import annotations

import json
import math
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def _drill(d: Any, path: list[str | int]) -> Any:
    """Walk a nested dict/list by a list of keys/indices; raise KeyError on miss."""
    cur = d
    for p in path:
        if isinstance(cur, dict):
            if p not in cur:
                raise KeyError(f"missing key {p!r} in {list(cur.keys())[:8]}...")
            cur = cur[p]
        elif isinstance(cur, list):
            if not isinstance(p, int) or p >= len(cur):
                raise KeyError(f"index {p} out of range (list len={len(cur)})")
            cur = cur[p]
        else:
            raise KeyError(f"cannot descend into {type(cur).__name__} at key {p!r}")
    return cur


@dataclass
class Claim:
    cid: str
    description: str
    file: str
    path: list
    expected: Any
    tol: float = 0.005
    labels: str = "1024tok"  # 256tok_NEW | 256tok_manifest | 1024tok | mixed | external_anchor | forward_looking
    note: str = ""
    # `kind` controls the readback semantics:
    #   "internal"        — back-check `file[path]` against `expected` (default)
    #   "external"        — paper-cited number, no project-side readback; kept on
    #                       the books so PAPER_INDEX cross-references stay
    #                       greppable. Result = "REGISTERED".
    #   "forward_looking" — project-side prediction (a hypothesis acceptance
    #                       threshold) that becomes a real readback target only
    #                       when the FE produces a result JSON. Result =
    #                       "PENDING_FE".
    kind: str = "internal"
    # --- Tier-1 regen (optional) -----------------------------------------
    # `regen` is a shell command run from ROOT that recomputes this number
    # from a cached intermediate (NPZ / parquet) and prints `key=value` lines
    # on stdout. `regen_key` selects the line to compare against `expected`.
    # When the script exits with "MISSING_REGEN_INPUT", the regen is treated
    # as advisory (regen_result=SKIP_NO_DATA) — JSON-vs-doc Tier-0 still runs.
    regen: str = ""
    regen_key: str = ""
    regen_tol: float = 0.0  # 0 ⇒ reuse `tol`
    result: str = field(default="", init=False)
    actual: Any = field(default=None, init=False)
    regen_result: str = field(default="", init=False)
    regen_actual: Any = field(default=None, init=False)


# --- CLAIMS ------------------------------------------------------------------
# Grouped roughly in the order of sections in RESEARCH_SUMMARY.md.

_BASELINES_REGEN = "python pathway11_h100/prefill_gated_compute/recompute_baselines.py"
_SELECTIVE_REGEN = "python pathway11_h100/prefill_gated_compute/recompute_selective.py"
_PR_RATIOS_REGEN = "python pathway11_h100/prefill_inversion/recompute_pr_ratios.py"
_BBH_PR_REGEN = "python pathway11_h100/prefill_inversion/recompute_bbh_pr_ratios.py"
_CLASSIFIER_REGEN = "python pathway11_h100/multi_signal_oracle/recompute_classifier.py"
_EXP2B_EXTRAS_REGEN = "python pathway11_h100/multi_signal_oracle/recompute_extras.py"
_GIB_PR_REGEN = "python pathway11_h100/gibberish_control/recompute_pr.py"
_NOCOT_PR_REGEN = "python pathway11_h100/no_cot_control/recompute_pr.py"
_CROSS_MODEL_REGEN = "python pathway11_h100/exp1_cross_model/recompute_acc.py"
_STAGE5_REGEN = "python pathway11_h100/recompute_stage5_bbh_transfer.py"

CLAIMS: list[Claim] = [
    # ----- Section 1 / §1: baselines -----
    # Tier-1: recompute_baselines.py replays K=1 greedy + K∈{2,4,8} majority over
    # phase1_majority_vote.npz, plus A/B/C/D buckets and oracle compute.
    Claim("baseline-1.5B-acc", "1.5B MATH-500 K=1 accuracy 0.486",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["accuracy_at_uniform_K", "K=1 (greedy T=0)"], 0.486, 0.002, "1024tok",
          regen=_BASELINES_REGEN, regen_key="baseline_K1_greedy"),
    Claim("baseline-K2", "1.5B K=2 majority accuracy 0.413",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["accuracy_at_uniform_K", "K=2 majority (T=0.7)"], 0.4132, 0.005, "1024tok",
          regen=_BASELINES_REGEN, regen_key="baseline_K2_maj"),
    Claim("baseline-K4", "1.5B K=4 majority 0.495",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["accuracy_at_uniform_K", "K=4 majority (T=0.7)"], 0.495, 0.003, "1024tok",
          regen=_BASELINES_REGEN, regen_key="baseline_K4_maj"),
    Claim("baseline-K8", "1.5B K=8 majority 0.550",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["accuracy_at_uniform_K", "K=8 majority (T=0.7)"], 0.55, 0.005, "1024tok",
          regen=_BASELINES_REGEN, regen_key="baseline_K8_maj"),
    Claim("oracle-K", "Oracle compute/problem 1.01",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["oracle_compute_per_problem"], 1.01, 0.02, "1024tok",
          regen=_BASELINES_REGEN, regen_key="oracle_K"),
    Claim("oracle-acc", "Oracle accuracy 0.589",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["oracle_overall_accuracy"], 0.589, 0.005, "1024tok",
          regen=_BASELINES_REGEN, regen_key="oracle_acc"),

    # ----- Bucket sizes (used across §7, §9, §10) -----
    Claim("buckets-A", "Bucket A always-right n=207",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["bucket_sizes", "A_always_right"], 207, 0, "1024tok",
          regen=_BASELINES_REGEN, regen_key="bucket_A"),
    Claim("buckets-B", "Bucket B recoverable n=68",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["bucket_sizes", "B_recoverable"], 68, 0, "1024tok",
          regen=_BASELINES_REGEN, regen_key="bucket_B"),
    Claim("buckets-C", "Bucket C never-right n=189",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["bucket_sizes", "C_never_right"], 189, 0, "1024tok",
          regen=_BASELINES_REGEN, regen_key="bucket_C"),
    Claim("buckets-D", "Bucket D pathological n=36",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["bucket_sizes", "D_pathological"], 36, 0, "1024tok",
          regen=_BASELINES_REGEN, regen_key="bucket_D"),

    # ----- §2 / §4 DoM signal -----
    # Tier-1 regen: recompute_dom_auroc.py replays roc_auc_score over the
    # OOF DoM scores cached in phase2_prefill_dom.npz. --mode=full re-derives
    # scores from per-problem NPZs, but is gated on the data dir existing.
    Claim("prefill-dom-auroc", "Prefill L19 DoM AUROC (OOF 5-fold) = 0.7731",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["prefill_DoM_auroc_oof"], 0.7731, 0.002, "1024tok",
          regen="python pathway11_h100/prefill_gated_compute/recompute_dom_auroc.py --key=prefill_oof",
          regen_key="prefill_oof"),
    Claim("finaltok-dom-auroc", "Final-token L19 DoM AUROC = 0.7186",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["final_token_DoM_auroc_oof"], 0.7186, 0.002, "1024tok",
          regen="python pathway11_h100/prefill_gated_compute/recompute_dom_auroc.py --key=final_oof",
          regen_key="final_oof"),

    # ----- FE719: Leave-One-Subject-Out CV on F-2 -----
    # Refutation test from 2603.18280: random-split CV is non-diagnostic; LOCO is.
    # Mean LOCO AUROC < 0.65 would refute F-2 as a correctness signal.
    Claim("loco-auroc-mean", "FE719 LOCO mean AUROC (1.5B prefill L19 DoM, 7 MATH-500 subjects) = 0.7427",
          "pathway11_h100/loco_subject/results.json",
          ["auroc_mean"], 0.7427, 0.002, "1024tok",
          regen="python pathway11_h100/loco_subject/recompute_fe719.py",
          regen_key="loco_auroc_mean"),
    Claim("loco-auroc-worst", "FE719 LOCO worst-case AUROC = 0.6032 (Number Theory)",
          "pathway11_h100/loco_subject/results.json",
          ["auroc_worst"], 0.6032, 0.002, "1024tok",
          regen="python pathway11_h100/loco_subject/recompute_fe719.py",
          regen_key="loco_auroc_worst"),
    Claim("loco-n-folds", "FE719 LOCO folds scored = 7",
          "pathway11_h100/loco_subject/results.json",
          ["n_folds_scored"], 7, 0, "1024tok",
          regen="python pathway11_h100/loco_subject/recompute_fe719.py",
          regen_key="loco_n_folds"),

    # ----- FE448: Length partial-correlation control on F-2 -----
    # Refutation: residual AUROC after partialing out predicted-length.
    # Threshold ≤0.55 ⇒ length artifact; >0.65 ⇒ survives partialing.
    Claim("fe448-length-r2", "FE448 length R² from L19 prefill (in-sample, ridge α=1e-3)",
          "pathway11_h100/length_partial/results.json",
          ["length_r2_from_prefill"], 1.0, 0.0001, "1024tok",
          regen="python pathway11_h100/length_partial/recompute_fe448.py",
          regen_key="length_r2_from_prefill"),
    Claim("fe448-auroc-resid", "FE448 DoM AUROC after partialing predicted-length = 0.6647",
          "pathway11_h100/length_partial/results.json",
          ["auroc_residualized"], 0.6647, 0.002, "1024tok",
          regen="python pathway11_h100/length_partial/recompute_fe448.py",
          regen_key="auroc_residualized"),
    Claim("fe448-auroc-length-alone", "FE448 seq_len-alone AUROC = 0.7986 (length is stronger than raw DoM 0.7731)",
          "pathway11_h100/length_partial/results.json",
          ["auroc_length_alone"], 0.7986, 0.002, "1024tok",
          regen="python pathway11_h100/length_partial/recompute_fe448.py",
          regen_key="auroc_length_alone"),
    Claim("fe448-auroc-drop", "FE448 AUROC drop from raw to residualized = 0.1187 (in-sample)",
          "pathway11_h100/length_partial/results.json",
          ["auroc_drop"], 0.1187, 0.002, "1024tok",
          regen="python pathway11_h100/length_partial/recompute_fe448.py",
          regen_key="auroc_drop"),

    # ----- FE145: Per-layer DoM AUROC sweep -----
    # Validates F-2's L19 choice. L19 must be within ±0.005 of peak.
    Claim("fe145-peak-layer", "FE145 peak DoM AUROC layer = 21",
          "pathway11_h100/per_layer_sweep/results.json",
          ["peak_layer"], 21, 0, "1024tok",
          regen="python pathway11_h100/per_layer_sweep/recompute_fe145.py",
          regen_key="peak_layer"),
    Claim("fe145-peak-auroc", "FE145 peak DoM AUROC = 0.7718 (L21)",
          "pathway11_h100/per_layer_sweep/results.json",
          ["peak_auroc"], 0.7718, 0.002, "1024tok",
          regen="python pathway11_h100/per_layer_sweep/recompute_fe145.py",
          regen_key="peak_auroc"),
    Claim("fe145-l19-auroc", "FE145 L19 DoM AUROC = 0.7705 (this script's 5-fold OOF, seed=9999)",
          "pathway11_h100/per_layer_sweep/results.json",
          ["auroc_l19"], 0.7705, 0.002, "1024tok",
          regen="python pathway11_h100/per_layer_sweep/recompute_fe145.py",
          regen_key="auroc_l19"),
    Claim("fe145-l19-within-band", "FE145 L19 within ±0.005 of peak: gap 0.0013",
          "pathway11_h100/per_layer_sweep/results.json",
          ["peak_minus_l19"], 0.0013, 0.001, "1024tok",
          regen="python pathway11_h100/per_layer_sweep/recompute_fe145.py",
          regen_key="peak_minus_l19"),
    Claim("fe145-l17-auroc", "FE145 L17 DoM AUROC = 0.7635 (CAA drop-off layer; not a sharp drop on Qwen)",
          "pathway11_h100/per_layer_sweep/results.json",
          ["auroc_l17"], 0.7635, 0.002, "1024tok",
          regen="python pathway11_h100/per_layer_sweep/recompute_fe145.py",
          regen_key="auroc_l17"),

    # ----- FE299: Within-topic prefill DoM AUROC -----
    # Sister to FE719: tests whether F-2 is a topic-detector. Within-topic
    # mean AUROC ≪ overall AUROC ⇒ F-2 is mostly topic-base-rate.
    Claim("fe299-within-mean", "FE299 within-topic mean AUROC = 0.7143 (vs 0.7731 overall)",
          "pathway11_h100/within_topic/results.json",
          ["within_topic_auroc_mean"], 0.7143, 0.002, "1024tok",
          regen="python pathway11_h100/within_topic/recompute_fe299.py",
          regen_key="within_topic_auroc_mean"),
    Claim("fe299-within-worst", "FE299 within-topic worst AUROC = 0.6000 (Number Theory, k-fold within-subject)",
          "pathway11_h100/within_topic/results.json",
          ["within_topic_auroc_worst"], 0.6000, 0.002, "1024tok",
          regen="python pathway11_h100/within_topic/recompute_fe299.py",
          regen_key="within_topic_auroc_worst"),
    Claim("fe299-n-scored", "FE299 topics scored = 7",
          "pathway11_h100/within_topic/results.json",
          ["n_topics_scored"], 7, 0, "1024tok",
          regen="python pathway11_h100/within_topic/recompute_fe299.py",
          regen_key="within_topic_n_scored"),

    # ----- FE101: LEACE linear-erasure null for F-2 -----
    # Strict-null test: per-fold LEACE on train, apply to test, refit DoM
    # on erased train, score erased test. AUROC ≈ 0.5 confirms F-2 is linear.
    Claim("fe101-auroc-raw", "FE101 raw DoM AUROC (5-fold OOF, sanity anchor) = 0.7705",
          "pathway11_h100/leace_erasure/results.json",
          ["auroc_raw_oof"], 0.7705, 0.002, "1024tok",
          regen="python pathway11_h100/leace_erasure/recompute_fe101.py",
          regen_key="auroc_raw_oof"),
    Claim("fe101-auroc-erased", "FE101 LEACE-erased DoM AUROC = 0.5000 (perfect linear collapse)",
          "pathway11_h100/leace_erasure/results.json",
          ["auroc_erased_oof"], 0.5000, 0.002, "1024tok",
          regen="python pathway11_h100/leace_erasure/recompute_fe101.py",
          regen_key="auroc_erased_oof"),
    Claim("fe101-collapse-pp", "FE101 LEACE AUROC collapse = 0.2705 (raw − erased, 5-fold OOF)",
          "pathway11_h100/leace_erasure/results.json",
          ["auroc_collapse_pp"], 0.2705, 0.002, "1024tok",
          regen="python pathway11_h100/leace_erasure/recompute_fe101.py",
          regen_key="auroc_collapse_pp"),

    # ----- FE115: Song-Zhong pos/ctx decomposition for F-3 -----
    # Direct refutation test: if F-3's prefill/final orthogonality is
    # positional, cos jumps after subtracting μ + pos_t + ctx_i. It does
    # NOT — cos drops from -0.062 to 0.0008 (F-3 reinforced).
    Claim("fe115-cos-raw", "FE115 raw cos(prefill_DoM, final_DoM) on pathway8 L19 cache = -0.0617",
          "pathway11_h100/song_zhong/results.json",
          ["cos_prefill_final_raw"], -0.0617, 0.005, "1024tok",
          regen="python pathway11_h100/song_zhong/recompute_fe115.py",
          regen_key="cos_prefill_final_raw"),
    Claim("fe115-cos-resid", "FE115 Song-Zhong residualized cos(prefill_DoM, final_DoM) = 0.0008 (≈0; F-3 reinforced)",
          "pathway11_h100/song_zhong/results.json",
          ["cos_prefill_final_resid"], 0.0008, 0.003, "1024tok",
          regen="python pathway11_h100/song_zhong/recompute_fe115.py",
          regen_key="cos_prefill_final_resid"),
    Claim("fe115-auroc-prefill-resid", "FE115 prefill DoM AUROC on residuals = 0.8016 (+3pp vs raw 0.7705)",
          "pathway11_h100/song_zhong/results.json",
          ["auroc_prefill_resid"], 0.8016, 0.003, "1024tok",
          regen="python pathway11_h100/song_zhong/recompute_fe115.py",
          regen_key="auroc_prefill_resid"),
    Claim("fe115-auroc-final-resid", "FE115 final-token DoM AUROC on residuals = 0.6865",
          "pathway11_h100/song_zhong/results.json",
          ["auroc_final_resid"], 0.6865, 0.003, "1024tok",
          regen="python pathway11_h100/song_zhong/recompute_fe115.py",
          regen_key="auroc_final_resid"),

    # ----- FE181: token-prob baseline vs F-2 DoM -----
    Claim("fe181-mean-logp-15b", "FE181 mean log-prob AUROC (1.5B) = 0.6721 (< F-2 DoM 0.7731 → does not subsume)",
          "pathway11_h100/token_prob/results.json",
          ["auroc_mean_logp_15b"], 0.6721, 0.003, "1024tok",
          regen="python pathway11_h100/token_prob/recompute_fe181.py",
          regen_key="auroc_mean_logp_15b"),
    Claim("fe181-sum-logp-15b", "FE181 sum log-prob AUROC (1.5B) = 0.8478 (length-confounded, not a clean baseline)",
          "pathway11_h100/token_prob/results.json",
          ["auroc_sum_logp_15b"], 0.8478, 0.003, "1024tok",
          regen="python pathway11_h100/token_prob/recompute_fe181.py",
          regen_key="auroc_sum_logp_15b"),
    Claim("fe181-joint-meanlogp-dom-15b", "FE181 joint [mean_logp, prefill_DoM_proj] AUROC (1.5B) = 0.7836 (+1pp over DoM alone)",
          "pathway11_h100/token_prob/results.json",
          ["joint_meanlogp_dom_auroc_15b"], 0.7836, 0.003, "1024tok",
          regen="python pathway11_h100/token_prob/recompute_fe181.py",
          regen_key="joint_meanlogp_dom_auroc_15b"),
    Claim("fe181-mean-logp-7b", "FE181 mean log-prob AUROC (7B) = 0.6336 (weaker than 1.5B; DoM not subsumed)",
          "pathway11_h100/token_prob/results.json",
          ["auroc_mean_logp_7b"], 0.6336, 0.003, "1024tok",
          regen="python pathway11_h100/token_prob/recompute_fe181.py",
          regen_key="auroc_mean_logp_7b"),
    Claim("fe181-sum-logp-7b", "FE181 sum log-prob AUROC (7B) = 0.8672 (length-confounded)",
          "pathway11_h100/token_prob/results.json",
          ["auroc_sum_logp_7b"], 0.8672, 0.003, "1024tok",
          regen="python pathway11_h100/token_prob/recompute_fe181.py",
          regen_key="auroc_sum_logp_7b"),

    # ----- FE749: spectral α (HT-SR) head-to-head with F-2 DoM -----
    # Tier-0 only: regen takes ~2h45m on 24 cores (29k SVDs across both
    # models), too expensive for the per-claim 600s validate_claims timeout.
    # Manual regen: `python pathway11_h100/spectral_alpha/recompute_fe749.py`.
    Claim("fe749-alpha-l19-15b", "FE749 spectral α at L19 AUROC (1.5B) = 0.5225 (≈ chance; α at peak DoM layer is uninformative)",
          "pathway11_h100/spectral_alpha/results.json",
          ["alpha_l19_auroc_15b"], 0.5225, 0.003, "1024tok"),
    Claim("fe749-alpha-best-15b", "FE749 best-layer spectral α AUROC (1.5B) = 0.7026 at L28 (final layer; below F-2 DoM 0.7731)",
          "pathway11_h100/spectral_alpha/results.json",
          ["alpha_best_auroc_15b"], 0.7026, 0.003, "1024tok"),
    Claim("fe749-alpha-best-layer-15b", "FE749 best α layer (1.5B) = L28 (last layer)",
          "pathway11_h100/spectral_alpha/results.json",
          ["alpha_best_layer_15b"], 28, 0, "1024tok"),
    Claim("fe749-joint-alpha-dom-15b", "FE749 joint [α_L28, prefill_DoM_proj_L19] AUROC (1.5B) = 0.7833 (+1pp over DoM alone)",
          "pathway11_h100/spectral_alpha/results.json",
          ["joint_alpha_dom_auroc_15b"], 0.7833, 0.003, "1024tok"),
    Claim("fe749-alpha-l19-7b", "FE749 spectral α at L19 AUROC (7B) = 0.5215 (≈ chance; matches 1.5B pattern)",
          "pathway11_h100/spectral_alpha/results.json",
          ["alpha_l19_auroc_7b"], 0.5215, 0.003, "1024tok"),
    Claim("fe749-alpha-best-7b", "FE749 best-layer spectral α AUROC (7B) = 0.7128 at L28 (still below F-2 DoM 0.7731)",
          "pathway11_h100/spectral_alpha/results.json",
          ["alpha_best_auroc_7b"], 0.7128, 0.003, "1024tok"),

    # ----- Phase 3 causal corroborators bundle (FE136, FE244, FE319, FE331, FE339, FE254) -----
    # Tier-1 regen via single bundled script (~5min CPU):
    Claim("fe136-cos-raw", "FE136 raw cos(prefill_DoM, final_DoM) = -0.0617 (matches FE115 anchor)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe136_cos_raw"], -0.0617, 0.003, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py",
          regen_key="fe136_cos_raw"),
    Claim("fe136-cos-whitened", "FE136 whitened cos via Park-Choe-Veitch causal inner product = -0.0315 (closer to 0; F-3 reinforced)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe136_cos_whitened"], -0.0315, 0.003, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py", regen_key="fe136_cos_whitened"),
    Claim("fe244-prefill-max-cos", "FE244 max |cos| of prefill DoM with top-50 answer-token unembed rows = 0.0667 (≪ NC3 threshold 0.3)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe244_max_abs_cos_prefill_unembed"], 0.0667, 0.003, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py", regen_key="fe244_max_abs_cos_prefill_unembed"),
    Claim("fe244-final-max-cos", "FE244 max |cos| of final DoM with top-50 answer-token unembed rows = 0.1282 (still below NC3 threshold 0.3)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe244_max_abs_cos_final_unembed"], 0.1282, 0.003, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py", regen_key="fe244_max_abs_cos_final_unembed"),
    Claim("fe319-quadratic-best", "FE319 quadratic-SVD probe peak AUROC = 0.7446 at k=10 (below F-2 linear DoM 0.7731)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe319_quadratic_auroc_k10"], 0.7446, 0.003, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py", regen_key="fe319_quadratic_auroc_k10"),
    Claim("fe319-quadratic-k1", "FE319 quadratic-SVD probe k=1 AUROC = 0.7186 (single squared SVD direction)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe319_quadratic_auroc_k1"], 0.7186, 0.003, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py", regen_key="fe319_quadratic_auroc_k1"),
    Claim("fe331-stolfo-c", "FE331 Stolfo principled steering coefficient c = signed_diff = 4.724 (replaces arbitrary alpha-sweep in H-1)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe331_c_signed_diff"], 4.724, 0.05, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py", regen_key="fe331_c_signed_diff"),
    Claim("fe339-linact-auroc", "FE339 Linear-AcT (variance-aware) probe AUROC = 0.7714 (≈ tied with F-2 DoM 0.7731; ≥0.02 lift hypothesis NOT met)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe339_linear_act_auroc"], 0.7714, 0.003, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py", regen_key="fe339_linear_act_auroc"),
    Claim("fe254-joint-concat", "FE254 joint [prefill, final_tok] DoM-style concat AUROC = 0.6946 (LOWER than prefill alone 0.7705 by 0.076)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe254_joint_auroc"], 0.6946, 0.003, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py", regen_key="fe254_joint_auroc"),
    Claim("fe254-prefill-alone", "FE254 prefill-alone DoM AUROC (1.5B prefill cache, simple mean-diff) = 0.7705 (matches F-2 baseline)",
          "pathway11_h100/phase3_corroborators/results.json",
          ["fe254_prefill_alone_auroc"], 0.7705, 0.003, "1024tok",
          regen="python pathway11_h100/phase3_corroborators/recompute_phase3.py", regen_key="fe254_prefill_alone_auroc"),

    # ----- §6c P11-FE321 zigzag-style PH on 28-layer trajectory (F-10 extension) -----
    # Tier-0 only — recompute_fe321.py takes ~8 min CPU, near 600s per-claim
    # timeout under contention. Manual regen: `python pathway11_h100/zigzag_ph/recompute_fe321.py`.
    Claim("fe321-real-bz-auroc", "FE321 B_1+bar_Z_1 real AUROC = 0.4952 (canonical zigzag pair degenerate on 28-layer trajectory)",
          "pathway11_h100/zigzag_ph/results.json",
          ["auroc_real_BZ_only"], 0.4952, 0.005, "1024tok"),
    Claim("fe321-null-bz-auroc", "FE321 B_1+bar_Z_1 null AUROC = 0.5611 (matched-cov Gaussian random-topology noise above real)",
          "pathway11_h100/zigzag_ph/results.json",
          ["auroc_null_BZ_only"], 0.5611, 0.005, "1024tok"),
    Claim("fe321-bz-gap", "FE321 B_1+bar_Z_1 gap real-null = -0.0659 (F-10 narrow form holds for zigzag descriptors)",
          "pathway11_h100/zigzag_ph/results.json",
          ["gap_BZ_only"], -0.0659, 0.005, "1024tok"),
    Claim("fe321-real-full-auroc", "FE321 7-descriptor real AUROC = 0.7156 (trajectory geometry signal; below F-2 0.7731)",
          "pathway11_h100/zigzag_ph/results.json",
          ["auroc_real_full"], 0.7156, 0.005, "1024tok"),
    Claim("fe321-null-full-auroc", "FE321 7-descriptor null AUROC = 0.6188 (matched-cov Gaussian baseline)",
          "pathway11_h100/zigzag_ph/results.json",
          ["auroc_null_full"], 0.6188, 0.005, "1024tok"),
    Claim("fe321-full-gap", "FE321 7-descriptor gap = +0.0968 (H_0 + step descriptors carry signal but not topology)",
          "pathway11_h100/zigzag_ph/results.json",
          ["gap_real_minus_null_full"], 0.0968, 0.005, "1024tok"),
    Claim("fe321-overall-prune-rate", "FE321 LOLO 10%-prune rate = 0.0003 (degenerate because bar_Z_1 ≈ 0 on real trajectories)",
          "pathway11_h100/zigzag_ph/results.json",
          ["overall_prune_rate"], 0.0003, 0.001, "1024tok"),

    # ----- §6d Phase-4 GPU bundle (P11-FE110 ActAdd + P10-FE23 softmax-conf + P11-FE455 ConCISE) -----
    Claim("fe110-best-cos", "FE110 best ActAdd contrast-pair cosine with supervised DoM = 0.0651 (certain/uncertain pair; ≪ 0.5 recovery threshold)",
          "pathway11_h100/gpu_bundle/results.json",
          ["fe110", "max_cos_with_dom"], 0.0651, 0.005, "1024tok",
          regen="python pathway11_h100/gpu_bundle/recompute_gpu_bundle.py", regen_key="fe110.max_cos_with_dom"),
    Claim("fe110-best-auroc", "FE110 best contrast-pair projection AUROC on cached prefill = 0.6615 (well below F-2 0.7731)",
          "pathway11_h100/gpu_bundle/results.json",
          ["fe110", "max_projection_auroc"], 0.6615, 0.005, "1024tok",
          regen="python pathway11_h100/gpu_bundle/recompute_gpu_bundle.py", regen_key="fe110.max_projection_auroc"),
    Claim("fe110-supervised-dom-oof", "FE110 supervised DoM AUROC OOF on m15b_prefill cache = 0.7711 (matches F-2 0.7731 ± 0.002)",
          "pathway11_h100/gpu_bundle/results.json",
          ["fe110", "supervised_dom_auroc_oof"], 0.7711, 0.003, "1024tok",
          regen="python pathway11_h100/gpu_bundle/recompute_gpu_bundle.py", regen_key="fe110.supervised_dom_auroc_oof"),
    Claim("fe23-softmax-conf", "FE23 softmax-confidence AUROC at PANL = 0.4395 (BELOW chance — anti-calibrated on this MATH-500 setting)",
          "pathway11_h100/gpu_bundle/results.json",
          ["fe23", "auroc_softmax_confidence"], 0.4395, 0.005, "1024tok",
          regen="python pathway11_h100/gpu_bundle/recompute_gpu_bundle.py", regen_key="fe23.auroc_softmax_confidence"),
    Claim("fe455-c-hat", "FE455 ConCISE c_hat AUROC = 0.5887 (well below F-2 0.7731)",
          "pathway11_h100/gpu_bundle/results.json",
          ["fe455", "auroc_c_hat"], 0.5887, 0.005, "1024tok",
          regen="python pathway11_h100/gpu_bundle/recompute_gpu_bundle.py", regen_key="fe455.auroc_c_hat"),
    Claim("fe455-p-confident", "FE455 P(' confident') alone AUROC = 0.5985 (weak positive signal; slightly better than composite c_hat)",
          "pathway11_h100/gpu_bundle/results.json",
          ["fe455", "auroc_p_confident"], 0.5985, 0.005, "1024tok",
          regen="python pathway11_h100/gpu_bundle/recompute_gpu_bundle.py", regen_key="fe455.auroc_p_confident"),
    Claim("fe455-p-sure", "FE455 P(' sure') alone AUROC = 0.2927 (anti-predictive — hesitancy correlates with correctness)",
          "pathway11_h100/gpu_bundle/results.json",
          ["fe455", "auroc_p_sure"], 0.2927, 0.005, "1024tok",
          regen="python pathway11_h100/gpu_bundle/recompute_gpu_bundle.py", regen_key="fe455.auroc_p_sure"),

    # ----- §6e P11-FE116 PH on Song-Zhong residualized L19 clouds (F-10 inversion) -----
    # Tier-0 only — recompute_fe116.py takes ~25-65 min CPU, exceeds 600s per-claim
    # timeout. Manual regen: `python pathway11_h100/ph_residuals/recompute_fe116.py`.
    Claim("fe116-real-ph-resid", "FE116 real PH AUROC on Song-Zhong residualized clouds = 0.6958",
          "pathway11_h100/ph_residuals/results.json",
          ["auroc_real_ph_resid"], 0.6958, 0.003, "1024tok"),
    Claim("fe116-null-ph-resid", "FE116 null PH AUROC = 0.7624 (covariance-structure signal exposed by residualization)",
          "pathway11_h100/ph_residuals/results.json",
          ["auroc_null_ph_resid"], 0.7624, 0.003, "1024tok"),
    Claim("fe116-diff-ph-resid", "FE116 paired diff [real − null] PH AUROC = 0.5854",
          "pathway11_h100/ph_residuals/results.json",
          ["auroc_diff_ph_resid"], 0.5854, 0.003, "1024tok"),
    Claim("fe116-gap", "FE116 gap real−null on residuals = -0.0666 (inverted vs raw F-10 −0.003)",
          "pathway11_h100/ph_residuals/results.json",
          ["gap_real_minus_null"], -0.0666, 0.003, "1024tok"),
    Claim("fe116-gap-change-vs-raw", "FE116 gap shift from raw F-10 = -0.0636 (null grew faster than real)",
          "pathway11_h100/ph_residuals/results.json",
          ["gap_change_vs_raw"], -0.0636, 0.005, "1024tok"),
    Claim("fe116-mu-norm", "FE116 Song-Zhong global mean ‖μ‖ at L19 = 44.652",
          "pathway11_h100/ph_residuals/results.json",
          ["song_zhong_norms", "mu_norm"], 44.652, 0.05, "1024tok"),
    Claim("fe116-pos-t-norm", "FE116 Song-Zhong per-position bias ‖pos_t‖ at L19 = 261.910 (substantial positional structure)",
          "pathway11_h100/ph_residuals/results.json",
          ["song_zhong_norms", "pos_t_norm"], 261.910, 0.5, "1024tok"),
    Claim("fe116-h1-n-features-real", "FE116 average H_1 cycle count per real residual cloud = 31.266",
          "pathway11_h100/ph_residuals/results.json",
          ["feature_comparison", "H1_n_features", "real_mean"], 31.266, 0.5, "1024tok"),
    Claim("fe116-h1-n-features-null", "FE116 average H_1 cycle count per matched-cov Gaussian = 87.666 (~3× real)",
          "pathway11_h100/ph_residuals/results.json",
          ["feature_comparison", "H1_n_features", "null_mean"], 87.666, 0.5, "1024tok"),

    # ----- §6f P11-FE291 CAST PCA-PC1 vs supervised DoM at L19 prefill (F-2 grounding) -----
    # Tier-1: recompute_pca_covariance.py is ~1.5s CPU (full SVD on 500×1536 + 50
    # logistic fits + 5-fold OOF). Deterministic seed=0 across rebuilds.
    Claim("pca-dom-oof-auroc", "FE291 supervised DoM single-feature OOF AUROC under matched 5-fold protocol = 0.7679 (vs F-2 1536-d logistic 0.7731)",
          "pathway11_h100/pca_covariance/results.json",
          ["dom_oof_auroc"], 0.7679, 0.003, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.dom_oof_auroc"),
    Claim("pca-best-pc-auroc", "FE291 best single PC AUROC = 0.7457 (matches unsupervised PC1)",
          "pathway11_h100/pca_covariance/results.json",
          ["best_pc_auroc"], 0.7457, 0.003, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.best_pc_auroc"),
    Claim("pca-best-pc-index", "FE291 best PC by AUROC is PC1 — the dominant variance direction is the correctness direction",
          "pathway11_h100/pca_covariance/results.json",
          ["best_pc_auroc_index"], 1, 0.5, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.best_pc_auroc_index"),
    Claim("pca-pc1-var-share", "FE291 PC1 carries 14.7% of total L19 prefill variance",
          "pathway11_h100/pca_covariance/results.json",
          ["pc_var_share_top10", 0], 0.1474, 0.005, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.pc1_var_share"),
    Claim("pca-dom-pc1-cosine", "FE291 cosine between supervised DoM and unsupervised PC1 = 0.9216 (near-identical directions)",
          "pathway11_h100/pca_covariance/results.json",
          ["dom_pc1_cosine"], 0.9216, 0.003, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.dom_pc1_cosine"),
    Claim("pca-dom-largest-pc-rank", "FE291 supervised DoM largest PC-basis coefficient is at PC1",
          "pathway11_h100/pca_covariance/results.json",
          ["dom_largest_coeff_pc_rank"], 1, 0.5, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.dom_largest_coeff_pc_rank"),
    Claim("pca-dom-energy-top10-share", "FE291 top-10 PCs explain 97.6% of unit-DoM energy",
          "pathway11_h100/pca_covariance/results.json",
          ["dom_energy_top10_share"], 0.9764, 0.005, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.dom_energy_top10_share"),
    Claim("pca-cast-pc1-auroc", "FE291 CAST class-mean-centered PCA-PC1 OOF AUROC = 0.7458 (matches unsupervised PC1 within 0.0001)",
          "pathway11_h100/pca_covariance/results.json",
          ["cast_pc1_auroc"], 0.7458, 0.003, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.cast_pc1_auroc"),
    Claim("pca-cast-pc1-dom-cosine", "FE291 CAST PC1 cosine with supervised DoM = 0.9217 (matches unsupervised PC1 cosine within 0.0001)",
          "pathway11_h100/pca_covariance/results.json",
          ["cast_pc1_dom_cosine"], 0.9217, 0.003, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.cast_pc1_dom_cosine"),
    Claim("pca-pc1-pc9-oof-auroc", "FE291 follow-up: 2-feature [PC1, PC9] OOF logistic AUROC = 0.7856, exceeds supervised DoM 0.7679 by 1.8pt — DoM under-weights PC9 vs OOF-optimal",
          "pathway11_h100/pca_covariance/results.json",
          ["pc1_pc9_oof_auroc"], 0.7856, 0.005, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.pc1_pc9_oof_auroc"),
    Claim("pca-pc1resid-ph-real-auroc", "FE291 follow-up #1 (FE880): 5 V-R PH features on PC1-residualized L19 clouds, real OOF AUROC = 0.6884 — fails F-10's pre-registered overturning condition (real >= null + 0.05)",
          "pathway11_h100/ph_residuals/pc1_resid_results.json",
          ["classifiers", "real_ph_pc1resid", "auroc_oof"], 0.6884, 0.005, "1024tok",
          regen="python pathway11_h100/ph_residuals/recompute_pc1_resid_ph.py", regen_key="pc1_resid.auroc_real_ph_pc1resid"),
    Claim("pca-pc1resid-ph-null-auroc", "FE291 follow-up #1 (FE880): matched-cov Gaussian null PH features on same PC1-residualized clouds, OOF AUROC = 0.7628 — null beats real by 7.4pt; F-10 sharpens",
          "pathway11_h100/ph_residuals/pc1_resid_results.json",
          ["classifiers", "null_ph_pc1resid", "auroc_oof"], 0.7628, 0.005, "1024tok",
          regen="python pathway11_h100/ph_residuals/recompute_pc1_resid_ph.py", regen_key="pc1_resid.auroc_null_ph_pc1resid"),
    Claim("pca-pc1resid-ph-gap", "FE291 follow-up #1 (FE880): real - null PH gap on PC1-residualized clouds = -0.0744; pre-registered F-10 overturning condition tested directly and failed by 12.4pt in the wrong direction",
          "pathway11_h100/ph_residuals/pc1_resid_results.json",
          ["gap_real_minus_null"], -0.0744, 0.005, "1024tok",
          regen="python pathway11_h100/ph_residuals/recompute_pc1_resid_ph.py", regen_key="pc1_resid.gap_real_minus_null"),
    Claim("pca-pc1resid-cov-spectrum-top20-auroc", "FE291 follow-up #2: top-20 log-eigvals of per-problem L19 cov (after PC1 residualization) OOF logistic AUROC = 0.7928 — beats supervised DoM 0.7679 by 2.5pt; F-2 signal is partly spectral",
          "pathway11_h100/cov_spectrum/pc1_resid_cov_spectrum_results.json",
          ["best_auroc_oof"], 0.7928, 0.005, "1024tok",
          regen="python pathway11_h100/cov_spectrum/recompute_pc1_resid_cov_spectrum.py", regen_key="cov_spectrum.best_auroc_oof"),
    Claim("pca-full-lr-best-auroc", "FE291 follow-up #3: full 1536-d L2-reg logistic OOF AUROC max = 0.7847 (at C=0.001) — essentially equal to 2-feat [PC1, PC9] 0.7856; the cov-spectrum 0.7928 still beats the directional ceiling, so F-2's extra signal is genuinely second-order",
          "pathway11_h100/pca_covariance/results.json",
          ["full_lr_best_auroc"], 0.7847, 0.005, "1024tok",
          regen="python pathway11_h100/pca_covariance/recompute_pca_covariance.py", regen_key="pca.full_lr_best_auroc"),

    # ----- §7 Selective prediction / refuse-and-spend -----
    # Tier-1: recompute_selective.py reimplements the quartile gate policies
    # over phase{1,2}_*.npz (refuse@0.5 + middle-heavy K=4.5).
    Claim("refuse-prefill-acc", "Prefill refuse@0.5 answered acc = 0.716",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["headline", "refuse_and_spend_coverage_0.5", "prefill", "acc_on_answered"],
          0.716, 0.005, "1024tok",
          regen=_SELECTIVE_REGEN, regen_key="refuse_prefill_acc"),
    Claim("refuse-random-acc", "Random refuse@0.5 answered acc = 0.494",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["headline", "refuse_and_spend_coverage_0.5", "random", "acc_on_answered"],
          0.494, 0.010, "1024tok",
          regen=_SELECTIVE_REGEN, regen_key="refuse_random_acc"),
    Claim("middle-heavy-acc", "Prefill middle-heavy K=4.5 acc = 0.526",
          "pathway11_h100/prefill_gated_compute/results.json",
          ["headline", "prefill_middle_heavy", "overall_accuracy"],
          0.526, 0.005, "1024tok",
          regen=_SELECTIVE_REGEN, regen_key="middle_heavy_acc"),

    # ----- §9 Prefill inversion (7B) -----
    # Tier-1: recompute_pr_ratios.py reloads cache/m{7b,15b}_prefill.npz and
    # recomputes participation ratios from scratch. Bootstrap CIs (inv-*-ci-*)
    # have no regen — bootstrap is non-deterministic across runs.
    Claim("inv-7B-ratio", "7B prefill PR ratio correct/incorrect = 1.377",
          "pathway11_h100/prefill_inversion/results.json",
          ["headline_ratios_point_estimates", "7B_prefill_PR_correct_over_incorrect"],
          1.377113639689133, 1e-5, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="ratio_7b_prefill"),
    Claim("inv-15B-ratio", "1.5B prefill PR ratio = 0.946",
          "pathway11_h100/prefill_inversion/results.json",
          ["headline_ratios_point_estimates", "1.5B_prefill_PR_correct_over_incorrect"],
          0.9456399969402085, 1e-5, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="ratio_15b_prefill"),
    Claim("inv-7B-final-ratio", "7B final-token PR ratio = 0.384",
          "pathway11_h100/prefill_inversion/results.json",
          ["headline_ratios_point_estimates", "7B_final_token_PR_correct_over_incorrect"],
          0.38397476790580454, 1e-5, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="ratio_7b_final"),
    Claim("inv-15B-final-ratio", "1.5B final-token PR ratio = 0.509",
          "pathway11_h100/prefill_inversion/results.json",
          ["headline_ratios_point_estimates", "1.5B_final_token_PR_correct_over_incorrect"],
          0.5086845084799126, 1e-5, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="ratio_15b_final"),
    # Bootstrap CIs: deterministic with SEED=9999, but 1000-iter bootstrap on
    # 500×3584 matrices takes ~7 min — too slow for routine gate. Opt in via
    # VALIDATE_BOOTSTRAP_CI=1 env var; default Tier-0-only (still back-checks
    # the JSON-vs-doc consistency). See phase1_bootstrap.py for the canonical run.
    Claim("inv-7B-ci-lo", "7B prefill ratio 95% CI lo = 1.226",
          "pathway11_h100/prefill_inversion/results.json",
          ["bootstrap_95_CI_on_ratio", "7B_prefill", 0], 1.2254645586938757, 1e-5, "1024tok"),
    Claim("inv-7B-ci-hi", "7B prefill ratio 95% CI hi = 1.757",
          "pathway11_h100/prefill_inversion/results.json",
          ["bootstrap_95_CI_on_ratio", "7B_prefill", 1], 1.7569904867197708, 1e-5, "1024tok"),
    Claim("inv-7B-balance-ratio", "7B prefill balance-controlled ratio = 1.227",
          "pathway11_h100/prefill_inversion/results.json",
          ["balance_controlled_ratios", "7B_prefill"], 1.2274623411828771, 1e-5, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="ratio_7b_balance_controlled"),

    # Phase 1 bootstrap PR values (7B prefill, used in §3 aggregate table)
    Claim("7B-prefill-PR-correct", "7B prefill PR correct = 26.87",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_prefill", "point", "pr_correct"], 26.872772216796875, 0.01, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="pr_7b_prefill_correct"),
    Claim("7B-prefill-PR-incorrect", "7B prefill PR incorrect = 19.51",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_prefill", "point", "pr_incorrect"], 19.513837814331055, 0.01, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="pr_7b_prefill_incorrect"),
    Claim("7B-final-PR-correct", "7B final-token PR correct = 2.63",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_final_token", "point", "pr_correct"], 2.629302501678467, 0.01, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="pr_7b_final_correct"),
    Claim("7B-final-PR-incorrect", "7B final-token PR incorrect = 6.85",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_final_token", "point", "pr_incorrect"], 6.847591876983643, 0.01, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="pr_7b_final_incorrect"),
    Claim("15B-prefill-PR-correct", "1.5B prefill PR correct = 19.41",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_prefill", "point", "pr_correct"], 19.405458450317383, 0.01, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="pr_15b_prefill_correct"),
    Claim("15B-prefill-PR-incorrect", "1.5B prefill PR incorrect = 20.52",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_prefill", "point", "pr_incorrect"], 20.520978927612305, 0.01, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="pr_15b_prefill_incorrect"),
    Claim("15B-final-PR-correct", "1.5B final-token PR correct = 4.33",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_final_token", "point", "pr_correct"], 4.329, 0.05, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="pr_15b_final_correct"),
    Claim("15B-final-PR-incorrect", "1.5B final-token PR incorrect = 8.51",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_final_token", "point", "pr_incorrect"], 8.51, 0.05, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="pr_15b_final_incorrect"),

    # Phase 1 counts
    Claim("7B-n-correct", "7B n_correct = 366",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_prefill", "n_correct"], 366, 0, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="n_7b_correct"),
    Claim("7B-n-incorrect", "7B n_incorrect = 134",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["7B_prefill", "n_incorrect"], 134, 0, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="n_7b_incorrect"),
    Claim("15B-n-correct", "1.5B n_correct = 243",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_prefill", "n_correct"], 243, 0, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="n_15b_correct"),
    Claim("15B-n-incorrect", "1.5B n_incorrect = 257",
          "pathway11_h100/prefill_inversion/phase1_bootstrap.json",
          ["1.5B_prefill", "n_incorrect"], 257, 0, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="n_15b_incorrect"),

    # Three-way split
    Claim("three-way-both-right-n", "Both-right n=230",
          "pathway11_h100/prefill_inversion/results.json",
          ["three_way_split", "both_right", "n"], 230, 0, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="three_way_both_right_n"),
    Claim("three-way-only7b-n", "Only-7B-right n=136",
          "pathway11_h100/prefill_inversion/results.json",
          ["three_way_split", "only_7b", "n"], 136, 0, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="three_way_only_7b_n"),
    Claim("three-way-only15b-n", "Only-1.5B-right n=13",
          "pathway11_h100/prefill_inversion/results.json",
          ["three_way_split", "only_15b", "n"], 13, 0, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="three_way_only_15b_n"),
    Claim("three-way-both-wrong-n", "Both-wrong n=121",
          "pathway11_h100/prefill_inversion/results.json",
          ["three_way_split", "both_wrong", "n"], 121, 0, "1024tok",
          regen=_PR_RATIOS_REGEN, regen_key="three_way_both_wrong_n"),

    # BBH prefill ratios (§9). Tier-1: recompute_bbh_pr_ratios.py reloads
    # pathway8_layerwise/data/bbh/<subset>/problem_*.npz and recomputes PR ratio.
    Claim("bbh-tracking-ratio", "BBH tracking_shuffled PR prefill ratio = 0.930",
          "pathway11_h100/prefill_inversion/results.json",
          ["bbh", "tracking_shuffled_objects_seven_objects", "pr_prefill_ratio"],
          0.9304261877758804, 1e-4, "1024tok",
          regen=_BBH_PR_REGEN,
          regen_key="pr_bbh_tracking_shuffled_objects_seven_objects_prefill_ratio"),
    Claim("bbh-logical-ratio", "BBH logical_deduction PR prefill ratio = 0.824",
          "pathway11_h100/prefill_inversion/results.json",
          ["bbh", "logical_deduction_seven_objects", "pr_prefill_ratio"],
          0.8240621037129747, 1e-4, "1024tok",
          regen=_BBH_PR_REGEN,
          regen_key="pr_bbh_logical_deduction_seven_objects_prefill_ratio"),
    Claim("bbh-web-ratio", "BBH web_of_lies PR prefill ratio = 1.009",
          "pathway11_h100/prefill_inversion/results.json",
          ["bbh", "web_of_lies", "pr_prefill_ratio"], 1.00919347747066, 1e-4, "1024tok",
          regen=_BBH_PR_REGEN, regen_key="pr_bbh_web_of_lies_prefill_ratio"),

    # ----- §10 Multi-signal oracle (Exp 2b) -----
    # Tier-1: recompute_classifier.py rebuilds macro_f1 / acc / D-recall from
    # oof_{logreg,rf}.npz (the OOF predictions cache).
    Claim("exp2b-logreg-f1", "Exp 2b logreg macro-F1 = 0.548",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_logreg", "macro_f1"], 0.5477024403341905, 1e-4, "1024tok",
          regen=_CLASSIFIER_REGEN, regen_key="logreg_macro_f1"),
    Claim("exp2b-rf-f1", "Exp 2b RF macro-F1 = 0.494",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_rf", "macro_f1"], 0.49397192549427976, 1e-4, "1024tok",
          regen=_CLASSIFIER_REGEN, regen_key="rf_macro_f1"),
    Claim("exp2b-logreg-acc", "Exp 2b logreg overall acc = 0.620",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_logreg", "overall_acc_oof"], 0.62, 0.005, "1024tok",
          regen=_CLASSIFIER_REGEN, regen_key="logreg_overall_acc"),
    Claim("exp2b-rf-acc", "Exp 2b RF overall acc = 0.636",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_rf", "overall_acc_oof"], 0.636, 0.005, "1024tok",
          regen=_CLASSIFIER_REGEN, regen_key="rf_overall_acc"),
    Claim("exp2b-logreg-Drecall", "Logreg D-recall to K=1 = 0.417",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_logreg", "D_recall_to_K1"], 0.4166666666666667, 1e-4, "1024tok",
          regen=_CLASSIFIER_REGEN, regen_key="logreg_D_recall_to_K1"),
    Claim("exp2b-rf-Drecall", "RF D-recall to K=1 = 0.444",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["classifier_rf", "D_recall_to_K1"], 0.4444444444444444, 1e-4, "1024tok",
          regen=_CLASSIFIER_REGEN, regen_key="rf_D_recall_to_K1"),
    # Tier-1 for the comparison-derived metrics: simulate K=1-fallback policy
    # over oof_logreg.npz, subtract best-single-signal (read from phase3b).
    Claim("exp2b-multi-vs-single", "Multi-signal vs best single = +0.4pp",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["multi_signal_minus_single_signal_pp"], 0.4, 0.05, "1024tok",
          regen=_EXP2B_EXTRAS_REGEN, regen_key="multi_vs_single_pp"),
    Claim("exp2b-2pp-met", "2pp bar met = False",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["user_threshold_2pp_met"], False, 0, "1024tok",
          regen=_EXP2B_EXTRAS_REGEN, regen_key="threshold_2pp_met"),
    Claim("exp2b-best-single", "Best single-signal compute 2.74, acc 0.516",
          "pathway11_h100/multi_signal_oracle/results.json",
          ["best_single_signal_at_matched_compute", "overall_accuracy"], 0.516, 0.003, "1024tok",
          regen=_EXP2B_EXTRAS_REGEN, regen_key="best_single_acc"),

    # ----- §3 Gibberish control -----
    # Tier-1: gibberish_control/recompute_pr.py reloads data/{math500,random,
    # stream}/problem_*.npz and recomputes participation_ratio per position.
    Claim("gib-math-pos50", "MATH-500 PR pos 50 = 30.55",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "math500_baseline", "pr_by_position", "50", "pr"],
          30.554419380797377, 0.01, "1024tok",
          regen=_GIB_PR_REGEN, regen_key="gib_math_pos50"),
    Claim("gib-math-pos1", "MATH-500 PR pos 1 = 13.74",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "math500_baseline", "pr_by_position", "1", "pr"],
          13.73524962240093, 0.01, "1024tok",
          regen=_GIB_PR_REGEN, regen_key="gib_math_pos1"),
    Claim("gib-math-final", "MATH-500 PR final = 22.19",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "math500_baseline", "pr_by_position", "final", "pr"],
          22.186663572792018, 0.01, "1024tok",
          regen=_GIB_PR_REGEN, regen_key="gib_math_posfinal"),
    Claim("gib-random-pos1", "Random PR pos 1 = 11.32",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "random_tokens", "pr_by_position", "1", "pr"],
          11.315935166610068, 0.01, "1024tok",
          regen=_GIB_PR_REGEN, regen_key="gib_random_pos1"),
    Claim("gib-random-final", "Random PR final = 8.96",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "random_tokens", "pr_by_position", "final", "pr"],
          8.963097360506318, 0.01, "1024tok",
          regen=_GIB_PR_REGEN, regen_key="gib_random_posfinal"),
    Claim("gib-stream-pos10", "Stream-of-consc PR pos 10 = 15.66",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "stream_of_consciousness", "pr_by_position", "10", "pr"],
          15.662493885959867, 0.01, "1024tok",
          regen=_GIB_PR_REGEN, regen_key="gib_stream_pos10"),
    Claim("gib-stream-pos1", "Stream-of-consc PR pos 1 = 4.60",
          "pathway11_h100/gibberish_control/pr_curves.json",
          ["conditions", "stream_of_consciousness", "pr_by_position", "1", "pr"],
          4.5966970751104155, 0.01, "1024tok",
          regen=_GIB_PR_REGEN, regen_key="gib_stream_pos1"),

    # ----- No-CoT control -----
    # Tier-1: no_cot_control/recompute_pr.py reloads data/nocot/problem_*.npz.
    Claim("nocot-pos1", "No-CoT PR pos 1 = 15.78",
          "pathway11_h100/no_cot_control/pr_curves.json",
          ["conditions", "nocot", "pr_by_position", "1", "pr"],
          15.78, 0.1, "1024tok",
          regen=_NOCOT_PR_REGEN, regen_key="nocot_pos1"),
    Claim("nocot-final", "No-CoT PR final = 19.80",
          "pathway11_h100/no_cot_control/pr_curves.json",
          ["conditions", "nocot", "pr_by_position", "final", "pr"],
          19.80, 0.1, "1024tok",
          regen=_NOCOT_PR_REGEN, regen_key="nocot_posfinal"),

    # ----- Exp 1 cross-model -----
    # Tier-1: exp1_cross_model/recompute_acc.py reloads data/{phi3mini,
    # llama32_1b}/problem_*.npz and counts correctness.
    Claim("phi3-acc", "Phi-3-mini MATH-500 acc = 0.448",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 0, "accuracy"], 0.448, 0.002, "1024tok",
          regen=_CROSS_MODEL_REGEN, regen_key="phi3_acc"),
    Claim("phi3-n-correct", "Phi-3-mini n_correct = 224",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 0, "n_correct"], 224, 0, "1024tok",
          regen=_CROSS_MODEL_REGEN, regen_key="phi3_n_correct"),
    Claim("llama-acc", "Llama-3.2-1B MATH-500 acc = 0.252",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 1, "accuracy"], 0.252, 0.002, "1024tok",
          regen=_CROSS_MODEL_REGEN, regen_key="llama_acc"),
    Claim("llama-n-correct", "Llama-3.2-1B n_correct = 126",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 1, "n_correct"], 126, 0, "1024tok",
          regen=_CROSS_MODEL_REGEN, regen_key="llama_n_correct"),
    Claim("phi3-2thirds-layer", "Phi-3 2/3-depth layer = 21",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 0, "twothirds_layer_idx"], 21, 0, "1024tok",
          regen=_CROSS_MODEL_REGEN, regen_key="phi3_2thirds_layer"),
    Claim("llama-2thirds-layer", "Llama 2/3-depth layer = 11",
          "pathway11_h100/exp1_cross_model/results.json",
          ["models", 1, "twothirds_layer_idx"], 11, 0, "1024tok",
          regen=_CROSS_MODEL_REGEN, regen_key="llama_2thirds_layer"),

    # ----- Stage 5 BBH L19 DoM transfer -----
    # Tier-1: recompute_stage5_bbh_transfer.py reloads pathway8_layerwise/data/
    # {math500,bbh/<subset>}/problem_*.npz, fits DoM, scores cross-domain. Slow
    # (~2 min) due to ~1250 NPZ loads.
    Claim("stage5-math-to-bbh-pooled", "MATH→BBH pooled AUROC = 0.747",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["math_to_bbh", "_pooled", "auroc"], 0.7471963025177446, 1e-4, "1024tok",
          regen=_STAGE5_REGEN, regen_key="stage5_math_to_bbh_pooled"),
    Claim("stage5-bbh-to-math", "BBH→MATH AUROC = 0.693",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["bbh_to_math", "auroc"], 0.6926230164448928, 1e-4, "1024tok",
          regen=_STAGE5_REGEN, regen_key="stage5_bbh_to_math"),
    Claim("stage5-symmetric", "Symmetric avg = 0.720",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["symmetric_avg"], 0.7199096594813187, 1e-4, "1024tok",
          regen=_STAGE5_REGEN, regen_key="stage5_symmetric_avg"),
    Claim("stage5-verdict-vs-coe", "Delta vs CoE symmetric = +0.004",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["verdict_vs_coe"], 0.003909659481318717, 1e-4, "1024tok",
          regen=_STAGE5_REGEN, regen_key="stage5_verdict_vs_coe"),
    Claim("stage5-within-math", "MATH within-benchmark AUROC = 0.731",
          "pathway11_h100/results/stage5_bbh_dom_transfer.json",
          ["within_benchmark_overfit", "math", "auroc"], 0.7305727690509358, 1e-4, "1024tok",
          regen=_STAGE5_REGEN, regen_key="stage5_within_math"),

    # ----- Pathway 9 claims (256-tok labels, many superseded) -----
    Claim("p9-abc44-holdout-raw", "ABC-44 holdout AUROC (raw) = 0.7961",
          "pathway9/results/exp1_raw_length_v2.json",
          ["baseline_auroc"], 0.7960784313725491, 1e-4, "256tok_NEW",
          note="SUPERSEDED: computed against 104/500 NEW labels; 1024-tok re-run gives prefill+final DoM ≈0.794"),
    Claim("p9-abc44-holdout-deconfounded", "ABC-44 deconfounded (raw length) AUROC = 0.7459",
          "pathway9/results/exp1_raw_length_v2.json",
          ["deconfounded_auroc"], 0.7458823529411764, 1e-4, "256tok_NEW",
          note="SUPERSEDED: length-confound was feature-engineering residue; raw residual-stream DoM is not length-confounded (cos=-0.09)"),
    Claim("p9-length-drop", "Drop from raw-length deconfounding = 0.050",
          "pathway9/results/exp1_raw_length_v2.json",
          ["drop_from_deconfounding"], 0.050, 0.001, "256tok_NEW"),
    Claim("p9-gaussian-null-real", "Real PH 5-feat AUROC = 0.690",
          "pathway9/results/exp3a_gaussian_null_v2.json",
          ["classifiers", "real_ph", "auroc_holdout"], 0.6901960784313725, 1e-4, "256tok_NEW"),
    Claim("p9-gaussian-null-null", "Null (empirical-cov) PH AUROC = 0.693",
          "pathway9/results/exp3a_gaussian_null_v2.json",
          ["classifiers", "null_ph", "auroc_holdout"], 0.6925490196078431, 1e-4, "256tok_NEW"),
    Claim("p9-lr-vs-xgb-lift", "XGB-LR lift CV5 = -0.005",
          "pathway9/results/exp4_xgboost_oinfo_v2.json",
          ["xgboost_lift_cv5"], -0.005228758169934511, 1e-4, "256tok_NEW"),
    Claim("p9-coe-math-to-bbh", "CoE MATH→BBH = 0.720",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "coe", "MATH_to_BBH_auroc"], 0.7204625669180751, 1e-4, "256tok_manifest"),
    Claim("p9-coe-bbh-to-math", "CoE BBH→MATH = 0.712",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "coe", "BBH_to_MATH_auroc"], 0.7122901881266194, 1e-4, "256tok_manifest"),
    Claim("p9-ph-math-to-bbh", "PH-168 MATH→BBH = 0.354",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "layerwise_ph", "MATH_to_BBH_auroc"], 0.3539202363906647, 1e-4, "256tok_manifest"),
    Claim("p9-ph-bbh-to-math", "PH-168 BBH→MATH = 0.497",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "layerwise_ph", "BBH_to_MATH_auroc"], 0.4967572699914708, 1e-4, "256tok_manifest"),
    Claim("p9-d2h-math-to-bbh", "D2H-lite MATH→BBH = 0.740",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "d2h_lite", "MATH_to_BBH_auroc"], 0.7403350691752775, 1e-4, "256tok_manifest"),
    Claim("p9-d2h-bbh-to-math", "D2H-lite BBH→MATH = 0.529",
          "pathway9/results/exp5_cross_domain.json",
          ["transfer_matrix", "d2h_lite", "BBH_to_MATH_auroc"], 0.5287339673956775, 1e-4, "256tok_manifest"),

    # ----- Pathway 8 layer-wise results (256-tok NEW labels) -----
    # Model array index: 0=ABC-44, 1=LayerwisePH-168, 2=CoE-60, 3=D2H-lite, 4=D2H-full, 5=PH+CoE, 6=PH+D2H, 7=All
    Claim("p8-abc44-auroc", "Pathway 8 ABC-44 holdout = 0.796",
          "pathway8_layerwise/results/exp2_results.json",
          ["models", 0, "auroc_holdout"], 0.7961, 1e-3, "256tok_NEW"),
    Claim("p8-coe-auroc", "Pathway 8 CoE-60 holdout = 0.811",
          "pathway8_layerwise/results/exp2_results.json",
          ["models", 2, "auroc_holdout"], 0.811, 1e-3, "256tok_NEW"),
    Claim("p8-d2h-lite-auroc", "Pathway 8 D2H-lite holdout = 0.806",
          "pathway8_layerwise/results/exp2_results.json",
          ["models", 3, "auroc_holdout"], 0.8055, 1e-3, "256tok_NEW"),
    Claim("p8-layerwise-ph", "Pathway 8 layer-wise PH = 0.646",
          "pathway8_layerwise/results/exp1_results.json",
          ["auroc_holdout"], 0.6463, 1e-3, "256tok_NEW"),
    Claim("p8-ph-coe-ensemble", "Pathway 8 PH+CoE combined = 0.758",
          "pathway8_layerwise/results/exp2_results.json",
          ["models", 5, "auroc_holdout"], 0.7584, 1e-3, "256tok_NEW"),

    # ----- Scratch pathway10 old-labels numbers (for §4 "superseded" provenance) -----
    Claim("scratch-rc-auroc", "Pathway10 rc 5fold prefill AUROC (old labels) = 0.7525",
          "scratch/pathway10_rc_5fold_results.json",
          ["methods", "1.5B_L19_DoM", "auroc"], 0.7524766899766899, 1e-4, "256tok_NEW"),
    Claim("scratch-rc-aurc", "Pathway10 rc 5fold AURC = 0.6346",
          "scratch/pathway10_rc_5fold_results.json",
          ["methods", "1.5B_L19_DoM", "aurc"], 0.6346043846421335, 1e-4, "256tok_NEW"),
    Claim("scratch-temporal-prefill", "Pathway10 temporal prefill AUROC (old) = 0.3921",
          "scratch/pathway10_temporal_and_verifier_results.json",
          ["T1_temporal_dom", "positions", 0, "dom_holdout_auroc"], 0.39210367691380343, 1e-4, "256tok_NEW",
          note="SUPERSEDED: reversal was label-distribution artifact at 104/500 correct"),
    Claim("scratch-temporal-cos-prefill-final", "Old prefill vs final DoM cosine = 0.046",
          "scratch/pathway10_temporal_and_verifier_results.json",
          ["T1_temporal_dom", "positions", 0, "cos_with_final_token_dom"], 0.046196311712265015, 1e-4, "256tok_NEW"),

    # ----- External anchors registered from triage briefs (2026-04-29 batch) -----
    # These are paper-cited numbers carried into PAPER_INDEX.md / brief footers
    # for cross-reference. They are NOT back-checked against any local JSON
    # (kind="external"); they exist so a future grep `validate_claims.py` will
    # surface their provenance and so the claims-gate count reflects the briefs
    # the project has chosen to graph. If we ever replicate one of these, swap
    # kind="external" → "internal" and add file/path/regen.

    # 2205.14334 — Lin et al., teaching language models to express calibration
    Claim("ext-2205.14334-verb-mse", "2205.14334 Tbl1: verbalized finetune MSE 22.0 (Multi-answer, GPT-3-175B)",
          "", [], 22.0, labels="external_anchor", kind="external",
          note="Lin et al. 2205.14334 Table 1"),
    Claim("ext-2205.14334-indirect-logit-mse", "2205.14334 Tbl1: indirect-logit finetune MSE 11.7 (Multiply-divide)",
          "", [], 11.7, labels="external_anchor", kind="external",
          note="Lin et al. 2205.14334 Table 1"),
    Claim("ext-2205.14334-answer-logit-zeroshot-mse", "2205.14334 Tbl1: answer-logit zero-shot MSE 10.4 (Multiply-divide)",
          "", [], 10.4, labels="external_anchor", kind="external",
          note="Lin et al. 2205.14334 Table 1"),
    Claim("ext-2205.14334-add-sub-acc", "2205.14334 §2.2: Add-subtract median accuracy 0.21 (label-shift)",
          "", [], 0.21, labels="external_anchor", kind="external",
          note="Lin et al. 2205.14334 §2.2"),
    Claim("ext-2205.14334-multi-answer-acc", "2205.14334 §2.2: Multi-answer median accuracy 0.65",
          "", [], 0.65, labels="external_anchor", kind="external",
          note="Lin et al. 2205.14334 §2.2"),

    # 2210.00069 — Euclidicity (von Rohrscheidt & Rieck)
    Claim("ext-2210.00069-mnist-misclass", "2210.00069: MNIST Euclidicity mean 0.39 (misclassified)",
          "", [], 0.39, labels="external_anchor", kind="external",
          note="von Rohrscheidt & Rieck 2210.00069"),
    Claim("ext-2210.00069-mnist-correct", "2210.00069: MNIST Euclidicity mean 0.33 (correctly classified)",
          "", [], 0.33, labels="external_anchor", kind="external",
          note="von Rohrscheidt & Rieck 2210.00069"),

    # 2311.04897 — Future Lens
    Claim("ext-2311.04897-fl-top1-t2", "2311.04897 Future Lens: >0.48 top-1 t+2 prediction (linear, GPT-J-6B)",
          "", [], 0.48, labels="external_anchor", kind="external",
          note="Pal et al. 2311.04897 — lower-bound; >48% reported"),

    # 2407.12404 — Tan et al., steerability
    Claim("ext-2407.12404-llama-id-ood", "2407.12404: ρ ID↔OOD steerability = 0.891 (Llama)",
          "", [], 0.891, labels="external_anchor", kind="external",
          note="Tan et al. 2407.12404"),
    Claim("ext-2407.12404-qwen-id-ood", "2407.12404: ρ ID↔OOD steerability = 0.694 (Qwen)",
          "", [], 0.694, labels="external_anchor", kind="external",
          note="Tan et al. 2407.12404"),
    Claim("ext-2407.12404-cross-id", "2407.12404: ρ cross-model (Llama↔Qwen) ID = 0.769",
          "", [], 0.769, labels="external_anchor", kind="external",
          note="Tan et al. 2407.12404"),
    Claim("ext-2407.12404-cross-ood", "2407.12404: ρ cross-model (Llama↔Qwen) OOD = 0.586",
          "", [], 0.586, labels="external_anchor", kind="external",
          note="Tan et al. 2407.12404"),

    # 2501.09929 — FGAA (fine-grained activation addition)
    Claim("ext-2501.09929-gemma2b-fgaa-coh", "2501.09929: FGAA behavioral-coherence 0.4702 (Gemma-2-2B)",
          "", [], 0.4702, labels="external_anchor", kind="external",
          note="Soo et al. 2501.09929 — vs CAA 0.2201"),
    Claim("ext-2501.09929-fgaa-inflection", "2501.09929: FGAA capability inflection α≈40 on MMLU/MMLU-Pro",
          "", [], 40, labels="external_anchor", kind="external",
          note="Soo et al. 2501.09929"),

    # 2501.17148 — AxBench (Wu et al.)
    Claim("ext-2501.17148-diffmean-auroc", "2501.17148 AxBench: DiffMean mean AUROC 0.942 (4 Gemma residual sites)",
          "", [], 0.942, labels="external_anchor", kind="external",
          note="Wu et al. 2501.17148 Table 1"),
    Claim("ext-2501.17148-diffmean-steer", "2501.17148 AxBench: DiffMean mean steering 0.239 / 2.0",
          "", [], 0.239, labels="external_anchor", kind="external",
          note="Wu et al. 2501.17148 Table 2"),
    Claim("ext-2501.17148-ssv-steer", "2501.17148 AxBench: SSV mean steering 0.026 / 2.0",
          "", [], 0.026, labels="external_anchor", kind="external",
          note="Wu et al. 2501.17148 Table 2"),
    Claim("ext-2501.17148-reft-steer", "2501.17148 AxBench: ReFT-r1 mean steering 0.543 / 2.0",
          "", [], 0.543, labels="external_anchor", kind="external",
          note="Wu et al. 2501.17148 Table 2"),
    Claim("ext-2501.17148-reft-winrate", "2501.17148 AxBench: ReFT-r1 winrate vs SAEs 0.818",
          "", [], 0.818, labels="external_anchor", kind="external",
          note="Wu et al. 2501.17148 Table 3"),
    Claim("ext-2501.17148-gemma-l10-auroc", "2501.17148 AxBench: Gemma-2-2B DiffMean L10 AUROC 0.948",
          "", [], 0.948, labels="external_anchor", kind="external",
          note="Wu et al. 2501.17148 Table 1"),

    # 2504.10063 — TOHA (Topology of Hallucination)
    Claim("ext-2504.10063-qwen-squad", "2504.10063 TOHA: AUROC 0.77 ± 0.02 on Qwen2.5-7B SQuAD",
          "", [], 0.77, labels="external_anchor", kind="external",
          note="Wu et al. 2504.10063 Table 2"),
    Claim("ext-2504.10063-mistral-squad", "2504.10063 TOHA: AUROC 0.89 ± 0.01 on Mistral-7B SQuAD",
          "", [], 0.89, labels="external_anchor", kind="external",
          note="Wu et al. 2504.10063 Table 1"),
    Claim("ext-2504.10063-mistral-xsum", "2504.10063 TOHA: AUROC 0.96 ± 0.01 on Mistral-7B XSum",
          "", [], 0.96, labels="external_anchor", kind="external",
          note="Wu et al. 2504.10063 Table 1"),

    # 2507.16806 — RLCR (calibrated reasoning)
    Claim("ext-2507.16806-base-ece", "2507.16806 Tbl1b: base Qwen-2.5-7B ECE 0.39 (Math avg)",
          "", [], 0.39, labels="external_anchor", kind="external",
          note="Damani et al. 2507.16806 Table 1b"),
    Claim("ext-2507.16806-base-brier", "2507.16806 Tbl1b: base Qwen-2.5-7B Brier 0.40 (Math avg)",
          "", [], 0.40, labels="external_anchor", kind="external",
          note="Damani et al. 2507.16806 Table 1b"),
    Claim("ext-2507.16806-rlvr-ece", "2507.16806 Tbl1b: RLVR ECE 0.26 (Math avg)",
          "", [], 0.26, labels="external_anchor", kind="external",
          note="Damani et al. 2507.16806 Table 1b"),
    Claim("ext-2507.16806-rlcr-ece", "2507.16806 Tbl1b: RLCR ECE 0.10 (Math avg)",
          "", [], 0.10, labels="external_anchor", kind="external",
          note="Damani et al. 2507.16806 Table 1b"),
    Claim("ext-2507.16806-rlvr-probe-auroc", "2507.16806 Tbl1b: RLVR+Probe Math AUROC 0.65",
          "", [], 0.65, labels="external_anchor", kind="external",
          note="Damani et al. 2507.16806 Table 1b — F-2 comparison anchor"),

    # 2509.24202 — LC+ (Logit Confidence Plus)
    Claim("ext-2509.24202-lcplus-min", "2509.24202 LC+ AUROC 0.7591 (SimpleQA, gpt-5-mini)",
          "", [], 0.7591, labels="external_anchor", kind="external",
          note="2509.24202 Table 2 — lower bound of reported range"),
    Claim("ext-2509.24202-lcplus-max", "2509.24202 LC+ AUROC 0.8083 (SimpleQA, gpt-5-mini)",
          "", [], 0.8083, labels="external_anchor", kind="external",
          note="2509.24202 Table 2 — upper bound of reported range"),
    Claim("ext-2509.24202-lc-sft-auroc", "2509.24202 LC (SFT) AUROC 0.7331 (NQ-Open Qwen3-8B)",
          "", [], 0.7331, labels="external_anchor", kind="external",
          note="2509.24202 Table 4"),

    # 2509.24248 — SpecExit
    Claim("ext-2509.24248-genlen-reduction", "2509.24248 SpecExit: 0.66 generation-length reduction (MATH-500)",
          "", [], 0.66, labels="external_anchor", kind="external",
          note="2509.24248 Table 1, Qwen3-4B-Thinking-2507"),
    Claim("ext-2509.24248-speedup", "2509.24248 SpecExit: 2.5× E2E latency speedup vs EAGLE3",
          "", [], 2.5, labels="external_anchor", kind="external",
          note="2509.24248"),

    # 2510.18147 — F-8 stability under GRPO
    Claim("ext-2510.18147-rho", "2510.18147: ρ = 0.88 (geometry↔correctness alignment)",
          "", [], 0.88, labels="external_anchor", kind="external",
          note="2510.18147"),
    Claim("ext-2510.18147-grpo-from", "2510.18147: pre-GRPO accuracy 64.7%",
          "", [], 64.7, labels="external_anchor", kind="external",
          note="2510.18147"),
    Claim("ext-2510.18147-grpo-to", "2510.18147: post-GRPO accuracy 76.2%",
          "", [], 76.2, labels="external_anchor", kind="external",
          note="2510.18147"),
    Claim("ext-2510.18147-beta-pos", "2510.18147: β = +6.66",
          "", [], 6.66, labels="external_anchor", kind="external",
          note="2510.18147"),
    Claim("ext-2510.18147-beta-neg", "2510.18147: β = -0.63",
          "", [], -0.63, labels="external_anchor", kind="external",
          note="2510.18147"),

    # 2601.19375 — SS HarmBench / Sparse Safety
    Claim("ext-2601.19375-ss-asr-qwen", "2601.19375 SS HarmBench ASR 74.04% (Qwen2.5-1.5B)",
          "", [], 74.04, labels="external_anchor", kind="external",
          note="2601.19375"),
    Claim("ext-2601.19375-sas-asr-qwen", "2601.19375 SAS HarmBench ASR 13.46% (Qwen2.5-1.5B; SS is ~5.5×)",
          "", [], 13.46, labels="external_anchor", kind="external",
          note="2601.19375"),
    Claim("ext-2601.19375-perp-violations", "2601.19375 SS perplexity-threshold violations: 0 across 8 models",
          "", [], 0, labels="external_anchor", kind="external",
          note="2601.19375"),

    # 2604.24712 — Lopez-Vazquez paraphrase / mutation thresholds
    # Forward-looking: these become live readbacks against
    # P11-FE71/72/73 result JSONs once those experiments run.
    Claim("fwd-h71-dom-stability", "H-71 threshold: prefill L19 DoM AUROC ≥ 0.733 under LV paraphrase",
          "", [], 0.733, labels="forward_looking", kind="forward_looking",
          note="from triage-2026-04-29-2604.24712.md; ≤0.04 drop from 0.7731. Live after P11-FE71 result JSON."),
    Claim("fwd-h72-bucket-migration", "H-72 threshold: D→{A,B} migration ≥ 33%, A→D ≤ 11% under LV mutation",
          "", [], 0.33, labels="forward_looking", kind="forward_looking",
          note="from triage-2026-04-29-2604.24712.md. Live after P11-FE72 result JSON."),
    Claim("fwd-h73-namedcue-acc-gain", "H-73 threshold: K=1 accuracy on named-cue subset ≥ +2pp under LV neutralization",
          "", [], 2.0, labels="forward_looking", kind="forward_looking",
          note="from triage-2026-04-29-2604.24712.md; expressed in pp. Live after P11-FE73-adjacent result JSON."),
]


# --- Driver ------------------------------------------------------------------

def _load(path: str) -> Any:
    return json.load(open(ROOT / path))


# Cache regen-script outputs by command string — a single script that prints
# many key=value lines (e.g. recompute_pr_ratios.py emits ~20) is shared across
# all Claims that reference it, so we don't re-spawn the interpreter per claim.
_REGEN_CACHE: dict[str, dict[str, Any]] = {}


def _exec_regen(cmd: str) -> dict[str, Any]:
    """Run cmd once, return {"status": "ok"|"missing"|"exit_<n>"|"timeout"|"error",
    "values": {key: float}, "stderr": str}."""
    if cmd in _REGEN_CACHE:
        return _REGEN_CACHE[cmd]
    try:
        proc = subprocess.run(
            shlex.split(cmd), cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        out = {"status": "timeout", "values": {}, "stderr": ""}
        _REGEN_CACHE[cmd] = out
        return out
    except Exception as exc:  # noqa: BLE001
        out = {"status": f"error:{exc}", "values": {}, "stderr": ""}
        _REGEN_CACHE[cmd] = out
        return out

    values: dict[str, Any] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        try:
            values[k.strip()] = float(v.strip())
        except ValueError:
            values[k.strip()] = v.strip()  # keep raw for boolean/str compares

    if proc.returncode == 0:
        status = "ok"
    else:
        msg = (proc.stderr or proc.stdout).strip().splitlines()
        first = msg[0] if msg else ""
        status = "missing" if "MISSING_REGEN_INPUT" in first else f"exit_{proc.returncode}"

    out = {"status": status, "values": values, "stderr": proc.stderr}
    _REGEN_CACHE[cmd] = out
    return out


def _run_regen(c: Claim) -> None:
    cached = _exec_regen(c.regen)
    status = cached["status"]
    if status == "missing":
        c.regen_result = "REGEN_SKIP_NO_DATA"
        return
    if status == "timeout":
        c.regen_result = "REGEN_TIMEOUT"
        return
    if status.startswith("error:"):
        c.regen_result = f"REGEN_RUN_ERROR:{status[6:]}"
        return
    if status.startswith("exit_"):
        c.regen_result = f"REGEN_EXIT_{status[5:]}"
        return

    if c.regen_key not in cached["values"]:
        c.regen_result = f"REGEN_KEY_MISSING:{c.regen_key}"
        return
    found = cached["values"][c.regen_key]
    c.regen_actual = found
    tol = c.regen_tol or c.tol
    if isinstance(c.expected, (int, float)) and isinstance(found, (int, float)):
        if math.isclose(found, c.expected, abs_tol=tol):
            c.regen_result = "REGEN_PASS"
        else:
            c.regen_result = f"REGEN_FAIL(Δ={found - c.expected:+.5g})"
    elif c.expected == found:
        c.regen_result = "REGEN_PASS"
    else:
        c.regen_result = f"REGEN_FAIL(got={found!r})"


def _check(c: Claim) -> None:
    if c.kind == "external":
        c.result = "REGISTERED"
        return
    if c.kind == "forward_looking":
        c.result = "PENDING_FE"
        return
    full = ROOT / c.file
    if not full.exists():
        c.result = "MISSING_FILE"
        return
    try:
        data = _load(c.file)
    except Exception as exc:  # noqa: BLE001
        c.result = f"LOAD_ERROR:{exc}"
        return
    try:
        val = _drill(data, c.path)
    except (KeyError, IndexError, TypeError) as exc:
        c.result = f"KEY_MISSING:{exc}"
        return
    c.actual = val
    exp = c.expected
    if isinstance(exp, bool) or isinstance(val, bool):
        c.result = "PASS" if bool(val) == bool(exp) else "FAIL"
    elif isinstance(exp, (int, float)) and isinstance(val, (int, float)):
        if math.isclose(val, exp, abs_tol=c.tol):
            c.result = "PASS"
        else:
            c.result = f"FAIL(Δ={val - exp:+.5g})"
    elif exp == val:
        c.result = "PASS"
    else:
        c.result = f"FAIL(got={val!r})"

    if c.regen:
        _run_regen(c)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-regen", action="store_true",
                    help="Skip Tier-1 regen scripts (Tier-0 JSON readback only).")
    args, _ = ap.parse_known_args()

    if args.no_regen:
        for c in CLAIMS:
            c.regen = ""  # forces _check to skip the regen branch

    for c in CLAIMS:
        _check(c)

    # Summary counts
    pass_n = sum(1 for c in CLAIMS if c.result == "PASS")
    fail_n = sum(1 for c in CLAIMS if c.result.startswith("FAIL"))
    missing = sum(1 for c in CLAIMS
                  if c.result.startswith("MISSING") or c.result.startswith("LOAD_ERROR")
                  or c.result.startswith("KEY_MISSING"))
    registered = sum(1 for c in CLAIMS if c.result == "REGISTERED")
    pending_fe = sum(1 for c in CLAIMS if c.result == "PENDING_FE")
    tok256 = sum(1 for c in CLAIMS if c.labels.startswith("256tok"))

    # Regen tier counts (Tier-1 reproducibility, not just JSON drift)
    has_regen = [c for c in CLAIMS if c.regen]
    regen_pass = sum(1 for c in has_regen if c.regen_result == "REGEN_PASS")
    regen_fail = sum(1 for c in has_regen if c.regen_result.startswith("REGEN_FAIL"))
    regen_skip = sum(1 for c in has_regen if c.regen_result == "REGEN_SKIP_NO_DATA")
    regen_err = len(has_regen) - regen_pass - regen_fail - regen_skip

    print(f"# validate_claims.py — {len(CLAIMS)} claims")
    print(f"PASS:           {pass_n}")
    print(f"FAIL:           {fail_n}")
    print(f"MISSING/ERROR:  {missing}")
    print(f"REGISTERED:     {registered}  (external paper anchors, no project readback)")
    print(f"PENDING_FE:     {pending_fe}  (forward-looking thresholds, become live after FE result JSON lands)")
    print(f"REGEN (Tier-1): {regen_pass} pass / {regen_fail} fail / {regen_skip} skip-no-data"
          f" / {regen_err} other (of {len(has_regen)} annotated)")
    print(f"(of which: {tok256} use the superseded 256-tok label scheme and should be re-baselined against 1024-tok if cited as current)\n")

    hdr = ("ID", "labels", "result", "regen", "expected", "actual", "description", "file")
    widths = (28, 18, 22, 22, 14, 16, 48, 60)
    line = "  ".join(h.ljust(w) for h, w in zip(hdr, widths))
    print(line)
    print("-" * len(line))

    for c in CLAIMS:
        actual = c.actual
        if isinstance(actual, float):
            actual_s = f"{actual:.6g}"
        else:
            actual_s = str(actual)[:16]
        exp = c.expected
        if isinstance(exp, float):
            exp_s = f"{exp:.6g}"
        else:
            exp_s = str(exp)[:14]
        desc = c.description[:48]
        path_s = c.file[:60]
        regen_s = (c.regen_result or "—")[:22]
        row = (c.cid[:28], c.labels[:18], c.result[:22], regen_s, exp_s[:14],
               actual_s[:16], desc, path_s)
        print("  ".join(r.ljust(w) for r, w in zip(row, widths)))

    # Detail section for failures and notes. REGISTERED / PENDING_FE are
    # bookkeeping states, not failures — exclude from the "issues" surface.
    had_notes = [c for c in CLAIMS if c.note]
    any_issue = [c for c in CLAIMS
                 if c.result not in ("PASS", "REGISTERED", "PENDING_FE")]

    if any_issue:
        print("\n## Issues requiring attention\n")
        for c in any_issue:
            print(f"- [{c.cid}] {c.result}: {c.description}")
            print(f"    file: {c.file}")
            print(f"    path: {c.path}")
            print(f"    expected: {c.expected!r}  actual: {c.actual!r}")
            if c.note:
                print(f"    note: {c.note}")

    # Regen issue surface — divergence between Tier-0 (json) and Tier-1 (regen)
    # is the high-signal failure mode this whole layer was added to catch.
    regen_issue = [c for c in has_regen if c.regen_result.startswith("REGEN_FAIL")
                   or c.regen_result.startswith("REGEN_EXIT")
                   or c.regen_result.startswith("REGEN_PARSE")
                   or c.regen_result.startswith("REGEN_KEY")
                   or c.regen_result == "REGEN_TIMEOUT"]
    if regen_issue:
        print("\n## Tier-1 regen issues (committed JSON ↔ recomputation drift)\n")
        for c in regen_issue:
            print(f"- [{c.cid}] {c.regen_result}: {c.description}")
            print(f"    regen: {c.regen}")
            print(f"    expected: {c.expected!r}  regen_actual: {c.regen_actual!r}")

    if had_notes:
        print("\n## Claims with superseded/outdated notes\n")
        for c in had_notes:
            print(f"- [{c.cid}] labels={c.labels}: {c.note}")

    # Tier-1 regen failures are real failures (committed-JSON ↔ regen drift).
    # SKIP_NO_DATA is advisory and does not flip exit code.
    regen_hard_fail = sum(1 for c in has_regen if c.regen_result.startswith("REGEN_FAIL")
                          or c.regen_result.startswith("REGEN_EXIT")
                          or c.regen_result.startswith("REGEN_PARSE")
                          or c.regen_result.startswith("REGEN_KEY")
                          or c.regen_result == "REGEN_TIMEOUT")
    return 0 if fail_n == 0 and missing == 0 and regen_hard_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
