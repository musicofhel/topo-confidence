"""Preflight — prompt-length pre-generation signal (Tier-A).

A logistic regression on the single scalar [n_prompt_tokens], usable BEFORE
any generation happens (route hopeless prompts straight to the large model).

Pinned anchor (results/phase2_arm2a.json, SPEC v6 arm-2A, Qwen2.5-1.5B
MATH-500, frozen-fold OOF): prompt_length_only AUROC 0.7056. The prompt-cloud
eigenspectrum did NOT add over prompt-length (gate delta -0.0096, p=0.42,
H-E refuted) — length alone is the whole free Tier-A signal.

Pinned deployment coefficients are fit on the full 500-problem cell
(prompt_len from cache/arm2a_prompt_cloud_qwen1.5b_math.npz, labels from
nocompute/cache/math500_1p5b.npz — same problems, same order, verified by
phase2_arm2a.py which pairs exactly these two artifacts).
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Sequence

import numpy as np

_RECIPE = "StandardScaler -> LogisticRegression(max_iter=2000, C=1.0) on [n_prompt_tokens]"


class PreflightGate:
    """P(correct | n_prompt_tokens) — pre-generation routing signal."""

    def __init__(self, scaler_mean: float, scaler_scale: float,
                 coef: float, intercept: float,
                 family: str | None = None, meta: dict | None = None):
        self.scaler_mean = float(scaler_mean)
        self.scaler_scale = float(scaler_scale)
        self.coef = float(coef)
        self.intercept = float(intercept)
        self.family = family
        self.meta = dict(meta or {})

    @classmethod
    def from_pinned(cls, family: str = "qwen2.5-1.5b") -> "PreflightGate":
        with resources.files("confgate").joinpath("data", "pinned_gates.json").open("r") as f:
            gates = json.load(f)
        pf = gates.get("preflight", {})
        if family not in pf:
            raise KeyError(f"No pinned preflight gate for {family!r}; pinned: {sorted(pf)}")
        g = pf[family]
        return cls(g["scaler_mean"], g["scaler_scale"], g["coef"], g["intercept"],
                   family=family, meta={k: v for k, v in g.items()
                                        if k not in ("scaler_mean", "scaler_scale",
                                                     "coef", "intercept")})

    @classmethod
    def fit(cls, prompt_lengths: Sequence[float], y,
            family: str | None = None) -> "PreflightGate":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(prompt_lengths, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(y).astype(int)
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)
        return cls(sc.mean_[0], sc.scale_[0], lr.coef_[0][0], lr.intercept_[0],
                   family=family, meta={"n": int(len(y)), "recipe": _RECIPE})

    def score(self, prompt_lengths) -> np.ndarray:
        """P(correct) per prompt, from token count alone."""
        x = np.asarray(prompt_lengths, dtype=np.float64)
        z = (x - self.scaler_mean) / self.scaler_scale
        return 1.0 / (1.0 + np.exp(-(z * self.coef + self.intercept)))

    def __repr__(self) -> str:  # pragma: no cover
        return (f"PreflightGate({self.family or 'unpinned'}, "
                f"coef={self.coef:.4f}, intercept={self.intercept:.4f})")
