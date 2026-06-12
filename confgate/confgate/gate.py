"""FreeGate — logistic regression on (n_gen_tokens, mean_logprob).

Recipe (EXACT parity with the pinned v6 evaluation code,
pathway11_h100/generalization_edge/probes.py::free_baseline and
phase3_heldout.py::eval_free_baseline):

    StandardScaler()  ->  LogisticRegression(max_iter=2000, C=1.0)

Features, in order: [n_gen_tokens, mean_logprob].

Pinned coefficients (data/pinned_gates.json) are DEPLOYMENT coefficients:
fit on the full per-family cache, no cross-validation. The honest
generalization numbers are the OOF/LOCO anchors recorded alongside them in
data/pinned_meta.json — resubstitution AUROCs stored here are a regression
check, not a performance claim.
"""
from __future__ import annotations

import json
from importlib import resources
from typing import Sequence

import numpy as np

FEATURES = ("n_gen_tokens", "mean_logprob")
_RECIPE = "StandardScaler -> LogisticRegression(max_iter=2000, C=1.0)"


def _load_data_json(name: str) -> dict:
    with resources.files("confgate").joinpath("data", name).open("r") as f:
        return json.load(f)


def pinned_families() -> list[str]:
    """Names of the families with pinned deployment coefficients."""
    return sorted(_load_data_json("pinned_gates.json")["families"].keys())


def pinned_meta() -> dict:
    """Provenance + pinned v6 anchors (source artifacts, dates, OOF AUROCs)."""
    return _load_data_json("pinned_meta.json")


class FreeGate:
    """Logistic gate on two free scalars of a greedy generation.

    Score = P(correct | n_gen_tokens, mean_logprob). Inference is pure numpy
    (scaler + sigmoid), so a pinned gate has zero sklearn runtime dependency.
    """

    def __init__(self, scaler_mean: Sequence[float], scaler_scale: Sequence[float],
                 coef: Sequence[float], intercept: float,
                 family: str | None = None, meta: dict | None = None):
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.family = family
        self.meta = dict(meta or {})
        if self.scaler_mean.shape != (2,) or self.coef.shape != (2,):
            raise ValueError("FreeGate is a 2-feature gate: [n_gen_tokens, mean_logprob]")

    # ------------------------------------------------------------------ ctor
    @classmethod
    def from_pinned(cls, family: str) -> "FreeGate":
        """Load pinned deployment coefficients for one of the 5 families."""
        gates = _load_data_json("pinned_gates.json")
        fams = gates["families"]
        if family not in fams:
            raise KeyError(f"Unknown family {family!r}; pinned: {sorted(fams)}")
        g = fams[family]
        return cls(g["scaler_mean"], g["scaler_scale"], g["coef"], g["intercept"],
                   family=family, meta={k: v for k, v in g.items()
                                        if k not in ("scaler_mean", "scaler_scale",
                                                     "coef", "intercept")})

    @classmethod
    def fit(cls, lengths, logprobs, y, family: str | None = None) -> "FreeGate":
        """Fit a fresh gate (full-data; this is how the pins were produced)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        X = np.column_stack([np.asarray(lengths, dtype=np.float64),
                             np.asarray(logprobs, dtype=np.float64)])
        y = np.asarray(y).astype(int)
        sc = StandardScaler().fit(X)
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)
        return cls(sc.mean_, sc.scale_, lr.coef_[0], lr.intercept_[0],
                   family=family, meta={"n": int(len(y)), "recipe": _RECIPE})

    # ----------------------------------------------------------------- score
    def score(self, lengths, logprobs) -> np.ndarray:
        """P(correct) for each (length, logprob) pair."""
        X = np.column_stack([np.asarray(lengths, dtype=np.float64),
                             np.asarray(logprobs, dtype=np.float64)])
        z = (X - self.scaler_mean) / self.scaler_scale
        logits = z @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-logits))

    def __repr__(self) -> str:  # pragma: no cover
        fam = self.family or "unpinned"
        return (f"FreeGate({fam}, coef={np.round(self.coef, 4).tolist()}, "
                f"intercept={self.intercept:.4f})")
