r"""Explainability checks — validating the explainer, not just running it.

SHAP output is usually presented as unfalsifiable colour: a bar chart of "drivers"
nobody can dispute. This module turns it into testable claims:

* :func:`check_local_accuracy` — SHAP's own **local-accuracy** axiom: per row the
  attributions must sum to the model output minus the base value,
  :math:`\sum_j \phi_{ij} = f(x_i) - \mathbb E[f]`. An explainer that fails this
  is broken *by its own definition*; asserting it (to tight tolerance) is the
  validate-the-validator move applied to the explainer.
* :func:`global_importance` / :func:`driver_recovery` — mean-|SHAP| importances,
  compared against a **known** driver ranking. On synthetic data the
  data-generating process is known, so "the explainer identified the drivers" is
  a checkable statement, not a narrative.
* :func:`rank_stability` — Spearman correlation of importance rankings across
  perturbed replications (subsamples or refits). Explanations that reorder under
  small perturbations cannot support governance use.
* :func:`attribution_direction` — the per-feature correlation between a
  feature's SHAP column and the feature itself. Sign checks against the known
  coefficients; a *weak* correlation on a truly nonlinear driver is itself
  informative (a U-shaped effect attenuates the linear correlation).

All functions consume SHAP **matrices** (and predictions) — never the model or
the explainer object — so the package stays dependency-lean and the checks apply
to any attribution method satisfying the same additivity contract.

References
----------
Lundberg & Lee (2017), "A unified approach to interpreting model predictions";
Fed/OCC SR 11-7 (2011), *Supervisory Guidance on Model Risk Management*.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from quantica.core.types import FloatArray

__all__ = [
    "DriverRecovery",
    "FeatureImportance",
    "LocalAccuracy",
    "RankStability",
    "attribution_direction",
    "check_local_accuracy",
    "driver_recovery",
    "global_importance",
    "rank_stability",
]

# Local accuracy should hold to numerical noise for exact explainers (TreeSHAP,
# LinearSHAP); the default tolerance leaves room only for float accumulation.
_DEFAULT_LOCAL_ACCURACY_TOL = 1e-8


@dataclass(frozen=True)
class LocalAccuracy:
    """Local-accuracy (additivity) check outcome for a SHAP matrix."""

    max_abs_error: float
    mean_abs_error: float
    n_rows: int
    tol: float

    @property
    def passed(self) -> bool:
        """Whether every row satisfies additivity within the tolerance."""
        return self.max_abs_error <= self.tol


def check_local_accuracy(
    shap_values: FloatArray,
    base_value: float,
    predictions: FloatArray,
    *,
    tol: float = _DEFAULT_LOCAL_ACCURACY_TOL,
) -> LocalAccuracy:
    r"""Verify :math:`\sum_j \phi_{ij} + \phi_0 = f(x_i)` row by row.

    ``predictions`` must be on the same output scale the explainer used (for
    tree/linear classifiers typically the **log-odds margin**, not the
    probability — passing probabilities against a log-odds explainer is the
    classic silent mistake this check catches).
    """
    sv = np.asarray(shap_values, dtype=np.float64)
    preds = np.asarray(predictions, dtype=np.float64)
    if sv.ndim != 2 or sv.shape[0] == 0:
        raise ValueError("shap_values must be a non-empty (n, k) matrix")
    if preds.shape != (sv.shape[0],):
        raise ValueError(f"predictions shape {preds.shape} must be ({sv.shape[0]},)")
    errors = np.abs(sv.sum(axis=1) + base_value - preds)
    return LocalAccuracy(
        max_abs_error=float(errors.max()),
        mean_abs_error=float(errors.mean()),
        n_rows=int(sv.shape[0]),
        tol=tol,
    )


@dataclass(frozen=True)
class FeatureImportance:
    """One feature's global importance (mean |SHAP|), with its rank (0 = top)."""

    name: str
    importance: float
    rank: int


def global_importance(
    shap_values: FloatArray, feature_names: tuple[str, ...]
) -> tuple[FeatureImportance, ...]:
    """Mean-|SHAP| global importances, sorted descending."""
    sv = np.asarray(shap_values, dtype=np.float64)
    if sv.ndim != 2 or sv.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values must be (n, {len(feature_names)}) to match feature_names, got {sv.shape}"
        )
    magnitudes = np.abs(sv).mean(axis=0)
    order = np.argsort(-magnitudes)
    return tuple(
        FeatureImportance(name=feature_names[j], importance=float(magnitudes[j]), rank=r)
        for r, j in enumerate(order)
    )


@dataclass(frozen=True)
class DriverRecovery:
    """Observed vs expected driver ranking — the known-truth explainability check."""

    observed: tuple[str, ...]
    expected: tuple[str, ...]
    exact_match: bool
    n_top_matched: int  # length of the matching prefix


def driver_recovery(
    importances: tuple[FeatureImportance, ...], expected_ranking: tuple[str, ...]
) -> DriverRecovery:
    """Compare the importance ranking against a known driver order.

    ``expected_ranking`` is the ground-truth order (strongest first) implied by
    the data-generating process — available for synthetic data, which is exactly
    what makes this a *verifiable* explainability claim.
    """
    observed = tuple(fi.name for fi in importances)
    if set(expected_ranking) - set(observed):
        raise ValueError("expected_ranking contains names not present in importances")
    prefix = 0
    for got, want in zip(observed, expected_ranking, strict=False):
        if got != want:
            break
        prefix += 1
    return DriverRecovery(
        observed=observed,
        expected=expected_ranking,
        exact_match=observed[: len(expected_ranking)] == expected_ranking,
        n_top_matched=prefix,
    )


@dataclass(frozen=True)
class RankStability:
    """Stability of importance rankings across perturbed replications."""

    mean_spearman: float
    min_spearman: float
    n_replications: int


def rank_stability(importance_vectors: FloatArray) -> RankStability:
    """Pairwise Spearman rank correlation of importances across replications.

    ``importance_vectors`` is ``(B, k)``: one importance vector per replication
    (subsample of the explanation set, or a bootstrap refit — the caller decides
    which stability question is being asked; refits are the stronger check).
    """
    v = np.asarray(importance_vectors, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 2 or v.shape[1] < 2:
        raise ValueError("importance_vectors must be (B >= 2, k >= 2)")
    rhos = [
        float(spearmanr(v[a], v[b]).statistic)
        for a in range(v.shape[0])
        for b in range(a + 1, v.shape[0])
    ]
    return RankStability(
        mean_spearman=float(np.mean(rhos)),
        min_spearman=float(np.min(rhos)),
        n_replications=int(v.shape[0]),
    )


def attribution_direction(shap_values: FloatArray, features: FloatArray) -> FloatArray:
    """Per-feature Pearson correlation between SHAP column and feature column.

    The sign should match the known effect direction; the magnitude is
    informative too — a genuinely nonlinear (e.g. U-shaped) driver shows a
    *weaker* linear correlation than a monotone one, which is the correct
    reading, not a defect.
    """
    sv = np.asarray(shap_values, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    if sv.shape != x.shape or sv.ndim != 2:
        raise ValueError(f"shap_values {sv.shape} and features {x.shape} must match (n, k)")
    out = np.empty(sv.shape[1], dtype=np.float64)
    for j in range(sv.shape[1]):
        out[j] = float(np.corrcoef(sv[:, j], x[:, j])[0, 1])
    return out
