r"""Stability — has the population the model scores drifted away from the one it
was built on?

A PD model validated on last year's portfolio silently degrades when the incoming
population shifts (new origination channels, macro regime changes). The standard
monitoring tool is the **Population Stability Index**:

.. math::

    \mathrm{PSI} = \sum_b (p^{\text{actual}}_b - p^{\text{expected}}_b)\,
                   \ln\!\frac{p^{\text{actual}}_b}{p^{\text{expected}}_b},

the symmetrised KL divergence between the binned development ("expected") and
monitoring ("actual") distributions. Bins are the expected sample's quantiles
(the industry convention: the development sample defines the yardstick). Applied
to the model *score* it flags overall drift; applied feature-by-feature
(:func:`characteristic_stability`, often called CSI) it points at *which* input
moved.

The 0.10 / 0.25 thresholds ("stable" / "monitor" / "shifted") are the widely used
industry rule of thumb — a convention, not a distributional result, and labelled
as such.

References
----------
Basel Committee WP 14 (2005); Siddiqi, *Credit Risk Scorecards* (2006).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from quantica.core.types import FloatArray

__all__ = [
    "CharacteristicStability",
    "PSIResult",
    "StabilityBand",
    "characteristic_stability",
    "psi",
]

_DEFAULT_N_BINS = 10
# Industry rule-of-thumb thresholds (convention, not a distributional result).
_PSI_STABLE = 0.10
_PSI_MONITOR = 0.25
# Floor on a bin proportion so empty bins contribute a large-but-finite term
# instead of an infinity; documented, and only binding under extreme drift.
_PROPORTION_FLOOR = 1e-6


class StabilityBand(Enum):
    """The conventional PSI interpretation bands."""

    STABLE = "stable"
    MONITOR = "monitor"
    SHIFTED = "shifted"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class PSIResult:
    """Population Stability Index outcome for one variable."""

    value: float
    band: StabilityBand
    n_bins: int


def psi(
    expected: FloatArray,
    actual: FloatArray,
    *,
    n_bins: int = _DEFAULT_N_BINS,
) -> PSIResult:
    """Population Stability Index of ``actual`` against ``expected``.

    Bin edges are the quantiles of the *expected* (development) sample — the
    monitoring sample is measured against the development yardstick. Tied edges
    are merged; each sample's bin proportions are floored at ``1e-6`` so an
    emptied bin contributes a large finite term rather than an infinity.
    """
    e = np.asarray(expected, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    if e.ndim != 1 or e.size == 0 or a.ndim != 1 or a.size == 0:
        raise ValueError("expected and actual must be non-empty 1-D arrays")
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    edges = np.unique(np.quantile(e, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]))
    e_bins = np.digitize(e, edges, right=True)
    a_bins = np.digitize(a, edges, right=True)
    k = edges.size + 1
    p_e = np.maximum(np.bincount(e_bins, minlength=k) / e.size, _PROPORTION_FLOOR)
    p_a = np.maximum(np.bincount(a_bins, minlength=k) / a.size, _PROPORTION_FLOOR)
    value = float(np.sum((p_a - p_e) * np.log(p_a / p_e)))
    return PSIResult(value=value, band=_band(value), n_bins=k)


def _band(value: float) -> StabilityBand:
    if value < _PSI_STABLE:
        return StabilityBand.STABLE
    if value < _PSI_MONITOR:
        return StabilityBand.MONITOR
    return StabilityBand.SHIFTED


@dataclass(frozen=True)
class CharacteristicStability:
    """Per-feature stability (CSI) row."""

    name: str
    psi: PSIResult


def characteristic_stability(
    expected_features: FloatArray,
    actual_features: FloatArray,
    feature_names: tuple[str, ...],
    *,
    n_bins: int = _DEFAULT_N_BINS,
) -> tuple[CharacteristicStability, ...]:
    """PSI per input characteristic — which feature drove the drift?

    ``expected_features`` and ``actual_features`` are ``(n, k)`` matrices over the
    same ``k`` named features (development vs monitoring samples).
    """
    e = np.asarray(expected_features, dtype=np.float64)
    a = np.asarray(actual_features, dtype=np.float64)
    if e.ndim != 2 or a.ndim != 2 or e.shape[1] != a.shape[1]:
        raise ValueError("feature matrices must be 2-D with matching column counts")
    if len(feature_names) != e.shape[1]:
        raise ValueError(f"got {len(feature_names)} names for {e.shape[1]} feature columns")
    return tuple(
        CharacteristicStability(name=name, psi=psi(e[:, j], a[:, j], n_bins=n_bins))
        for j, name in enumerate(feature_names)
    )
