r"""Strategies — the glue binding a covariance estimator to a construction rule.

A :class:`~quantica.portfolio.backtest.Strategy` maps a trailing training window to
target weights; these classes implement that seam by combining a
:class:`~quantica.factor.estimators.CovarianceEstimator` (so the stage-2 estimator
comparison plugs straight into the backtest) with one of the constructors in
:mod:`quantica.portfolio.construction`.

Mean-variance additionally needs an expected-return **signal** — a callable mapping
the training returns to an alpha vector :math:`\mu`. The default is the trailing
sample mean, but the *point* of a signal seam is that a deliberately-overfit search
over many candidate signals (the backtest-validity headline) can be run through the
same machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from quantica.portfolio.construction import (
    PortfolioConstraints,
    mean_variance_weights,
    minimum_variance_weights,
    risk_parity_weights,
)

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.factor.estimators import CovarianceEstimator

__all__ = [
    "MeanVarianceStrategy",
    "MinimumVarianceStrategy",
    "RiskParityStrategy",
    "Signal",
    "historical_mean_signal",
]


class Signal(Protocol):
    """Maps a ``(train, n)`` return window to an expected-return vector ``mu``."""

    def __call__(self, asset_returns: FloatArray) -> FloatArray: ...


def historical_mean_signal(asset_returns: FloatArray) -> FloatArray:
    """The trailing sample mean return per asset — the simplest alpha signal."""
    return np.asarray(np.mean(np.asarray(asset_returns, dtype=np.float64), axis=0))


@dataclass(frozen=True)
class MinimumVarianceStrategy:
    """Minimum-variance construction from an estimated covariance."""

    estimator: CovarianceEstimator
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    name: str = "min-variance"

    def target_weights(
        self,
        asset_returns: FloatArray,
        factor_returns: FloatArray | None,
        w_prev: FloatArray | None,
    ) -> FloatArray:
        cov = self.estimator.estimate(asset_returns, factor_returns)
        return minimum_variance_weights(cov, self.constraints, w_prev)


@dataclass(frozen=True)
class RiskParityStrategy:
    """Equal-risk-contribution construction from an estimated covariance."""

    estimator: CovarianceEstimator
    name: str = "risk-parity"

    def target_weights(
        self,
        asset_returns: FloatArray,
        factor_returns: FloatArray | None,
        w_prev: FloatArray | None,
    ) -> FloatArray:
        cov = self.estimator.estimate(asset_returns, factor_returns)
        return risk_parity_weights(cov)


@dataclass(frozen=True)
class MeanVarianceStrategy:
    """Markowitz construction from an estimated covariance and an alpha signal."""

    estimator: CovarianceEstimator
    risk_aversion: float
    signal: Signal = historical_mean_signal
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    name: str = "mean-variance"

    def target_weights(
        self,
        asset_returns: FloatArray,
        factor_returns: FloatArray | None,
        w_prev: FloatArray | None,
    ) -> FloatArray:
        cov = self.estimator.estimate(asset_returns, factor_returns)
        mu = self.signal(asset_returns)
        return mean_variance_weights(mu, cov, self.risk_aversion, self.constraints, w_prev)
