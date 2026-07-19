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

from quantica.portfolio.black_litterman import black_litterman
from quantica.portfolio.construction import (
    PortfolioConstraints,
    mean_variance_weights,
    minimum_variance_weights,
    risk_parity_weights,
)
from quantica.portfolio.hrp import hrp_weights

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.factor.estimators import CovarianceEstimator

__all__ = [
    "BlackLittermanStrategy",
    "HRPStrategy",
    "MeanVarianceStrategy",
    "MinimumVarianceStrategy",
    "RiskParityStrategy",
    "Signal",
    "ViewGenerator",
    "absolute_mean_views",
    "historical_mean_signal",
]


class Signal(Protocol):
    """Maps a ``(train, n)`` return window to an expected-return vector ``mu``."""

    def __call__(self, asset_returns: FloatArray) -> FloatArray: ...


class ViewGenerator(Protocol):
    """Maps a training window to Black--Litterman views ``(P, Q)``."""

    def __call__(self, asset_returns: FloatArray) -> tuple[FloatArray, FloatArray]: ...


def historical_mean_signal(asset_returns: FloatArray) -> FloatArray:
    """The trailing sample mean return per asset — the simplest alpha signal."""
    return np.asarray(np.mean(np.asarray(asset_returns, dtype=np.float64), axis=0))


def absolute_mean_views(asset_returns: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Absolute Black--Litterman views: the trailing sample mean on every asset.

    Returns an identity picking matrix (one view per asset) and the sample-mean target
    returns. Blending these noisy estimates toward the market equilibrium is exactly how
    Black--Litterman stabilises the mean-variance weights.

    Parameters
    ----------
    asset_returns : ndarray, shape (train, n)
        The trailing training-window asset returns.

    Returns
    -------
    tuple of ndarray
        ``(P, Q)`` — the ``(n, n)`` identity picking matrix and the ``(n,)`` sample-mean
        view targets.
    """
    r = np.asarray(asset_returns, dtype=np.float64)
    n = r.shape[1]
    return np.eye(n), historical_mean_signal(r)


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
        """Estimate the covariance and return the minimum-variance target weights.

        Parameters
        ----------
        asset_returns : ndarray, shape (train, n)
            The trailing training-window asset returns.
        factor_returns : ndarray or None
            Aligned factor returns, forwarded to a factor-model estimator (ignored by
            the sample and shrinkage estimators).
        w_prev : ndarray or None
            Current holdings, used only by a turnover constraint.

        Returns
        -------
        ndarray, shape (n,)
            The target portfolio weights.
        """
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
        """Estimate the covariance and return the equal-risk-contribution weights.

        Parameters
        ----------
        asset_returns : ndarray, shape (train, n)
            The trailing training-window asset returns.
        factor_returns : ndarray or None
            Aligned factor returns, forwarded to a factor-model estimator.
        w_prev : ndarray or None
            Ignored — risk parity is long-only and fully invested by construction, so
            it takes no turnover constraint (accepted for interface compatibility).

        Returns
        -------
        ndarray, shape (n,)
            The equal-risk-contribution target weights.
        """
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
        """Estimate the covariance and alpha signal and return the mean-variance weights.

        Parameters
        ----------
        asset_returns : ndarray, shape (train, n)
            The trailing training-window asset returns (fed to both the covariance
            estimator and the alpha :attr:`signal`).
        factor_returns : ndarray or None
            Aligned factor returns, forwarded to a factor-model estimator.
        w_prev : ndarray or None
            Current holdings, used only by a turnover constraint.

        Returns
        -------
        ndarray, shape (n,)
            The Markowitz mean-variance target weights.
        """
        cov = self.estimator.estimate(asset_returns, factor_returns)
        mu = self.signal(asset_returns)
        return mean_variance_weights(mu, cov, self.risk_aversion, self.constraints, w_prev)


@dataclass(frozen=True)
class HRPStrategy:
    """Hierarchical Risk Parity construction from an estimated covariance."""

    estimator: CovarianceEstimator
    linkage_method: str = "single"
    name: str = "hrp"

    def target_weights(
        self,
        asset_returns: FloatArray,
        factor_returns: FloatArray | None,
        w_prev: FloatArray | None,
    ) -> FloatArray:
        """Estimate the covariance and return the HRP weights (no matrix inversion).

        Parameters
        ----------
        asset_returns : ndarray, shape (train, n)
            The trailing training-window asset returns.
        factor_returns : ndarray or None
            Aligned factor returns, forwarded to a factor-model estimator.
        w_prev : ndarray or None
            Ignored — HRP is long-only and fully invested by construction (accepted for
            interface compatibility).

        Returns
        -------
        ndarray, shape (n,)
            The Hierarchical Risk Parity target weights.
        """
        cov = self.estimator.estimate(asset_returns, factor_returns)
        return hrp_weights(cov, self.linkage_method)


@dataclass(frozen=True)
class BlackLittermanStrategy:
    """Black--Litterman construction: equilibrium + views → posterior → mean-variance.

    Reverse-optimises the ``market_weights`` benchmark into equilibrium returns, blends
    them with the views from :attr:`views`, and feeds the posterior into mean-variance.
    ``market_weights`` defaults to equal weight (a neutral benchmark when no market-cap
    data is available); :attr:`views` defaults to :func:`absolute_mean_views`, so the
    strategy shrinks the noisy sample mean toward equilibrium.
    """

    estimator: CovarianceEstimator
    risk_aversion: float
    views: ViewGenerator = absolute_mean_views
    market_weights: FloatArray | None = None
    tau: float = 0.05
    constraints: PortfolioConstraints = field(default_factory=PortfolioConstraints)
    name: str = "black-litterman"

    def target_weights(
        self,
        asset_returns: FloatArray,
        factor_returns: FloatArray | None,
        w_prev: FloatArray | None,
    ) -> FloatArray:
        """Estimate the covariance, form the BL posterior, and mean-variance optimise.

        Parameters
        ----------
        asset_returns : ndarray, shape (train, n)
            The trailing training-window asset returns (drive both the covariance
            estimate and the views).
        factor_returns : ndarray or None
            Aligned factor returns, forwarded to a factor-model estimator.
        w_prev : ndarray or None
            Current holdings, used only by a turnover constraint.

        Returns
        -------
        ndarray, shape (n,)
            The Black--Litterman mean-variance target weights.
        """
        cov = self.estimator.estimate(asset_returns, factor_returns)
        n = cov.shape[0]
        market = self.market_weights if self.market_weights is not None else np.full(n, 1.0 / n)
        views_p, views_q = self.views(asset_returns)
        posterior = black_litterman(
            cov, market, self.risk_aversion, views_p=views_p, views_q=views_q, tau=self.tau
        )
        return mean_variance_weights(
            posterior.posterior_returns,
            posterior.posterior_cov,
            self.risk_aversion,
            self.constraints,
            w_prev,
        )
