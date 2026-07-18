r"""Covariance estimators, behind one interface — the three horses in the race.

Estimating the :math:`n \times n` asset covariance is the hard part of any risk or
portfolio model, and *which* estimator to trust is an empirical question the stage-2
:mod:`quantica.factor.evaluation` layer answers. This module supplies the three
contenders behind a common :class:`CovarianceEstimator` interface — **none of them
re-implemented** (CLAUDE.md §3):

* :class:`SampleCovariance` — the textbook sample covariance (``numpy.cov``). The
  maximum-likelihood-ish baseline that is unbiased but *high variance*, and
  near-singular once the asset count approaches the number of observations.
* :class:`LedoitWolfCovariance` — Ledoit--Wolf (2004) shrinkage toward a scaled
  identity, with the analytically optimal intensity, from
  ``sklearn.covariance.LedoitWolf`` (lazily imported).
* :class:`FactorCovariance` — the stage-1 factor model
  :math:`\Sigma = B F B^\top + D`, which is well-conditioned *by construction*
  (a low-rank factor part plus a positive diagonal).

The package's contribution is not the estimators but the comparison framework around
them; keeping them behind one small interface is what lets that framework treat them
uniformly (the factor estimator additionally consumes the observable factor returns,
which the others ignore).

References
----------
Ledoit, O. & Wolf, M. (2004), "A well-conditioned estimator for large-dimensional
covariance matrices", *J. Multivariate Analysis*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from quantica.factor.model import FactorRiskModel

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "CovarianceEstimator",
    "FactorCovariance",
    "LedoitWolfCovariance",
    "SampleCovariance",
    "condition_number",
    "min_variance_weights",
]


@runtime_checkable
class CovarianceEstimator(Protocol):
    """A named strategy mapping a training return panel to a covariance matrix."""

    @property
    def name(self) -> str:
        """Short label used in comparison tables."""
        ...

    def estimate(
        self, asset_returns: FloatArray, factor_returns: FloatArray | None = None
    ) -> FloatArray:
        """Return the estimated ``(n, n)`` covariance from ``(T, n)`` asset returns."""
        ...


class SampleCovariance:
    """The plain sample covariance (``numpy.cov``, unbiased ``ddof=1``)."""

    name = "sample"

    def estimate(
        self, asset_returns: FloatArray, factor_returns: FloatArray | None = None
    ) -> FloatArray:
        """Return the sample covariance of ``asset_returns`` (``numpy.cov``, ``ddof=1``).

        Parameters
        ----------
        asset_returns : ndarray, shape (T, n)
            The asset return panel (``T`` observations of ``n`` assets).
        factor_returns : ndarray, optional
            Unused by this estimator; accepted for interface compatibility with
            :class:`CovarianceEstimator`.

        Returns
        -------
        ndarray, shape (n, n)
            The unbiased sample covariance matrix.
        """
        return np.atleast_2d(np.cov(np.asarray(asset_returns, dtype=np.float64), rowvar=False))


class LedoitWolfCovariance:
    """Ledoit--Wolf shrinkage covariance from scikit-learn (analytic intensity)."""

    name = "ledoit-wolf"

    def estimate(
        self, asset_returns: FloatArray, factor_returns: FloatArray | None = None
    ) -> FloatArray:
        """Return the Ledoit--Wolf shrinkage covariance of ``asset_returns``.

        Fits ``sklearn.covariance.LedoitWolf`` (lazily imported), which shrinks the
        sample covariance toward a scaled identity with the analytically optimal
        intensity.

        Parameters
        ----------
        asset_returns : ndarray, shape (T, n)
            The asset return panel.
        factor_returns : ndarray, optional
            Unused by this estimator; accepted for interface compatibility with
            :class:`CovarianceEstimator`.

        Returns
        -------
        ndarray, shape (n, n)
            The well-conditioned shrinkage covariance matrix.
        """
        from sklearn.covariance import LedoitWolf  # lazy: sklearn import is heavy

        fitted = LedoitWolf().fit(np.asarray(asset_returns, dtype=np.float64))
        return np.asarray(fitted.covariance_, dtype=np.float64)


class FactorCovariance:
    """The stage-1 factor-model covariance :math:`B F B^\\top + D`.

    Requires the aligned observable factor returns; raises if they are absent.
    """

    name = "factor"

    def __init__(self, factor_names: tuple[str, ...] | None = None) -> None:
        self.factor_names = factor_names

    def estimate(
        self, asset_returns: FloatArray, factor_returns: FloatArray | None = None
    ) -> FloatArray:
        r"""Return the factor-model covariance :math:`B F B^\top + D`.

        Fits the stage-1 :class:`~quantica.factor.model.FactorRiskModel` on the aligned
        observable factor returns and returns its assembled covariance, which is
        well-conditioned by construction (a low-rank factor part plus a positive
        diagonal).

        Parameters
        ----------
        asset_returns : ndarray, shape (T, n)
            The asset return panel.
        factor_returns : ndarray, shape (T, k)
            The aligned observable factor returns (required by this estimator).

        Returns
        -------
        ndarray, shape (n, n)
            The factor-model covariance matrix.

        Raises
        ------
        ValueError
            If ``factor_returns`` is ``None``.
        """
        if factor_returns is None:
            raise ValueError("FactorCovariance requires factor_returns")
        model = FactorRiskModel.fit(asset_returns, factor_returns, factor_names=self.factor_names)
        return model.covariance()


def condition_number(cov: FloatArray) -> float:
    """The 2-norm condition number :math:`\\lambda_{\\max}/\\lambda_{\\min}` of ``cov``.

    Large values mean near-singularity: inverting the matrix (as portfolio
    optimisation does) amplifies estimation error. ``inf`` for a singular matrix.
    """
    return float(np.linalg.cond(np.asarray(cov, dtype=np.float64)))


def min_variance_weights(cov: FloatArray) -> FloatArray:
    r"""Global minimum-variance weights :math:`w \propto \Sigma^{-1}\mathbf 1` (renormalised).

    The classic long-short GMV portfolio (weights sum to one, shorts allowed). It
    inverts the covariance, so it is *maximally* sensitive to estimation error —
    Michaud's "error maximiser" — which is exactly why it is the sharp test of an
    estimator's quality. A near-singular ``cov`` yields wild weights (the failure
    mode we mean to expose), not an exception.
    """
    sigma = np.asarray(cov, dtype=np.float64)
    n = sigma.shape[0]
    ones = np.ones(n)
    z = np.linalg.solve(sigma, ones)
    return np.asarray(z / z.sum(), dtype=np.float64)
