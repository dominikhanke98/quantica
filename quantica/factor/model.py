r"""The multi-factor risk model — exposures, decomposition, and :math:`\Sigma`.

A factor risk model represents the :math:`n \times n` asset covariance through a
small number ``k`` of factors:

.. math:: \Sigma = B\,F\,B^\top + D,

where ``B`` (``n x k``) are the factor loadings, ``F`` (``k x k``) is the factor
covariance, and ``D`` is the diagonal of specific (idiosyncratic) variances. This
is the workhorse of both **market-risk decomposition** (split a portfolio's risk
into factor and specific pieces) and **portfolio construction** (a well-conditioned
covariance estimate when assets outnumber observations) — which is why this lives
in its own top-level :mod:`quantica.factor` package rather than under ``risk`` or a
future ``portfolio``: it is the shared foundation, consumed by both.

Estimation leans entirely on established libraries (CLAUDE.md §3): the loadings
come from :func:`~quantica.factor.exposures.estimate_exposures` (statsmodels OLS),
and the factor covariance ``F`` is the sample covariance of the factor returns
(``numpy.cov``). The package's own contribution is the *assembly* into a valid
(symmetric PSD) covariance, the risk **decomposition**, and — the real deliverable,
in stage 2 — the out-of-sample estimator-comparison layer that neither statsmodels
nor scikit-learn ships.

Assumptions and their consequences are named, not hidden: idiosyncratic returns are
taken uncorrelated across assets (``D`` diagonal), so any residual cross-correlation
is *not* captured — the standard factor-model simplification, and a natural thing
for the stage-2 out-of-sample check to probe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from quantica.core.types import FloatArray
from quantica.factor.exposures import FactorExposures, estimate_exposures

__all__ = ["AssetVarianceDecomposition", "FactorRiskModel", "PortfolioRiskDecomposition"]


class AssetVarianceDecomposition(NamedTuple):
    """One asset's variance split into systematic and specific parts."""

    asset: str
    total_variance: float
    systematic_variance: float
    specific_variance: float

    @property
    def systematic_fraction(self) -> float:
        """Share of variance explained by the factors (model-implied)."""
        return self.systematic_variance / self.total_variance


class PortfolioRiskDecomposition(NamedTuple):
    """A portfolio's variance split into systematic and specific parts."""

    total_variance: float
    systematic_variance: float
    specific_variance: float
    factor_exposure: FloatArray  # B^T w: the portfolio's net loading on each factor

    @property
    def total_volatility(self) -> float:
        """Portfolio volatility :math:`\\sqrt{w^\\top \\Sigma w}`."""
        return float(np.sqrt(self.total_variance))

    @property
    def systematic_fraction(self) -> float:
        """Share of portfolio variance coming from factor exposure."""
        return self.systematic_variance / self.total_variance


@dataclass(frozen=True)
class FactorRiskModel:
    """A fitted linear factor risk model :math:`\\Sigma = B F B^\\top + D`.

    Build with :meth:`fit`. Holds the loadings ``B`` (:attr:`betas`), the factor
    covariance ``F`` (:attr:`factor_cov`), the specific variances (the diagonal of
    ``D``), and the per-asset regression detail.
    """

    asset_names: tuple[str, ...]
    factor_names: tuple[str, ...]
    betas: FloatArray  # B: (n_assets, k)
    alphas: FloatArray  # (n_assets,)
    factor_cov: FloatArray  # F: (k, k)
    specific_var: FloatArray  # diag(D): (n_assets,)
    exposures: tuple[FactorExposures, ...]

    # --------------------------------------------------------------- fitting

    @classmethod
    def fit(
        cls,
        asset_returns: FloatArray,
        factor_returns: FloatArray,
        *,
        asset_names: tuple[str, ...] | None = None,
        factor_names: tuple[str, ...] | None = None,
    ) -> FactorRiskModel:
        r"""Estimate the model from aligned asset and factor return panels.

        Parameters
        ----------
        asset_returns : ndarray, shape (T, n_assets)
            Excess returns of the assets.
        factor_returns : ndarray, shape (T, k)
            The factor excess returns.
        asset_names, factor_names : tuple of str, optional
            Labels; default to ``A1..An`` and ``F1..Fk``.

        Notes
        -----
        Each asset is regressed independently on the factors (the loadings share a
        design matrix but the specific variances differ), and ``F`` is the sample
        covariance of the factor returns.
        """
        r = np.asarray(asset_returns, dtype=np.float64)
        f = np.asarray(factor_returns, dtype=np.float64)
        if r.ndim != 2 or f.ndim != 2:
            raise ValueError("asset_returns and factor_returns must both be 2-D")
        if r.shape[0] != f.shape[0]:
            raise ValueError(
                f"time dimension mismatch: assets T={r.shape[0]}, factors T={f.shape[0]}"
            )
        n_assets, k = r.shape[1], f.shape[1]
        factor_names = factor_names or tuple(f"F{j + 1}" for j in range(k))
        asset_names = asset_names or tuple(f"A{i + 1}" for i in range(n_assets))
        if len(factor_names) != k:
            raise ValueError(f"{len(factor_names)} factor_names for {k} factors")
        if len(asset_names) != n_assets:
            raise ValueError(f"{len(asset_names)} asset_names for {n_assets} assets")

        exposures = tuple(estimate_exposures(r[:, i], f, factor_names) for i in range(n_assets))
        betas = np.array([e.betas for e in exposures], dtype=np.float64)
        alphas = np.array([e.alpha for e in exposures], dtype=np.float64)
        specific_var = np.array([e.specific_variance for e in exposures], dtype=np.float64)
        factor_cov = np.atleast_2d(np.cov(f, rowvar=False, ddof=1))
        return cls(
            asset_names=asset_names,
            factor_names=factor_names,
            betas=betas,
            alphas=alphas,
            factor_cov=factor_cov,
            specific_var=specific_var,
            exposures=exposures,
        )

    # --------------------------------------------------------------- covariance

    def systematic_covariance(self) -> FloatArray:
        """The factor-driven covariance :math:`B F B^\\top`."""
        return np.asarray(self.betas @ self.factor_cov @ self.betas.T, dtype=np.float64)

    def covariance(self) -> FloatArray:
        r"""The assembled asset covariance :math:`\Sigma = B F B^\top + D`.

        Symmetrised against floating-point asymmetry; PSD by construction (``F`` is
        a sample covariance, so ``B F B^\top`` is PSD, and ``D`` is non-negative
        diagonal).
        """
        sigma = self.systematic_covariance()
        sigma = sigma + np.diag(self.specific_var)
        return np.asarray(0.5 * (sigma + sigma.T), dtype=np.float64)

    # --------------------------------------------------------------- decomposition

    def variance_decomposition(self) -> tuple[AssetVarianceDecomposition, ...]:
        """Per-asset split of total variance into systematic and specific parts."""
        systematic = np.diag(self.systematic_covariance())
        return tuple(
            AssetVarianceDecomposition(
                asset=name,
                total_variance=float(systematic[i] + self.specific_var[i]),
                systematic_variance=float(systematic[i]),
                specific_variance=float(self.specific_var[i]),
            )
            for i, name in enumerate(self.asset_names)
        )

    def portfolio_factor_exposure(self, weights: FloatArray) -> FloatArray:
        r"""The portfolio's net factor loadings :math:`B^\top w`."""
        w = self._check_weights(weights)
        return np.asarray(self.betas.T @ w, dtype=np.float64)

    def portfolio_variance(self, weights: FloatArray) -> float:
        r"""Portfolio variance :math:`w^\top \Sigma w`."""
        w = self._check_weights(weights)
        return float(w @ self.covariance() @ w)

    def portfolio_risk_decomposition(self, weights: FloatArray) -> PortfolioRiskDecomposition:
        r"""Split portfolio variance into systematic (:math:`(B^\top w)^\top F (B^\top w)`)
        and specific (:math:`\sum_i w_i^2 D_i`) parts.
        """
        w = self._check_weights(weights)
        exposure = self.betas.T @ w
        systematic = float(exposure @ self.factor_cov @ exposure)
        specific = float(np.sum(w * w * self.specific_var))
        return PortfolioRiskDecomposition(
            total_variance=systematic + specific,
            systematic_variance=systematic,
            specific_variance=specific,
            factor_exposure=np.asarray(exposure, dtype=np.float64),
        )

    def _check_weights(self, weights: FloatArray) -> FloatArray:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (len(self.asset_names),):
            raise ValueError(f"weights must have shape ({len(self.asset_names)},), got {w.shape}")
        return w
