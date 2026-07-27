r"""Dynamic Conditional Correlation (DCC-GARCH) — a time-varying covariance matrix.

Engle's (2002) DCC model builds a conditional covariance matrix in two stages:

1. a **univariate GARCH** per asset (reused from :mod:`quantica.timeseries.models`), giving each
   asset's conditional volatility :math:`\sigma_{i,t}` and standardised residuals
   :math:`z_{i,t} = \varepsilon_{i,t}/\sigma_{i,t}`;
2. a **dynamic correlation** recursion on the standardised residuals,

   .. math::

       Q_t = (1 - a - b)\,\bar Q + a\,z_{t-1}z_{t-1}' + b\,Q_{t-1}, \qquad
       R_t = \operatorname{diag}(Q_t)^{-1/2} Q_t \operatorname{diag}(Q_t)^{-1/2},

   with :math:`(a, b)` estimated by maximising the DCC quasi-likelihood (``scipy``). The conditional
   covariance is :math:`H_t = D_t R_t D_t`, :math:`D_t = \operatorname{diag}(\sigma_{i,t})`.

Setting :math:`a = b = 0` collapses :math:`R_t` to the constant :math:`\bar Q` — the CCC (constant
conditional correlation) special case, the reduction anchor.

The payoff is cross-pillar: DCC produces exactly the conditional covariance that the portfolio
pillar's min-variance construction and the risk pillar's VaR consume, so :class:`DccCovariance`
wraps DCC's one-step-ahead forecast as a drop-in :class:`~quantica.factor.CovarianceEstimator`,
lettng it race the static estimators from factor stage 2 on out-of-sample risk forecasting.

References
----------
Engle, R. (2002). "Dynamic Conditional Correlation." *Journal of Business & Economic Statistics*
20(3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from quantica.timeseries.models import fit_volatility_model

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "DccCovariance",
    "DccResult",
    "fit_dcc",
]

_DCC_SCALE = 100.0  # GARCH is fit on percent returns; covariances are rescaled back to native units


def _correlation_from_q(q_matrix: FloatArray) -> FloatArray:
    """Normalise a quasi-correlation matrix ``Q`` to a correlation matrix ``R``."""
    inv_sqrt = 1.0 / np.sqrt(np.diag(q_matrix))
    return np.asarray(q_matrix * np.outer(inv_sqrt, inv_sqrt), dtype=np.float64)


def _dcc_recursion(
    z: FloatArray, qbar: FloatArray, a: float, b: float
) -> tuple[FloatArray, FloatArray, float]:
    """Run the DCC correlation recursion; return (R_path, Q_{T+1}, quasi-loglikelihood)."""
    n_obs, n = z.shape
    correlations = np.empty((n_obs, n, n), dtype=np.float64)
    q_current = qbar.copy()
    loglik = 0.0
    for t in range(n_obs):
        r_t = _correlation_from_q(q_current)
        correlations[t] = r_t
        _sign, logdet = np.linalg.slogdet(r_t)
        solved = np.linalg.solve(r_t, z[t])
        loglik += -0.5 * (logdet + z[t] @ solved)
        q_current = (1.0 - a - b) * qbar + a * np.outer(z[t], z[t]) + b * q_current
    return correlations, q_current, loglik


@dataclass(frozen=True)
class DccResult:
    r"""A fitted DCC-GARCH model producing a time-varying covariance matrix.

    Attributes
    ----------
    a, b : float
        The DCC correlation parameters (``a`` news impact, ``b`` persistence; ``a + b < 1``).
    unconditional_correlation : ndarray, shape (n, n)
        The target correlation :math:`\bar Q` (the sample correlation of the standardised resids).
    conditional_correlations : ndarray, shape (T, n, n)
        The in-sample conditional correlation path :math:`R_t`.
    conditional_covariances : ndarray, shape (T, n, n)
        The in-sample conditional covariance path :math:`H_t = D_t R_t D_t` (native return units).
    univariate_volatility : ndarray, shape (T, n)
        Each asset's conditional volatility :math:`\sigma_{i,t}` (native units).
    standardized_residuals : ndarray, shape (T, n)
        The standardised residuals :math:`z_{i,t}` fed to the correlation recursion.
    forecast_correlation : ndarray, shape (n, n)
        The one-step-ahead correlation forecast :math:`R_{T+1}`.
    forecast_volatility : ndarray, shape (n,)
        The one-step-ahead volatility forecast :math:`\sigma_{i,T+1}` (native units).
    loglikelihood : float
        The maximised DCC (correlation-stage) quasi-log-likelihood.
    """

    a: float
    b: float
    unconditional_correlation: FloatArray
    conditional_correlations: FloatArray
    conditional_covariances: FloatArray
    univariate_volatility: FloatArray
    standardized_residuals: FloatArray
    forecast_correlation: FloatArray
    forecast_volatility: FloatArray
    loglikelihood: float

    def forecast_covariance(self) -> FloatArray:
        r"""The one-step-ahead conditional covariance :math:`H_{T+1} = D_{T+1} R_{T+1} D_{T+1}`.

        This is the object the portfolio and risk pillars consume — a forward covariance forecast
        conditioned on everything observed through ``T``.

        Returns
        -------
        ndarray, shape (n, n)
            The one-step covariance forecast in native return units.
        """
        d = np.diag(self.forecast_volatility)
        return np.asarray(d @ self.forecast_correlation @ d, dtype=np.float64)


def fit_dcc(returns: FloatArray, *, a_init: float = 0.02, b_init: float = 0.95) -> DccResult:
    r"""Fit a DCC-GARCH model: univariate GARCH per asset, then a dynamic correlation.

    Each column is fitted with a GARCH(1,1) (reusing
    :func:`~quantica.timeseries.fit_volatility_model`); the standardised residuals drive the DCC
    correlation recursion whose :math:`(a, b)` are estimated by maximising the quasi-likelihood.

    Parameters
    ----------
    returns : ndarray, shape (T, n)
        The asset return panel, in native units (scaled internally to percent for the GARCH fits).
    a_init, b_init : float, optional
        Starting values for the news-impact and persistence parameters.

    Returns
    -------
    DccResult
        The fitted parameters and the in-sample / one-step-ahead conditional covariances.

    Raises
    ------
    ValueError
        If ``returns`` is not a 2-D panel with at least two assets.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.ndim != 2 or r.shape[1] < 2:
        raise ValueError("returns must be a 2-D panel (T, n) with n >= 2")
    n_obs, n = r.shape

    # Stage 1: univariate GARCH per asset (percent returns), collecting vols and one-step forecasts.
    vols = np.empty((n_obs, n), dtype=np.float64)
    std_resid = np.empty((n_obs, n), dtype=np.float64)
    forecast_vol = np.empty(n, dtype=np.float64)
    for i in range(n):
        series = r[:, i] * _DCC_SCALE
        fit = fit_volatility_model(series, "GARCH")
        sigma = fit.conditional_volatility
        resid = series - fit.params["mu"]
        vols[:, i] = sigma / _DCC_SCALE
        std_resid[:, i] = resid / sigma
        next_var = (
            fit.params["omega"]
            + fit.params["alpha[1]"] * resid[-1] ** 2
            + fit.params["beta[1]"] * sigma[-1] ** 2
        )
        forecast_vol[i] = np.sqrt(next_var) / _DCC_SCALE

    # Stage 2: DCC correlation parameters by quasi-maximum-likelihood.
    qbar = np.asarray(np.corrcoef(std_resid, rowvar=False), dtype=np.float64)

    def negative_loglik(params: FloatArray) -> float:
        a, b = float(params[0]), float(params[1])
        if a < 0.0 or b < 0.0 or a + b >= 0.9999:
            return 1e10
        _corr, _q_next, loglik = _dcc_recursion(std_resid, qbar, a, b)
        return -loglik

    opt = minimize(
        negative_loglik,
        np.array([a_init, b_init]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6},
    )
    a, b = float(opt.x[0]), float(opt.x[1])

    correlations, q_next, loglik = _dcc_recursion(std_resid, qbar, a, b)
    covariances = np.einsum("ti,tij,tj->tij", vols, correlations, vols)
    forecast_correlation = _correlation_from_q(q_next)

    return DccResult(
        a=a,
        b=b,
        unconditional_correlation=np.asarray(qbar, dtype=np.float64),
        conditional_correlations=correlations,
        conditional_covariances=np.asarray(covariances, dtype=np.float64),
        univariate_volatility=vols,
        standardized_residuals=std_resid,
        forecast_correlation=forecast_correlation,
        forecast_volatility=forecast_vol,
        loglikelihood=float(loglik),
    )


class DccCovariance:
    """DCC one-step-ahead covariance as a drop-in :class:`~quantica.factor.CovarianceEstimator`.

    Fits DCC on a training panel and returns its one-step-ahead conditional covariance forecast —
    the cross-pillar tie-back that lets the econometrics pillar's dynamic covariance compete against
    the static estimators (sample, Ledoit--Wolf, factor) inside the factor pillar's out-of-sample
    comparison harness.
    """

    name = "dcc"

    def estimate(
        self, asset_returns: FloatArray, factor_returns: FloatArray | None = None
    ) -> FloatArray:
        """Return DCC's one-step-ahead covariance forecast for the panel.

        Parameters
        ----------
        asset_returns : ndarray, shape (T, n)
            The training return panel.
        factor_returns : ndarray, optional
            Unused; accepted for interface compatibility with :class:`CovarianceEstimator`.

        Returns
        -------
        ndarray, shape (n, n)
            The one-step-ahead conditional covariance (native units).
        """
        return fit_dcc(np.asarray(asset_returns, dtype=np.float64)).forecast_covariance()
