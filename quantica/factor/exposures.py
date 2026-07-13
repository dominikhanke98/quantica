r"""Factor-exposure estimation by time-series regression.

An asset's factor loadings are the slopes of its excess return regressed on the
factor returns,

.. math:: r_{t,i} = \alpha_i + \beta_i^\top f_t + \varepsilon_{t,i},

estimated by ordinary least squares over the sample. The **estimator is not the
deliverable** — we lean on ``statsmodels`` (which ships OLS with inference:
t-statistics, :math:`R^2`, residual variance) rather than hand-roll the normal
equations (CLAUDE.md §3). The package's contribution is the risk-model assembly
and out-of-sample validation built on top (see :mod:`quantica.factor.model` and
stage 2). The tests anchor the fitted betas against an *independent*
``numpy.linalg.lstsq`` so "we called statsmodels correctly" is itself checked.

The residual variance is the **specific (idiosyncratic) risk** :math:`\sigma^2_i`
that becomes the diagonal of ``D`` in :math:`\Sigma = B F B^\top + D`. We report
the unbiased estimate (statsmodels' ``mse_resid``, dividing by :math:`T-k-1`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantica.core.types import FloatArray

__all__ = ["FactorExposures", "estimate_exposures"]


@dataclass(frozen=True)
class FactorExposures:
    """One asset's estimated factor exposures and regression diagnostics.

    Attributes
    ----------
    alpha : float
        Regression intercept.
    betas : ndarray, shape (k,)
        Factor loadings, aligned with ``factor_names``.
    t_stats : ndarray, shape (k,)
        t-statistics on the betas (from ``statsmodels``).
    r_squared : float
        Coefficient of determination — the systematic fraction of return variance.
    specific_variance : float
        Residual (idiosyncratic) variance, unbiased (``SSR / (T - k - 1)``).
    n_obs : int
        Number of time periods in the regression.
    factor_names : tuple of str
        Factor labels.
    """

    alpha: float
    betas: FloatArray
    t_stats: FloatArray
    r_squared: float
    specific_variance: float
    n_obs: int
    factor_names: tuple[str, ...]


def estimate_exposures(
    asset_excess_returns: FloatArray,
    factor_returns: FloatArray,
    factor_names: tuple[str, ...],
) -> FactorExposures:
    r"""OLS of one asset's excess returns on the factors (with an intercept).

    Parameters
    ----------
    asset_excess_returns : ndarray, shape (T,)
        The asset's excess returns.
    factor_returns : ndarray, shape (T, k)
        The factor excess returns, aligned in time with the asset.
    factor_names : tuple of str
        Names of the ``k`` factors.

    Returns
    -------
    FactorExposures
        Alpha, betas, t-statistics, :math:`R^2`, and the specific variance.
    """
    import statsmodels.api as sm  # lazy: statsmodels import is heavy

    y = np.asarray(asset_excess_returns, dtype=np.float64)
    f = np.asarray(factor_returns, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"factor_returns must be 2-D (T, k), got shape {f.shape}")
    if y.ndim != 1 or y.shape[0] != f.shape[0]:
        raise ValueError(
            f"asset_excess_returns must be 1-D of length T={f.shape[0]}, got {y.shape}"
        )
    k = f.shape[1]
    if k != len(factor_names):
        raise ValueError(f"factor_returns has {k} columns but {len(factor_names)} names")
    if f.shape[0] <= k + 1:
        raise ValueError(f"need more than k+1={k + 1} observations, got {f.shape[0]}")

    design = sm.add_constant(f, has_constant="add")  # prepend the intercept column
    result = sm.OLS(y, design).fit()
    params = np.asarray(result.params, dtype=np.float64)
    tvalues = np.asarray(result.tvalues, dtype=np.float64)
    return FactorExposures(
        alpha=float(params[0]),
        betas=params[1:].copy(),
        t_stats=tvalues[1:].copy(),
        r_squared=float(result.rsquared),
        specific_variance=float(result.mse_resid),
        n_obs=int(f.shape[0]),
        factor_names=tuple(factor_names),
    )
