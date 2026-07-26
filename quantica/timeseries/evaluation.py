r"""Forecast evaluation — the deliverable the libraries do not ship.

Fitting a GARCH model is a solved problem (:mod:`arch`); deciding whether one volatility forecast
is *genuinely* better than another is the part that needs care, and it is what a model-validation
team is actually paid to get right. This module implements that layer:

* :func:`diebold_mariano` — the Diebold--Mariano (1995) test of equal predictive accuracy, with a
  **HAC / Newey--West** long-run-variance correction. This is the subtlety most implementations
  miss: forecast-error loss differentials are serially correlated (volatility errors cluster), so
  the naive ``Var(\bar d) = s^2 / T`` is wrong and the test mis-sizes. The HAC estimator is the
  fix, and validating that it restores the correct size is the pillar's headline
  (:mod:`~quantica.timeseries.data` supplies the known-truth fixture).
* :func:`mse_loss` and :func:`qlike_loss` — the two loss functions that are *robust* to using a
  noisy volatility proxy (Patton 2011): their expected ranking is unaffected by the proxy's
  noise, which matters because true volatility is **latent** and we score against a squared-return
  proxy. QLIKE is the more robust of the two.
* :func:`mincer_zarnowitz` — the forecast-efficiency regression: regress the realised proxy on the
  forecast and test intercept ``= 0``, slope ``= 1`` jointly (an unbiased, efficient forecast
  passes).

References
----------
Diebold, F.X. and Mariano, R.S. (1995). "Comparing Predictive Accuracy." *JBES* 13(3).
Patton, A.J. (2011). "Volatility forecast comparison using imperfect volatility proxies."
*Journal of Econometrics* 160(1).
Mincer, J. and Zarnowitz, V. (1969). "The Evaluation of Economic Forecasts."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "DieboldMarianoResult",
    "MincerZarnowitzResult",
    "diebold_mariano",
    "mincer_zarnowitz",
    "mse_loss",
    "qlike_loss",
]


def mse_loss(forecast_variance: FloatArray, proxy: FloatArray) -> FloatArray:
    r"""Mean-squared-error loss per observation: :math:`(\text{proxy} - \hat\sigma^2)^2`.

    One of the two loss functions robust to volatility-proxy noise (Patton 2011): scoring the
    variance forecast against a noisy but unbiased proxy leaves the *expected* ranking unchanged.

    Parameters
    ----------
    forecast_variance : ndarray
        The variance forecasts :math:`\hat\sigma^2` (percent-squared).
    proxy : ndarray
        The realised-variance proxy (e.g. squared returns), same units and shape.

    Returns
    -------
    ndarray
        The per-observation squared error.
    """
    f = np.asarray(forecast_variance, dtype=np.float64)
    h = np.asarray(proxy, dtype=np.float64)
    return np.asarray((h - f) ** 2, dtype=np.float64)


def qlike_loss(forecast_variance: FloatArray, proxy: FloatArray) -> FloatArray:
    r"""QLIKE loss per observation, in ranking-equivalent form.

    Returns :math:`h/\hat\sigma^2 + \ln\hat\sigma^2` (``h`` the proxy). This is Patton's (2011)
    QLIKE, :math:`h/\hat\sigma^2 - \ln(h/\hat\sigma^2) - 1`, dropping the forecast-independent term
    :math:`\ln h + 1`. That term cancels in every loss *differential*,
    ranking and Mincer--Zarnowitz comparison, so the two forms are interchangeable for evaluation
    — but this one stays finite when the squared-return proxy is exactly zero (the full form's
    :math:`\ln h` diverges), which is why it is the robust choice in practice. QLIKE is more
    robust to proxy noise than MSE and penalises under-prediction of variance more heavily.

    Parameters
    ----------
    forecast_variance : ndarray
        The variance forecasts :math:`\hat\sigma^2`, strictly positive (percent-squared).
    proxy : ndarray
        The realised-variance proxy (e.g. squared returns), non-negative.

    Returns
    -------
    ndarray
        The per-observation QLIKE loss.

    Raises
    ------
    ValueError
        If any forecast variance is non-positive.
    """
    f = np.asarray(forecast_variance, dtype=np.float64)
    h = np.asarray(proxy, dtype=np.float64)
    if np.any(f <= 0.0):
        raise ValueError("QLIKE requires strictly positive variance forecasts")
    return np.asarray(h / f + np.log(f), dtype=np.float64)


def _newey_west_lrv(centered: FloatArray, lags: int) -> float:
    """Newey--West (Bartlett-kernel) long-run variance of a mean-zero series."""
    lrv = float(np.mean(centered * centered))  # gamma_0
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1)  # Bartlett kernel, guarantees a non-negative estimate
        gamma_k = float(np.mean(centered[k:] * centered[:-k]))
        lrv += 2.0 * weight * gamma_k
    return lrv


@dataclass(frozen=True)
class DieboldMarianoResult:
    """The Diebold--Mariano test of equal predictive accuracy.

    Attributes
    ----------
    statistic : float
        The DM statistic. Positive ⇒ ``loss_a`` exceeds ``loss_b`` on average, i.e. forecast *A*
        is **worse**; negative ⇒ *A* is better.
    p_value : float
        Two-sided p-value under the asymptotic standard-normal null of equal accuracy.
    mean_loss_diff : float
        The mean loss differential :math:`\\bar d = \\overline{L_A - L_B}`.
    lags : int
        The Newey--West truncation lag used (0 when ``hac=False``).
    hac : bool
        Whether the HAC long-run-variance correction was applied.
    """

    statistic: float
    p_value: float
    mean_loss_diff: float
    lags: int
    hac: bool


def diebold_mariano(
    loss_a: FloatArray,
    loss_b: FloatArray,
    *,
    hac: bool = True,
    lags: int | None = None,
) -> DieboldMarianoResult:
    r"""Diebold--Mariano test of equal predictive accuracy between two forecasts.

    Given per-observation losses of two forecasts, tests :math:`H_0: E[d_t] = 0` for the loss
    differential :math:`d_t = L_{A,t} - L_{B,t}`. The statistic is
    :math:`\bar d / \sqrt{\widehat{\operatorname{Var}}(\bar d)}`; the variance is the crux.

    Because loss differentials are **serially correlated** — one-step volatility forecast errors
    cluster, so consecutive :math:`d_t` are dependent — the naive
    :math:`\widehat{\operatorname{Var}}(\bar d) = \gamma_0 / T` understates the true sampling
    variability and the test **over-rejects**. With ``hac=True`` (the default) the long-run
    variance :math:`\gamma_0 + 2\sum_{k=1}^{L} w_k \gamma_k` is estimated by the Newey--West
    (Bartlett-kernel) HAC estimator, which restores the correct size. Passing ``hac=False``
    reproduces the naive test, for the size comparison.

    Parameters
    ----------
    loss_a, loss_b : ndarray
        Per-observation losses of forecasts *A* and *B* (e.g. from :func:`qlike_loss`), same shape.
    hac : bool, optional
        Apply the Newey--West HAC variance correction (default ``True``).
    lags : int, optional
        Truncation lag for the HAC estimator. Defaults to the automatic rule
        :math:`\lfloor 4 (T/100)^{2/9} \rfloor` when ``None``; ignored when ``hac=False``.

    Returns
    -------
    DieboldMarianoResult
        The statistic, two-sided p-value, mean differential and the lag used.

    Raises
    ------
    ValueError
        If the loss arrays differ in length or have fewer than two observations.
    """
    d = np.asarray(loss_a, dtype=np.float64) - np.asarray(loss_b, dtype=np.float64)
    n = d.size
    if np.asarray(loss_a).shape != np.asarray(loss_b).shape:
        raise ValueError("loss_a and loss_b must have the same shape")
    if n < 2:
        raise ValueError("need at least two observations")

    mean_diff = float(d.mean())
    centered = d - mean_diff
    if hac:
        used_lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))) if lags is None else lags
        used_lags = max(0, min(used_lags, n - 1))
        lrv = _newey_west_lrv(centered, used_lags)
    else:
        used_lags = 0
        lrv = float(np.mean(centered * centered))

    lrv = max(lrv, 1e-300)  # guard a degenerate (perfectly anti-correlated) differential
    statistic = mean_diff / np.sqrt(lrv / n)
    p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))
    return DieboldMarianoResult(
        statistic=float(statistic),
        p_value=p_value,
        mean_loss_diff=mean_diff,
        lags=used_lags,
        hac=hac,
    )


@dataclass(frozen=True)
class MincerZarnowitzResult:
    """The Mincer--Zarnowitz forecast-efficiency regression.

    Attributes
    ----------
    intercept, slope : float
        The estimated intercept :math:`a` and slope :math:`b` of ``proxy = a + b * forecast``.
    intercept_se, slope_se : float
        Their HAC standard errors.
    joint_p_value : float
        p-value of the joint Wald test of :math:`H_0: a = 0, b = 1` (forecast efficiency /
        unbiasedness). A low value rejects an efficient forecast.
    r_squared : float
        The regression :math:`R^2`.
    """

    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    joint_p_value: float
    r_squared: float


def mincer_zarnowitz(
    proxy: FloatArray, forecast_variance: FloatArray, *, hac_lags: int = 10
) -> MincerZarnowitzResult:
    r"""Mincer--Zarnowitz forecast-efficiency regression with a joint efficiency test.

    Regresses the realised proxy on the variance forecast, ``proxy = a + b * forecast + e``, and
    tests :math:`H_0: a = 0, b = 1` jointly (Wald). Under this null the forecast is unbiased and
    efficient: it needs no level correction (``a = 0``) and no rescaling (``b = 1``). A HAC
    covariance is used because the regression residuals are themselves serially correlated.

    Parameters
    ----------
    proxy : ndarray
        The realised-variance proxy (e.g. squared returns), the regressand.
    forecast_variance : ndarray
        The variance forecasts, the regressor.
    hac_lags : int, optional
        Newey--West lag for the regression's HAC covariance (default 10).

    Returns
    -------
    MincerZarnowitzResult
        The intercept, slope, their HAC standard errors, the joint-test p-value and :math:`R^2`.

    Raises
    ------
    ImportError
        If the optional :mod:`statsmodels` dependency is not installed.
    """
    try:
        import statsmodels.api as sm
    except ImportError as exc:  # pragma: no cover - exercised only without statsmodels
        raise ImportError(
            "mincer_zarnowitz needs the 'statsmodels' dependency; install it with "
            "`pip install statsmodels`."
        ) from exc

    y = np.asarray(proxy, dtype=np.float64)
    x = sm.add_constant(np.asarray(forecast_variance, dtype=np.float64))
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    restriction = (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 1.0]))
    wald = fit.wald_test(restriction, scalar=True)
    return MincerZarnowitzResult(
        intercept=float(fit.params[0]),
        slope=float(fit.params[1]),
        intercept_se=float(fit.bse[0]),
        slope_se=float(fit.bse[1]),
        joint_p_value=float(wald.pvalue),
        r_squared=float(fit.rsquared),
    )
