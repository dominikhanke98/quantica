r"""Post-estimation residual diagnostics for the fEGarch port (Phase 6).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Fisher & Gallagher 2012 for the weighted portmanteau; Engle & Ng 1993 for the sign-bias test;
    the probability-integral-transform Pearson goodness-of-fit), **never** the `fEGarch` source.
    Validated against committed `fEGarch` *output* fixtures (``diag_garch11_norm_*``).

The three fit-tests fEGarch reports on a fitted model, all computed from the **standardized
residuals** :math:`\hat z_t = (r_t - \hat\mu_t)/\hat\sigma_t` (and, for the sign-bias size terms,
the raw model residuals :math:`\hat\varepsilon_t = r_t - \hat\mu_t = \hat z_t\,\hat\sigma_t`):

* :func:`weighted_ljung_box` — the **weighted** Ljung-Box portmanteau test (Fisher & Gallagher 2012,
  *not* the classic equal-weight Ljung-Box 1978), with linearly-decreasing weights
  :math:`w_k = (m+1-k)/m` and the null obtained from a **Gamma approximation** to the weighted
  sum of :math:`\chi^2_1` terms (not a :math:`\chi^2`). Run on the simple residuals (remaining mean
  autocorrelation) and on the squared residuals (remaining volatility autocorrelation).
* :func:`sign_bias_test` — the Engle-Ng (1993) test for remaining sign/size asymmetry: an OLS
  regression of :math:`\hat z_t^2` on lagged sign and signed-size regressors, with the three
  coefficient t-tests and a **joint LM test** :math:`n R^2 \sim \chi^2_3`.
* :func:`goodness_of_fit_test` — the adjusted Pearson :math:`\chi^2` goodness-of-fit test on the
  probability integral transform :math:`u_t = F_\eta(\hat z_t)` of the standardized residuals under
  the fitted conditional distribution, binned into equal-probability bins (:math:`\mathrm{df} =
  n_{\text{bins}} - 1`, i.e. the estimated parameters are **not** subtracted).

:func:`fit_test_suite` bundles all four reports (LB simple + squared, sign-bias, GoF) from one fit,
mirroring fEGarch's ``fit_test_suite``. **fEGarch exposes only p-values** (the weighted-LB lag and
the GoF bin-count / df alongside), never the raw statistics, so validation reproduces the p-values;
the standardized residuals are machine-exact, so the reproductions match the fixtures to ``~1e-14``
— the null-distribution CDFs (Gamma / :math:`\chi^2` / Student-t) agree with R's to machine
precision, well inside any R↔SciPy distribution-function tolerance.

References
----------
Fisher, T. J. & Gallagher, C. M. (2012). "New Weighted Portmanteau Statistics for Time Series
Goodness of Fit Testing." *JASA* 107(498) — the weighted portmanteau and its Gamma approximation.
Engle, R. F. & Ng, V. K. (1993). "Measuring and Testing the Impact of News on Volatility."
*Journal of Finance* 48(5) — the sign-bias / size-bias tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from quantica.timeseries.fegarch.forecast import _distribution_for_fit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantica.core.types import FloatArray
    from quantica.timeseries.fegarch.distributions import ConditionalDistribution
    from quantica.timeseries.fegarch.garch import GarchFit

__all__ = [
    "FitTestSuite",
    "GoodnessOfFitResult",
    "LjungBoxResult",
    "SignBiasResult",
    "fit_test_suite",
    "goodness_of_fit_test",
    "sign_bias_test",
    "standardized_residuals",
    "weighted_ljung_box",
]

#: fEGarch defaults: the weighted-LB maximum lag and the goodness-of-fit bin counts.
_DEFAULT_M_MAX = 20
_DEFAULT_N_BINS: tuple[int, ...] = (20, 30, 40, 50)


@dataclass(frozen=True)
class LjungBoxResult:
    """Weighted Ljung-Box p-values across lags ``1..m_max``.

    Attributes
    ----------
    lags : ndarray of int
        The maximum lags :math:`m = 1, \\dots, m_{\\max}`.
    pvalue : ndarray
        The weighted-portmanteau p-value at each lag (Gamma approximation).
    fitdf : int
        The degrees-of-freedom adjustment applied (fitted mean-ARMA parameters for the simple
        residuals; 0 for the squared residuals).
    """

    lags: FloatArray
    pvalue: FloatArray
    fitdf: int


@dataclass(frozen=True)
class SignBiasResult:
    """Engle-Ng sign-bias test p-values (all two-sided / upper-tail as appropriate)."""

    sign_bias_pvalue: float
    negative_size_pvalue: float
    positive_size_pvalue: float
    joint_pvalue: float


@dataclass(frozen=True)
class GoodnessOfFitResult:
    """Adjusted Pearson goodness-of-fit p-values across bin counts.

    Attributes
    ----------
    n_bins : ndarray of int
        The bin counts tested.
    dof : ndarray of int
        The degrees of freedom for each (``n_bins - 1``).
    pvalue : ndarray
        The Pearson :math:`\\chi^2` p-value at each bin count.
    """

    n_bins: FloatArray
    dof: FloatArray
    pvalue: FloatArray


@dataclass(frozen=True)
class FitTestSuite:
    """The bundled post-estimation fit-tests (mirrors fEGarch's ``fit_test_suite``)."""

    ljung_box_simple: LjungBoxResult
    ljung_box_squared: LjungBoxResult
    sign_bias: SignBiasResult
    goodness_of_fit: GoodnessOfFitResult


def standardized_residuals(fit: GarchFit, returns: FloatArray) -> FloatArray:
    r"""The standardized residuals :math:`\hat z_t = (r_t - \hat\mu_t)/\hat\sigma_t` of a fit.

    For the constant-mean GARCH the conditional mean is the fitted :math:`\mu`, so
    :math:`\hat z_t = (r_t - \mu)/\hat\sigma_t` with :math:`\hat\sigma_t` the fitted
    conditional-volatility series.

    Parameters
    ----------
    fit : GarchFit
        The fitted model (supplies :math:`\mu` and the conditional-volatility series).
    returns : ndarray, shape (T,)
        The return series the model was fit on.

    Returns
    -------
    ndarray, shape (T,)
        The standardized residuals.

    Raises
    ------
    ValueError
        If ``returns`` and the fitted conditional-volatility series differ in length.
    """
    y = np.asarray(returns, dtype=np.float64)
    sigma = np.asarray(fit.conditional_volatility, dtype=np.float64)
    if y.shape != sigma.shape:
        raise ValueError(f"returns length {y.shape} != conditional_volatility length {sigma.shape}")
    return np.asarray((y - fit.params["mu"]) / sigma, dtype=np.float64)


def _sample_autocorrelations(x: FloatArray, m_max: int) -> FloatArray:
    r"""Biased (divide-by-:math:`n`) sample autocorrelations of ``x`` for lags ``1..m_max``.

    The standard estimator :math:`\hat\rho_k = \hat\gamma_k/\hat\gamma_0` with
    :math:`\hat\gamma_k = \frac1n\sum_{t=1}^{n-k}(x_t-\bar x)(x_{t+k}-\bar x)` (mean-corrected,
    divided by :math:`n`) — matching R's ``acf``.
    """
    v = np.asarray(x, dtype=np.float64)
    n = v.size
    v = v - v.mean()
    gamma0 = float(np.mean(v * v))
    return np.array(
        [float(np.sum(v[: n - k] * v[k:]) / n / gamma0) for k in range(1, m_max + 1)],
        dtype=np.float64,
    )


def weighted_ljung_box(
    x: FloatArray, *, m_max: int = _DEFAULT_M_MAX, fitdf: int = 0
) -> LjungBoxResult:
    r"""Weighted Ljung-Box portmanteau test (Fisher & Gallagher 2012), p-values for ``1..m_max``.

    For each maximum lag :math:`m`, the weighted statistic (linearly-decreasing weights
    :math:`w_k = (m+1-k)/m`, Fisher-Gallagher 2012)

    .. math::

        Q_W(m) = n(n+2) \sum_{k=1}^{m} w_k\, \frac{\hat\rho_k^2}{n-k}

    is referred to a **Gamma approximation** of the weighted sum of :math:`\chi^2_1` terms — with an
    optional degrees-of-freedom correction ``fitdf`` for estimated parameters — via the shape and
    scale

    .. math::

        \alpha = \frac{\tfrac34 (m+1)^2 m}{2m^2 + 3m + 1 - 6\,m\,\mathrm{fitdf}}, \qquad
        \beta  = \frac{\tfrac23 (2m^2 + 3m + 1 - 6\,m\,\mathrm{fitdf})}{m(m+1)},

    and :math:`p = P(\mathrm{Gamma}(\alpha,\beta) > Q_W(m))`. At ``fitdf = 0`` the parameters reduce
    to the moment-matched Gamma of :math:`\sum_k w_k \chi^2_{1}`. This is **not** the classic
    equal-weight Ljung-Box with a :math:`\chi^2` null.

    Parameters
    ----------
    x : ndarray, shape (T,)
        The series to test — the standardized residuals (mean autocorrelation) or their squares
        (volatility autocorrelation).
    m_max : int, optional
        The maximum lag; p-values are returned for :math:`m = 1, \dots, m_{\max}` (default 20).
    fitdf : int, optional
        Degrees-of-freedom correction for estimated parameters (default 0): for the simple
        residuals the number of fitted mean-ARMA parameters, for the squared residuals 0.

    Returns
    -------
    LjungBoxResult
        The lags, the Gamma p-values, and ``fitdf``.

    Raises
    ------
    ValueError
        If ``m_max`` is not positive, or ``fitdf`` makes a Gamma parameter non-positive.
    """
    v = np.asarray(x, dtype=np.float64)
    n = v.size
    if m_max < 1:
        raise ValueError(f"m_max must be >= 1, got {m_max}")
    rho = _sample_autocorrelations(v, m_max)
    lags = np.arange(1, m_max + 1)
    pvalue = np.empty(m_max, dtype=np.float64)
    for i, m in enumerate(lags):
        k = np.arange(1, m + 1)
        w = (m + 1 - k) / m
        stat = n * (n + 2) * float(np.sum(w * rho[:m] ** 2 / (n - k)))
        denom = 2.0 * m * m + 3.0 * m + 1.0 - 6.0 * m * fitdf
        if denom <= 0.0:
            raise ValueError(
                f"fitdf={fitdf} too large for lag m={m} (non-positive Gamma parameter)"
            )
        shape = 0.75 * (m + 1) ** 2 * m / denom
        scale = (2.0 / 3.0) * denom / (m * (m + 1))
        pvalue[i] = float(stats.gamma.sf(stat, a=shape, scale=scale))
    return LjungBoxResult(lags=lags.astype(np.float64), pvalue=pvalue, fitdf=int(fitdf))


def sign_bias_test(std_resid: FloatArray, residuals: FloatArray) -> SignBiasResult:
    r"""Engle-Ng (1993) sign-bias test for remaining asymmetry in the standardized residuals.

    OLS regression of the squared standardized residuals on a constant and three lagged
    sign/signed-size regressors,

    .. math::

        \hat z_t^2 = b_0 + b_1 S^-_{t-1} + b_2 S^-_{t-1}\hat\varepsilon_{t-1}
                     + b_3 S^+_{t-1}\hat\varepsilon_{t-1} + e_t,

    where :math:`S^-_t = \mathbf 1\{\hat z_t < 0\}`, :math:`S^+_t = 1 - S^-_t`, and the **size**
    regressors use the *raw* model residual :math:`\hat\varepsilon` (not the standardized
    :math:`\hat z`). The three coefficient two-sided t-tests give the sign-bias, negative-size-bias
    and positive-size-bias p-values; the **joint** test is the Lagrange-multiplier statistic
    :math:`n R^2 \sim \chi^2_3` (not the F-test — the fixture distinguishes them).

    Parameters
    ----------
    std_resid : ndarray, shape (T,)
        The standardized residuals :math:`\hat z_t`.
    residuals : ndarray, shape (T,)
        The raw model residuals :math:`\hat\varepsilon_t = r_t - \hat\mu_t` (:math:`= \hat z_t
        \hat\sigma_t`), aligned with ``std_resid``.

    Returns
    -------
    SignBiasResult
        The three individual p-values and the joint LM p-value.

    Raises
    ------
    ValueError
        If ``std_resid`` and ``residuals`` differ in length or are too short.
    """
    z = np.asarray(std_resid, dtype=np.float64)
    eps = np.asarray(residuals, dtype=np.float64)
    if z.shape != eps.shape:
        raise ValueError(f"std_resid length {z.shape} != residuals length {eps.shape}")
    if z.size < 6:
        raise ValueError(f"need at least 6 observations, got {z.size}")

    z_lag = z[:-1]
    eps_lag = eps[:-1]
    y = z[1:] ** 2
    s_neg = (z_lag < 0.0).astype(np.float64)
    s_pos = 1.0 - s_neg
    design = np.column_stack([np.ones_like(z_lag), s_neg, s_neg * eps_lag, s_pos * eps_lag])
    m, k = design.shape

    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = m - k
    rss = float(resid @ resid)
    total = float(np.sum((y - y.mean()) ** 2))

    sigma2 = rss / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    p_ind = 2.0 * stats.t.sf(np.abs(tvals), dof)

    r_squared = 1.0 - rss / total
    lm = m * r_squared
    joint_p = float(stats.chi2.sf(lm, k - 1))

    return SignBiasResult(
        sign_bias_pvalue=float(p_ind[1]),
        negative_size_pvalue=float(p_ind[2]),
        positive_size_pvalue=float(p_ind[3]),
        joint_pvalue=joint_p,
    )


def goodness_of_fit_test(
    std_resid: FloatArray,
    distribution: ConditionalDistribution,
    *,
    dist_params: Sequence[float] | None = None,
    n_bins: Sequence[int] = _DEFAULT_N_BINS,
) -> GoodnessOfFitResult:
    r"""Adjusted Pearson :math:`\chi^2` goodness-of-fit on the PIT of the standardized residuals.

    The probability integral transform :math:`u_t = F_\eta(\hat z_t)` under the fitted conditional
    distribution is binned into ``n_bins`` **equal-probability** bins on :math:`[0, 1]`; the Pearson
    statistic :math:`\sum_i (O_i - E_i)^2 / E_i` with :math:`E_i = n / n_{\text{bins}}` is referred
    to :math:`\chi^2` with :math:`\mathrm{df} = n_{\text{bins}} - 1` — the estimated parameters are
    **not** subtracted (matching fEGarch).

    Parameters
    ----------
    std_resid : ndarray, shape (T,)
        The standardized residuals :math:`\hat z_t`.
    distribution : ConditionalDistribution
        The fitted conditional distribution providing the CDF :math:`F_\eta` for the PIT.
    dist_params : sequence of float or None, optional
        The distribution's shape parameters (default ``None`` uses its ``param_start``).
    n_bins : sequence of int, optional
        The bin counts to test (default ``(20, 30, 40, 50)``).

    Returns
    -------
    GoodnessOfFitResult
        The bin counts, degrees of freedom, and p-values.
    """
    z = np.asarray(std_resid, dtype=np.float64)
    n = z.size
    u = np.asarray(distribution.cdf(z, dist_params), dtype=np.float64)
    counts = np.array(list(n_bins), dtype=np.int64)
    dof = counts - 1
    pvalue = np.empty(counts.size, dtype=np.float64)
    for i, nb in enumerate(counts):
        edges = np.linspace(0.0, 1.0, int(nb) + 1)
        observed, _ = np.histogram(u, bins=edges)
        expected = n / nb
        chi_sq = float(np.sum((observed - expected) ** 2 / expected))
        pvalue[i] = float(stats.chi2.sf(chi_sq, int(nb) - 1))
    return GoodnessOfFitResult(
        n_bins=counts.astype(np.float64), dof=dof.astype(np.float64), pvalue=pvalue
    )


def fit_test_suite(
    fit: GarchFit,
    returns: FloatArray,
    *,
    m_max: int = _DEFAULT_M_MAX,
    n_bins: Sequence[int] = _DEFAULT_N_BINS,
) -> FitTestSuite:
    r"""The bundled post-estimation fit-tests for a fitted model (mirrors fEGarch's suite).

    Computes the standardized residuals :math:`\hat z_t` and the raw residuals
    :math:`\hat\varepsilon_t` from ``fit`` and runs: the weighted Ljung-Box on :math:`\hat z_t`
    (simple; ``fitdf`` = number of fitted mean-ARMA params, 0 for the constant mean) and on
    :math:`\hat z_t^2` (squared; ``fitdf = 0``); the Engle-Ng sign-bias test; and the PIT Pearson
    goodness-of-fit test under the fitted conditional distribution.

    Parameters
    ----------
    fit : GarchFit
        The fitted model.
    returns : ndarray, shape (T,)
        The return series the model was fit on.
    m_max : int, optional
        The maximum lag for the Ljung-Box tests (default 20).
    n_bins : sequence of int, optional
        The bin counts for the goodness-of-fit test (default ``(20, 30, 40, 50)``).

    Returns
    -------
    FitTestSuite
        The Ljung-Box (simple + squared), sign-bias, and goodness-of-fit results.
    """
    z = standardized_residuals(fit, returns)
    eps = np.asarray(returns, dtype=np.float64) - fit.params["mu"]
    # Constant-mean GARCH has no ARMA mean parameters, so the simple-residual df correction is 0.
    fitdf_simple = sum(1 for name in fit.params if name in ("ar1", "ma1"))
    distribution, dist_params = _distribution_for_fit(fit)
    return FitTestSuite(
        ljung_box_simple=weighted_ljung_box(z, m_max=m_max, fitdf=fitdf_simple),
        ljung_box_squared=weighted_ljung_box(z * z, m_max=m_max, fitdf=0),
        sign_bias=sign_bias_test(z, eps),
        goodness_of_fit=goodness_of_fit_test(
            z, distribution, dist_params=dist_params, n_bins=n_bins
        ),
    )
