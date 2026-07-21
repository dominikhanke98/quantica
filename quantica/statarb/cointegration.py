r"""Cointegration testing — Engle--Granger and Johansen.

Two series can each wander (be :math:`I(1)`) yet move together, so that a linear
combination is stationary: they are **cointegrated**, and that stationary combination is
the tradeable *spread* of a pairs trade. The whole discipline of statistical arbitrage
rests on telling genuine cointegration from a *spurious* relationship between two
independent random walks — two series that drift apart forever but look correlated in
any finite sample. Getting that wrong is exactly the failure mode that sinks naive pairs
trading, so the test is the deliverable, not an afterthought.

Two tests, hand-implemented on top of the library primitives (CLAUDE.md §3):

* :func:`engle_granger` — the two-step residual test: regress one series on the other
  (OLS), then test the residual for a unit root. The **subtlety that a naive
  implementation gets wrong**: the residual is *estimated*, so its Dickey--Fuller
  statistic does **not** follow the standard ADF distribution — it needs the
  Engle--Granger / MacKinnon cointegration critical values (which depend on the number of
  series). We take the ADF statistic from ``statsmodels`` but apply the correct MacKinnon
  cointegration p-value, and anchor the whole thing to ``statsmodels.tsa.stattools.coint``.
* :func:`johansen` — the multivariate reduced-rank test for the *number* of cointegrating
  relations. The eigenvalue computation (the actual statistical content) is implemented
  here from the moment matrices; the trace and maximum-eigenvalue statistics and the
  cointegrating vectors are read off the eigen-system, and compared against the standard
  Osterwald--Lenum critical values embedded below. Anchored to
  ``statsmodels.tsa.vector_ar.vecm.coint_johansen`` to machine precision.

References
----------
Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and error correction",
*Econometrica* 55, 251--276.
Johansen, S. (1991). "Estimation and hypothesis testing of cointegration vectors in
Gaussian vector autoregressive models", *Econometrica* 59, 1551--1580.
Osterwald-Lenum, M. (1992). "A note with quantiles of the asymptotic distribution of the
maximum likelihood cointegration rank test statistics", *Oxford Bull. Econ. Stat.* 54.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "EngleGrangerResult",
    "JohansenResult",
    "engle_granger",
    "johansen",
]


# --------------------------------------------------------------------------- #
# Engle--Granger two-step residual test
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngleGrangerResult:
    """Outcome of the Engle--Granger cointegration test of ``y`` on ``x``.

    Attributes
    ----------
    hedge_ratio : float
        The OLS slope :math:`\\beta` — units of ``x`` per unit of ``y``, i.e. the pairs
        hedge ratio.
    intercept : float
        The regression intercept (present when ``trend='c'``, else 0).
    cointegrating_vector : ndarray, shape (2,)
        ``[1, -beta]`` — the combination ``y - beta*x`` that is (tested to be) stationary.
    spread : ndarray, shape (T,)
        The regression residual :math:`y - \\beta x - \\alpha`, i.e. the tradeable spread.
    adf_stat : float
        The augmented Dickey--Fuller statistic of the residual.
    pvalue : float
        The MacKinnon cointegration p-value (not the naive ADF p-value — see the module
        docstring).
    critical_values : dict of str to float
        Cointegration critical values at the ``"10%"``, ``"5%"`` and ``"1%"`` levels.
    n_obs : int
        Number of observations.
    """

    hedge_ratio: float
    intercept: float
    cointegrating_vector: FloatArray
    spread: FloatArray
    adf_stat: float
    pvalue: float
    critical_values: dict[str, float]
    n_obs: int

    def is_cointegrated(self, significance: float = 0.05) -> bool:
        """Whether the null of *no* cointegration is rejected at ``significance``."""
        return self.pvalue < significance


def engle_granger(
    y: FloatArray,
    x: FloatArray,
    *,
    trend: str = "c",
    autolag: str | None = "AIC",
    maxlag: int | None = None,
) -> EngleGrangerResult:
    r"""Engle--Granger residual-based cointegration test of ``y`` on ``x``.

    Regresses ``y`` on ``x`` (with an intercept when ``trend='c'``) and tests the residual
    for a unit root; the residual is the pairs spread. The residual Dickey--Fuller
    statistic is evaluated against the **MacKinnon cointegration** distribution (correct
    for an *estimated* residual with two series), not the standard ADF distribution.

    Parameters
    ----------
    y, x : ndarray, shape (T,)
        The two candidate :math:`I(1)` series; ``y`` is regressed on ``x``.
    trend : {"c", "n"}, optional
        ``"c"`` includes an intercept in the cointegrating regression (default); ``"n"``
        omits it.
    autolag : str or None, optional
        Lag-selection criterion for the residual ADF regression (default ``"AIC"``).
    maxlag : int, optional
        Maximum ADF lag; ``None`` lets ``statsmodels`` choose.

    Returns
    -------
    EngleGrangerResult
        The hedge ratio, spread, ADF statistic, MacKinnon p-value and critical values.

    Raises
    ------
    ValueError
        If ``y`` and ``x`` are not 1-D of equal length, or ``trend`` is unsupported.
    """
    import statsmodels.api as sm
    from statsmodels.tsa.adfvalues import mackinnoncrit
    from statsmodels.tsa.stattools import adfuller

    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if y_arr.ndim != 1 or x_arr.ndim != 1:
        raise ValueError("y and x must be 1-D series")
    if y_arr.shape != x_arr.shape:
        raise ValueError(f"y and x must have equal length, got {y_arr.shape} vs {x_arr.shape}")
    if trend not in ("c", "n"):
        raise ValueError(f"trend must be 'c' or 'n', got {trend!r}")

    design = sm.add_constant(x_arr) if trend == "c" else x_arr.reshape(-1, 1)
    ols = sm.OLS(y_arr, design).fit()
    params = np.asarray(ols.params, dtype=np.float64)
    intercept = float(params[0]) if trend == "c" else 0.0
    beta = float(params[-1])
    spread = np.asarray(ols.resid, dtype=np.float64)

    adf_stat = float(adfuller(spread, maxlag=maxlag, autolag=autolag, regression="n")[0])
    n_obs = int(y_arr.shape[0])
    # Two-series cointegration distribution (N=2), finite-sample adjusted by nobs.
    pvalue = float(_mackinnon_pvalue(adf_stat, trend, n_series=2))
    crit = np.asarray(mackinnoncrit(N=2, regression=trend, nobs=n_obs), dtype=np.float64)
    critical_values = {"1%": float(crit[0]), "5%": float(crit[1]), "10%": float(crit[2])}

    return EngleGrangerResult(
        hedge_ratio=beta,
        intercept=intercept,
        cointegrating_vector=np.array([1.0, -beta], dtype=np.float64),
        spread=spread,
        adf_stat=adf_stat,
        pvalue=pvalue,
        critical_values=critical_values,
        n_obs=n_obs,
    )


def _mackinnon_pvalue(stat: float, trend: str, n_series: int) -> float:
    """MacKinnon cointegration p-value for an Engle--Granger residual statistic."""
    from statsmodels.tsa.adfvalues import mackinnonp

    return float(mackinnonp(stat, regression=trend, N=n_series))


# --------------------------------------------------------------------------- #
# Johansen reduced-rank test
# --------------------------------------------------------------------------- #

# Osterwald--Lenum (1992) critical values, indexed [det_order][n_minus_r - 1] -> (10%, 5%,
# 1%). Rows run n - r = 1 .. 12. Validated against statsmodels' embedded tables in the
# test suite (a transcription guard).
_TRACE_CV: dict[int, FloatArray] = {
    -1: np.array(
        [
            [2.98, 4.13, 6.94],
            [10.47, 12.32, 16.36],
            [21.78, 24.28, 29.51],
            [37.03, 40.17, 46.57],
            [56.28, 60.06, 67.64],
            [79.53, 83.94, 92.71],
            [106.74, 111.78, 121.74],
            [138.0, 143.67, 154.8],
            [173.23, 179.52, 191.81],
            [212.47, 219.41, 232.83],
            [255.67, 263.26, 278.0],
            [302.91, 311.13, 326.97],
        ]
    ),
    0: np.array(
        [
            [2.71, 3.84, 6.63],
            [13.43, 15.49, 19.93],
            [27.07, 29.8, 35.46],
            [44.49, 47.85, 54.68],
            [65.82, 69.82, 77.82],
            [91.11, 95.75, 104.96],
            [120.37, 125.62, 135.98],
            [153.63, 159.53, 171.09],
            [190.87, 197.38, 210.04],
            [232.1, 239.25, 253.25],
            [277.37, 285.14, 300.28],
            [326.54, 334.98, 351.22],
        ]
    ),
    1: np.array(
        [
            [2.71, 3.84, 6.63],
            [16.16, 18.4, 23.15],
            [32.06, 35.01, 41.08],
            [51.65, 55.25, 62.52],
            [75.1, 79.34, 87.77],
            [102.47, 107.34, 116.98],
            [133.79, 139.28, 150.08],
            [169.06, 175.16, 187.19],
            [208.36, 215.13, 228.22],
            [251.63, 259.03, 273.38],
            [298.88, 306.9, 322.43],
            [350.11, 358.72, 375.32],
        ]
    ),
}
_MAX_EIG_CV: dict[int, FloatArray] = {
    -1: np.array(
        [
            [2.98, 4.13, 6.94],
            [9.47, 11.22, 15.09],
            [15.72, 17.8, 22.25],
            [21.84, 24.16, 29.06],
            [27.92, 30.44, 35.74],
            [33.93, 36.63, 42.23],
            [39.91, 42.77, 48.66],
            [45.89, 48.88, 55.03],
            [51.85, 54.96, 61.34],
            [57.8, 61.04, 67.64],
            [63.72, 67.08, 73.89],
            [69.65, 73.09, 80.09],
        ]
    ),
    0: np.array(
        [
            [2.71, 3.84, 6.63],
            [12.3, 14.26, 18.52],
            [18.89, 21.13, 25.86],
            [25.12, 27.59, 32.72],
            [31.24, 33.88, 39.37],
            [37.28, 40.08, 45.87],
            [43.29, 46.23, 52.31],
            [49.29, 52.36, 58.66],
            [55.24, 58.43, 65.0],
            [61.2, 64.5, 71.25],
            [67.13, 70.54, 77.49],
            [73.06, 76.57, 83.71],
        ]
    ),
    1: np.array(
        [
            [2.71, 3.84, 6.63],
            [15.0, 17.15, 21.75],
            [21.87, 24.25, 29.26],
            [28.24, 30.82, 36.19],
            [34.42, 37.16, 42.86],
            [40.52, 43.42, 49.41],
            [46.56, 49.59, 55.82],
            [52.59, 55.73, 62.17],
            [58.53, 61.81, 68.5],
            [64.53, 67.9, 74.74],
            [70.46, 73.94, 81.07],
            [76.41, 79.99, 87.24],
        ]
    ),
}


@dataclass(frozen=True)
class JohansenResult:
    """Outcome of the Johansen reduced-rank cointegration test.

    Attributes
    ----------
    eigenvalues : ndarray, shape (n,)
        The ordered eigenvalues :math:`\\lambda_1 > \\dots > \\lambda_n` of the
        reduced-rank problem.
    cointegrating_vectors : ndarray, shape (n, n)
        The eigenvectors (columns), the leading ones being the estimated cointegrating
        vectors.
    trace_stats : ndarray, shape (n,)
        The trace statistic for :math:`H_0:\\text{rank} \\le r`, ``r = 0 .. n-1``.
    max_eig_stats : ndarray, shape (n,)
        The maximum-eigenvalue statistic for the same hypotheses.
    trace_crit_values, max_eig_crit_values : ndarray, shape (n, 3)
        Critical values (columns ``10%, 5%, 1%``) aligned with the statistic rows.
    det_order : int
        Deterministic assumption: ``-1`` none, ``0`` constant, ``1`` linear trend.
    k_ar_diff : int
        Number of lagged differences in the underlying VECM.
    n_obs : int
        Effective number of observations used.
    """

    eigenvalues: FloatArray
    cointegrating_vectors: FloatArray
    trace_stats: FloatArray
    max_eig_stats: FloatArray
    trace_crit_values: FloatArray
    max_eig_crit_values: FloatArray
    det_order: int
    k_ar_diff: int
    n_obs: int

    def rank(self, significance: float = 0.05, *, statistic: str = "trace") -> int:
        r"""The estimated number of cointegrating relations at ``significance``.

        Steps up through the sequential hypotheses :math:`H_0:\text{rank}\le r` and returns
        the first ``r`` that is *not* rejected — the standard Johansen decision rule.

        Parameters
        ----------
        significance : {0.10, 0.05, 0.01}, optional
            Test level (default 0.05).
        statistic : {"trace", "max_eig"}, optional
            Which statistic to use (default the trace test).

        Returns
        -------
        int
            The inferred cointegration rank, ``0 .. n``.
        """
        try:
            col = {0.10: 0, 0.05: 1, 0.01: 2}[significance]
        except KeyError:
            raise ValueError(
                f"significance must be one of 0.10, 0.05, 0.01, got {significance}"
            ) from None
        if statistic == "trace":
            stats, crit = self.trace_stats, self.trace_crit_values
        elif statistic == "max_eig":
            stats, crit = self.max_eig_stats, self.max_eig_crit_values
        else:
            raise ValueError(f"statistic must be 'trace' or 'max_eig', got {statistic!r}")
        for r in range(stats.shape[0]):
            if stats[r] <= crit[r, col]:  # first hypothesis not rejected
                return r
        return int(stats.shape[0])


def johansen(
    data: FloatArray,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    r"""Johansen reduced-rank cointegration test on a panel of :math:`I(1)` series.

    Implements the reduced-rank regression eigenvalue computation directly: partial out the
    lagged differences (and the deterministic terms) from both the differences
    :math:`\Delta Y_t` and the lagged levels :math:`Y_{t-1}`, form the moment matrices
    :math:`S_{00}, S_{k0}, S_{kk}` of the residuals, and solve the eigenvalue problem
    :math:`|\lambda S_{kk} - S_{k0} S_{00}^{-1} S_{k0}^\top| = 0`. The trace and
    maximum-eigenvalue statistics follow from the eigenvalues; the eigenvectors are the
    cointegrating vectors. Critical values are the embedded Osterwald--Lenum tables.

    Parameters
    ----------
    data : ndarray, shape (T, n)
        The panel of candidate series (``n >= 2``).
    det_order : {-1, 0, 1}, optional
        Deterministic terms: ``-1`` none, ``0`` constant (default), ``1`` linear trend.
    k_ar_diff : int, optional
        Number of lagged differences in the VECM (default 1).

    Returns
    -------
    JohansenResult
        Eigenvalues, cointegrating vectors, trace/max-eigenvalue statistics and their
        critical values.

    Raises
    ------
    ValueError
        If ``data`` is not 2-D with at least two columns, ``det_order`` is unsupported, or
        there are too few observations for the requested lag.
    """
    from statsmodels.tsa.tsatools import lagmat

    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("data must be 2-D with at least two columns")
    if det_order not in (-1, 0, 1):
        raise ValueError(f"det_order must be -1, 0 or 1, got {det_order}")
    if k_ar_diff < 1:
        raise ValueError(f"k_ar_diff must be at least 1, got {k_ar_diff}")
    n_series = arr.shape[1]
    if arr.shape[0] <= k_ar_diff + n_series + 1:
        raise ValueError("too few observations for the requested k_ar_diff")

    # f is the detrend order applied to the differenced/lagged blocks (0 whenever any
    # deterministic term is present, matching the standard VECM treatment).
    f = 0 if det_order > -1 else det_order
    endog = _detrend(arr, det_order)
    dx = np.diff(endog, axis=0)
    z = _detrend(lagmat(dx, k_ar_diff)[k_ar_diff:], f)
    dx = _detrend(dx[k_ar_diff:], f)
    r0t = _residualise(dx, z)  # differences, purged of short-run dynamics
    lagged_levels = _detrend(endog[: endog.shape[0] - k_ar_diff][1:], f)
    rkt = _residualise(lagged_levels, z)  # lagged levels, purged

    n_eff = rkt.shape[0]
    skk = rkt.T @ rkt / n_eff
    sk0 = rkt.T @ r0t / n_eff
    s00 = r0t.T @ r0t / n_eff
    sig = sk0 @ np.linalg.inv(s00) @ sk0.T
    raw_vals, raw_vecs = np.linalg.eig(np.linalg.inv(skk) @ sig)

    order = np.argsort(raw_vals.real)[::-1]
    eigenvalues = np.clip(raw_vals.real[order], None, 1.0 - 1e-12)
    vectors = _normalise_eigenvectors(raw_vecs.real[:, order], skk)

    trace_stats = np.array(
        [-n_eff * np.sum(np.log(1.0 - eigenvalues[r:])) for r in range(n_series)],
        dtype=np.float64,
    )
    max_eig_stats = np.array(
        [-n_eff * np.log(1.0 - eigenvalues[r]) for r in range(n_series)], dtype=np.float64
    )
    # Row r tests rank <= r, i.e. n - r remaining relations -> table row (n - r) - 1.
    rows = [n_series - r - 1 for r in range(n_series)]
    return JohansenResult(
        eigenvalues=eigenvalues,
        cointegrating_vectors=vectors,
        trace_stats=trace_stats,
        max_eig_stats=max_eig_stats,
        trace_crit_values=_TRACE_CV[det_order][rows],
        max_eig_crit_values=_MAX_EIG_CV[det_order][rows],
        det_order=det_order,
        k_ar_diff=k_ar_diff,
        n_obs=n_eff,
    )


def _detrend(y: FloatArray, order: int) -> FloatArray:
    """Remove polynomial deterministic terms (``-1`` none, ``0`` mean, ``1`` linear)."""
    if order == -1:
        return y
    basis = np.vander(np.linspace(-1.0, 1.0, len(y)), order + 1)
    resid: FloatArray = y - basis @ np.linalg.pinv(basis) @ y
    return resid


def _residualise(y: FloatArray, x: FloatArray) -> FloatArray:
    """Residual of ``y`` after projecting on ``x`` (identity when ``x`` is empty)."""
    if x.size == 0:
        return y
    resid: FloatArray = y - x @ np.linalg.pinv(x) @ y
    return resid


def _normalise_eigenvectors(vectors: FloatArray, skk: FloatArray) -> FloatArray:
    """Scale eigenvectors to unit :math:`v^\\top S_{kk} v` and a stable sign convention."""
    scale = np.linalg.cholesky(vectors.T @ skk @ vectors)
    normalised = vectors @ np.linalg.inv(scale)
    flat = normalised.ravel()
    non_zero = np.flatnonzero(flat)
    if non_zero.size:  # fix the sign by the first non-zero entry (matches statsmodels)
        normalised = normalised * np.sign(flat[non_zero[0]])
    return np.asarray(normalised, dtype=np.float64)
