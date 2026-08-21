r"""FIGARCH(1,d,1) — fractionally-integrated GARCH (Phase 4, variance-recursion long memory).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Baillie-Bollerslev-Mikkelsen 1996 for FIGARCH; Conrad-Haag 2006 for the ARCH(∞) coefficient
    conditions; WP175 §2.1 Eqs. 2.3-2.4 for the ω-direct form), **never** the `fEGarch`
    source. The pre-sample convention (``presample=50``, ``trunc="none"``) was resolved by matching
    the committed `fEGarch` output fixture to machine precision (4.86e-17), not by reading code.

FIGARCH is the first **variance-recursion** long-memory model — a fundamentally different seam from
the EGF (FIEGARCH) family, where the fractional operator entered the *log*-variance. Here it enters
the conditional-**variance** polynomial directly (WP175 Eq. 2.4):

.. math::

    \sigma_t^2 = \omega + \sum_{i=1}^{\infty}\theta_i\,\varepsilon_{t-i}^2,\qquad
    \varepsilon_t = r_t - \mu,\qquad
    \theta(B) = 1 - \frac{(1-\phi_1 B)(1-B)^d}{1-\beta_1 B} = \sum_{i\ge 1}\theta_i B^i,

with :math:`\theta_1 = d + \phi_1 - \beta_1` and all :math:`\theta_i \ge 0` (Conrad-Haag 2006). Two
things distinguish this seam from every prior model:

* **The intercept** :math:`\omega` is the **variance** intercept used **directly** (WP175
  :math:`\omega^*`), **not** the Baillie-Bollerslev-Mikkelsen / ``arch``-package ARCH(∞) intercept
  :math:`(1-\beta_1)^{-1}\omega`. (The ``arch`` package uses the latter, which is why it diverges
  from `fEGarch` at ~2e-3 — a documented-divergence anchor, not a machine-precision one.)
* **There is no** :math:`\eta` **/** :math:`\sigma^2` **feedback.** :math:`\sigma_t^2` is a *pure
  linear filter* of the **observed** squared residuals :math:`\varepsilon^2`, so the fit recursion
  is a straight convolution (:func:`figarch_variance_filter`), not the coupled step-by-step
  recursion of the GARCH/EGF families. (Simulation *does* couple — there the innovations are drawn.)

The :math:`(1-B)^d` factor reuses the Phase-3 :func:`~quantica.timeseries.fegarch.fracdiff_coeffs`
engine at **positive** exponent ``+d`` (fractional *differencing*, vs FIEGARCH's ``-d``).

**Pre-sample (resolved, machine-exact).** ``presample=50`` and ``trunc="none"`` are `fEGarch`'s
defaults. Resolved against the fixture: the pre-sample terms :math:`\varepsilon_{t-i}^2` for
:math:`t-i \le 0` are seeded at the **unbiased sample variance** :math:`\operatorname{Var}(r)`
(``ddof=1``) for exactly the first **50** lags, and ``0`` beyond; ``trunc="none"`` means the
:math:`\theta(B)` series runs to full length (no early coefficient truncation). Both the ``ddof=1``
and the count ``50`` are decisive — ``ddof=0`` misses at 2.3e-6, and 49 or 51 terms miss at ~8e-6,
while 50 terms with ``ddof=1`` reconstructs the fixture :math:`\sigma`-series to 4.86e-17.

The parameter vector is ``(mu, omega, phi1, beta1, d)`` with :math:`d \in (0, 1)` (``fEGarch``'s
``drange``; the fitted ``d = 0.678`` is upper-regime). :math:`\phi_1, \beta_1` are the ARCH / GARCH
lag coefficients.

**Shared seam for the short-memory FI group.** :func:`figarch_coefficients` (the
:math:`\text{fracdiff}(+d) \star \text{ARMA}` :math:`\theta`-composition) and
:func:`figarch_variance_filter` (the ω-direct linear filter with the 50-term ``Var(ddof=1)``
pre-sample) are the reusable primitives that FIAPARCH / FITGARCH / FIGJR inherit — they differ only
in the *news* term fed to the filter (a power/asymmetry transform of :math:`\varepsilon` instead of
:math:`\varepsilon^2`), exactly as the EGF FI group shared ``type1_news_impact``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import fft as sp_fft

from quantica.timeseries.fegarch.distributions import get_distribution
from quantica.timeseries.fegarch.fracdiff import fracdiff_coeffs
from quantica.timeseries.fegarch.garch import GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "FIGARCH_PRESAMPLE",
    "figarch_coefficients",
    "figarch_recursion",
    "figarch_sim",
    "figarch_variance_filter",
    "fit_figarch",
]

#: `fEGarch`'s ``presample=50``: the number of pre-sample news terms seeded at ``Var(r, ddof=1)``.
FIGARCH_PRESAMPLE = 50

_VAR_NAMES = ("mu", "omega", "phi1", "beta1", "d")


def figarch_coefficients(phi1: float, beta1: float, d: float, length: int) -> FloatArray:
    r"""ARCH(∞) coefficients of :math:`\theta(B) = 1 - (1-\phi_1 B)(1-B)^d/(1-\beta_1 B)`.

    The causal convolution of the fractional-**differencing** coefficients :math:`b_i(+d)` of
    :math:`(1-B)^d` (Phase-3 :func:`fracdiff_coeffs` at *positive* exponent) with the MA factor
    :math:`(1-\phi_1 B)` and the geometric :math:`(1-\beta_1 B)^{-1}=\sum_j \beta_1^j B^j` series.
    ``d = 0`` recovers the short-memory GARCH :math:`\theta_i=(\phi_1-\beta_1)\beta_1^{i-1}`.

    Parameters
    ----------
    phi1 : float
        The ARCH lag coefficient (the :math:`\phi_1` of :math:`(1-\phi_1 B)`).
    beta1 : float
        The GARCH lag coefficient (``|beta1| < 1`` for the geometric expansion).
    d : float
        The fractional differencing order.
    length : int
        Number of coefficients ``theta_0 .. theta_{length-1}``.

    Returns
    -------
    ndarray, shape (length,)
        The coefficients, with ``theta[0] == 0`` (no contemporaneous :math:`\varepsilon_t^2` term)
        and ``theta[1] == d + phi1 - beta1``.
    """
    frac = fracdiff_coeffs(
        d, length
    )  # (1-B)^d coefficients (Phase-3, POSITIVE exponent = differencing)
    a = frac.copy()
    a[1:] -= phi1 * frac[:-1]  # (1 - phi1 B)(1 - B)^d
    geometric = beta1 ** np.arange(length, dtype=np.float64)  # (1 - beta1 B)^{-1}
    fft_len = sp_fft.next_fast_len(2 * length - 1)  # FFT convolution a * geometric (O(L log L))
    product = np.fft.rfft(a, fft_len) * np.fft.rfft(geometric, fft_len)
    c = np.fft.irfft(product, fft_len)[:length]  # (1-phi1 B)(1-B)^d / (1-beta1 B); c_0 = 1
    theta = -c
    theta[0] = 1.0 - c[0]  # theta(B) = 1 - c(B); c_0 = 1 => theta_0 = 0
    return np.asarray(theta, dtype=np.float64)


def figarch_variance_filter(
    news: FloatArray,
    theta: FloatArray,
    omega: float,
    presample_value: float,
    *,
    n_presample: int = FIGARCH_PRESAMPLE,
) -> FloatArray:
    r"""Apply the ω-direct ARCH(∞) filter :math:`\omega + \sum_i \theta_i\,\text{news}_{t-i}`.

    The shared variance-recursion primitive for the short-memory FI group. Because there is no
    feedback, the whole series is one causal convolution of ``theta`` with the ``news`` (observed at
    fit time), pre-padded by ``n_presample`` terms seeded at ``presample_value`` (0 beyond).

    Parameters
    ----------
    news : ndarray, shape (T,)
        The observed news series (:math:`\varepsilon^2` for FIGARCH; a power/asymmetry transform for
        FIAPARCH / FITGARCH / FIGJR).
    theta : ndarray
        The ARCH(∞) coefficients from :func:`figarch_coefficients`; must have length at least
        ``T + n_presample`` so the full ``trunc="none"`` series is used.
    omega : float
        The variance intercept, used **directly** (not ``(1-beta1)^{-1} omega``).
    presample_value : float
        The value seeding the first ``n_presample`` pre-sample news terms (``Var(r, ddof=1)`` for
        FIGARCH).
    n_presample : int, optional
        The number of pre-sample terms (default :data:`FIGARCH_PRESAMPLE` = 50).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`.

    Raises
    ------
    ValueError
        If ``theta`` is shorter than ``T + n_presample``.
    """
    news = np.asarray(news, dtype=np.float64)
    n = news.size
    if theta.size < n + n_presample:
        raise ValueError(f"theta must have length >= {n + n_presample} (got {theta.size})")
    extended = np.empty(n_presample + n, dtype=np.float64)
    extended[:n_presample] = presample_value  # pre-sample news at actual indices -n_presample .. -1
    extended[n_presample:] = news
    # sigma^2_t = omega + sum_i theta_i * extended[(t + n_presample) - i]  (theta_0 = 0)
    conv = np.convolve(theta, extended)
    return omega + conv[n_presample : n_presample + n]


def figarch_recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
    r"""FIGARCH(1,d,1) conditional variance :math:`\sigma_t^2` (a QMLE ``VarianceRecursion``).

    A pure linear filter of the observed squared residuals — no feedback (see the module docstring).

    Parameters
    ----------
    params : ndarray, shape (5,)
        ``(mu, omega, phi1, beta1, d)`` — ``omega`` is the variance intercept (used directly) and
        ``d`` the fractional differencing order.
    returns : ndarray, shape (T,)
        The return series.

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`, with the first
        :data:`FIGARCH_PRESAMPLE` pre-sample :math:`\varepsilon^2` seeded at ``Var(r, ddof=1)``.
    """
    mu, omega, phi1, beta1, d = (float(p) for p in params)
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    eps2 = (y - mu) ** 2
    presample_value = float(np.var(y, ddof=1))  # Var(r, ddof=1) — the machine-exact 50-term seed
    theta = figarch_coefficients(phi1, beta1, d, n + FIGARCH_PRESAMPLE)
    return figarch_variance_filter(eps2, theta, omega, presample_value)


def fit_figarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit FIGARCH(1,d,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``, the only validated one).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega, phi1, beta1, d``, log-likelihood, per-observation AIC/BIC, and the
        conditional-volatility series (original return units).
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude mean/omega
    scaled = y / scale
    distribution = get_distribution(cond_dist)

    var_start = (float(np.mean(scaled)), 0.05, 0.2, 0.5, 0.3)
    var_bounds = (
        (-10.0, 10.0),
        (1e-8, 100.0),  # omega > 0 (variance intercept)
        (1e-6, 0.9999),
        (1e-6, 0.9999),
        (1e-6, 0.9999),  # d in (0, 1) -- figarch's drange, NOT clamped at 0.5
    )

    result = quasi_max_likelihood(
        scaled,
        figarch_recursion,
        distribution,
        var_start=var_start,
        var_bounds=var_bounds,
        var_names=_VAR_NAMES,
        mean=True,
    )

    # Undo the scaling: mu scales (~scale); omega is a VARIANCE intercept, so it scales by scale^2
    # (multiplicatively, unlike the EGF omega_sig which shifts additively); phi1/beta1/d invariant.
    names = result.param_names
    values = np.asarray(result.params, dtype=np.float64).copy()
    std_errors_arr = np.asarray(result.std_errors, dtype=np.float64).copy()
    values[0] *= scale
    values[1] *= scale**2
    std_errors_arr[0] *= scale
    std_errors_arr[1] *= scale**2
    params = {name: float(v) for name, v in zip(names, values, strict=True)}
    std_errors = {name: float(s) for name, s in zip(names, std_errors_arr, strict=True)}

    loglik = result.loglikelihood - n * np.log(scale)  # Jacobian of the rescaling
    k = result.params.size
    aic = (2.0 * k - 2.0 * loglik) / n
    bic = (k * np.log(n) - 2.0 * loglik) / n
    conditional_volatility = np.sqrt(result.conditional_variance) * scale

    return GarchFit(
        cond_dist=cond_dist,
        params=params,
        std_errors=std_errors,
        loglikelihood=float(loglik),
        aic=float(aic),
        bic=float(bic),
        conditional_volatility=np.asarray(conditional_volatility, dtype=np.float64),
        n_obs=n,
        converged=result.converged,
    )


def figarch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega: float,
    phi1: float,
    beta1: float,
    d: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 1000,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a FIGARCH(1,d,1) process under a chosen conditional distribution.

    Unlike the fit (a pure filter of observed :math:`\varepsilon^2`), simulation **couples**:
    :math:`\varepsilon_t^2 = \sigma_t^2\eta_t^2` is generated, so :math:`\sigma_t^2` is built
    step-by-step from the past :math:`\varepsilon^2`. The pre-sample terms are seeded at the level
    :math:`\omega/(1-\beta_1)` and discarded via a long burn-in (long-memory transients
    decay slowly).

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega : float
        The variance intercept (``omega > 0``).
    phi1, beta1, d : float
        FIGARCH parameters; requires ``|beta1| < 1`` and ``0 <= d < 1``.
    cond_dist : str, optional
        Conditional-distribution code (default ``"norm"``).
    dist_params : tuple of float, optional
        Shape parameters for the distribution.
    rng : numpy.random.Generator
        Seeded generator (keyword-only).
    n_burn : int, optional
        Burn-in samples discarded (default 1000; larger than the EGF models' 500 because FIGARCH's
        long-memory transient decays hyperbolically).

    Returns
    -------
    tuple of ndarray
        ``(returns, sigma)`` of shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``n`` is not positive, ``omega <= 0``, ``|beta1| >= 1``, or ``d`` is outside ``[0, 1)``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if omega <= 0.0:
        raise ValueError("require omega > 0 for the variance intercept")
    if not abs(beta1) < 1.0:
        raise ValueError("require |beta1| < 1 for a stationary FIGARCH(1,d,1)")
    if not 0.0 <= d < 1.0:
        raise ValueError("require 0 <= d < 1 for the fractional order")

    distribution = get_distribution(cond_dist)
    total = n + n_burn
    eta = distribution.sample(total, rng, dist_params)
    theta = figarch_coefficients(phi1, beta1, d, total + FIGARCH_PRESAMPLE)
    seed = omega / (1.0 - beta1)  # rough operating variance level; washes out over the burn-in

    # eps2_hist holds the pre-sample seed (FIGARCH_PRESAMPLE terms) followed by generated eps2.
    eps2_hist = np.empty(FIGARCH_PRESAMPLE + total, dtype=np.float64)
    eps2_hist[:FIGARCH_PRESAMPLE] = seed
    sigma = np.empty(total, dtype=np.float64)
    for t in range(total):
        length = FIGARCH_PRESAMPLE + t  # number of available past news terms
        # sigma^2_t = omega + sum_{i=1}^{length} theta_i * eps2_hist[length - i]
        sigma2 = omega + float(theta[1 : length + 1] @ eps2_hist[length - 1 :: -1])
        sigma[t] = np.sqrt(sigma2)
        eps2_hist[FIGARCH_PRESAMPLE + t] = (sigma[t] * eta[t]) ** 2
    returns = mu + sigma * eta
    return (
        np.asarray(returns[n_burn:], dtype=np.float64),
        np.asarray(sigma[n_burn:], dtype=np.float64),
    )
