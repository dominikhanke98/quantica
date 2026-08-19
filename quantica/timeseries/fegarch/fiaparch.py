r"""FIAPARCH(1,d,1) — fractionally-integrated APARCH (Phase 4, δ-power variance-recursion).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Ding-Granger-Engle 1993 APARCH; Tse 1998 FIAPARCH; Baillie-Bollerslev-Mikkelsen 1996 +
    Conrad-Haag 2006 for the FIGARCH ARCH(∞); WP175 §2.1 Eqs. 2.9-2.10 for the ω-direct form),
    **never** the `fEGarch` source. Validated against two committed `fEGarch` output fixtures (a
    boundary d≈1 and an interior d≈0). The presample-seed convention is a **documented bounded
    limit** — see below.

FIAPARCH is FIGARCH's variance-recursion seam raised to the APARCH **power/asymmetry** kernel. WP175
Eq. 2.10 gives it as the **same ARCH(∞) coefficient sequence** :math:`\{\theta_i\}` as FIGARCH,
applied to the APARCH news :math:`(|\varepsilon|-\gamma\varepsilon)^\delta`:

.. math::

    \sigma_t^\delta = \omega + \sum_{i=1}^{\infty}\theta_i\,
    (|\varepsilon_{t-i}|-\gamma\varepsilon_{t-i})^\delta,\qquad \varepsilon_t = r_t - \mu,\qquad
    \theta(B) = 1 - \frac{(1-\phi_1 B)(1-B)^d}{1-\beta_1 B},

then :math:`\sigma_t = (\sigma_t^\delta)^{1/\delta}`. So it is built **entirely by reuse**:
:func:`~quantica.timeseries.fegarch.figarch_coefficients` (unchanged — WP175 confirms the identical
θ(B), :math:`\theta_1 = d+\phi_1-\beta_1`) and
:func:`~quantica.timeseries.fegarch.figarch_variance_filter` (the ω-direct linear filter) with the
**news term swapped** from :math:`\varepsilon^2` to
:math:`(|\varepsilon|-\gamma\varepsilon)^\delta`. ω is in ``sigma^delta`` units, used directly.
There is no feedback — the ``sigma^delta`` is a pure linear filter of the observed news (fit-time).

The parameter vector is ``(mu, omega, phi1, beta1, gamma, delta, d)`` — FIGARCH's four plus APARCH's
leverage :math:`\gamma` (``|gamma| < 1``) and estimated power :math:`\delta > 0`, plus
fractional :math:`d`. **Reduction anchors** (WP175): :math:`\delta=2, \gamma=0` gives FIGARCH (news
:math:`(|\varepsilon|)^2 = \varepsilon^2`); :math:`d\to 0` gives short-memory APARCH.

**The Conrad-Haag non-negativity floor.** :math:`\theta_1 = d+\phi_1-\beta_1 \ge 0` requires
:math:`d \ge \beta_1-\phi_1`, so a high GARCH lag :math:`\beta_1` **forces** :math:`d` toward 1
for a non-negative conditional variance. This is why the high-persistence fixture fits
:math:`d \approx 0.9999999` (β₁=0.913 ⇒ d ≥ 0.912) while the low-persistence fixture fits
:math:`d \approx 0`. The QMLE ``d``-bound therefore spans essentially all of ``(0, 1)``, and at the
boundary the parameters are weakly identified (a near-flat likelihood ridge — the fixture fit is
reproduced in log-likelihood, not param-by-param there).

**Presample seed — a documented bounded limit (irreducible-from-output).** FIGARCH seeds its 50-term
presample at the sample mean of *its* news (``Var(r, ddof=1)`` = sample mean of
:math:`\varepsilon^2`). FIAPARCH generalizes that rule: the 50-term presample is seeded at the
**sample mean of the news** :math:`\operatorname{mean}[(|\varepsilon|-\gamma\varepsilon)^\delta]`
(which reduces to FIGARCH's ``Var(ddof=1)`` at :math:`\delta=2, \gamma=0` up to the μ-vs-r̄/ddof
difference). This is **not** `fEGarch`'s exact presample value: two fixtures at
:math:`d\approx 0` and :math:`d\approx 1` disconfirmed **every** closed form — the closest,
:math:`\operatorname{Var}^{\delta/2}\!\operatorname{E}[(|z|-\gamma z)^\delta]`, is 1.6% off at
interior d and 44% off at the boundary; ``mean(news)`` is exact to **0.07%** at interior d but
carries a **d-dependent inflation** (factor ``1.0 → 1.49`` as ``d: 0 → 1``). The exact `fEGarch`
value is an internal backcast not recoverable from output — the **second irreducible-from-output**
limit in the
port (after the APARCH :math:`\sigma_0` fork). The **seam itself is machine-exact** (the
recursion at a fixture's own empirical seed reproduces its sigma-series to ~1e-16); only the seed
*convention* carries a residual that grows from ~1e-6 (interior d) to ~1e-3 (boundary d≈1, where
the long-memory transient decays slowly). This bound is asserted honestly in the tests, not hidden.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import get_distribution
from quantica.timeseries.fegarch.figarch import (
    FIGARCH_PRESAMPLE,
    figarch_coefficients,
    figarch_variance_filter,
)
from quantica.timeseries.fegarch.garch import GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "fiaparch_news",
    "fiaparch_recursion",
    "fiaparch_sim",
    "fit_fiaparch",
]

_VAR_NAMES = ("mu", "omega", "phi1", "beta1", "gamma", "delta", "d")


def fiaparch_news(resid: FloatArray, gamma: float, delta: float) -> FloatArray:
    r"""The APARCH power-asymmetry news :math:`(|\varepsilon| - \gamma\varepsilon)^\delta`.

    Non-negative for ``|gamma| < 1`` (Ding-Granger-Engle 1993). ``gamma > 0`` amplifies negative
    shocks (leverage). At ``gamma = 0, delta = 2`` this is :math:`\varepsilon^2` — FIGARCH's news.

    Parameters
    ----------
    resid : ndarray, shape (T,)
        The residuals :math:`\varepsilon = r - \mu`.
    gamma : float
        The leverage parameter (``|gamma| < 1``).
    delta : float
        The power (``delta > 0``).

    Returns
    -------
    ndarray, shape (T,)
        The power-asymmetry news term.
    """
    resid = np.asarray(resid, dtype=np.float64)
    return np.asarray((np.abs(resid) - gamma * resid) ** delta, dtype=np.float64)


def fiaparch_recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
    r"""FIAPARCH(1,d,1) conditional variance :math:`\sigma_t^2` (a QMLE ``VarianceRecursion``).

    Reuses the FIGARCH ARCH(∞) coefficients and ω-direct filter with the APARCH power news; the
    :math:`\sigma^\delta` output is mapped back to :math:`\sigma^2 = (\sigma^\delta)^{2/\delta}`.

    Parameters
    ----------
    params : ndarray, shape (7,)
        ``(mu, omega, phi1, beta1, gamma, delta, d)`` — ``omega`` is the ``sigma^delta`` intercept
        (used directly), ``gamma`` the leverage, ``delta`` the power, ``d`` the fractional order.
    returns : ndarray, shape (T,)
        The return series.

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`. The 50-term presample news is seeded at the
        sample mean of the news (the FIGARCH-generalized bounded-limit convention).
    """
    mu, omega, phi1, beta1, gamma, delta, d = (float(p) for p in params)
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    news = fiaparch_news(y - mu, gamma, delta)
    presample_value = float(np.mean(news))  # mean(news): FIGARCH's Var(ddof=1) seed generalized
    theta = figarch_coefficients(phi1, beta1, d, n + FIGARCH_PRESAMPLE)
    sigma_delta = figarch_variance_filter(news, theta, omega, presample_value)
    return np.asarray(
        sigma_delta ** (2.0 / delta), dtype=np.float64
    )  # sigma^2 = (sigma^delta)^(2/d)


def fit_fiaparch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit FIAPARCH(1,d,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``, the only validated one).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega, phi1, beta1, gamma, delta, d``, log-likelihood, per-observation
        AIC/BIC, and the conditional-volatility series (original return units).
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude mean/omega
    scaled = y / scale
    distribution = get_distribution(cond_dist)

    var_start = (float(np.mean(scaled)), 0.05, 0.2, 0.5, 0.05, 1.5, 0.3)
    var_bounds = (
        (-10.0, 10.0),
        (1e-8, 100.0),  # omega > 0 (sigma^delta intercept)
        (1e-6, 0.9999),
        (1e-6, 0.9999),
        (-0.9999, 0.9999),  # gamma leverage, |gamma| < 1
        (0.25, 4.0),  # delta > 0 (APARCH power)
        (1e-7, 0.9999999),  # d in (0, 1) -- must permit the boundary d approx 1 (high-beta1 series)
    )

    result = quasi_max_likelihood(
        scaled,
        fiaparch_recursion,
        distribution,
        var_start=var_start,
        var_bounds=var_bounds,
        var_names=_VAR_NAMES,
        mean=True,
    )

    # Undo the scaling: mu scales (~scale); omega is a sigma^delta intercept, so it scales by
    # scale^delta; phi1/beta1/gamma/delta/d are scale-invariant.
    names = result.param_names
    values = np.asarray(result.params, dtype=np.float64).copy()
    std_errors_arr = np.asarray(result.std_errors, dtype=np.float64).copy()
    delta_hat = float(values[5])
    values[0] *= scale
    values[1] *= scale**delta_hat
    std_errors_arr[0] *= scale
    std_errors_arr[1] *= scale**delta_hat
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


def fiaparch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega: float,
    phi1: float,
    beta1: float,
    gamma: float,
    delta: float,
    d: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 1000,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a FIAPARCH(1,d,1) process under a chosen conditional distribution.

    Like FIGARCH, simulation **couples** (the news :math:`(|\varepsilon|-\gamma\varepsilon)^\delta`
    is generated from :math:`\varepsilon = \sigma\eta`), so :math:`\sigma_t^\delta` is built
    step-by-step. Pre-sample news is seeded at :math:`\omega/(1-\beta_1)` and discarded via a long
    burn-in.

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega : float
        The sigma^delta intercept (``omega > 0``).
    phi1, beta1, gamma, delta, d : float
        FIAPARCH parameters; requires ``|beta1| < 1``, ``|gamma| < 1``, ``delta > 0`` and
        ``0 <= d < 1``.
    cond_dist : str, optional
        Conditional-distribution code (default ``"norm"``).
    dist_params : tuple of float, optional
        Shape parameters for the distribution.
    rng : numpy.random.Generator
        Seeded generator (keyword-only).
    n_burn : int, optional
        Burn-in samples discarded (default 1000).

    Returns
    -------
    tuple of ndarray
        ``(returns, sigma)`` of shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``n`` is not positive, ``omega <= 0``, ``|beta1| >= 1``, ``|gamma| >= 1``,
        ``delta <= 0``, or ``d`` is outside ``[0, 1)``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if omega <= 0.0:
        raise ValueError("require omega > 0 for the sigma^delta intercept")
    if not abs(beta1) < 1.0:
        raise ValueError("require |beta1| < 1 for a stationary FIAPARCH(1,d,1)")
    if not abs(gamma) < 1.0:
        raise ValueError("require |gamma| < 1 for a non-negative news term")
    if delta <= 0.0:
        raise ValueError("require delta > 0 for the APARCH power")
    if not 0.0 <= d < 1.0:
        raise ValueError("require 0 <= d < 1 for the fractional order")

    distribution = get_distribution(cond_dist)
    total = n + n_burn
    eta = distribution.sample(total, rng, dist_params)
    theta = figarch_coefficients(phi1, beta1, d, total + FIGARCH_PRESAMPLE)
    seed = omega / (1.0 - beta1)  # rough sigma^delta level; washes out over the burn-in

    news_hist = np.empty(FIGARCH_PRESAMPLE + total, dtype=np.float64)
    news_hist[:FIGARCH_PRESAMPLE] = seed
    sigma = np.empty(total, dtype=np.float64)
    for t in range(total):
        length = FIGARCH_PRESAMPLE + t
        sigma_delta = omega + float(theta[1 : length + 1] @ news_hist[length - 1 :: -1])
        sigma[t] = sigma_delta ** (1.0 / delta)
        eps_t = sigma[t] * eta[t]
        news_hist[FIGARCH_PRESAMPLE + t] = (abs(eps_t) - gamma * eps_t) ** delta
    returns = mu + sigma * eta
    return (
        np.asarray(returns[n_burn:], dtype=np.float64),
        np.asarray(sigma[n_burn:], dtype=np.float64),
    )
