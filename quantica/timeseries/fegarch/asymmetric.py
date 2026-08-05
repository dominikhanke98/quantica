r"""Asymmetric short-memory models — GJR-GARCH, TGARCH, APARCH (Phase 1).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Glosten, Jagannathan & Runkle 1993 for GJR; Zakoian 1994 for TGARCH; Ding, Granger & Engle
    1993 for APARCH; WP 2026-04 App. C.3 for the QMLE conditioning), **never** the `fEGarch` source.
    Validated against committed `fEGarch` *output* fixtures.

**One recursion, three models.** Reconciling the committed `fEGarch` fixtures shows that its
``gjrgarch``, ``tgarch`` and ``aparch`` are the **single APARCH power recursion**
(Ding-Granger-Engle 1993) at three values of the power :math:`\delta`:

.. math::

    \sigma_t^\delta = \omega
        + \phi_1\,\big(|\varepsilon_{t-1}| - \gamma_1\,\varepsilon_{t-1}\big)^{\delta}
        + \beta_1\,\sigma_{t-1}^\delta,
    \qquad \varepsilon_t = r_t - \mu,

with a constant mean :math:`\mu` and :math:`\omega,\phi_1,\beta_1 \ge 0`, :math:`|\gamma_1| < 1`,
:math:`\delta > 0`:

* **GJR-GARCH** — :math:`\delta = 2` (a recursion on the **variance** :math:`\sigma^2`). The
  ``(|\varepsilon| - \gamma\varepsilon)^2` kernel is the Glosten-Jagannathan-Runkle (1993)
  variance-threshold asymmetry in `fEGarch`'s parameterization: it equals
  :math:`\varepsilon^2(1-\gamma_1\operatorname{sign}\varepsilon)^2`, i.e. slope
  :math:`\phi_1(1-\gamma_1)^2` on good news and :math:`\phi_1(1+\gamma_1)^2` on bad news.
* **TGARCH** — :math:`\delta = 1` (a recursion on the **standard deviation** :math:`\sigma`), the
  Zakoian (1994) threshold-ARCH form :math:`|\varepsilon| - \gamma_1\varepsilon`. The intercept
  :math:`\omega` is therefore in :math:`\sigma`-units (~``2.3e-4`` here) versus GJR's
  :math:`\sigma^2`-units (~``3e-6``) — a scale difference the fixtures confirm.
* **APARCH** — :math:`\delta` a **free, continuously estimated** parameter (fitted ``~2.41`` on the
  synthetic series); ``fEGarch``'s default ``fix_delta = NA``. Unlike the ALD's discrete profiled
  ``P``, :math:`\delta` is part of the QMLE parameter vector with bounds :math:`\delta \in (0, 4]`.

Each model supplies only this recursion (returned as the **variance** :math:`\sigma_t^2` the engine
expects); the likelihood, optimizer and Hessian standard errors come from the shared Phase-0 engine
(:func:`~quantica.timeseries.fegarch.quasi_max_likelihood`), and all eight conditional distributions
route through it. The fits are done on internally rescaled returns (the MLE is scale-equivariant).

**Pre-sample conditioning (reconciled against the fixtures).** The recursion is seeded with

.. math::

    \sigma_0^\delta = \operatorname{Var}(r)^{\delta/2}, \qquad
    \text{kernel}_0 = \frac1n\sum_t |\varepsilon_t|^\delta,

i.e. the :math:`\sigma^\delta` state from the **unbiased** sample variance (``ddof=1``) and the
pre-sample news-impact from the **:math:`\delta`-th absolute sample moment**
:math:`\operatorname{E}|\varepsilon|^\delta` (the expected symmetric news impact — the leverage term
:math:`-\gamma_1\varepsilon` has zero pre-sample mean by symmetry). Reconstructing each fixture's
conditional-SD series from its reported parameters under this convention matches GJR to ``~3e-9``
and TGARCH to ``~1e-8``; the recursion **form** is machine-exact for all three (verified by seeding
from the fixture's own :math:`\sigma_0`). APARCH carries a larger pre-sample residual (``~9e-5``
absolute at :math:`\sigma_0`, decaying) because the :math:`\delta`-th absolute moment at the fitted
:math:`\delta \approx 2.41` does not exactly reproduce fEGarch's (unpublished) pre-sample state;
this is flagged in ``docs/fegarch-spec-notes.md`` and affects only :math:`\sigma_0`, not the
recursion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import get_distribution
from quantica.timeseries.fegarch.garch import GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.timeseries.fegarch.qmle import VarianceRecursion

__all__ = [
    "aparch_recursion",
    "aparch_sim",
    "fit_aparch",
    "fit_gjr",
    "fit_tgarch",
    "gjr_recursion",
    "gjr_sim",
    "tgarch_recursion",
    "tgarch_sim",
]

_GJR_TGARCH_NAMES = ("mu", "omega", "phi1", "beta1", "gamma1")
_APARCH_NAMES = ("mu", "omega", "phi1", "beta1", "gamma1", "delta")


def _aparch_family_variance(
    mu: float,
    omega: float,
    phi1: float,
    beta1: float,
    gamma1: float,
    delta: float,
    returns: FloatArray,
) -> FloatArray:
    r"""The APARCH-power recursion, returned as the conditional variance :math:`\sigma_t^2`.

    Implements :math:`\sigma_t^\delta = \omega + \phi_1(|\varepsilon_{t-1}| -
    \gamma_1\varepsilon_{t-1})^\delta + \beta_1\sigma_{t-1}^\delta` with the reconciled pre-sample
    (:math:`\sigma_0^\delta = \operatorname{Var}(r)^{\delta/2}`, news-impact
    :math:`\tfrac1n\sum|\varepsilon_t|^\delta`), then converts back to :math:`\sigma_t^2` for the
    QMLE engine. Since :math:`|\gamma_1| < 1` the kernel base :math:`|\varepsilon|(1 -
    \gamma_1\operatorname{sign}\varepsilon)` is non-negative, so the fractional power is real.
    """
    y = np.asarray(returns, dtype=np.float64)
    resid = y - mu
    n = y.size
    sig_delta_0 = float(np.var(y, ddof=1)) ** (delta / 2.0)
    kernel_0 = float(np.mean(np.abs(resid) ** delta))
    sig_delta = np.empty(n, dtype=np.float64)
    sig_delta[0] = omega + phi1 * kernel_0 + beta1 * sig_delta_0
    for t in range(1, n):
        kernel = (abs(resid[t - 1]) - gamma1 * resid[t - 1]) ** delta
        sig_delta[t] = omega + phi1 * kernel + beta1 * sig_delta[t - 1]
    return sig_delta ** (2.0 / delta)


def gjr_recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
    r"""GJR-GARCH(1,1) conditional variance (APARCH power :math:`\delta = 2`).

    Parameters
    ----------
    params : ndarray, shape (5,)
        ``(mu, omega, phi1, beta1, gamma1)``.
    returns : ndarray, shape (T,)
        The return series.

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`.
    """
    mu, omega, phi1, beta1, gamma1 = (float(p) for p in params)
    return _aparch_family_variance(mu, omega, phi1, beta1, gamma1, 2.0, returns)


def tgarch_recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
    r"""TGARCH(1,1) conditional variance (APARCH power :math:`\delta = 1`, on :math:`\sigma`).

    Parameters
    ----------
    params : ndarray, shape (5,)
        ``(mu, omega, phi1, beta1, gamma1)``.
    returns : ndarray, shape (T,)
        The return series.

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`.
    """
    mu, omega, phi1, beta1, gamma1 = (float(p) for p in params)
    return _aparch_family_variance(mu, omega, phi1, beta1, gamma1, 1.0, returns)


def aparch_recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
    r"""APARCH(1,1) conditional variance with a free power :math:`\delta`.

    Parameters
    ----------
    params : ndarray, shape (6,)
        ``(mu, omega, phi1, beta1, gamma1, delta)``.
    returns : ndarray, shape (T,)
        The return series.

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`.
    """
    mu, omega, phi1, beta1, gamma1, delta = (float(p) for p in params)
    return _aparch_family_variance(mu, omega, phi1, beta1, gamma1, delta, returns)


def _fit_aparch_family(
    returns: FloatArray, cond_dist: str, *, delta_fixed: float | None
) -> GarchFit:
    """Shared QMLE fit (``delta_fixed`` set for GJR/TGARCH, ``None`` for the free-delta APARCH)."""
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude omega
    scaled = y / scale
    distribution = get_distribution(cond_dist)

    variance = float(np.var(scaled, ddof=1))
    mean_start = float(np.mean(scaled))
    recursion: VarianceRecursion
    var_names: tuple[str, ...]
    var_start: tuple[float, ...]
    var_bounds: tuple[tuple[float, float], ...]
    # box bounds shared by every leading parameter (mean/omega/phi1/beta1/gamma1)
    _shared_bounds = ((-10.0, 10.0), (1e-8, 1e6), (0.0, 0.9999), (0.0, 0.9999), (-0.9999, 0.9999))
    if delta_fixed is None:
        recursion = aparch_recursion
        var_names = _APARCH_NAMES
        omega_start = variance ** (1.5 / 2.0) * 0.05
        var_start = (mean_start, omega_start, 0.05, 0.90, 0.0, 1.5)
        var_bounds = (*_shared_bounds, (0.05, 4.0))
    else:
        recursion = gjr_recursion if delta_fixed == 2.0 else tgarch_recursion
        var_names = _GJR_TGARCH_NAMES
        omega_start = variance ** (delta_fixed / 2.0) * 0.05
        var_start = (mean_start, omega_start, 0.05, 0.90, 0.0)
        var_bounds = _shared_bounds

    result = quasi_max_likelihood(
        scaled,
        recursion,
        distribution,
        var_start=var_start,
        var_bounds=var_bounds,
        var_names=var_names,
        mean=True,
    )

    names = result.param_names
    delta = float(delta_fixed) if delta_fixed is not None else float(result.params[5])
    # Undo the scaling: mu ~ scale, omega ~ scale^delta; phi1/beta1/gamma1/delta/shape invariant.
    n_extra = len(names) - len(var_names)  # distribution shape parameters
    if delta_fixed is None:
        var_factors = [scale, scale**delta, 1.0, 1.0, 1.0, 1.0]
    else:
        var_factors = [scale, scale**delta, 1.0, 1.0, 1.0]
    factors = np.array(var_factors + [1.0] * n_extra, dtype=np.float64)
    values = result.params * factors
    params = {name: float(v) for name, v in zip(names, values, strict=True)}
    std_errors = {
        name: float(se * f) for name, se, f in zip(names, result.std_errors, factors, strict=True)
    }

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


def fit_gjr(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit GJR-GARCH(1,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega, phi1, beta1, gamma1`` (+ shape parameters), log-likelihood,
        per-observation AIC/BIC, and the conditional-volatility series.
    """
    return _fit_aparch_family(returns, cond_dist, delta_fixed=2.0)


def fit_tgarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit TGARCH(1,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega, phi1, beta1, gamma1`` (+ shape parameters), log-likelihood,
        per-observation AIC/BIC, and the conditional-volatility series.
    """
    return _fit_aparch_family(returns, cond_dist, delta_fixed=1.0)


def fit_aparch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    r"""Fit APARCH(1,1) with a free power :math:`\delta` and a constant mean by QMLE.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega, phi1, beta1, gamma1, delta`` (+ shape parameters), log-likelihood,
        per-observation AIC/BIC, and the conditional-volatility series.
    """
    return _fit_aparch_family(returns, cond_dist, delta_fixed=None)


def _aparch_family_sim(
    n: int,
    mu: float,
    omega: float,
    phi1: float,
    beta1: float,
    gamma1: float,
    delta: float,
    cond_dist: str,
    dist_params: tuple[float, ...],
    rng: np.random.Generator,
    n_burn: int,
) -> tuple[FloatArray, FloatArray]:
    """Shared simulator for the APARCH family; returns ``(returns, sigma)`` after burn-in."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not abs(gamma1) < 1.0:
        raise ValueError("require |gamma1| < 1")
    total = n + n_burn
    innovations = get_distribution(cond_dist).sample(total, rng, dist_params)
    # Unconditional sigma^delta uses kappa = E[(|z| - gamma1 z)^delta] over the drawn innovations.
    kappa = float(np.mean((np.abs(innovations) - gamma1 * innovations) ** delta))
    persistence = beta1 + phi1 * kappa
    if persistence >= 1.0:
        raise ValueError("require beta1 + phi1 * E[(|z| - gamma1 z)^delta] < 1 for stationarity")
    sig_delta = np.empty(total, dtype=np.float64)
    eps = np.empty(total, dtype=np.float64)
    sig_delta[0] = omega / (1.0 - persistence)
    eps[0] = sig_delta[0] ** (1.0 / delta) * innovations[0]
    for t in range(1, total):
        kernel = (abs(eps[t - 1]) - gamma1 * eps[t - 1]) ** delta
        sig_delta[t] = omega + phi1 * kernel + beta1 * sig_delta[t - 1]
        eps[t] = sig_delta[t] ** (1.0 / delta) * innovations[t]
    returns = mu + eps
    sigma = sig_delta ** (1.0 / delta)
    return (
        np.asarray(returns[n_burn:], dtype=np.float64),
        np.asarray(sigma[n_burn:], dtype=np.float64),
    )


def gjr_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega: float,
    phi1: float,
    beta1: float,
    gamma1: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a GJR-GARCH(1,1) process (APARCH power :math:`\delta = 2`).

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega, phi1, beta1, gamma1 : float
        GJR parameters; requires ``|gamma1| < 1`` and stationarity ``beta1 + phi1 * E[kernel] < 1``.
    cond_dist : str, optional
        Conditional-distribution code for the standardized innovations (default ``"norm"``).
    dist_params : tuple of float, optional
        Shape parameters for the distribution.
    rng : numpy.random.Generator
        Seeded generator (keyword-only).
    n_burn : int, optional
        Burn-in samples discarded (default 500).

    Returns
    -------
    tuple of ndarray
        ``(returns, sigma)`` of shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``n`` is not positive, ``|gamma1| >= 1``, or the process is non-stationary.
    """
    return _aparch_family_sim(
        n, mu, omega, phi1, beta1, gamma1, 2.0, cond_dist, dist_params, rng, n_burn
    )


def tgarch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega: float,
    phi1: float,
    beta1: float,
    gamma1: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a TGARCH(1,1) process (APARCH power :math:`\delta = 1`).

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega, phi1, beta1, gamma1 : float
        TGARCH parameters; requires ``|gamma1| < 1`` and stationary ``beta1 + phi1*E[kernel] < 1``.
    cond_dist : str, optional
        Conditional-distribution code for the standardized innovations (default ``"norm"``).
    dist_params : tuple of float, optional
        Shape parameters for the distribution.
    rng : numpy.random.Generator
        Seeded generator (keyword-only).
    n_burn : int, optional
        Burn-in samples discarded (default 500).

    Returns
    -------
    tuple of ndarray
        ``(returns, sigma)`` of shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``n`` is not positive, ``|gamma1| >= 1``, or the process is non-stationary.
    """
    return _aparch_family_sim(
        n, mu, omega, phi1, beta1, gamma1, 1.0, cond_dist, dist_params, rng, n_burn
    )


def aparch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega: float,
    phi1: float,
    beta1: float,
    gamma1: float,
    delta: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate an APARCH(1,1) process with a free power :math:`\delta`.

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega, phi1, beta1, gamma1, delta : float
        APARCH parameters; requires ``|gamma1| < 1``, ``delta > 0`` and stationarity
        ``beta1 + phi1 * E[(|z| - gamma1 z)^delta] < 1``.
    cond_dist : str, optional
        Conditional-distribution code for the standardized innovations (default ``"norm"``).
    dist_params : tuple of float, optional
        Shape parameters for the distribution.
    rng : numpy.random.Generator
        Seeded generator (keyword-only).
    n_burn : int, optional
        Burn-in samples discarded (default 500).

    Returns
    -------
    tuple of ndarray
        ``(returns, sigma)`` of shape ``(n,)``.

    Raises
    ------
    ValueError
        If ``n`` is not positive, ``|gamma1| >= 1``, ``delta <= 0``, or the process is
        non-stationary.
    """
    if delta <= 0.0:
        raise ValueError("require delta > 0")
    return _aparch_family_sim(
        n, mu, omega, phi1, beta1, gamma1, delta, cond_dist, dist_params, rng, n_burn
    )
