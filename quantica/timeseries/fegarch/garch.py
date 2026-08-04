r"""GARCH(1,1) — the first fEGarch short-memory model, on the Phase-0 QMLE engine (Phase 1).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Bollerslev 1986 for the recursion; WP 2026-04 App. C.3 for the QMLE conditioning), **never**
    the `fEGarch` source. Validated against committed `fEGarch` *output* fixtures.

The plain GARCH(1,1) conditional variance (Bollerslev 1986)

.. math::

    \sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2,
    \qquad \varepsilon_t = r_t - \mu,

with a constant mean :math:`\mu` (fEGarch's default mean specification, orders ``P=Q=D=0``) and
:math:`\omega > 0`, :math:`\alpha \ge 0`, :math:`\beta \ge 0`, :math:`\alpha + \beta < 1`.

The **model only supplies the variance recursion** (:func:`garch_recursion`); the likelihood,
optimizer and Hessian standard errors come from the shared Phase-0 engine
(:func:`~quantica.timeseries.fegarch.quasi_max_likelihood`), and all eight conditional distributions
are available through it. :func:`fit_garch` fits on internally rescaled returns for numerical
conditioning (the MLE is scale-equivariant) and converts the estimates back to the original units.

**Pre-sample conditioning (confirmed against the fixture, machine precision).** The recursion is
seeded with :math:`\sigma_0^2 = \varepsilon_0^2 = \operatorname{Var}(r)` — the **unbiased** sample
variance of the returns (``ddof=1``). Reconstructing the fEGarch fixture's conditional-SD series
from its reported parameters under this convention matches to ``~1e-17``; the biased
:math:`\overline{\varepsilon^2}` and the unconditional :math:`\omega/(1-\alpha-\beta)` do **not**
(``~1e-4`` and ``~1e-2`` relative). fEGarch's ``presample=50`` argument does not change this output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import get_distribution
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "GarchFit",
    "fit_garch",
    "garch_recursion",
    "garch_sim",
]

_VAR_NAMES = ("mu", "omega", "alpha", "beta")


def garch_recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
    r"""The GARCH(1,1) conditional-variance path :math:`\sigma_t^2` (a QMLE ``VarianceRecursion``).

    Parameters
    ----------
    params : ndarray, shape (4,)
        ``(mu, omega, alpha, beta)``.
    returns : ndarray, shape (T,)
        The return series.

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances, seeded with :math:`\sigma_0^2 = \varepsilon_0^2 =
        \operatorname{Var}(r)` (unbiased sample variance).
    """
    mu, omega, alpha, beta = (float(p) for p in params)
    y = np.asarray(returns, dtype=np.float64)
    resid = y - mu
    presample = float(np.var(y, ddof=1))  # sigma_0^2 = eps_0^2, mean-invariant
    sigma2 = np.empty(y.size, dtype=np.float64)
    sigma2[0] = omega + alpha * presample + beta * presample
    for t in range(1, y.size):
        sigma2[t] = omega + alpha * resid[t - 1] ** 2 + beta * sigma2[t - 1]
    return sigma2


@dataclass(frozen=True)
class GarchFit:
    """A fitted GARCH(1,1) model (parameters in the original return units).

    Attributes
    ----------
    cond_dist : str
        The conditional-distribution code (``"norm"``, ``"std"``, ...).
    params : dict of str to float
        Estimated parameters ``mu, omega, alpha, beta`` plus any distribution shape parameters.
    std_errors : dict of str to float
        Standard errors (inverse numerical Hessian), aligned with ``params``.
    loglikelihood : float
        Maximized log-likelihood on the original returns.
    aic, bic : float
        Information criteria **per observation** (``(2k - 2*loglik)/n`` and ``(k*ln(n) -
        2*loglik)/n``), matching fEGarch's reporting convention.
    conditional_volatility : ndarray
        The fitted conditional standard-deviation series :math:`\\sigma_t`.
    n_obs : int
        Number of observations.
    converged : bool
        Whether the optimizer reported success.
    """

    cond_dist: str
    params: dict[str, float]
    std_errors: dict[str, float]
    loglikelihood: float
    aic: float
    bic: float
    conditional_volatility: FloatArray
    n_obs: int
    converged: bool


def fit_garch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit GARCH(1,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``).

    Returns
    -------
    GarchFit
        The estimates (original units), log-likelihood, per-observation AIC/BIC, and the
        conditional-volatility series.
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude omega
    scaled = y / scale
    distribution = get_distribution(cond_dist)

    variance = float(np.var(scaled, ddof=1))
    var_start = (float(np.mean(scaled)), variance * 0.05, 0.05, 0.90)
    var_bounds = ((-10.0, 10.0), (1e-8, 1e6), (0.0, 0.9999), (0.0, 0.9999))

    result = quasi_max_likelihood(
        scaled,
        garch_recursion,
        distribution,
        var_start=var_start,
        var_bounds=var_bounds,
        var_names=_VAR_NAMES,
        mean=True,
    )

    # Undo the scaling: mu ~ scale, omega ~ scale^2, alpha/beta and shape params invariant.
    factors = np.array([scale, scale**2, 1.0, 1.0] + [1.0] * len(distribution.param_names))
    values = result.params * factors
    names = result.param_names
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


def garch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega: float,
    alpha: float,
    beta: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a GARCH(1,1) process under a chosen conditional distribution.

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega, alpha, beta : float
        GARCH parameters; requires ``alpha + beta < 1`` for stationarity.
    cond_dist : str, optional
        Conditional-distribution code for the standardized innovations (default ``"norm"``).
    dist_params : tuple of float, optional
        Shape parameters for the distribution (e.g. ``(nu,)`` for ``std``).
    rng : numpy.random.Generator
        Seeded generator (keyword-only).
    n_burn : int, optional
        Burn-in samples discarded so the recursion forgets its start (default 500).

    Returns
    -------
    tuple of ndarray
        ``(returns, sigma)`` of shape ``(n,)`` — the simulated returns and their conditional
        standard deviations.

    Raises
    ------
    ValueError
        If ``n`` is not positive or ``alpha + beta >= 1``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if alpha + beta >= 1.0:
        raise ValueError("require alpha + beta < 1 for a stationary GARCH(1,1)")

    total = n + n_burn
    innovations = get_distribution(cond_dist).sample(total, rng, dist_params)
    sigma2 = np.empty(total, dtype=np.float64)
    eps = np.empty(total, dtype=np.float64)
    sigma2[0] = omega / (1.0 - alpha - beta)  # unconditional variance
    eps[0] = np.sqrt(sigma2[0]) * innovations[0]
    for t in range(1, total):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * innovations[t]
    returns = mu + eps
    return (
        np.asarray(returns[n_burn:], dtype=np.float64),
        np.asarray(np.sqrt(sigma2[n_burn:]), dtype=np.float64),
    )
