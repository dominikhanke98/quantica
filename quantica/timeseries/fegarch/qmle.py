r"""Quasi-maximum-likelihood estimation engine for the fEGarch clean-room port (Phase 0).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent clean-room reimplementation from the published
    mathematics (fEGarch reference manual + cited papers), never from the fEGarch source; validated
    against committed fEGarch *output* fixtures (a later step).

A generic QMLE engine that fits any conditional-variance model to a mean-zero return series under
any standardized innovation distribution in :mod:`~quantica.timeseries.fegarch.distributions`.
It is the shared estimation machinery every fEGarch-family model plugs into; the models themselves
(the variance recursions) arrive in later phases.

**Likelihood.** With :math:`\varepsilon_t = \sigma_t z_t` and a standardized innovation density
:math:`f_z` (mean 0, variance 1), the conditional log-likelihood is

.. math::

    \ell(\theta) = \sum_{t} \Big[ -\ln \sigma_t(\theta)
    + \ln f_z\!\big(\varepsilon_t/\sigma_t(\theta)\big) \Big],

so the :math:`-\ln\sigma_t` Jacobian term carries the variance scale and the distribution supplies
the shape. The series is assumed mean-zero (mean modelling is a later "dual mean" phase); pass
residuals if a mean has been removed.

**Presample conditioning (documented convention).** The variance recursion is *conditional on
presample values*: the recursion function is responsible for initialising its pre-sample state
(:math:`\sigma_0^2`, :math:`\varepsilon_0^2`, ...). :func:`initial_variance` provides the default
this port uses — the **sample variance of the series** as the pre-sample :math:`\sigma_0^2`. This
is a standard, defensible conditioning; whether fEGarch uses exactly this (vs. a backcast or a fixed
warm-up length) is a ``# RECONCILE`` item to confirm against the fixtures, and models can pass their
own initialisation without changing this engine.

**Standard errors.** The variance-covariance matrix is the inverse of the numerical Hessian of the
negative log-likelihood at the optimum (observed information); standard errors are its square-root
diagonal. Optimization leans on :func:`scipy.optimize.minimize` with box bounds — the demonstrable
work is the likelihood construction and the conditioning, not the optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantica.core.types import FloatArray
    from quantica.timeseries.fegarch.distributions import ConditionalDistribution

__all__ = [
    "QMLEResult",
    "VarianceRecursion",
    "initial_variance",
    "quasi_max_likelihood",
]

_HESSIAN_STEP = 1e-4  # central-difference step for the numerical Hessian


def initial_variance(returns: FloatArray) -> float:
    """The default pre-sample conditional variance: the sample variance of the series.

    Parameters
    ----------
    returns : ndarray
        The (mean-zero) return series.

    Returns
    -------
    float
        The sample variance, used to condition the variance recursion's pre-sample state.
    """
    return float(np.var(np.asarray(returns, dtype=np.float64)))


class VarianceRecursion(Protocol):
    """A conditional-variance model: parameters + data → the conditional-variance path.

    The callable receives the variance parameters and the (mean-zero) return series and returns the
    conditional variances :math:`\\sigma_t^2` for every observation, **conditioned on its own
    pre-sample initialisation** (see :func:`initial_variance`). It must return strictly positive
    values for admissible parameters.
    """

    def __call__(self, params: FloatArray, returns: FloatArray) -> FloatArray:
        """Return the conditional-variance path for ``params`` on ``returns``."""
        ...


@dataclass(frozen=True)
class QMLEResult:
    """The outcome of a quasi-maximum-likelihood fit.

    Attributes
    ----------
    params : ndarray
        The estimated parameters, variance parameters first then distribution shape parameters.
    param_names : tuple of str
        Names aligned with ``params``.
    loglikelihood : float
        The maximized conditional log-likelihood.
    std_errors : ndarray
        Standard errors from the inverse numerical Hessian (``nan`` if it is not positive-definite).
    vcov : ndarray
        The variance-covariance matrix (inverse observed information).
    conditional_variance : ndarray
        The fitted conditional-variance path at the optimum.
    n_var_params : int
        How many leading entries of ``params`` are variance parameters (the rest are distribution
        shape parameters).
    converged : bool
        Whether the optimizer reported success.
    """

    params: FloatArray
    param_names: tuple[str, ...]
    loglikelihood: float
    std_errors: FloatArray
    vcov: FloatArray
    conditional_variance: FloatArray
    n_var_params: int
    converged: bool


def _numerical_hessian(func, x: FloatArray, step: float) -> FloatArray:  # type: ignore[no-untyped-def]
    """Central-difference Hessian of a scalar ``func`` at ``x``."""
    n = x.size
    hessian = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            x_pp, x_pm, x_mp, x_mm = x.copy(), x.copy(), x.copy(), x.copy()
            x_pp[i] += step
            x_pp[j] += step
            x_pm[i] += step
            x_pm[j] -= step
            x_mp[i] -= step
            x_mp[j] += step
            x_mm[i] -= step
            x_mm[j] -= step
            value = (func(x_pp) - func(x_pm) - func(x_mp) + func(x_mm)) / (4.0 * step * step)
            hessian[i, j] = hessian[j, i] = value
    return hessian


def quasi_max_likelihood(
    returns: FloatArray,
    variance_recursion: VarianceRecursion,
    distribution: ConditionalDistribution,
    *,
    var_start: Sequence[float],
    var_bounds: Sequence[tuple[float, float]],
    var_names: Sequence[str],
    dist_start: Sequence[float] | None = None,
    dist_bounds: Sequence[tuple[float, float]] | None = None,
    mean: bool = False,
) -> QMLEResult:
    r"""Fit a conditional-variance model by quasi-maximum likelihood.

    Maximizes :math:`\sum_t [-\ln\sigma_t + \ln f_z(\varepsilon_t/\sigma_t)]` over the variance
    parameters and the distribution's shape parameters jointly, conditioned on the recursion's
    pre-sample initialisation. With ``mean=True`` a constant mean :math:`\mu` is the **first**
    variance parameter and the standardized residual is :math:`(y-\mu)/\sigma` (the recursion is
    passed the raw ``y`` and forms :math:`\varepsilon = y - \mu` itself); with ``mean=False``
    (default) the series is treated as mean-zero and :math:`z = y/\sigma`.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series (mean-zero unless ``mean=True``).
    variance_recursion : VarianceRecursion
        The model mapping variance parameters + returns to the conditional-variance path.
    distribution : ConditionalDistribution
        The standardized innovation distribution (its shape parameters are estimated too).
    var_start, var_bounds, var_names : sequences
        Starting values, ``(low, high)`` bounds, and names for the variance parameters (the first
        entry is the constant mean :math:`\mu` when ``mean=True``).
    dist_start, dist_bounds : sequences, optional
        Starting values and bounds for the distribution's shape parameters; default to the
        distribution's own ``param_start`` / ``param_bounds``.
    mean : bool, optional
        Treat the first variance parameter as a constant mean :math:`\mu` used in the standardized
        residual (default ``False``, i.e. mean-zero).

    Returns
    -------
    QMLEResult
        The estimates, log-likelihood, standard errors, and fitted conditional variance.

    Raises
    ------
    ValueError
        If the variance metadata lengths disagree.
    """
    y = np.asarray(returns, dtype=np.float64)
    n_var = len(var_start)
    if not (len(var_bounds) == len(var_names) == n_var):
        raise ValueError("var_start, var_bounds and var_names must have the same length")
    d_start = tuple(distribution.param_start if dist_start is None else dist_start)
    d_bounds = tuple(distribution.param_bounds if dist_bounds is None else dist_bounds)

    start = np.array([*var_start, *d_start], dtype=np.float64)
    bounds = [*var_bounds, *d_bounds]

    def negative_loglik(theta: FloatArray) -> float:
        var_params = theta[:n_var]
        dist_params = tuple(float(p) for p in theta[n_var:])
        sigma2 = np.asarray(variance_recursion(var_params, y), dtype=np.float64)
        if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0.0):
            return 1e10
        sigma = np.sqrt(sigma2)
        residuals = y - var_params[0] if mean else y
        z = residuals / sigma
        loglik = np.sum(-np.log(sigma) + distribution.logpdf(z, dist_params))
        return float(-loglik) if np.isfinite(loglik) else 1e10

    optimum = minimize(negative_loglik, start, method="L-BFGS-B", bounds=bounds)
    theta = np.asarray(optimum.x, dtype=np.float64)

    hessian = _numerical_hessian(negative_loglik, theta, _HESSIAN_STEP)
    try:
        vcov = np.linalg.inv(hessian)
        diagonal = np.diag(vcov)
        std_errors = np.where(diagonal > 0.0, np.sqrt(np.abs(diagonal)), np.nan)
    except np.linalg.LinAlgError:  # pragma: no cover - singular Hessian is rare on real fits
        vcov = np.full((theta.size, theta.size), np.nan)
        std_errors = np.full(theta.size, np.nan)

    sigma2 = np.asarray(variance_recursion(theta[:n_var], y), dtype=np.float64)
    return QMLEResult(
        params=theta,
        param_names=(*var_names, *distribution.param_names),
        loglikelihood=float(-optimum.fun),
        std_errors=np.asarray(std_errors, dtype=np.float64),
        vcov=np.asarray(vcov, dtype=np.float64),
        conditional_variance=sigma2,
        n_var_params=n_var,
        converged=bool(optimum.success),
    )
