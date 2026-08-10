r"""EGARCH(1,1) — the first EGARCH-family (EGF) model, on the Phase-0 QMLE engine (Phase 2).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Nelson 1991 for EGARCH; WP 2026-04 §2.1 + App. C.3 for the EGF spec and QMLE conditioning;
    the EGF papers Ayensu et al. 2026 / Peitz et al. 2026), **never** the `fEGarch` source.
    Validated against committed `fEGarch` *output* fixtures.

EGARCH is the **Type-I** EGF model (an explicit asymmetry term). Its log-variance recursion, in the
package's representation (WP 2026-04 Eq. 5) for orders ``(1, 1)``, is

.. math::

    r_t = \mu + \sigma_t\,\eta_t,\qquad
    \ln\sigma_t^2 = \omega + g(\eta_{t-1}) + \phi_1\,\ln\sigma_{t-1}^2,

with the **magnitude/asymmetry transformation** (Nelson 1991; WP 2026-04 Eq. 3)

.. math::

    g(\eta) = \kappa\,\eta + \gamma\,\big(|\eta| - \operatorname{E}|\eta|\big),

where :math:`\kappa` weights the **asymmetry** term (on :math:`\eta`) and :math:`\gamma` the
**magnitude** term (on :math:`|\eta| - \operatorname{E}|\eta|`) — the WP 2026-04 / WP173
orientation, confirmed by the fixture (:math:`\kappa < 0` leverage, :math:`\gamma > 0` magnitude).
(Note WP175 swaps the :math:`\kappa`/:math:`\gamma` letters; `fEGarch` uses the orientation here.)
The :math:`\operatorname{E}(\eta) = 0` term drops out because :math:`\eta` is standardized, so only
the :math:`\operatorname{E}|\eta|` centering survives — making :math:`\operatorname{E}[g] = 0`.

**Reported intercept.** `fEGarch` reports :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]`
(the *unconditional mean* of the log-variance, named ``omega_sig``), not the recursion intercept.
They are linked by :math:`\omega = \omega_\sigma\,\phi(1) = \omega_\sigma\,(1 - \phi_1)`, applied
internally. The parameter vector is ``(mu, omega_sig, phi1, kappa, gamma)`` (+ any distribution
shape parameters); for orders ``(1, 1)`` there is no ``psi`` term (``q - 1 = 0``).

**Pre-sample conditioning (confirmed against the fixture, machine precision).** Per WP 2026-04
App. C.3, the pre-sample news-impact history is zero (:math:`g(\eta_t) = 0` for :math:`t \le 0`) and
the pre-sample log-variance is the log of the **unbiased** sample variance:
:math:`\ln\sigma_0^2 = \omega + \phi_1\ln\operatorname{Var}(r)` with ``ddof=1``. Reconstructing the
fixture's :math:`\sigma`-series from its reported parameters under this convention matches to
``~4e-17`` (``ddof=0`` gives ``~2e-6``); the whole series, not just :math:`\sigma_0`, is reproduced.

**Distribution seam.** :math:`\operatorname{E}|\eta|` is sourced from the Phase-0 distribution
layer's :meth:`abs_moment` (``norm`` → :math:`\sqrt{2/\pi}`), passed into the recursion as a
captured value — never hard-coded. Only the normal is wired and validated here; jointly-shaped
distributions (``std`` / ``ged``) and the skewed variants are the documented Phase-2 follow-up (see
``docs/fegarch-spec-notes.md``): the FS-skew wrapper does not yet expose ``abs_moment`` for the
standardized skewed variable, and a jointly-estimated shape needs :math:`\operatorname{E}|\eta|`
recomputed at the current shape rather than captured once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import get_distribution
from quantica.timeseries.fegarch.garch import GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "egarch_recursion",
    "egarch_sim",
    "fit_egarch",
]

_VAR_NAMES = ("mu", "omega_sig", "phi1", "kappa", "gamma")


def egarch_recursion(params: FloatArray, returns: FloatArray, *, abs_moment: float) -> FloatArray:
    r"""EGARCH(1,1) conditional variance :math:`\sigma_t^2` (a QMLE ``VarianceRecursion``).

    Parameters
    ----------
    params : ndarray, shape (5,)
        ``(mu, omega_sig, phi1, kappa, gamma)`` — ``omega_sig`` is
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]` (the recursion intercept
        :math:`\omega = \omega_\sigma(1 - \phi_1)` is applied internally).
    returns : ndarray, shape (T,)
        The return series.
    abs_moment : float
        :math:`\operatorname{E}|\eta|`, the first absolute moment of the standardized innovation,
        from the conditional distribution's :meth:`abs_moment` (keyword-only).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`, seeded with
        :math:`\ln\sigma_0^2 = \omega + \phi_1\ln\operatorname{Var}(r)` (``ddof=1``) and a zero
        pre-sample news impact.
    """
    mu, omega_sig, phi1, kappa, gamma = (float(p) for p in params)
    y = np.asarray(returns, dtype=np.float64)
    resid = y - mu
    n = y.size
    omega = omega_sig * (1.0 - phi1)  # recursion intercept omega = omega_sig * phi(1)
    log_var = np.empty(n, dtype=np.float64)
    log_var[0] = omega + phi1 * np.log(np.var(y, ddof=1))  # g pre-sample = 0
    for t in range(1, n):
        eta = resid[t - 1] / np.exp(log_var[t - 1] / 2.0)
        g = kappa * eta + gamma * (abs(eta) - abs_moment)
        log_var[t] = omega + g + phi1 * log_var[t - 1]
    return np.exp(log_var)


def fit_egarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit EGARCH(1,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``). Only ``"norm"`` is
        validated in this phase (see the module note on the distribution seam).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega_sig, phi1, kappa, gamma`` (+ shape parameters), log-likelihood,
        per-observation AIC/BIC, and the conditional-volatility series (original return units).
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude mean
    scaled = y / scale
    distribution = get_distribution(cond_dist)

    # E|eta| from the distribution layer (norm -> sqrt(2/pi)); captured, not hard-coded.
    abs_moment = distribution.abs_moment(distribution.param_start)

    def recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
        return egarch_recursion(params, returns, abs_moment=abs_moment)

    var_start = (float(np.mean(scaled)), float(np.log(np.var(scaled, ddof=1))), 0.9, 0.0, 0.1)
    var_bounds = ((-10.0, 10.0), (-50.0, 50.0), (-0.9999, 0.9999), (-5.0, 5.0), (-5.0, 5.0))

    result = quasi_max_likelihood(
        scaled,
        recursion,
        distribution,
        var_start=var_start,
        var_bounds=var_bounds,
        var_names=_VAR_NAMES,
        mean=True,
    )

    # Undo the scaling: mu is multiplicative (~scale); omega_sig is a log-variance intercept, so it
    # shifts ADDITIVELY by ln(scale^2); phi1/kappa/gamma and shape parameters are scale-invariant.
    names = result.param_names
    values = np.asarray(result.params, dtype=np.float64).copy()
    std_errors_arr = np.asarray(result.std_errors, dtype=np.float64).copy()
    values[0] *= scale
    values[1] += np.log(scale**2)
    std_errors_arr[0] *= scale  # additive shift on omega_sig leaves its SE unchanged
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


def egarch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega_sig: float,
    phi1: float,
    kappa: float,
    gamma: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate an EGARCH(1,1) process under a chosen conditional distribution.

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega_sig : float
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]`, the unconditional log-variance.
    phi1, kappa, gamma : float
        EGARCH parameters; requires ``|phi1| < 1`` for stationarity of the log-variance.
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
        ``(returns, sigma)`` of shape ``(n,)`` — the simulated returns and conditional SDs.

    Raises
    ------
    ValueError
        If ``n`` is not positive or ``|phi1| >= 1``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not abs(phi1) < 1.0:
        raise ValueError("require |phi1| < 1 for a stationary EGARCH(1,1)")

    distribution = get_distribution(cond_dist)
    abs_moment = distribution.abs_moment(dist_params)
    omega = omega_sig * (1.0 - phi1)
    total = n + n_burn
    eta = distribution.sample(total, rng, dist_params)
    log_var = np.empty(total, dtype=np.float64)
    log_var[0] = omega_sig  # start at the unconditional mean E[ln sigma^2]
    for t in range(1, total):
        g = kappa * eta[t - 1] + gamma * (abs(eta[t - 1]) - abs_moment)
        log_var[t] = omega + g + phi1 * log_var[t - 1]
    sigma = np.exp(log_var / 2.0)
    returns = mu + sigma * eta
    return (
        np.asarray(returns[n_burn:], dtype=np.float64),
        np.asarray(sigma[n_burn:], dtype=np.float64),
    )
