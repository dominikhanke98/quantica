r"""FILog-GARCH(1,d,1) — fractionally-integrated Log-GARCH (Type-II EGF, Phase 4, last model).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Geweke 1986 / Pantula 1986 / Milhøj 1987 Log-GARCH; Feng et al. 2020a FILog-GARCH; WP 2026-04
    §2.1 Eqs. 10-13 for the Type-II fractional form), **never** the `fEGarch` source. Validated
    against the committed `fEGarch` output fixture.

FILog-GARCH is the **Type-II** counterpart of FIEGARCH: the fractionally-integrated Log-GARCH. Where
FIEGARCH's truncated MA(∞) loads the Type-I news impact :math:`g(\eta)`, FILog-GARCH loads the
**log-square innovation** :math:`\xi_t = \ln(\eta_t^2) - \operatorname{E}[\ln\eta_t^2]` (Log-GARCH's
news). WP 2026-04 Eqs. 10-11 give it as

.. math::

    \ln\sigma_t^2 = \omega_\sigma + \gamma(B)\xi_t,\qquad
    \gamma(B) = \phi^{-1}(B)(1-B)^{-d}\psi(B) - 1 = \sum_{i=1}^{\infty}\gamma_i B^i,

which for orders ``(1, d, 1)`` (:math:`\phi(B)=1-\phi_1 B`, :math:`\psi(B)=1+\psi_1 B`) is
:math:`\gamma(B) = (1-\phi_1 B)^{-1}(1-B)^{-d}(1+\psi_1 B) - 1`, with :math:`\gamma_0 = 0`
(the ``-1`` drops the constant term, so the news loads from lag 1:
:math:`\ln\sigma_t^2 = \omega_\sigma + \sum_{i\ge 1}\gamma_i\xi_{t-i}`). So it is built **by
composition**, reusing FIEGARCH's
:func:`~quantica.timeseries.fegarch.theta_coefficients` (the geometric :math:`\phi^{-1}` ⊛ Phase-3
:func:`~quantica.timeseries.fegarch.fracdiff_coeffs` at ``-d``, fractional *integration*) convolved
with the **Type-II MA factor** :math:`(1+\psi_1 B)`. The log-square centering
:math:`\operatorname{E}[\ln\eta^2]` reuses the Log-GARCH distribution moment
:meth:`~quantica.timeseries.fegarch.ConditionalDistribution.mean_log_sq` (``norm`` →
:math:`-\gamma_E - \ln 2 = -1.2703628`).

The parameter vector is ``(mu, omega_sig, phi1, psi1, d)`` — short-memory Log-GARCH's Type-II set
plus the fractional :math:`d`; :math:`\psi_1` enters the :math:`\gamma(B)` **numerator**
:math:`(1+\psi_1 B)` (the ARMA MA structure). The **MA(∞) pre-sample** is FIEGARCH's: the intercept
is :math:`\omega_\sigma` directly (no ``ln Var`` seed), the :math:`\xi`-history is ``0`` for
:math:`t \le 0`, so :math:`\ln\sigma_0^2 = \omega_\sigma` and
:math:`\sigma_0 = \exp(\omega_\sigma/2)`. Truncation is ``L = n - 1`` (App. C.3).

**The relaxed ridge + honest misfit.** Short-memory Log-GARCH sits on a near-common-root ridge
(:math:`\phi_1 \approx -\psi_1`, weakly identified). Here the fractional :math:`d` **absorbs the
persistence** — the fixture fits :math:`\phi_1 = 0.306, \psi_1 = -0.556` (well separated, :math:`d =
0.289` interior), the FIEGARCH pattern (:math:`\phi_1` dropped from ~0.98) — so all coefficients are
identified and recover tight (no ridge tolerance). Note FILog-GARCH fits the GARCH-simulated fixture
series ~160 log-likelihood **worse** than the rest of the family: the data is not Type-II
log-long-memory, so this is **honest model misfit**, not a defect.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import get_distribution
from quantica.timeseries.fegarch.fiegarch import theta_coefficients
from quantica.timeseries.fegarch.garch import GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "filoggarch_gamma_coefficients",
    "filoggarch_recursion",
    "filoggarch_sim",
    "fit_filoggarch",
]

_VAR_NAMES = ("mu", "omega_sig", "phi1", "psi1", "d")
_MIN_ETA_SQ = 1e-300  # floor for ln(eta^2) at a rejected optimizer point (mirror loggarch)


def filoggarch_gamma_coefficients(phi1: float, psi1: float, d: float, length: int) -> FloatArray:
    r"""Coefficients :math:`\gamma_i`, :math:`\gamma(B)=(1-\phi_1 B)^{-1}(1-B)^{-d}(1+\psi_1 B)-1`.

    Reuses FIEGARCH's :func:`theta_coefficients` (geometric :math:`\phi^{-1}` ⊛ ``fracdiff_coeffs``,
    at ``-d`` fractional integration) convolved with the Type-II MA factor :math:`(1+\psi_1 B)`;
    the trailing ``-1`` sets :math:`\gamma_0 = 0`. ``d = 0`` recovers short-memory Log-GARCH's
    :math:`\gamma_i = (\psi_1+\phi_1)\phi_1^{i-1}`.

    Parameters
    ----------
    phi1 : float
        The AR coefficient (``|phi1| < 1``).
    psi1 : float
        The MA coefficient (in the numerator :math:`1+\psi_1 B`).
    d : float
        The fractional order.
    length : int
        Number of coefficients ``gamma_0 .. gamma_{length-1}``.

    Returns
    -------
    ndarray, shape (length,)
        The coefficients, with ``gamma[0] == 0``.
    """
    base = theta_coefficients(phi1, d, length)  # (1-phi1 B)^{-1}(1-B)^{-d}, base_0 = 1
    gamma = base.copy()
    gamma[1:] += psi1 * base[:-1]  # * (1 + psi1 B)
    gamma[0] = 0.0  # gamma(B) = [...] - 1  =>  gamma_0 = c_0 - 1 = 0
    return np.asarray(gamma, dtype=np.float64)


def filoggarch_recursion(
    params: FloatArray, returns: FloatArray, *, mean_log_sq: float
) -> FloatArray:
    r"""FILog-GARCH(1,d,1) conditional variance :math:`\sigma_t^2` (a QMLE ``VarianceRecursion``).

    The truncated MA(∞) Type-II recursion :math:`\ln\sigma_t^2 = \omega_\sigma + \sum_{i\ge 1}
    \gamma_i\xi_{t-i}` with :math:`\xi_t = \ln(\eta_t^2) - \operatorname{E}[\ln\eta^2]`,
    :math:`\eta_t = (r_t-\mu)/\sigma_t`. Couples (:math:`\xi_t` depends on :math:`\sigma_t`), so it
    is a step-by-step recursion, like FIEGARCH.

    Parameters
    ----------
    params : ndarray, shape (5,)
        ``(mu, omega_sig, phi1, psi1, d)`` — ``omega_sig`` is
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]` (the MA(∞) intercept, used
        directly), ``psi1`` the MA coefficient, ``d`` the fractional order.
    returns : ndarray, shape (T,)
        The return series.
    mean_log_sq : float
        :math:`\operatorname{E}[\ln\eta^2]`, the log-square centering, from
        :meth:`~quantica.timeseries.fegarch.ConditionalDistribution.mean_log_sq` (keyword-only).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`, seeded with
        :math:`\ln\sigma_0^2 = \omega_\sigma` (zero pre-sample :math:`\xi`, no ``ln Var`` seed).
    """
    mu, omega_sig, phi1, psi1, d = (float(p) for p in params)
    y = np.asarray(returns, dtype=np.float64)
    resid = y - mu
    n = y.size
    gamma = filoggarch_gamma_coefficients(phi1, psi1, d, n)  # gamma_0 = 0, gamma_1 .. gamma_{n-1}
    log_var = np.empty(n, dtype=np.float64)
    xi = np.empty(n, dtype=np.float64)  # log-square innovation, built as the recursion advances
    for t in range(n):
        # ln sigma^2[t] = omega_sig + sum_{i=1}^{t} gamma_i xi_{t-i}; t=0 has empty history.
        log_var[t] = omega_sig + (float(gamma[1 : t + 1] @ xi[t - 1 :: -1]) if t > 0 else 0.0)
        eta = resid[t] / np.exp(log_var[t] / 2.0)
        # floor eta^2 so a rejected optimizer point gives a finite (large) xi, not a -inf warning.
        xi[t] = np.log(max(eta * eta, _MIN_ETA_SQ)) - mean_log_sq
    return np.exp(log_var)


def fit_filoggarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit FILog-GARCH(1,d,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``, the only validated one).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega_sig, phi1, psi1, d``, log-likelihood, per-observation AIC/BIC, and the
        conditional-volatility series (original return units).
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude mean
    scaled = y / scale
    distribution = get_distribution(cond_dist)

    # E[ln eta^2] from the distribution layer (norm -> -gamma_E - ln2); captured, not hard-coded.
    mean_log_sq = distribution.mean_log_sq(distribution.param_start)

    def recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
        return filoggarch_recursion(params, returns, mean_log_sq=mean_log_sq)

    var_start = (float(np.mean(scaled)), float(np.log(np.var(scaled, ddof=1))), 0.3, -0.3, 0.3)
    var_bounds = (
        (-10.0, 10.0),
        (-50.0, 50.0),
        (-0.9999, 0.9999),
        (-0.9999, 0.9999),
        (1e-6, 0.9999),  # d in (0, 1) -- fitted 0.289 is interior, no boundary handling needed
    )

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
    # shifts ADDITIVELY by ln(scale^2); phi1/psi1/d and shape parameters are scale-invariant.
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


def filoggarch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega_sig: float,
    phi1: float,
    psi1: float,
    d: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a FILog-GARCH(1,d,1) process under a chosen conditional distribution.

    Unlike the fit, simulation does **not** couple: the innovations :math:`\eta_t` are drawn, so
    :math:`\xi_t = \ln(\eta_t^2) - \operatorname{E}[\ln\eta^2]` is precomputed and the MA(∞) is one
    convolution (``O(n log n)``), like :func:`~quantica.timeseries.fegarch.fiegarch_sim`.

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega_sig : float
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]`, the MA(∞) intercept.
    phi1, psi1, d : float
        FILog-GARCH parameters; requires ``|phi1| < 1`` and ``0 <= d < 1``.
    cond_dist : str, optional
        Conditional-distribution code (default ``"norm"``). Non-norm needs ``mean_log_sq``, which is
        implemented only for the normal (deferred, as for MLog-GARCH).
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
        If ``n`` is not positive, ``|phi1| >= 1``, or ``d`` is outside ``[0, 1)``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not abs(phi1) < 1.0:
        raise ValueError("require |phi1| < 1 for a stationary FILog-GARCH(1,d,1)")
    if not 0.0 <= d < 1.0:
        raise ValueError("require 0 <= d < 1 for the fractional order")

    distribution = get_distribution(cond_dist)
    mean_log_sq = distribution.mean_log_sq(dist_params)
    total = n + n_burn
    eta = distribution.sample(total, rng, dist_params)
    # xi depends only on the drawn eta (no sigma feedback in simulation) -> one convolution.
    xi = np.log(np.maximum(eta * eta, _MIN_ETA_SQ)) - mean_log_sq
    gamma = filoggarch_gamma_coefficients(phi1, psi1, d, total)  # gamma_0 = 0
    conv = np.convolve(gamma, xi)[:total]  # (gamma * xi)[t] = sum_{i>=1} gamma_i xi_{t-i}
    log_var = omega_sig + conv  # conv[0] = 0 (gamma_0 = 0) -> ln sigma^2[0] = omega_sig
    sigma = np.exp(log_var / 2.0)
    returns = mu + sigma * eta
    return (
        np.asarray(returns[n_burn:], dtype=np.float64),
        np.asarray(sigma[n_burn:], dtype=np.float64),
    )
