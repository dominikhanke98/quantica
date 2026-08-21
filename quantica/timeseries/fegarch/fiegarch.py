r"""Fractionally-integrated (long-memory) Type-I EGF family: FIEGARCH / FIMEGARCH / FIMLog-GARCH.

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Nelson 1991 for EGARCH; Bollerslev-Mikkelsen 1996 for the fractional extension; WP 2026-04 §2.1
    Eqs. 4-6 + App. C.3 Eq. 50 for the long-memory recursion, truncation and QMLE conditioning),
    **never** the `fEGarch` source. Validated against committed `fEGarch` *output* fixtures.

FIEGARCH is the first long-memory model, and it is built **by composition** — it reuses the Phase-2
Type-I news-impact :func:`~quantica.timeseries.fegarch.type1_news_impact` (at the EGARCH
:data:`~quantica.timeseries.fegarch.EGARCH_CONSTANTS` constant-set) and the Phase-3
:func:`~quantica.timeseries.fegarch.fracdiff_coeffs` operator, adding only the ``theta(B)``
composition and the truncated-MA(∞) recursion. **FIMEGARCH and FIMLog-GARCH are its exact
long-memory analogues** — the *same* ``theta(B)`` machinery and MA(∞) pre-sample, differing only in
the Type-I
constant-set (and hence the magnitude-centering moment), precisely as MEGARCH / MLog-GARCH differed
from EGARCH in Phase 2: FIMEGARCH uses :data:`~quantica.timeseries.fegarch.MEGARCH_CONSTANTS`
(modulus-log asymmetry, :math:`|\eta|` magnitude ⇒ ``abs_moment`` centering) and FIMLog-GARCH uses
:data:`~quantica.timeseries.fegarch.MLOGGARCH_CONSTANTS` (modulus-log in both terms ⇒
``mean_log_modulus`` centering). It splices the fractional operator into EGARCH's coefficient
polynomial (WP 2026-04 Eq. 6):

.. math::

    \theta(B) = \phi^{-1}(B)\,(1 - B)^{-d}\,\psi(B) = \sum_i \theta_i B^i,\qquad \theta_0 = 1,

which for orders ``(1, d, 1)`` (``p = q = 1`` ⇒ :math:`\psi(B) = 1`) is
:math:`\theta(B) = (1 - \phi_1 B)^{-1}(1 - B)^{-d}`. The coefficients are the causal convolution
of the geometric :math:`\phi^{-1}(B) = \sum_k \phi_1^k B^k` series with the
**fractional-integration** coefficients :math:`b_i(-d)` of :math:`(1 - B)^{-d}` (Phase-3 engine),
FFT-accelerated (the Nielsen-Noël 2021 product App. C.3 cites):
:math:`\theta_i = \sum_{k=0}^{i} \phi_1^k\, b_{i-k}(-d)`.

The conditional variance is then the **truncated MA(∞)** form (WP 2026-04 App. C.3 Eq. 50)

.. math::

    \ln\sigma_t^2 = \omega_\sigma + \sum_{i=0}^{L-1} \theta_i\, g(\eta_{t-1-i}),\qquad
    g(\eta) = \kappa\,\eta + \gamma\,(|\eta| - \operatorname{E}|\eta|),

the EGARCH ``(0,1,0,1)`` transform. The parameter vector is
``(mu, omega_sig, phi1, kappa, gamma, d)``.

**Pre-sample — the one real divergence from EGARCH (fixture-pinned).** The MA(∞) form has **no**
:math:`\ln\sigma_{t-i}^2` feedback and **no** ``ln Var(r)`` seed: the intercept is
:math:`\omega_\sigma` **directly** (not the AR-form :math:`\omega_\sigma(1-\phi_1)`), and the
:math:`g`-history is ``0`` for :math:`t \le 0`, so :math:`\ln\sigma_0^2 = \omega_\sigma` and
:math:`\sigma_0 = \exp(\omega_\sigma/2)`. Reconstructing the committed fixture's
:math:`\sigma`-series under this convention matches to ``~1e-16`` (the AR-form intercept gives
relative ``4.9`` — decisively wrong).

**The fractional order.** :math:`d \in (0, 1)`: ``d = 0`` collapses to short-memory EGARCH,
:math:`d \in (0, 0.5)` is weakly stationary long memory, and :math:`d \in (0.5, 1)` is
mean-reverting non-stationary long memory (fEGarch's fitted ``d = 0.744`` on the synthetic series is
in this upper regime — the high-persistence GARCH data is captured by a large :math:`d` and a small
:math:`\phi_1 = 0.39`, the persistence reparameterized into the slow :math:`\theta` tail). The QMLE
bound is therefore :math:`d \in (0, 1)`, **not** clamped at ``0.5``. Truncation is ``L = n - 1``
(App. C.3), so the whole (slow-decaying) :math:`\theta` tail is used — load-bearing at large ``d``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import fft as sp_fft

from quantica.timeseries.fegarch.distributions import get_distribution
from quantica.timeseries.fegarch.egarch import (
    EGARCH_CONSTANTS,
    MEGARCH_CONSTANTS,
    MLOGGARCH_CONSTANTS,
    type1_news_impact,
)
from quantica.timeseries.fegarch.fracdiff import fracdiff_coeffs
from quantica.timeseries.fegarch.garch import GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "fiegarch_recursion",
    "fiegarch_sim",
    "fimegarch_recursion",
    "fimegarch_sim",
    "fimloggarch_recursion",
    "fimloggarch_sim",
    "fit_fiegarch",
    "fit_fimegarch",
    "fit_fimloggarch",
    "theta_coefficients",
]

_VAR_NAMES = ("mu", "omega_sig", "phi1", "kappa", "gamma", "d")


def theta_coefficients(phi1: float, d: float, length: int) -> FloatArray:
    r"""Coefficients :math:`\theta_i` of :math:`\theta(B) = (1 - \phi_1 B)^{-1}(1 - B)^{-d}`.

    The causal convolution of the geometric :math:`\phi^{-1}(B) = \sum_k \phi_1^k B^k` series with
    the fractional-integration coefficients :math:`b_i(-d)` (Phase-3 :func:`fracdiff_coeffs` at
    negative exponent), FFT-accelerated. ``d = 0`` recovers the EGARCH :math:`\theta_i = \phi_1^i`.

    Parameters
    ----------
    phi1 : float
        The AR coefficient (``|phi1| < 1``).
    d : float
        The fractional order.
    length : int
        Number of coefficients ``theta_0 .. theta_{length-1}``.

    Returns
    -------
    ndarray, shape (length,)
        The coefficients, with ``theta[0] == 1``.
    """
    frac = fracdiff_coeffs(-d, length)  # (1-B)^{-d} coefficients (Phase-3, negative exponent)
    geometric = phi1 ** np.arange(length, dtype=np.float64)  # phi^{-1}(B) = sum_k phi1^k B^k
    fft_len = sp_fft.next_fast_len(2 * length - 1)
    product = np.fft.rfft(geometric, fft_len) * np.fft.rfft(frac, fft_len)
    return np.asarray(np.fft.irfft(product, fft_len)[:length], dtype=np.float64)


def _fiegarch_variance(
    params: FloatArray,
    returns: FloatArray,
    *,
    constants: tuple[float, float, float, float],
    mean_asy: float,
    mean_mag: float,
) -> FloatArray:
    r"""Shared fractionally-integrated Type-I conditional variance :math:`\sigma_t^2`.

    The truncated-MA(∞) recursion :math:`\ln\sigma_t^2 = \omega_\sigma + \sum_{i} \theta_i\,
    g(\eta_{t-1-i})` for a given ``constants`` set — the FI analogue of
    :func:`~quantica.timeseries.fegarch.egarch._type1_variance`. FIEGARCH / FIMEGARCH / FIMLog-GARCH
    differ only in the constant-set (and hence the magnitude-centering moment ``mean_mag``).

    Parameters
    ----------
    params : ndarray, shape (6,)
        ``(mu, omega_sig, phi1, kappa, gamma, d)``.
    returns : ndarray, shape (T,)
        The return series.
    constants : tuple of float
        The Type-I ``(M_asy, p_asy, M_mag, p_mag)`` construction constants (WP 2026-04 Eqs. 7-9).
    mean_asy, mean_mag : float
        The asymmetry and magnitude centering moments (``mean_asy = 0`` for symmetric bases).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances, seeded with :math:`\ln\sigma_0^2 = \omega_\sigma` (zero
        pre-sample news impact, no ``ln Var`` seed) and truncated at :math:`L = n - 1`.
    """
    mu, omega_sig, phi1, kappa, gamma, d = (float(p) for p in params)
    y = np.asarray(returns, dtype=np.float64)
    resid = y - mu
    n = y.size
    theta = theta_coefficients(phi1, d, n)  # L = n-1 -> n coefficients theta_0..theta_{n-1}
    log_var = np.empty(n, dtype=np.float64)
    news = np.empty(n, dtype=np.float64)  # g(eta_t), built as the recursion advances
    for t in range(n):
        # ln sigma^2[t] = omega_sig + sum_{i=0}^{t-1} theta_i g(eta_{t-1-i}); t=0 has empty history.
        log_var[t] = omega_sig + (float(theta[:t] @ news[t - 1 :: -1]) if t > 0 else 0.0)
        eta = resid[t] / np.exp(log_var[t] / 2.0)
        news[t] = type1_news_impact(
            eta, kappa, gamma, constants=constants, mean_asy=mean_asy, mean_mag=mean_mag
        )
    return np.exp(log_var)


def fiegarch_recursion(params: FloatArray, returns: FloatArray, *, abs_moment: float) -> FloatArray:
    r"""FIEGARCH(1,d,1) conditional variance :math:`\sigma_t^2` (a QMLE ``VarianceRecursion``).

    Parameters
    ----------
    params : ndarray, shape (6,)
        ``(mu, omega_sig, phi1, kappa, gamma, d)`` — ``omega_sig`` is
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]` (used **directly** as the MA(∞)
        intercept), and ``d`` is the fractional order.
    returns : ndarray, shape (T,)
        The return series.
    abs_moment : float
        :math:`\operatorname{E}|\eta|`, the magnitude centering, from :meth:`abs_moment` (keyword).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`, seeded with
        :math:`\ln\sigma_0^2 = \omega_\sigma` (zero pre-sample news impact, no ``ln Var`` seed) and
        truncated at :math:`L = n - 1`.
    """
    return _fiegarch_variance(
        params, returns, constants=EGARCH_CONSTANTS, mean_asy=0.0, mean_mag=abs_moment
    )


def fimegarch_recursion(
    params: FloatArray, returns: FloatArray, *, abs_moment: float
) -> FloatArray:
    r"""FIMEGARCH(1,d,1) conditional variance — FIEGARCH with the MEGARCH constant-set.

    The modulus-log asymmetry :math:`\operatorname{sgn}(\eta)\ln(|\eta|+1)` with the EGARCH
    :math:`|\eta|` magnitude (:data:`~quantica.timeseries.fegarch.MEGARCH_CONSTANTS`), spliced into
    the same long-memory :math:`\theta(B)` recursion.

    Parameters
    ----------
    params : ndarray, shape (6,)
        ``(mu, omega_sig, phi1, kappa, gamma, d)``.
    returns : ndarray, shape (T,)
        The return series.
    abs_moment : float
        :math:`\operatorname{E}|\eta|`, the magnitude centering, from :meth:`abs_moment` (keyword).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2` (same MA(∞) pre-sample as FIEGARCH).
    """
    return _fiegarch_variance(
        params, returns, constants=MEGARCH_CONSTANTS, mean_asy=0.0, mean_mag=abs_moment
    )


def fimloggarch_recursion(
    params: FloatArray, returns: FloatArray, *, mean_log_modulus: float
) -> FloatArray:
    r"""FIMLog-GARCH(1,d,1) conditional variance — FIEGARCH with the MLog-GARCH constant-set.

    The modulus-log transform :math:`\operatorname{sgn}(\eta)\ln(|\eta|+1)` in **both** the
    asymmetry and the magnitude (:data:`~quantica.timeseries.fegarch.MLOGGARCH_CONSTANTS`), spliced
    into the same long-memory :math:`\theta(B)` recursion.

    Parameters
    ----------
    params : ndarray, shape (6,)
        ``(mu, omega_sig, phi1, kappa, gamma, d)``.
    returns : ndarray, shape (T,)
        The return series.
    mean_log_modulus : float
        :math:`\operatorname{E}[\ln(|\eta|+1)]`, the magnitude centering, from
        :meth:`mean_log_modulus` (keyword-only).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2` (same MA(∞) pre-sample as FIEGARCH).
    """
    return _fiegarch_variance(
        params, returns, constants=MLOGGARCH_CONSTANTS, mean_asy=0.0, mean_mag=mean_log_modulus
    )


def _fit_fiegarch(
    returns: FloatArray,
    cond_dist: str,
    *,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
) -> GarchFit:
    """Shared fractionally-integrated Type-I QMLE fit for a constant-set (norm-only validated).

    The FI analogue of :func:`~quantica.timeseries.fegarch.egarch._fit_type1`, with the extra
    fractional order ``d`` (bound ``(1e-6, 0.9999)`` — long memory extends to the unit root, NOT
    clamped at ``0.5``). Scale-equivariant: ``mu`` scales, ``omega_sig`` shifts additively by
    ``ln(scale^2)``, ``phi1/kappa/gamma/d`` are invariant.
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude mean
    scaled = y / scale
    distribution = get_distribution(cond_dist)

    # Magnitude centering from the distribution layer; captured, not hard-coded.
    if log_modulus_magnitude:
        mean_mag = distribution.mean_log_modulus(distribution.param_start)
    else:
        mean_mag = distribution.abs_moment(distribution.param_start)

    def recursion(params: FloatArray, returns: FloatArray) -> FloatArray:
        return _fiegarch_variance(
            params, returns, constants=constants, mean_asy=0.0, mean_mag=mean_mag
        )

    var_start = (float(np.mean(scaled)), float(np.log(np.var(scaled, ddof=1))), 0.5, 0.0, 0.1, 0.3)
    var_bounds = (
        (-10.0, 10.0),
        (-50.0, 50.0),
        (-0.9999, 0.9999),
        (-5.0, 5.0),
        (-5.0, 5.0),
        (1e-6, 0.9999),  # d in (0, 1) -- NOT clamped at 0.5 (long memory extends to the unit root)
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
    # shifts ADDITIVELY by ln(scale^2); phi1/kappa/gamma/d and shape parameters are scale-invariant.
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


def fit_fiegarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit FIEGARCH(1,d,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``, the only validated one).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega_sig, phi1, kappa, gamma, d``, log-likelihood, per-observation AIC/BIC,
        and the conditional-volatility series (original return units).
    """
    return _fit_fiegarch(
        returns, cond_dist, constants=EGARCH_CONSTANTS, log_modulus_magnitude=False
    )


def fit_fimegarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit FIMEGARCH(1,d,1) (modulus-log asymmetry, EGARCH magnitude) by long-memory QMLE.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``, the only validated one).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega_sig, phi1, kappa, gamma, d`` (+ log-likelihood, AIC/BIC, conditional
        SD).
    """
    return _fit_fiegarch(
        returns, cond_dist, constants=MEGARCH_CONSTANTS, log_modulus_magnitude=False
    )


def fit_fimloggarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit FIMLog-GARCH(1,d,1) (modulus-log asymmetry and magnitude) by long-memory QMLE.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        Conditional-distribution code (default ``"norm"``, the only validated one). Non-norm needs
        ``mean_log_modulus``, which is currently implemented only for the normal (deferred).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega_sig, phi1, kappa, gamma, d`` (+ log-likelihood, AIC/BIC, conditional
        SD).
    """
    return _fit_fiegarch(
        returns, cond_dist, constants=MLOGGARCH_CONSTANTS, log_modulus_magnitude=True
    )


def _sim_fiegarch(
    n: int,
    *,
    mu: float,
    omega_sig: float,
    phi1: float,
    kappa: float,
    gamma: float,
    d: float,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
    cond_dist: str,
    dist_params: tuple[float, ...],
    rng: np.random.Generator,
    n_burn: int,
    model: str,
) -> tuple[FloatArray, FloatArray]:
    """Shared fractionally-integrated Type-I simulator; returns ``(returns, sigma)`` after burn-in.

    The MA(∞) is one convolution (no feedback): the innovations are given, so ``g(eta_t)`` is
    precomputed, then convolved with the :func:`theta_coefficients`. The FI analogue of
    :func:`~quantica.timeseries.fegarch.egarch._sim_type1`.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not abs(phi1) < 1.0:
        raise ValueError(f"require |phi1| < 1 for a stationary {model}(1,d,1)")
    if not 0.0 <= d < 1.0:
        raise ValueError("require 0 <= d < 1 for the fractional order")

    distribution = get_distribution(cond_dist)
    if log_modulus_magnitude:
        mean_mag = distribution.mean_log_modulus(dist_params)
    else:
        mean_mag = distribution.abs_moment(dist_params)
    total = n + n_burn
    eta = distribution.sample(total, rng, dist_params)
    news = np.array(
        [
            type1_news_impact(e, kappa, gamma, constants=constants, mean_asy=0.0, mean_mag=mean_mag)
            for e in eta
        ],
        dtype=np.float64,
    )
    theta = theta_coefficients(phi1, d, total)
    log_var = np.empty(total, dtype=np.float64)
    log_var[0] = omega_sig
    if total > 1:
        conv = np.convolve(theta, news)[: total - 1]  # (theta * g)[0 .. total-2]
        log_var[1:] = omega_sig + conv
    sigma = np.exp(log_var / 2.0)
    returns = mu + sigma * eta
    return (
        np.asarray(returns[n_burn:], dtype=np.float64),
        np.asarray(sigma[n_burn:], dtype=np.float64),
    )


def fiegarch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega_sig: float,
    phi1: float,
    kappa: float,
    gamma: float,
    d: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a FIEGARCH(1,d,1) process under a chosen conditional distribution.

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega_sig : float
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]`, the MA(∞) intercept.
    phi1, kappa, gamma, d : float
        FIEGARCH parameters; requires ``|phi1| < 1`` and ``0 <= d < 1`` for a well-defined process.
    cond_dist : str, optional
        Conditional-distribution code (default ``"norm"``).
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
    return _sim_fiegarch(
        n,
        mu=mu,
        omega_sig=omega_sig,
        phi1=phi1,
        kappa=kappa,
        gamma=gamma,
        d=d,
        constants=EGARCH_CONSTANTS,
        log_modulus_magnitude=False,
        cond_dist=cond_dist,
        dist_params=dist_params,
        rng=rng,
        n_burn=n_burn,
        model="FIEGARCH",
    )


def fimegarch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega_sig: float,
    phi1: float,
    kappa: float,
    gamma: float,
    d: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a FIMEGARCH(1,d,1) process (modulus-log asymmetry, EGARCH magnitude).

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega_sig : float
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]`, the MA(∞) intercept.
    phi1, kappa, gamma, d : float
        FIMEGARCH parameters; requires ``|phi1| < 1`` and ``0 <= d < 1``.
    cond_dist : str, optional
        Conditional-distribution code (default ``"norm"``).
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
    return _sim_fiegarch(
        n,
        mu=mu,
        omega_sig=omega_sig,
        phi1=phi1,
        kappa=kappa,
        gamma=gamma,
        d=d,
        constants=MEGARCH_CONSTANTS,
        log_modulus_magnitude=False,
        cond_dist=cond_dist,
        dist_params=dist_params,
        rng=rng,
        n_burn=n_burn,
        model="FIMEGARCH",
    )


def fimloggarch_sim(
    n: int,
    *,
    mu: float = 0.0,
    omega_sig: float,
    phi1: float,
    kappa: float,
    gamma: float,
    d: float,
    cond_dist: str = "norm",
    dist_params: tuple[float, ...] = (),
    rng: np.random.Generator,
    n_burn: int = 500,
) -> tuple[FloatArray, FloatArray]:
    r"""Simulate a FIMLog-GARCH(1,d,1) process (modulus-log asymmetry and magnitude).

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega_sig : float
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]`, the MA(∞) intercept.
    phi1, kappa, gamma, d : float
        FIMLog-GARCH parameters; requires ``|phi1| < 1`` and ``0 <= d < 1``.
    cond_dist : str, optional
        Conditional-distribution code (default ``"norm"``). Non-norm needs ``mean_log_modulus``,
        currently implemented only for the normal (deferred).
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
    return _sim_fiegarch(
        n,
        mu=mu,
        omega_sig=omega_sig,
        phi1=phi1,
        kappa=kappa,
        gamma=gamma,
        d=d,
        constants=MLOGGARCH_CONSTANTS,
        log_modulus_magnitude=True,
        cond_dist=cond_dist,
        dist_params=dist_params,
        rng=rng,
        n_burn=n_burn,
        model="FIMLog-GARCH",
    )
