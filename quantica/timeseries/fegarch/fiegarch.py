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

from quantica.timeseries.fegarch.distributions import (
    AverageLaplace,
    ConditionalDistribution,
    FernandezSteelSkew,
    get_distribution,
)
from quantica.timeseries.fegarch.egarch import (
    EGARCH_CONSTANTS,
    MEGARCH_CONSTANTS,
    MLOGGARCH_CONSTANTS,
    type1_news_impact,
)
from quantica.timeseries.fegarch.fracdiff import fracdiff_coeffs
from quantica.timeseries.fegarch.garch import _ALD_PRANGE, GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.timeseries.fegarch.qmle import QMLEResult

# Nelder-Mead options for the shape-parameter FI-EGF fits (flat shape ridge + shape-dependent
# centering + the fractional d), plus simplex-collapse restart controls (as the short-memory EGF).
_SHAPE_OPTIONS: dict[str, object] = {"maxiter": 20000, "maxfev": 20000, "fatol": 1e-10}
_MAX_FIEGF_RESTARTS = 3
_FIEGF_RESTART_TOL = 1e-6

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
    params: FloatArray,
    returns: FloatArray,
    *,
    abs_moment: float,
    mean_signed_log_modulus: float = 0.0,
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
    mean_signed_log_modulus : float, optional
        :math:`\operatorname{E}[\operatorname{sgn}(\eta)\ln(|\eta|+1)]`, the modulus-log
        **asymmetry**
        centering (``0`` for symmetric innovations, nonzero under skew). Default ``0``.

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2` (same MA(∞) pre-sample as FIEGARCH).
    """
    return _fiegarch_variance(
        params,
        returns,
        constants=MEGARCH_CONSTANTS,
        mean_asy=mean_signed_log_modulus,
        mean_mag=abs_moment,
    )


def fimloggarch_recursion(
    params: FloatArray,
    returns: FloatArray,
    *,
    mean_log_modulus: float,
    mean_signed_log_modulus: float = 0.0,
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
    mean_signed_log_modulus : float, optional
        :math:`\operatorname{E}[\operatorname{sgn}(\eta)\ln(|\eta|+1)]`, the asymmetry centering
        (``0`` for symmetric, nonzero under skew). Default ``0``.

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2` (same MA(∞) pre-sample as FIEGARCH).
    """
    return _fiegarch_variance(
        params,
        returns,
        constants=MLOGGARCH_CONSTANTS,
        mean_asy=mean_signed_log_modulus,
        mean_mag=mean_log_modulus,
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
    var_start = (float(np.mean(scaled)), float(np.log(np.var(scaled, ddof=1))), 0.5, 0.0, 0.1, 0.3)

    if cond_dist in ("ald", "sald"):
        return _fit_fiegarch_ald(
            scaled,
            scale,
            n,
            constants,
            log_modulus_magnitude,
            var_start,
            skewed=cond_dist == "sald",
        )

    distribution = get_distribution(cond_dist)
    result = _run_fiegarch_fit(scaled, distribution, constants, log_modulus_magnitude, var_start)
    return _build_fiegarch_fit(result, cond_dist, scale, n, k=result.params.size)


_FIEGARCH_BOUNDS = (
    (-10.0, 10.0),
    (-50.0, 50.0),
    (-0.9999, 0.9999),
    (-5.0, 5.0),
    (-5.0, 5.0),
    (1e-6, 0.9999),  # d in (0, 1) -- NOT clamped at 0.5 (long memory extends to the unit root)
)


def _fiegarch_centered_recursion(
    distribution: ConditionalDistribution,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
) -> tuple[object, bool, str, dict[str, object] | None]:
    """The FI-EGF recursion with the right magnitude + asymmetry centering for a distribution.

    Mirrors the short-memory EGF: a continuous shape recomputes the centering moments from the
    *current* shape each iteration (Nelder-Mead); ``norm`` / fixed-``P`` ALD use precomputed consts
    (default L-BFGS-B), bit-identical. The magnitude is ``E[ln(|eta|+1)]`` (log-modulus models) or
    ``E|eta|``; the asymmetry is the signed-log-modulus moment for the modulus-log-asymmetry models
    (``M_asy = 1, p_asy = 0``) — ``0`` for symmetric, nonzero under skew — else ``0``.
    """
    moment = distribution.mean_log_modulus if log_modulus_magnitude else distribution.abs_moment
    modulus_log_asy = constants[0] == 1.0 and constants[1] == 0.0

    def _mean_asy(dist_params: tuple[float, ...] | None) -> float:
        return distribution.mean_signed_log_modulus(dist_params) if modulus_log_asy else 0.0

    if distribution.param_names:

        def per_iter(
            params: FloatArray, returns: FloatArray, dist_params: tuple[float, ...]
        ) -> FloatArray:
            return _fiegarch_variance(
                params,
                returns,
                constants=constants,
                mean_asy=_mean_asy(dist_params),
                mean_mag=moment(dist_params),
            )

        return per_iter, True, "Nelder-Mead", _SHAPE_OPTIONS

    mean_mag = moment(distribution.param_start)
    mean_asy_const = _mean_asy(distribution.param_start)

    def constant(params: FloatArray, returns: FloatArray) -> FloatArray:
        return _fiegarch_variance(
            params, returns, constants=constants, mean_asy=mean_asy_const, mean_mag=mean_mag
        )

    return constant, False, "L-BFGS-B", None


def _run_fiegarch_fit(
    scaled: FloatArray,
    distribution: ConditionalDistribution,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
    var_start: tuple[float, ...],
) -> QMLEResult:
    """Fit an FI-EGF model under one distribution, restarting Nelder-Mead if the simplex stalls."""
    recursion, uses_dist, method, options = _fiegarch_centered_recursion(
        distribution, constants, log_modulus_magnitude
    )
    kw: dict[str, object] = {
        "var_bounds": _FIEGARCH_BOUNDS,
        "var_names": _VAR_NAMES,
        "mean": True,
        "method": method,
        "options": options,
        "recursion_uses_dist_params": uses_dist,
    }
    result = quasi_max_likelihood(
        scaled,
        recursion,  # type: ignore[arg-type]
        distribution,
        var_start=var_start,
        **kw,  # type: ignore[arg-type]
    )
    if method != "Nelder-Mead":
        return result
    n_var = len(var_start)
    for _ in range(_MAX_FIEGF_RESTARTS):
        var_restart = tuple(float(p) for p in result.params[:n_var])
        dist_restart = tuple(float(p) for p in result.params[n_var:])
        restarted = quasi_max_likelihood(
            scaled,
            recursion,  # type: ignore[arg-type]
            distribution,
            var_start=var_restart,
            dist_start=dist_restart,
            **kw,  # type: ignore[arg-type]
        )
        if restarted.loglikelihood <= result.loglikelihood + _FIEGF_RESTART_TOL:
            if restarted.loglikelihood > result.loglikelihood:
                result = restarted
            break
        result = restarted
    return result


def _build_fiegarch_fit(
    result: QMLEResult,
    cond_dist: str,
    scale: float,
    n: int,
    *,
    k: int,
    extra_params: dict[str, float] | None = None,
    profile: tuple[tuple[int, float], ...] | None = None,
) -> GarchFit:
    """Assemble a :class:`GarchFit` from an FI-EGF result: undo the scaling and form AIC/BIC.

    Parameters
    ----------
    result : QMLEResult
        The fitted engine result on the rescaled returns.
    cond_dist : str
        The conditional-distribution code.
    scale : float
        The return scale divided out before fitting.
    n : int
        Number of observations.
    k : int
        Parameter count for the AIC/BIC penalty (includes the profiled ``P`` where one applies).
    extra_params : dict of str to float, optional
        Extra parameters not produced by the optimizer (the ALD's profiled ``P``).
    profile : tuple of (int, float) or None, optional
        The ALD ``(P, log-likelihood)`` grid, forwarded to :class:`GarchFit`.

    Returns
    -------
    GarchFit
        The fitted model in original units.
    """
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
    if extra_params:
        params.update(extra_params)

    loglik = result.loglikelihood - n * np.log(scale)  # Jacobian of the rescaling
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
        profile=profile,
    )


def _fit_fiegarch_ald(
    scaled: FloatArray,
    scale: float,
    n: int,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
    var_start: tuple[float, ...],
    *,
    skewed: bool,
) -> GarchFit:
    """Profile the ALD degree ``P`` over :data:`_ALD_PRANGE` for an FI-EGF model (ald / sald).

    Each fixed-``P`` ALD has a constant magnitude centering; ``sald`` adds the FS ``skew``
    (per-iteration). ``P`` never enters the optimizer but counts in the AIC/BIC penalty.
    """
    best: QMLEResult | None = None
    best_p = 0
    profile: list[tuple[int, float]] = []
    for p in _ALD_PRANGE:
        base = AverageLaplace(p=p)
        distribution = FernandezSteelSkew(base) if skewed else base
        result = _run_fiegarch_fit(
            scaled, distribution, constants, log_modulus_magnitude, var_start
        )
        profile.append((p, float(result.loglikelihood - n * np.log(scale))))
        if best is None or result.loglikelihood > best.loglikelihood:
            best, best_p = result, p

    assert best is not None  # _ALD_PRANGE is non-empty
    return _build_fiegarch_fit(
        best,
        "sald" if skewed else "ald",
        scale,
        n,
        k=best.params.size + 1,
        extra_params={"P": float(best_p)},
        profile=tuple(profile),
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
    # FIMEGARCH/FIMLog-GARCH ((M_asy, p_asy) = (1, 0)) center g_asy on E[sgn(eta) ln(|eta|+1)]
    # (0 for symmetric, nonzero under skew); FIEGARCH keeps 0. The sim matches the fit's centering.
    modulus_log_asy = constants[0] == 1.0 and constants[1] == 0.0
    mean_asy = distribution.mean_signed_log_modulus(dist_params) if modulus_log_asy else 0.0
    total = n + n_burn
    eta = distribution.sample(total, rng, dist_params)
    news = np.array(
        [
            type1_news_impact(
                e, kappa, gamma, constants=constants, mean_asy=mean_asy, mean_mag=mean_mag
            )
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
