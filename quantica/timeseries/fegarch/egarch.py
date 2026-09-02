r"""Type-I EGARCH-family (EGF) models — EGARCH / MEGARCH / MLog-GARCH (Phase 2).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (Nelson 1991 for EGARCH; WP 2026-04 §2.1 Eqs. 7-9 + App. C.3 for the generalized EGF spec and
    QMLE conditioning; the EGF papers Ayensu et al. 2026 / Peitz et al. 2026, and John-Draper 1980
    for the modulus-log transform), **never** the `fEGarch` source. Validated against committed
    `fEGarch` *output* fixtures.

The **Type-I** EGF models share one log-variance recursion (WP 2026-04 Eq. 5), for orders ``(1, 1)``

.. math::

    r_t = \mu + \sigma_t\,\eta_t,\qquad
    \ln\sigma_t^2 = \omega + g(\eta_{t-1}) + \phi_1\,\ln\sigma_{t-1}^2,

and differ only in the **news-impact transformation** :math:`g` (WP 2026-04 Eq. 7)

.. math::

    g(\eta) = \kappa\,\{g_{\mathrm{asy}}(\eta) - \operatorname{E}[g_{\mathrm{asy}}]\}
            + \gamma\,\{g_{\mathrm{mag}}(\eta) - \operatorname{E}[g_{\mathrm{mag}}]\},

where :math:`\kappa` weights the **asymmetry** term and :math:`\gamma` the **magnitude** term (the
WP 2026-04 / WP173 orientation; the fixtures confirm :math:`\kappa < 0` leverage, :math:`\gamma > 0`
magnitude — WP175 swaps the letters). :math:`g_{\mathrm{asy}}` and :math:`g_{\mathrm{mag}}` are the
generalized forms (WP 2026-04 Eqs. 8-9), parameterized by **fixed construction constants**
:math:`(M_{\mathrm{asy}}, p_{\mathrm{asy}}, M_{\mathrm{mag}}, p_{\mathrm{mag}})` (chosen per model,
not fitted):

.. math::

    g_{\mathrm{asy}}(\eta) &= \operatorname{sgn}(\eta)\cdot
        \begin{cases}\ln(|\eta| + M) & p = 0\\ [(|\eta| + M)^p - M]/p & p > 0\end{cases}, \\
    g_{\mathrm{mag}}(\eta) &=
        \begin{cases}\ln(|\eta| + M) & p = 0\\ [(|\eta| + M)^p - M]/p & p > 0\end{cases}.

The three models are constant-sets of this one recursion:

============  ================  ================  ======================  ==================
model         :math:`(M,p)_a`   :math:`(M,p)_m`   :math:`g_{\mathrm{asy}}`  :math:`g_{\mathrm{mag}}`
============  ================  ================  ======================  ==================
EGARCH        ``(0, 1)``        ``(0, 1)``        :math:`\eta`            :math:`|\eta|`
MEGARCH       ``(1, 0)``        ``(0, 1)``        :math:`\zeta(\eta)`     :math:`|\eta|`
MLog-GARCH    ``(1, 0)``        ``(1, 0)``        :math:`\zeta(\eta)`     :math:`\ln(|\eta|+1)`
============  ================  ================  ======================  ==================

with the modulus-log transform :math:`\zeta(\eta) = \operatorname{sgn}(\eta)\ln(|\eta|+1)`
(John-Draper 1980). EGARCH is thus recovered exactly by the ``(0,1,0,1)`` constant-set.

**Centering.** :math:`\operatorname{E}[g(\eta)] = 0` (keeping :math:`\omega_\sigma =
\operatorname{E}[\ln\sigma^2]`) needs each term centered. The **asymmetry** centering
:math:`\operatorname{E}[g_{\mathrm{asy}}]` is **0 for symmetric** innovations
(:math:`g_{\mathrm{asy}}` is odd, the density even), and **0 under the FS-skew wrapper too** for
EGARCH (:math:`g_{\mathrm{asy}} = \eta`, :math:`\operatorname{E}[\eta] = 0` by standardization). The
**magnitude** centering is :math:`\operatorname{E}|\eta|` (``abs_moment``) for EGARCH/MEGARCH
(:math:`g_{\mathrm{mag}} = |\eta|`), and :math:`\operatorname{E}[\ln(|\eta|+1)]`
(``mean_log_modulus``) for MLog-GARCH — both sourced from the distribution layer, never hard-coded.
**All eight distributions are wired** (the EGF distribution infrastructure, Phase 5): the FS-skew
wrapper's moments are by quadrature over its density, and for a jointly-estimated continuous shape
the centering is re-computed from the *current* shape each optimizer iteration (per-iteration
centering, via the QMLE ``recursion_uses_dist_params`` hook, with Nelder-Mead + a restart for the
flat shape ridge). Validated on EGARCH against the five converging fixtures + known-truth for
``std`` / ``sstd`` (which fEGarch's own optimizer failed to fit).

**Reported intercept & pre-sample.** `fEGarch` reports :math:`\omega_\sigma =
\operatorname{E}[\ln\sigma_t^2]` (``omega_sig``); the recursion intercept :math:`\omega =
\omega_\sigma(1 - \phi_1)` is applied internally, and the parameter vector is
``(mu, omega_sig, phi1, kappa, gamma)`` for all three (no ``psi`` at order ``(1, 1)``). Per
App. C.3, the pre-sample news-impact history is zero and :math:`\ln\sigma_0^2 = \omega + \phi_1\ln
\operatorname{Var}(r)` (``ddof=1``) — reconstructing each fixture's :math:`\sigma`-series matches to
``~5e-17``. This generalized Type-I seam is what Phase-4's fractionally-integrated variants extend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import (
    AverageLaplace,
    ConditionalDistribution,
    FernandezSteelSkew,
    get_distribution,
)
from quantica.timeseries.fegarch.garch import _ALD_PRANGE, GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.timeseries.fegarch.qmle import QMLEResult

# Optimizer options for the derivative-free Nelder-Mead used on the shape-parameter EGF fits (the
# flat shape ridge plus the shape-dependent E[g(eta)] centering), matching the GARCH shape-fit path.
_SHAPE_OPTIONS: dict[str, object] = {"maxiter": 20000, "maxfev": 20000, "fatol": 1e-10}

__all__ = [
    "EGARCH_CONSTANTS",
    "MEGARCH_CONSTANTS",
    "MLOGGARCH_CONSTANTS",
    "egarch_recursion",
    "egarch_sim",
    "fit_egarch",
    "fit_megarch",
    "fit_mloggarch",
    "megarch_recursion",
    "megarch_sim",
    "mloggarch_recursion",
    "mloggarch_sim",
    "type1_news_impact",
]

_VAR_NAMES = ("mu", "omega_sig", "phi1", "kappa", "gamma")

# Construction constants (M_asy, p_asy, M_mag, p_mag) — fixed per model, not fitted.
EGARCH_CONSTANTS = (0.0, 1.0, 0.0, 1.0)
MEGARCH_CONSTANTS = (1.0, 0.0, 0.0, 1.0)
MLOGGARCH_CONSTANTS = (1.0, 0.0, 1.0, 0.0)


def _modulus_core(a: float, modulus: float, power: float) -> float:
    """The shared modulus/power core of WP 2026-04 Eqs. (8)-(9) on ``a = |eta|`` (non-negative).

    ``power == 1`` returns ``a`` directly (the ``[(a+M)^1 - M]/1 = a`` linear branch, kept exact so
    the EGARCH/MEGARCH ``p = 1`` instances reproduce ``|eta|`` bit-for-bit); ``power == 0`` is the
    log-modulus ``ln(a + M)``; otherwise the general power form ``[(a + M)^power - M]/power``.
    """
    if power == 1.0:
        return a
    if power == 0.0:
        return float(np.log(a + modulus))
    return float(((a + modulus) ** power - modulus) / power)


def _g_asy(eta: float, modulus: float, power: float) -> float:
    r"""Asymmetry transform :math:`g_{\mathrm{asy}}(\eta) = \operatorname{sgn}(\eta)\cdot` core."""
    return float(np.copysign(_modulus_core(abs(eta), modulus, power), eta))


def _g_mag(eta: float, modulus: float, power: float) -> float:
    r"""Magnitude transform :math:`g_{\mathrm{mag}}(\eta) = \text{core}(|\eta|)` (no sign)."""
    return _modulus_core(abs(eta), modulus, power)


def type1_news_impact(
    eta: float,
    kappa: float,
    gamma: float,
    *,
    constants: tuple[float, float, float, float],
    mean_asy: float,
    mean_mag: float,
) -> float:
    r"""The Type-I EGF news-impact transform :math:`g(\eta)` for a given constant-set.

    :math:`g(\eta) = \kappa\{g_{\mathrm{asy}}(\eta) - \operatorname{E}[g_{\mathrm{asy}}]\} +
    \gamma\{g_{\mathrm{mag}}(\eta) - \operatorname{E}[g_{\mathrm{mag}}]\}` (WP 2026-04 Eq. 7). It is
    the shared seam reused by the short-memory Type-I models and by the fractionally-integrated
    variants (FIEGARCH / FIMEGARCH / FIMLog-GARCH), which compose it with the ``(1-L)^d`` operator.

    Parameters
    ----------
    eta : float
        The standardized residual :math:`\eta_t`.
    kappa, gamma : float
        The asymmetry and magnitude coefficients.
    constants : tuple of float
        ``(M_asy, p_asy, M_mag, p_mag)`` — the model's fixed construction constants.
    mean_asy, mean_mag : float
        The centering moments :math:`\operatorname{E}[g_{\mathrm{asy}}]` (0 for symmetric) and
        :math:`\operatorname{E}[g_{\mathrm{mag}}]`, from the conditional distribution.

    Returns
    -------
    float
        The news-impact value :math:`g(\eta)`.
    """
    m_asy, p_asy, m_mag, p_mag = constants
    return kappa * (_g_asy(eta, m_asy, p_asy) - mean_asy) + gamma * (
        _g_mag(eta, m_mag, p_mag) - mean_mag
    )


def _type1_variance(
    params: FloatArray,
    returns: FloatArray,
    *,
    constants: tuple[float, float, float, float],
    mean_asy: float,
    mean_mag: float,
) -> FloatArray:
    r"""The shared Type-I EGF conditional variance :math:`\sigma_t^2` for a given constant-set.

    Parameters
    ----------
    params : ndarray, shape (5,)
        ``(mu, omega_sig, phi1, kappa, gamma)``.
    returns : ndarray, shape (T,)
        The return series.
    constants : tuple of float
        ``(M_asy, p_asy, M_mag, p_mag)`` — the model's fixed construction constants.
    mean_asy, mean_mag : float
        Centering moments :math:`\operatorname{E}[g_{\mathrm{asy}}]` (0 for symmetric) and
        :math:`\operatorname{E}[g_{\mathrm{mag}}]`, from the conditional distribution.

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances, seeded with :math:`\ln\sigma_0^2 = \omega +
        \phi_1\ln\operatorname{Var}(r)` (``ddof=1``) and a zero pre-sample news impact.
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
        g = type1_news_impact(
            eta, kappa, gamma, constants=constants, mean_asy=mean_asy, mean_mag=mean_mag
        )
        log_var[t] = omega + g + phi1 * log_var[t - 1]
    return np.exp(log_var)


def egarch_recursion(params: FloatArray, returns: FloatArray, *, abs_moment: float) -> FloatArray:
    r"""EGARCH(1,1) conditional variance (the ``(0,1,0,1)`` Type-I instance).

    Parameters
    ----------
    params : ndarray, shape (5,)
        ``(mu, omega_sig, phi1, kappa, gamma)`` — ``omega_sig`` is
        :math:`\omega_\sigma = \operatorname{E}[\ln\sigma_t^2]`.
    returns : ndarray, shape (T,)
        The return series.
    abs_moment : float
        :math:`\operatorname{E}|\eta|`, the magnitude centering, from :meth:`abs_moment` (keyword).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`.
    """
    return _type1_variance(
        params, returns, constants=EGARCH_CONSTANTS, mean_asy=0.0, mean_mag=abs_moment
    )


def megarch_recursion(
    params: FloatArray,
    returns: FloatArray,
    *,
    abs_moment: float,
    mean_signed_log_modulus: float = 0.0,
) -> FloatArray:
    r"""MEGARCH(1,1) conditional variance (the ``(1,0,0,1)`` Type-I instance).

    Modulus-log asymmetry :math:`\zeta(\eta)` with the EGARCH magnitude :math:`|\eta|`.

    Parameters
    ----------
    params : ndarray, shape (5,)
        ``(mu, omega_sig, phi1, kappa, gamma)``.
    returns : ndarray, shape (T,)
        The return series.
    abs_moment : float
        :math:`\operatorname{E}|\eta|`, the magnitude centering, from :meth:`abs_moment` (keyword).
    mean_signed_log_modulus : float, optional
        :math:`\operatorname{E}[\operatorname{sgn}(\eta)\ln(|\eta|+1)]`, the modulus-log
        **asymmetry**
        centering (``0`` for symmetric innovations, nonzero under skew). Default ``0`` (symmetric).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`.
    """
    return _type1_variance(
        params,
        returns,
        constants=MEGARCH_CONSTANTS,
        mean_asy=mean_signed_log_modulus,
        mean_mag=abs_moment,
    )


def mloggarch_recursion(
    params: FloatArray,
    returns: FloatArray,
    *,
    mean_log_modulus: float,
    mean_signed_log_modulus: float = 0.0,
) -> FloatArray:
    r"""MLog-GARCH(1,1) conditional variance (the ``(1,0,1,0)`` Type-I instance).

    Modulus-log in **both** terms: asymmetry :math:`\zeta(\eta)`, magnitude :math:`\ln(|\eta|+1)`.

    Parameters
    ----------
    params : ndarray, shape (5,)
        ``(mu, omega_sig, phi1, kappa, gamma)``.
    returns : ndarray, shape (T,)
        The return series.
    mean_log_modulus : float
        :math:`\operatorname{E}[\ln(|\eta|+1)]`, the magnitude centering, from
        :meth:`mean_log_modulus` (keyword-only).
    mean_signed_log_modulus : float, optional
        :math:`\operatorname{E}[\operatorname{sgn}(\eta)\ln(|\eta|+1)]`, the asymmetry centering
        (``0`` for symmetric, nonzero under skew). Default ``0`` (symmetric).

    Returns
    -------
    ndarray, shape (T,)
        The conditional variances :math:`\sigma_t^2`.
    """
    return _type1_variance(
        params,
        returns,
        constants=MLOGGARCH_CONSTANTS,
        mean_asy=mean_signed_log_modulus,
        mean_mag=mean_log_modulus,
    )


_TYPE1_VAR_START = (0.9, 0.0, 0.1)  # phi1, kappa, gamma starts (mu, omega_sig set from the data)
_MAX_TYPE1_RESTARTS = 3  # Nelder-Mead simplex-collapse restarts for the shape-parameter EGF fits
_TYPE1_RESTART_TOL = 1e-6  # stop restarting when the scaled log-likelihood stops improving
_TYPE1_VAR_BOUNDS = (
    (-10.0, 10.0),
    (-50.0, 50.0),
    (-0.9999, 0.9999),
    (-5.0, 5.0),
    (-5.0, 5.0),
)


def _type1_centered_recursion(
    distribution: ConditionalDistribution,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
) -> tuple[object, bool, str, dict[str, object] | None]:
    """Build the Type-I recursion with the right E[g(eta)] magnitude centering for a distribution.

    Returns ``(recursion, uses_dist_params, method, options)``. A distribution with shape parameters
    re-computes the centering moments from the *current* shape each iteration (per-iteration
    centering, Nelder-Mead); a shapeless distribution (``norm``, or a fixed-``P`` ALD) uses a
    precomputed constant (L-BFGS-B), keeping the norm path bit-identical.

    The **magnitude** centering is ``E|eta|`` (EGARCH/MEGARCH) or ``E[ln(|eta|+1)]`` (MLog-GARCH).
    The **asymmetry** centering is ``0`` for EGARCH (``g_asy = eta``, ``E[eta] = 0`` even under
    skew), but for the modulus-log-asymmetry models (MEGARCH/MLog-GARCH, ``M_asy=1, p_asy=0``) it is
    the 4th
    EGF moment ``E[sgn(eta) ln(|eta|+1)]`` (``mean_signed_log_modulus``) — ``0`` for symmetric
    innovations but nonzero under skew, so it too recomputes per iteration.
    """
    moment = distribution.mean_log_modulus if log_modulus_magnitude else distribution.abs_moment
    # g_asy = sgn(eta) ln(|eta|+1) when (M_asy, p_asy) = (1, 0); its centering is the signed
    # modulus-log moment (0 for symmetric, nonzero under skew). EGARCH (0, 1) keeps mean_asy = 0.
    modulus_log_asy = constants[0] == 1.0 and constants[1] == 0.0

    def _mean_asy(dist_params: tuple[float, ...] | None) -> float:
        return distribution.mean_signed_log_modulus(dist_params) if modulus_log_asy else 0.0

    if distribution.param_names:

        def per_iter(
            params: FloatArray, returns: FloatArray, dist_params: tuple[float, ...]
        ) -> FloatArray:
            mean_mag = moment(dist_params)
            return _type1_variance(
                params,
                returns,
                constants=constants,
                mean_asy=_mean_asy(dist_params),
                mean_mag=mean_mag,
            )

        return per_iter, True, "Nelder-Mead", _SHAPE_OPTIONS

    mean_mag = moment(distribution.param_start)  # precomputed constant (norm / fixed-P ALD)
    mean_asy_const = _mean_asy(distribution.param_start)  # 0 for symmetric bases, hence norm

    def constant(params: FloatArray, returns: FloatArray) -> FloatArray:
        return _type1_variance(
            params, returns, constants=constants, mean_asy=mean_asy_const, mean_mag=mean_mag
        )

    return constant, False, "L-BFGS-B", None


def _run_type1_fit(
    scaled: FloatArray,
    distribution: ConditionalDistribution,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
    var_start: tuple[float, ...],
) -> QMLEResult:
    """Fit a Type-I EGF model under one distribution, restarting Nelder-Mead if its simplex stalls.

    The shape-parameter fits (Nelder-Mead over a flat shape ridge with a shape-dependent centering)
    can converge prematurely on a collapsed simplex; restarting from the solution escapes that. The
    ``norm`` / fixed-``P`` ALD path is gradient-based (L-BFGS-B) and needs no restart.
    """
    recursion, uses_dist, method, options = _type1_centered_recursion(
        distribution, constants, log_modulus_magnitude
    )
    kw: dict[str, object] = {
        "var_bounds": _TYPE1_VAR_BOUNDS,
        "var_names": _VAR_NAMES,
        "mean": True,
        "method": method,
        "options": options,
        "recursion_uses_dist_params": uses_dist,
    }
    result = quasi_max_likelihood(
        scaled, recursion, distribution, var_start=var_start, **kw  # type: ignore[arg-type]
    )
    if method != "Nelder-Mead":
        return result
    n_var = len(var_start)
    for _ in range(_MAX_TYPE1_RESTARTS):
        var_restart = tuple(float(p) for p in result.params[:n_var])
        dist_restart = tuple(float(p) for p in result.params[n_var:])
        restarted = quasi_max_likelihood(
            scaled,
            recursion,
            distribution,
            var_start=var_restart,
            dist_start=dist_restart,
            **kw,  # type: ignore[arg-type]
        )
        if restarted.loglikelihood <= result.loglikelihood + _TYPE1_RESTART_TOL:
            if restarted.loglikelihood > result.loglikelihood:
                result = restarted
            break
        result = restarted
    return result


def _build_type1_fit(
    result: QMLEResult,
    cond_dist: str,
    scale: float,
    n: int,
    *,
    k: int,
    extra_params: dict[str, float] | None = None,
    profile: tuple[tuple[int, float], ...] | None = None,
) -> GarchFit:
    """Assemble a :class:`GarchFit` from a Type-I EGF result: undo the scaling and form AIC/BIC.

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
    # shifts ADDITIVELY by ln(scale^2); phi1/kappa/gamma and shape parameters are scale-invariant.
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


def _fit_type1(
    returns: FloatArray,
    cond_dist: str,
    *,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
) -> GarchFit:
    """Shared Type-I EGF QMLE fit under any conditional distribution (the EGF infrastructure).

    The magnitude term is centered by the distribution's ``E[g(eta)]`` (``E|eta|`` for
    EGARCH/MEGARCH, ``E[ln(|eta|+1)]`` for MLog-GARCH). For a continuous shape (std df, ged shape)
    that centering is re-computed each optimizer iteration from the current shape; the ALD profiles
    ``P`` over the grid;
    norm keeps the precomputed constant, bit-identical.
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude mean
    scaled = y / scale
    var_start = (float(np.mean(scaled)), float(np.log(np.var(scaled, ddof=1))), *_TYPE1_VAR_START)

    if cond_dist in ("ald", "sald"):
        return _fit_type1_ald(
            scaled,
            scale,
            n,
            constants,
            log_modulus_magnitude,
            var_start,
            skewed=cond_dist == "sald",
        )

    distribution = get_distribution(cond_dist)
    result = _run_type1_fit(scaled, distribution, constants, log_modulus_magnitude, var_start)
    return _build_type1_fit(result, cond_dist, scale, n, k=result.params.size)


def _fit_type1_ald(
    scaled: FloatArray,
    scale: float,
    n: int,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
    var_start: tuple[float, ...],
    *,
    skewed: bool,
) -> GarchFit:
    """Profile the ALD degree ``P`` over :data:`_ALD_PRANGE` for a Type-I EGF model (ald / sald).

    Each fixed-``P`` ALD has a constant magnitude centering (no continuous shape); ``sald`` adds the
    FS ``skew`` (per-iteration centering, Nelder-Mead). ``P`` never enters the optimizer but counts
    in the AIC/BIC penalty.
    """
    best: QMLEResult | None = None
    best_p = 0
    profile: list[tuple[int, float]] = []
    for p in _ALD_PRANGE:
        base = AverageLaplace(p=p)
        distribution = FernandezSteelSkew(base) if skewed else base
        result = _run_type1_fit(scaled, distribution, constants, log_modulus_magnitude, var_start)
        profile.append((p, float(result.loglikelihood - n * np.log(scale))))
        if best is None or result.loglikelihood > best.loglikelihood:
            best, best_p = result, p

    assert best is not None  # _ALD_PRANGE is non-empty
    return _build_type1_fit(
        best,
        "sald" if skewed else "ald",
        scale,
        n,
        k=best.params.size + 1,  # P is profiled, not optimized, but counts in the penalty
        extra_params={"P": float(best_p)},
        profile=tuple(profile),
    )


def fit_egarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit EGARCH(1,1) with a constant mean by QMLE under a chosen conditional distribution.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``, the only validated one).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega_sig, phi1, kappa, gamma``, log-likelihood, per-observation AIC/BIC,
        and the conditional-volatility series (original return units).
    """
    return _fit_type1(returns, cond_dist, constants=EGARCH_CONSTANTS, log_modulus_magnitude=False)


def fit_megarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit MEGARCH(1,1) (modulus-log asymmetry, EGARCH magnitude) by QMLE.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``, the only validated one).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega_sig, phi1, kappa, gamma`` (+ log-likelihood, AIC/BIC, conditional SD).
    """
    return _fit_type1(returns, cond_dist, constants=MEGARCH_CONSTANTS, log_modulus_magnitude=False)


def fit_mloggarch(returns: FloatArray, *, cond_dist: str = "norm") -> GarchFit:
    """Fit MLog-GARCH(1,1) (modulus-log asymmetry and magnitude) by QMLE.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    cond_dist : str, optional
        One of the eight fEGarch distribution codes (default ``"norm"``, the only validated one).

    Returns
    -------
    GarchFit
        Estimates ``mu, omega_sig, phi1, kappa, gamma`` (+ log-likelihood, AIC/BIC, conditional SD).
    """
    return _fit_type1(returns, cond_dist, constants=MLOGGARCH_CONSTANTS, log_modulus_magnitude=True)


def _sim_type1(
    n: int,
    *,
    mu: float,
    omega_sig: float,
    phi1: float,
    kappa: float,
    gamma: float,
    constants: tuple[float, float, float, float],
    log_modulus_magnitude: bool,
    cond_dist: str,
    dist_params: tuple[float, ...],
    rng: np.random.Generator,
    n_burn: int,
    model: str,
) -> tuple[FloatArray, FloatArray]:
    """Shared Type-I EGF simulator; returns ``(returns, sigma)`` after burn-in."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not abs(phi1) < 1.0:
        raise ValueError(f"require |phi1| < 1 for a stationary {model}(1,1)")

    distribution = get_distribution(cond_dist)
    # The modulus-log-asymmetry models (MEGARCH/MLog-GARCH, (M_asy, p_asy) = (1, 0)) center g_asy on
    # E[sgn(eta) ln(|eta|+1)] (0 for symmetric, nonzero under skew); EGARCH keeps mean_asy = 0. The
    # sim must use the SAME centering as the fit so known-truth recovers omega_sig under skew.
    modulus_log_asy = constants[0] == 1.0 and constants[1] == 0.0
    mean_asy = distribution.mean_signed_log_modulus(dist_params) if modulus_log_asy else 0.0
    if log_modulus_magnitude:
        mean_mag = distribution.mean_log_modulus(dist_params)
    else:
        mean_mag = distribution.abs_moment(dist_params)
    omega = omega_sig * (1.0 - phi1)
    total = n + n_burn
    eta = distribution.sample(total, rng, dist_params)
    log_var = np.empty(total, dtype=np.float64)
    log_var[0] = omega_sig  # start at the unconditional mean E[ln sigma^2]
    for t in range(1, total):
        g = type1_news_impact(
            eta[t - 1], kappa, gamma, constants=constants, mean_asy=mean_asy, mean_mag=mean_mag
        )
        log_var[t] = omega + g + phi1 * log_var[t - 1]
    sigma = np.exp(log_var / 2.0)
    returns = mu + sigma * eta
    return (
        np.asarray(returns[n_burn:], dtype=np.float64),
        np.asarray(sigma[n_burn:], dtype=np.float64),
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
    return _sim_type1(
        n,
        mu=mu,
        omega_sig=omega_sig,
        phi1=phi1,
        kappa=kappa,
        gamma=gamma,
        constants=EGARCH_CONSTANTS,
        log_modulus_magnitude=False,
        cond_dist=cond_dist,
        dist_params=dist_params,
        rng=rng,
        n_burn=n_burn,
        model="EGARCH",
    )


def megarch_sim(
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
    r"""Simulate a MEGARCH(1,1) process (modulus-log asymmetry, EGARCH magnitude).

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega_sig, phi1, kappa, gamma : float
        MEGARCH parameters; requires ``|phi1| < 1`` for stationarity.
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
        If ``n`` is not positive or ``|phi1| >= 1``.
    """
    return _sim_type1(
        n,
        mu=mu,
        omega_sig=omega_sig,
        phi1=phi1,
        kappa=kappa,
        gamma=gamma,
        constants=MEGARCH_CONSTANTS,
        log_modulus_magnitude=False,
        cond_dist=cond_dist,
        dist_params=dist_params,
        rng=rng,
        n_burn=n_burn,
        model="MEGARCH",
    )


def mloggarch_sim(
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
    r"""Simulate an MLog-GARCH(1,1) process (modulus-log asymmetry and magnitude).

    Parameters
    ----------
    n : int
        Number of observations to return (after burn-in).
    mu : float, optional
        Constant mean (default 0).
    omega_sig, phi1, kappa, gamma : float
        MLog-GARCH parameters; requires ``|phi1| < 1`` for stationarity.
    cond_dist : str, optional
        Conditional-distribution code (default ``"norm"``). Non-norm needs ``mean_log_modulus``
        (deferred), so only ``norm`` is available.
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
        If ``n`` is not positive or ``|phi1| >= 1``.
    """
    return _sim_type1(
        n,
        mu=mu,
        omega_sig=omega_sig,
        phi1=phi1,
        kappa=kappa,
        gamma=gamma,
        constants=MLOGGARCH_CONSTANTS,
        log_modulus_magnitude=True,
        cond_dist=cond_dist,
        dist_params=dist_params,
        rng=rng,
        n_burn=n_burn,
        model="MLog-GARCH",
    )
