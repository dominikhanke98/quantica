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

**The relaxed ridge.** Short-memory Log-GARCH sits on a near-common-root ridge
(:math:`\phi_1 \approx -\psi_1`, weakly identified). Here the fractional :math:`d` **absorbs the
persistence** — the fixture fits :math:`\phi_1 = 0.306, \psi_1 = -0.556` (well separated, :math:`d =
0.289` interior), the FIEGARCH pattern (:math:`\phi_1` dropped from ~0.98) — so all coefficients are
identified and recover tight (no ridge tolerance).

**Effective challenge — fEGarch's fixture is a strictly-dominated local optimum.** The committed
fixture reports :math:`\mu = -0.003, \text{loglik} = 7436`, but this is a strictly-dominated local
optimum: every data-driven optimizer start (the data mean, random starts, and even a start from
fEGarch's own reported params) escapes to a basin ``~+120`` log-likelihood higher
(:math:`\text{loglik} \approx 7556`, :math:`\mu` near the data mean) — only an optimizer *placed* at
:math:`\mu = -0.003` stays there. Since :func:`filoggarch_recursion` at those same params reproduces
the fixture :math:`\sigma`-series to ``~1e-15``, it is fEGarch's **optimizer** landing in a poor
local optimum, not the model. :func:`fit_filoggarch` (data-mean started) reaches the dominant basin;
the tests assert strict domination from every sensible start rather than matching the fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import (
    AverageLaplace,
    FernandezSteelSkew,
    get_distribution,
)
from quantica.timeseries.fegarch.fiegarch import theta_coefficients
from quantica.timeseries.fegarch.garch import _ALD_PRANGE, GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

# Nelder-Mead options + restart controls for the shape-parameter fits (near-common-root ridge +
# shape-dependent centering + fractional d) — norm keeps the gradient path, bit-identical.
_SHAPE_OPTIONS: dict[str, object] = {"maxiter": 20000, "maxfev": 20000, "fatol": 1e-10}
_FILOGGARCH_BOUNDS = (
    (-10.0, 10.0),
    (-50.0, 50.0),
    (-0.9999, 0.9999),
    (-0.9999, 0.9999),
    (1e-6, 0.9999),
)
_MAX_FILOGGARCH_RESTARTS = 3
_FILOGGARCH_RESTART_TOL = 1e-6

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.timeseries.fegarch.qmle import QMLEResult

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
    var_start = (float(np.mean(scaled)), float(np.log(np.var(scaled, ddof=1))), 0.3, -0.3, 0.3)

    if cond_dist in ("ald", "sald"):
        return _fit_filoggarch_ald(scaled, scale, n, var_start, skewed=cond_dist == "sald")

    distribution = get_distribution(cond_dist)
    result = _run_filoggarch_fit(scaled, distribution, var_start)
    return _build_filoggarch_fit(result, cond_dist, scale, n, k=result.params.size)


def _run_filoggarch_fit(
    scaled: FloatArray, distribution: object, var_start: tuple[float, ...]
) -> QMLEResult:
    """Fit FILog-GARCH under one distribution; per-iteration ``mean_log_sq`` centering for a shape.

    ``norm`` / fixed-``P`` ALD capture a precomputed ``mean_log_sq`` and use the gradient path
    (bit-identical). A continuous shape recomputes it from the *current* shape each iteration, using
    Nelder-Mead + a restart (the near-common-root ridge + a shape-dependent centering can stall).
    """
    if distribution.param_names:  # type: ignore[attr-defined]

        def per_iter(
            params: FloatArray, returns: FloatArray, dist_params: tuple[float, ...]
        ) -> FloatArray:
            return filoggarch_recursion(
                params,
                returns,
                mean_log_sq=distribution.mean_log_sq(dist_params),  # type: ignore[attr-defined]
            )

        kw: dict[str, object] = {
            "var_bounds": _FILOGGARCH_BOUNDS,
            "var_names": _VAR_NAMES,
            "mean": True,
            "method": "Nelder-Mead",
            "options": _SHAPE_OPTIONS,
            "recursion_uses_dist_params": True,
        }
        result = quasi_max_likelihood(
            scaled,
            per_iter,  # type: ignore[arg-type]
            distribution,  # type: ignore[arg-type]
            var_start=var_start,
            **kw,  # type: ignore[arg-type]
        )
        n_var = len(var_start)
        for _ in range(_MAX_FILOGGARCH_RESTARTS):
            v = tuple(float(p) for p in result.params[:n_var])
            ds = tuple(float(p) for p in result.params[n_var:])
            restarted = quasi_max_likelihood(
                scaled,
                per_iter,  # type: ignore[arg-type]
                distribution,  # type: ignore[arg-type]
                var_start=v,
                dist_start=ds,
                **kw,  # type: ignore[arg-type]
            )
            if restarted.loglikelihood <= result.loglikelihood + _FILOGGARCH_RESTART_TOL:
                if restarted.loglikelihood > result.loglikelihood:
                    result = restarted
                break
            result = restarted
        return result

    mean_log_sq = distribution.mean_log_sq(distribution.param_start)  # type: ignore[attr-defined]

    def constant(params: FloatArray, returns: FloatArray) -> FloatArray:
        return filoggarch_recursion(params, returns, mean_log_sq=mean_log_sq)

    return quasi_max_likelihood(
        scaled,
        constant,
        distribution,  # type: ignore[arg-type]
        var_start=var_start,
        var_bounds=_FILOGGARCH_BOUNDS,
        var_names=_VAR_NAMES,
        mean=True,
    )


def _build_filoggarch_fit(
    result: QMLEResult,
    cond_dist: str,
    scale: float,
    n: int,
    *,
    k: int,
    extra_params: dict[str, float] | None = None,
    profile: tuple[tuple[int, float], ...] | None = None,
) -> GarchFit:
    """Assemble a :class:`GarchFit` from a FILog-GARCH result: undo the scaling and form AIC/BIC.

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
    # shifts ADDITIVELY by ln(scale^2); phi1/psi1/d and shape parameters are scale-invariant.
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


def _fit_filoggarch_ald(
    scaled: FloatArray, scale: float, n: int, var_start: tuple[float, ...], *, skewed: bool
) -> GarchFit:
    """Profile the ALD degree ``P`` over :data:`_ALD_PRANGE` for FILog-GARCH (ald / sald).

    Selects the ``P`` with the highest log-likelihood -- for FILog-GARCH x sald the **interior
    P = 3**, not the boundary. ``P`` never enters the optimizer but counts in the AIC/BIC penalty.
    """
    best: QMLEResult | None = None
    best_p = 0
    profile: list[tuple[int, float]] = []
    for p in _ALD_PRANGE:
        base = AverageLaplace(p=p)
        distribution = FernandezSteelSkew(base) if skewed else base
        result = _run_filoggarch_fit(scaled, distribution, var_start)
        profile.append((p, float(result.loglikelihood - n * np.log(scale))))
        if best is None or result.loglikelihood > best.loglikelihood:
            best, best_p = result, p

    assert best is not None  # _ALD_PRANGE is non-empty
    return _build_filoggarch_fit(
        best,
        "sald" if skewed else "ald",
        scale,
        n,
        k=best.params.size + 1,
        extra_params={"P": float(best_p)},
        profile=tuple(profile),
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
