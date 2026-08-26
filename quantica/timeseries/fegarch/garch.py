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

from quantica.timeseries.fegarch.distributions import (
    AverageLaplace,
    FernandezSteelSkew,
    get_distribution,
)
from quantica.timeseries.fegarch.qmle import QMLEResult, quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "GarchFit",
    "fit_garch",
    "garch_recursion",
    "garch_sim",
]

_VAR_NAMES = ("mu", "omega", "alpha", "beta")

#: fEGarch's ``Prange = c(1, 5)`` for the ALD: the integer grid over which the polynomial-degree
#: P is *profiled* (fit the continuous params at each fixed P, keep the best log-likelihood) rather
#: than optimized continuously. P is discrete, so it never enters the inner QMLE optimizer.
_ALD_PRANGE = (1, 2, 3, 4, 5)


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
    profile : tuple of (int, float) or None
        Only for the ALD (whose degree ``P`` is profiled over an integer grid, not optimized): the
        ``(P, log-likelihood)`` pairs from the grid search, so the profile shape and the selected
        ``P`` are inspectable. ``None`` for continuous-shape distributions.
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
    profile: tuple[tuple[int, float], ...] | None = None


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

    variance = float(np.var(scaled, ddof=1))
    var_start = (float(np.mean(scaled)), variance * 0.05, 0.05, 0.90)
    var_bounds = ((-10.0, 10.0), (1e-8, 1e6), (0.0, 0.9999), (0.0, 0.9999))

    # The ALD's degree P is a discrete construction parameter that fEGarch *profiles* over an
    # integer grid (it never enters the continuous optimizer); every other distribution's shape
    # parameters are estimated jointly by the QMLE engine. These are structurally different fits.
    # ``sald`` is the compound case: the same P-grid, with the FS ``skew`` fit jointly at each P.
    if cond_dist in ("ald", "sald"):
        return _fit_garch_ald(scaled, scale, n, var_start, var_bounds, skewed=cond_dist == "sald")

    distribution = get_distribution(cond_dist)

    # Distribution shape parameters can lie on a near-flat likelihood ridge (e.g. Student-t df on
    # near-normal data, where the MLE is df -> infinity): L-BFGS-B's projected-gradient step stalls
    # there and stops far short, so the shape-parameter fits use derivative-free Nelder-Mead, which
    # climbs the flat ridge to the boundary. The norm path (no shape parameter) keeps L-BFGS-B, so
    # it stays bit-identical to the validated Phase-1 fixture.
    if distribution.param_names:
        method: str = "Nelder-Mead"
        options: dict[str, object] | None = {"maxiter": 20000, "maxfev": 20000, "fatol": 1e-10}
    else:
        method, options = "L-BFGS-B", None

    result = quasi_max_likelihood(
        scaled,
        garch_recursion,
        distribution,
        var_start=var_start,
        var_bounds=var_bounds,
        var_names=_VAR_NAMES,
        mean=True,
        method=method,
        options=options,
    )
    return _garch_fit_from_result(result, cond_dist, scale, n, k=result.params.size)


def _garch_fit_from_result(
    result: QMLEResult,
    cond_dist: str,
    scale: float,
    n: int,
    *,
    k: int,
    extra_params: dict[str, float] | None = None,
    profile: tuple[tuple[int, float], ...] | None = None,
) -> GarchFit:
    """Assemble a :class:`GarchFit` from a QMLE result: undo the scaling and form AIC/BIC.

    Parameters
    ----------
    result : QmleResult
        The fitted engine result on the rescaled returns.
    cond_dist : str
        The conditional-distribution code.
    scale : float
        The return scale that was divided out before fitting.
    n : int
        Number of observations.
    k : int
        Parameter count for the AIC/BIC penalty (includes a profiled degree where one applies).
    extra_params : dict of str to float, optional
        Extra parameters not produced by the optimizer (e.g. the ALD's profiled ``P``).
    profile : tuple of (int, float) or None, optional
        The ALD's ``(P, log-likelihood)`` grid, forwarded to :class:`GarchFit`.

    Returns
    -------
    GarchFit
        The fitted model in original units.
    """
    n_shape = len(result.param_names) - len(_VAR_NAMES)
    # Undo the scaling: mu ~ scale, omega ~ scale^2, alpha/beta and shape params invariant.
    factors = np.array([scale, scale**2, 1.0, 1.0] + [1.0] * n_shape)
    values = result.params * factors
    names = result.param_names
    params = {name: float(v) for name, v in zip(names, values, strict=True)}
    std_errors = {
        name: float(se * f) for name, se, f in zip(names, result.std_errors, factors, strict=True)
    }
    if extra_params:
        params.update(extra_params)

    loglik = result.loglikelihood - n * np.log(scale)
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


def _fit_garch_ald(
    scaled: FloatArray,
    scale: float,
    n: int,
    var_start: tuple[float, ...],
    var_bounds: tuple[tuple[float, float], ...],
    *,
    skewed: bool = False,
) -> GarchFit:
    """Fit GARCH(1,1)/ALD (or /sALD) by profiling the degree ``P`` over fEGarch's grid ``Prange``.

    For each fixed ``P`` in :data:`_ALD_PRANGE` the continuous parameters are fit at that ``P``, and
    the ``P`` with the highest log-likelihood is selected. ``P`` never enters the continuous
    optimizer but *is* counted in the AIC/BIC penalty (``k``), matching fEGarch. For ``ald`` the
    inner fit is just ``(mu, omega, alpha, beta)`` (a fixed-shape ALD, so gradient-based L-BFGS-B,
    ``k = 5``). For ``sald`` the Fernández-Steel ``skew`` rides in the inner vector too — the
    compound P-grid x continuous-skew case — fit by Nelder-Mead (flat skew ridge near 1), ``k = 6``.

    Parameters
    ----------
    scaled : ndarray
        The rescaled return series.
    scale : float
        The scale divided out.
    n : int
        Number of observations.
    var_start, var_bounds : tuple
        Starts and bounds for the four continuous parameters.
    skewed : bool, optional
        If ``True`` fit the skewed ``sald`` (FS skew fit jointly at each ``P``); else plain ``ald``.

    Returns
    -------
    GarchFit
        The best-``P`` fit, carrying the full ``(P, log-likelihood)`` profile.
    """
    method = "Nelder-Mead" if skewed else "L-BFGS-B"
    options: dict[str, object] | None = (
        {"maxiter": 20000, "maxfev": 20000, "fatol": 1e-10} if skewed else None
    )
    best: QMLEResult | None = None
    best_p = 0
    profile: list[tuple[int, float]] = []
    for p in _ALD_PRANGE:
        base = AverageLaplace(p=p)
        distribution = FernandezSteelSkew(base) if skewed else base
        result = quasi_max_likelihood(
            scaled,
            garch_recursion,
            distribution,
            var_start=var_start,
            var_bounds=var_bounds,
            var_names=_VAR_NAMES,
            mean=True,
            method=method,
            options=options,
        )
        profile.append((p, float(result.loglikelihood - n * np.log(scale))))
        if best is None or result.loglikelihood > best.loglikelihood:
            best, best_p = result, p

    assert best is not None  # _ALD_PRANGE is non-empty
    # P is profiled, not optimized, but fEGarch counts it in the penalty: k = continuous_params + 1.
    return _garch_fit_from_result(
        best,
        "sald" if skewed else "ald",
        scale,
        n,
        k=best.params.size + 1,
        extra_params={"P": float(best_p)},
        profile=tuple(profile),
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
