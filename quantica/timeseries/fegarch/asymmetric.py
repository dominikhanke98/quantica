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

**Pre-sample conditioning (reconciled per recursion against the fixtures).** The
:math:`\sigma^\delta` **state** is always seeded from the **unbiased** sample variance,
:math:`\sigma_0^\delta = \operatorname{Var}(r)^{\delta/2}` (``ddof=1``). The pre-sample
**news-impact** ``kernel_0`` is fEGarch's estimate of :math:`\operatorname{E}|\varepsilon|^\delta`,
which the fixtures show it computes two different ways depending on the recursion:

.. math::

    \text{kernel}_0 =
    \begin{cases}
      \operatorname{Var}(r)^{\delta/2} & \sigma^2 / \sigma^\delta\text{ recursions (GJR, APARCH)},\\
      \tfrac1n\sum_t |\varepsilon_t| & \sigma\text{-recursion (TGARCH,}\ \delta = 1).
    \end{cases}

The two forms — the **variance-power** and the **first absolute sample moment** — coincide at
:math:`\delta = 2` and fork otherwise. This is an **empirical reconciliation against the committed
output**, not a proven internal identity: the fEGarch source is never consulted (CLAUDE.md §12), and
neither single form fits all four models (each nails three of four; TGARCH at :math:`\delta = 1` and
APARCH at :math:`\delta \approx 2.41` give opposite verdicts). Under this per-recursion convention
every fixture's conditional-SD series is reproduced to :math:`\lesssim 10^{-5}` relative; the
recursion **form** is separately machine-exact (verified by seeding from the fixture's own
:math:`\sigma_0`). Full derivation, the fork numbers and the corroborating ``ddof`` evidence are in
``docs/fegarch-spec-notes.md`` §4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import (
    AverageLaplace,
    FernandezSteelSkew,
    get_distribution,
)
from quantica.timeseries.fegarch.garch import _ALD_PRANGE, GarchFit
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.timeseries.fegarch.qmle import QMLEResult, VarianceRecursion

# Optimizer options for the derivative-free Nelder-Mead used on the near-flat shape/skew ridges
# (Student-t df -> inf, FS skew -> 1 on near-symmetric data), matching the GARCH shape-fit path.
_SHAPE_OPTIONS: dict[str, object] = {"maxiter": 20000, "maxfev": 20000, "fatol": 1e-10}

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

# Pre-sample news-impact seed, reconciled per recursion against the fixtures (spec-notes §4).
# fEGarch's estimate of E[|eps|^delta] takes two forms that coincide at delta=2 and fork otherwise:
# the variance-power (Var r)^(delta/2) for the sigma^2 / sigma^delta recursions, and the first
# absolute sample moment mean|eps| for the sigma-recursion.
_SEED_VARIANCE_POWER = "variance_power"  # GJR (delta=2) and APARCH (free delta)
_SEED_ABS_MOMENT = "abs_moment"  # TGARCH (delta=1, the sigma-recursion)


def _aparch_family_variance(
    mu: float,
    omega: float,
    phi1: float,
    beta1: float,
    gamma1: float,
    delta: float,
    returns: FloatArray,
    *,
    kernel_seed: str,
) -> FloatArray:
    r"""The APARCH-power recursion, returned as the conditional variance :math:`\sigma_t^2`.

    Implements :math:`\sigma_t^\delta = \omega + \phi_1(|\varepsilon_{t-1}| -
    \gamma_1\varepsilon_{t-1})^\delta + \beta_1\sigma_{t-1}^\delta`, then converts back to
    :math:`\sigma_t^2` for the QMLE engine. Since :math:`|\gamma_1| < 1` the kernel base
    :math:`|\varepsilon|(1 - \gamma_1\operatorname{sign}\varepsilon)` is non-negative, so the
    fractional power is real.

    Pre-sample conditioning (reconciled per recursion, spec-notes §4). The :math:`\sigma^\delta`
    state is always seeded from the **unbiased** sample variance,
    :math:`\sigma_0^\delta = \operatorname{Var}(r)^{\delta/2}`. The pre-sample news-impact
    ``kernel_0`` is fEGarch's estimate of :math:`\operatorname{E}|\varepsilon|^\delta`, which forks
    by ``kernel_seed``: the **variance-power** :math:`\operatorname{Var}(r)^{\delta/2}` for the
    :math:`\sigma^2` / :math:`\sigma^\delta` recursions (GJR, APARCH), or the **first absolute
    moment** :math:`\tfrac1n\sum|\varepsilon_t|` for the :math:`\sigma`-recursion (TGARCH). The two
    forms coincide at :math:`\delta = 2` and diverge otherwise.
    """
    y = np.asarray(returns, dtype=np.float64)
    resid = y - mu
    n = y.size
    var1 = float(np.var(y, ddof=1))  # unbiased sample variance (ddof=1)
    sig_delta_0 = var1 ** (delta / 2.0)  # sigma^delta state seed (all recursions)
    if kernel_seed == _SEED_VARIANCE_POWER:
        kernel_0 = var1 ** (delta / 2.0)  # sigma^2 / sigma^delta recursions: (Var r)^(delta/2)
    else:  # _SEED_ABS_MOMENT
        kernel_0 = float(np.mean(np.abs(resid) ** delta))  # sigma-recursion: first absolute moment
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
    return _aparch_family_variance(
        mu, omega, phi1, beta1, gamma1, 2.0, returns, kernel_seed=_SEED_VARIANCE_POWER
    )


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
    return _aparch_family_variance(
        mu, omega, phi1, beta1, gamma1, 1.0, returns, kernel_seed=_SEED_ABS_MOMENT
    )


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
    return _aparch_family_variance(
        mu, omega, phi1, beta1, gamma1, delta, returns, kernel_seed=_SEED_VARIANCE_POWER
    )


def _build_asym_fit(
    result: QMLEResult,
    cond_dist: str,
    scale: float,
    n: int,
    delta_fixed: float | None,
    var_names: tuple[str, ...],
    *,
    k: int,
    extra_params: dict[str, float] | None = None,
    profile: tuple[tuple[int, float], ...] | None = None,
) -> GarchFit:
    """Assemble a :class:`GarchFit` from a QMLE result: undo the delta-dependent scaling + AIC/BIC.

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
    delta_fixed : float or None
        The fixed power (GJR/TGARCH) or ``None`` for the free-delta APARCH.
    var_names : tuple of str
        The variance-recursion parameter names (five for GJR/TGARCH, six for APARCH).
    k : int
        Parameter count for the AIC/BIC penalty (includes a profiled degree where one applies).
    extra_params : dict of str to float, optional
        Extra parameters not produced by the optimizer (e.g. the ALD's profiled ``P``).
    profile : tuple of (int, float) or None, optional
        The ALD ``(P, log-likelihood)`` grid, forwarded to :class:`GarchFit`.

    Returns
    -------
    GarchFit
        The fitted model in original units.
    """
    names = result.param_names
    delta = float(delta_fixed) if delta_fixed is not None else float(result.params[5])
    # Undo the scaling: mu ~ scale, omega ~ scale^delta; phi1/beta1/gamma1/delta/shape invariant.
    n_extra = len(names) - len(var_names)  # distribution shape parameters (0 for ald's profiled P)
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


def _fit_aparch_family(
    returns: FloatArray, cond_dist: str, *, delta_fixed: float | None
) -> GarchFit:
    """Shared QMLE fit (``delta_fixed`` set for GJR/TGARCH, ``None`` for the free-delta APARCH).

    Inherits the GARCH distribution machinery wholesale: the shape parameters ride in the fitted
    vector, near-flat shape/skew ridges use derivative-free Nelder-Mead, and the ALD's degree ``P``
    is profiled over :data:`_ALD_PRANGE` (an outer grid, counted in the AIC/BIC penalty). APARCH
    additionally carries a jointly-estimated ``delta`` (the 7/8-parameter compounds).
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude omega
    scaled = y / scale

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

    # The ALD (and sALD) profiles its integer degree P over the grid, exactly as GARCH; every other
    # distribution's shape/skew parameters are estimated jointly by the engine.
    if cond_dist in ("ald", "sald"):
        return _fit_aparch_ald(
            scaled,
            scale,
            n,
            recursion,
            var_names,
            var_start,
            var_bounds,
            delta_fixed,
            skewed=cond_dist == "sald",
        )

    distribution = get_distribution(cond_dist)
    # Near-flat shape/skew ridges (Student-t df, FS skew) stall L-BFGS-B, so shape-parameter fits
    # Nelder-Mead; the norm path (no shape parameter) keeps L-BFGS-B and stays bit-identical.
    if distribution.param_names:
        method: str = "Nelder-Mead"
        options: dict[str, object] | None = _SHAPE_OPTIONS
    else:
        method, options = "L-BFGS-B", None

    result = quasi_max_likelihood(
        scaled,
        recursion,
        distribution,
        var_start=var_start,
        var_bounds=var_bounds,
        var_names=var_names,
        mean=True,
        method=method,
        options=options,
    )
    return _build_asym_fit(
        result, cond_dist, scale, n, delta_fixed, var_names, k=result.params.size
    )


def _fit_aparch_ald(
    scaled: FloatArray,
    scale: float,
    n: int,
    recursion: VarianceRecursion,
    var_names: tuple[str, ...],
    var_start: tuple[float, ...],
    var_bounds: tuple[tuple[float, float], ...],
    delta_fixed: float | None,
    *,
    skewed: bool,
) -> GarchFit:
    """Profile the ALD degree ``P`` over :data:`_ALD_PRANGE` for a GJR/TGARCH/APARCH recursion.

    Mirrors the GARCH ``_fit_garch_ald`` fork: fit the continuous parameters at each fixed ``P``
    (plus the FS ``skew`` for ``sald``), select the best log-likelihood, and count ``P`` in the
    AIC/BIC penalty. ``P`` never enters the continuous optimizer.
    """
    method = "Nelder-Mead" if skewed else "L-BFGS-B"
    options = _SHAPE_OPTIONS if skewed else None
    best: QMLEResult | None = None
    best_p = 0
    profile: list[tuple[int, float]] = []
    for p in _ALD_PRANGE:
        base = AverageLaplace(p=p)
        distribution = FernandezSteelSkew(base) if skewed else base
        result = quasi_max_likelihood(
            scaled,
            recursion,
            distribution,
            var_start=var_start,
            var_bounds=var_bounds,
            var_names=var_names,
            mean=True,
            method=method,
            options=options,
        )
        profile.append((p, float(result.loglikelihood - n * np.log(scale))))
        if best is None or result.loglikelihood > best.loglikelihood:
            best, best_p = result, p

    assert best is not None  # _ALD_PRANGE is non-empty
    return _build_asym_fit(
        best,
        "sald" if skewed else "ald",
        scale,
        n,
        delta_fixed,
        var_names,
        k=best.params.size + 1,  # P is profiled, not optimized, but counts in the penalty
        extra_params={"P": float(best_p)},
        profile=tuple(profile),
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
