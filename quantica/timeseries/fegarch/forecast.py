r"""Rolling one-step forecasting and VaR/ES assembly for the fEGarch port (Phase 6).

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (the fEGarch reference manual's ``predict_roll`` / ``measure_risk`` help and WP171 Eqs. 61--62),
    **never** the `fEGarch` source. Validated against committed `fEGarch` *output* fixtures.

The **no-refit rolling forecast** (fEGarch's ``predict_roll(refit_after = NULL)``): a model is fit
once on the training window, then iterated one step ahead over the held-out test returns with the
parameters *frozen at the training fit* — no refitting. For a constant-mean GARCH this is exactly
the existing conditional-variance recursion
(:func:`~quantica.timeseries.fegarch.garch_recursion`) *continued* past the training window over
the realized test returns:

.. math::

    \hat\sigma_t^2 = \omega + \alpha\,\varepsilon_{t-1}^2 + \beta\,\hat\sigma_{t-1}^2,
    \qquad \hat\mu_t = \mu \quad(\text{constant mean}),

so :func:`predict_roll` simply runs :func:`garch_recursion` with the *training* parameters over the
**full** series and keeps the last ``n_test`` values. The **train-then-continue seed is
machine-exact for any reasonable training seed** — the long training window decays the pre-sample
transient to nothing before the test region — so there is *no* new seed convention and *no* bounded
limit here (unlike the dual-mean :math:`\sigma_0^2`); reconstructing the fixture's rolling
:math:`\hat\sigma_t` this way matches to ``~7e-18``.

The **risk assembly** (:func:`measure_risk`, WP171 Eqs. 61--62) scales and shifts the standardized
innovation's tail quantile / ES by the conditional mean and SD:

.. math::

    \mathrm{VaR}_\alpha(t) = \hat\mu_t + \hat\sigma_t\, q_\eta(1-\alpha),
    \qquad
    \mathrm{ES}_\alpha(t) = \hat\mu_t + \hat\sigma_t\, \mathrm{ES}_\eta(\alpha),

with :math:`q_\eta` the existing distribution :meth:`ppf` and :math:`\mathrm{ES}_\eta` the new
:meth:`~quantica.timeseries.fegarch.ConditionalDistribution.expected_shortfall`. Both are
**return-space** thresholds (losses carry a negative sign, matching fEGarch); the sign map into the
loss-space risk-pillar backtests is
:func:`~quantica.risk.backtest.backtest_return_forecasts`. Reconstructing the fixture's VaR/ES this
way matches to ``~1e-6`` — **not** machine-exact: :math:`\hat\sigma_t` is machine-exact, so the
residual is :math:`\hat\sigma_t\,(q_{\mathrm{scipy}} - q_{\mathrm R})`, the R-vs-SciPy
quantile-implementation difference (documented in ``docs/fegarch-spec-notes.md``).

References
----------
WP171 (Feng, Peitz & co-authors) — the fEGarch forecasting appendix and the VaR/ES definitions
(Eqs. 61--62). fEGarch reference manual — ``predict_roll`` / ``measure_risk`` help pages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import (
    AverageLaplace,
    ConditionalDistribution,
    FernandezSteelSkew,
    get_distribution,
)
from quantica.timeseries.fegarch.garch import GarchFit, garch_recursion

if TYPE_CHECKING:
    from collections.abc import Sequence

    from quantica.core.types import FloatArray

__all__ = [
    "MultiStepForecast",
    "RiskForecast",
    "RollingForecast",
    "measure_risk",
    "predict",
    "predict_roll",
]

#: fEGarch's default risk levels for ``measure_risk`` (the 97.5% and 99% VaR/ES).
_DEFAULT_LEVELS: tuple[float, ...] = (0.975, 0.99)


@dataclass(frozen=True)
class RollingForecast:
    """A no-refit rolling one-step forecast over a held-out test window.

    Attributes
    ----------
    sigma : ndarray, shape (n_test,)
        The rolling one-step conditional standard deviations :math:`\\hat\\sigma_t`.
    cmeans : ndarray, shape (n_test,)
        The rolling one-step conditional means :math:`\\hat\\mu_t` (constant for a plain GARCH).
    n_test : int
        The number of held-out test points forecast.
    """

    sigma: FloatArray
    cmeans: FloatArray
    n_test: int


@dataclass(frozen=True)
class RiskForecast:
    """Conditional VaR/ES forecasts over the test window, one series per confidence level.

    The values are **return-space** thresholds: losses carry a negative sign (fEGarch's
    convention), so e.g. a 97.5% VaR is a negative number. Sign-mapped into the loss-space
    backtests by :func:`~quantica.risk.backtest.backtest_return_forecasts`.

    Attributes
    ----------
    levels : tuple of float
        The confidence levels :math:`\\alpha` (e.g. ``(0.975, 0.99)``).
    var : dict of float to ndarray
        Value-at-Risk series keyed by level, each shape ``(n_test,)``.
    es : dict of float to ndarray
        Expected-Shortfall series keyed by level, each shape ``(n_test,)``.
    """

    levels: tuple[float, ...]
    var: dict[float, FloatArray]
    es: dict[float, FloatArray]


def predict_roll(
    fit: GarchFit, returns: FloatArray, *, n_test: int | None = None
) -> RollingForecast:
    r"""No-refit rolling one-step forecast of :math:`\hat\sigma_t` / :math:`\hat\mu_t`.

    Freezes the parameters at the training ``fit`` and continues the GARCH variance recursion past
    the training window over the realized test returns — equivalently, runs
    :func:`~quantica.timeseries.fegarch.garch_recursion` with the training parameters over the
    **full** ``returns`` series and keeps the last ``n_test`` values (the train-then-continue seed
    is machine-exact for any reasonable training seed; see the module docstring). The mean is the
    constant fitted :math:`\mu` (plain GARCH), so :math:`\hat\mu_t` is a constant series.

    Parameters
    ----------
    fit : GarchFit
        A GARCH(1,1) fit on the **training** window (its ``params`` supply ``mu, omega, alpha,
        beta`` and ``n_obs`` the training length).
    returns : ndarray, shape (n_train + n_test,)
        The full return series — the training returns followed by the realized test returns.
    n_test : int, optional
        The number of test points to forecast (default ``len(returns) - fit.n_obs``).

    Returns
    -------
    RollingForecast
        The rolling :math:`\hat\sigma_t` / :math:`\hat\mu_t` over the ``n_test`` test points.

    Raises
    ------
    ValueError
        If ``n_test`` is not positive or exceeds the series length.
    """
    y = np.asarray(returns, dtype=np.float64)
    if n_test is None:
        n_test = y.size - fit.n_obs
    if not 0 < n_test <= y.size:
        raise ValueError(f"n_test must satisfy 0 < n_test <= {y.size}, got {n_test}")

    params = np.array(
        [fit.params["mu"], fit.params["omega"], fit.params["alpha"], fit.params["beta"]],
        dtype=np.float64,
    )
    sigma2 = garch_recursion(params, y)
    sigma = np.sqrt(sigma2[y.size - n_test :])
    cmeans = np.full(n_test, fit.params["mu"], dtype=np.float64)
    return RollingForecast(sigma=sigma, cmeans=cmeans, n_test=n_test)


def _distribution_for_fit(fit: GarchFit) -> tuple[ConditionalDistribution, Sequence[float] | None]:
    """The conditional distribution + its shape parameters for a fitted model.

    Rebuilds the exact standardized innovation the fit used: the ALD's profiled degree ``P`` is a
    construction parameter (not in ``param_names``), so ``ald`` / ``sald`` are reconstructed with
    it; every other distribution reads its shape parameters straight from ``fit.params``.
    """
    if fit.cond_dist in ("ald", "sald"):
        base = AverageLaplace(p=int(fit.params["P"]))
        dist: ConditionalDistribution = (
            FernandezSteelSkew(base) if fit.cond_dist == "sald" else base
        )
    else:
        dist = get_distribution(fit.cond_dist)
    shape = [fit.params[name] for name in dist.param_names]
    return dist, (shape or None)


def measure_risk(
    forecast: RollingForecast,
    fit: GarchFit,
    *,
    levels: Sequence[float] = _DEFAULT_LEVELS,
) -> RiskForecast:
    r"""Assemble conditional VaR/ES from a rolling forecast (WP171 Eqs. 61--62).

    For each level :math:`\alpha`, :math:`\mathrm{VaR}_\alpha(t) = \hat\mu_t + \hat\sigma_t\,
    q_\eta(1-\alpha)` and :math:`\mathrm{ES}_\alpha(t) = \hat\mu_t + \hat\sigma_t\,
    \mathrm{ES}_\eta(\alpha)`, where :math:`q_\eta = F^{-1}(1-\alpha)` is the fitted distribution's
    :meth:`ppf` and :math:`\mathrm{ES}_\eta` its
    :meth:`~quantica.timeseries.fegarch.ConditionalDistribution.expected_shortfall`. The outputs
    are return-space thresholds (losses negative), matching the fEGarch fixtures.

    Parameters
    ----------
    forecast : RollingForecast
        The rolling :math:`\hat\sigma_t` / :math:`\hat\mu_t` from :func:`predict_roll`.
    fit : GarchFit
        The fitted model — supplies the conditional distribution and its shape parameters.
    levels : sequence of float, optional
        The confidence levels :math:`\alpha` (default ``(0.975, 0.99)``).

    Returns
    -------
    RiskForecast
        The VaR and ES series, one per level.
    """
    dist, shape = _distribution_for_fit(fit)
    var: dict[float, FloatArray] = {}
    es: dict[float, FloatArray] = {}
    for alpha in levels:
        q_eta = float(dist.ppf(np.array([1.0 - alpha], dtype=np.float64), shape)[0])
        es_eta = dist.expected_shortfall(alpha, shape)
        var[alpha] = np.asarray(forecast.cmeans + forecast.sigma * q_eta, dtype=np.float64)
        es[alpha] = np.asarray(forecast.cmeans + forecast.sigma * es_eta, dtype=np.float64)
    return RiskForecast(levels=tuple(levels), var=var, es=es)


@dataclass(frozen=True)
class MultiStepForecast:
    r"""A multi-step (``n_ahead``) point forecast over a fixed horizon from the fitted-sample end.

    Distinct from :class:`RollingForecast` (one-step rolling over held-out data): this is the pure
    :math:`h`-step-ahead forecast, :math:`h = 1, \dots, n_{\text{ahead}}`.

    Attributes
    ----------
    sigma : ndarray, shape (n_ahead,)
        The :math:`h`-step-ahead conditional standard-deviation forecasts :math:`\hat\sigma_{n+h}`.
    cmeans : ndarray, shape (n_ahead,)
        The :math:`h`-step-ahead conditional-mean forecasts (constant for a plain constant-mean
        model).
    n_ahead : int
        The forecast horizon.
    """

    sigma: FloatArray
    cmeans: FloatArray
    n_ahead: int


def predict(fit: GarchFit, returns: FloatArray, *, n_ahead: int = 10) -> MultiStepForecast:
    r"""Multi-step (``n_ahead``) point forecast — the naive iterated forecast fEGarch ships.

    Iterates the fitted recursion forward from the fitted-sample-end state, replacing each future
    news term by its expectation (the ``predict`` mechanism, distinct from :func:`predict_roll`'s
    one-step rolling). Two families are supported, both reproducing the committed fixtures to
    machine precision:

    * **Variance-recursion GARCH(1,1)** (params ``mu, omega, alpha, beta``): the standard iterated
      forecast :math:`\hat\sigma_{n+h}^2 = \omega + (\alpha+\beta)\hat\sigma_{n+h-1}^2` for
      :math:`h \ge 2` (the future :math:`\varepsilon_{n+h-1}^2 \to` its expectation
      :math:`\hat\sigma_{n+h-1}^2`); :math:`h = 1` uses the realized :math:`\varepsilon_n`. It is
      **unbiased** and converges to the unconditional variance :math:`\omega/(1-\alpha-\beta)`.
    * **Type-I EGF (EGARCH)** (params ``mu, omega_sig, phi1, kappa, gamma``): iterate the
      log-variance with the future news :math:`g(\eta) = 0` (its expectation by construction),
      :math:`\ln\hat\sigma_{n+h}^2 = \omega + \phi_1\ln\hat\sigma_{n+h-1}^2` for :math:`h \ge 2`
      (:math:`h = 1` uses the realized :math:`\eta_n` through :math:`g(\eta_n) = \kappa\eta_n +
      \gamma(|\eta_n| - \operatorname{E}|z|)`) and report
      :math:`\hat\sigma = \exp(\ln\hat\sigma^2/2)`. This is the **naive/biased** iterate fEGarch
      ships: :math:`\hat\sigma` converges to
      :math:`\exp(\operatorname{E}[\ln\sigma^2]/2) = \exp(\omega_\sigma/2)`, **not**
      :math:`\operatorname{E}[\sigma_{n+h}]` (the Jensen / :math:`\operatorname{E}[e^{g/2}]` gap is
      left in). This reproduces fEGarch's flagged-biased multi-step forecast and — per the
      clean-room brief — is deliberately **not** bias-corrected (a correction would be original
      work, not a port).

    The mean forecast is the constant fitted :math:`\mu` (constant-mean models; the dual-mean
    multi-step mean is deferred with the other dual-mean forecasting). ``predict`` at :math:`h = 1`
    equals :func:`predict_roll`'s first :math:`\hat\sigma` (the same one-step-ahead value).

    Parameters
    ----------
    fit : GarchFit
        The fitted model (its ``params``, ``conditional_volatility`` end-state and ``n_obs`` seed
        the iterate). The family is inferred from the parameter names.
    returns : ndarray
        The return series the model was fit on; only the fitted window ``returns[:fit.n_obs]`` is
        used (its last value supplies the realized :math:`h = 1` news).
    n_ahead : int, optional
        The forecast horizon (default 10).

    Returns
    -------
    MultiStepForecast
        The :math:`\hat\sigma` / :math:`\hat\mu` forecasts over ``n_ahead`` steps.

    Raises
    ------
    ValueError
        If ``n_ahead`` is not positive, or the fitted window is empty.
    NotImplementedError
        If the fit is not a supported family (multi-step for the asymmetric/power, fractionally
        integrated and MEGARCH/MLog-GARCH models is deferred until their fixtures exist).
    """
    if n_ahead < 1:
        raise ValueError(f"n_ahead must be >= 1, got {n_ahead}")
    y = np.asarray(returns, dtype=np.float64)[: fit.n_obs]
    if y.size == 0:
        raise ValueError("fitted window returns[:fit.n_obs] is empty")
    mu = fit.params["mu"]
    sigma_n = float(fit.conditional_volatility[-1])
    keys = set(fit.params)

    if {"omega_sig", "phi1", "kappa", "gamma"} <= keys:
        # Type-I EGF (EGARCH): iterate ln sigma^2 forward with future g(eta) = 0.
        omega = fit.params["omega_sig"] * (1.0 - fit.params["phi1"])
        phi1, kappa, gamma = fit.params["phi1"], fit.params["kappa"], fit.params["gamma"]
        dist, shape = _distribution_for_fit(fit)
        e_abs = dist.abs_moment(shape)
        eta_n = (y[-1] - mu) / sigma_n
        log_var = np.empty(n_ahead, dtype=np.float64)
        log_var[0] = (
            omega + kappa * eta_n + gamma * (abs(eta_n) - e_abs) + phi1 * 2.0 * np.log(sigma_n)
        )
        for h in range(1, n_ahead):
            log_var[h] = omega + phi1 * log_var[h - 1]
        sigma = np.exp(log_var / 2.0)
    elif {"omega", "alpha", "beta"} <= keys and not (keys & {"gamma", "delta", "d"}):
        # Variance-recursion GARCH(1,1): iterate sigma^2 with future eps^2 -> its expectation.
        omega, alpha, beta = fit.params["omega"], fit.params["alpha"], fit.params["beta"]
        eps_n = y[-1] - mu
        variance = np.empty(n_ahead, dtype=np.float64)
        variance[0] = omega + alpha * eps_n**2 + beta * sigma_n**2
        for h in range(1, n_ahead):
            variance[h] = omega + (alpha + beta) * variance[h - 1]
        sigma = np.sqrt(variance)
    else:
        raise NotImplementedError(
            f"multi-step predict is not built for a fit with params {sorted(keys)}; "
            "only constant-mean GARCH(1,1) and EGARCH(1,1) have committed fixtures"
        )

    cmeans = np.full(n_ahead, mu, dtype=np.float64)
    return MultiStepForecast(
        sigma=np.asarray(sigma, dtype=np.float64), cmeans=cmeans, n_ahead=n_ahead
    )
