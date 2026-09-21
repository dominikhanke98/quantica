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
    "RiskForecast",
    "RollingForecast",
    "measure_risk",
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
