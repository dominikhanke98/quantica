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

from quantica.timeseries.fegarch.asymmetric import (
    aparch_recursion,
    gjr_recursion,
    tgarch_recursion,
)
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
    egarch_recursion,
    megarch_recursion,
    mloggarch_recursion,
    type1_news_impact,
)
from quantica.timeseries.fegarch.fiegarch import (
    fiegarch_recursion,
    fimegarch_recursion,
    fimloggarch_recursion,
    theta_coefficients,
)
from quantica.timeseries.fegarch.figarch import figarch_coefficients, figarch_variance_filter
from quantica.timeseries.fegarch.filoggarch import (
    filoggarch_gamma_coefficients,
    filoggarch_recursion,
)
from quantica.timeseries.fegarch.fracdiff import fracdiff_coeffs
from quantica.timeseries.fegarch.garch import GarchFit, garch_recursion
from quantica.timeseries.fegarch.loggarch import loggarch_recursion
from quantica.timeseries.fegarch.mean import arma_mean_residuals, farima_mean_residuals

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


def _aparch_family_variance(fit: GarchFit, y: FloatArray) -> FloatArray:
    r"""Conditional variance :math:`\sigma_t^2` for an APARCH-family fit, continued over ``y``.

    Dispatches on the fitted power :math:`\delta` to the recursion that carries that model's own
    pre-sample seed convention (spec-notes §4): :math:`\delta = 2` → :func:`gjr_recursion`,
    :math:`\delta = 1` → :func:`tgarch_recursion`, free :math:`\delta` → :func:`aparch_recursion`.
    The pre-sample seed transient decays to nothing long before the test window, so the choice does
    not affect the rolling forecast there (it is seam-exact regardless; §33.3).
    """
    mu, omega, phi1 = fit.params["mu"], fit.params["omega"], fit.params["phi1"]
    beta1, gamma1, delta = fit.params["beta1"], fit.params["gamma1"], fit.params["delta"]
    if delta == 2.0:
        return gjr_recursion(np.array([mu, omega, phi1, beta1, gamma1], dtype=np.float64), y)
    if delta == 1.0:
        return tgarch_recursion(np.array([mu, omega, phi1, beta1, gamma1], dtype=np.float64), y)
    return aparch_recursion(np.array([mu, omega, phi1, beta1, gamma1, delta], dtype=np.float64), y)


def _fi_variance(fit: GarchFit, y: FloatArray) -> FloatArray:
    r"""Rolling-forecast conditional variance for a long-memory FI fit over ``y`` (spec-notes §34).

    The one-step rolling forecast uses the **forecast** truncation convention, not the in-sample
    fit: the long-memory weights condition on the observed sample with a **pre-sample 0** (§34.2 --
    the in-sample warm-up seed, FIGARCH's 50-term ``Var``, is not re-applied; gate-confirmed).
    Dispatches the two structures: FIEGARCH (EGF MA(inf) in ``g``, already 0-seeded in-sample) and
    the variance-recursion FI family (FIGARCH / FIGJR / FITGARCH / FIAPARCH), a pre-sample-0
    ARCH(inf) filter of the observed news ``(|e|-gamma e)^delta`` (``e^2`` at ``gamma=0, delta=2``).
    """
    p = fit.params
    y = np.asarray(y, dtype=np.float64)
    if (
        "omega_sig" in p
    ):  # FIEGARCH: g-history is 0 for t <= 0 in-sample, so the recursion is 0-seeded
        dist, shape = _distribution_for_fit(fit)
        vec = [p["mu"], p["omega_sig"], p["phi1"], p["kappa"], p["gamma"], p["d"]]
        return fiegarch_recursion(
            np.array(vec, dtype=np.float64), y, abs_moment=dist.abs_moment(shape)
        )
    mu, phi1, beta1, d = p["mu"], p["phi1"], p["beta1"], p["d"]
    gamma = p.get("gamma", 0.0)
    delta = p.get("delta", 2.0)
    eps = y - mu
    news = (np.abs(eps) - gamma * eps) ** delta  # eps^2 at gamma=0, delta=2 (FIGARCH)
    theta = figarch_coefficients(phi1, beta1, d, y.size + 1)
    sig_delta = figarch_variance_filter(news, theta, p["omega"], 0.0, n_presample=0)
    return np.asarray(sig_delta ** (2.0 / delta), dtype=np.float64)


# Type-I EGF model -> (constant-set, magnitude centering moment, asymmetry uses signed-log-modulus).
# The FI variants map to their short-memory base (same g; only the weight propagation differs).
_TYPE1_EGF: dict[str, tuple[tuple[float, float, float, float], str, bool]] = {
    "egarch": (EGARCH_CONSTANTS, "abs_moment", False),
    "megarch": (MEGARCH_CONSTANTS, "abs_moment", True),
    "mloggarch": (MLOGGARCH_CONSTANTS, "mean_log_modulus", True),
}
_FI_TYPE1_BASE = {"fiegarch": "egarch", "fimegarch": "megarch", "fimloggarch": "mloggarch"}


def _egf_type1_news(
    base_model: str,
    kappa: float,
    gamma: float,
    eta: FloatArray,
    dist: ConditionalDistribution,
    shape: Sequence[float] | None,
) -> FloatArray:
    r"""The Type-I EGF news impact :math:`g(\eta_t)` over an array, for one short-memory base model.

    Reuses the fit-layer :func:`~quantica.timeseries.fegarch.type1_news_impact` with the model's
    constant-set and the distribution's centering moments (spec-notes §35): magnitude centering is
    ``abs_moment`` (EGARCH/MEGARCH) or ``mean_log_modulus`` (MLog-GARCH), asymmetry centering is
    ``mean_signed_log_modulus`` for the modulus-log-asymmetry models (0 for EGARCH's ``g_asy=eta``).
    """
    constants, mag_attr, asy_slm = _TYPE1_EGF[base_model]
    mean_mag = float(getattr(dist, mag_attr)(shape))
    mean_asy = float(dist.mean_signed_log_modulus(shape)) if asy_slm else 0.0
    return np.array(
        [
            type1_news_impact(
                float(e), kappa, gamma, constants=constants, mean_asy=mean_asy, mean_mag=mean_mag
            )
            for e in np.asarray(eta, dtype=np.float64)
        ],
        dtype=np.float64,
    )


def _egf_variance(fit: GarchFit, y: FloatArray) -> FloatArray:
    r"""Conditional variance for an EGF fit over ``y`` -- the model's own in-sample recursion.

    Dispatches the EGF family (spec-notes §35): Type-II (``psi1``) -> Log/FILog-GARCH; Type-I
    (``kappa``/``gamma`` + model tag) -> EGARCH / MEGARCH / MLog-GARCH + FI. The EGF recursion
    is 0-seeded in the news/``xi`` history in-sample, so continuing it over the test window is
    seam-exact (the ``ln Var`` intercept seed decays), giving the rolling one-step forecast.
    """
    p = fit.params
    keys = set(p)
    dist, shape = _distribution_for_fit(fit)
    if "psi1" in keys:  # Type-II (Log-GARCH / FILog-GARCH)
        e_lsq = float(dist.mean_log_sq(shape))
        if "d" in keys:
            vec = [p["mu"], p["omega_sig"], p["phi1"], p["psi1"], p["d"]]
            return filoggarch_recursion(np.array(vec, dtype=np.float64), y, mean_log_sq=e_lsq)
        vec = [p["mu"], p["omega_sig"], p["phi1"], p["psi1"]]
        return loggarch_recursion(np.array(vec, dtype=np.float64), y, mean_log_sq=e_lsq)
    # Type-I: pick the recursion by the model tag (default EGARCH for a bare Type-I signature).
    model = fit.model or "egarch"
    base = _FI_TYPE1_BASE.get(model, model)
    vec = [p["mu"], p["omega_sig"], p["phi1"], p["kappa"], p["gamma"]]
    if "d" in keys:
        vec.append(p["d"])
        v = np.array(vec, dtype=np.float64)
        if base == "egarch":
            return fiegarch_recursion(v, y, abs_moment=dist.abs_moment(shape))
        if base == "megarch":
            return fimegarch_recursion(
                v,
                y,
                abs_moment=dist.abs_moment(shape),
                mean_signed_log_modulus=dist.mean_signed_log_modulus(shape),
            )
        return fimloggarch_recursion(
            v,
            y,
            mean_log_modulus=dist.mean_log_modulus(shape),
            mean_signed_log_modulus=dist.mean_signed_log_modulus(shape),
        )
    v = np.array(vec, dtype=np.float64)
    if base == "egarch":
        return egarch_recursion(v, y, abs_moment=dist.abs_moment(shape))
    if base == "megarch":
        return megarch_recursion(
            v,
            y,
            abs_moment=dist.abs_moment(shape),
            mean_signed_log_modulus=dist.mean_signed_log_modulus(shape),
        )
    return mloggarch_recursion(
        v,
        y,
        mean_log_modulus=dist.mean_log_modulus(shape),
        mean_signed_log_modulus=dist.mean_signed_log_modulus(shape),
    )


def _egf_multistep(fit: GarchFit, returns: FloatArray, n_ahead: int) -> FloatArray:
    r"""EGF multi-step forecast of the log-variance (spec-notes §35) -- Type-I/II, short or FI.

    All four cases iterate ``ln sigma^2`` with the *future* news centered to 0, so the
    forecast is the model's news on the observed history only. Short memory is the AR(1)
    ``ln sigma^2_{n+h} = omega + phi1 ln sigma^2_{n+h-1}`` (h>=2), realized news at h=1; FI is
    the §34 weight-propagation convolution (window extended by the horizon, pre-sample 0). Report
    ``sigma = exp(ln sigma^2 / 2)``, the Jensen-biased plug-in (reproduced-not-corrected, §29.2).
    """
    p = fit.params
    keys = set(p)
    n = fit.n_obs
    mu, omega_sig, phi1 = p["mu"], p["omega_sig"], p["phi1"]
    sigma_n = float(fit.conditional_volatility[-1])
    dist, shape = _distribution_for_fit(fit)
    eta = (np.asarray(returns, dtype=np.float64)[:n] - mu) / np.asarray(fit.conditional_volatility)
    is_fi = "d" in keys
    k = np.arange(n)
    if "psi1" in keys:  # Type-II (Log-GARCH / FILog-GARCH): news xi = ln eta^2 - E[ln eta^2]
        xi = np.log(eta * eta) - float(dist.mean_log_sq(shape))
        if is_fi:  # FILog: gamma-coefficient MA(inf), index h+k (gamma_0 = 0)
            gam = filoggarch_gamma_coefficients(phi1, p["psi1"], p["d"], n + n_ahead + 1)
            xi_rev = xi[::-1]
            log_var = np.array(
                [omega_sig + float(np.sum(gam[h + k] * xi_rev)) for h in range(1, n_ahead + 1)]
            )
        else:  # short Log-GARCH: AR(1), news loading psi1 + phi1
            omega = omega_sig * (1.0 - phi1)
            log_var = np.empty(n_ahead, dtype=np.float64)
            log_var[0] = omega + phi1 * np.log(sigma_n**2) + (p["psi1"] + phi1) * xi[-1]
            for h in range(1, n_ahead):
                log_var[h] = omega + phi1 * log_var[h - 1]
    else:  # Type-I (EGARCH / MEGARCH / MLog-GARCH + FI)
        base = _FI_TYPE1_BASE.get(fit.model or "egarch", fit.model or "egarch")
        news = _egf_type1_news(base, p["kappa"], p["gamma"], eta, dist, shape)
        if is_fi:  # §34 theta convolution, index h-1+k (theta_0 = 1)
            theta = theta_coefficients(phi1, p["d"], n + n_ahead)
            news_rev = news[::-1]
            log_var = np.array(
                [
                    omega_sig + float(np.sum(theta[(h - 1) + k] * news_rev))
                    for h in range(1, n_ahead + 1)
                ]
            )
        else:  # short-memory AR(1), realized g(eta_n) at h=1
            omega = omega_sig * (1.0 - phi1)
            log_var = np.empty(n_ahead, dtype=np.float64)
            log_var[0] = omega + float(news[-1]) + phi1 * np.log(sigma_n**2)
            for h in range(1, n_ahead):
                log_var[h] = omega + phi1 * log_var[h - 1]
    return np.asarray(np.exp(log_var / 2.0), dtype=np.float64)


def _fi_power_multistep(fit: GarchFit, returns: FloatArray, n_ahead: int) -> FloatArray:
    r"""FIGARCH-family multi-step forecast -- the ARCH(inf) iterate (spec-notes §34.1).

    ``m_h = omega + sum_{i=1}^{h-1} theta_i kappa m_{h-i} + sum_{i>=h} theta_i news_{n+h-i}``, with
    the §33 moment ``kappa = E[(|eta|-gamma eta)^delta]`` for future news, the observed news tail
    conditioned back to ``t=1`` with **pre-sample 0**, and the ``theta`` window **extended by the
    horizon** (§34.2). Report ``sigma = (m_h)^(1/delta)``. FIGARCH: ``delta=2, gamma=0`` (kappa=1).
    """
    p = fit.params
    mu, omega, phi1, beta1, d = p["mu"], p["omega"], p["phi1"], p["beta1"], p["d"]
    gamma = p.get("gamma", 0.0)
    delta = p.get("delta", 2.0)
    n = fit.n_obs
    eps = np.asarray(returns, dtype=np.float64)[:n] - mu
    news = (
        np.abs(eps) - gamma * eps
    ) ** delta  # observed (|eps| - gamma eps)^delta (eps^2 if gamma=0)
    dist, shape = _distribution_for_fit(fit)
    kappa = dist.news_moment(delta, gamma, shape)
    theta = figarch_coefficients(phi1, beta1, d, n + n_ahead + 1)  # theta[0] = 0; window extended
    m = np.empty(n_ahead, dtype=np.float64)
    for h in range(1, n_ahead + 1):
        imax = min(theta.size - 1, n + h - 1)
        iobs = np.arange(h, imax + 1)  # observed tail: news time s = n+h-i in [1, n]
        s = n + h - iobs
        valid = (s >= 1) & (s <= n)
        tot = float(np.sum(theta[iobs[valid]] * news[s[valid] - 1]))
        for i in range(1, h):  # future news -> kappa * m_{h-i}
            tot += theta[i] * kappa * m[h - i - 1]
        m[h - 1] = omega + tot
    return np.asarray(m ** (1.0 / delta), dtype=np.float64)


#: The dual-mean ARMA/FARIMA coefficient names (any present => an ARMA/FARIMA-in-mean fit).
_MEAN_KEYS = frozenset({"ar1", "ma1", "D"})


def _is_dual_mean(fit: GarchFit) -> bool:
    """True if the fit carries an ARMA/FARIMA-in-mean block (``ar1`` / ``ma1`` / ``D``)."""
    return bool(_MEAN_KEYS & set(fit.params))


def _conditional_variance(fit: GarchFit, y: FloatArray) -> FloatArray:
    r"""The fitted model's conditional variance :math:`\sigma_t^2` over ``y`` (the family dispatch).

    The shared variance-recursion dispatch used by :func:`predict_roll` and by the dual-mean
    composition (where ``y`` is the mean-residual series). Covers the EGF family, the FI
    variance-recursion family, the APARCH family and plain GARCH.
    """
    keys = set(fit.params)
    if {"omega_sig", "phi1"} <= keys:  # EGF family (EGARCH/MEGARCH/MLog/Log-GARCH + FI variants)
        return _egf_variance(fit, y)
    if "d" in keys:  # long-memory variance-recursion FI (FIGARCH / FIGJR / FITGARCH / FIAPARCH)
        return _fi_variance(fit, y)
    if {"phi1", "beta1", "gamma1"} <= keys:  # APARCH family (GJR / TGARCH / APARCH)
        return _aparch_family_variance(fit, y)
    params = np.array(  # plain GARCH
        [fit.params["mu"], fit.params["omega"], fit.params["alpha"], fit.params["beta"]],
        dtype=np.float64,
    )
    return garch_recursion(params, y)


def _mean_residuals(fit: GarchFit, y: FloatArray) -> FloatArray:
    r"""Mean-residuals :math:`r_t = y_t - \mu_t` of the fit's ARMA/FARIMA-in-mean recursion.

    ARMA(1,1) when there is no fractional mean order ``D``; FARIMA(1,d,1) (demean, fractionally
    difference by ``D``, then the ARMA recursion) when ``D`` is present (spec-notes §36).
    """
    mu = fit.params["mu"]
    ar1 = fit.params.get("ar1", 0.0)
    ma1 = fit.params.get("ma1", 0.0)
    if "D" in fit.params:
        return farima_mean_residuals(y, mu, ar1, ma1, fit.params["D"])
    return arma_mean_residuals(y, mu, ar1, ma1)


def _variance_subfit(fit: GarchFit) -> GarchFit:
    """A variance-only view of a dual-mean fit: the mean block stripped and ``mu`` set to 0.

    The mean-residuals already have the conditional mean removed, so the variance forecast runs on
    them as a zero-mean series. The variance parameters, ``model`` tag and fitted conditional
    volatility (the residual recursion's :math:`\\sigma`) are kept.
    """
    var_params = {k: v for k, v in fit.params.items() if k not in _MEAN_KEYS}
    var_params["mu"] = 0.0
    return GarchFit(
        cond_dist=fit.cond_dist,
        params=var_params,
        std_errors={},
        loglikelihood=fit.loglikelihood,
        aic=fit.aic,
        bic=fit.bic,
        conditional_volatility=fit.conditional_volatility,
        n_obs=fit.n_obs,
        converged=fit.converged,
        model=fit.model,
    )


def _mean_forecast(fit: GarchFit, returns: FloatArray, n_ahead: int) -> FloatArray:
    r"""Multi-step conditional-mean forecast of an ARMA/FARIMA-in-mean fit (spec-notes §36).

    Future mean-residuals have expectation 0, so the forecast iterates the mean recursion with the
    realized residual only at :math:`h = 1`. ARMA(1,1):
    :math:`\hat\mu_{n+1} = \mu + \beta_1(y_n - \mu) + \alpha_1 r_n`, then
    :math:`\hat\mu_{n+h} = \mu + \beta_1(\hat\mu_{n+h-1} - \mu)`. FARIMA(1,d,1) forecasts the
    fractionally-differenced :math:`x = (1-B)^D(y-\mu)` by its ARMA iterate and inverts
    :math:`(1-B)^D` (the same operator the FI variance uses, at ``+D``; ``d_mean`` is independent of
    any variance-side ``d``).
    """
    y = np.asarray(returns, dtype=np.float64)
    n = fit.n_obs
    mu = fit.params["mu"]
    ar1 = fit.params.get("ar1", 0.0)
    ma1 = fit.params.get("ma1", 0.0)
    r = _mean_residuals(fit, y[:n])
    mfc = np.empty(n_ahead, dtype=np.float64)
    if "D" not in fit.params:  # ARMA(1,1)-in-mean
        mfc[0] = mu + ar1 * (y[n - 1] - mu) + ma1 * r[-1]
        for h in range(1, n_ahead):
            mfc[h] = mu + ar1 * (mfc[h - 1] - mu)
        return mfc
    # FARIMA(1,d,1)-in-mean: forecast x = (1-B)^D (y-mu) by its ARMA iterate, then invert (1-B)^D.
    d_mean = fit.params["D"]
    w = y[:n] - mu
    x = np.convolve(fracdiff_coeffs(d_mean, n), w)[:n]  # x_t = (1-B)^D (y_t - mu)
    xf = np.empty(n_ahead, dtype=np.float64)
    xf[0] = ar1 * x[-1] + ma1 * r[-1]
    for h in range(1, n_ahead):
        xf[h] = ar1 * xf[h - 1]
    b = fracdiff_coeffs(d_mean, n + n_ahead)  # invert: (y-mu) = x - sum_{k>=1} b_k (y-mu)_{.-k}
    w_ext = np.concatenate([w, np.zeros(n_ahead, dtype=np.float64)])
    for h in range(n_ahead):
        tail = float(np.sum(b[1 : n + h + 1] * w_ext[n + h - 1 :: -1]))
        w_ext[n + h] = xf[h] - tail
    return np.asarray(mu + w_ext[n:], dtype=np.float64)


def predict_roll(
    fit: GarchFit, returns: FloatArray, *, n_test: int | None = None
) -> RollingForecast:
    r"""No-refit rolling one-step forecast of :math:`\hat\sigma_t` / :math:`\hat\mu_t`.

    Freezes the parameters at the training ``fit`` and continues the fitted variance recursion past
    the training window over the realized test returns — equivalently, runs the model's recursion
    with the training parameters over the **full** ``returns`` series and keeps the last ``n_test``
    values (the train-then-continue seed is machine-exact for any reasonable training seed; see the
    module docstring). Supports **GARCH** (via
    :func:`~quantica.timeseries.fegarch.garch_recursion`) and the **APARCH family** — GJR / TGARCH /
    APARCH, detected by the ``phi1, beta1, gamma1`` parameters and dispatched by the fitted power
    :math:`\delta` (spec-notes §33). The mean is the constant fitted :math:`\mu` (constant-mean
    models), so :math:`\hat\mu_t` is a constant series.

    Parameters
    ----------
    fit : GarchFit
        A fit on the **training** window — GARCH (``mu, omega, alpha, beta``) or an APARCH-family
        model (``mu, omega, phi1, beta1, gamma1, delta``); ``n_obs`` is the training length.
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

    if _is_dual_mean(fit):
        # ARMA/FARIMA-in-mean (spec-notes §36): the variance recursion runs on the mean-RESIDUALS
        # (no sigma->mean feedback), and the rolling conditional mean at each test point is y - r.
        resid = _mean_residuals(fit, y)
        sigma2 = _conditional_variance(_variance_subfit(fit), resid)
        sigma = np.sqrt(sigma2[y.size - n_test :])
        cmeans = (y - resid)[y.size - n_test :]
    else:  # constant mean
        sigma2 = _conditional_variance(fit, y)
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
    one-step rolling). The short-memory families below, plus the **long-memory FI family** (FIGARCH,
    FIEGARCH, FIGJR, FITGARCH, FIAPARCH, detected by the fractional ``d`` and dispatched to the §34
    weight-propagation helpers -- weights extended with the horizon, observed sample pre-sample 0),
    are supported, reproducing the fixtures to machine precision (FS-skew / free-``delta`` cases to
    the kappa-quadrature floor):

    * **Variance-recursion GARCH(1,1)** (params ``mu, omega, alpha, beta``): the standard iterated
      forecast :math:`\hat\sigma_{n+h}^2 = \omega + (\alpha+\beta)\hat\sigma_{n+h-1}^2` for
      :math:`h \ge 2` (the future :math:`\varepsilon_{n+h-1}^2 \to` its expectation
      :math:`\hat\sigma_{n+h-1}^2`); :math:`h = 1` uses the realized :math:`\varepsilon_n`. It is
      **unbiased** and converges to the unconditional variance :math:`\omega/(1-\alpha-\beta)`.
    * **APARCH family (GJR / TGARCH / APARCH)** (params ``mu, omega, phi1, beta1, gamma1, delta``):
      the iterated :math:`\sigma^\delta` forecast :math:`m_h = \omega + (\phi_1\kappa + \beta_1)
      m_{h-1}` for :math:`h \ge 2`, with the asymmetry-corrected news moment
      :math:`\kappa = \operatorname{E}[(|\eta| - \gamma_1\eta)^\delta]` (full quadrature — fEGarch
      integrates the whole kernel, spec-notes §33); :math:`h = 1` uses the realized news
      :math:`(|\varepsilon_n| - \gamma_1\varepsilon_n)^\delta`. Report
      :math:`\hat\sigma = (m_h)^{1/\delta}`, the **Jensen-biased plug-in** fEGarch ships (for
      :math:`\delta \ne 2`), reproduced **not** bias-corrected. GJR (:math:`\delta=2`) and TGARCH
      (:math:`\delta=1`) are the fixed-power specializations of this path.
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
    if _is_dual_mean(fit):
        # ARMA/FARIMA-in-mean (spec-notes §36): compose the mean forecast with the variance forecast
        # of the mean-residuals (no sigma->mean feedback; future residuals enter the variance as 0).
        resid = _mean_residuals(fit, np.asarray(returns, dtype=np.float64))
        sigma = predict(_variance_subfit(fit), resid, n_ahead=n_ahead).sigma
        cmeans = _mean_forecast(fit, returns, n_ahead)
        return MultiStepForecast(sigma=sigma, cmeans=cmeans, n_ahead=n_ahead)
    mu = fit.params["mu"]
    sigma_n = float(fit.conditional_volatility[-1])
    keys = set(fit.params)

    if {"omega_sig", "phi1"} <= keys:
        # EGF family (spec-notes §35): EGARCH / MEGARCH / MLog-GARCH / Log-GARCH + FI variants. All
        # iterate ln sigma^2 with future news = 0; short memory AR(1), FI the §34 weight
        # convolution. The model tag + param keys select the link g and centering moment.
        sigma = _egf_multistep(fit, returns, n_ahead)
    elif "d" in keys:
        # Long-memory variance-recursion FI (FIGARCH / FIGJR / FITGARCH / FIAPARCH, spec-notes §34):
        # the ARCH(inf) iterate reusing the §33 kappa; weights extended by h, pre-sample 0.
        sigma = _fi_power_multistep(fit, returns, n_ahead)
    elif {"omega", "alpha", "beta"} <= keys and not (keys & {"gamma", "delta", "d"}):
        # Variance-recursion GARCH(1,1): iterate sigma^2 with future eps^2 -> its expectation.
        omega, alpha, beta = fit.params["omega"], fit.params["alpha"], fit.params["beta"]
        eps_n = y[-1] - mu
        variance = np.empty(n_ahead, dtype=np.float64)
        variance[0] = omega + alpha * eps_n**2 + beta * sigma_n**2
        for h in range(1, n_ahead):
            variance[h] = omega + (alpha + beta) * variance[h - 1]
        sigma = np.sqrt(variance)
    elif {"omega", "phi1", "beta1", "gamma1"} <= keys:
        # APARCH family (GJR delta=2, TGARCH delta=1, APARCH free delta): iterate the sigma^delta
        # recursion m_h = omega + (phi1*kappa + beta1) m_{h-1} for h >= 2, with the asymmetry-
        # corrected news moment kappa = E[(|z| - gamma1 z)^delta] (§33). The h=1 term uses
        # the realized news (|eps_n| - gamma1 eps_n)^delta; report sigma = (m_h)^(1/delta), the
        # Jensen-biased plug-in fEGarch ships (reproduced, not corrected -- as the EGARCH bias).
        omega, phi1 = fit.params["omega"], fit.params["phi1"]
        beta1, gamma1, delta = fit.params["beta1"], fit.params["gamma1"], fit.params["delta"]
        dist, shape = _distribution_for_fit(fit)
        kappa = dist.news_moment(delta, gamma1, shape)
        rho = phi1 * kappa + beta1  # reduced persistence
        eps_n = y[-1] - mu
        m = np.empty(n_ahead, dtype=np.float64)
        m[0] = omega + phi1 * (abs(eps_n) - gamma1 * eps_n) ** delta + beta1 * sigma_n**delta
        for h in range(1, n_ahead):
            m[h] = omega + rho * m[h - 1]
        sigma = m ** (1.0 / delta)
    else:
        raise NotImplementedError(
            f"multi-step predict is not built for a fit with params {sorted(keys)}; "
            "only constant-mean GARCH(1,1), EGARCH(1,1), the APARCH family (GJR/TGARCH/APARCH) and "
            "the long-memory FI family (FIGARCH/FIEGARCH/FIGJR/FITGARCH/FIAPARCH) are built"
        )

    cmeans = np.full(n_ahead, mu, dtype=np.float64)
    return MultiStepForecast(
        sigma=np.asarray(sigma, dtype=np.float64), cmeans=cmeans, n_ahead=n_ahead
    )
