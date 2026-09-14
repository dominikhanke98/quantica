r"""ARMA-in-mean dual fits: a non-constant (ARMA) conditional mean jointly with a GARCH variance.

.. note::

    **Clean-room (CLAUDE.md §12).** Independent reimplementation from the published mathematics
    (WP 2026-04 §2.2.2, Eqs. 19-22 — the FARIMA-in-mean structure, here the ``D = 0`` ARMA case),
    **never** the ``fEGarch`` source. Validated against committed ``fEGarch`` OUTPUT fixtures.

The first **dual mean+variance** model of Phase 5: the conditional mean is an ARMA(P, Q) recursion
(P, Q in {0, 1}) instead of the constant :math:`\mu`, estimated **jointly** with the GARCH(1,1)
variance in one likelihood. WP 2026-04 Eq. 22 (``D = 0``) gives the mean as

.. math::

    y_t - \mu = \sum_{i=1}^{P} \beta_i (y_{t-i} - \mu) + \sum_{j=1}^{Q} \alpha_j r_{t-j} + r_t,

i.e. the conditional mean and the mean-residual are

.. math::

    \mu_t = \mu + \sum_{i=1}^{P} \beta_i (y_{t-i} - \mu) + \sum_{j=1}^{Q} \alpha_j r_{t-j},
    \qquad r_t = y_t - \mu_t,

with ``ar1`` :math:`= \beta_1` (the AR coefficient on the lagged demeaned level) and ``ma1``
:math:`= \alpha_1` (the MA coefficient on the lagged residual). The **mean-residuals** :math:`r_t`
then drive the **existing** GARCH(1,1) variance recursion
(:func:`~quantica.timeseries.fegarch.garch._garch_variance`) in place of the constant-mean residual
:math:`(y - \mu)` — the variance recursion is unchanged, only its input residual series changes.

**Mean pre-sample (the conditional convention).** :math:`(y_{t-i} - \mu)` and :math:`r_{t-j}` are
``0`` for :math:`t \le 0`, so :math:`\mu_1 = \mu` and :math:`r_1 = y_1 - \mu`. This conditional
initialisation reproduces ``fEGarch``'s :math:`r_t` **machine-identically for every** :math:`t`
past the variance seed (verified: the reconstruction pins to ``6.9e-18`` at the fixture's own
:math:`\sigma_0^2`).

**The** :math:`\sigma_0^2` **seed — a documented bounded limit.** The variance recursion is seeded
with :math:`\sigma_0^2 = \omega + (\alpha+\beta)\operatorname{Var}(r,\,\text{ddof}=1)`, the
reduction-consistent published-math analog of the constant-mean :math:`\operatorname{Var}(y)` seed.
``fEGarch``'s exact dual-case seed is an **internal preliminary-residual quantity** (a non-standard
effective denominator :math:`\approx n-1.4`) that no published-math formula reproduces and that §12
forbids reverse-engineering from source — so, as with the FIAPARCH bounded-limit seed (spec-notes
§19), this analog is used and the residual **decaying seed transient** (:math:`\sim 10^{-6}` at
:math:`t=0`, decaying at :math:`\approx\beta` per step to machine-zero by :math:`t\approx 200`) is a
documented bound. At ARMA(0, 0) the mean-residuals are exactly :math:`y-\mu`, so the seed collapses
to the pure-GARCH :math:`\operatorname{Var}(y)` seed and the fit reduces **exactly** to
:func:`~quantica.timeseries.fegarch.fit_garch`.

**Weak identification on Gaussian data.** On a pure-GARCH series (no mean dynamics) ``ar1``/``ma1``
sit on a flat likelihood ridge and estimate near-zero; ``fEGarch``'s Hessian fails there (no
standard errors), and for the full ARMA(1,1) the AR and MA roots nearly cancel, so our optimizer
lands a comparable-or-higher likelihood at a different ``(ar1, ma1)`` than ``fEGarch`` (a
dominate-or-tie, validated by loglik + the variance block, not by a mean-parameter match). The mean
estimation is validated to recover genuine dynamics by a known-truth simulation with identified
``ar1``/``ma1``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.timeseries.fegarch.distributions import get_distribution
from quantica.timeseries.fegarch.garch import GarchFit, _garch_fit_from_result, _garch_variance
from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = ["arma_mean_residuals", "fit_arma_garch"]

_SHAPE_OPTIONS: dict[str, object] = {"maxiter": 20000, "maxfev": 20000, "fatol": 1e-10}


def arma_mean_residuals(
    returns: FloatArray, mu: float, ar1: float = 0.0, ma1: float = 0.0
) -> FloatArray:
    r"""Mean-residuals :math:`r_t = y_t - \mu_t` of an ARMA(1,1)-in-mean recursion (WP Eq. 22, D=0).

    :math:`\mu_t = \mu + \text{ar1}\,(y_{t-1}-\mu) + \text{ma1}\,r_{t-1}`, with the conditional
    pre-sample :math:`(y_0-\mu) = r_0 = 0` (so :math:`\mu_1 = \mu`). ARMA(1,0) sets ``ma1 = 0``;
    ARMA(0,1) sets ``ar1 = 0``; ARMA(0,0) (both ``0``) returns exactly :math:`y - \mu`.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series :math:`y_t`.
    mu : float
        The unconditional mean :math:`\mu`.
    ar1 : float, optional
        The AR(1) coefficient :math:`\beta_1` (default ``0``).
    ma1 : float, optional
        The MA(1) coefficient :math:`\alpha_1` (default ``0``).

    Returns
    -------
    ndarray, shape (T,)
        The mean-residuals :math:`r_t`.
    """
    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    r = np.empty(n, dtype=np.float64)
    for t in range(n):
        ydev_prev = (y[t - 1] - mu) if t >= 1 else 0.0  # (y_{t-1} - mu), 0 pre-sample
        r_prev = r[t - 1] if t >= 1 else 0.0  # r_{t-1}, 0 pre-sample
        r[t] = y[t] - (mu + ar1 * ydev_prev + ma1 * r_prev)
    return r


def fit_arma_garch(
    returns: FloatArray, *, mean_orders: tuple[int, int] = (1, 1), cond_dist: str = "norm"
) -> GarchFit:
    r"""Fit ARMA(P,Q)-in-mean with a GARCH(1,1) variance jointly by QMLE (P, Q in {0, 1}).

    The mean block :math:`\{\mu, \text{ar1}, \text{ma1}\}` and the variance block
    :math:`\{\omega, \alpha, \beta\}` are estimated **together** in one likelihood: each candidate
    vector gives :math:`\mu_t \to r_t = y_t-\mu_t \to \sigma_t` (the existing GARCH recursion on
    :math:`r_t`) :math:`\to \ell(y_t, \mu_t, \sigma_t)`.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The return series.
    mean_orders : tuple of (int, int), optional
        The ARMA mean orders ``(P, Q)`` with ``P, Q in {0, 1}`` (default ``(1, 1)``): ``(1, 0)``
        AR-only, ``(0, 1)`` MA-only, ``(1, 1)`` the full ARMA(1,1) mean. ``(0, 0)`` reduces to a
        constant mean (use :func:`~quantica.timeseries.fegarch.fit_garch`).
    cond_dist : str, optional
        Conditional-distribution code (default ``"norm"``, the validated one). ``"ald"``/``"sald"``
        need discrete-``P`` profiling and are not supported for the dual fit yet.

    Returns
    -------
    GarchFit
        Estimates ``mu``, ``ar1``/``ma1`` (as present), ``omega, alpha, beta`` (fEGarch's
        ``phi1``/``beta1``), the log-likelihood, per-observation AIC/BIC, and the conditional-SD
        series.

    Raises
    ------
    ValueError
        If ``mean_orders`` are outside ``{0, 1}``.
    NotImplementedError
        If ``cond_dist`` is ``"ald"`` or ``"sald"`` (discrete-P profiling, deferred).
    """
    p_order, q_order = mean_orders
    if p_order not in (0, 1) or q_order not in (0, 1):
        raise ValueError("mean_orders (P, Q) must each be 0 or 1 for the ARMA(1,1) dual fit")
    if cond_dist in ("ald", "sald"):
        raise NotImplementedError(
            "ALD/sALD dual fits need discrete-P profiling (not yet supported)"
        )

    y = np.asarray(returns, dtype=np.float64)
    n = y.size
    scale = float(np.std(y))  # scale-equivariant fit; conditions the small-magnitude omega
    scaled = y / scale
    distribution = get_distribution(cond_dist)
    n_mean = 1 + p_order + q_order  # mu (+ ar1) (+ ma1)

    # Parameter layout: mu, [ar1], [ma1], omega, alpha, beta.  Names use quantica's variance names
    # (alpha/beta = fEGarch's phi1/beta1); ar1/ma1 are the ARMA mean coefficients.
    var_names = ("mu", *(("ar1",) if p_order else ()), *(("ma1",) if q_order else ()),
                 "omega", "alpha", "beta")  # fmt: skip
    variance = float(np.var(scaled, ddof=1))
    var_start = (float(np.mean(scaled)), *(0.0,) * (p_order + q_order), variance * 0.05, 0.05, 0.90)
    var_bounds = ((-10.0, 10.0), *(((-0.9999, 0.9999),) * (p_order + q_order)),
                  (1e-8, 1e6), (0.0, 0.9999), (0.0, 0.9999))  # fmt: skip
    # Unscale: mu ~ scale, ar1/ma1 scale-invariant, omega ~ scale^2, alpha/beta invariant.
    factors = np.array([scale, *(1.0,) * (p_order + q_order), scale**2, 1.0, 1.0])

    def mean_recursion(var_params: FloatArray, series: FloatArray) -> FloatArray:
        mu = float(var_params[0])
        ar1 = float(var_params[1]) if p_order else 0.0
        ma1 = float(var_params[1 + p_order]) if q_order else 0.0
        return arma_mean_residuals(series, mu, ar1, ma1)

    def variance_recursion(var_params: FloatArray, resid: FloatArray) -> FloatArray:
        omega = float(var_params[n_mean])
        alpha = float(var_params[n_mean + 1])
        beta = float(var_params[n_mean + 2])
        r = np.asarray(resid, dtype=np.float64)
        # Bounded-limit seed: Var(r, ddof=1) -- the reduction-consistent analog of the pure-GARCH
        # Var(y) seed (see the module docstring). Reduces to Var(y) exactly at ARMA(0,0).
        return _garch_variance(r, omega, alpha, beta, float(np.var(r, ddof=1)))

    # A continuous shape parameter (std df, ged shape) sits on a flat ridge -> Nelder-Mead; the
    # norm path (no shape parameter) keeps L-BFGS-B, matching fit_garch's convention.
    if distribution.param_names:
        method: str = "Nelder-Mead"
        options: dict[str, object] | None = _SHAPE_OPTIONS
    else:
        method, options = "L-BFGS-B", None

    result = quasi_max_likelihood(
        scaled,
        variance_recursion,
        distribution,
        var_start=var_start,
        var_bounds=var_bounds,
        var_names=var_names,
        mean=True,
        method=method,
        options=options,
        mean_recursion=mean_recursion,
    )
    return _garch_fit_from_result(
        result, cond_dist, scale, n, k=result.params.size, factors=factors
    )
