r"""Implied volatility of a European option.

Given an observed option *price*, recover the Black--Scholes volatility
:math:`\sigma` that reproduces it. The Black--Scholes price is strictly
increasing in :math:`\sigma` on :math:`(0, \infty)`, so the inverse is unique
whenever the price lies strictly inside the no-arbitrage band

.. math::

    \max(\omega (S e^{-qT} - K e^{-rT}), 0) \;<\; V \;<\;
    \begin{cases} S e^{-qT} & \text{call} \\ K e^{-rT} & \text{put.}\end{cases}

The lower bound is the :math:`\sigma\to 0` price (discounted intrinsic on the
forward); the upper bound is the :math:`\sigma\to\infty` limit. Outside the band
no implied volatility exists and a :class:`ValueError` is raised.

Method
------
A safeguarded hybrid: a Newton fast path driven by the analytic vega, wrapped in
a bracket that it may never leave. Newton gives quadratic convergence when vega
is well behaved; the moment a step would leave the bracket or vega becomes tiny
(deep in/out of the money), we fall back to Brent's method
(:func:`scipy.optimize.brentq`) on the maintained bracket, which is guaranteed
to converge. Pricing and vega are delegated to
:class:`~quantica.pricing.engines.analytic.AnalyticEuropeanEngine`, so this
solver inherits that engine's validation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from scipy.optimize import brentq

from quantica.core.types import OptionType
from quantica.pricing.engines.analytic import AnalyticEuropeanEngine
from quantica.pricing.processes import BlackScholesProcess

if TYPE_CHECKING:
    from quantica.pricing.instruments import EuropeanOption
    from quantica.pricing.processes import Market

__all__ = ["implied_volatility"]

# Numerical parameters (named, not magic — CLAUDE.md §6).
_VOL_FLOOR = 1e-9  # lower volatility bracket (effectively zero)
_VOL_CEIL_START = 5.0  # initial upper bracket: 500% vol
_VOL_CEIL_MAX = 50.0  # cap when doubling out the upper bracket: 5000% vol
_MIN_VEGA = 1e-12  # below this vega the Newton step is unreliable -> bracket
_BOUND_ATOL = 1e-12  # absolute tolerance on the no-arbitrage bound checks
_DEFAULT_PRICE_TOL = 1e-10  # convergence tolerance on the price residual
_DEFAULT_MAX_ITER = 100
_BRENT_XTOL = 1e-12  # absolute tolerance on sigma for the Brent fallback

# One shared, stateless engine instance for repricing during the solve.
_ENGINE = AnalyticEuropeanEngine()


def implied_volatility(
    price: float,
    option: EuropeanOption,
    market: Market,
    *,
    tol: float = _DEFAULT_PRICE_TOL,
    max_iter: int = _DEFAULT_MAX_ITER,
    initial_guess: float | None = None,
) -> float:
    r"""Return the Black--Scholes implied volatility reproducing ``price``.

    Parameters
    ----------
    price : float
        Observed option price to invert.
    option : EuropeanOption
        The contract (strike, expiry, call/put).
    market : Market
        The market state (spot, rate, dividend). Volatility is precisely the
        unknown being solved for, so — unlike a pricing call — no vol is passed.
    tol : float, optional
        Convergence tolerance on the absolute price residual.
    max_iter : int, optional
        Maximum iterations for both the Newton phase and the Brent fallback.
    initial_guess : float, optional
        Starting volatility for Newton. Defaults to the Brenner--Subrahmanyam
        at-the-money approximation :math:`\sigma_0 \approx \sqrt{2\pi/T}\,V/S`.

    Returns
    -------
    float
        The implied volatility :math:`\sigma > 0` (or ``0.0`` when the price
        sits on the discounted-intrinsic lower bound).

    Raises
    ------
    ValueError
        If ``option.expiry <= 0``, or ``price`` lies outside the no-arbitrage
        band (below discounted intrinsic, or at/above the upper bound), so that
        no finite positive implied volatility exists.
    """
    S = market.spot
    K = option.strike
    r = market.rate
    q = market.div
    T = option.expiry

    if T <= 0.0:
        raise ValueError("implied volatility is undefined for non-positive time to expiry")

    omega = option.option_type.sign
    disc_spot = S * math.exp(-q * T)
    disc_strike = K * math.exp(-r * T)
    lower = max(omega * (disc_spot - disc_strike), 0.0)
    upper = disc_spot if option.option_type is OptionType.CALL else disc_strike

    # No-arbitrage band checks with clear diagnostics.
    if price < lower - _BOUND_ATOL:
        raise ValueError(
            f"price {price:.10g} is below the no-arbitrage lower bound {lower:.10g} "
            "(discounted intrinsic value); no implied volatility exists"
        )
    if price > upper + _BOUND_ATOL:
        raise ValueError(
            f"price {price:.10g} exceeds the no-arbitrage upper bound {upper:.10g}; "
            "no implied volatility exists"
        )
    if price <= lower + _BOUND_ATOL:
        # On the lower bound: the sigma -> 0 limit.
        return 0.0
    if price >= upper - _BOUND_ATOL:
        raise ValueError(
            f"price {price:.10g} is at the no-arbitrage upper bound {upper:.10g}; "
            "implied volatility is unbounded"
        )

    def price_at(sigma: float) -> float:
        return _ENGINE.calculate(option, BlackScholesProcess.from_market(market, sigma))

    def vega_at(sigma: float) -> float:
        return _ENGINE.greeks(option, BlackScholesProcess.from_market(market, sigma)).vega

    # Establish a bracket [lo, hi] with price_at(lo) < price < price_at(hi).
    # price_at(_VOL_FLOOR) ~ lower < price holds by the checks above.
    lo, hi = _VOL_FLOOR, _VOL_CEIL_START
    while price_at(hi) < price:
        hi *= 2.0
        if hi > _VOL_CEIL_MAX:
            raise ValueError(
                f"could not bracket implied volatility below {_VOL_CEIL_MAX:.0f}; "
                f"price {price:.10g} is too close to the upper bound {upper:.10g}"
            )

    # --- Newton fast path, safeguarded to stay inside the bracket ---------- #
    sigma = initial_guess if initial_guess is not None else _initial_guess(price, disc_spot, T)
    sigma = min(max(sigma, lo), hi)
    for _ in range(max_iter):
        diff = price_at(sigma) - price
        if abs(diff) <= tol:
            return sigma
        # Tighten the bracket using the sign of the residual (price is monotone
        # increasing in sigma), so the fallback keeps a valid bracket.
        if diff > 0.0:
            hi = sigma
        else:
            lo = sigma
        vega = vega_at(sigma)
        if vega < _MIN_VEGA:
            break  # gradient too flat: hand over to Brent
        step = diff / vega
        nxt = sigma - step
        if not (lo < nxt < hi):
            break  # Newton left the bracket: hand over to Brent
        sigma = nxt

    # --- Robust bracketed fallback ---------------------------------------- #
    root = brentq(
        lambda s: price_at(s) - price,
        lo,
        hi,
        xtol=_BRENT_XTOL,
        maxiter=max_iter,
    )
    return float(root)


def _initial_guess(price: float, scale: float, expiry: float) -> float:
    r"""Brenner--Subrahmanyam ATM approximation :math:`\sqrt{2\pi/T}\,V/S`.

    Cheap and dimensionally correct; the safeguarded solver corrects it from
    there, so its only job is to land Newton in a sensible neighbourhood.

    References
    ----------
    Brenner, M. & Subrahmanyam, M. (1988). "A Simple Formula to Compute the
    Implied Standard Deviation", *Financial Analysts Journal*.
    """
    return math.sqrt(2.0 * math.pi / expiry) * price / scale
