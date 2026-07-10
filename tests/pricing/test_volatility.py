"""Validation of the implied-volatility solver (numerical-validation skill).

Checks applied:

1. Analytical sanity — price->IV->price round-trip to tight tolerance across a
   grid; recovery of a known volatility; the no-arbitrage no-solution cases.
2. Robustness — deep ITM/OTM, very short and very long maturities, calls and
   puts, and convergence from a deliberately poor initial guess.
3. QuantLib benchmark (skill §4, behind the ``benchmark`` marker) — agreement
   with QuantLib's own implied-volatility solver.

Cross-method convergence (§3) and Monte Carlo discipline (§5) are N/A: implied
vol is a deterministic inverse of the analytic price.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BlackScholesProcess,
    EuropeanOption,
    Market,
    OptionType,
    implied_volatility,
)

ENGINE = AnalyticEuropeanEngine()

# The solver takes just a Market (spot, rate, div) — volatility is the unknown.
MARKET = Market(spot=100.0, rate=0.05, div=0.02)


def _price(option: EuropeanOption, sigma: float, market: Market = MARKET) -> float:
    return ENGINE.calculate(option, BlackScholesProcess.from_market(market, sigma))


# --------------------------------------------------------------------------- #
# 1. Round-trip and known-vol recovery
# --------------------------------------------------------------------------- #


# Below this vega, price carries essentially no information about volatility
# (deep ITM/OTM at low vol): the price is numerically pinned to a no-arbitrage
# bound, so vol is not identifiable and only the reprice identity is meaningful.
_IDENTIFIABLE_VEGA = 1e-2


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("sigma", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("strike", [50.0, 80.0, 100.0, 120.0, 175.0])
@pytest.mark.parametrize("expiry", [0.1, 1.0, 5.0])
def test_price_iv_price_round_trip(
    kind: OptionType, sigma: float, strike: float, expiry: float
) -> None:
    option = EuropeanOption(strike=strike, expiry=expiry, option_type=kind)
    target = _price(option, sigma)
    iv = implied_volatility(target, option, MARKET)

    # The fundamental identity: whatever IV we return must reprice to the target.
    assert _price(option, iv) == pytest.approx(target, abs=1e-9)

    # Recovering the volatility itself is only well-posed where vega is not
    # vanishing; elsewhere many vols give the same (bound-pinned) price.
    vega = ENGINE.greeks(option, BlackScholesProcess.from_market(MARKET, sigma)).vega
    if vega >= _IDENTIFIABLE_VEGA:
        assert iv == pytest.approx(sigma, rel=1e-6, abs=1e-7)


def test_recovers_known_volatility() -> None:
    option = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    target = _price(option, 0.2)  # 10.4506-ish family with dividends
    assert implied_volatility(target, option, MARKET) == pytest.approx(0.2, rel=1e-9)


# --------------------------------------------------------------------------- #
# 2. Edge cases / robustness
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("strike", [20.0, 300.0])  # deep ITM / deep OTM
def test_deep_in_and_out_of_the_money(kind: OptionType, strike: float) -> None:
    option = EuropeanOption(strike=strike, expiry=1.0, option_type=kind)
    target = _price(option, 0.35)
    if target < 1e-8:
        pytest.skip("price below solver resolution")
    assert implied_volatility(target, option, MARKET) == pytest.approx(0.35, rel=1e-6)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("expiry", [1.0 / 365.0, 1.0 / 52.0, 30.0])  # ~1 day, 1 week, 30y
def test_short_and_long_maturities(kind: OptionType, expiry: float) -> None:
    option = EuropeanOption(strike=100.0, expiry=expiry, option_type=kind)
    target = _price(option, 0.25)
    assert implied_volatility(target, option, MARKET) == pytest.approx(0.25, rel=1e-6)


def test_converges_from_poor_initial_guess() -> None:
    # A wildly wrong starting point must still converge (safeguarded Newton /
    # Brent fallback), not diverge.
    option = EuropeanOption(strike=100.0, expiry=2.0, option_type=OptionType.PUT)
    target = _price(option, 0.4)
    assert implied_volatility(target, option, MARKET, initial_guess=1e-6) == pytest.approx(
        0.4, rel=1e-6
    )
    assert implied_volatility(target, option, MARKET, initial_guess=40.0) == pytest.approx(
        0.4, rel=1e-6
    )


def test_price_on_lower_bound_returns_zero_vol() -> None:
    option = EuropeanOption(strike=90.0, expiry=1.0, option_type=OptionType.CALL)
    lower = max(MARKET.spot * math.exp(-MARKET.div) - option.strike * math.exp(-MARKET.rate), 0.0)
    assert implied_volatility(lower, option, MARKET) == 0.0


# --------------------------------------------------------------------------- #
# No-solution handling
# --------------------------------------------------------------------------- #


def test_price_below_intrinsic_raises() -> None:
    option = EuropeanOption(strike=90.0, expiry=1.0, option_type=OptionType.CALL)
    lower = MARKET.spot * math.exp(-MARKET.div) - option.strike * math.exp(-MARKET.rate)
    with pytest.raises(ValueError, match="below the no-arbitrage lower bound"):
        implied_volatility(lower - 1.0, option, MARKET)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_price_above_upper_bound_raises(kind: OptionType) -> None:
    option = EuropeanOption(strike=100.0, expiry=1.0, option_type=kind)
    upper = (
        MARKET.spot * math.exp(-MARKET.div)
        if kind is OptionType.CALL
        else (option.strike * math.exp(-MARKET.rate))
    )
    with pytest.raises(ValueError, match="upper bound"):
        implied_volatility(upper + 1.0, option, MARKET)


def test_at_upper_bound_raises_unbounded() -> None:
    option = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    upper = MARKET.spot * math.exp(-MARKET.div)
    with pytest.raises(ValueError, match="unbounded"):
        implied_volatility(upper, option, MARKET)


def test_unbracketable_price_raises() -> None:
    # A 1-day ATM call priced at 90 is unreachable even at 5000% vol (which
    # tops out near 81 here), yet sits below the upper bound (~100): the solver
    # cannot bracket it and says so rather than looping forever.
    option = EuropeanOption(strike=100.0, expiry=1.0 / 365.0, option_type=OptionType.CALL)
    with pytest.raises(ValueError, match="could not bracket"):
        implied_volatility(90.0, option, MARKET)


def test_non_positive_expiry_raises() -> None:
    option = EuropeanOption(strike=100.0, expiry=0.0, option_type=OptionType.CALL)
    with pytest.raises(ValueError, match="non-positive time to expiry"):
        implied_volatility(5.0, option, MARKET)


def test_tolerance_is_respected() -> None:
    option = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    target = _price(option, 0.2)
    iv = implied_volatility(target, option, MARKET, tol=1e-12)
    assert abs(_price(option, iv) - target) <= 1e-10


# --------------------------------------------------------------------------- #
# 3. QuantLib benchmark
# --------------------------------------------------------------------------- #

ql = pytest.importorskip("QuantLib")


@pytest.mark.benchmark
@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("sigma", [0.1, 0.2, 0.5])
def test_implied_vol_matches_quantlib(kind: OptionType, sigma: float) -> None:
    spot, rate, div = 100.0, 0.05, 0.02
    strike, expiry = 105.0, 1.0
    market = Market(spot=spot, rate=rate, div=div)
    option = EuropeanOption(strike=strike, expiry=expiry, option_type=kind)
    target = ENGINE.calculate(option, BlackScholesProcess.from_market(market, sigma))
    ours = implied_volatility(target, option, market)

    today = ql.Date(15, 6, 2020)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    spot_h = ql.QuoteHandle(ql.SimpleQuote(spot))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, rate, day_count))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, div, day_count))
    vol_h = ql.SimpleQuote(0.20)  # seed; QuantLib solves from here
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, ql.QuoteHandle(vol_h), day_count)
    )
    ql_process = ql.BlackScholesMertonProcess(spot_h, q_ts, r_ts, vol_ts)
    ql_kind = ql.Option.Call if kind is OptionType.CALL else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(ql_kind, strike)
    exercise = ql.EuropeanExercise(today + ql.Period(round(365 * expiry), ql.Days))
    ql_option = ql.VanillaOption(payoff, exercise)
    theirs = ql_option.impliedVolatility(target, ql_process, 1e-10, 200, 1e-7, 10.0)

    assert np.isclose(ours, theirs, rtol=1e-6)
    assert ours == pytest.approx(sigma, rel=1e-6)
