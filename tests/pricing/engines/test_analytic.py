"""Validation of the Black--Scholes analytic engine (numerical-validation skill).

Checks applied (skill checklist):

1. Analytical sanity — known textbook values, put--call parity, arbitrage
   bounds, monotonicity, and the vol->0 / T->0 / deep-ITM limits.
2. Greeks — every analytic Greek vs a central bump-and-reval finite difference.

Not applied here, by design:

* Cross-method convergence (skill §3) needs a *second* engine; it is added with
  the binomial engine in the next step, along with the README convergence table
  (skill §6).
* QuantLib benchmark (skill §4) lives in ``test_benchmark_quantlib.py`` behind
  the ``benchmark`` marker.
* Monte Carlo discipline (skill §5) is N/A — this engine is deterministic.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BlackScholesProcess,
    EuropeanOption,
    OptionType,
)
from quantica.pricing.engines import GreeksEngine, PricingEngine

ENGINE = AnalyticEuropeanEngine()


def _option(strike: float, expiry: float, kind: OptionType) -> EuropeanOption:
    return EuropeanOption(strike=strike, expiry=expiry, option_type=kind).set_engine(ENGINE)


# --------------------------------------------------------------------------- #
# 1. Analytical sanity checks
# --------------------------------------------------------------------------- #
# Reference values computed independently from the closed form (scipy norm),
# not from this engine, so the check is not circular. Assert to ~1e-9.


def test_known_atm_value() -> None:
    # Textbook ATM case: S=K=100, r=5%, q=0, sigma=20%, T=1 (Hull; skill table).
    proc = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    assert _option(100.0, 1.0, OptionType.CALL).npv(proc) == pytest.approx(
        10.450583572185565, abs=1e-9
    )
    assert _option(100.0, 1.0, OptionType.PUT).npv(proc) == pytest.approx(
        5.573526022256971, abs=1e-9
    )


def test_known_hull_value() -> None:
    # Hull, Options Futures & Other Derivatives: S=42, K=40, r=10%, sigma=20%, T=0.5.
    proc = BlackScholesProcess(spot=42.0, rate=0.10, vol=0.2)
    assert _option(40.0, 0.5, OptionType.CALL).npv(proc) == pytest.approx(
        4.759422392871532, abs=1e-9
    )
    assert _option(40.0, 0.5, OptionType.PUT).npv(proc) == pytest.approx(
        0.8085993729000922, abs=1e-9
    )


def test_put_call_parity() -> None:
    # C - P == S e^{-qT} - K e^{-rT} to machine precision (with dividends).
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.2)
    K, T = 95.0, 1.5
    lhs = _option(K, T, OptionType.CALL).npv(proc) - _option(K, T, OptionType.PUT).npv(proc)
    rhs = proc.spot * math.exp(-proc.div * T) - K * math.exp(-proc.rate * T)
    assert lhs == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_arbitrage_bounds(kind: OptionType) -> None:
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.25)
    T = 1.0
    disc_spot = proc.spot * math.exp(-proc.div * T)
    for K in (60.0, 100.0, 140.0):
        price = _option(K, T, kind).npv(proc)
        disc_strike = K * math.exp(-proc.rate * T)
        assert price >= -1e-12  # non-negative
        if kind is OptionType.CALL:
            lower = max(disc_spot - disc_strike, 0.0)
            assert lower - 1e-12 <= price <= disc_spot + 1e-12
        else:
            lower = max(disc_strike - disc_spot, 0.0)
            assert lower - 1e-12 <= price <= disc_strike + 1e-12


def test_call_monotone_increasing_in_spot() -> None:
    spots = np.linspace(50.0, 150.0, 21)
    prices = [
        _option(100.0, 1.0, OptionType.CALL).npv(BlackScholesProcess(spot=s, rate=0.05, vol=0.2))
        for s in spots
    ]
    assert np.all(np.diff(prices) > 0.0)


def test_put_monotone_decreasing_in_spot() -> None:
    spots = np.linspace(50.0, 150.0, 21)
    prices = [
        _option(100.0, 1.0, OptionType.PUT).npv(BlackScholesProcess(spot=s, rate=0.05, vol=0.2))
        for s in spots
    ]
    assert np.all(np.diff(prices) < 0.0)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_monotone_increasing_in_vol(kind: OptionType) -> None:
    # Vega is positive for both calls and puts.
    vols = np.linspace(0.05, 0.8, 21)
    prices = [
        _option(100.0, 1.0, kind).npv(BlackScholesProcess(spot=100.0, rate=0.05, vol=v))
        for v in vols
    ]
    assert np.all(np.diff(prices) > 0.0)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("spot", [80.0, 100.0, 130.0])
def test_zero_vol_limit_is_discounted_intrinsic(kind: OptionType, spot: float) -> None:
    # sigma -> 0: underlying is deterministic; price is discounted intrinsic on
    # the forward, max(omega (S e^{-qT} - K e^{-rT}), 0).
    proc = BlackScholesProcess(spot=spot, rate=0.05, div=0.02, vol=0.0)
    K, T = 100.0, 1.0
    expected = max(kind.sign * (spot * math.exp(-proc.div * T) - K * math.exp(-proc.rate * T)), 0.0)
    assert _option(K, T, kind).npv(proc) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("spot", [80.0, 100.0, 130.0])
def test_zero_expiry_limit_is_intrinsic(kind: OptionType, spot: float) -> None:
    proc = BlackScholesProcess(spot=spot, rate=0.05, vol=0.2)
    expected = max(kind.sign * (spot - 100.0), 0.0)
    assert _option(100.0, 0.0, kind).npv(proc) == pytest.approx(expected, abs=1e-12)


def test_deep_itm_call_approaches_forward_minus_discounted_strike() -> None:
    # Deep ITM call -> S e^{-qT} - K e^{-rT} (probability of finishing ITM -> 1).
    proc = BlackScholesProcess(spot=1000.0, rate=0.05, div=0.02, vol=0.2)
    K, T = 100.0, 1.0
    price = _option(K, T, OptionType.CALL).npv(proc)
    expected = proc.spot * math.exp(-proc.div * T) - K * math.exp(-proc.rate * T)
    assert price == pytest.approx(expected, rel=1e-9)


# --------------------------------------------------------------------------- #
# 2. Greeks validation (central bump-and-reval)
# --------------------------------------------------------------------------- #

# Central-difference step sizes. First-order Greeks use h ~ 1e-4 (truncation
# error O(h^2) ~ 1e-8, well inside the tolerances); gamma is a second difference
# and uses a larger h to keep round-off from dominating.
_H = 1e-4
_H_GAMMA = 1e-2


def _central(f, x: float, h: float) -> float:  # type: ignore[no-untyped-def]
    return (f(x + h) - f(x - h)) / (2.0 * h)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_delta_matches_finite_difference(kind: OptionType) -> None:
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.2)
    opt = _option(105.0, 1.0, kind)
    fd = _central(lambda s: opt.npv(proc.with_spot(s)), proc.spot, _H)
    assert opt.greeks(proc).delta == pytest.approx(fd, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_gamma_matches_second_difference(kind: OptionType) -> None:
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.2)
    opt = _option(105.0, 1.0, kind)
    h = _H_GAMMA
    up = opt.npv(proc.with_spot(proc.spot + h))
    mid = opt.npv(proc)
    dn = opt.npv(proc.with_spot(proc.spot - h))
    fd = (up - 2.0 * mid + dn) / (h * h)
    assert opt.greeks(proc).gamma == pytest.approx(fd, rel=1e-5, abs=1e-8)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_vega_matches_finite_difference(kind: OptionType) -> None:
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.2)
    opt = _option(105.0, 1.0, kind)
    fd = _central(lambda v: opt.npv(proc.with_vol(v)), proc.vol, _H)
    assert opt.greeks(proc).vega == pytest.approx(fd, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_rho_matches_finite_difference(kind: OptionType) -> None:
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.2)
    opt = _option(105.0, 1.0, kind)
    fd = _central(lambda r: opt.npv(proc.with_rate(r)), proc.rate, _H)
    assert opt.greeks(proc).rho == pytest.approx(fd, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_theta_matches_finite_difference(kind: OptionType) -> None:
    # theta = dV/dt (calendar time) = -dV/dT. Bump the option's expiry.
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.2)
    opt = _option(105.0, 1.0, kind)

    def price_at_expiry(expiry: float) -> float:
        bumped = dataclasses.replace(opt, expiry=expiry).set_engine(ENGINE)
        return bumped.npv(proc)

    fd_dT = _central(price_at_expiry, opt.expiry, _H)
    assert opt.greeks(proc).theta == pytest.approx(-fd_dT, rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_greeks_undefined_in_degenerate_limits(kind: OptionType) -> None:
    opt = _option(100.0, 1.0, kind)
    with pytest.raises(ValueError, match="undefined"):
        opt.greeks(BlackScholesProcess(spot=100.0, rate=0.05, vol=0.0))  # sigma = 0
    zero_expiry = _option(100.0, 0.0, kind)
    with pytest.raises(ValueError, match="undefined"):
        zero_expiry.greeks(BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2))  # T = 0


# --------------------------------------------------------------------------- #
# Engine wiring / capability protocol
# --------------------------------------------------------------------------- #


def test_engine_satisfies_both_protocols() -> None:
    assert isinstance(ENGINE, PricingEngine)
    assert isinstance(ENGINE, GreeksEngine)


def test_option_greeks_delegate_to_engine() -> None:
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.2)
    opt = _option(100.0, 1.0, OptionType.CALL)
    assert opt.greeks(proc) == ENGINE.greeks(opt, proc)


def test_greeks_without_engine_raises() -> None:
    opt = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    with pytest.raises(RuntimeError, match="no pricing engine attached"):
        opt.greeks(BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2))


def test_greeks_with_price_only_engine_raises() -> None:
    class PriceOnly:
        def calculate(self, instrument: EuropeanOption, process: BlackScholesProcess) -> float:
            return 0.0

    opt = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    opt.set_engine(PriceOnly())
    with pytest.raises(RuntimeError, match="does not support Greeks"):
        opt.greeks(BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2))
