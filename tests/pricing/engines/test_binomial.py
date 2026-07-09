"""Validation of the CRR binomial engine (numerical-validation skill).

Checks applied:

1. Cross-method convergence (§3) — the CRR price converges to the analytic
   Black--Scholes price as N grows: a tight bound at high N, and the stronger
   evidence of first-order O(1/N) convergence via a log(error)-vs-log(N) slope
   along an even-N subsequence (avoiding the even/odd CRR sawtooth), plus a
   demonstration that averaging consecutive N tames that oscillation.
2. Analytical sanity — put--call parity on tree prices, and the sigma->0 /
   T->0 discounted-intrinsic limit.

The QuantLib CRR benchmark (§4) lives in ``test_benchmark_quantlib.py`` behind
the ``benchmark`` marker.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BinomialEngine,
    BlackScholesProcess,
    EuropeanOption,
    OptionType,
)
from quantica.pricing.engines import GreeksEngine, PricingEngine

ANALYTIC = AnalyticEuropeanEngine()
PROC = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.2)


def _option(strike: float, expiry: float, kind: OptionType) -> EuropeanOption:
    return EuropeanOption(strike=strike, expiry=expiry, option_type=kind)


# --------------------------------------------------------------------------- #
# 1. Cross-method convergence
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_converges_to_analytic_at_high_steps(kind: OptionType) -> None:
    option = _option(100.0, 1.0, kind)
    reference = ANALYTIC.calculate(option, PROC)
    price = BinomialEngine(steps=5000).calculate(option, PROC)
    assert abs(price - reference) < 1e-3


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_first_order_convergence_slope(kind: OptionType) -> None:
    # Along an even-N geometric subsequence the CRR error is a clean O(1/N),
    # so a log-log fit of error vs N should have slope ~ -1.
    option = _option(100.0, 1.0, kind)
    reference = ANALYTIC.calculate(option, PROC)
    steps = np.array([50, 100, 200, 400, 800, 1600])
    errors = np.array(
        [abs(BinomialEngine(steps=int(n)).calculate(option, PROC) - reference) for n in steps]
    )
    slope = np.polyfit(np.log(steps), np.log(errors), 1)[0]
    assert -1.15 < slope < -0.85
    # And the error should shrink roughly by half each time N doubles.
    ratios = errors[:-1] / errors[1:]
    assert np.all(ratios > 1.7)


def test_averaging_consecutive_steps_tames_oscillation() -> None:
    # The even/odd sawtooth: averaging price(N) and price(N+1) is far more
    # accurate than either single tree at the same order of work.
    option = _option(100.0, 1.0, OptionType.CALL)
    reference = ANALYTIC.calculate(option, PROC)
    n = 400
    single = abs(BinomialEngine(steps=n).calculate(option, PROC) - reference)
    averaged = abs(
        0.5
        * (
            BinomialEngine(steps=n).calculate(option, PROC)
            + BinomialEngine(steps=n + 1).calculate(option, PROC)
        )
        - reference
    )
    assert averaged < single / 5.0


# --------------------------------------------------------------------------- #
# 2. Analytical sanity
# --------------------------------------------------------------------------- #


def test_put_call_parity_on_tree_prices() -> None:
    # Parity holds exactly on the lattice (at any N): the discounted expectation
    # of the linear payoff S_T - K is S e^{-qT} - K e^{-rT}.
    engine = BinomialEngine(steps=500)
    K, T = 95.0, 1.5
    call = engine.calculate(_option(K, T, OptionType.CALL), PROC)
    put = engine.calculate(_option(K, T, OptionType.PUT), PROC)
    rhs = PROC.spot * math.exp(-PROC.div * T) - K * math.exp(-PROC.rate * T)
    assert (call - put) == pytest.approx(rhs, abs=1e-10)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("spot", [80.0, 100.0, 130.0])
def test_zero_vol_limit_is_discounted_intrinsic(kind: OptionType, spot: float) -> None:
    proc = BlackScholesProcess(spot=spot, rate=0.05, div=0.02, vol=0.0)
    K, T = 100.0, 1.0
    expected = max(kind.sign * (spot * math.exp(-proc.div * T) - K * math.exp(-proc.rate * T)), 0.0)
    assert BinomialEngine(steps=64).calculate(_option(K, T, kind), proc) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_zero_expiry_limit_is_intrinsic(kind: OptionType) -> None:
    proc = BlackScholesProcess(spot=120.0, rate=0.05, vol=0.2)
    expected = max(kind.sign * (120.0 - 100.0), 0.0)
    assert BinomialEngine(steps=64).calculate(_option(100.0, 0.0, kind), proc) == pytest.approx(
        expected, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# Engine wiring
# --------------------------------------------------------------------------- #


def test_satisfies_pricing_protocol_only() -> None:
    engine = BinomialEngine()
    assert isinstance(engine, PricingEngine)
    assert not isinstance(engine, GreeksEngine)  # price-only engine


def test_invalid_steps_raises() -> None:
    with pytest.raises(ValueError, match="steps must be a positive integer"):
        BinomialEngine(steps=0)


def test_prices_through_option_npv() -> None:
    option = _option(100.0, 1.0, OptionType.CALL)
    engine = BinomialEngine(steps=500)
    option.set_engine(engine)
    assert option.npv(PROC) == pytest.approx(engine.calculate(option, PROC))
