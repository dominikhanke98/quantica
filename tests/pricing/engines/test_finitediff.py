"""Validation of the Crank--Nicolson PDE engine (numerical-validation skill).

Checks applied:

1. Cross-method convergence (§3) — the PDE price converges to the analytic
   Black--Scholes price as the grid refines: a tight bound at a fine grid and —
   stronger — the second-order O(h^2) rate of Crank--Nicolson via a
   log(error)-vs-log(steps) slope of ~ -2 (error quarters as steps double).
2. Analytical sanity — put--call parity on PDE prices (up to discretisation
   error), and the sigma->0 / T->0 discounted-intrinsic limit.

The QuantLib FD benchmark (§4) lives in ``test_benchmark_quantlib.py`` behind
the ``benchmark`` marker.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BlackScholesProcess,
    EuropeanOption,
    FiniteDifferenceEngine,
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
def test_converges_to_analytic_on_fine_grid(kind: OptionType) -> None:
    option = _option(100.0, 1.0, kind)
    reference = ANALYTIC.calculate(option, PROC)
    price = FiniteDifferenceEngine(space_steps=500, time_steps=500).calculate(option, PROC)
    assert abs(price - reference) < 1e-3


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_second_order_convergence_slope(kind: OptionType) -> None:
    # Refining space and time together, Crank--Nicolson error is O(h^2), so a
    # log-log fit of error vs grid steps should have slope ~ -2.
    option = _option(100.0, 1.0, kind)
    reference = ANALYTIC.calculate(option, PROC)
    steps = np.array([20, 40, 80, 160, 320])
    errors = np.array(
        [
            abs(
                FiniteDifferenceEngine(space_steps=m, time_steps=m).calculate(option, PROC)
                - reference
            )
            for m in steps
        ]
    )
    slope = np.polyfit(np.log(steps), np.log(errors), 1)[0]
    assert -2.3 < slope < -1.7
    # And the error should shrink by ~4x each time the grid doubles.
    ratios = errors[:-1] / errors[1:]
    assert np.all(ratios > 3.4)


# --------------------------------------------------------------------------- #
# 2. Analytical sanity
# --------------------------------------------------------------------------- #


def test_put_call_parity_up_to_discretisation() -> None:
    # Unlike the tree, parity is not exact on the PDE grid (clamped Dirichlet
    # boundaries + log-grid), but holds to the O(h^2) discretisation error.
    engine = FiniteDifferenceEngine(space_steps=400, time_steps=400)
    K, T = 95.0, 1.5
    call = engine.calculate(_option(K, T, OptionType.CALL), PROC)
    put = engine.calculate(_option(K, T, OptionType.PUT), PROC)
    rhs = PROC.spot * math.exp(-PROC.div * T) - K * math.exp(-PROC.rate * T)
    assert (call - put) == pytest.approx(rhs, abs=1e-3)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("spot", [80.0, 100.0, 130.0])
def test_zero_vol_limit_is_discounted_intrinsic(kind: OptionType, spot: float) -> None:
    proc = BlackScholesProcess(spot=spot, rate=0.05, div=0.02, vol=0.0)
    K, T = 100.0, 1.0
    expected = max(kind.sign * (spot * math.exp(-proc.div * T) - K * math.exp(-proc.rate * T)), 0.0)
    assert FiniteDifferenceEngine().calculate(_option(K, T, kind), proc) == pytest.approx(
        expected, abs=1e-12
    )


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_zero_expiry_limit_is_intrinsic(kind: OptionType) -> None:
    proc = BlackScholesProcess(spot=120.0, rate=0.05, vol=0.2)
    expected = max(kind.sign * (120.0 - 100.0), 0.0)
    assert FiniteDifferenceEngine().calculate(_option(100.0, 0.0, kind), proc) == pytest.approx(
        expected, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# Engine wiring
# --------------------------------------------------------------------------- #


def test_satisfies_pricing_protocol_only() -> None:
    engine = FiniteDifferenceEngine()
    assert isinstance(engine, PricingEngine)
    assert not isinstance(engine, GreeksEngine)


def test_odd_space_steps_round_down_to_even() -> None:
    # space_steps is rounded down to even so ln(spot) is exactly a node: an odd
    # count gives the identical price to the next even-down count.
    option = _option(100.0, 1.0, OptionType.CALL)
    odd = FiniteDifferenceEngine(space_steps=201, time_steps=200).calculate(option, PROC)
    even = FiniteDifferenceEngine(space_steps=200, time_steps=200).calculate(option, PROC)
    assert odd == even


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"space_steps": 1}, "space_steps must be at least 2"),
        ({"time_steps": 0}, "time_steps must be at least 1"),
        ({"num_std": 0.0}, "num_std must be positive"),
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        FiniteDifferenceEngine(**kwargs)


def test_prices_through_option_npv() -> None:
    option = _option(100.0, 1.0, OptionType.CALL)
    engine = FiniteDifferenceEngine(space_steps=300, time_steps=300)
    option.set_engine(engine)
    assert option.npv(PROC) == pytest.approx(engine.calculate(option, PROC))
