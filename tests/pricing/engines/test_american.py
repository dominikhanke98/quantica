"""Validation of American exercise (numerical-validation skill, no analytic reference).

American options have no closed form, so the validation strategy shifts from
"agree with Black--Scholes" to:

1. **Cross-method** — the binomial tree and the Crank--Nicolson PDE (LCP via
   PSOR) agree on the American price to their combined discretisation tolerance.
2. **Exact structural theorems** — the strongest checks available:
   * *No-dividend American call = European call.* Early exercise of a call on a
     non-dividend-paying stock is never optimal, so on a given engine the two
     prices are identical to machine/solver precision (the tree never takes
     intrinsic; the PDE obstacle never binds).
   * *Early-exercise premium >= 0.* American >= European on the same engine, for
     both puts and calls, exactly (no discretisation slack, since it is the same
     lattice/grid).
3. **Qualitative sanity** — a strictly positive American-put premium, and
   monotonicity of the put in spot and volatility.

QuantLib American benchmarks live in ``test_benchmark_quantlib.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.pricing import (
    AmericanOption,
    AnalyticEuropeanEngine,
    BinomialEngine,
    BlackScholesProcess,
    EuropeanOption,
    FiniteDifferenceEngine,
    MonteCarloEngine,
    OptionType,
)

# Grids and their justified cross-method tolerance (no analytic anchor here).
TREE_STEPS = 2000  # CRR, O(1/N)
PDE_STEPS = 300  # Crank--Nicolson + PSOR, ~O(h^2)
CROSS_TOL = 3e-3  # combined tree/PDE discretisation bound (measured <= ~2e-3)


def _tree() -> BinomialEngine:
    return BinomialEngine(steps=TREE_STEPS)


def _pde() -> FiniteDifferenceEngine:
    return FiniteDifferenceEngine(space_steps=PDE_STEPS, time_steps=PDE_STEPS)


def _pair(strike: float, expiry: float, kind: OptionType) -> tuple[AmericanOption, EuropeanOption]:
    return (
        AmericanOption(strike=strike, expiry=expiry, option_type=kind),
        EuropeanOption(strike=strike, expiry=expiry, option_type=kind),
    )


# --------------------------------------------------------------------------- #
# 1. Cross-method: tree vs PDE
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("strike", [90.0, 100.0, 110.0])
def test_tree_and_pde_agree_on_american_price(kind: OptionType, strike: float) -> None:
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.04, vol=0.25)
    option = AmericanOption(strike=strike, expiry=1.0, option_type=kind)
    tree = _tree().calculate(option, proc)
    pde = _pde().calculate(option, proc)
    assert abs(tree - pde) < CROSS_TOL


# --------------------------------------------------------------------------- #
# 2. Exact structural theorems
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("expiry", [1.0, 2.0])
def test_no_dividend_american_call_equals_european_call(strike: float, expiry: float) -> None:
    # Highlighted validation: with q = 0 early exercise of a call is never
    # optimal, so American == European on the SAME engine to tight tolerance.
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=0.0, vol=0.2)
    american, european = _pair(strike, expiry, OptionType.CALL)

    tree = BinomialEngine(steps=TREE_STEPS)
    assert tree.calculate(american, proc) == pytest.approx(
        tree.calculate(european, proc), abs=1e-10
    )  # the tree never takes intrinsic -> bit-for-bit equality

    pde = _pde()
    assert pde.calculate(american, proc) == pytest.approx(
        pde.calculate(european, proc), abs=1e-8
    )  # the PSOR obstacle never binds -> equal to solver precision


@pytest.mark.parametrize("engine_name", ["tree", "pde"])
@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("div", [0.0, 0.05])
def test_american_at_least_european_same_engine(
    engine_name: str, kind: OptionType, div: float
) -> None:
    # Early-exercise premium >= 0, checked on the same engine/grid so there is no
    # discretisation slack to hide behind.
    proc = BlackScholesProcess(spot=100.0, rate=0.05, div=div, vol=0.2)
    american, european = _pair(100.0, 1.0, kind)
    engine = BinomialEngine(steps=TREE_STEPS) if engine_name == "tree" else _pde()
    premium = engine.calculate(american, proc) - engine.calculate(european, proc)
    assert premium >= -1e-10


# --------------------------------------------------------------------------- #
# 3. Qualitative sanity
# --------------------------------------------------------------------------- #


def test_american_put_has_positive_premium() -> None:
    # An ATM American put carries a real early-exercise premium (from the rate),
    # even with no dividends. Anchor the magnitude against QuantLib-validated tree.
    proc = BlackScholesProcess(spot=100.0, rate=0.08, div=0.0, vol=0.3)
    american, european = _pair(100.0, 1.0, OptionType.PUT)
    tree = BinomialEngine(steps=TREE_STEPS)
    premium = tree.calculate(american, proc) - tree.calculate(european, proc)
    assert premium > 1e-2


def test_american_put_decreasing_in_spot() -> None:
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    engine = BinomialEngine(steps=500)
    prices = [
        engine.calculate(option, BlackScholesProcess(spot=s, rate=0.05, div=0.02, vol=0.2))
        for s in np.linspace(70.0, 130.0, 13)
    ]
    assert np.all(np.diff(prices) < 0.0)


def test_american_put_increasing_in_vol() -> None:
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    engine = BinomialEngine(steps=500)
    prices = [
        engine.calculate(option, BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=v))
        for v in np.linspace(0.1, 0.6, 11)
    ]
    assert np.all(np.diff(prices) > 0.0)


# --------------------------------------------------------------------------- #
# Engines without early-exercise support reject American options
# --------------------------------------------------------------------------- #


def test_analytic_engine_rejects_american() -> None:
    proc = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    engine = AnalyticEuropeanEngine()
    with pytest.raises(ValueError, match="European exercise only"):
        engine.calculate(option, proc)
    with pytest.raises(ValueError, match="European exercise only"):
        engine.greeks(option, proc)


def test_monte_carlo_engine_rejects_american() -> None:
    proc = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    engine = MonteCarloEngine(10_000, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="European exercise only"):
        engine.estimate(option, proc)
