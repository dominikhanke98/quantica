"""Validation of barrier options (numerical-validation skill).

- :func:`barrier_price` is the continuous-monitoring closed form
  (Reiner--Rubinstein), the analytic anchor; in + out = vanilla holds exactly.
- The Monte Carlo engine monitors *discretely*, which biases it versus the
  continuous contract — a knock-out high, a knock-in low — because between-step
  crossings are missed. The bias is named, its direction reasoned, and shown to
  shrink as monitoring frequency rises.
- The **Brownian-bridge** correction analytically restores the missed-crossing
  probability, cutting the bias at a fixed step count (the differentiator).
- In-out parity (knock-in + knock-out = vanilla) is an exact structural check.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BarrierMonteCarloEngine,
    BarrierOption,
    BarrierType,
    BlackScholesProcess,
    EuropeanOption,
    OptionType,
    barrier_price,
)

MARKET = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.25)
DOWN_H = 90.0
UP_H = 115.0
N_PATHS = 200_000


def _barrier(barrier_type: BarrierType, level: float, n: int = 100) -> BarrierOption:
    return BarrierOption(
        strike=100.0,
        expiry=1.0,
        option_type=OptionType.CALL,
        barrier=level,
        barrier_type=barrier_type,
        n_monitoring_dates=n,
    )


def _closed_form(barrier_type: BarrierType, level: float) -> float:
    return barrier_price(
        MARKET.spot,
        100.0,
        level,
        MARKET.rate,
        MARKET.div,
        MARKET.vol,
        1.0,
        barrier_type,
        OptionType.CALL,
    )


def _engine(seed: int, *, bridge: bool = False, n: int = N_PATHS) -> BarrierMonteCarloEngine:
    return BarrierMonteCarloEngine(
        n, rng=np.random.default_rng(seed), antithetic=True, brownian_bridge=bridge
    )


# --------------------------------------------------------------------------- #
# Closed form: analytic in-out parity and vanilla bounds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "out_type, in_type, level",
    [
        (BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN, DOWN_H),
        (BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN, UP_H),
    ],
)
def test_closed_form_in_out_parity(
    out_type: BarrierType, in_type: BarrierType, level: float
) -> None:
    vanilla = AnalyticEuropeanEngine().calculate(
        EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL), MARKET
    )
    assert _closed_form(out_type, level) + _closed_form(in_type, level) == pytest.approx(
        vanilla, abs=1e-10
    )


def test_knock_out_cheaper_than_vanilla() -> None:
    vanilla = AnalyticEuropeanEngine().calculate(
        EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL), MARKET
    )
    assert 0.0 < _closed_form(BarrierType.DOWN_AND_OUT, DOWN_H) < vanilla


# --------------------------------------------------------------------------- #
# Discrete-monitoring bias: named, its direction reasoned, and shrinking with n
# --------------------------------------------------------------------------- #


def test_discrete_knock_out_biased_high_and_shrinks() -> None:
    # Discrete monitoring misses between-step crossings -> fewer knock-outs
    # detected -> a knock-out is biased HIGH vs the continuous contract, and the
    # bias shrinks as monitoring frequency rises.
    cf = _closed_form(BarrierType.DOWN_AND_OUT, DOWN_H)
    coarse = _engine(0, n=N_PATHS).estimate(
        _barrier(BarrierType.DOWN_AND_OUT, DOWN_H, n=25), MARKET
    )
    fine = _engine(0, n=N_PATHS).estimate(_barrier(BarrierType.DOWN_AND_OUT, DOWN_H, n=250), MARKET)
    assert coarse.price > cf  # biased high
    assert fine.price > cf  # still high...
    assert (fine.price - cf) < (coarse.price - cf)  # ...but the bias shrank


def test_discrete_knock_in_biased_low() -> None:
    # The mirror image: a knock-in activates too few paths, so it is biased LOW.
    cf = _closed_form(BarrierType.DOWN_AND_IN, DOWN_H)
    discrete = _engine(1, n=N_PATHS).estimate(
        _barrier(BarrierType.DOWN_AND_IN, DOWN_H, n=25), MARKET
    )
    assert discrete.price < cf


# --------------------------------------------------------------------------- #
# Brownian bridge: recovers the continuous price and beats discrete at fixed n
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "barrier_type, level",
    [
        (BarrierType.DOWN_AND_OUT, DOWN_H),
        (BarrierType.UP_AND_OUT, UP_H),
        (BarrierType.DOWN_AND_IN, DOWN_H),
    ],
)
def test_bridge_recovers_continuous_closed_form(barrier_type: BarrierType, level: float) -> None:
    cf = _closed_form(barrier_type, level)
    result = _engine(2, bridge=True, n=N_PATHS).estimate(
        _barrier(barrier_type, level, n=100), MARKET
    )
    assert abs(result.price - cf) <= 3.0 * result.std_error


def test_bridge_beats_discrete_at_fixed_step_count() -> None:
    cf = _closed_form(BarrierType.DOWN_AND_OUT, DOWN_H)
    option = _barrier(BarrierType.DOWN_AND_OUT, DOWN_H, n=50)
    discrete = _engine(3, bridge=False).estimate(option, MARKET).price
    bridge = _engine(3, bridge=True).estimate(option, MARKET).price
    assert abs(bridge - cf) < abs(discrete - cf)


# --------------------------------------------------------------------------- #
# In-out parity (exact structural check on the same paths)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bridge", [False, True])
def test_monte_carlo_in_out_parity(bridge: bool) -> None:
    # On identical paths, knock-in + knock-out = vanilla exactly (each path either
    # knocks or does not; the payoffs partition). Seed all three the same way.
    knock_out = _engine(4, bridge=bridge, n=100_000).calculate(
        _barrier(BarrierType.DOWN_AND_OUT, DOWN_H, n=50), MARKET
    )
    knock_in = _engine(4, bridge=bridge, n=100_000).calculate(
        _barrier(BarrierType.DOWN_AND_IN, DOWN_H, n=50), MARKET
    )
    # A barrier that can never be hit reproduces the vanilla payoff on these paths.
    vanilla = _engine(4, bridge=bridge, n=100_000).calculate(
        _barrier(BarrierType.DOWN_AND_OUT, 1e-9, n=50), MARKET
    )
    assert knock_out + knock_in == pytest.approx(vanilla, abs=1e-10)


# --------------------------------------------------------------------------- #
# Discipline / wiring
# --------------------------------------------------------------------------- #


def test_same_seed_same_result() -> None:
    option = _barrier(BarrierType.DOWN_AND_OUT, DOWN_H, n=50)
    assert _engine(5, n=10_000).estimate(option, MARKET) == _engine(5, n=10_000).estimate(
        option, MARKET
    )


def test_rejects_non_barrier_option() -> None:
    european = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    with pytest.raises(TypeError, match="BarrierOption"):
        _engine(0, n=1000).estimate(european, MARKET)  # type: ignore[arg-type]


def test_barrier_instrument_validation() -> None:
    with pytest.raises(ValueError, match="barrier must be positive"):
        BarrierOption(
            strike=100.0,
            expiry=1.0,
            option_type=OptionType.CALL,
            barrier=0.0,
            barrier_type=BarrierType.DOWN_AND_OUT,
        )
    with pytest.raises(ValueError, match="n_monitoring_dates must be at least 1"):
        BarrierOption(
            strike=100.0,
            expiry=1.0,
            option_type=OptionType.CALL,
            barrier=90.0,
            barrier_type=BarrierType.DOWN_AND_OUT,
            n_monitoring_dates=0,
        )


def test_barrier_is_european_exercise() -> None:
    from quantica.core.types import ExerciseStyle

    assert _barrier(BarrierType.DOWN_AND_OUT, DOWN_H).exercise is ExerciseStyle.EUROPEAN
