"""Tests for the vanilla option contracts, payoff, and exercise style."""

from __future__ import annotations

import numpy as np
import pytest
from quantica.core.types import ExerciseStyle, OptionType
from quantica.pricing.instruments import AmericanOption, EuropeanOption, VanillaOption
from quantica.pricing.processes import BlackScholesProcess


def test_call_payoff_scalar() -> None:
    call = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    assert call.payoff(120.0) == pytest.approx(20.0)  # in the money
    assert call.payoff(80.0) == pytest.approx(0.0)  # out of the money
    assert call.payoff(100.0) == pytest.approx(0.0)  # at the money


def test_put_payoff_scalar() -> None:
    put = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    assert put.payoff(80.0) == pytest.approx(20.0)  # in the money
    assert put.payoff(120.0) == pytest.approx(0.0)  # out of the money
    assert put.payoff(100.0) == pytest.approx(0.0)  # at the money


def test_payoff_scalar_returns_python_float() -> None:
    call = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    result = call.payoff(120.0)
    assert isinstance(result, float)


def test_payoff_vectorised() -> None:
    call = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    spots = np.array([80.0, 100.0, 120.0])
    result = call.payoff(spots)
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result, np.array([0.0, 0.0, 20.0]))


def test_put_call_payoff_parity_identity() -> None:
    # call_payoff - put_payoff == S_T - K at every spot.
    call = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    put = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    spots = np.linspace(50.0, 150.0, 11)
    np.testing.assert_allclose(call.payoff(spots) - put.payoff(spots), spots - 100.0)


def test_invalid_strike_rejected() -> None:
    with pytest.raises(ValueError, match="strike must be positive"):
        EuropeanOption(strike=0.0, expiry=1.0, option_type=OptionType.CALL)
    with pytest.raises(ValueError, match="strike must be positive"):
        EuropeanOption(strike=-10.0, expiry=1.0, option_type=OptionType.CALL)


def test_negative_expiry_rejected() -> None:
    with pytest.raises(ValueError, match="expiry must be non-negative"):
        EuropeanOption(strike=100.0, expiry=-1.0, option_type=OptionType.CALL)


def test_zero_expiry_allowed() -> None:
    opt = EuropeanOption(strike=100.0, expiry=0.0, option_type=OptionType.CALL)
    assert opt.payoff(120.0) == pytest.approx(20.0)


def test_npv_without_engine_raises() -> None:
    call = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    process = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    with pytest.raises(RuntimeError, match="no pricing engine attached"):
        call.npv(process)


def test_set_engine_delegates_and_chains() -> None:
    # A stub engine standing in for a real one (added in later steps) verifies
    # the delegation contract in isolation.
    class StubEngine:
        def calculate(self, instrument: EuropeanOption, process: BlackScholesProcess) -> float:
            return 42.0

    call = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    process = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    returned = call.set_engine(StubEngine())
    assert returned is call  # chainable
    assert call.npv(process) == pytest.approx(42.0)


# --------------------------------------------------------------------------- #
# Exercise style (European vs American)
# --------------------------------------------------------------------------- #


def test_exercise_style_of_each_subclass() -> None:
    assert EuropeanOption(100.0, 1.0, OptionType.CALL).exercise is ExerciseStyle.EUROPEAN
    assert AmericanOption(100.0, 1.0, OptionType.PUT).exercise is ExerciseStyle.AMERICAN


def test_base_vanilla_option_has_no_exercise_style() -> None:
    base = VanillaOption(100.0, 1.0, OptionType.CALL)
    with pytest.raises(NotImplementedError, match="EuropeanOption or AmericanOption"):
        _ = base.exercise


def test_american_shares_the_intrinsic_payoff() -> None:
    # Payoff (immediate-exercise value) is identical to the European terminal one.
    spots = np.array([80.0, 100.0, 120.0])
    american = AmericanOption(100.0, 1.0, OptionType.PUT)
    european = EuropeanOption(100.0, 1.0, OptionType.PUT)
    np.testing.assert_allclose(american.payoff(spots), european.payoff(spots))
    np.testing.assert_allclose(american.payoff(spots), np.array([20.0, 0.0, 0.0]))


def test_set_engine_preserves_concrete_type() -> None:
    # set_engine returns Self, so chaining keeps the American type (not the base).
    class StubEngine:
        def calculate(self, instrument: VanillaOption, process: BlackScholesProcess) -> float:
            return 1.0

    american = AmericanOption(100.0, 1.0, OptionType.PUT).set_engine(StubEngine())
    assert isinstance(american, AmericanOption)


def test_same_terms_different_style_are_unequal() -> None:
    assert AmericanOption(100.0, 1.0, OptionType.PUT) != EuropeanOption(100.0, 1.0, OptionType.PUT)


def test_american_validates_strike_and_expiry() -> None:
    with pytest.raises(ValueError, match="strike must be positive"):
        AmericanOption(0.0, 1.0, OptionType.PUT)
    with pytest.raises(ValueError, match="expiry must be non-negative"):
        AmericanOption(100.0, -1.0, OptionType.PUT)


def test_engine_excluded_from_equality() -> None:
    # Two contracts with the same terms are equal regardless of engine state.
    a = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    b = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    assert a == b
    a.set_engine(type("E", (), {"calculate": lambda *_: 1.0})())
    assert a == b
