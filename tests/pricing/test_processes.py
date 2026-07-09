"""Tests for the BlackScholesProcess market state."""

from __future__ import annotations

import math

import pytest
from quantica.pricing.processes import BlackScholesProcess


def test_construction_and_fields() -> None:
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2, div=0.01)
    assert p.spot == 100.0
    assert p.rate == 0.05
    assert p.vol == 0.2
    assert p.div == 0.01


def test_div_defaults_to_zero() -> None:
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    assert p.div == 0.0


def test_is_frozen() -> None:
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    with pytest.raises(AttributeError):
        p.spot = 110.0  # type: ignore[misc]


def test_negative_rate_and_div_allowed() -> None:
    # Negative rates are a real feature of modern fixed-income markets.
    p = BlackScholesProcess(spot=100.0, rate=-0.005, vol=0.2, div=-0.01)
    assert p.rate == -0.005
    assert p.div == -0.01


def test_zero_vol_allowed() -> None:
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.0)
    assert p.vol == 0.0


def test_invalid_spot_rejected() -> None:
    with pytest.raises(ValueError, match="spot must be positive"):
        BlackScholesProcess(spot=0.0, rate=0.05, vol=0.2)
    with pytest.raises(ValueError, match="spot must be positive"):
        BlackScholesProcess(spot=-100.0, rate=0.05, vol=0.2)


def test_negative_vol_rejected() -> None:
    with pytest.raises(ValueError, match="vol must be non-negative"):
        BlackScholesProcess(spot=100.0, rate=0.05, vol=-0.2)


def test_discount_factor() -> None:
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    assert p.discount_factor(0.0) == pytest.approx(1.0)
    assert p.discount_factor(2.0) == pytest.approx(math.exp(-0.05 * 2.0))


def test_forward_no_dividends_equals_compounded_spot() -> None:
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    assert p.forward(0.0) == pytest.approx(100.0)
    assert p.forward(1.0) == pytest.approx(100.0 * math.exp(0.05))


def test_forward_with_dividends() -> None:
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2, div=0.03)
    assert p.forward(1.0) == pytest.approx(100.0 * math.exp(0.05 - 0.03))


def test_discount_and_forward_reject_negative_time() -> None:
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    with pytest.raises(ValueError, match="t must be non-negative"):
        p.discount_factor(-1.0)
    with pytest.raises(ValueError, match="t must be non-negative"):
        p.forward(-1.0)
