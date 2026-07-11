"""Tests for the market carrier and the vol/variance processes."""

from __future__ import annotations

import math

import pytest
from quantica.pricing.processes import (
    BlackScholesProcess,
    HestonProcess,
    Market,
    MertonProcess,
)


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


def test_bump_helpers_return_new_process_with_one_field_changed() -> None:
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2, div=0.01)
    assert p.with_spot(110.0) == BlackScholesProcess(spot=110.0, rate=0.05, vol=0.2, div=0.01)
    assert p.with_rate(0.03) == BlackScholesProcess(spot=100.0, rate=0.03, vol=0.2, div=0.01)
    assert p.with_vol(0.25) == BlackScholesProcess(spot=100.0, rate=0.05, vol=0.25, div=0.01)
    assert p.with_div(0.02) == BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2, div=0.02)
    # The original is untouched (frozen; bumps are copies).
    assert p == BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2, div=0.01)


def test_bump_helpers_revalidate() -> None:
    # A bump that produces an invalid state is rejected by __post_init__.
    p = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    with pytest.raises(ValueError, match="spot must be positive"):
        p.with_spot(-1.0)
    with pytest.raises(ValueError, match="vol must be non-negative"):
        p.with_vol(-0.1)


# --------------------------------------------------------------------------- #
# Market carrier and its composition with the processes
# --------------------------------------------------------------------------- #


def test_market_construction_and_helpers() -> None:
    m = Market(spot=100.0, rate=0.05, div=0.02)
    assert m.spot == 100.0 and m.rate == 0.05 and m.div == 0.02
    assert m.discount_factor(2.0) == pytest.approx(math.exp(-0.05 * 2.0))
    assert m.forward(1.0) == pytest.approx(100.0 * math.exp(0.05 - 0.02))
    assert m.with_spot(110.0) == Market(spot=110.0, rate=0.05, div=0.02)
    assert m.with_rate(0.03) == Market(spot=100.0, rate=0.03, div=0.02)
    assert m.with_div(0.0) == Market(spot=100.0, rate=0.05, div=0.0)


def test_market_rejects_non_positive_spot() -> None:
    with pytest.raises(ValueError, match="spot must be positive"):
        Market(spot=0.0, rate=0.05)


def test_black_scholes_market_view_and_from_market() -> None:
    m = Market(spot=100.0, rate=0.05, div=0.02)
    p = BlackScholesProcess.from_market(m, vol=0.2)
    assert p == BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2, div=0.02)
    assert p.market == m  # round-trips back to the carrier


def test_heston_construction_and_market_view() -> None:
    p = HestonProcess(
        spot=100.0, rate=0.05, v0=0.04, kappa=1.5, theta=0.04, xi=0.3, rho=-0.6, div=0.02
    )
    assert p.spot == 100.0 and p.v0 == 0.04 and p.rho == -0.6
    assert p.market == Market(spot=100.0, rate=0.05, div=0.02)
    assert p.discount_factor(1.0) == pytest.approx(math.exp(-0.05))
    assert p.forward(1.0) == pytest.approx(100.0 * math.exp(0.05 - 0.02))
    assert (
        HestonProcess.from_market(
            Market(spot=100.0, rate=0.05, div=0.02),
            v0=0.04,
            kappa=1.5,
            theta=0.04,
            xi=0.3,
            rho=-0.6,
        )
        == p
    )


def test_heston_feller_condition() -> None:
    # 2*kappa*theta >= xi^2
    ok = HestonProcess(spot=100.0, rate=0.05, v0=0.04, kappa=2.0, theta=0.04, xi=0.3, rho=-0.5)
    assert ok.feller_satisfied  # 2*2*0.04 = 0.16 >= 0.09
    violated = HestonProcess(
        spot=100.0, rate=0.05, v0=0.04, kappa=0.5, theta=0.04, xi=0.5, rho=-0.5
    )
    assert not violated.feller_satisfied  # 2*0.5*0.04 = 0.04 < 0.25


def test_merton_construction_and_market_view() -> None:
    p = MertonProcess(spot=100.0, rate=0.05, vol=0.2, lam=0.75, mu_j=-0.1, sigma_j=0.15, div=0.02)
    assert p.spot == 100.0 and p.vol == 0.2 and p.lam == 0.75 and p.mu_j == -0.1
    assert p.market == Market(spot=100.0, rate=0.05, div=0.02)
    assert p.discount_factor(1.0) == pytest.approx(math.exp(-0.05))
    # The jump compensator leaves the forward at the plain no-jump value.
    assert p.forward(1.0) == pytest.approx(100.0 * math.exp(0.05 - 0.02))
    assert p.compensator == pytest.approx(math.exp(-0.1 + 0.5 * 0.15**2) - 1.0)
    assert (
        MertonProcess.from_market(
            Market(spot=100.0, rate=0.05, div=0.02),
            vol=0.2,
            lam=0.75,
            mu_j=-0.1,
            sigma_j=0.15,
        )
        == p
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"spot": 0.0}, "spot must be positive"),
        ({"v0": -0.1}, "v0 must be non-negative"),
        ({"kappa": -1.0}, "kappa must be non-negative"),
        ({"theta": -0.1}, "theta must be non-negative"),
        ({"xi": -0.1}, "xi must be non-negative"),
        ({"rho": 1.5}, r"rho must be in \[-1, 1\]"),
        ({"rho": -1.5}, r"rho must be in \[-1, 1\]"),
    ],
)
def test_heston_validation(kwargs: dict[str, float], match: str) -> None:
    base: dict[str, float] = {
        "spot": 100.0,
        "rate": 0.05,
        "v0": 0.04,
        "kappa": 1.5,
        "theta": 0.04,
        "xi": 0.3,
        "rho": -0.6,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        HestonProcess(**base)  # type: ignore[arg-type]
