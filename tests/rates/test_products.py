"""Validation of the interest-rate products — swaps and Black-76 (numerical-validation skill).

The curve tie-back is the anchor: a swap struck at :func:`par_swap_rate` values to **zero**
off the very curve it was bootstrapped from (self-consistency with step 1). Black-76 is pinned
by two model-free identities that hold for *any* volatility — the undiscounted call/put spread
is the forward-minus-strike, and a **cap minus a floor equals the underlying swap** (the vol
terms cancel), which doubles as the put-call parity of the strip.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from quantica.rates import (
    Cap,
    Caplet,
    Deposit,
    DiscountCurve,
    Swap,
    Swaption,
    black76,
    bootstrap,
    monotone_cubic_zero,
    par_swap_rate,
    swap_value,
)


def _curve() -> DiscountCurve:
    market = [
        Deposit(0.25, 0.030),
        Deposit(0.5, 0.032),
        Deposit(1.0, 0.035),
        Swap(2, 0.037),
        Swap(3, 0.039),
        Swap(5, 0.042),
        Swap(7, 0.044),
        Swap(10, 0.045),
    ]
    return bootstrap(market, monotone_cubic_zero())


# --------------------------------------------------------------------------- #
# Swaps: the par-rate tie-back to the bootstrap
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("maturity", [2.0, 5.0, 10.0])
@pytest.mark.parametrize("frequency", [1, 2, 4])
def test_swap_struck_at_par_values_to_zero(maturity: float, frequency: int) -> None:
    """A swap struck at its par rate has zero present value — the curve self-consistency check."""
    curve = _curve()
    par = par_swap_rate(curve, maturity, frequency=frequency)
    value = swap_value(curve, maturity, par, frequency=frequency)
    assert abs(value) < 1e-12


def test_bootstrapped_par_rate_recovers_the_input_swap_quote() -> None:
    """The par rate of a curve-input maturity equals the market quote it was bootstrapped from."""
    curve = _curve()
    assert np.isclose(par_swap_rate(curve, 5.0, frequency=1), 0.042, atol=1e-12)
    assert np.isclose(par_swap_rate(curve, 10.0, frequency=1), 0.045, atol=1e-12)


def test_payer_and_receiver_are_mirror_images() -> None:
    """Pay-fixed and receive-fixed values are equal and opposite; sign flips around par."""
    curve = _curve()
    par = par_swap_rate(curve, 5.0, frequency=2)
    payer = swap_value(curve, 5.0, par + 0.01, frequency=2, pay_fixed=True)
    receiver = swap_value(curve, 5.0, par + 0.01, frequency=2, pay_fixed=False)
    assert np.isclose(payer, -receiver, atol=1e-14)
    assert payer < 0.0  # paying an above-par fixed rate is a loss to the payer


def test_swap_value_scales_with_notional() -> None:
    """The swap value is linear in the notional."""
    curve = _curve()
    one = swap_value(curve, 5.0, 0.03, frequency=2, notional=1.0)
    million = swap_value(curve, 5.0, 0.03, frequency=2, notional=1_000_000.0)
    assert np.isclose(million, 1_000_000.0 * one, rtol=1e-14)


# --------------------------------------------------------------------------- #
# Black-76: model-free identities
# --------------------------------------------------------------------------- #


def test_black76_call_put_spread_is_forward_minus_strike() -> None:
    """Undiscounted call minus put equals ``F - K`` for any volatility (put-call parity)."""
    call = black76(0.04, 0.038, 0.25, 2.0, call=True)
    put = black76(0.04, 0.038, 0.25, 2.0, call=False)
    assert np.isclose(call - put, 0.04 - 0.038, atol=1e-14)


def test_black76_zero_vol_is_intrinsic() -> None:
    """With zero volatility Black-76 collapses to the forward intrinsic value."""
    assert np.isclose(black76(0.05, 0.04, 0.0, 1.0, call=True), 0.01, atol=1e-14)
    assert black76(0.03, 0.04, 0.0, 1.0, call=True) == 0.0
    assert np.isclose(black76(0.03, 0.04, 0.0, 1.0, call=False), 0.01, atol=1e-14)


def test_black76_monotone_in_vol() -> None:
    """An option is worth more the higher the volatility."""
    prices = [black76(0.04, 0.04, v, 2.0, call=True) for v in (0.1, 0.2, 0.3, 0.4)]
    assert all(b > a for a, b in pairwise(prices))


# --------------------------------------------------------------------------- #
# Cap - floor = swap: the strip put-call parity
# --------------------------------------------------------------------------- #


def test_cap_minus_floor_equals_underlying_swap() -> None:
    """cap(K) - floor(K) equals the value of paying fixed ``K`` on the cap's schedule."""
    curve = _curve()
    strike = 0.04
    cap = Cap(0.5, 5.0, strike, frequency=2)
    floor = Cap(0.5, 5.0, strike, frequency=2, is_floor=True)
    # Reference swap value: sum over the strip of tau * P(pay) * (forward - K).
    swap = sum(
        c.accrual * float(curve.discount_factor(c.payment)) * (c.forward_rate(curve) - strike)
        for c in cap.caplets
    )
    parity_gap = cap.black_price(curve, 0.30) - floor.black_price(curve, 0.30) - swap
    assert abs(parity_gap) < 1e-14


def test_cap_is_the_sum_of_its_caplets() -> None:
    """A cap price is exactly the sum of the individual caplet prices at the same vol."""
    curve = _curve()
    cap = Cap(0.5, 3.0, 0.04, frequency=2)
    strip = sum(c.black_price(curve, 0.25) for c in cap.caplets)
    assert np.isclose(cap.black_price(curve, 0.25), strip, atol=1e-15)
    assert len(cap.caplets) == 5  # (3.0 - 0.5) * 2 = 5 periods


def test_caplet_schedule_validation() -> None:
    """Bad reset/payment ordering and degenerate schedules are rejected."""
    with pytest.raises(ValueError, match="0 < reset < payment"):
        Caplet(1.0, 0.5, 0.04)
    with pytest.raises(ValueError, match="0 < start < end"):
        Cap(2.0, 1.0, 0.04)
    with pytest.raises(ValueError, match="frequency must be at least 1"):
        Cap(0.5, 2.0, 0.04, frequency=0)


# --------------------------------------------------------------------------- #
# Swaption forward measure: annuity and forward swap rate
# --------------------------------------------------------------------------- #


def test_swaption_forward_swap_rate_matches_spot_par_rate_when_starting_now() -> None:
    """A swaption's forward swap rate equals the forward par rate rebuilt from discount factors."""
    curve = _curve()
    swaption = Swaption(1.0, 4.0, 0.04, frequency=1)
    forward = swaption.forward_swap_rate(curve)
    # Rebuild the forward par rate directly from discount factors over [1, 5].
    times = swaption.payment_times
    annuity = float(np.sum(curve.discount_factor(times)))
    p_start = float(curve.discount_factor(1.0))
    p_end = float(curve.discount_factor(5.0))
    assert np.isclose(forward, (p_start - p_end) / annuity, atol=1e-14)


def test_payer_and_receiver_swaption_parity() -> None:
    """payer - receiver = annuity * (forward - strike): the forward swap value (any vol)."""
    curve = _curve()
    payer = Swaption(1.0, 4.0, 0.043, frequency=1, payer=True)
    receiver = Swaption(1.0, 4.0, 0.043, frequency=1, payer=False)
    forward = payer.forward_swap_rate(curve)
    annuity = payer.annuity(curve)
    spread = payer.black_price(curve, 0.20) - receiver.black_price(curve, 0.20)
    assert np.isclose(spread, annuity * (forward - 0.043), atol=1e-14)


def test_swaption_validation() -> None:
    """Non-positive expiry/tenor and sub-annual frequency are rejected."""
    with pytest.raises(ValueError, match="expiry and tenor must be positive"):
        Swaption(0.0, 4.0, 0.04)
    with pytest.raises(ValueError, match="frequency must be at least 1"):
        Swaption(1.0, 4.0, 0.04, frequency=0)
