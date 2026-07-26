r"""Validation of Hull--White option pricing and volatility calibration (validation skill).

Three independent checks pin the analytic formulas. **Analytic vs Monte Carlo:** the closed-form
caplet (put-on-bond) and swaption (Jamshidian) prices agree with an exact-transition Hull--White
MC estimate within standard error — the model-independent oracle. **HW vs Black-76:** the Black
implied volatility backed out of a Hull--White price reprices it exactly, so the two routes are
consistent. **Volatility identification (the headline):** Hull--White reprices the curve for
*any* :math:`\sigma` (so the curve carries no volatility information), yet caps and swaptions are
strongly :math:`\sigma`-dependent — a least-squares fit to option prices recovers a known
:math:`\sigma` tightly, closing the step-2 finding that the curve only weakly identifies it.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.rates import (
    Cap,
    Caplet,
    Deposit,
    DiscountCurve,
    HullWhite,
    Swap,
    Swaption,
    bootstrap,
    calibrate_hull_white_volatility,
    hull_white_caplet,
    hull_white_price,
    hull_white_price_mc,
    hull_white_swaption,
    monotone_cubic_zero,
)
from scipy.optimize import brentq


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
# Analytic vs Monte Carlo — the independent oracle
# --------------------------------------------------------------------------- #


def test_caplet_analytic_matches_monte_carlo() -> None:
    """The closed-form Hull--White caplet agrees with the exact-transition MC estimate within SE."""
    model = HullWhite.from_curve(_curve(), a=0.1, sigma=0.015)
    caplet = Caplet(1.0, 1.5, 0.04)
    analytic = hull_white_caplet(model, caplet)
    estimate, se = hull_white_price_mc(
        model, caplet, n_paths=150_000, n_steps=120, rng=np.random.default_rng(0)
    )
    assert abs(estimate - analytic) < 4.0 * se


def test_floorlet_analytic_matches_monte_carlo() -> None:
    """The Hull--White floorlet closed form also agrees with Monte Carlo within SE."""
    model = HullWhite.from_curve(_curve(), a=0.1, sigma=0.015)
    floorlet = Caplet(1.0, 1.5, 0.04, is_floor=True)
    analytic = hull_white_caplet(model, floorlet)
    estimate, se = hull_white_price_mc(
        model, floorlet, n_paths=150_000, n_steps=120, rng=np.random.default_rng(1)
    )
    assert abs(estimate - analytic) < 4.0 * se


def test_swaption_analytic_matches_monte_carlo() -> None:
    """The Jamshidian swaption price agrees with the exact-transition MC estimate within SE."""
    model = HullWhite.from_curve(_curve(), a=0.1, sigma=0.015)
    swaption = Swaption(1.0, 4.0, 0.042, frequency=1, payer=True)
    analytic = hull_white_swaption(model, swaption)
    estimate, se = hull_white_price_mc(
        model, swaption, n_paths=150_000, n_steps=120, rng=np.random.default_rng(2)
    )
    assert abs(estimate - analytic) < 4.0 * se


# --------------------------------------------------------------------------- #
# HW vs Black-76 — consistency through the implied volatility
# --------------------------------------------------------------------------- #


def test_hull_white_caplet_reprices_through_black_implied_vol() -> None:
    """The Black vol implied by a Hull--White caplet reprices it exactly (routes are consistent)."""
    curve = _curve()
    model = HullWhite.from_curve(curve, a=0.1, sigma=0.015)
    caplet = Caplet(1.0, 1.5, 0.04)
    hw_price = hull_white_caplet(model, caplet)
    implied = brentq(lambda v: caplet.black_price(curve, v) - hw_price, 1e-6, 5.0, xtol=1e-14)
    assert implied > 0.0
    assert np.isclose(caplet.black_price(curve, implied), hw_price, atol=1e-14)


def test_payer_receiver_swaption_parity_under_hull_white() -> None:
    """payer - receiver = annuity * (forward - strike) under Hull--White (the forward swap)."""
    curve = _curve()
    model = HullWhite.from_curve(curve, a=0.1, sigma=0.015)
    payer = Swaption(1.0, 4.0, 0.043, frequency=1, payer=True)
    receiver = Swaption(1.0, 4.0, 0.043, frequency=1, payer=False)
    spread = hull_white_swaption(model, payer) - hull_white_swaption(model, receiver)
    forward = payer.forward_swap_rate(curve)
    assert np.isclose(spread, payer.annuity(curve) * (forward - 0.043), atol=1e-8)


# --------------------------------------------------------------------------- #
# The headline: volatility identification
# --------------------------------------------------------------------------- #


def test_volatility_is_recovered_from_option_prices() -> None:
    """A known sigma, invisible to the curve, is recovered tightly by calibrating to cap prices."""
    curve = _curve()
    sigma_true = 0.012
    truth = HullWhite.from_curve(curve, a=0.1, sigma=sigma_true)
    caps = [Cap(0.5, m, 0.04, frequency=2) for m in (2.0, 3.0, 5.0)]
    quotes = [(cap, hull_white_price(truth, cap)) for cap in caps]

    result = calibrate_hull_white_volatility(curve, quotes, a=0.1)
    assert abs(result.sigma - sigma_true) < 1e-4  # sigma pinned to well within a bp of vol
    assert result.rmse < 1e-8  # and the fit is essentially exact


def test_curve_carries_no_volatility_information() -> None:
    """Two very different sigmas reprice the curve identically but the caps very differently."""
    curve = _curve()
    low = HullWhite.from_curve(curve, a=0.1, sigma=0.005)
    high = HullWhite.from_curve(curve, a=0.1, sigma=0.025)
    pillars = curve.times
    # Same discount curve to machine precision (the curve cannot tell the two sigmas apart)...
    assert np.allclose(low.discount_bond(pillars), high.discount_bond(pillars), atol=1e-12)
    # ... but the cap price is materially different (the vol instrument identifies sigma).
    cap = Cap(0.5, 5.0, 0.04, frequency=2)
    assert hull_white_price(high, cap) > 3.0 * hull_white_price(low, cap)


def test_calibration_requires_quotes() -> None:
    """Calibrating with no quotes is an error."""
    with pytest.raises(ValueError, match="at least one quote"):
        calibrate_hull_white_volatility(_curve(), [], a=0.1)


def test_hull_white_price_rejects_unknown_product() -> None:
    """The analytic dispatcher rejects an unsupported product type."""
    with pytest.raises(TypeError, match="unsupported product type"):
        hull_white_price(HullWhite.from_curve(_curve(), a=0.1, sigma=0.01), object())  # type: ignore[arg-type]
