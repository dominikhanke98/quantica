"""Validation of the one-factor short-rate models (numerical-validation skill).

The signature cross-method check: each model's **analytic** zero-coupon bond price must match a
**Monte Carlo** estimate of ``E[exp(-∫ r ds)]`` within standard error — validating the closed
form and the exact-transition simulation against each other, no external reference needed. The
exact simulation is separately pinned against the closed-form transition mean/variance. Anchors
cover the bond-price limits (``P(T,T)=1``; the ``sigma -> 0`` deterministic discount), the
Feller condition for CIR (the same square-root process as Heston's variance), and the Gaussian
vs square-root distinction (Vasicek can go negative, CIR cannot).
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.rates import (
    CIR,
    Deposit,
    HullWhite,
    Swap,
    Vasicek,
    bootstrap,
    monotone_cubic_zero,
    monte_carlo_discount,
)


def _curve():  # type: ignore[no-untyped-def]
    market = [
        Deposit(0.25, 0.030),
        Deposit(1.0, 0.035),
        Swap(2, 0.037),
        Swap(5, 0.042),
        Swap(10, 0.045),
    ]
    return bootstrap(market, monotone_cubic_zero())


def _models():  # type: ignore[no-untyped-def]
    return {
        "vasicek": Vasicek(a=0.3, b=0.04, sigma=0.015, r0=0.03),
        "cir": CIR(a=0.5, b=0.04, sigma=0.05, r0=0.03),
        "hull_white": HullWhite.from_curve(_curve(), a=0.1, sigma=0.01),
    }


# --------------------------------------------------------------------------- #
# Bond-price anchors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["vasicek", "cir", "hull_white"])
def test_zero_coupon_bond_at_maturity_is_one(name: str) -> None:
    """``P(T, T) = 1`` for every model."""
    model = _models()[name]
    assert np.isclose(float(model.zero_coupon_bond(5.0, 5.0, 0.03)), 1.0, atol=1e-12)


def test_vasicek_zero_volatility_is_deterministic_discount() -> None:
    """With ``sigma = 0`` the Vasicek bond is the deterministic discount of the rate path."""
    a, b, r0, big_t = 0.4, 0.05, 0.03, 3.0
    model = Vasicek(a=a, b=b, sigma=0.0, r0=r0)
    integral = b * big_t + (r0 - b) * (1.0 - np.exp(-a * big_t)) / a  # ∫ r(t) dt, r deterministic
    assert np.isclose(float(model.discount_bond(big_t)), np.exp(-integral), atol=1e-12)


# --------------------------------------------------------------------------- #
# The cross-method check: analytic bond == Monte Carlo
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["vasicek", "cir", "hull_white"])
def test_analytic_bond_matches_monte_carlo(name: str) -> None:
    """The closed-form bond price agrees with the exact-transition MC estimate within SE."""
    model = _models()[name]
    analytic = float(model.discount_bond(3.0))
    estimate, std_error = monte_carlo_discount(
        model, 3.0, n_paths=60_000, n_steps=300, rng=np.random.default_rng(0)
    )
    assert abs(estimate - analytic) < 4.0 * std_error


def test_vasicek_simulation_matches_exact_transition_law() -> None:
    """The simulated terminal rate matches the closed-form Gaussian transition mean/variance."""
    a, b, sigma, r0, big_t = 0.4, 0.05, 0.02, 0.03, 3.0
    model = Vasicek(a=a, b=b, sigma=sigma, r0=r0)
    terminal = model.simulate(np.array([0.0, big_t]), 200_000, rng=np.random.default_rng(1))[:, -1]
    exact_mean = b + (r0 - b) * np.exp(-a * big_t)
    exact_var = sigma**2 / (2 * a) * (1.0 - np.exp(-2 * a * big_t))
    assert abs(terminal.mean() - exact_mean) < 5e-4
    assert abs(terminal.std() - np.sqrt(exact_var)) < 5e-4


# --------------------------------------------------------------------------- #
# Model-specific behaviour: Feller, sign
# --------------------------------------------------------------------------- #


def test_cir_feller_condition() -> None:
    """The Feller flag is ``2ab >= sigma^2``, and violating it lets the rate reach ~zero."""
    assert CIR(a=0.5, b=0.04, sigma=0.10, r0=0.03).feller_satisfied  # 2ab=0.04 >= 0.01
    violated = CIR(a=0.1, b=0.02, sigma=0.15, r0=0.03)  # 2ab=0.004 < 0.0225
    assert not violated.feller_satisfied
    times = np.linspace(0.0, 10.0, 500)
    paths = violated.simulate(times, 2_000, rng=np.random.default_rng(2))
    assert paths.min() < 1e-4  # the rate touches zero when Feller is violated
    assert paths.min() >= 0.0  # but never goes negative (square-root process)


def test_vasicek_can_go_negative_but_cir_cannot() -> None:
    """Gaussian Vasicek admits negative rates; the square-root CIR stays non-negative."""
    times = np.linspace(0.0, 10.0, 400)
    vasicek = Vasicek(a=0.2, b=0.005, sigma=0.03, r0=0.01)  # low mean, high vol
    cir = CIR(a=0.5, b=0.04, sigma=0.08, r0=0.03)
    assert vasicek.simulate(times, 5_000, rng=np.random.default_rng(3)).min() < 0.0
    assert cir.simulate(times, 5_000, rng=np.random.default_rng(4)).min() >= 0.0


def test_hull_white_reprices_the_curve_exactly() -> None:
    """Hull--White reproduces the initial discount curve to machine precision (any a, sigma)."""
    curve = _curve()
    for a, sigma in [(0.05, 0.005), (0.1, 0.01), (0.5, 0.02)]:
        model = HullWhite.from_curve(curve, a=a, sigma=sigma)
        pillars = curve.times
        assert np.allclose(model.discount_bond(pillars), curve.discount_factor(pillars), atol=1e-12)


def test_rejects_bad_parameters() -> None:
    """Non-positive mean reversion and negative volatility/level are rejected."""
    with pytest.raises(ValueError, match="a must be positive"):
        Vasicek(a=0.0, b=0.04, sigma=0.01, r0=0.03)
    with pytest.raises(ValueError, match="non-negative"):
        CIR(a=0.3, b=0.04, sigma=-0.01, r0=0.03)
    with pytest.raises(ValueError, match="a must be positive"):
        HullWhite.from_curve(_curve(), a=-0.1, sigma=0.01)
