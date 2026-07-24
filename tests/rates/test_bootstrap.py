"""Validation of the curve bootstrap (numerical-validation skill).

The foundational anchor is **self-consistency**: whatever the interpolation, the bootstrapped
curve must reprice every input instrument back to par to machine precision — the known-truth
proof that the bootstrap is correct. The headline is that interpolation is a *modelling
decision*: the same market inputs, all repriced exactly, produce materially different forward
curves — and the smooth cubic schemes can even manufacture **negative** forwards where the
robust log-linear scheme cannot (Hagan & West). A hand-computed deposit+swap case pins the
arithmetic, and the usual arbitrage sanity (positive, decreasing discount factors) is checked.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.rates import (
    Deposit,
    Swap,
    bootstrap,
    linear_zero,
    log_linear_discount,
    monotone_cubic_zero,
    natural_cubic_zero,
)

_NORMAL = [
    Deposit(0.25, 0.030),
    Deposit(0.5, 0.032),
    Deposit(1.0, 0.035),
    Swap(2, 0.037),
    Swap(3, 0.039),
    Swap(5, 0.042),
    Swap(7, 0.044),
    Swap(10, 0.045),
]
# A curve flat at 3% with one tenor (4y) trading rich — a mild, realistic stress.
_BUMP = [
    Deposit(0.5, 0.030),
    Deposit(1.0, 0.030),
    Swap(2, 0.030),
    Swap(3, 0.030),
    Swap(4, 0.036),
    Swap(5, 0.030),
    Swap(7, 0.030),
    Swap(10, 0.030),
]
_SCHEMES = [linear_zero(), log_linear_discount(), natural_cubic_zero(), monotone_cubic_zero()]


# --------------------------------------------------------------------------- #
# The foundational anchor: reprice every input to par
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_reprices_all_inputs_to_par(scheme: object) -> None:
    """Whatever the interpolation, the curve values every input instrument at par."""
    curve = bootstrap(_NORMAL, scheme)  # type: ignore[arg-type]
    residuals = [abs(inst.value(curve)) for inst in _NORMAL]
    assert max(residuals) < 1e-12  # machine-precision self-consistency


def test_hand_computed_deposit_and_swap() -> None:
    """A single deposit + a 2y annual swap match the closed-form discount factors."""
    curve = bootstrap([Deposit(1.0, 0.03), Swap(2, 0.035)], log_linear_discount())
    p1 = 1.0 / (1.0 + 0.03)  # deposit: P(1) = 1/(1 + r*tau)
    p2 = (1.0 - 0.035 * p1) / (1.0 + 0.035)  # par swap with coupons on the 1y and 2y pillars
    assert np.isclose(float(curve.discount_factor(1.0)), p1, atol=1e-14)
    assert np.isclose(float(curve.discount_factor(2.0)), p2, atol=1e-12)


# --------------------------------------------------------------------------- #
# The headline: interpolation is a modelling choice
# --------------------------------------------------------------------------- #


def test_interpolation_materially_changes_forwards() -> None:
    """All schemes reprice the inputs, yet imply forwards that differ by tens of bps."""
    tq = np.linspace(0.1, 10.0, 400)
    forwards = []
    for scheme in _SCHEMES:
        curve = bootstrap(_NORMAL, scheme)
        assert max(abs(inst.value(curve)) for inst in _NORMAL) < 1e-12  # all still at par
        forwards.append(curve.instantaneous_forward(tq))
    stacked = np.array(forwards)
    max_divergence = float((stacked.max(axis=0) - stacked.min(axis=0)).max())
    assert max_divergence > 0.0020  # > 20 bps of cross-scheme forward disagreement


def test_cubic_forwards_go_negative_where_log_linear_stays_positive() -> None:
    """Under a mild stress the smooth cubic manufactures negative forwards; log-linear cannot.

    Log-linear on discount factors gives piecewise-flat forwards that stay positive whenever
    the discount factors decrease (a positive-rate curve). The natural cubic on zero rates is
    not shape-preserving, so it overshoots and produces negative (arbitrageable) instantaneous
    forwards on the *same* inputs — the classic spline artifact.
    """
    tq = np.linspace(0.05, 10.0, 800)
    log_linear = bootstrap(_BUMP, log_linear_discount()).instantaneous_forward(tq)
    natural = bootstrap(_BUMP, natural_cubic_zero()).instantaneous_forward(tq)
    assert log_linear.min() > 0.0  # robust: never negative
    assert natural.min() < 0.0  # the cubic-spline oscillation artifact


def test_log_linear_forwards_are_piecewise_constant() -> None:
    """Log-linear discount factors imply piecewise-flat instantaneous forwards."""
    curve = bootstrap(_NORMAL, log_linear_discount())
    # 0.6 and 0.9 lie in the same (0.5, 1.0) pillar interval -> identical forward.
    assert np.isclose(
        float(curve.instantaneous_forward(0.6)), float(curve.instantaneous_forward(0.9)), atol=1e-12
    )
    # 1.5 and 2.5 lie in different intervals -> different forwards.
    assert not np.isclose(
        float(curve.instantaneous_forward(1.5)), float(curve.instantaneous_forward(2.5)), atol=1e-6
    )


# --------------------------------------------------------------------------- #
# Arbitrage sanity and mechanics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_discount_factors_positive_and_decreasing(scheme: object) -> None:
    """In a positive-rate regime discount factors are positive and strictly decreasing."""
    curve = bootstrap(_NORMAL, scheme)  # type: ignore[arg-type]
    tq = np.linspace(0.05, 10.0, 500)
    dfs = curve.discount_factor(tq)
    assert np.all(dfs > 0.0)
    assert np.all(np.diff(dfs) < 0.0)  # monotonically decreasing
    assert np.all(np.isfinite(curve.instantaneous_forward(tq)))


def test_deterministic() -> None:
    """The bootstrap is a pure function of its inputs."""
    a = bootstrap(_NORMAL, monotone_cubic_zero())
    b = bootstrap(_NORMAL, monotone_cubic_zero())
    assert np.array_equal(a.discount_factors, b.discount_factors)


def test_rejects_bad_inputs() -> None:
    """Empty instrument sets and duplicate maturities are rejected."""
    with pytest.raises(ValueError, match="at least one instrument"):
        bootstrap([])
    with pytest.raises(ValueError, match="strictly increasing"):
        bootstrap([Swap(2, 0.03), Deposit(2.0, 0.03)])  # two instruments at t=2
