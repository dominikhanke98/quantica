"""Validation of the discount curve's rate conversions (numerical-validation skill).

The curve's job is to convert consistently between discount factors, zero rates and forward
rates. These checks pin the identities: ``P(0)=1``; the zero rate round-trips through the
discount factor; the instantaneous forward equals ``-d ln P/dt`` (finite-difference checked);
the simple and continuous forwards match their definitions; and the pillars are reproduced
exactly whatever the interpolation.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.rates import (
    DiscountCurve,
    linear_zero,
    log_linear_discount,
    monotone_cubic_zero,
    natural_cubic_zero,
)

_TIMES = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
_ZEROS = np.array([0.030, 0.032, 0.035, 0.037, 0.040, 0.043])
_DFS = np.exp(-_ZEROS * _TIMES)
_SCHEMES = [linear_zero(), log_linear_discount(), natural_cubic_zero(), monotone_cubic_zero()]


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_reproduces_pillars_and_unit_at_zero(scheme: object) -> None:
    """The curve hits its pillar discount factors exactly and has ``P(0)=1``."""
    curve = DiscountCurve(_TIMES, _DFS, scheme)  # type: ignore[arg-type]
    assert np.allclose(curve.discount_factor(_TIMES), _DFS, atol=1e-12)
    assert np.isclose(float(curve.discount_factor(0.0)), 1.0)


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_zero_rate_round_trips(scheme: object) -> None:
    """The zero rate reconstructs the discount factor: ``P(t) = exp(-z(t) t)``."""
    curve = DiscountCurve(_TIMES, _DFS, scheme)  # type: ignore[arg-type]
    tq = np.array([0.75, 1.5, 4.0, 8.0])
    assert np.allclose(curve.discount_factor(tq), np.exp(-curve.zero_rate(tq) * tq), atol=1e-14)
    assert np.allclose(curve.zero_rate(_TIMES), _ZEROS, atol=1e-12)  # pillar zeros recovered


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_instantaneous_forward_matches_derivative_of_log_discount(scheme: object) -> None:
    """The instantaneous forward equals ``-d ln P/dt`` (finite-difference check)."""
    curve = DiscountCurve(_TIMES, _DFS, scheme)  # type: ignore[arg-type]
    tq = np.array([0.75, 1.5, 4.0, 8.0])  # away from pillar kinks
    h = 1e-6
    fd = -(np.log(curve.discount_factor(tq + h)) - np.log(curve.discount_factor(tq - h))) / (2 * h)
    assert np.allclose(curve.instantaneous_forward(tq), fd, atol=1e-5)


def test_simple_and_continuous_forward_definitions() -> None:
    """The forward-rate helper matches the simple and continuous definitions."""
    curve = DiscountCurve(_TIMES, _DFS, log_linear_discount())
    t1, t2 = 2.0, 5.0
    p1, p2 = float(curve.discount_factor(t1)), float(curve.discount_factor(t2))
    assert np.isclose(float(curve.forward_rate(t1, t2, simple=True)), (p1 / p2 - 1.0) / (t2 - t1))
    assert np.isclose(
        float(curve.forward_rate(t1, t2, simple=False)), (np.log(p1) - np.log(p2)) / (t2 - t1)
    )


def test_single_pillar_curve_is_flat() -> None:
    """A one-pillar curve is a flat zero-rate curve (constant forward)."""
    curve = DiscountCurve(np.array([2.0]), np.array([np.exp(-0.03 * 2.0)]), monotone_cubic_zero())
    assert np.isclose(float(curve.zero_rate(0.5)), 0.03, atol=1e-12)
    assert np.isclose(float(curve.instantaneous_forward(7.0)), 0.03, atol=1e-12)


def test_rejects_bad_inputs() -> None:
    """Non-monotone times and out-of-range discount factors are rejected."""
    with pytest.raises(ValueError, match="strictly increasing"):
        DiscountCurve(np.array([1.0, 1.0]), np.array([0.99, 0.98]))
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        DiscountCurve(np.array([1.0, 2.0]), np.array([0.99, 1.5]))
    with pytest.raises(ValueError, match="quantity"):
        from quantica.rates import CurveInterpolation, LinearInterpolation

        CurveInterpolation("swap_rate", LinearInterpolation(), "bad")
