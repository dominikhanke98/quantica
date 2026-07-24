"""Validation of the 1-D interpolation schemes (numerical-validation skill).

Each hand-rolled scheme is anchored to an independent reference: linear to ``numpy.interp``,
the natural cubic to ``scipy`` natural ``CubicSpline``, the monotone cubic to ``scipy``'s
``PchipInterpolator`` — all to machine precision. The analytic derivatives (needed for forward
rates) are checked against finite differences, node interpolation is exact, and outside the
fitted range every scheme flat-extrapolates (constant value, zero derivative).
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.rates.interpolation import (
    LinearInterpolation,
    MonotoneCubicInterpolation,
    NaturalCubicInterpolation,
)

_X = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
_Y = np.array([0.020, 0.022, 0.025, 0.030, 0.032])


def test_linear_matches_numpy_interp() -> None:
    """Piecewise-linear values match ``numpy.interp`` exactly (including flat extrapolation)."""
    fit = LinearInterpolation().fit(_X, _Y)
    xq = np.linspace(-1.0, 12.0, 60)
    assert np.allclose(fit.value(xq), np.interp(xq, _X, _Y))


def test_natural_cubic_matches_scipy() -> None:
    """The natural cubic spline matches ``scipy`` natural ``CubicSpline`` to machine precision."""
    cs = pytest.importorskip("scipy.interpolate").CubicSpline(_X, _Y, bc_type="natural")
    fit = NaturalCubicInterpolation().fit(_X, _Y)
    xq = np.linspace(0.5, 10.0, 100)
    assert np.allclose(fit.value(xq), cs(xq), atol=1e-12)
    assert np.allclose(fit.derivative(xq), cs.derivative()(xq), atol=1e-10)


def test_monotone_cubic_matches_scipy_pchip() -> None:
    """The monotone cubic matches ``scipy``'s ``PchipInterpolator`` (value and derivative)."""
    pchip = pytest.importorskip("scipy.interpolate").PchipInterpolator(_X, _Y)
    fit = MonotoneCubicInterpolation().fit(_X, _Y)
    xq = np.linspace(0.5, 10.0, 100)
    assert np.allclose(fit.value(xq), pchip(xq), atol=1e-12)
    assert np.allclose(fit.derivative(xq), pchip.derivative()(xq), atol=1e-10)


@pytest.mark.parametrize(
    "scheme",
    [LinearInterpolation(), NaturalCubicInterpolation(), MonotoneCubicInterpolation()],
)
def test_interpolates_nodes_and_derivative_matches_finite_difference(scheme: object) -> None:
    """Every scheme passes through the nodes and has a derivative matching finite differences."""
    fit = scheme.fit(_X, _Y)  # type: ignore[attr-defined]
    assert np.allclose(fit.value(_X), _Y, atol=1e-12)
    xm = np.array([0.8, 1.5, 3.0, 7.0])
    h = 1e-6
    fd = (fit.value(xm + h) - fit.value(xm - h)) / (2 * h)
    assert np.allclose(fd, fit.derivative(xm), atol=1e-5)


@pytest.mark.parametrize(
    "scheme",
    [LinearInterpolation(), NaturalCubicInterpolation(), MonotoneCubicInterpolation()],
)
def test_flat_extrapolation(scheme: object) -> None:
    """Outside the fitted range the value is flat and the derivative is zero."""
    fit = scheme.fit(_X, _Y)  # type: ignore[attr-defined]
    assert np.isclose(fit.value(np.array([-2.0]))[0], _Y[0])
    assert np.isclose(fit.value(np.array([20.0]))[0], _Y[-1])
    assert np.allclose(fit.derivative(np.array([-2.0, 20.0])), 0.0)


def test_monotone_cubic_preserves_monotonicity() -> None:
    """On monotone data the PCHIP interpolant stays monotone (no overshoot); the natural
    cubic need not."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0])  # a step-like monotone increasing set
    xq = np.linspace(0.0, 4.0, 200)
    mono = MonotoneCubicInterpolation().fit(x, y).value(xq)
    assert mono.min() >= -1e-12 and mono.max() <= 1.0 + 1e-12  # no overshoot beyond the data
    natural = NaturalCubicInterpolation().fit(x, y).value(xq)
    assert natural.min() < -1e-6 or natural.max() > 1.0 + 1e-6  # the spline overshoots


def test_rejects_bad_nodes() -> None:
    """Non-increasing or too-few nodes are rejected."""
    with pytest.raises(ValueError, match="strictly increasing"):
        LinearInterpolation().fit(np.array([1.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="at least two"):
        MonotoneCubicInterpolation().fit(np.array([1.0]), np.array([0.0]))
