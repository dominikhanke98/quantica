r"""1-D interpolation schemes for yield-curve construction.

Interpolation is a **first-class modelling decision** on a curve, not a hidden default: the
scheme chosen for the pillar quantities (zero rates or log-discount factors) determines the
shape of the *forward* curve between pillars, and different schemes — all repricing the same
market inputs exactly — imply materially different forwards (Hagan & West 2006). Each scheme
here is hand-implemented (the demonstrable skill) and exposes both the interpolated value and
its **derivative**, because the instantaneous forward rate is a derivative of the curve.

Three schemes, spanning the smoothness/robustness trade-off:

* :class:`LinearInterpolation` — piecewise linear; continuous but kinked (its derivative
  jumps at every pillar), so forwards are discontinuous or piecewise flat depending on the
  quantity interpolated.
* :class:`NaturalCubicInterpolation` — the natural cubic spline; smooth (``C²``) but *not*
  shape-preserving, so it can **oscillate** and produce negative forwards — the classic
  cubic-spline artifact.
* :class:`MonotoneCubicInterpolation` — the Fritsch--Carlson / PCHIP monotone cubic Hermite;
  smooth (``C¹``) *and* shape-preserving, so it does not overshoot — the practitioners' fix.

All three flat-extrapolate outside the pillar range (constant value, zero derivative).

References
----------
Fritsch, F. N. & Carlson, R. E. (1980). "Monotone piecewise cubic interpolation",
*SIAM J. Numer. Anal.* 17, 238--246.
Hagan, P. & West, G. (2006). "Interpolation methods for curve construction",
*Applied Mathematical Finance* 13, 89--129.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "Interpolant",
    "InterpolationScheme",
    "LinearInterpolation",
    "MonotoneCubicInterpolation",
    "NaturalCubicInterpolation",
]


class Interpolant(Protocol):
    """A fitted interpolation: evaluate the value and its first derivative anywhere."""

    def value(self, x: FloatArray) -> FloatArray:
        """Interpolated value at ``x`` (flat-extrapolated outside the fitted range)."""
        ...

    def derivative(self, x: FloatArray) -> FloatArray:
        """First derivative at ``x`` (zero outside the fitted range)."""
        ...


class InterpolationScheme(Protocol):
    """A named strategy that fits an :class:`Interpolant` to nodes ``(x, y)``."""

    @property
    def name(self) -> str:
        """Short label used in comparison tables."""
        ...

    def fit(self, x: FloatArray, y: FloatArray) -> Interpolant:
        """Fit the scheme to strictly increasing nodes ``x`` with values ``y``."""
        ...


def _check_nodes(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if xs.ndim != 1 or ys.shape != xs.shape:
        raise ValueError("x and y must be 1-D arrays of equal length")
    if xs.size < 2:
        raise ValueError("need at least two nodes to interpolate")
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError("x must be strictly increasing")
    return xs, ys


# --------------------------------------------------------------------------- #
# Piecewise linear
# --------------------------------------------------------------------------- #


class _LinearInterpolant:
    """Piecewise-linear interpolant with flat extrapolation."""

    def __init__(self, x: FloatArray, y: FloatArray) -> None:
        self._x = x
        self._y = y
        self._slopes = np.diff(y) / np.diff(x)

    def value(self, x: FloatArray) -> FloatArray:
        # np.interp clamps to the endpoints -> flat extrapolation, exactly what we want.
        return np.asarray(np.interp(np.asarray(x, dtype=np.float64), self._x, self._y))

    def derivative(self, x: FloatArray) -> FloatArray:
        xq = np.atleast_1d(np.asarray(x, dtype=np.float64))
        idx = np.clip(np.searchsorted(self._x, xq, side="right") - 1, 0, self._slopes.size - 1)
        deriv = self._slopes[idx]
        deriv = np.where((xq < self._x[0]) | (xq > self._x[-1]), 0.0, deriv)  # flat outside
        return np.asarray(deriv.reshape(np.shape(x)), dtype=np.float64)


class LinearInterpolation:
    """Piecewise-linear interpolation."""

    name = "linear"

    def fit(self, x: FloatArray, y: FloatArray) -> Interpolant:
        """Fit a piecewise-linear interpolant to ``(x, y)``."""
        xs, ys = _check_nodes(x, y)
        return _LinearInterpolant(xs, ys)


# --------------------------------------------------------------------------- #
# Cubic Hermite core (shared by the natural and monotone cubic schemes)
# --------------------------------------------------------------------------- #


class _CubicHermiteInterpolant:
    """Piecewise cubic Hermite interpolant from node tangents, with flat extrapolation."""

    def __init__(self, x: FloatArray, y: FloatArray, tangents: FloatArray) -> None:
        self._x = x
        self._y = y
        self._m = tangents

    def _segment(self, xq: FloatArray) -> FloatArray:
        return np.clip(np.searchsorted(self._x, xq, side="right") - 1, 0, self._x.size - 2)

    def value(self, x: FloatArray) -> FloatArray:
        xq = np.atleast_1d(np.asarray(x, dtype=np.float64))
        clamped = np.clip(xq, self._x[0], self._x[-1])  # flat extrapolation
        i = self._segment(clamped)
        h = self._x[i + 1] - self._x[i]
        t = (clamped - self._x[i]) / h
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        out = (
            h00 * self._y[i]
            + h10 * h * self._m[i]
            + h01 * self._y[i + 1]
            + h11 * h * self._m[i + 1]
        )
        return np.asarray(out.reshape(np.shape(x)), dtype=np.float64)

    def derivative(self, x: FloatArray) -> FloatArray:
        xq = np.atleast_1d(np.asarray(x, dtype=np.float64))
        inside = (xq >= self._x[0]) & (xq <= self._x[-1])
        clamped = np.clip(xq, self._x[0], self._x[-1])
        i = self._segment(clamped)
        h = self._x[i + 1] - self._x[i]
        t = (clamped - self._x[i]) / h
        d00 = 6 * t**2 - 6 * t
        d10 = 3 * t**2 - 4 * t + 1
        d01 = -6 * t**2 + 6 * t
        d11 = 3 * t**2 - 2 * t
        slope = (
            d00 * self._y[i] / h
            + d10 * self._m[i]
            + d01 * self._y[i + 1] / h
            + d11 * self._m[i + 1]
        )
        slope = np.where(inside, slope, 0.0)  # flat outside -> zero derivative
        return np.asarray(slope.reshape(np.shape(x)), dtype=np.float64)


# --------------------------------------------------------------------------- #
# Natural cubic spline
# --------------------------------------------------------------------------- #


class NaturalCubicInterpolation:
    """The natural cubic spline (second derivative zero at both ends).

    Smooth (``C²``) but not shape-preserving: it can overshoot between nodes and produce
    oscillatory — even negative — forward rates. Included precisely to exhibit that artifact.
    """

    name = "natural-cubic"

    def fit(self, x: FloatArray, y: FloatArray) -> Interpolant:
        """Fit a natural cubic spline; tangents come from the solved second derivatives."""
        xs, ys = _check_nodes(x, y)
        tangents = _natural_cubic_tangents(xs, ys)
        return _CubicHermiteInterpolant(xs, ys, tangents)


def _natural_cubic_tangents(x: FloatArray, y: FloatArray) -> FloatArray:
    """Node first-derivatives of the natural cubic spline (via a tridiagonal solve)."""
    n = x.size
    h = np.diff(x)
    delta = np.diff(y) / h
    if n == 2:
        return np.array([delta[0], delta[0]], dtype=np.float64)

    # Solve the standard symmetric tridiagonal system for second derivatives M (M_0=M_n=0).
    lower = h[:-1].copy()
    diag = 2.0 * (h[:-1] + h[1:])
    upper = h[1:].copy()
    rhs = 6.0 * (delta[1:] - delta[:-1])
    second = np.zeros(n, dtype=np.float64)
    second[1:-1] = _solve_tridiagonal(lower, diag, upper, rhs)

    # Convert second derivatives to node first-derivatives (tangents) for the Hermite form.
    tangents = delta[0] - h[0] * (2.0 * second[0] + second[1]) / 6.0
    left = np.empty(n, dtype=np.float64)
    left[0] = tangents
    left[1:] = delta + h * (2.0 * second[1:] + second[:-1]) / 6.0
    return left


def _solve_tridiagonal(
    lower: FloatArray, diag: FloatArray, upper: FloatArray, rhs: FloatArray
) -> FloatArray:
    """Thomas algorithm for an ``m x m`` tridiagonal system (interior spline equations)."""
    m = diag.size
    c = np.empty(m, dtype=np.float64)
    d = np.empty(m, dtype=np.float64)
    c[0] = upper[0] / diag[0]
    d[0] = rhs[0] / diag[0]
    for i in range(1, m):
        denom = diag[i] - lower[i] * c[i - 1]
        c[i] = upper[i] / denom if i < m - 1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i - 1]) / denom
    solution = np.empty(m, dtype=np.float64)
    solution[-1] = d[-1]
    for i in range(m - 2, -1, -1):
        solution[i] = d[i] - c[i] * solution[i + 1]
    return solution


# --------------------------------------------------------------------------- #
# Monotone cubic (Fritsch--Carlson / PCHIP)
# --------------------------------------------------------------------------- #


class MonotoneCubicInterpolation:
    """The Fritsch--Carlson / PCHIP monotone cubic Hermite interpolation.

    Smooth (``C¹``) and shape-preserving: monotone data give a monotone interpolant with no
    overshoot, so forward rates stay well-behaved — the practitioners' answer to the cubic
    spline's oscillation.
    """

    name = "monotone-cubic"

    def fit(self, x: FloatArray, y: FloatArray) -> Interpolant:
        """Fit the PCHIP monotone cubic, computing shape-preserving node tangents."""
        xs, ys = _check_nodes(x, y)
        tangents = _pchip_tangents(xs, ys)
        return _CubicHermiteInterpolant(xs, ys, tangents)


def _pchip_tangents(x: FloatArray, y: FloatArray) -> FloatArray:
    """PCHIP node tangents: weighted-harmonic interior slopes, shape-limited endpoints."""
    h = np.diff(x)
    delta = np.diff(y) / h
    n = x.size
    m = np.zeros(n, dtype=np.float64)

    # Interior: zero at local extrema, else the Fritsch--Carlson weighted harmonic mean.
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0.0:
            m[i] = 0.0
        else:
            w1 = 2.0 * h[i] + h[i - 1]
            w2 = h[i] + 2.0 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    m[0] = _pchip_end_slope(h[0], h[1], delta[0], delta[1]) if n > 2 else delta[0]
    m[-1] = _pchip_end_slope(h[-1], h[-2], delta[-1], delta[-2]) if n > 2 else delta[-1]
    return m


def _pchip_end_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    """One-sided endpoint slope with the standard PCHIP shape-preserving limiter."""
    slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if np.sign(slope) != np.sign(d0):
        return 0.0
    if np.sign(d0) != np.sign(d1) and abs(slope) > 3.0 * abs(d0):
        return 3.0 * d0
    return float(slope)
