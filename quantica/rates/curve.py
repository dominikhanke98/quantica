r"""The discount curve — discount factors, zero rates, and forward rates.

A :class:`DiscountCurve` is the object every rates product prices off: a set of pillar times
with their discount factors :math:`P(0,t_i)`, plus an **interpolation scheme** that defines
the curve everywhere in between. It exposes discount factors, (continuously-compounded) zero
rates, simple forward rates between two dates, and the **instantaneous forward**
:math:`f(t) = -\partial \ln P/\partial t` — the quantity most sensitive to the interpolation
choice, and the reason interpolation is treated here as a first-class modelling decision
(:mod:`quantica.rates.interpolation`).

Which *quantity* is interpolated is part of the scheme: linear on **zero rates**, or linear
on **log-discount factors** (equivalently, piecewise-flat instantaneous forwards), or a cubic
on the zero rates. The convenience presets below cover the common choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantica.rates.interpolation import (
    InterpolationScheme,
    LinearInterpolation,
    MonotoneCubicInterpolation,
    NaturalCubicInterpolation,
)

if TYPE_CHECKING:
    from quantica.core.types import FloatArray, FloatLike
    from quantica.rates.interpolation import Interpolant

__all__ = [
    "LOG_LINEAR_DISCOUNT",
    "MONOTONE_CUBIC_ZERO",
    "NATURAL_CUBIC_ZERO",
    "CurveInterpolation",
    "DiscountCurve",
    "linear_zero",
    "log_linear_discount",
    "monotone_cubic_zero",
    "natural_cubic_zero",
]

_ZERO = "zero"
_LOG_DISCOUNT = "log_discount"


@dataclass(frozen=True)
class CurveInterpolation:
    """An interpolation choice for a curve: *what* is interpolated, and *how*.

    Attributes
    ----------
    quantity : {"zero", "log_discount"}
        The pillar quantity interpolated — continuously-compounded zero rates, or
        log-discount factors (whose linear interpolation is piecewise-flat forwards).
    scheme : InterpolationScheme
        The 1-D interpolation method (linear / natural-cubic / monotone-cubic).
    name : str
        Short label used in comparison tables.
    """

    quantity: str
    scheme: InterpolationScheme
    name: str

    def __post_init__(self) -> None:
        """Validate the interpolated quantity."""
        if self.quantity not in (_ZERO, _LOG_DISCOUNT):
            raise ValueError(f"quantity must be 'zero' or 'log_discount', got {self.quantity!r}")


def linear_zero() -> CurveInterpolation:
    """Linear interpolation on zero rates (continuous, kinked forwards)."""
    return CurveInterpolation(_ZERO, LinearInterpolation(), "linear-zero")


def log_linear_discount() -> CurveInterpolation:
    """Linear interpolation on log-discount factors (piecewise-flat instantaneous forwards)."""
    return CurveInterpolation(_LOG_DISCOUNT, LinearInterpolation(), "log-linear-discount")


def natural_cubic_zero() -> CurveInterpolation:
    """Natural cubic spline on zero rates (smooth but can oscillate — negative forwards)."""
    return CurveInterpolation(_ZERO, NaturalCubicInterpolation(), "natural-cubic-zero")


def monotone_cubic_zero() -> CurveInterpolation:
    """Monotone (PCHIP) cubic on zero rates (smooth *and* shape-preserving forwards)."""
    return CurveInterpolation(_ZERO, MonotoneCubicInterpolation(), "monotone-cubic-zero")


#: Preset: linear on log-discount factors — the market-standard robust default.
LOG_LINEAR_DISCOUNT = log_linear_discount()
#: Preset: natural cubic spline on zero rates (exhibits the oscillation artifact).
NATURAL_CUBIC_ZERO = natural_cubic_zero()
#: Preset: monotone cubic on zero rates (smooth, shape-preserving).
MONOTONE_CUBIC_ZERO = monotone_cubic_zero()


class _ConstantInterpolant:
    """A flat interpolant used when a curve has a single pillar (constant value)."""

    def __init__(self, value: float) -> None:
        self._value = value

    def value(self, x: FloatArray) -> FloatArray:
        return np.full(np.shape(x), self._value, dtype=np.float64)

    def derivative(self, x: FloatArray) -> FloatArray:
        return np.zeros(np.shape(x), dtype=np.float64)


class DiscountCurve:
    r"""A discount curve over pillar times, with a configurable interpolation scheme.

    Parameters
    ----------
    times : array_like, shape (n,)
        Strictly increasing pillar maturities in years, all positive.
    discount_factors : array_like, shape (n,)
        The discount factors :math:`P(0, t_i)` at the pillars, in ``(0, 1]``.
    interpolation : CurveInterpolation, optional
        The interpolation choice (default :data:`LOG_LINEAR_DISCOUNT`).

    Raises
    ------
    ValueError
        If the pillars are not strictly increasing positive times, or the discount factors
        are out of range or mismatched in length.
    """

    def __init__(
        self,
        times: FloatArray,
        discount_factors: FloatArray,
        interpolation: CurveInterpolation | None = None,
    ) -> None:
        t = np.asarray(times, dtype=np.float64)
        df = np.asarray(discount_factors, dtype=np.float64)
        if t.ndim != 1 or df.shape != t.shape:
            raise ValueError("times and discount_factors must be 1-D of equal length")
        if t.size < 1:
            raise ValueError("need at least one pillar")
        if np.any(t <= 0.0) or (t.size > 1 and np.any(np.diff(t) <= 0.0)):
            raise ValueError("times must be strictly increasing and positive")
        if np.any(df <= 0.0) or np.any(df > 1.0 + 1e-12):
            raise ValueError("discount_factors must lie in (0, 1]")

        self._times = t
        self._dfs = df
        self._interp = interpolation if interpolation is not None else LOG_LINEAR_DISCOUNT
        self._fitted = self._fit()

    @property
    def times(self) -> FloatArray:
        """The pillar maturities (years)."""
        return self._times

    @property
    def discount_factors(self) -> FloatArray:
        """The pillar discount factors."""
        return self._dfs

    @property
    def interpolation(self) -> CurveInterpolation:
        """The interpolation scheme in use."""
        return self._interp

    def _fit(self) -> Interpolant:
        if self._interp.quantity == _LOG_DISCOUNT:
            # Anchor at (0, 0): ln P(0) = 0, so P(0) = 1 and the short end is well defined.
            nodes_x = np.concatenate([[0.0], self._times])
            nodes_y = np.concatenate([[0.0], np.log(self._dfs)])
            return self._interp.scheme.fit(nodes_x, nodes_y)
        zeros = -np.log(self._dfs) / self._times
        if self._times.size == 1:
            return _ConstantInterpolant(float(zeros[0]))
        return self._interp.scheme.fit(self._times, zeros)

    def discount_factor(self, t: FloatLike) -> FloatArray:
        r"""The discount factor :math:`P(0, t)` (with :math:`P(0,0)=1`)."""
        tq = np.asarray(t, dtype=np.float64)
        if self._interp.quantity == _LOG_DISCOUNT:
            df = np.exp(self._fitted.value(tq))
        else:
            df = np.exp(-self._fitted.value(tq) * tq)
        df = np.where(tq == 0.0, 1.0, df)
        return np.asarray(df, dtype=np.float64)

    def zero_rate(self, t: FloatLike) -> FloatArray:
        r"""The continuously-compounded zero rate :math:`z(t) = -\ln P(0,t)/t`."""
        tq = np.asarray(t, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = -np.log(self.discount_factor(tq)) / tq
        return np.asarray(z, dtype=np.float64)

    def instantaneous_forward(self, t: FloatLike) -> FloatArray:
        r"""The instantaneous forward rate :math:`f(t) = -\partial \ln P/\partial t`."""
        tq = np.asarray(t, dtype=np.float64)
        if self._interp.quantity == _LOG_DISCOUNT:
            fwd = -self._fitted.derivative(tq)
        else:
            fwd = self._fitted.value(tq) + tq * self._fitted.derivative(tq)
        return np.asarray(fwd, dtype=np.float64)

    def forward_rate(self, t1: FloatLike, t2: FloatLike, *, simple: bool = True) -> FloatArray:
        r"""The forward rate between ``t1`` and ``t2``.

        Parameters
        ----------
        t1, t2 : float or ndarray
            Start and end times in years (``t2 > t1``).
        simple : bool, optional
            If ``True`` (default) return the simple (money-market) forward
            :math:`(P(t_1)/P(t_2) - 1)/(t_2 - t_1)`; else the continuously-compounded forward
            :math:`(\ln P(t_1) - \ln P(t_2))/(t_2 - t_1)`.

        Returns
        -------
        ndarray
            The forward rate(s).
        """
        a = np.asarray(t1, dtype=np.float64)
        b = np.asarray(t2, dtype=np.float64)
        tau = b - a
        p1, p2 = self.discount_factor(a), self.discount_factor(b)
        if simple:
            return np.asarray((p1 / p2 - 1.0) / tau, dtype=np.float64)
        return np.asarray((np.log(p1) - np.log(p2)) / tau, dtype=np.float64)
