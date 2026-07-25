r"""Calibrating short-rate models to a discount curve — exact fit vs best fit.

The structural point of this step: **Hull--White fits the initial term structure exactly** (by
construction, for any volatility parameters — its time-dependent drift absorbs the whole
curve), while **Vasicek and CIR, with constant parameters, can only best-fit** an arbitrary
curve and leave a residual. This module calibrates the two constant-parameter models to a
:class:`~quantica.rates.curve.DiscountCurve` by least squares (``scipy.optimize`` — the only
place calibration needs it, CLAUDE.md §3); Hull--White needs no fitting to the curve, so it is
built directly with :meth:`~quantica.rates.short_rate.HullWhite.from_curve`.

The parameters ``(a, b, sigma, r_0)`` are fitted to the pillar zero rates by least squares.
Note the honest caveat surfaced by the fit: the *curve* barely identifies :math:`sigma` —
volatility enters bond prices only through a small convexity term, so pinning down
:math:`sigma` really needs volatility instruments (caps/swaptions, a later step). The residual,
not the fitted :math:`sigma`, is the deliverable here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

from quantica.rates.short_rate import CIR, Vasicek

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.rates.curve import DiscountCurve
    from quantica.rates.short_rate import ShortRateModel

_ModelBuilder = Callable[[float, float, float, float], "ShortRateModel"]

__all__ = ["CalibrationResult", "calibrate_cir", "calibrate_vasicek"]

_SHORT_END = 1.0e-8


@dataclass(frozen=True)
class CalibrationResult:
    """The fitted model and how well it reproduces the curve.

    Attributes
    ----------
    model : ShortRateModel
        The calibrated model.
    rmse : float
        Root-mean-square zero-rate error across the fitted pillars (in rate units).
    max_abs_error : float
        Largest absolute zero-rate error across the pillars (in rate units).
    n_pillars : int
        Number of curve pillars used in the fit.
    """

    model: ShortRateModel
    rmse: float
    max_abs_error: float
    n_pillars: int


def calibrate_vasicek(curve: DiscountCurve) -> CalibrationResult:
    """Best-fit a :class:`~quantica.rates.short_rate.Vasicek` model to ``curve``.

    Parameters
    ----------
    curve : DiscountCurve
        The target discount curve.

    Returns
    -------
    CalibrationResult
        The fitted Vasicek model and its (non-zero) fit residual.
    """
    return _calibrate(curve, _build_vasicek, r0_bounds=(-0.2, 0.5))


def calibrate_cir(curve: DiscountCurve) -> CalibrationResult:
    """Best-fit a :class:`~quantica.rates.short_rate.CIR` model to ``curve``.

    Parameters
    ----------
    curve : DiscountCurve
        The target discount curve.

    Returns
    -------
    CalibrationResult
        The fitted CIR model and its (non-zero) fit residual.
    """
    return _calibrate(curve, _build_cir, r0_bounds=(1e-8, 0.5))


def _build_vasicek(a: float, b: float, sigma: float, r0: float) -> ShortRateModel:
    return Vasicek(a=a, b=b, sigma=sigma, r0=r0)


def _build_cir(a: float, b: float, sigma: float, r0: float) -> ShortRateModel:
    return CIR(a=a, b=b, sigma=sigma, r0=r0)


def _calibrate(
    curve: DiscountCurve, build: _ModelBuilder, *, r0_bounds: tuple[float, float]
) -> CalibrationResult:
    """Fit ``(a, b, sigma, r0)`` to the curve zero rates by least squares."""
    times = np.asarray(curve.times, dtype=np.float64)
    market_zero = np.asarray(curve.zero_rate(times), dtype=np.float64)
    short_rate = float(curve.instantaneous_forward(_SHORT_END))

    def residuals(params: FloatArray) -> FloatArray:
        a, b, sigma, r0 = params
        model = build(a, b, sigma, r0)
        model_zero = -np.log(model.discount_bond(times)) / times
        return np.asarray(model_zero - market_zero, dtype=np.float64)

    r0_lo, r0_hi = r0_bounds
    initial = np.array([0.2, max(market_zero[-1], 1e-3), 0.01, short_rate], dtype=np.float64)
    bounds = ([1e-3, 1e-4, 1e-5, r0_lo], [5.0, 0.5, 0.5, r0_hi])
    fit = least_squares(residuals, initial, bounds=bounds)
    a, b, sigma, r0 = fit.x
    model = build(a, b, sigma, r0)
    errors = np.abs(residuals(fit.x))
    return CalibrationResult(
        model=model,
        rmse=float(np.sqrt(np.mean(errors**2))),
        max_abs_error=float(np.max(errors)),
        n_pillars=int(times.size),
    )
