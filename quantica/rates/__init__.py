r"""Fixed income & rates — the first non-equity asset class in ``quantica``.

The rates pillar starts where every rates desk starts: **yield-curve construction**. A
discount curve is bootstrapped from market instruments (deposits at the short end, par swaps
at the long end) so that it reprices every input to par, and the *interpolation scheme* — a
first-class modelling decision, not a hidden default — determines the forward rates between
the pillars.

This foundational step ships:

* **Discount curve** (:mod:`~quantica.rates.curve`) — discount factors, zero rates, simple and
  instantaneous forward rates, under a configurable interpolation.
* **Interpolation schemes** (:mod:`~quantica.rates.interpolation`) — linear, natural cubic
  (which can oscillate) and monotone cubic (shape-preserving), each hand-implemented and aware
  of its own derivative (for the forwards).
* **Instruments + bootstrap** (:mod:`~quantica.rates.instruments`,
  :mod:`~quantica.rates.bootstrap`) — deposits and par swaps, and the sequential bootstrap that
  makes the curve self-consistent with them.

Building on the curve, the **short-rate models** (:mod:`~quantica.rates.short_rate`) — Vasicek,
CIR and Hull--White — add the dynamics: each gives the analytic zero-coupon bond price and
exact-transition Monte Carlo simulation, and is calibrated to the curve
(:mod:`~quantica.rates.calibration`), where Hull--White fits it exactly and Vasicek/CIR
best-fit.

Later step (not yet built): the interest-rate products (swaps, caps/floors, swaptions) that
price off the curve and the models.
"""

from __future__ import annotations

from quantica.rates.bootstrap import bootstrap
from quantica.rates.calibration import (
    CalibrationResult,
    calibrate_cir,
    calibrate_vasicek,
)
from quantica.rates.curve import (
    LOG_LINEAR_DISCOUNT,
    MONOTONE_CUBIC_ZERO,
    NATURAL_CUBIC_ZERO,
    CurveInterpolation,
    DiscountCurve,
    linear_zero,
    log_linear_discount,
    monotone_cubic_zero,
    natural_cubic_zero,
)
from quantica.rates.instruments import Deposit, RateInstrument, Swap
from quantica.rates.interpolation import (
    Interpolant,
    InterpolationScheme,
    LinearInterpolation,
    MonotoneCubicInterpolation,
    NaturalCubicInterpolation,
)
from quantica.rates.short_rate import (
    CIR,
    HullWhite,
    ShortRateModel,
    Vasicek,
    monte_carlo_discount,
)

__all__ = [
    "CIR",
    "LOG_LINEAR_DISCOUNT",
    "MONOTONE_CUBIC_ZERO",
    "NATURAL_CUBIC_ZERO",
    "CalibrationResult",
    "CurveInterpolation",
    "Deposit",
    "DiscountCurve",
    "HullWhite",
    "Interpolant",
    "InterpolationScheme",
    "LinearInterpolation",
    "MonotoneCubicInterpolation",
    "NaturalCubicInterpolation",
    "RateInstrument",
    "ShortRateModel",
    "Swap",
    "Vasicek",
    "bootstrap",
    "calibrate_cir",
    "calibrate_vasicek",
    "linear_zero",
    "log_linear_discount",
    "monotone_cubic_zero",
    "monte_carlo_discount",
    "natural_cubic_zero",
]
