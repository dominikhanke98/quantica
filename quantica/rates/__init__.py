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

Later steps (not in this foundation): short-rate models (Vasicek, Hull--White) and the rates
products that price off the curve.
"""

from __future__ import annotations

from quantica.rates.bootstrap import bootstrap
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

__all__ = [
    "LOG_LINEAR_DISCOUNT",
    "MONOTONE_CUBIC_ZERO",
    "NATURAL_CUBIC_ZERO",
    "CurveInterpolation",
    "Deposit",
    "DiscountCurve",
    "Interpolant",
    "InterpolationScheme",
    "LinearInterpolation",
    "MonotoneCubicInterpolation",
    "NaturalCubicInterpolation",
    "RateInstrument",
    "Swap",
    "bootstrap",
    "linear_zero",
    "log_linear_discount",
    "monotone_cubic_zero",
    "natural_cubic_zero",
]
