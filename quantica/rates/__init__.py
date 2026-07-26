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

Completing the pillar, the **interest-rate products** (:mod:`~quantica.rates.products`,
:mod:`~quantica.rates.hull_white_options`) price off the curve and the models: swaps (curve-only),
and caps/floors and swaptions priced both with market-standard Black-76 and analytically under
Hull--White (bond options / Jamshidian), with a Hull--White **volatility calibration** that finally
identifies :math:`\sigma` where the curve could not.
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
from quantica.rates.hull_white_options import (
    HullWhiteVolResult,
    calibrate_hull_white_volatility,
    hull_white_bond_option,
    hull_white_cap,
    hull_white_caplet,
    hull_white_price,
    hull_white_price_mc,
    hull_white_swaption,
)
from quantica.rates.instruments import Deposit, RateInstrument, Swap
from quantica.rates.interpolation import (
    Interpolant,
    InterpolationScheme,
    LinearInterpolation,
    MonotoneCubicInterpolation,
    NaturalCubicInterpolation,
)
from quantica.rates.products import (
    Cap,
    Caplet,
    Swaption,
    black76,
    par_swap_rate,
    swap_value,
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
    "Cap",
    "Caplet",
    "CurveInterpolation",
    "Deposit",
    "DiscountCurve",
    "HullWhite",
    "HullWhiteVolResult",
    "Interpolant",
    "InterpolationScheme",
    "LinearInterpolation",
    "MonotoneCubicInterpolation",
    "NaturalCubicInterpolation",
    "RateInstrument",
    "ShortRateModel",
    "Swap",
    "Swaption",
    "Vasicek",
    "black76",
    "bootstrap",
    "calibrate_cir",
    "calibrate_hull_white_volatility",
    "calibrate_vasicek",
    "hull_white_bond_option",
    "hull_white_cap",
    "hull_white_caplet",
    "hull_white_price",
    "hull_white_price_mc",
    "hull_white_swaption",
    "linear_zero",
    "log_linear_discount",
    "monotone_cubic_zero",
    "monte_carlo_discount",
    "natural_cubic_zero",
    "par_swap_rate",
    "swap_value",
]
