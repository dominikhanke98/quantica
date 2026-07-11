"""Pricing subpackage: instruments, processes, and numerical engines.

The design follows the Instrument / Process / Engine separation (CLAUDE.md §4):
an *instrument* is the contract, a *process* is the market dynamics, and an
*engine* is a numerical method that prices an instrument under a process.
"""

from __future__ import annotations

from quantica.core.types import AveragingType, BarrierType, ExerciseStyle, OptionType
from quantica.pricing.calibration import (
    DEFAULT_BOUNDS,
    HestonCalibrationResult,
    HestonParams,
    ObjectiveProfile,
    ParamBounds,
    VolQuote,
    calibrate_heston,
    profile_objective,
    vol_surface_from_grid,
)
from quantica.pricing.engines.analytic import AnalyticEuropeanEngine
from quantica.pricing.engines.asian import AsianMonteCarloEngine, geometric_asian_price
from quantica.pricing.engines.barrier import BarrierMonteCarloEngine, barrier_price
from quantica.pricing.engines.binomial import BinomialEngine
from quantica.pricing.engines.finitediff import FiniteDifferenceEngine
from quantica.pricing.engines.heston import HestonFFTEngine, heston_characteristic_function
from quantica.pricing.engines.lsm import LongstaffSchwartzEngine
from quantica.pricing.engines.merton import (
    MertonClosedFormEngine,
    MertonFFTEngine,
    merton_characteristic_function,
    merton_jump_price,
)
from quantica.pricing.engines.montecarlo import MCResult, MonteCarloEngine
from quantica.pricing.greeks import Greeks
from quantica.pricing.instruments import (
    AmericanOption,
    AsianOption,
    BarrierOption,
    EuropeanOption,
    VanillaOption,
)
from quantica.pricing.processes import (
    BlackScholesProcess,
    HestonProcess,
    Market,
    MertonProcess,
)
from quantica.pricing.volatility import implied_volatility

__all__ = [
    "DEFAULT_BOUNDS",
    "AmericanOption",
    "AnalyticEuropeanEngine",
    "AsianMonteCarloEngine",
    "AsianOption",
    "AveragingType",
    "BarrierMonteCarloEngine",
    "BarrierOption",
    "BarrierType",
    "BinomialEngine",
    "BlackScholesProcess",
    "EuropeanOption",
    "ExerciseStyle",
    "FiniteDifferenceEngine",
    "Greeks",
    "HestonCalibrationResult",
    "HestonFFTEngine",
    "HestonParams",
    "HestonProcess",
    "LongstaffSchwartzEngine",
    "MCResult",
    "Market",
    "MertonClosedFormEngine",
    "MertonFFTEngine",
    "MertonProcess",
    "MonteCarloEngine",
    "ObjectiveProfile",
    "OptionType",
    "ParamBounds",
    "VanillaOption",
    "VolQuote",
    "barrier_price",
    "calibrate_heston",
    "geometric_asian_price",
    "heston_characteristic_function",
    "implied_volatility",
    "merton_characteristic_function",
    "merton_jump_price",
    "profile_objective",
    "vol_surface_from_grid",
]
