"""Pricing subpackage: instruments, processes, and numerical engines.

The design follows the Instrument / Process / Engine separation (CLAUDE.md §4):
an *instrument* is the contract, a *process* is the market dynamics, and an
*engine* is a numerical method that prices an instrument under a process.
"""

from __future__ import annotations

from quantica.core.types import AveragingType, BarrierType, ExerciseStyle, OptionType
from quantica.pricing.engines.analytic import AnalyticEuropeanEngine
from quantica.pricing.engines.asian import AsianMonteCarloEngine, geometric_asian_price
from quantica.pricing.engines.barrier import BarrierMonteCarloEngine, barrier_price
from quantica.pricing.engines.binomial import BinomialEngine
from quantica.pricing.engines.finitediff import FiniteDifferenceEngine
from quantica.pricing.engines.heston import HestonFFTEngine, heston_characteristic_function
from quantica.pricing.engines.lsm import LongstaffSchwartzEngine
from quantica.pricing.engines.montecarlo import MCResult, MonteCarloEngine
from quantica.pricing.greeks import Greeks
from quantica.pricing.instruments import (
    AmericanOption,
    AsianOption,
    BarrierOption,
    EuropeanOption,
    VanillaOption,
)
from quantica.pricing.processes import BlackScholesProcess, HestonProcess, Market
from quantica.pricing.volatility import implied_volatility

__all__ = [
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
    "HestonFFTEngine",
    "HestonProcess",
    "LongstaffSchwartzEngine",
    "MCResult",
    "Market",
    "MonteCarloEngine",
    "OptionType",
    "VanillaOption",
    "barrier_price",
    "geometric_asian_price",
    "heston_characteristic_function",
    "implied_volatility",
]
