"""Pricing subpackage: instruments, processes, and numerical engines.

The design follows the Instrument / Process / Engine separation (CLAUDE.md §4):
an *instrument* is the contract, a *process* is the market dynamics, and an
*engine* is a numerical method that prices an instrument under a process.
"""

from __future__ import annotations

from quantica.core.types import ExerciseStyle, OptionType
from quantica.pricing.engines.analytic import AnalyticEuropeanEngine
from quantica.pricing.engines.binomial import BinomialEngine
from quantica.pricing.engines.finitediff import FiniteDifferenceEngine
from quantica.pricing.engines.lsm import LongstaffSchwartzEngine
from quantica.pricing.engines.montecarlo import MCResult, MonteCarloEngine
from quantica.pricing.greeks import Greeks
from quantica.pricing.instruments import AmericanOption, EuropeanOption, VanillaOption
from quantica.pricing.processes import BlackScholesProcess
from quantica.pricing.volatility import implied_volatility

__all__ = [
    "AmericanOption",
    "AnalyticEuropeanEngine",
    "BinomialEngine",
    "BlackScholesProcess",
    "EuropeanOption",
    "ExerciseStyle",
    "FiniteDifferenceEngine",
    "Greeks",
    "LongstaffSchwartzEngine",
    "MCResult",
    "MonteCarloEngine",
    "OptionType",
    "VanillaOption",
    "implied_volatility",
]
