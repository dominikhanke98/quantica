"""Pricing subpackage: instruments, processes, and numerical engines.

The design follows the Instrument / Process / Engine separation (CLAUDE.md §4):
an *instrument* is the contract, a *process* is the market dynamics, and an
*engine* is a numerical method that prices an instrument under a process.
"""

from __future__ import annotations

from quantica.core.types import OptionType
from quantica.pricing.engines.analytic import AnalyticEuropeanEngine
from quantica.pricing.greeks import Greeks
from quantica.pricing.instruments import EuropeanOption
from quantica.pricing.processes import BlackScholesProcess

__all__ = [
    "AnalyticEuropeanEngine",
    "BlackScholesProcess",
    "EuropeanOption",
    "Greeks",
    "OptionType",
]
