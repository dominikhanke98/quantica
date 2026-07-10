"""quantica — a tested and validated quantitative-finance library.

Phase 1 exposes the pricing primitives: the option contract
(:class:`~quantica.pricing.instruments.EuropeanOption`), the market dynamics
(:class:`~quantica.pricing.processes.BlackScholesProcess`), and the shared
enums (:class:`~quantica.core.types.OptionType`). Numerical engines are added
in later steps and attached to an instrument via ``set_engine``.
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

__version__ = "0.1.0"
