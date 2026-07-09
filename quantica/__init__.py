"""quantica — a tested and validated quantitative-finance library.

Phase 1 exposes the pricing primitives: the option contract
(:class:`~quantica.pricing.instruments.EuropeanOption`), the market dynamics
(:class:`~quantica.pricing.processes.BlackScholesProcess`), and the shared
enums (:class:`~quantica.core.types.OptionType`). Numerical engines are added
in later steps and attached to an instrument via ``set_engine``.
"""

from __future__ import annotations

from quantica.core.types import OptionType
from quantica.pricing.engines.analytic import AnalyticEuropeanEngine
from quantica.pricing.greeks import Greeks
from quantica.pricing.instruments import EuropeanOption
from quantica.pricing.processes import BlackScholesProcess
from quantica.pricing.volatility import implied_volatility

__all__ = [
    "AnalyticEuropeanEngine",
    "BlackScholesProcess",
    "EuropeanOption",
    "Greeks",
    "OptionType",
    "implied_volatility",
]

__version__ = "0.1.0"
