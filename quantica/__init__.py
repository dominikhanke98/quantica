"""quantica — a tested and validated quantitative-finance library.

Phase 1 exposes the pricing primitives: the option contract
(:class:`~quantica.pricing.instruments.EuropeanOption`), the market dynamics
(:class:`~quantica.pricing.processes.BlackScholesProcess`), and the shared
enums (:class:`~quantica.core.types.OptionType`). Numerical engines are added
in later steps and attached to an instrument via ``set_engine``.
"""

from __future__ import annotations

from quantica.core.types import OptionType
from quantica.pricing.instruments import EuropeanOption
from quantica.pricing.processes import BlackScholesProcess

__all__ = [
    "BlackScholesProcess",
    "EuropeanOption",
    "OptionType",
]

__version__ = "0.1.0"
