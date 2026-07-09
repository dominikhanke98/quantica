"""Tests for the pricing-engine interface.

No concrete engine exists yet (they arrive in later steps of Phase 1); this
pins down the Protocol so a class must expose ``calculate(instrument, process)``
to count as an engine.
"""

from __future__ import annotations

from quantica.pricing.engines import PricingEngine
from quantica.pricing.instruments import EuropeanOption
from quantica.pricing.processes import BlackScholesProcess


def test_engine_with_calculate_satisfies_protocol() -> None:
    class GoodEngine:
        def calculate(self, instrument: EuropeanOption, process: BlackScholesProcess) -> float:
            return 0.0

    assert isinstance(GoodEngine(), PricingEngine)


def test_engine_without_calculate_does_not_satisfy_protocol() -> None:
    class NotAnEngine:
        def price(self) -> float:
            return 0.0

    assert not isinstance(NotAnEngine(), PricingEngine)
