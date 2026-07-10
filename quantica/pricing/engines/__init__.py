"""Numerical pricing engines.

This module defines only the *interface* an engine must satisfy. Concrete
engines (analytic Black--Scholes, binomial tree, Monte Carlo, Crank--Nicolson
PDE) are added in later steps of Phase 1 (CLAUDE.md §8) and register themselves
here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from quantica.pricing.greeks import Greeks
    from quantica.pricing.instruments import VanillaOption
    from quantica.pricing.processes import BlackScholesProcess


@runtime_checkable
class PricingEngine(Protocol):
    """Protocol for a numerical method that prices an instrument.

    An engine is a small, stateless strategy object: it takes an instrument
    and the market process and returns a present value. Keeping engines behind
    this Protocol is what lets a single instrument be re-priced by swapping the
    engine, which in turn makes cross-method convergence tests natural
    (CLAUDE.md §4).
    """

    def calculate(
        self,
        instrument: VanillaOption,
        process: BlackScholesProcess,
    ) -> float:
        """Return the present value of ``instrument`` under ``process``."""
        ...


@runtime_checkable
class GreeksEngine(PricingEngine, Protocol):
    """A pricing engine that can also report first-order sensitivities.

    Kept separate from :class:`PricingEngine` because computing Greeks is a
    distinct capability: some engines (e.g. a bare Monte Carlo price) price
    without providing analytic sensitivities. An instrument's ``greeks`` method
    requires the attached engine to satisfy this narrower Protocol.
    """

    def greeks(
        self,
        instrument: VanillaOption,
        process: BlackScholesProcess,
    ) -> Greeks:
        """Return the Greeks of ``instrument`` under ``process``."""
        ...


__all__ = ["GreeksEngine", "PricingEngine"]
