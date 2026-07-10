"""Option contracts (payoffs).

An instrument knows *what* it pays, not *how* it is priced. Pricing is delegated
to a :class:`~quantica.pricing.engines.PricingEngine` attached via
:meth:`VanillaOption.set_engine`; the same instrument can be re-priced by any
engine (CLAUDE.md §4).

The vanilla payoff and the engine seam are shared in :class:`VanillaOption`; the
concrete :class:`EuropeanOption` and :class:`AmericanOption` differ only in their
exercise right, which is the single fact an engine branches on to decide whether
early exercise is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Self

import numpy as np

from quantica.core.types import ExerciseStyle, FloatLike, OptionType

if TYPE_CHECKING:
    from quantica.pricing.engines import PricingEngine
    from quantica.pricing.greeks import Greeks
    from quantica.pricing.processes import BlackScholesProcess


@dataclass
class VanillaOption:
    """Base for a vanilla option — strike, expiry, call/put, and an engine seam.

    Not used directly; instantiate :class:`EuropeanOption` or
    :class:`AmericanOption`, which fix :attr:`exercise`.

    Parameters
    ----------
    strike : float
        Strike price ``K``. Must be positive.
    expiry : float
        Time to expiry ``T`` in years. Must be non-negative; ``T == 0`` is a
        valid degenerate contract that pays its intrinsic value immediately.
    option_type : OptionType
        Call or put.

    Notes
    -----
    The engine is intentionally *not* a constructor argument: it is set after
    construction so a single contract can be priced by several methods in turn
    (see :meth:`set_engine`). ``npv`` raises until an engine is attached.
    """

    strike: float
    expiry: float
    option_type: OptionType
    _engine: PricingEngine | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.expiry < 0.0:
            raise ValueError(f"expiry must be non-negative, got {self.expiry}")

    @property
    def exercise(self) -> ExerciseStyle:
        """When the option may be exercised (fixed by the concrete subclass)."""
        raise NotImplementedError("instantiate EuropeanOption or AmericanOption")

    def payoff(self, spot: FloatLike) -> FloatLike:
        r"""Intrinsic payoff at a given underlying price.

        Computes :math:`\max(\omega (S - K), 0)` where :math:`\omega` is ``+1``
        for a call and ``-1`` for a put. This is the terminal payoff for a
        European option and the immediate-exercise value at any time for an
        American one.

        Parameters
        ----------
        spot : float or ndarray
            Underlying price(s) :math:`S`.

        Returns
        -------
        float or ndarray
            Payoff, matching the shape of ``spot``.
        """
        intrinsic = self.option_type.sign * (np.asarray(spot, dtype=np.float64) - self.strike)
        payoff = np.maximum(intrinsic, 0.0)
        # Preserve scalar-in / scalar-out ergonomics.
        if np.isscalar(spot) or np.ndim(spot) == 0:
            return float(payoff)
        return payoff

    def set_engine(self, engine: PricingEngine) -> Self:
        """Attach a pricing engine and return ``self`` (of the same type) for chaining."""
        self._engine = engine
        return self

    def npv(self, process: BlackScholesProcess) -> float:
        """Present value under ``process`` using the attached engine.

        Raises
        ------
        RuntimeError
            If no engine has been attached via :meth:`set_engine`.
        """
        if self._engine is None:
            raise RuntimeError("no pricing engine attached; call set_engine(...) before npv()")
        return self._engine.calculate(self, process)

    def greeks(self, process: BlackScholesProcess) -> Greeks:
        """First-order sensitivities under ``process`` via the attached engine.

        Raises
        ------
        RuntimeError
            If no engine has been attached, or the attached engine does not
            support Greeks (i.e. is not a
            :class:`~quantica.pricing.engines.GreeksEngine`).
        """
        # Imported here to keep the runtime isinstance check local; the type is
        # only needed for the capability check, not the class definition.
        from quantica.pricing.engines import GreeksEngine

        if self._engine is None:
            raise RuntimeError("no pricing engine attached; call set_engine(...) before greeks()")
        if not isinstance(self._engine, GreeksEngine):
            raise RuntimeError(
                f"attached engine {type(self._engine).__name__} does not support Greeks"
            )
        return self._engine.greeks(self, process)


@dataclass
class EuropeanOption(VanillaOption):
    """A vanilla option exercisable only at expiry."""

    @property
    def exercise(self) -> ExerciseStyle:
        return ExerciseStyle.EUROPEAN


@dataclass
class AmericanOption(VanillaOption):
    """A vanilla option exercisable at any time up to expiry.

    Has no closed form; price it with a lattice (:class:`BinomialEngine`) or a
    PDE solve of the free-boundary problem (:class:`FiniteDifferenceEngine`).
    """

    @property
    def exercise(self) -> ExerciseStyle:
        return ExerciseStyle.AMERICAN
