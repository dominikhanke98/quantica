"""Option contracts (payoffs).

An instrument knows *what* it pays, not *how* it is priced. Pricing is
delegated to a :class:`~quantica.pricing.engines.PricingEngine` attached via
:meth:`EuropeanOption.set_engine`; the same instrument can be re-priced by any
engine (CLAUDE.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from quantica.core.types import FloatLike, OptionType

if TYPE_CHECKING:
    from quantica.pricing.engines import PricingEngine
    from quantica.pricing.processes import BlackScholesProcess


@dataclass
class EuropeanOption:
    """A vanilla European option, exercisable only at expiry.

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

    def payoff(self, spot: FloatLike) -> FloatLike:
        r"""Terminal payoff for a given underlying price at expiry.

        Computes :math:`\max(\omega (S_T - K), 0)` where :math:`\omega` is
        ``+1`` for a call and ``-1`` for a put.

        Parameters
        ----------
        spot : float or ndarray
            Underlying price(s) at expiry, :math:`S_T`.

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

    def set_engine(self, engine: PricingEngine) -> EuropeanOption:
        """Attach a pricing engine and return ``self`` for chaining."""
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
