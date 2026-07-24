r"""Curve-building instruments — deposits (short end) and par swaps (long end).

The market quotes a curve through instruments, and a bootstrap is the process of finding the
discount factors that reprice every one of them to par. Each instrument here exposes
:meth:`value` — its present value off a given curve — so "prices to par" means ``value ≈ 0``,
the self-consistency anchor the bootstrap is validated against.

Conventions are deliberately simplified (the modelling content is the bootstrap and the
interpolation, not the calendar): year fractions are plain time differences in years, and the
swap is a single-curve vanilla fixed-for-float where the projected float leg discounts to
:math:`P(0,t_{\text{start}}) - P(0,t_{\text{end}})`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.rates.curve import DiscountCurve

__all__ = ["Deposit", "RateInstrument", "Swap"]


@runtime_checkable
class RateInstrument(Protocol):
    """A market instrument used to build a curve; ``value(curve) == 0`` means priced to par."""

    @property
    def maturity(self) -> float:
        """The instrument's final maturity in years (its bootstrap pillar)."""
        ...

    def value(self, curve: DiscountCurve) -> float:
        """Present value off ``curve`` (zero at par)."""
        ...


@dataclass(frozen=True)
class Deposit:
    r"""A simple-compounded money-market deposit maturing at ``maturity``.

    Lend 1 today, receive :math:`1 + r\,\tau` at maturity, so the fair (par) discount factor
    is :math:`P(0,T) = 1/(1 + r\,\tau)` — the short-end pillars.

    Parameters
    ----------
    maturity : float
        Maturity in years (must be positive).
    rate : float
        The simple deposit rate (decimal).
    year_fraction : float, optional
        Accrual year fraction; defaults to ``maturity``.
    """

    maturity: float
    rate: float
    year_fraction: float | None = None

    def __post_init__(self) -> None:
        """Validate the maturity."""
        if self.maturity <= 0.0:
            raise ValueError(f"maturity must be positive, got {self.maturity}")

    @property
    def accrual(self) -> float:
        """The accrual year fraction used (``year_fraction`` or ``maturity``)."""
        return self.maturity if self.year_fraction is None else self.year_fraction

    def par_discount_factor(self) -> float:
        r"""The discount factor that prices the deposit to par, :math:`1/(1 + r\,\tau)`."""
        return 1.0 / (1.0 + self.rate * self.accrual)

    def value(self, curve: DiscountCurve) -> float:
        """Present value ``(1 + r*tau) * P(T) - 1`` (zero at par)."""
        p_t = float(curve.discount_factor(self.maturity))
        return (1.0 + self.rate * self.accrual) * p_t - 1.0


@dataclass(frozen=True)
class Swap:
    r"""A par vanilla fixed-for-float interest-rate swap.

    The fixed leg pays ``rate`` on ``frequency`` dates a year to ``maturity``; the single-curve
    float leg has present value :math:`1 - P(0,T)`. The swap value (receive-fixed) is

    .. math:: V = \text{rate}\sum_i \tau_i\,P(0,t_i) - \big(1 - P(0,T)\big),

    which is zero at the par rate — the long-end pillars.

    Parameters
    ----------
    maturity : float
        Final maturity in years (must be a positive multiple of ``1/frequency``).
    rate : float
        The fixed (par-swap) rate (decimal).
    frequency : int, optional
        Fixed-leg payments per year (default 1, i.e. annual).
    """

    maturity: float
    rate: float
    frequency: int = 1

    def __post_init__(self) -> None:
        """Validate the maturity and payment frequency."""
        if self.frequency < 1:
            raise ValueError(f"frequency must be at least 1, got {self.frequency}")
        n = self.maturity * self.frequency
        if self.maturity <= 0.0 or abs(n - round(n)) > 1e-9:
            raise ValueError("maturity must be a positive multiple of 1/frequency")

    @property
    def payment_times(self) -> FloatArray:
        """The fixed-leg payment times ``1/f, 2/f, ..., maturity`` (years)."""
        n = round(self.maturity * self.frequency)
        return np.asarray(np.arange(1, n + 1, dtype=np.float64) / self.frequency)

    def annuity(self, curve: DiscountCurve) -> float:
        r"""The fixed-leg annuity :math:`\sum_i \tau_i P(0,t_i)`."""
        tau = 1.0 / self.frequency
        return float(tau * np.sum(curve.discount_factor(self.payment_times)))

    def value(self, curve: DiscountCurve) -> float:
        """Receive-fixed present value ``rate*annuity - (1 - P(T))`` (zero at par)."""
        p_t = float(curve.discount_factor(self.maturity))
        return self.rate * self.annuity(curve) - (1.0 - p_t)
