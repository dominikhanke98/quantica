r"""Interest-rate products — swaps, caps/floors and swaptions (curve + Black-76).

The final rates step: products that price off the step-1 curve and the step-2 models. Two
kinds live here:

* **Linear** — the vanilla fixed-for-float **swap**, which is *curve-only* (no volatility).
  :func:`par_swap_rate` and :func:`swap_value` price it off the discount curve; a swap struck
  at its par rate values to zero, the tie-back to the bootstrap's self-consistency.
* **Optional** — **caps/floors** (strips of caplets/floorlets) and European **swaptions**,
  which *do* depend on volatility. Their definitions live here with the market-standard
  **Black-76** pricing (:func:`black76`); the arbitrage-free **Hull--White** pricing (bond
  options / Jamshidian) and the volatility calibration are in
  :mod:`quantica.rates.hull_white_options`.

Day-count conventions are simplified as in the rest of the pillar (year fractions are plain
time differences); the modelling content is the pricing and the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.rates.curve import DiscountCurve

__all__ = [
    "Cap",
    "Caplet",
    "Swaption",
    "black76",
    "par_swap_rate",
    "swap_value",
]


# --------------------------------------------------------------------------- #
# Linear: swaps (curve-only)
# --------------------------------------------------------------------------- #


def _swap_times(maturity: float, frequency: int) -> FloatArray:
    n = round(maturity * frequency)
    return np.arange(1, n + 1, dtype=np.float64) / frequency


def par_swap_rate(curve: DiscountCurve, maturity: float, *, frequency: int = 1) -> float:
    r"""The par (fair) fixed rate of a fixed-for-float swap: :math:`(1-P(T))/\sum_i \tau_i P(t_i)`.

    Parameters
    ----------
    curve : DiscountCurve
        The discount curve.
    maturity : float
        Swap maturity in years (a positive multiple of ``1/frequency``).
    frequency : int, optional
        Fixed-leg payments per year (default 1).

    Returns
    -------
    float
        The par swap rate.
    """
    times = _swap_times(maturity, frequency)
    annuity = float(np.sum(curve.discount_factor(times))) / frequency
    return float((1.0 - float(curve.discount_factor(maturity))) / annuity)


def swap_value(
    curve: DiscountCurve,
    maturity: float,
    fixed_rate: float,
    *,
    frequency: int = 1,
    notional: float = 1.0,
    pay_fixed: bool = True,
) -> float:
    r"""Present value of a vanilla swap off the curve.

    The receive-fixed value is :math:`N\big(\text{rate}\sum_i\tau_i P(t_i) - (1-P(T))\big)`;
    the pay-fixed value is its negative. Zero when ``fixed_rate`` equals the par swap rate.

    Parameters
    ----------
    curve : DiscountCurve
        The discount curve.
    maturity : float
        Swap maturity in years.
    fixed_rate : float
        The fixed rate paid/received.
    frequency : int, optional
        Fixed-leg payments per year (default 1).
    notional : float, optional
        Notional (default 1.0).
    pay_fixed : bool, optional
        ``True`` for a payer swap (pay fixed), ``False`` for a receiver (default ``True``).

    Returns
    -------
    float
        The swap present value.
    """
    times = _swap_times(maturity, frequency)
    annuity = float(np.sum(curve.discount_factor(times))) / frequency
    receive_fixed = fixed_rate * annuity - (1.0 - float(curve.discount_factor(maturity)))
    value = notional * receive_fixed
    return float(-value if pay_fixed else value)


# --------------------------------------------------------------------------- #
# Black-76
# --------------------------------------------------------------------------- #


def black76(
    forward: float, strike: float, vol: float, expiry: float, *, call: bool = True
) -> float:
    r"""The (undiscounted) Black-76 value on a forward.

    Returns :math:`\omega[F N(\omega d_1) - K N(\omega d_2)]`.

    Multiply by the appropriate discount factor (a caplet) or annuity (a swaption) to get the
    present value. Handles the degenerate ``vol*sqrt(expiry) = 0`` limit (intrinsic value).

    Parameters
    ----------
    forward, strike : float
        Forward level ``F`` and strike ``K``.
    vol : float
        Lognormal (Black) volatility.
    expiry : float
        Time to expiry in years.
    call : bool, optional
        ``True`` for a call/caplet/payer, ``False`` for a put/floorlet/receiver.

    Returns
    -------
    float
        The undiscounted Black-76 value.
    """
    omega = 1.0 if call else -1.0
    std = vol * np.sqrt(expiry)
    if std <= 0.0:  # no time value -> intrinsic
        return float(max(omega * (forward - strike), 0.0))
    d1 = (np.log(forward / strike) + 0.5 * std * std) / std
    d2 = d1 - std
    return float(omega * (forward * norm.cdf(omega * d1) - strike * norm.cdf(omega * d2)))


# --------------------------------------------------------------------------- #
# Caplets, caps / floors
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Caplet:
    r"""A single caplet (``is_floor=False``) or floorlet on the simple forward rate.

    Pays :math:`\tau\max(\omega(L(T_r, T_p) - K), 0)` at the payment date ``payment``, where
    :math:`L` is the simple forward over ``[reset, payment]`` fixed at ``reset`` and
    :math:`\omega=+1` for a caplet, :math:`-1` for a floorlet.

    Parameters
    ----------
    reset : float
        Fixing time in years (``> 0``).
    payment : float
        Payment time in years (``> reset``).
    strike : float
        Strike rate ``K``.
    is_floor : bool, optional
        ``True`` for a floorlet (default ``False`` -- a caplet).
    """

    reset: float
    payment: float
    strike: float
    is_floor: bool = False

    def __post_init__(self) -> None:
        """Validate the reset/payment ordering."""
        if not 0.0 < self.reset < self.payment:
            raise ValueError("require 0 < reset < payment")

    @property
    def accrual(self) -> float:
        """The accrual year fraction ``payment - reset``."""
        return self.payment - self.reset

    def forward_rate(self, curve: DiscountCurve) -> float:
        """The simple forward rate over ``[reset, payment]``."""
        return float(curve.forward_rate(self.reset, self.payment, simple=True))

    def black_price(self, curve: DiscountCurve, vol: float) -> float:
        """Black-76 present value at flat volatility ``vol``."""
        forward = self.forward_rate(curve)
        undiscounted = black76(forward, self.strike, vol, self.reset, call=not self.is_floor)
        return float(self.accrual * float(curve.discount_factor(self.payment)) * undiscounted)


@dataclass(frozen=True)
class Cap:
    r"""A cap (``is_floor=False``) or floor: a strip of caplets/floorlets on a fixed schedule.

    Covers ``[start, end]`` with ``frequency`` resets a year; caplet ``k`` spans
    ``[start + k/f, start + (k+1)/f]`` and fixes at its period start.

    Parameters
    ----------
    start : float
        First reset in years (``> 0``).
    end : float
        Final payment in years (``> start``).
    strike : float
        Common strike rate.
    frequency : int, optional
        Resets per year (default 2, i.e. semi-annual).
    is_floor : bool, optional
        ``True`` for a floor (default ``False``).
    """

    start: float
    end: float
    strike: float
    frequency: int = 2
    is_floor: bool = False

    def __post_init__(self) -> None:
        """Validate the schedule."""
        if not 0.0 < self.start < self.end:
            raise ValueError("require 0 < start < end")
        if self.frequency < 1:
            raise ValueError("frequency must be at least 1")

    @property
    def caplets(self) -> tuple[Caplet, ...]:
        """The strip of caplets/floorlets making up the cap/floor."""
        step = 1.0 / self.frequency
        n = round((self.end - self.start) * self.frequency)
        return tuple(
            Caplet(self.start + k * step, self.start + (k + 1) * step, self.strike, self.is_floor)
            for k in range(n)
        )

    def black_price(self, curve: DiscountCurve, vol: float) -> float:
        """Black-76 present value: the sum of the caplet/floorlet prices at flat ``vol``."""
        return float(sum(c.black_price(curve, vol) for c in self.caplets))


# --------------------------------------------------------------------------- #
# Swaptions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Swaption:
    r"""A European swaption on a vanilla fixed-for-float swap.

    The option, expiring at ``expiry``, is to enter a swap running ``expiry -> expiry + tenor``
    at the fixed rate ``strike``; a **payer** swaption is the option to *pay* fixed.

    Parameters
    ----------
    expiry : float
        Option expiry in years (``> 0``), also the underlying swap's start.
    tenor : float
        Length of the underlying swap in years.
    strike : float
        Fixed strike rate.
    frequency : int, optional
        Fixed-leg payments per year (default 1).
    payer : bool, optional
        ``True`` for a payer swaption (default), ``False`` for a receiver.
    """

    expiry: float
    tenor: float
    strike: float
    frequency: int = 1
    payer: bool = True

    def __post_init__(self) -> None:
        """Validate the expiry, tenor and frequency."""
        if self.expiry <= 0.0 or self.tenor <= 0.0:
            raise ValueError("expiry and tenor must be positive")
        if self.frequency < 1:
            raise ValueError("frequency must be at least 1")

    @property
    def payment_times(self) -> FloatArray:
        """The underlying swap's fixed-leg payment times (after expiry)."""
        n = round(self.tenor * self.frequency)
        return self.expiry + np.arange(1, n + 1, dtype=np.float64) / self.frequency

    def annuity(self, curve: DiscountCurve) -> float:
        r"""The forward annuity :math:`\sum_i \tau_i P(0, t_i)` of the underlying swap."""
        return float(np.sum(curve.discount_factor(self.payment_times))) / self.frequency

    def forward_swap_rate(self, curve: DiscountCurve) -> float:
        r"""The forward par swap rate :math:`(P(T_0) - P(T_n))/\text{annuity}`."""
        p_start = float(curve.discount_factor(self.expiry))
        p_end = float(curve.discount_factor(self.payment_times[-1]))
        return float((p_start - p_end) / self.annuity(curve))

    def black_price(self, curve: DiscountCurve, vol: float) -> float:
        """Black-76 present value: ``annuity * Black(S, K, vol, expiry)`` at flat ``vol``."""
        forward = self.forward_swap_rate(curve)
        undiscounted = black76(forward, self.strike, vol, self.expiry, call=self.payer)
        return float(self.annuity(curve) * undiscounted)
