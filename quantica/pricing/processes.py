"""Market dynamics for the underlying.

A *process* holds the current market state and the model parameters that
describe how the underlying evolves. It knows nothing about any particular
contract or numerical method.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class BlackScholesProcess:
    r"""Geometric Brownian motion under the risk-neutral measure.

    The spot follows

    .. math::

        dS_t = (r - q)\, S_t\, dt + \sigma\, S_t\, dW_t,

    with constant risk-free rate :math:`r`, continuous dividend yield
    :math:`q`, and volatility :math:`\sigma` (Black--Scholes--Merton, 1973).

    Parameters
    ----------
    spot : float
        Current underlying price :math:`S_0`. Must be positive.
    rate : float
        Continuously-compounded risk-free rate :math:`r`. May be negative.
    div : float, optional
        Continuous dividend yield :math:`q`. May be negative; defaults to 0.
    vol : float
        Volatility :math:`\sigma`, annualised. Must be non-negative;
        ``vol == 0`` is the deterministic (zero-diffusion) limit.

    Notes
    -----
    The object is frozen (immutable): a change in market data is a *new*
    process, which keeps pricing calls referentially transparent and safe to
    reuse across engines.
    """

    spot: float
    rate: float
    vol: float
    div: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.vol < 0.0:
            raise ValueError(f"vol must be non-negative, got {self.vol}")

    def discount_factor(self, t: float) -> float:
        r"""Risk-free discount factor :math:`e^{-r t}` to time ``t``."""
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        return float(np.exp(-self.rate * t))

    def forward(self, t: float) -> float:
        r"""Forward price :math:`S_0 e^{(r - q) t}` for delivery at ``t``."""
        if t < 0.0:
            raise ValueError(f"t must be non-negative, got {t}")
        return float(self.spot * np.exp((self.rate - self.div) * t))

    # -- bumped copies (support bump-and-reval Greek validation) ------------- #
    # The process is frozen, so a bump returns a *new* process. These make the
    # finite-difference Greek checks read cleanly, e.g. ``proc.with_spot(s)``.

    def with_spot(self, spot: float) -> BlackScholesProcess:
        """Return a copy with the spot replaced by ``spot``."""
        return replace(self, spot=spot)

    def with_vol(self, vol: float) -> BlackScholesProcess:
        """Return a copy with the volatility replaced by ``vol``."""
        return replace(self, vol=vol)

    def with_rate(self, rate: float) -> BlackScholesProcess:
        """Return a copy with the risk-free rate replaced by ``rate``."""
        return replace(self, rate=rate)

    def with_div(self, div: float) -> BlackScholesProcess:
        """Return a copy with the dividend yield replaced by ``div``."""
        return replace(self, div=div)
