"""Market data and the processes that model the underlying's dynamics.

A :class:`Market` is the model-independent state (spot, rate, dividend). A
*process* composes a market with the parameters of a particular vol/variance
model — :class:`BlackScholesProcess` (constant vol) or :class:`HestonProcess`
(stochastic variance). Splitting the two lets a volatility-model-free routine
such as :func:`~quantica.pricing.volatility.implied_volatility` take just a
``Market``, and lets every process share one carrier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Market:
    r"""Model-independent market state: spot, risk-free rate, dividend yield.

    Parameters
    ----------
    spot : float
        Current underlying price :math:`S_0`. Must be positive.
    rate : float
        Continuously-compounded risk-free rate :math:`r`. May be negative.
    div : float, optional
        Continuous dividend yield :math:`q`. May be negative; defaults to 0.

    Notes
    -----
    Frozen (immutable): a change in market data is a *new* ``Market``, which
    keeps pricing referentially transparent.
    """

    spot: float
    rate: float
    div: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError(f"spot must be positive, got {self.spot}")

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

    def with_spot(self, spot: float) -> Market:
        """Return a copy with the spot replaced by ``spot``."""
        return replace(self, spot=spot)

    def with_rate(self, rate: float) -> Market:
        """Return a copy with the risk-free rate replaced by ``rate``."""
        return replace(self, rate=rate)

    def with_div(self, div: float) -> Market:
        """Return a copy with the dividend yield replaced by ``div``."""
        return replace(self, div=div)


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
    Kept as a flat record (``spot, rate, vol, div``) for the constant-vol case;
    :attr:`market` exposes the shared :class:`Market` carrier and
    :meth:`from_market` builds one from a market plus a vol. Frozen (immutable).
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

    @property
    def market(self) -> Market:
        """The model-independent :class:`Market` carrier (spot, rate, div)."""
        return Market(spot=self.spot, rate=self.rate, div=self.div)

    @classmethod
    def from_market(cls, market: Market, vol: float) -> BlackScholesProcess:
        """Build a process from a :class:`Market` plus a volatility."""
        return cls(spot=market.spot, rate=market.rate, vol=vol, div=market.div)

    def discount_factor(self, t: float) -> float:
        r"""Risk-free discount factor :math:`e^{-r t}` to time ``t``."""
        return self.market.discount_factor(t)

    def forward(self, t: float) -> float:
        r"""Forward price :math:`S_0 e^{(r - q) t}` for delivery at ``t``."""
        return self.market.forward(t)

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


@dataclass(frozen=True)
class HestonProcess:
    r"""Heston (1993) stochastic-variance model under the risk-neutral measure.

    .. math::

        dS_t &= (r - q)\, S_t\, dt + \sqrt{v_t}\, S_t\, dW_t^{(1)}, \\
        dv_t &= \kappa (\theta - v_t)\, dt + \xi \sqrt{v_t}\, dW_t^{(2)},
        \qquad d\langle W^{(1)}, W^{(2)} \rangle_t = \rho\, dt.

    The variance is a CIR (square-root) process, so it is mean-reverting and
    non-negative.

    Parameters
    ----------
    spot : float
        Current underlying price :math:`S_0`. Must be positive.
    rate : float
        Continuously-compounded risk-free rate :math:`r`.
    v0 : float
        Initial variance :math:`v_0` (units of variance, i.e. vol squared). ``>= 0``.
    kappa : float
        Mean-reversion speed :math:`\kappa`. ``>= 0``.
    theta : float
        Long-run variance :math:`\theta`. ``>= 0``.
    xi : float
        Volatility of variance :math:`\xi` ("vol of vol"). ``>= 0``; ``xi == 0``
        is the deterministic-variance (Black--Scholes) limit.
    rho : float
        Correlation :math:`\rho` between the spot and variance Brownian motions,
        in ``[-1, 1]``.
    div : float, optional
        Continuous dividend yield :math:`q`. Defaults to 0.

    Notes
    -----
    The **Feller condition** :math:`2\kappa\theta \ge \xi^2` (see
    :attr:`feller_satisfied`) guarantees the variance never reaches zero; a fit
    that violates it is still valid but the variance can touch zero.
    """

    spot: float
    rate: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float
    div: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0.0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.v0 < 0.0:
            raise ValueError(f"v0 must be non-negative, got {self.v0}")
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be non-negative, got {self.kappa}")
        if self.theta < 0.0:
            raise ValueError(f"theta must be non-negative, got {self.theta}")
        if self.xi < 0.0:
            raise ValueError(f"xi must be non-negative, got {self.xi}")
        if not -1.0 <= self.rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    @property
    def market(self) -> Market:
        """The model-independent :class:`Market` carrier (spot, rate, div)."""
        return Market(spot=self.spot, rate=self.rate, div=self.div)

    @classmethod
    def from_market(
        cls,
        market: Market,
        *,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
    ) -> HestonProcess:
        """Build a process from a :class:`Market` plus the Heston parameters."""
        return cls(
            spot=market.spot,
            rate=market.rate,
            v0=v0,
            kappa=kappa,
            theta=theta,
            xi=xi,
            rho=rho,
            div=market.div,
        )

    @property
    def feller_satisfied(self) -> bool:
        r"""Whether the Feller condition :math:`2\kappa\theta \ge \xi^2` holds."""
        return 2.0 * self.kappa * self.theta >= self.xi * self.xi

    def discount_factor(self, t: float) -> float:
        r"""Risk-free discount factor :math:`e^{-r t}` to time ``t``."""
        return self.market.discount_factor(t)

    def forward(self, t: float) -> float:
        r"""Forward price :math:`S_0 e^{(r - q) t}` for delivery at ``t``."""
        return self.market.forward(t)
