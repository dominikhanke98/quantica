r"""One-factor short-rate models — Vasicek, CIR and Hull--White.

The curve (step 1) is a static snapshot of *today's* discount factors; a short-rate model is a
*dynamic* description of how the whole curve can evolve, driven by a single stochastic factor
:math:`r_t`. All three models here are **affine**: the zero-coupon bond has the closed form

.. math:: P(t,T) = A(t,T)\,e^{-B(t,T)\,r_t},

so bonds (and, later, options on them) price analytically. The three span the standard
trade-offs:

* :class:`Vasicek` — Gaussian Ornstein--Uhlenbeck, ``dr = a(b-r)dt + sigma dW``. Mean-reverting and
  analytically the simplest, but the Gaussian law lets rates go **negative**.
* :class:`CIR` — the square-root diffusion ``dr = a(b-r)dt + sigma*sqrt(r) dW``. Non-negative
  when the **Feller condition** :math:`2ab \ge \sigma^2` holds (else the rate can touch zero) —
  the same square-root process, and the same honesty, as the Heston variance in the pricing pillar.
* :class:`HullWhite` — Vasicek with a *time-dependent* drift ``dr = (theta(t) - a r)dt + sigma dW``
  whose ``theta(t)`` is chosen to reproduce the initial term structure **exactly**. That exact fit
  is why it, not Vasicek/CIR, is used for anything that must be arbitrage-free to the current curve.

Simulation uses the **exact** transition law of each model (Gaussian for Vasicek/Hull--White,
non-central chi-squared for CIR), not an Euler discretisation — the marginals then carry no
time-stepping bias, and the analytic bond price is cross-checked against the Monte Carlo
estimate of :math:`\mathbb E[e^{-\int_0^T r\,ds}]`.

References
----------
Vasicek, O. (1977). "An equilibrium characterization of the term structure", *JFE* 5.
Cox, J., Ingersoll, J. & Ross, S. (1985). "A theory of the term structure of interest rates",
*Econometrica* 53.
Hull, J. & White, A. (1990). "Pricing interest-rate-derivative securities", *RFS* 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray, FloatLike
    from quantica.rates.curve import DiscountCurve

__all__ = ["CIR", "HullWhite", "ShortRateModel", "Vasicek", "monte_carlo_discount"]


class ShortRateModel(ABC):
    """Common interface for a one-factor short-rate model.

    Concrete models expose the analytic zero-coupon bond price :meth:`zero_coupon_bond`, the
    present-value discount bond :meth:`discount_bond` (``P(0, T)`` from the initial rate), and
    exact-transition path simulation :meth:`simulate`.
    """

    r0: float

    @abstractmethod
    def zero_coupon_bond(self, t: float, maturity: FloatLike, rate: FloatLike) -> FloatArray:
        r"""The bond price :math:`P(t,T) = A(t,T)e^{-B(t,T)r_t}` given the short rate at ``t``."""

    def discount_bond(self, maturity: FloatLike) -> FloatArray:
        r"""The present-value discount factor :math:`P(0, T)` implied by the model."""
        t_arr = np.asarray(maturity, dtype=np.float64)
        return np.asarray(self.zero_coupon_bond(0.0, t_arr, self.r0), dtype=np.float64)

    @abstractmethod
    def simulate(self, times: FloatArray, n_paths: int, *, rng: np.random.Generator) -> FloatArray:
        """Simulate short-rate paths on ``times`` (shape ``(n_paths, len(times))``, exact law)."""


# --------------------------------------------------------------------------- #
# Vasicek
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Vasicek(ShortRateModel):
    r"""The Vasicek Gaussian short-rate model ``dr = a(b - r)dt + sigma dW``.

    Parameters
    ----------
    a : float
        Mean-reversion speed (``> 0``).
    b : float
        Long-run mean level.
    sigma : float
        Volatility (``>= 0``).
    r0 : float
        Initial short rate.
    """

    a: float
    b: float
    sigma: float
    r0: float

    def __post_init__(self) -> None:
        """Validate the mean-reversion speed and volatility."""
        if self.a <= 0.0:
            raise ValueError(f"a must be positive, got {self.a}")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")

    def _b_factor(self, tau: FloatArray) -> FloatArray:
        return np.asarray((1.0 - np.exp(-self.a * tau)) / self.a, dtype=np.float64)

    def zero_coupon_bond(self, t: float, maturity: FloatLike, rate: FloatLike) -> FloatArray:
        r"""Analytic Vasicek bond price :math:`P(t,T)`."""
        tau = np.asarray(maturity, dtype=np.float64) - t
        big_b = self._b_factor(tau)
        a, sig = self.a, self.sigma
        log_a = (big_b - tau) * (a * a * self.b - 0.5 * sig * sig) / (a * a) - (
            sig * sig * big_b * big_b
        ) / (4.0 * a)
        return np.asarray(np.exp(log_a) * np.exp(-big_b * np.asarray(rate)), dtype=np.float64)

    def simulate(self, times: FloatArray, n_paths: int, *, rng: np.random.Generator) -> FloatArray:
        """Exact Gaussian-transition simulation of the Vasicek short rate."""
        return _simulate_gaussian_ou(
            times, n_paths, rng=rng, a=self.a, sigma=self.sigma, x0=self.r0, mean=lambda _t: self.b
        )


# --------------------------------------------------------------------------- #
# CIR
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CIR(ShortRateModel):
    r"""The Cox--Ingersoll--Ross square-root model ``dr = a(b - r)dt + sigma*sqrt(r) dW``.

    Parameters
    ----------
    a : float
        Mean-reversion speed (``> 0``).
    b : float
        Long-run mean level (``> 0``).
    sigma : float
        Volatility (``>= 0``).
    r0 : float
        Initial short rate (``>= 0``).
    """

    a: float
    b: float
    sigma: float
    r0: float

    def __post_init__(self) -> None:
        """Validate the parameters (``a``/``b`` positive, ``sigma``/``r0`` non-negative)."""
        if self.a <= 0.0:
            raise ValueError(f"a must be positive, got {self.a}")
        if self.b < 0.0 or self.sigma < 0.0 or self.r0 < 0.0:
            raise ValueError("b, sigma and r0 must be non-negative")

    @property
    def feller_satisfied(self) -> bool:
        r"""Whether the Feller condition :math:`2ab \ge \sigma^2` holds (rate stays positive)."""
        return bool(2.0 * self.a * self.b >= self.sigma * self.sigma)

    def zero_coupon_bond(self, t: float, maturity: FloatLike, rate: FloatLike) -> FloatArray:
        r"""Analytic CIR bond price :math:`P(t,T)`."""
        tau = np.asarray(maturity, dtype=np.float64) - t
        a, b, sig = self.a, self.b, self.sigma
        gamma = np.sqrt(a * a + 2.0 * sig * sig)
        exp_gt = np.exp(gamma * tau)
        denom = (gamma + a) * (exp_gt - 1.0) + 2.0 * gamma
        big_b = 2.0 * (exp_gt - 1.0) / denom
        big_a = (2.0 * gamma * np.exp((a + gamma) * tau / 2.0) / denom) ** (
            2.0 * a * b / (sig * sig)
        )
        return np.asarray(big_a * np.exp(-big_b * np.asarray(rate)), dtype=np.float64)

    def simulate(self, times: FloatArray, n_paths: int, *, rng: np.random.Generator) -> FloatArray:
        """Exact non-central chi-squared transition simulation of the CIR short rate."""
        t = np.asarray(times, dtype=np.float64)
        a, sig = self.a, self.sigma
        df = 4.0 * self.a * self.b / (sig * sig)
        paths = np.empty((n_paths, t.size), dtype=np.float64)
        paths[:, 0] = self.r0
        for i in range(1, t.size):
            dt = t[i] - t[i - 1]
            decay = np.exp(-a * dt)
            c = 2.0 * a / (sig * sig * (1.0 - decay))
            nc = 2.0 * c * paths[:, i - 1] * decay  # non-centrality
            paths[:, i] = rng.noncentral_chisquare(df, nc) / (2.0 * c)
        return paths


# --------------------------------------------------------------------------- #
# Hull--White (extended Vasicek, fitted to the initial curve)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HullWhite(ShortRateModel):
    r"""Hull--White ``dr = (theta(t) - a r)dt + sigma dW``, fitted to an initial discount curve.

    The drift ``theta(t)`` is implied by the market curve so that the model reprices the initial
    term structure **exactly** for any ``a``/``sigma``; those two are volatility parameters (to be
    calibrated to caps/swaptions in a later step). Build with :meth:`from_curve`.

    Parameters
    ----------
    a : float
        Mean-reversion speed (``> 0``).
    sigma : float
        Volatility (``>= 0``).
    curve : DiscountCurve
        The initial discount curve the model reproduces exactly.
    """

    a: float
    sigma: float
    curve: DiscountCurve

    def __post_init__(self) -> None:
        """Validate the mean-reversion speed and volatility; set ``r0`` from the curve."""
        if self.a <= 0.0:
            raise ValueError(f"a must be positive, got {self.a}")
        if self.sigma < 0.0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")
        object.__setattr__(self, "r0", float(self.curve.instantaneous_forward(_SHORT_END)))

    @classmethod
    def from_curve(cls, curve: DiscountCurve, *, a: float, sigma: float) -> HullWhite:
        """Build a Hull--White model fitting ``curve`` exactly, with volatility ``a``/``sigma``."""
        return cls(a=a, sigma=sigma, curve=curve)

    def _b_factor(self, tau: FloatArray) -> FloatArray:
        return np.asarray((1.0 - np.exp(-self.a * tau)) / self.a, dtype=np.float64)

    def zero_coupon_bond(self, t: float, maturity: FloatLike, rate: FloatLike) -> FloatArray:
        r"""Analytic Hull--White bond price :math:`P(t,T)` consistent with the fitted curve."""
        big_t = np.asarray(maturity, dtype=np.float64)
        tau = big_t - t
        big_b = self._b_factor(tau)
        pm_t = self.curve.discount_factor(t)
        pm_big_t = self.curve.discount_factor(big_t)
        fwd_t = self.curve.instantaneous_forward(t)
        adj = (
            big_b * fwd_t
            - (self.sigma * self.sigma)
            / (4.0 * self.a)
            * (1.0 - np.exp(-2.0 * self.a * t))
            * big_b
            * big_b
        )
        return np.asarray(
            (pm_big_t / pm_t) * np.exp(adj) * np.exp(-big_b * np.asarray(rate)), dtype=np.float64
        )

    def _alpha(self, t: FloatArray) -> FloatArray:
        r"""The deterministic shift :math:`\alpha(t) = f^M(0,t) + \sigma^2/(2a^2)(1-e^{-at})^2`."""
        fwd = self.curve.instantaneous_forward(t)
        return np.asarray(
            fwd
            + (self.sigma * self.sigma)
            / (2.0 * self.a * self.a)
            * (1.0 - np.exp(-self.a * t)) ** 2,
            dtype=np.float64,
        )

    def simulate(self, times: FloatArray, n_paths: int, *, rng: np.random.Generator) -> FloatArray:
        """Exact simulation: ``r_t = x_t + alpha(t)`` where ``x`` is a mean-zero Gaussian OU."""
        t = np.asarray(times, dtype=np.float64)
        alpha = self._alpha(t)
        x = _simulate_gaussian_ou(
            t, n_paths, rng=rng, a=self.a, sigma=self.sigma, x0=0.0, mean=lambda _t: 0.0
        )
        return np.asarray(x + alpha[np.newaxis, :], dtype=np.float64)


_SHORT_END = 1.0e-8  # a tiny maturity used to read f(0,0) off the curve


def _simulate_gaussian_ou(
    times: FloatArray,
    n_paths: int,
    *,
    rng: np.random.Generator,
    a: float,
    sigma: float,
    x0: float,
    mean: Callable[[float], float],
) -> FloatArray:
    """Exact Gaussian-transition simulation of ``dx = a(mean - x)dt + sigma dW``."""
    t = np.asarray(times, dtype=np.float64)
    paths = np.empty((n_paths, t.size), dtype=np.float64)
    paths[:, 0] = x0
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        decay = np.exp(-a * dt)
        m = mean(t[i])
        cond_mean = m + (paths[:, i - 1] - m) * decay
        cond_var = sigma * sigma * (1.0 - decay * decay) / (2.0 * a)
        paths[:, i] = cond_mean + np.sqrt(cond_var) * rng.standard_normal(n_paths)
    return paths


def monte_carlo_discount(
    model: ShortRateModel,
    maturity: float,
    *,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    r"""Monte Carlo estimate of :math:`P(0,T) = \mathbb E[e^{-\int_0^T r\,ds}]` and its SE.

    Simulates the short rate on an ``n_steps`` grid (exact transitions) and integrates it by
    the trapezoidal rule along each path — the cross-check for the analytic bond price.

    Parameters
    ----------
    model : ShortRateModel
        The model to simulate.
    maturity : float
        Bond maturity ``T`` in years.
    n_paths, n_steps : int
        Number of paths and time steps on ``[0, T]``.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    tuple of float
        ``(discount_estimate, standard_error)``.
    """
    grid = np.linspace(0.0, maturity, n_steps + 1)
    paths = model.simulate(grid, n_paths, rng=rng)
    integral = np.trapezoid(paths, grid, axis=1)
    discounts = np.exp(-integral)
    estimate = float(discounts.mean())
    std_error = float(discounts.std(ddof=1) / np.sqrt(n_paths))
    return estimate, std_error
