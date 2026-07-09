"""Closed-form Black--Scholes--Merton engine for European options.

Prices and first-order Greeks in the constant-parameter Black--Scholes model
with a continuous dividend yield (Black & Scholes 1973; Merton 1973). All
formulae are written with the option's payoff sign :math:`\\omega` (``+1`` call,
``-1`` put) so a single expression covers both flavours.

References
----------
Hull, J. C. (2018). *Options, Futures, and Other Derivatives*, 10th ed.,
ch. 15 & 19 (pricing and Greeks). Merton, R. C. (1973). "Theory of Rational
Option Pricing", *Bell J. Econ.*, on the continuous-dividend extension.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from scipy.stats import norm

from quantica.pricing.greeks import Greeks

if TYPE_CHECKING:
    from quantica.pricing.instruments import EuropeanOption
    from quantica.pricing.processes import BlackScholesProcess


class AnalyticEuropeanEngine:
    """Black--Scholes--Merton closed-form pricer for a :class:`EuropeanOption`.

    Satisfies the :class:`~quantica.pricing.engines.GreeksEngine` protocol:
    provides both ``calculate`` (price) and ``greeks`` (analytic delta, gamma,
    vega, theta, rho). The engine is stateless, so one instance can price any
    number of options and processes.
    """

    def calculate(
        self,
        instrument: EuropeanOption,
        process: BlackScholesProcess,
    ) -> float:
        r"""Present value :math:`V = \omega [S e^{-qT} N(\omega d_1) - K e^{-rT} N(\omega d_2)]`.

        In the degenerate :math:`\sigma\sqrt{T} \to 0` limit (zero vol or zero
        time to expiry) the price collapses to the discounted intrinsic value
        of the forward, :math:`\max(\omega (S e^{-qT} - K e^{-rT}), 0)`.
        """
        S, K, r, q, sigma, T, omega = _unpack(instrument, process)
        disc_spot = S * math.exp(-q * T)
        disc_strike = K * math.exp(-r * T)

        if sigma * math.sqrt(T) == 0.0:
            # Deterministic underlying: discounted intrinsic on the forward.
            return max(omega * (disc_spot - disc_strike), 0.0)

        d1, d2 = _d1_d2(S, K, r, q, sigma, T)
        return omega * (disc_spot * _cdf(omega * d1) - disc_strike * _cdf(omega * d2))

    def greeks(
        self,
        instrument: EuropeanOption,
        process: BlackScholesProcess,
    ) -> Greeks:
        r"""Analytic first-order Greeks (see :class:`~quantica.pricing.greeks.Greeks`).

        Raises
        ------
        ValueError
            If :math:`\sigma = 0` or :math:`T = 0`, where the Greeks are not
            well defined (delta becomes a step, gamma a Dirac spike).
        """
        S, K, r, q, sigma, T, omega = _unpack(instrument, process)
        if sigma == 0.0 or T == 0.0:
            raise ValueError("Greeks are undefined in the zero-vol / zero-expiry limit")

        sqrt_T = math.sqrt(T)
        d1, d2 = _d1_d2(S, K, r, q, sigma, T)
        disc_r = math.exp(-r * T)  # e^{-rT}
        disc_q = math.exp(-q * T)  # e^{-qT}
        pdf_d1 = _phi(d1)

        delta = omega * disc_q * _cdf(omega * d1)
        gamma = disc_q * pdf_d1 / (S * sigma * sqrt_T)
        vega = S * disc_q * pdf_d1 * sqrt_T
        rho = omega * K * T * disc_r * _cdf(omega * d2)
        # theta = dV/dt (calendar time); the first term is the common time-decay
        # of optionality, the other two the carry on strike and spot legs.
        theta = (
            -disc_q * S * pdf_d1 * sigma / (2.0 * sqrt_T)
            - omega * r * K * disc_r * _cdf(omega * d2)
            + omega * q * S * disc_q * _cdf(omega * d1)
        )
        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def _unpack(
    instrument: EuropeanOption,
    process: BlackScholesProcess,
) -> tuple[float, float, float, float, float, float, int]:
    """Pull the scalar model parameters out of the instrument and process."""
    return (
        process.spot,
        instrument.strike,
        process.rate,
        process.div,
        process.vol,
        instrument.expiry,
        instrument.option_type.sign,
    )


def _d1_d2(S: float, K: float, r: float, q: float, sigma: float, T: float) -> tuple[float, float]:
    r"""The Black--Scholes :math:`d_1` and :math:`d_2`. Requires :math:`\sigma\sqrt{T} > 0`."""
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return d1, d2


def _cdf(x: float) -> float:
    """Standard normal CDF, :math:`N(x)`."""
    return float(norm.cdf(x))


def _phi(x: float) -> float:
    """Standard normal PDF, :math:`\\phi(x)`."""
    return float(norm.pdf(x))
