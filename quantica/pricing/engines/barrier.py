r"""Barrier options: closed-form (continuous) and Monte Carlo (discrete monitoring).

A single-barrier option is a vanilla payoff that is switched on (knock-*in*) or
off (knock-*out*) according to whether the underlying touches a barrier
:math:`H`.

Two prices live here:

* :func:`barrier_price` — the **continuous-monitoring** closed form
  (Reiner--Rubinstein / Merton), the analytic anchor.
* :class:`BarrierMonteCarloEngine` — **discrete-monitoring** Monte Carlo, which
  is what a real contract with periodic observations actually is.

Discrete-monitoring bias (named, and its direction reasoned)
------------------------------------------------------------
A discretely-monitored barrier can only be knocked at the observation dates, so
Monte Carlo **misses crossings that happen between steps**. It therefore
*under-detects* barrier hits: a knock-**out** keeps too many paths alive and is
biased **high** versus the continuous contract, while a knock-**in** activates
too few paths and is biased **low**. The bias shrinks as the monitoring
frequency rises (more steps -> fewer missed crossings), vanishing in the
continuous limit.

Brownian-bridge correction (the differentiator)
-----------------------------------------------
Rather than only brute-forcing more steps, we can *analytically* account for the
missed crossings. Conditional on the two endpoints :math:`S_{t_i}, S_{t_{i+1}}`
of a step (both on the safe side of :math:`H`), a Brownian bridge crosses the
barrier with probability

.. math::

    p_i = \exp\!\Big(-\frac{2\,\ln(H/S_{t_i})\,\ln(H/S_{t_{i+1}})}{\sigma^2\,\Delta t}\Big),

so a path's continuous survival probability is :math:`\prod_i (1 - p_i)`.
Weighting each path's payoff by this survival probability (``brownian_bridge=True``)
estimates the *continuous* contract and removes most of the discretisation bias
at a fixed step count.

Randomness is an injected, seeded :class:`numpy.random.Generator`; standard
errors are reported (numerical-validation skill §5).

References
----------
Reiner, E. & Rubinstein, M. (1991). "Breaking Down the Barriers", *Risk*.
Beaglehole, Dybvig & Zhou (1997) and Broadie, Glasserman & Kou (1997) on the
Brownian-bridge / discrete-monitoring correction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import norm

from quantica.core.types import FloatArray, OptionType
from quantica.pricing.engines._common import unpack
from quantica.pricing.engines._paths import GBMPathSimulator
from quantica.pricing.engines.montecarlo import MCResult
from quantica.pricing.instruments import BarrierOption

if TYPE_CHECKING:
    from quantica.core.types import BarrierType
    from quantica.pricing.processes import BlackScholesProcess


def _n(x: float) -> float:
    return float(norm.cdf(x))


def barrier_price(
    spot: float,
    strike: float,
    barrier: float,
    rate: float,
    div: float,
    vol: float,
    expiry: float,
    barrier_type: BarrierType,
    option_type: OptionType,
) -> float:
    """Continuous-monitoring barrier price (Reiner--Rubinstein), zero rebate.

    Matches QuantLib's ``AnalyticBarrierEngine`` to machine precision. Uses the
    standard A/B/C/D decomposition; in-out parity (in + out = vanilla) holds by
    construction.
    """
    S, X, H, r, q, sigma, T = spot, strike, barrier, rate, div, vol, expiry
    b = r - q
    phi = float(option_type.sign)  # +1 call, -1 put
    eta = -1.0 if barrier_type.is_up else 1.0
    vsqt = sigma * np.sqrt(T)
    mu = (b - 0.5 * sigma * sigma) / (sigma * sigma)

    x1 = np.log(S / X) / vsqt + (1 + mu) * vsqt
    x2 = np.log(S / H) / vsqt + (1 + mu) * vsqt
    y1 = np.log(H * H / (S * X)) / vsqt + (1 + mu) * vsqt
    y2 = np.log(H / S) / vsqt + (1 + mu) * vsqt
    carry, disc = np.exp((b - r) * T), np.exp(-r * T)

    a = phi * S * carry * _n(phi * x1) - phi * X * disc * _n(phi * x1 - phi * vsqt)
    bb = phi * S * carry * _n(phi * x2) - phi * X * disc * _n(phi * x2 - phi * vsqt)
    c = phi * S * carry * (H / S) ** (2 * (mu + 1)) * _n(eta * y1) - phi * X * disc * (H / S) ** (
        2 * mu
    ) * _n(eta * y1 - eta * vsqt)
    d = phi * S * carry * (H / S) ** (2 * (mu + 1)) * _n(eta * y2) - phi * X * disc * (H / S) ** (
        2 * mu
    ) * _n(eta * y2 - eta * vsqt)

    is_call = option_type is OptionType.CALL
    up = barrier_type.is_up
    knock_in = barrier_type.is_knock_in
    strike_above_barrier = X > H

    # Reiner--Rubinstein type table (zero rebate). In + out = vanilla (= A here).
    if knock_in:
        if is_call and not up:  # down-and-in call
            value = c if strike_above_barrier else a - bb + d
        elif is_call and up:  # up-and-in call
            value = a if strike_above_barrier else bb - c + d
        elif not is_call and not up:  # down-and-in put
            value = bb - c + d if strike_above_barrier else a
        else:  # up-and-in put
            value = a - bb + d if strike_above_barrier else c
    else:
        if is_call and not up:  # down-and-out call
            value = a - c if strike_above_barrier else bb - d
        elif is_call and up:  # up-and-out call
            value = 0.0 if strike_above_barrier else a - bb + c - d
        elif not is_call and not up:  # down-and-out put
            value = a - bb + c - d if strike_above_barrier else 0.0
        else:  # up-and-out put
            value = bb - d if strike_above_barrier else a - c
    return float(value)


class BarrierMonteCarloEngine:
    """Discrete-monitoring Monte Carlo pricer for a :class:`BarrierOption`.

    Parameters
    ----------
    n_paths : int
        Number of simulated paths (must be >= 2).
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.
    antithetic : bool, optional
        Mirror the Brownian increments (variance reduction).
    brownian_bridge : bool, optional
        If ``True``, weight each surviving path by its Brownian-bridge continuous
        survival probability, estimating the *continuous* barrier contract and
        largely removing the discrete-monitoring bias. If ``False`` (default),
        knock is decided purely at the monitoring dates (the discrete contract).

    Notes
    -----
    Satisfies the :class:`~quantica.pricing.engines.PricingEngine` protocol
    (price only). Use :meth:`estimate` for the price *and* its standard error.
    """

    def __init__(
        self,
        n_paths: int,
        *,
        rng: np.random.Generator,
        antithetic: bool = False,
        brownian_bridge: bool = False,
    ) -> None:
        self.n_paths = n_paths
        self.brownian_bridge = brownian_bridge
        self._sim = GBMPathSimulator(n_paths, rng=rng, antithetic=antithetic)

    def calculate(self, instrument: BarrierOption, process: BlackScholesProcess) -> float:
        """Present value (the point estimate; see :meth:`estimate` for the SE)."""
        return self.estimate(instrument, process).price

    def estimate(self, instrument: BarrierOption, process: BlackScholesProcess) -> MCResult:
        """Price ``instrument`` under ``process`` with its Monte Carlo standard error."""
        if not isinstance(instrument, BarrierOption):
            raise TypeError(
                f"BarrierMonteCarloEngine prices BarrierOption, got {type(instrument).__name__}"
            )
        S, K, r, q, sigma, T, omega = unpack(instrument, process)
        H = instrument.barrier
        up = instrument.barrier_type.is_up
        n = instrument.n_monitoring_dates
        dt = T / n
        disc = float(np.exp(-r * T))

        paths = self._sim.simulate(spot=S, rate=r, div=q, vol=sigma, dt=dt, n_steps=n)
        terminal_payoff = np.maximum(omega * (paths[:, -1] - K), 0.0)

        survival = self._knock_out_survival(paths, H, up, sigma, dt)
        weight = 1.0 - survival if instrument.barrier_type.is_knock_in else survival
        per_path = disc * terminal_payoff * weight

        samples = self._sim.combine_antithetic(per_path)
        price = float(samples.mean())
        std_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
        return MCResult(price=price, std_error=std_error, n_paths=self.n_paths)

    def _knock_out_survival(
        self, paths: FloatArray, barrier: float, up: bool, vol: float, dt: float
    ) -> FloatArray:
        """Per-path knock-out survival weight (0/1 discrete, or bridge probability)."""
        monitored = paths[:, 1:]  # observations at t_1 .. t_n
        if up:
            discretely_hit = (monitored >= barrier).any(axis=1)
        else:
            discretely_hit = (monitored <= barrier).any(axis=1)

        if not self.brownian_bridge:
            return np.where(discretely_hit, 0.0, 1.0)

        # Brownian-bridge continuous survival over every step (t_0..t_n). For
        # paths that discretely hit, the weight is overridden to 0 below, so the
        # (possibly overflowing) bridge product on those paths is discarded.
        left, right = paths[:, :-1], paths[:, 1:]
        with np.errstate(over="ignore", invalid="ignore"):
            cross_prob = np.exp(
                -2.0 * np.log(barrier / left) * np.log(barrier / right) / (vol * vol * dt)
            )
            survival: FloatArray = np.prod(1.0 - cross_prob, axis=1)
        return np.where(discretely_hit, 0.0, survival)
