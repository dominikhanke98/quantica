r"""Cox--Ross--Rubinstein binomial-tree engine for vanilla options.

A recombining binomial lattice under the risk-neutral measure with continuous
dividend yield. Over a step :math:`\Delta t = T/N` the spot moves up by
:math:`u = e^{\sigma\sqrt{\Delta t}}` or down by :math:`d = 1/u`, with
risk-neutral up-probability

.. math::

    p = \frac{e^{(r-q)\Delta t} - d}{u - d}.

The value is the discounted risk-neutral expectation of the payoff, computed by
backward induction over the lattice. For a **European** option this is a plain
roll-back; for an **American** option early exercise is handled by taking
:math:`\max(\text{continuation}, \text{intrinsic})` at every node — the whole
generalisation is that one line, which is the point: the lattice machinery is
unchanged. As :math:`N\to\infty` the European price converges to Black--Scholes
at first order, :math:`O(1/N)` (with the familiar even/odd-:math:`N` oscillation
as the strike moves relative to the terminal nodes).

References
----------
Cox, J., Ross, S. & Rubinstein, M. (1979). "Option Pricing: A Simplified
Approach", *Journal of Financial Economics*. Hull (2018), ch. 21.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from quantica.core.types import ExerciseStyle
from quantica.pricing.engines._common import unpack

if TYPE_CHECKING:
    from quantica.pricing.instruments import VanillaOption
    from quantica.pricing.processes import BlackScholesProcess

_DEFAULT_STEPS = 256


class BinomialEngine:
    """CRR binomial-tree pricer for a vanilla option (European or American).

    Parameters
    ----------
    steps : int, optional
        Number of time steps :math:`N` in the lattice (default 256). More steps
        reduce discretisation error at :math:`O(1/N)`; pass a large value for a
        reference-quality price.

    Notes
    -----
    Satisfies the :class:`~quantica.pricing.engines.PricingEngine` protocol
    (price only). The engine is stateless apart from ``steps``, so one instance
    can price any number of options.
    """

    def __init__(self, steps: int = _DEFAULT_STEPS) -> None:
        if steps < 1:
            raise ValueError(f"steps must be a positive integer, got {steps}")
        self.steps = steps

    def calculate(
        self,
        instrument: VanillaOption,
        process: BlackScholesProcess,
    ) -> float:
        """Present value of ``instrument`` under ``process`` on the CRR lattice."""
        S, K, r, q, sigma, T, omega = unpack(instrument, process)

        # Deterministic limit: a degenerate (zero-width) tree is exactly the
        # discounted intrinsic value on the forward, matching the analytic
        # sigma->0 / T->0 limit.
        if sigma == 0.0 or T == 0.0:
            disc_spot = S * math.exp(-q * T)
            disc_strike = K * math.exp(-r * T)
            return max(omega * (disc_spot - disc_strike), 0.0)

        n = self.steps
        dt = T / n
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        p = (math.exp((r - q) * dt) - d) / (u - d)
        american = instrument.exercise is ExerciseStyle.AMERICAN

        # Terminal spot at node j (j up-moves, N-j down-moves): S * u^(2j - N).
        j = np.arange(n + 1)
        terminal_spot = S * u ** (2 * j - n)
        values = np.maximum(omega * (terminal_spot - K), 0.0)

        # Backward induction: V_i <- disc * (p * V_up + (1-p) * V_down). For an
        # American option, also allow immediate exercise at each node:
        # V_i <- max(continuation, intrinsic).
        one_minus_p = 1.0 - p
        for _ in range(n):
            continuation = disc * (p * values[1:] + one_minus_p * values[:-1])
            if american:
                level = continuation.size - 1  # intervals at this time layer
                spot = S * u ** (2 * np.arange(level + 1) - level)
                intrinsic = np.maximum(omega * (spot - K), 0.0)
                values = np.maximum(continuation, intrinsic)
            else:
                values = continuation

        return float(values[0])
