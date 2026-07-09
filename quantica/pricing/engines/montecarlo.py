r"""Monte Carlo engine for European options under Black--Scholes.

For a vanilla European payoff only the *terminal* spot matters, and geometric
Brownian motion has an exact solution at maturity,

.. math::

    S_T = S_0 \exp\!\big((r - q - \tfrac12\sigma^2) T + \sigma \sqrt{T}\, Z\big),
    \qquad Z \sim \mathcal N(0, 1),

so we sample :math:`S_T` directly — no time-stepping, no discretisation bias
(the only error is statistical). The price is the discounted sample mean of the
payoff, reported with its standard error.

Two variance-reduction techniques are supported and composable:

* **Antithetic variates** — pair each draw :math:`Z` with :math:`-Z`. For a
  monotone payoff the pair is negatively correlated, shrinking the variance.
* **Control variate** — the discounted terminal spot :math:`e^{-rT} S_T`, whose
  expectation is known exactly, :math:`\mathbb E[e^{-rT} S_T] = S_0 e^{-qT}`.
  It is highly correlated with a vanilla payoff, an ideal control.

Randomness is always an injected, seeded :class:`numpy.random.Generator`; the
global ``numpy.random`` state is never touched (CLAUDE.md §2, reproducibility).

References
----------
Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*,
ch. 4 (variance reduction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantica.core.types import FloatArray
from quantica.pricing.engines._common import unpack

if TYPE_CHECKING:
    from quantica.pricing.instruments import EuropeanOption
    from quantica.pricing.processes import BlackScholesProcess

# Below this control-variate variance the control carries no information
# (deterministic limit); fall back to no adjustment.
_MIN_CONTROL_VAR = 1e-300


@dataclass(frozen=True)
class MCResult:
    """Outcome of a Monte Carlo pricing run.

    Attributes
    ----------
    price : float
        Discounted sample-mean estimate of the option value.
    std_error : float
        Standard error of ``price``: ``sample_std / sqrt(n_samples)``. Note
        ``n_samples`` is the number of *independent* samples, which is halved
        under antithetic pairing.
    n_paths : int
        Number of simulated terminal prices used.
    """

    price: float
    std_error: float
    n_paths: int


class MonteCarloEngine:
    """Monte Carlo pricer for a :class:`EuropeanOption`.

    Parameters
    ----------
    n_paths : int
        Number of simulated terminal prices (must be >= 2). Under antithetic
        variates this is the number of payoff evaluations; the number of
        independent samples is ``n_paths // 2``.
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility. Each call consumes draws
        and advances the generator.
    antithetic : bool, optional
        Enable antithetic variates.
    control_variate : bool, optional
        Enable the discounted-terminal-spot control variate.

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
        control_variate: bool = False,
    ) -> None:
        if n_paths < 2:
            raise ValueError(f"n_paths must be at least 2, got {n_paths}")
        self.n_paths = n_paths
        self.rng = rng
        self.antithetic = antithetic
        self.control_variate = control_variate

    def calculate(
        self,
        instrument: EuropeanOption,
        process: BlackScholesProcess,
    ) -> float:
        """Present value (the point estimate; see :meth:`estimate` for the SE)."""
        return self.estimate(instrument, process).price

    def estimate(
        self,
        instrument: EuropeanOption,
        process: BlackScholesProcess,
    ) -> MCResult:
        """Price ``instrument`` under ``process`` with its Monte Carlo standard error."""
        S, K, r, q, sigma, T, omega = unpack(instrument, process)
        disc = np.exp(-r * T)

        z = self._draw_normals()
        drift = (r - q - 0.5 * sigma * sigma) * T
        diffusion = sigma * np.sqrt(T)
        terminal_spot = S * np.exp(drift + diffusion * z)
        disc_payoff = disc * np.maximum(omega * (terminal_spot - K), 0.0)

        if self.control_variate:
            # Control W = e^{-rT} S_T, with known mean E[W] = S_0 e^{-qT}.
            control = disc * terminal_spot
            control_mean = S * np.exp(-q * T)
            per_path = self._apply_control_variate(disc_payoff, control, control_mean)
        else:
            per_path = disc_payoff

        samples = self._combine_antithetic(per_path)
        price = float(samples.mean())
        std_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
        return MCResult(price=price, std_error=std_error, n_paths=self.n_paths)

    # -- internals ---------------------------------------------------------- #

    def _draw_normals(self) -> FloatArray:
        """Standard normals; the second half mirrors the first under antithetic."""
        if self.antithetic:
            half = self.n_paths // 2
            base = self.rng.standard_normal(half)
            return np.concatenate([base, -base])
        draws: FloatArray = self.rng.standard_normal(self.n_paths)
        return draws

    @staticmethod
    def _apply_control_variate(
        disc_payoff: FloatArray,
        control: FloatArray,
        control_mean: float,
    ) -> FloatArray:
        r"""Adjust payoffs by the control: ``Y = P - beta (W - E[W])``.

        The optimal ``beta = Cov(P, W) / Var(W)`` is estimated from the sample
        (a plug-in that adds only O(1/n) bias). The adjustment is mean-preserving
        because ``E[W]`` is known exactly.
        """
        cov = np.cov(disc_payoff, control, ddof=1)
        var_control = float(cov[1, 1])
        if var_control <= _MIN_CONTROL_VAR:
            return disc_payoff  # deterministic control carries no information
        beta = float(cov[0, 1]) / var_control
        adjusted: FloatArray = disc_payoff - beta * (control - control_mean)
        return adjusted

    def _combine_antithetic(self, per_path: FloatArray) -> FloatArray:
        """Average antithetic pairs into independent samples (identity otherwise)."""
        if self.antithetic:
            half = self.n_paths // 2
            combined: FloatArray = 0.5 * (per_path[:half] + per_path[half : 2 * half])
            return combined
        return per_path
