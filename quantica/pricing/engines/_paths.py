"""Shared Monte Carlo path machinery.

Full geometric-Brownian-motion path simulation plus antithetic pairing, factored
out of the Longstaff--Schwartz engine once the exotic (Asian, barrier) engines
became the second consumer (CLAUDE.md §2, "extract on the second use"). The
terminal-only :class:`~quantica.pricing.engines.montecarlo.MonteCarloEngine`
keeps its own lightweight sampling — it never needs a whole path.
"""

from __future__ import annotations

import numpy as np

from quantica.core.types import FloatArray


class GBMPathSimulator:
    r"""Simulates full GBM paths with exact log-stepping and injected randomness.

    Over a step :math:`\Delta t` the log-price increment is *exactly* normal,
    :math:`\Delta \ln S = (r - q - \tfrac12\sigma^2)\Delta t + \sigma\sqrt{\Delta t}\,Z`,
    so the simulated marginals are exact at every grid point — there is no
    path-discretisation bias, only the statistical Monte Carlo error (and, for a
    path-dependent payoff, whatever discrete-monitoring error the contract's own
    grid introduces).

    Parameters
    ----------
    n_paths : int
        Number of simulated paths (must be >= 2). Under antithetic variates this
        is the number of paths; the number of *independent* samples is
        ``n_paths // 2``.
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.
    antithetic : bool, optional
        Mirror the Brownian increments (row ``i`` paired with its negation) for
        variance reduction.
    """

    def __init__(
        self,
        n_paths: int,
        *,
        rng: np.random.Generator,
        antithetic: bool = False,
    ) -> None:
        if n_paths < 2:
            raise ValueError(f"n_paths must be at least 2, got {n_paths}")
        self.n_paths = n_paths
        self.rng = rng
        self.antithetic = antithetic

    def draw_normals(self, n_steps: int) -> FloatArray:
        """Standard normals of shape ``(n_paths, n_steps)`` (antithetic mirrors rows)."""
        if self.antithetic:
            half = self.n_paths // 2
            base = self.rng.standard_normal((half, n_steps))
            return np.concatenate([base, -base], axis=0)
        draws: FloatArray = self.rng.standard_normal((self.n_paths, n_steps))
        return draws

    def simulate(
        self,
        *,
        spot: float,
        rate: float,
        div: float,
        vol: float,
        dt: float,
        n_steps: int,
    ) -> FloatArray:
        """Simulate paths of shape ``(n_paths, n_steps + 1)``; column 0 is ``spot``."""
        drift = (rate - div - 0.5 * vol * vol) * dt
        vol_step = vol * np.sqrt(dt)
        log_increments = drift + vol_step * self.draw_normals(n_steps)
        log_paths = np.cumsum(log_increments, axis=1)
        paths = np.empty((self.n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = spot
        paths[:, 1:] = spot * np.exp(log_paths)
        return paths

    def combine_antithetic(self, per_path: FloatArray) -> FloatArray:
        """Average antithetic path pairs into independent samples (identity otherwise)."""
        if self.antithetic:
            half = self.n_paths // 2
            combined: FloatArray = 0.5 * (per_path[:half] + per_path[half : 2 * half])
            return combined
        return per_path
