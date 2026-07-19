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


class MertonPathSimulator:
    r"""Simulates Merton jump-diffusion paths (exact marginals, no time-discretisation bias).

    Over a step :math:`\Delta t` the log-price increment is the sum of an *exact* normal
    diffusion and an *exact* compound-Poisson jump,

    .. math::

        \Delta \ln S = \big(r - q - \lambda\bar\kappa - \tfrac12\sigma^2\big)\Delta t
                       + \sigma\sqrt{\Delta t}\,Z
                       + \sum_{k=1}^{N}\!Y_k,\qquad
        N\sim\mathrm{Poisson}(\lambda\Delta t),\;\; Y_k\sim\mathcal N(\mu_J,\sigma_J^2),

    with the compensator :math:`\bar\kappa = e^{\mu_J+\sigma_J^2/2}-1` keeping the
    discounted spot a martingale (so the simulated forward matches
    :meth:`~quantica.pricing.processes.MertonProcess.forward`). Both pieces are exact over
    any step, so — like :class:`GBMPathSimulator` — the marginals carry no
    path-discretisation bias, only Monte Carlo error. A step's jump total is
    :math:`\mathcal N(N\mu_J,\,N\sigma_J^2)` conditional on the count :math:`N`, drawn
    directly rather than jump-by-jump.

    Parameters
    ----------
    n_paths : int
        Number of simulated paths (must be >= 2).
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.

    References
    ----------
    Merton, R. C. (1976). "Option pricing when underlying stock returns are
    discontinuous", *Journal of Financial Economics* 3, 125--144.
    """

    def __init__(self, n_paths: int, *, rng: np.random.Generator) -> None:
        if n_paths < 2:
            raise ValueError(f"n_paths must be at least 2, got {n_paths}")
        self.n_paths = n_paths
        self.rng = rng

    def simulate(
        self,
        *,
        spot: float,
        rate: float,
        div: float,
        vol: float,
        lam: float,
        mu_j: float,
        sigma_j: float,
        dt: float,
        n_steps: int,
    ) -> FloatArray:
        """Simulate paths of shape ``(n_paths, n_steps + 1)``; column 0 is ``spot``."""
        compensator = float(np.exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0)
        drift = (rate - div - lam * compensator - 0.5 * vol * vol) * dt
        vol_step = vol * np.sqrt(dt)
        diffusion = drift + vol_step * self.rng.standard_normal((self.n_paths, n_steps))

        counts = self.rng.poisson(lam * dt, size=(self.n_paths, n_steps)).astype(np.float64)
        jump_normals = self.rng.standard_normal((self.n_paths, n_steps))
        jumps = counts * mu_j + np.sqrt(counts) * sigma_j * jump_normals

        log_paths = np.cumsum(diffusion + jumps, axis=1)
        paths = np.empty((self.n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = spot
        paths[:, 1:] = spot * np.exp(log_paths)
        return paths


class HestonPathSimulator:
    r"""Simulates Heston stochastic-volatility paths (full-truncation Euler).

    Evolves the coupled spot/variance system

    .. math::

        d\ln S_t = \big(r - q - \tfrac12 v_t\big)\,dt + \sqrt{v_t}\,dW^S_t,\qquad
        dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^v_t,\qquad
        d\langle W^S,W^v\rangle_t = \rho\,dt,

    with the **full-truncation** scheme of Lord et al. (2010): the variance is allowed to
    go negative but is floored at zero, :math:`v^+=\max(v,0)`, wherever it enters a drift,
    diffusion, or square root. This is the most accurate of the simple Euler fixes and
    keeps a small, controllable discretisation bias that shrinks as ``n_substeps`` rises.
    Unlike :class:`GBMPathSimulator` / :class:`MertonPathSimulator`, the marginals are
    *not* exact, so each observation step is refined into ``n_substeps`` Euler sub-steps;
    the path is recorded only at the ``n_steps`` observation dates.

    Parameters
    ----------
    n_paths : int
        Number of simulated paths (must be >= 2).
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.
    n_substeps : int, optional
        Euler sub-steps *per observation step* (default 32). More sub-steps reduce the
        variance-discretisation bias at linear cost.

    References
    ----------
    Heston, S. (1993). "A closed-form solution for options with stochastic volatility",
    *Review of Financial Studies* 6, 327--343.
    Lord, R., Koekkoek, R. & van Dijk, D. (2010). "A comparison of biased simulation
    schemes for stochastic volatility models", *Quantitative Finance* 10, 177--194.
    """

    def __init__(self, n_paths: int, *, rng: np.random.Generator, n_substeps: int = 32) -> None:
        if n_paths < 2:
            raise ValueError(f"n_paths must be at least 2, got {n_paths}")
        if n_substeps < 1:
            raise ValueError(f"n_substeps must be at least 1, got {n_substeps}")
        self.n_paths = n_paths
        self.rng = rng
        self.n_substeps = n_substeps

    def simulate(
        self,
        *,
        spot: float,
        rate: float,
        div: float,
        v0: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        dt: float,
        n_steps: int,
    ) -> FloatArray:
        """Simulate observed paths of shape ``(n_paths, n_steps + 1)``; column 0 is ``spot``."""
        h = dt / self.n_substeps
        sqrt_h = np.sqrt(h)
        rho_perp = np.sqrt(max(1.0 - rho * rho, 0.0))

        log_spot = np.zeros(self.n_paths, dtype=np.float64)
        variance = np.full(self.n_paths, v0, dtype=np.float64)
        observed = np.empty((self.n_paths, n_steps + 1), dtype=np.float64)
        observed[:, 0] = spot

        for step in range(n_steps):
            for _ in range(self.n_substeps):
                z_v = self.rng.standard_normal(self.n_paths)
                z_s = rho * z_v + rho_perp * self.rng.standard_normal(self.n_paths)
                v_pos = np.maximum(variance, 0.0)
                sqrt_v = np.sqrt(v_pos)
                log_spot += (rate - div - 0.5 * v_pos) * h + sqrt_v * sqrt_h * z_s
                variance += kappa * (theta - v_pos) * h + xi * sqrt_v * sqrt_h * z_v
            observed[:, step + 1] = spot * np.exp(log_spot)
        return observed
