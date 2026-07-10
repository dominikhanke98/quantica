r"""Longstaff--Schwartz Monte Carlo (LSM) for American options.

American exercise by simulation. The vanilla :class:`MonteCarloEngine` samples
only the *terminal* price, which is enough for a European payoff; early exercise
needs the whole path, so LSM simulates full geometric-Brownian-motion paths on a
grid of exercise dates :math:`0 = t_0 < t_1 < \dots < t_M = T` and works the
optimal-stopping policy out by backward induction.

At each exercise date the (discounted) value of *continuing* is unknown per path,
so it is **estimated by least-squares regression** of the realized discounted
future cashflow on a basis of the current spot — and, following the
Longstaff--Schwartz refinement, the regression uses **only in-the-money paths**
(an out-of-the-money path has no exercise decision to make and would only add
noise to the fit). A path is exercised where its immediate intrinsic exceeds the
fitted continuation, and every path is finally valued by the **realized
cashflow** under the resulting policy (never by the fitted value).

Estimator bias
--------------
The exercise policy learned from regression is *sub-optimal* (finite paths,
finite basis), and a sub-optimal stopping rule can only leave value on the table,
so realized-cashflow LSM is a **low-biased, lower-bound** estimator of the
American price. Landing at or a little below the tree/PDE reference — within a
few standard errors — is therefore the *expected* correctness signature, not a
failure; a richer basis or more paths reduces the downward bias.

Stepping is exact log-GBM (the log-increment over any step is exactly normal), so
there is **no path-discretisation bias**: the only errors are the exercise-date
grid (a Bermudan approximation to the continuous American, vanishing as
:math:`M \to \infty`) and the statistical Monte Carlo error.

References
----------
Longstaff, F. & Schwartz, E. (2001). "Valuing American Options by Simulation: A
Simple Least-Squares Approach", *Review of Financial Studies*.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from quantica.core.types import ExerciseStyle, FloatArray
from quantica.pricing.engines._common import unpack
from quantica.pricing.engines.montecarlo import MCResult

if TYPE_CHECKING:
    from quantica.pricing.instruments import VanillaOption
    from quantica.pricing.processes import BlackScholesProcess

_DEFAULT_EXERCISE_DATES = 50
_DEFAULT_BASIS_DEGREE = 3


class LongstaffSchwartzEngine:
    """Least-squares Monte Carlo pricer for an :class:`AmericanOption`.

    Parameters
    ----------
    n_paths : int
        Number of simulated paths (must be >= 2). Under antithetic variates this
        is the number of paths; the number of independent samples is
        ``n_paths // 2``.
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.
    exercise_dates : int, optional
        Number of equally spaced exercise opportunities :math:`M` up to expiry
        (default 50). More dates approach the continuous American limit.
    basis_degree : int, optional
        Degree :math:`k` of the monomial regression basis
        :math:`\\{1, x, x^2, \\dots, x^k\\}` in the (strike-scaled) spot
        (default 3). A richer basis fits the continuation value better and
        reduces the downward bias, up to a point.
    antithetic : bool, optional
        Mirror the Brownian increments to pair paths (variance reduction).

    Notes
    -----
    Satisfies the :class:`~quantica.pricing.engines.PricingEngine` protocol
    (price only). Use :meth:`estimate` for the price *and* its standard error.
    Prices American options only — for European exercise use
    :class:`~quantica.pricing.engines.montecarlo.MonteCarloEngine`.
    """

    def __init__(
        self,
        n_paths: int,
        *,
        rng: np.random.Generator,
        exercise_dates: int = _DEFAULT_EXERCISE_DATES,
        basis_degree: int = _DEFAULT_BASIS_DEGREE,
        antithetic: bool = False,
    ) -> None:
        if n_paths < 2:
            raise ValueError(f"n_paths must be at least 2, got {n_paths}")
        if exercise_dates < 1:
            raise ValueError(f"exercise_dates must be at least 1, got {exercise_dates}")
        if basis_degree < 1:
            raise ValueError(f"basis_degree must be at least 1, got {basis_degree}")
        self.n_paths = n_paths
        self.rng = rng
        self.exercise_dates = exercise_dates
        self.basis_degree = basis_degree
        self.antithetic = antithetic

    def calculate(
        self,
        instrument: VanillaOption,
        process: BlackScholesProcess,
    ) -> float:
        """Present value (the point estimate; see :meth:`estimate` for the SE)."""
        return self.estimate(instrument, process).price

    def estimate(
        self,
        instrument: VanillaOption,
        process: BlackScholesProcess,
    ) -> MCResult:
        """Price ``instrument`` under ``process`` with its Monte Carlo standard error.

        Raises
        ------
        ValueError
            If the option is not American (LSM is an early-exercise scheme; use
            the terminal-only Monte Carlo engine for European options).
        """
        if instrument.exercise is not ExerciseStyle.AMERICAN:
            raise ValueError(
                "the Longstaff--Schwartz engine prices American exercise only; "
                "use MonteCarloEngine for European options"
            )
        S, K, r, q, sigma, T, omega = unpack(instrument, process)
        n_steps = self.exercise_dates
        dt = T / n_steps
        disc_step = float(np.exp(-r * dt))

        paths = self._simulate_paths(S, r, q, sigma, dt, n_steps)
        intrinsic = np.maximum(omega * (paths - K), 0.0)

        # Backward induction over exercise dates t_{M-1}, ..., t_1. `cashflow`
        # holds each path's realized cashflow discounted to the current date.
        cashflow: FloatArray = intrinsic[:, n_steps].copy()  # value if held to expiry
        for step in range(n_steps - 1, 0, -1):
            cashflow *= disc_step  # move the future cashflow back one step to t_step
            in_money = intrinsic[:, step] > 0.0
            if np.count_nonzero(in_money) <= self.basis_degree:
                continue  # too few points to fit a continuation surface; hold
            continuation = self._fit_continuation(
                spot=paths[in_money, step], discounted_future=cashflow[in_money], strike=K
            )
            exercise = intrinsic[in_money, step] > continuation
            in_money_idx = np.nonzero(in_money)[0]
            exercised = in_money_idx[exercise]
            cashflow[exercised] = intrinsic[exercised, step]  # take intrinsic at t_step
        cashflow *= disc_step  # discount t_1 -> t_0

        samples = self._combine_antithetic(cashflow)
        mean = float(samples.mean())
        std_error = float(samples.std(ddof=1) / np.sqrt(samples.size))
        # The value at t=0 cannot be below immediate exercise (S_0 is known).
        price = max(mean, max(omega * (S - K), 0.0))
        return MCResult(price=price, std_error=std_error, n_paths=self.n_paths)

    # -- internals ---------------------------------------------------------- #

    def _simulate_paths(
        self, spot: float, rate: float, div: float, vol: float, dt: float, n_steps: int
    ) -> FloatArray:
        """Full GBM paths, exact log-Euler: shape ``(n_paths, n_steps + 1)``."""
        drift = (rate - div - 0.5 * vol * vol) * dt
        vol_step = vol * np.sqrt(dt)
        log_increments = drift + vol_step * self._draw_normals(n_steps)
        log_paths = np.cumsum(log_increments, axis=1)
        paths = np.empty((self.n_paths, n_steps + 1), dtype=np.float64)
        paths[:, 0] = spot
        paths[:, 1:] = spot * np.exp(log_paths)
        return paths

    def _draw_normals(self, n_steps: int) -> FloatArray:
        """Standard normals of shape ``(n_paths, n_steps)`` (antithetic mirrors rows)."""
        if self.antithetic:
            half = self.n_paths // 2
            base = self.rng.standard_normal((half, n_steps))
            return np.concatenate([base, -base], axis=0)
        draws: FloatArray = self.rng.standard_normal((self.n_paths, n_steps))
        return draws

    def _fit_continuation(
        self, spot: FloatArray, discounted_future: FloatArray, strike: float
    ) -> FloatArray:
        """Least-squares fit of continuation value on a monomial basis of the spot.

        The regressor is scaled by the strike to keep the Vandermonde matrix well
        conditioned across moneyness.
        """
        x = spot / strike
        design = np.vander(x, self.basis_degree + 1, increasing=True)
        coeffs, *_ = np.linalg.lstsq(design, discounted_future, rcond=None)
        fitted: FloatArray = design @ coeffs
        return fitted

    def _combine_antithetic(self, per_path: FloatArray) -> FloatArray:
        """Average antithetic path pairs into independent samples (identity otherwise)."""
        if self.antithetic:
            half = self.n_paths // 2
            combined: FloatArray = 0.5 * (per_path[:half] + per_path[half : 2 * half])
            return combined
        return per_path
