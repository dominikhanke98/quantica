r"""Crank--Nicolson finite-difference engine for European/American options, with Greeks.

Prices by solving the Black--Scholes PDE on a grid. Writing the value in
log-price :math:`x = \ln S` turns the equation into a constant-coefficient
convection--diffusion problem in time-to-maturity :math:`\tau = T - t`,

.. math::

    \frac{\partial V}{\partial \tau}
    = \tfrac12\sigma^2 \frac{\partial^2 V}{\partial x^2}
      + \big(r - q - \tfrac12\sigma^2\big)\frac{\partial V}{\partial x}
      - r V,

with initial condition the payoff at :math:`\tau = 0`. The Crank--Nicolson
scheme (the trapezoidal average of explicit and implicit Euler) is
unconditionally stable and second-order accurate in both :math:`\Delta x` and
:math:`\Delta \tau`; each step solves a tridiagonal system.

Dirichlet boundaries use the discounted-intrinsic asymptotics
:math:`\max(\omega(S e^{-q\tau} - K e^{-r\tau}), 0)`, which are exact in the
:math:`S \to 0` and :math:`S \to \infty` limits, so a wide-enough domain makes
the boundary error negligible against the interior discretisation error.

Greeks
------
The engine satisfies the :class:`~quantica.pricing.engines.GreeksEngine`
protocol. Because the solve *is* the value surface :math:`V(S, t)` on the grid,
the spatial Greeks come almost for free from finite differences of adjacent
nodes, mapped from the log-grid by the chain rule (:math:`\partial_S = S^{-1}\partial_x`):

.. math::

    \Delta = \frac{V_x}{S}, \qquad
    \Gamma = \frac{V_{xx} - V_x}{S^2},

with :math:`V_x, V_{xx}` central differences at the spot node. **Theta** is a
central difference in the time direction (one extra step past today).
**Vega** and **rho** are bump-and-reval: re-solve at :math:`\sigma \pm h` and
:math:`r \pm h` (reusing the process ``with_*`` helpers) and central-difference.

Rannacher start-up (L-stability near the payoff kink)
-----------------------------------------------------
Crank--Nicolson is A-stable but **not L-stable**: it damps high-frequency error
modes only weakly. The vanilla payoff has a kink at the strike, whose
non-smooth data excites high-frequency modes that CN fails to damp, producing
spurious oscillations concentrated near the strike. The *price* is smooth enough
that this is invisible, but **gamma** read off a raw CN grid rings visibly around
the strike. The remedy is **Rannacher start-up** (Giles & Carter 2006): replace
the first ``rannacher_steps`` Crank--Nicolson steps nearest expiry with pairs of
fully-implicit backward-Euler half-steps (:math:`\Delta\tau/2`). Backward Euler is
L-stable and annihilates the offending modes before CN takes over, restoring
second-order accuracy with a clean, oscillation-free gamma. It is on by default
(``rannacher_steps=2``); set ``rannacher_steps=0`` for pure CN to see the
oscillation (``scripts/rannacher_gamma_demo.py`` quantifies the before/after).

American exercise (linear complementarity)
------------------------------------------
An American option adds the early-exercise constraint :math:`V \ge g`, where
:math:`g = \max(\omega(S-K), 0)` is the immediate-exercise (intrinsic) value.
Each step then solves a **linear complementarity problem** by **projected SOR**
(PSOR) — Gauss--Seidel over-relaxation with a :math:`\max(\cdot, g)` projection
after each node update — iterated to convergence and warm-started from the
previous step, which keeps the sweep count small. The far-field Dirichlet value
becomes :math:`\max(\text{European discounted intrinsic},\, g)`.

References
----------
Giles, M. & Carter, R. (2006), "Convergence analysis of Crank--Nicolson and
Rannacher time-marching", *Journal of Computational Finance*. Wilmott, Howison &
Dewynne (1995), *The Mathematics of Financial Derivatives*. Duffy (2006),
*Finite Difference Methods in Financial Engineering*.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from scipy.linalg import solve_banded

from quantica.core.types import ExerciseStyle, FloatArray
from quantica.pricing.engines._common import unpack
from quantica.pricing.greeks import Greeks

if TYPE_CHECKING:
    from quantica.pricing.instruments import VanillaOption
    from quantica.pricing.processes import BlackScholesProcess

_DEFAULT_SPACE_STEPS = 200
_DEFAULT_TIME_STEPS = 200
_DEFAULT_NUM_STD = 6.0
_DEFAULT_RANNACHER_STEPS = 2  # first CN steps replaced by backward-Euler half-step pairs

# Central-difference bump sizes for the re-solved Greeks (vega, rho). Small enough
# that the O(h^2) truncation is negligible; the log-grid places ln(spot) exactly on a
# node in every re-solve, so the discretisation error is common-mode and cancels.
_VEGA_BUMP = 1e-3  # absolute bump in volatility
_RHO_BUMP = 1e-3  # absolute bump in the rate

# Projected SOR parameters for the American LCP (see _psor_solve).
_PSOR_RELAXATION = 1.2  # over-relaxation factor omega in (1, 2)
_PSOR_TOL = 1e-9  # convergence tol on the max node update (<< discretisation error)
_PSOR_MAX_ITER = 10_000  # safety cap; warm-started sweeps converge in a handful


class _Grid(NamedTuple):
    """The fixed log-price grid (shared across bump re-solves so errors cancel)."""

    x: FloatArray  # log-price nodes, uniform spacing dx
    spot_grid: FloatArray  # exp(x)
    dx: float
    i_spot: int  # index of the node at ln(spot)


class _Scheme(NamedTuple):
    """A theta-scheme step: banded implicit matrix + operator coefficients.

    A step solves ``M V^{n+1} = N V^n`` with ``M = I - imp*dt*L`` (banded ``ab``,
    coefficients ``m_*``) and ``N = I + (1-imp)*dt*L`` (coefficients ``e_*``);
    ``imp = 1/2`` is Crank--Nicolson, ``imp = 1`` is backward Euler (``e_*`` the identity).
    """

    ab: FloatArray
    m_lower: float
    m_diag: float
    m_upper: float
    e_lower: float
    e_diag: float
    e_upper: float


class _Solution(NamedTuple):
    """The solved value surface at today (``tau = T``), plus theta when requested."""

    values: FloatArray
    theta: float | None


class FiniteDifferenceEngine:
    """Crank--Nicolson PDE pricer + Greeks for a vanilla option (European or American).

    Parameters
    ----------
    space_steps : int, optional
        Number of intervals along the log-price axis (default 200). Rounded down
        to an even number so that ``ln(spot)`` lands exactly on a grid node (no
        interpolation error at the evaluation point).
    time_steps : int, optional
        Number of time steps from maturity back to today (default 200).
    num_std : float, optional
        Half-width of the log-price domain in standard deviations
        :math:`\\sigma\\sqrt{T}` (default 6), widened if needed to enclose the
        strike. Larger values push the Dirichlet boundaries further out.
    rannacher_steps : int, optional
        Number of initial Crank--Nicolson steps (nearest expiry) replaced by
        backward-Euler half-step pairs to damp the payoff-kink oscillation in
        gamma (default 2). ``0`` disables Rannacher (pure Crank--Nicolson). Clamped
        to ``time_steps``.

    Notes
    -----
    Satisfies :class:`~quantica.pricing.engines.GreeksEngine`. A European option is
    solved by a tridiagonal solve per step and is second-order accurate — halving
    both step sizes cuts the error by ~4. An American option solves the
    early-exercise LCP per step by projected SOR (see the module docstring).
    """

    def __init__(
        self,
        space_steps: int = _DEFAULT_SPACE_STEPS,
        time_steps: int = _DEFAULT_TIME_STEPS,
        num_std: float = _DEFAULT_NUM_STD,
        rannacher_steps: int = _DEFAULT_RANNACHER_STEPS,
    ) -> None:
        if space_steps < 2:
            raise ValueError(f"space_steps must be at least 2, got {space_steps}")
        if time_steps < 1:
            raise ValueError(f"time_steps must be at least 1, got {time_steps}")
        if num_std <= 0.0:
            raise ValueError(f"num_std must be positive, got {num_std}")
        if rannacher_steps < 0:
            raise ValueError(f"rannacher_steps must be non-negative, got {rannacher_steps}")
        self.space_steps = space_steps
        self.time_steps = time_steps
        self.num_std = num_std
        self.rannacher_steps = rannacher_steps

    def calculate(
        self,
        instrument: VanillaOption,
        process: BlackScholesProcess,
    ) -> float:
        """Present value of ``instrument`` under ``process`` via Crank--Nicolson.

        Parameters
        ----------
        instrument : VanillaOption
            The contract (European or American vanilla).
        process : BlackScholesProcess
            The market dynamics (spot, rate, dividend, volatility).

        Returns
        -------
        float
            The present value at the spot node of the solved grid.
        """
        S, K, r, q, sigma, T, omega = unpack(instrument, process)
        american = instrument.exercise is ExerciseStyle.AMERICAN

        # Degenerate (no diffusion) limit: discounted intrinsic on the forward,
        # matching the analytic sigma->0 / T->0 result.
        if sigma == 0.0 or T == 0.0:
            return max(omega * (S * math.exp(-q * T) - K * math.exp(-r * T)), 0.0)

        grid = self._make_grid(S, K, sigma, T)
        solution = self._solve(grid, K, r, q, sigma, T, omega, american, with_theta=False)
        return float(solution.values[grid.i_spot])

    def greeks(
        self,
        instrument: VanillaOption,
        process: BlackScholesProcess,
    ) -> Greeks:
        r"""First-order Greeks of ``instrument`` under ``process`` from the PDE solve.

        Delta and gamma are read off the solved value surface at the spot node
        (chain-rule central differences on the log-grid); theta is a central
        difference in the time direction; vega and rho are bump-and-reval on
        :math:`\sigma` and :math:`r`. Rannacher start-up (on by default) keeps the
        gamma free of the Crank--Nicolson payoff-kink oscillation.

        Parameters
        ----------
        instrument : VanillaOption
            The contract (European or American vanilla).
        process : BlackScholesProcess
            The market dynamics.

        Returns
        -------
        Greeks
            Delta, gamma, vega, theta and rho (same conventions as
            :class:`~quantica.pricing.greeks.Greeks`).

        Raises
        ------
        ValueError
            If :math:`\sigma = 0` or :math:`T = 0`, where the Greeks are not well
            defined (delta becomes a step, gamma a Dirac spike).
        """
        S, K, r, q, sigma, T, omega = unpack(instrument, process)
        if sigma == 0.0 or T == 0.0:
            raise ValueError("Greeks are undefined in the zero-vol / zero-expiry limit")
        american = instrument.exercise is ExerciseStyle.AMERICAN

        grid = self._make_grid(S, K, sigma, T)
        solution = self._solve(grid, K, r, q, sigma, T, omega, american, with_theta=True)
        i = grid.i_spot
        values = solution.values
        s_i = float(grid.spot_grid[i])
        dx = grid.dx

        # Spatial Greeks via the chain rule from the log-grid (d/dS = (1/S) d/dx):
        #   delta = V_x / S,   gamma = (V_xx - V_x) / S^2.
        v_x = (values[i + 1] - values[i - 1]) / (2.0 * dx)
        v_xx = (values[i + 1] - 2.0 * values[i] + values[i - 1]) / (dx * dx)
        delta = float(v_x / s_i)
        gamma = float((v_xx - v_x) / (s_i * s_i))
        theta = float(solution.theta) if solution.theta is not None else 0.0

        # Vega and rho by bump-and-reval, reusing the process bump helpers. Each
        # re-solve rebuilds the grid but with ln(spot) exactly on a node, so the
        # discretisation error is common-mode across the +/- bumps and cancels.
        vega = (
            self.calculate(instrument, process.with_vol(sigma + _VEGA_BUMP))
            - self.calculate(instrument, process.with_vol(sigma - _VEGA_BUMP))
        ) / (2.0 * _VEGA_BUMP)
        rho = (
            self.calculate(instrument, process.with_rate(r + _RHO_BUMP))
            - self.calculate(instrument, process.with_rate(r - _RHO_BUMP))
        ) / (2.0 * _RHO_BUMP)

        return Greeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _make_grid(self, S: float, K: float, sigma: float, T: float) -> _Grid:
        """Build the centred log-price grid so ``ln(spot)`` is exactly a node."""
        ln_spot = math.log(S)
        sig_sqrt_t = sigma * math.sqrt(T)
        half_width = max(self.num_std * sig_sqrt_t, abs(math.log(K) - ln_spot) + 3.0 * sig_sqrt_t)
        half_nodes = self.space_steps // 2
        dx = half_width / half_nodes
        n_x = 2 * half_nodes  # even number of intervals
        i_spot = half_nodes
        x = np.asarray(ln_spot + (np.arange(n_x + 1) - i_spot) * dx, dtype=np.float64)
        return _Grid(x=x, spot_grid=np.asarray(np.exp(x), dtype=np.float64), dx=dx, i_spot=i_spot)

    def _solve(
        self,
        grid: _Grid,
        K: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        omega: int,
        american: bool,
        *,
        with_theta: bool,
    ) -> _Solution:
        """March the PDE from the payoff to today on ``grid``; optionally return theta."""
        spot_grid = grid.spot_grid
        dx = grid.dx
        n_x = spot_grid.size - 1
        n_int = n_x - 1
        n_t = self.time_steps
        dt = T / n_t

        # Constant spatial operator coefficients (log-price, constant in x and tau).
        a = 0.5 * sigma * sigma  # diffusion
        b = r - q - 0.5 * sigma * sigma  # convection
        lower = a / (dx * dx) - b / (2.0 * dx)
        diag = -2.0 * a / (dx * dx) - r
        upper = a / (dx * dx) + b / (2.0 * dx)

        cn = _make_scheme(lower, diag, upper, dt, 0.5, n_int)  # Crank--Nicolson
        be = _make_scheme(lower, diag, upper, 0.5 * dt, 1.0, n_int)  # backward-Euler half step

        intrinsic: FloatArray = np.maximum(omega * (spot_grid - K), 0.0)
        obstacle = intrinsic[1:-1]  # interior obstacle for the American LCP
        s_min = float(spot_grid[0])
        s_max = float(spot_grid[-1])

        def boundary(spot: float, tau: float) -> float:
            european = max(omega * (spot * math.exp(-q * tau) - K * math.exp(-r * tau)), 0.0)
            if not american:
                return european
            return max(european, max(omega * (spot - K), 0.0))

        # Time schedule: the first `rannacher_steps` CN steps become backward-Euler
        # half-step pairs (Rannacher start-up); the rest are Crank--Nicolson.
        rann = min(self.rannacher_steps, n_t)
        schedule: list[tuple[_Scheme, float]] = []
        for k in range(n_t):
            if k < rann:
                schedule.append((be, 0.5 * dt))
                schedule.append((be, 0.5 * dt))
            else:
                schedule.append((cn, dt))

        values = intrinsic.copy()
        prev_values = values
        final_scheme, final_dtau = cn, dt
        tau = 0.0
        for scheme, dtau in schedule:
            prev_values = values
            final_scheme, final_dtau = scheme, dtau
            values = _advance(
                scheme, values, boundary, tau, tau + dtau, american, obstacle, s_min, s_max
            )
            tau += dtau

        theta: float | None = None
        if with_theta:
            # Central difference of V in the time direction around today (tau = T):
            # one extra step past today gives V(T + dtau); prev_values is V(T - dtau).
            values_next = _advance(
                final_scheme, values, boundary, T, T + final_dtau, american, obstacle, s_min, s_max
            )
            i = grid.i_spot
            # dV/dtau ~ (V(T+dtau) - V(T-dtau)) / (2 dtau); theta = dV/dt = -dV/dtau.
            theta = -(float(values_next[i]) - float(prev_values[i])) / (2.0 * final_dtau)

        return _Solution(values=values, theta=theta)


def _make_scheme(
    lower: float, diag: float, upper: float, dt_step: float, implicitness: float, n_int: int
) -> _Scheme:
    """Build the banded implicit matrix and operator coefficients for one theta-step."""
    explicitness = 1.0 - implicitness
    m_lower = -implicitness * dt_step * lower
    m_diag = 1.0 - implicitness * dt_step * diag
    m_upper = -implicitness * dt_step * upper
    e_lower = explicitness * dt_step * lower
    e_diag = 1.0 + explicitness * dt_step * diag
    e_upper = explicitness * dt_step * upper
    ab = np.zeros((3, n_int))
    ab[0, 1:] = m_upper  # superdiagonal
    ab[1, :] = m_diag  # main diagonal
    ab[2, :-1] = m_lower  # subdiagonal
    return _Scheme(ab, m_lower, m_diag, m_upper, e_lower, e_diag, e_upper)


def _advance(
    scheme: _Scheme,
    values: FloatArray,
    boundary: Callable[[float, float], float],
    tau_now: float,
    tau_next: float,
    american: bool,
    obstacle: FloatArray,
    s_min: float,
    s_max: float,
) -> FloatArray:
    """Advance one theta-step (Crank--Nicolson or backward Euler); returns a new array.

    Applies the explicit operator ``N`` to the current level (with its Dirichlet
    boundaries), folds the next level's known boundary terms into the right-hand side,
    then solves ``M V^{n+1} = rhs`` (banded solve for European, projected SOR for the
    American LCP). The input array is not mutated.
    """
    cur = values.copy()
    cur[0] = boundary(s_min, tau_now)
    cur[-1] = boundary(s_max, tau_now)

    rhs = scheme.e_lower * cur[:-2] + scheme.e_diag * cur[1:-1] + scheme.e_upper * cur[2:]
    rhs[0] -= scheme.m_lower * boundary(s_min, tau_next)
    rhs[-1] -= scheme.m_upper * boundary(s_max, tau_next)

    out = np.empty_like(values)
    if american:
        out[1:-1] = _psor_solve(
            scheme.m_lower, scheme.m_diag, scheme.m_upper, rhs, obstacle, cur[1:-1]
        )
    else:
        out[1:-1] = solve_banded((1, 1), scheme.ab, rhs)
    out[0] = boundary(s_min, tau_next)
    out[-1] = boundary(s_max, tau_next)
    return out


def _psor_solve(
    lower: float,
    diag: float,
    upper: float,
    rhs: FloatArray,
    obstacle: FloatArray,
    warm: FloatArray,
) -> FloatArray:
    r"""Projected SOR for the American LCP on the interior nodes.

    Solves ``M v = rhs`` subject to ``v >= obstacle``, where ``M`` is the constant
    tridiagonal step matrix (sub-diagonal ``lower``, diagonal ``diag``, super-diagonal
    ``upper``). Each sweep is Gauss--Seidel with over-relaxation followed by the
    projection ``v_i <- max(v_i, obstacle_i)`` that enforces early exercise. ``warm``
    (the previous time step's interior values) seeds the iteration, so it converges in
    a handful of sweeps.

    Parameters
    ----------
    lower, diag, upper : float
        The constant sub-, main- and super-diagonal of the tridiagonal step matrix.
    rhs : ndarray
        The right-hand side (boundary contributions already folded in).
    obstacle : ndarray
        The early-exercise lower bound on each interior node.
    warm : ndarray
        Warm-start values (the previous step's interior solution).

    Returns
    -------
    ndarray
        The interior solution satisfying the complementarity conditions.

    Notes
    -----
    The work is done in Python floats: the sweep is inherently sequential (each node
    uses its just-updated neighbour), so a NumPy vectorised form buys nothing and the
    per-element overhead is lower this way.
    """
    n = rhs.size
    v = [max(float(warm[i]), float(obstacle[i])) for i in range(n)]  # feasible start
    b = rhs.tolist()
    g = obstacle.tolist()
    relax = _PSOR_RELAXATION
    inv_diag = 1.0 / diag

    for _ in range(_PSOR_MAX_ITER):
        err = 0.0
        for i in range(n):
            residual = b[i]
            if i > 0:
                residual -= lower * v[i - 1]
            if i < n - 1:
                residual -= upper * v[i + 1]
            candidate = v[i] + relax * (residual * inv_diag - v[i])
            if candidate < g[i]:
                candidate = g[i]
            change = abs(candidate - v[i])
            if change > err:
                err = change
            v[i] = candidate
        if err < _PSOR_TOL:
            break

    result: FloatArray = np.asarray(v, dtype=np.float64)
    return result
