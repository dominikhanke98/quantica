r"""Crank--Nicolson finite-difference engine for European options.

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

Stability caveat (relevant only for future PDE Greeks)
------------------------------------------------------
Crank--Nicolson is A-stable but **not L-stable**: it damps high-frequency error
modes only weakly. The vanilla payoff has a kink at the strike (and, for
digitals, a jump), so those non-smooth initial data excite high-frequency modes
that CN fails to damp, producing spurious oscillations concentrated near the
strike. The *price* is smooth enough that this is invisible at the tolerances
here, but higher-order Greeks (gamma, and worse) read off a CN grid can ring.
The standard remedy is **Rannacher start-up**: replace the first one or two CN
steps with two half-steps of fully implicit (backward) Euler, which is L-stable
and annihilates the offending modes before CN takes over, restoring second-order
accuracy with clean Greeks. This engine prices only (no PDE Greeks yet), so
Rannacher is not implemented — noted here so it is applied if/when PDE
sensitivities are added. See Giles & Carter (2006), "Convergence analysis of
Crank--Nicolson and Rannacher time-marching".

References
----------
Wilmott, Howison & Dewynne (1995), *The Mathematics of Financial Derivatives*,
ch. on finite differences. Duffy (2006), *Finite Difference Methods in Financial
Engineering*.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import solve_banded

from quantica.core.types import FloatArray
from quantica.pricing.engines._common import unpack

if TYPE_CHECKING:
    from quantica.pricing.instruments import EuropeanOption
    from quantica.pricing.processes import BlackScholesProcess

_DEFAULT_SPACE_STEPS = 200
_DEFAULT_TIME_STEPS = 200
_DEFAULT_NUM_STD = 6.0


class FiniteDifferenceEngine:
    """Crank--Nicolson PDE pricer for a :class:`EuropeanOption`.

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

    Notes
    -----
    Satisfies the :class:`~quantica.pricing.engines.PricingEngine` protocol
    (price only). Second-order accurate: halving both step sizes cuts the error
    by ~4.
    """

    def __init__(
        self,
        space_steps: int = _DEFAULT_SPACE_STEPS,
        time_steps: int = _DEFAULT_TIME_STEPS,
        num_std: float = _DEFAULT_NUM_STD,
    ) -> None:
        if space_steps < 2:
            raise ValueError(f"space_steps must be at least 2, got {space_steps}")
        if time_steps < 1:
            raise ValueError(f"time_steps must be at least 1, got {time_steps}")
        if num_std <= 0.0:
            raise ValueError(f"num_std must be positive, got {num_std}")
        self.space_steps = space_steps
        self.time_steps = time_steps
        self.num_std = num_std

    def calculate(
        self,
        instrument: EuropeanOption,
        process: BlackScholesProcess,
    ) -> float:
        """Present value of ``instrument`` under ``process`` via Crank--Nicolson."""
        S, K, r, q, sigma, T, omega = unpack(instrument, process)

        # Degenerate (no diffusion) limit: discounted intrinsic on the forward,
        # matching the analytic sigma->0 / T->0 result.
        if sigma == 0.0 or T == 0.0:
            return max(omega * (S * math.exp(-q * T) - K * math.exp(-r * T)), 0.0)

        # -- log-price grid, centred so ln(spot) is a node ------------------- #
        ln_spot = math.log(S)
        sig_sqrt_t = sigma * math.sqrt(T)
        half_width = max(self.num_std * sig_sqrt_t, abs(math.log(K) - ln_spot) + 3.0 * sig_sqrt_t)

        half_nodes = self.space_steps // 2
        dx = half_width / half_nodes
        n_x = 2 * half_nodes  # even number of intervals
        i_spot = half_nodes  # node holding ln(spot)
        x = ln_spot + (np.arange(n_x + 1) - i_spot) * dx
        spot_grid = np.exp(x)

        n_t = self.time_steps
        dt = T / n_t

        # -- spatial operator coefficients (constant in x and tau) ----------- #
        a = 0.5 * sigma * sigma  # diffusion
        b = r - q - 0.5 * sigma * sigma  # convection
        lower = a / (dx * dx) - b / (2.0 * dx)
        diag = -2.0 * a / (dx * dx) - r
        upper = a / (dx * dx) + b / (2.0 * dx)

        # Crank--Nicolson: (I - dt/2 L) V^{n+1} = (I + dt/2 L) V^n.
        m_lower = -0.5 * dt * lower
        m_diag = 1.0 - 0.5 * dt * diag
        m_upper = -0.5 * dt * upper
        e_lower = 0.5 * dt * lower
        e_diag = 1.0 + 0.5 * dt * diag
        e_upper = 0.5 * dt * upper

        # Implicit tridiagonal matrix in banded form for the interior unknowns.
        n_int = n_x - 1
        ab = np.zeros((3, n_int))
        ab[0, 1:] = m_upper  # superdiagonal
        ab[1, :] = m_diag  # main diagonal
        ab[2, :-1] = m_lower  # subdiagonal

        values: FloatArray = np.maximum(omega * (spot_grid - K), 0.0)  # payoff at tau = 0
        s_min = float(spot_grid[0])
        s_max = float(spot_grid[-1])

        def boundary(spot: float, tau: float) -> float:
            return max(omega * (spot * math.exp(-q * tau) - K * math.exp(-r * tau)), 0.0)

        for n in range(n_t):
            tau_now = n * dt
            tau_next = (n + 1) * dt
            values[0] = boundary(s_min, tau_now)
            values[-1] = boundary(s_max, tau_now)

            interior = values[1:-1]
            rhs = e_lower * values[:-2] + e_diag * interior + e_upper * values[2:]
            # Move the known next-level boundary terms of the implicit operator
            # to the right-hand side.
            rhs[0] -= m_lower * boundary(s_min, tau_next)
            rhs[-1] -= m_upper * boundary(s_max, tau_next)

            values[1:-1] = solve_banded((1, 1), ab, rhs)
            values[0] = boundary(s_min, tau_next)
            values[-1] = boundary(s_max, tau_next)

        return float(values[i_spot])
