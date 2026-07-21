r"""Synthetic data for statistical-arbitrage validation, with a *known* structure.

The effective challenge for a cointegration test is to catch genuine cointegration **and**
reject spurious pairs, so the validation needs both truths on demand:

* :func:`generate_cointegrated_pair` — two series sharing a common stochastic trend plus a
  stationary spread, so ``y - beta*x`` is stationary *by construction* (the null of no
  cointegration is false).
* :func:`generate_independent_random_walks` — independent :math:`I(1)` walks with no shared
  trend, the textbook *spurious* case (the null is true; a good test must not reject it).
* :func:`simulate_ou_process` — an exact-discretisation Ornstein--Uhlenbeck path with known
  parameters, so OU estimation can be checked against ground truth.

Randomness is always an injected, seeded :class:`numpy.random.Generator` (CLAUDE.md §3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "generate_cointegrated_pair",
    "generate_independent_random_walks",
    "simulate_ou_process",
]


def generate_cointegrated_pair(
    n_obs: int,
    rng: np.random.Generator,
    *,
    beta: float = 1.5,
    alpha: float = 0.0,
    trend_vol: float = 1.0,
    spread_kappa: float = 0.1,
    spread_vol: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
    r"""Two cointegrated series: a shared random-walk trend plus a stationary OU spread.

    Builds ``x`` as a random walk (:math:`I(1)`) and ``y = alpha + beta*x + spread`` with a
    stationary Ornstein--Uhlenbeck ``spread``, so ``y - beta*x`` is stationary and the pair
    is cointegrated with vector ``[1, -beta]``.

    Parameters
    ----------
    n_obs : int
        Series length (``>= 2``).
    rng : numpy.random.Generator
        Seeded generator.
    beta : float, optional
        The true hedge ratio (default 1.5).
    alpha : float, optional
        The spread's mean offset (default 0).
    trend_vol : float, optional
        Innovation volatility of the common random-walk trend (default 1.0).
    spread_kappa : float, optional
        Mean-reversion speed of the stationary spread (default 0.1 -> half-life ~ 6.9).
    spread_vol : float, optional
        Volatility of the stationary spread (default 1.0).

    Returns
    -------
    tuple of ndarray
        ``(y, x)``, each of shape ``(n_obs,)``.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be at least 2, got {n_obs}")
    x = np.cumsum(rng.normal(0.0, trend_vol, size=n_obs))
    spread = simulate_ou_process(
        n_obs, rng, kappa=spread_kappa, mu=alpha, sigma=spread_vol, x0=alpha
    )
    y = beta * x + spread
    return np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)


def generate_independent_random_walks(
    n_obs: int, n_series: int, rng: np.random.Generator, *, vol: float = 1.0
) -> FloatArray:
    r"""Independent :math:`I(1)` random walks — the spurious (non-cointegrated) case.

    Parameters
    ----------
    n_obs : int
        Series length (``>= 2``).
    n_series : int
        Number of independent walks (``>= 1``).
    rng : numpy.random.Generator
        Seeded generator.
    vol : float, optional
        Per-step innovation volatility (default 1.0).

    Returns
    -------
    ndarray, shape (n_obs, n_series)
        The independent random walks in columns.
    """
    if n_obs < 2:
        raise ValueError(f"n_obs must be at least 2, got {n_obs}")
    if n_series < 1:
        raise ValueError(f"n_series must be at least 1, got {n_series}")
    innovations = rng.normal(0.0, vol, size=(n_obs, n_series))
    return np.asarray(np.cumsum(innovations, axis=0), dtype=np.float64)


def simulate_ou_process(
    n_obs: int,
    rng: np.random.Generator,
    *,
    kappa: float,
    mu: float,
    sigma: float,
    dt: float = 1.0,
    x0: float | None = None,
) -> FloatArray:
    r"""Exact-discretisation Ornstein--Uhlenbeck path with known parameters.

    Uses the exact transition :math:`X_t = \mu + (X_{t-1}-\mu)e^{-\kappa\Delta t} +
    \varepsilon_t` with :math:`\varepsilon_t \sim \mathcal N(0, \sigma^2(1-e^{-2\kappa
    \Delta t})/(2\kappa))`, so the simulated marginals carry no discretisation bias.

    Parameters
    ----------
    n_obs : int
        Path length (``>= 1``).
    rng : numpy.random.Generator
        Seeded generator.
    kappa : float
        Mean-reversion speed (``> 0``).
    mu : float
        Long-run mean.
    sigma : float
        Instantaneous volatility.
    dt : float, optional
        Time step (default 1.0).
    x0 : float, optional
        Initial value (default the long-run mean ``mu``).

    Returns
    -------
    ndarray, shape (n_obs,)
        The simulated OU path.
    """
    if n_obs < 1:
        raise ValueError(f"n_obs must be at least 1, got {n_obs}")
    if kappa <= 0.0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    phi = np.exp(-kappa * dt)
    step_var = sigma * sigma * (1.0 - phi * phi) / (2.0 * kappa)
    step_sd = np.sqrt(step_var)
    shocks = rng.normal(0.0, step_sd, size=n_obs)
    path = np.empty(n_obs, dtype=np.float64)
    path[0] = mu if x0 is None else x0
    for t in range(1, n_obs):
        path[t] = mu + (path[t - 1] - mu) * phi + shocks[t]
    return path
