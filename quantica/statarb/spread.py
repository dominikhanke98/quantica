r"""Spread modelling — the Ornstein--Uhlenbeck process and its half-life.

A cointegrated spread is mean-reverting, and the natural continuous-time model of mean
reversion is the **Ornstein--Uhlenbeck** process

.. math:: dX_t = \kappa(\mu - X_t)\,dt + \sigma\,dW_t,

with mean-reversion speed :math:`\kappa`, long-run mean :math:`\mu` and volatility
:math:`\sigma`. Sampled at a fixed step :math:`\Delta t` the OU process is *exactly* a
Gaussian AR(1),

.. math:: X_t = \mu(1-\phi) + \phi X_{t-1} + \varepsilon_t,\qquad \phi = e^{-\kappa\Delta t},

so the parameters are recovered from a single AR(1) regression of the spread on its own
lag — the estimation implemented here (leaning on ``numpy`` for the least-squares fit,
CLAUDE.md §3). The practically useful summary is the **half-life** of mean reversion,

.. math:: t_{1/2} = \frac{\ln 2}{\kappa},

the time for a deviation to decay halfway back to the mean — it sets the natural holding
period of the trade and screens pairs that revert too slowly to be tradeable.

References
----------
Uhlenbeck, G. E. & Ornstein, L. S. (1930). "On the theory of the Brownian motion",
*Physical Review* 36, 823--841.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = ["OUProcess", "estimate_ou_process", "ou_half_life"]


def ou_half_life(mean_reversion_speed: float) -> float:
    r"""The mean-reversion half-life :math:`\ln 2 / \kappa` (``inf`` if not mean-reverting)."""
    if mean_reversion_speed <= 0.0:
        return float("inf")
    return float(np.log(2.0) / mean_reversion_speed)


@dataclass(frozen=True)
class OUProcess:
    """Estimated Ornstein--Uhlenbeck parameters of a mean-reverting spread.

    Attributes
    ----------
    mean_reversion_speed : float
        :math:`\\kappa` — the speed of mean reversion (per unit time); ``<= 0`` means the
        fitted series is not mean-reverting.
    long_run_mean : float
        :math:`\\mu` — the level the spread reverts to.
    volatility : float
        :math:`\\sigma` — the instantaneous volatility of the OU process.
    half_life : float
        :math:`\\ln 2 / \\kappa` — time for a deviation to decay halfway (``inf`` if not
        mean-reverting).
    ar1_coefficient : float
        The fitted AR(1) autoregressive coefficient :math:`\\phi = e^{-\\kappa\\Delta t}`.
    dt : float
        The sampling step used in the estimation.
    """

    mean_reversion_speed: float
    long_run_mean: float
    volatility: float
    half_life: float
    ar1_coefficient: float
    dt: float

    @property
    def is_mean_reverting(self) -> bool:
        """Whether the fit implies mean reversion (``0 < phi < 1``, i.e. ``kappa > 0``)."""
        return self.mean_reversion_speed > 0.0


def estimate_ou_process(spread: FloatArray, *, dt: float = 1.0) -> OUProcess:
    r"""Estimate OU parameters from a spread by exact-discretisation AR(1) regression.

    Fits :math:`X_t = a + \phi X_{t-1} + \varepsilon_t` by least squares and inverts the
    exact OU--AR(1) mapping: :math:`\kappa = -\ln\phi/\Delta t`,
    :math:`\mu = a/(1-\phi)`, and :math:`\sigma^2 = 2\kappa\,\mathrm{Var}(\varepsilon)/(1-\phi^2)`.

    Parameters
    ----------
    spread : ndarray, shape (T,)
        The mean-reverting spread series (``T >= 3``).
    dt : float, optional
        Time between observations (default 1.0, i.e. parameters per period).

    Returns
    -------
    OUProcess
        The estimated speed, long-run mean, volatility, half-life and AR(1) coefficient. A
        non-mean-reverting fit (:math:`\phi \ge 1` or :math:`\phi \le 0`) yields
        ``mean_reversion_speed <= 0`` and an infinite half-life rather than an error.

    Raises
    ------
    ValueError
        If ``spread`` is not a 1-D series of length at least 3, or ``dt`` is not positive.
    """
    s = np.asarray(spread, dtype=np.float64)
    if s.ndim != 1 or s.shape[0] < 3:
        raise ValueError(f"spread must be a 1-D series of length >= 3, got shape {s.shape}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")

    lagged, current = s[:-1], s[1:]
    design = np.column_stack([np.ones_like(lagged), lagged])
    (intercept, phi), *_ = np.linalg.lstsq(design, current, rcond=None)
    residuals = current - design @ np.array([intercept, phi])
    resid_var = float(np.var(residuals, ddof=2))  # two estimated parameters

    if phi <= 0.0 or phi >= 1.0:  # not mean-reverting: leave kappa <= 0, half-life infinite
        kappa = -np.log(phi) / dt if phi > 0.0 else -1.0
        long_run_mean = float(intercept / (1.0 - phi)) if phi != 1.0 else float("nan")
        return OUProcess(
            mean_reversion_speed=float(kappa),
            long_run_mean=long_run_mean,
            volatility=float("nan"),
            half_life=float("inf"),
            ar1_coefficient=float(phi),
            dt=dt,
        )

    kappa = float(-np.log(phi) / dt)
    long_run_mean = float(intercept / (1.0 - phi))
    sigma = float(np.sqrt(resid_var * 2.0 * kappa / (1.0 - phi * phi)))
    return OUProcess(
        mean_reversion_speed=kappa,
        long_run_mean=long_run_mean,
        volatility=sigma,
        half_life=ou_half_life(kappa),
        ar1_coefficient=float(phi),
        dt=dt,
    )
