r"""Dynamic hedge ratio via a Kalman filter — a time-varying cointegrating coefficient.

The Engle--Granger / Johansen coefficient of :mod:`~quantica.statarb.cointegration` is
**static**: it assumes the hedge ratio between two assets is constant over the whole
sample. Real relationships drift — sector rotation, changing fundamentals, liquidity — so a
single number fitted once is a compromise between the early and late regimes and tracks
*neither* well. The Kalman filter treats the hedge ratio as an **unobserved state that
evolves**, re-estimated as each new price arrives: more realistic, and a genuine
quant-craft signal.

State-space formulation
-----------------------
The pair is written as a linear Gaussian state-space model. The latent state is the hedge
ratio (and, optionally, an intercept), following a random walk; the observation relates one
price to the other through the current state:

.. math::

    \beta_t = \beta_{t-1} + w_t, \quad w_t \sim \mathcal N(0, Q)
    \qquad\text{(state: the drifting hedge ratio)}

    y_t = H_t\,\beta_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal N(0, R),
    \quad H_t = [\,x_t,\ 1\,]
    \qquad\text{(observation)}

The standard predict/update recursion is implemented directly (:func:`kalman_hedge_ratio`),
in numpy, with no external filtering library — the recursion *is* the demonstrable skill.
Each step yields the filtered hedge ratio, its uncertainty, and the one-step-ahead
prediction error :math:`e_t = y_t - H_t\beta_{t-1}` — the **dynamic spread**, the innovation
series a mean-reversion strategy trades.

The tuning knob
---------------
The process variance :math:`Q = q\,I` (state-evolution) and the observation variance
:math:`R = r` are exposed, not hard-coded. Their **ratio** :math:`q/r` is the signal-to-noise
knob: a *large* ratio lets the hedge ratio adapt quickly (it trusts recent observations over
the prior state), a *small* ratio pins it down (in the limit :math:`q \to 0` the filter is
recursive least squares and its estimate converges to the static OLS coefficient). There is
no universally correct value — it trades responsiveness against noise — so it is a documented
parameter, tuned to the pair's drift speed, rather than a magic constant.

References
----------
Kalman, R. E. (1960). "A new approach to linear filtering and prediction problems",
*J. Basic Engineering* 82, 35--45.
Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*, Wiley
(the pairs-trading Kalman formulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = ["KalmanHedgeResult", "kalman_hedge_ratio"]


@dataclass(frozen=True)
class KalmanHedgeResult:
    """Filtered output of the dynamic-hedge-ratio Kalman filter.

    Attributes
    ----------
    hedge_ratio : ndarray, shape (T,)
        The filtered time-varying hedge ratio :math:`\\beta_t` (the loading on ``x``).
    intercept : ndarray, shape (T,)
        The filtered intercept path (all zeros when ``fit_intercept=False``).
    spread : ndarray, shape (T,)
        The one-step-ahead prediction error :math:`e_t = y_t - H_t\\beta_{t-1}` — the
        dynamic spread / innovation series.
    innovation_variance : ndarray, shape (T,)
        The variance :math:`S_t` of each innovation; ``spread / sqrt(innovation_variance)``
        is the standardised innovation (white when the model fits).
    hedge_ratio_var : ndarray, shape (T,)
        The filtered variance of the hedge-ratio state, :math:`P_t[0,0]`; its square root is
        the one-sigma uncertainty band on :attr:`hedge_ratio`.
    process_var : float
        The state-evolution variance :math:`q` used (diagonal of ``Q``).
    obs_var : float
        The observation variance :math:`r` used (``R``).
    """

    hedge_ratio: FloatArray
    intercept: FloatArray
    spread: FloatArray
    innovation_variance: FloatArray
    hedge_ratio_var: FloatArray
    process_var: float
    obs_var: float

    @property
    def hedge_ratio_std(self) -> FloatArray:
        """One-sigma uncertainty band on the filtered hedge ratio."""
        return np.sqrt(self.hedge_ratio_var)


def kalman_hedge_ratio(
    y: FloatArray,
    x: FloatArray,
    *,
    process_var: float,
    obs_var: float,
    fit_intercept: bool = True,
    initial_state: FloatArray | None = None,
    initial_cov: float = 1.0e5,
) -> KalmanHedgeResult:
    r"""Estimate a time-varying hedge ratio with a Kalman filter (predict/update).

    Runs the linear-Gaussian filter for the state-space model in the module docstring: the
    hedge ratio (and optional intercept) is a random-walk state; ``y`` is observed as
    :math:`H_t\beta_t + \varepsilon_t` with :math:`H_t = [x_t, 1]`. A diffuse prior
    (``initial_cov`` large) lets the estimate lock on quickly regardless of the starting
    guess.

    Parameters
    ----------
    y, x : ndarray, shape (T,)
        The two price series; ``y`` is explained by ``x`` through the hedge ratio.
    process_var : float
        The state-evolution variance :math:`q` (diagonal of ``Q``), non-negative. Larger
        values let the hedge ratio adapt faster (see the module docstring on ``q/r``).
    obs_var : float
        The observation variance :math:`r` (``R``), positive.
    fit_intercept : bool, optional
        Include a drifting intercept in the state (default ``True``); if ``False`` the state
        is the hedge ratio alone.
    initial_state : ndarray, optional
        Initial state ``[hedge_ratio, intercept]`` (or ``[hedge_ratio]``); default zeros.
    initial_cov : float, optional
        Initial state-covariance scale (a diffuse prior); default ``1e5``.

    Returns
    -------
    KalmanHedgeResult
        The filtered hedge-ratio path, intercept path, dynamic spread (innovations), and
        the uncertainty series.

    Raises
    ------
    ValueError
        If ``y`` and ``x`` are not 1-D of equal length, the noise parameters are invalid, or
        ``initial_state`` has the wrong shape.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if y_arr.ndim != 1 or x_arr.ndim != 1:
        raise ValueError("y and x must be 1-D series")
    if y_arr.shape != x_arr.shape:
        raise ValueError(f"y and x must have equal length, got {y_arr.shape} vs {x_arr.shape}")
    if process_var < 0.0:
        raise ValueError(f"process_var must be non-negative, got {process_var}")
    if obs_var <= 0.0:
        raise ValueError(f"obs_var must be positive, got {obs_var}")

    dim = 2 if fit_intercept else 1
    if initial_state is None:
        beta = np.zeros(dim, dtype=np.float64)
    else:
        beta = np.asarray(initial_state, dtype=np.float64)
        if beta.shape != (dim,):
            raise ValueError(f"initial_state must have shape ({dim},), got {beta.shape}")

    n = y_arr.shape[0]
    cov = np.eye(dim, dtype=np.float64) * initial_cov
    process_cov = np.eye(dim, dtype=np.float64) * process_var

    hedge_ratio = np.empty(n, dtype=np.float64)
    intercept = np.zeros(n, dtype=np.float64)
    spread = np.empty(n, dtype=np.float64)
    innovation_variance = np.empty(n, dtype=np.float64)
    hedge_ratio_var = np.empty(n, dtype=np.float64)

    for t in range(n):
        obs_row = np.array([x_arr[t], 1.0]) if fit_intercept else np.array([x_arr[t]])

        # Predict: the state is a random walk (transition = identity), so only the
        # covariance grows by the process noise.
        cov = cov + process_cov

        # Update against the new observation.
        innovation = float(y_arr[t] - obs_row @ beta)  # one-step-ahead error = the spread
        innovation_var = float(obs_row @ cov @ obs_row + obs_var)
        gain = cov @ obs_row / innovation_var
        beta = beta + gain * innovation
        cov = cov - np.outer(gain, obs_row) @ cov

        hedge_ratio[t] = beta[0]
        if fit_intercept:
            intercept[t] = beta[1]
        spread[t] = innovation
        innovation_variance[t] = innovation_var
        hedge_ratio_var[t] = cov[0, 0]

    return KalmanHedgeResult(
        hedge_ratio=hedge_ratio,
        intercept=intercept,
        spread=spread,
        innovation_variance=innovation_variance,
        hedge_ratio_var=hedge_ratio_var,
        process_var=process_var,
        obs_var=obs_var,
    )
