r"""Synthetic time-series data with a *known* structure, for forecast-evaluation validation.

The forecast-evaluation layer (:mod:`~quantica.timeseries.evaluation`) is only trustworthy if
its statistics have the right properties, so the validation needs ground truth on demand:

* :func:`simulate_garch` — a return series from a **known** GARCH / GJR-GARCH / EGARCH process,
  so the estimation leaned on (:mod:`arch`) can be checked against the true parameters, and a
  correctly-specified model can be pitted against a misspecified one.
* :func:`simulate_loss_differential` — a serially-correlated forecast-**loss differential** with
  a known mean and autocorrelation. This is the fixture that lets us verify the Diebold--Mariano
  test's *size* (mean zero ⇒ should reject at the nominal rate) and *power* (mean shifted ⇒
  should reject), and demonstrate why the HAC variance correction is needed — a positively
  autocorrelated differential breaks the naive variance, exactly the case that arises because
  volatility forecast errors cluster.
* :func:`simulate_markov_switching` — a return series from a **known** Gaussian regime-switching
  process, returning the hidden state path so a fitted model can be checked both on parameter
  recovery and on how well its smoothed probabilities classify the regime each point was really in.

Randomness is always an injected, seeded :class:`numpy.random.Generator` (CLAUDE.md §3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray, IntArray

__all__ = [
    "simulate_garch",
    "simulate_loss_differential",
    "simulate_markov_switching",
]

_SQRT_2_OVER_PI = np.sqrt(2.0 / np.pi)  # E|z| for a standard normal, the EGARCH centering


@overload
def simulate_garch(
    n: int,
    *,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float = ...,
    model: Literal["GARCH", "GJR", "EGARCH"] = ...,
    mu: float = ...,
    rng: np.random.Generator,
    n_burn: int = ...,
    return_variance: Literal[False] = ...,
) -> FloatArray: ...


@overload
def simulate_garch(
    n: int,
    *,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float = ...,
    model: Literal["GARCH", "GJR", "EGARCH"] = ...,
    mu: float = ...,
    rng: np.random.Generator,
    n_burn: int = ...,
    return_variance: Literal[True],
) -> tuple[FloatArray, FloatArray]: ...


def simulate_garch(
    n: int,
    *,
    omega: float,
    alpha: float,
    beta: float,
    gamma: float = 0.0,
    model: Literal["GARCH", "GJR", "EGARCH"] = "GARCH",
    mu: float = 0.0,
    rng: np.random.Generator,
    n_burn: int = 1000,
    return_variance: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray]:
    r"""Simulate a return series from a known GARCH-family process.

    The recursions match :mod:`arch`'s constant-mean, normal-innovation parameterisation, so a
    fit of the same model to the output recovers ``(omega, alpha, beta, gamma)`` (the known-truth
    anchor). With :math:`\varepsilon_t = \sigma_t z_t`, :math:`z_t \sim N(0,1)`:

    * ``GARCH``:  :math:`\sigma_t^2 = \omega + \alpha\varepsilon_{t-1}^2 + \beta\sigma_{t-1}^2`.
    * ``GJR``:  adds a leverage term
      :math:`\gamma\,\varepsilon_{t-1}^2\mathbf{1}\{\varepsilon_{t-1}<0\}` (bad news lifts vol
      more).
    * ``EGARCH``:
      :math:`\ln\sigma_t^2 = \omega + \alpha(|z_{t-1}|-\sqrt{2/\pi}) + \gamma z_{t-1}
      + \beta\ln\sigma_{t-1}^2` (asymmetry via the signed :math:`\gamma z_{t-1}`).

    Parameters
    ----------
    n : int
        Number of returns to return (after burn-in).
    omega, alpha, beta : float
        The GARCH constant, ARCH and lagged-variance coefficients. For a stationary ``GARCH`` /
        ``GJR`` process ``alpha + beta (+ gamma/2) < 1``; for ``EGARCH`` ``|beta| < 1``.
    gamma : float, optional
        The leverage/asymmetry coefficient (default 0, i.e. symmetric). Ignored for ``GARCH``.
    model : {"GARCH", "GJR", "EGARCH"}, optional
        Which process to simulate (default ``"GARCH"``).
    mu : float, optional
        The constant mean return (default 0).
    rng : numpy.random.Generator
        Seeded generator (keyword-only).
    n_burn : int, optional
        Burn-in samples discarded so the variance recursion forgets its start (default 1000).
    return_variance : bool, optional
        If ``True``, also return the true conditional-variance path :math:`\sigma_t^2` — the
        *oracle* forecast, used to validate that QLIKE ranks the true variance best and that the
        Diebold--Mariano test detects a genuinely better forecast (default ``False``).

    Returns
    -------
    ndarray or tuple of ndarray
        The simulated return series of shape ``(n,)``; or, when ``return_variance=True``, the pair
        ``(returns, conditional_variance)``.

    Raises
    ------
    ValueError
        If ``n`` is not positive or ``model`` is unknown.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if model not in ("GARCH", "GJR", "EGARCH"):
        raise ValueError(f"unknown model {model!r}")

    total = n + n_burn
    z = rng.standard_normal(total)
    returns = np.empty(total, dtype=np.float64)
    variance = np.empty(total, dtype=np.float64)

    if model == "EGARCH":
        log_var = omega / (1.0 - beta) if beta != 1.0 else 0.0  # stationary start
        prev_z = 0.0
        for t in range(total):
            log_var = (
                omega + alpha * (abs(prev_z) - _SQRT_2_OVER_PI) + gamma * prev_z + beta * log_var
            )
            variance[t] = np.exp(log_var)
            returns[t] = mu + np.sqrt(variance[t]) * z[t]
            prev_z = z[t]
    else:
        uncond = omega / max(1.0 - alpha - beta - 0.5 * gamma, 1e-6)
        var = uncond
        prev_eps = 0.0
        for t in range(total):
            leverage = gamma * prev_eps * prev_eps if (model == "GJR" and prev_eps < 0.0) else 0.0
            var = omega + alpha * prev_eps * prev_eps + leverage + beta * var
            variance[t] = var
            eps = np.sqrt(var) * z[t]
            returns[t] = mu + eps
            prev_eps = eps

    if return_variance:
        return returns[n_burn:], variance[n_burn:]
    return returns[n_burn:]


def simulate_loss_differential(
    n: int,
    *,
    mean: float,
    phi: float,
    sigma: float = 1.0,
    rng: np.random.Generator,
) -> FloatArray:
    r"""Simulate a serially-correlated forecast-loss differential as a Gaussian AR(1).

    Produces :math:`d_t = \mu + u_t`, :math:`u_t = \phi u_{t-1} + \sigma\,e_t`,
    :math:`e_t \sim N(0,1)` — a loss differential with a **known** mean :math:`\mu` and a known
    autocorrelation :math:`\phi`. Setting ``mean=0`` gives two equally-accurate forecasts (the
    Diebold--Mariano *size* case); ``mean>0`` gives a genuinely worse first forecast (the *power*
    case). A positive ``phi`` is the realistic case for volatility forecasts — clustered errors
    make the differential autocorrelated, which is precisely what the HAC variance correction
    must handle and the naive variance cannot.

    Parameters
    ----------
    n : int
        Length of the series.
    mean : float
        The true expected loss differential :math:`\mu` (0 under equal accuracy).
    phi : float
        AR(1) coefficient of the noise (``|phi| < 1``); the serial correlation the HAC
        correction exists to absorb.
    sigma : float, optional
        Innovation standard deviation (default 1).
    rng : numpy.random.Generator
        Seeded generator (keyword-only).

    Returns
    -------
    ndarray, shape (n,)
        The simulated loss-differential series.

    Raises
    ------
    ValueError
        If ``n`` is not positive or ``|phi| >= 1``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if abs(phi) >= 1.0:
        raise ValueError("require |phi| < 1 for a stationary AR(1)")

    e = rng.standard_normal(n)
    u = np.empty(n, dtype=np.float64)
    u[0] = e[0] / np.sqrt(1.0 - phi * phi)  # draw from the stationary distribution
    for t in range(1, n):
        u[t] = phi * u[t - 1] + e[t]
    return np.asarray(mean + sigma * u, dtype=np.float64)


def simulate_markov_switching(
    n: int,
    means: FloatArray,
    variances: FloatArray,
    transition_matrix: FloatArray,
    *,
    rng: np.random.Generator,
    initial_state: int = 0,
) -> tuple[FloatArray, IntArray]:
    r"""Simulate a Gaussian Markov regime-switching series, returning the hidden states.

    A latent state :math:`s_t` follows a Markov chain with transition matrix
    ``P[i, j] = P(s_t = j | s_{t-1} = i)``, and each observation is drawn
    :math:`y_t \sim N(\mu_{s_t}, \sigma^2_{s_t})`. Returning the true state path is what makes the
    known-truth validation possible: a fitted model must recover the parameters *and* classify each
    observation into the regime it was actually generated from.

    Parameters
    ----------
    n : int
        Number of observations.
    means, variances : ndarray, shape (K,)
        The per-regime means and (positive) variances.
    transition_matrix : ndarray, shape (K, K)
        Row-stochastic transition matrix.
    rng : numpy.random.Generator
        Seeded generator (keyword-only).
    initial_state : int, optional
        The state at ``t = 0`` (default 0).

    Returns
    -------
    tuple of ndarray
        ``(returns, states)`` — the observed series and the integer hidden-state path, each of
        length ``n``.

    Raises
    ------
    ValueError
        If ``n`` is not positive or ``transition_matrix`` is not square with matching dimensions.
    """
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    p_matrix = np.asarray(transition_matrix, dtype=np.float64)
    k = means.size
    if n <= 0:
        raise ValueError("n must be positive")
    if p_matrix.shape != (k, k) or variances.size != k:
        raise ValueError("means, variances and transition_matrix dimensions must agree")

    states = np.empty(n, dtype=np.intp)
    states[0] = initial_state
    for t in range(1, n):
        states[t] = rng.choice(k, p=p_matrix[states[t - 1]])
    returns = rng.normal(means[states], np.sqrt(variances[states]))
    return np.asarray(returns, dtype=np.float64), np.asarray(states, dtype=np.intp)
