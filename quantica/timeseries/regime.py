r"""Markov regime-switching models — the Hamilton filter, Kim smoother and EM, hand-implemented.

A regime-switching model lets the data-generating parameters jump between a small number of hidden
**states** governed by a Markov chain — the canonical description of markets that alternate between
a *calm* regime (low volatility) and a *crisis* regime (high volatility). The inference machinery
is a clean, self-contained algorithm, so it is hand-implemented here (CLAUDE.md §3); only the
Gaussian densities and linear algebra lean on :mod:`numpy`.

The pieces, for a ``K``-state Gaussian switching mean/variance model:

* :func:`hamilton_filter` — the forward recursion giving the **filtered** state probabilities
  :math:`P(s_t = k \mid y_{1:t})` and the log-likelihood (Hamilton 1989).
* :func:`kim_smoother` — the backward pass giving the **smoothed** probabilities
  :math:`P(s_t = k \mid y_{1:T})`, using all the data (Kim 1994).
* :func:`fit_markov_switching` — **EM** (Baum--Welch) estimation of the state means, variances,
  transition matrix and initial distribution, with the Gaussian M-step in closed form (so no inner
  optimiser is needed) and multiple random starts to guard against local optima and label
  switching. The expectation-maximisation log-likelihood is non-decreasing by construction.

References
----------
Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of Nonstationary Time Series and
the Business Cycle." *Econometrica* 57(2). Kim, C.-J. (1994). "Dynamic linear models with
Markov-switching." *Journal of Econometrics* 60.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "HamiltonFilterResult",
    "MarkovSwitchingResult",
    "fit_markov_switching",
    "hamilton_filter",
    "kim_smoother",
]

_VARIANCE_FLOOR = 1e-8  # keeps a collapsing regime's variance away from zero
_TINY = 1e-300  # guards divisions by a vanishing predicted probability


def _gaussian_densities(
    returns: FloatArray, means: FloatArray, variances: FloatArray
) -> FloatArray:
    """The ``(T, K)`` matrix of Gaussian densities of each observation under each regime."""
    y = returns[:, None]
    resid = y - means[None, :]
    return np.asarray(
        np.exp(-0.5 * resid * resid / variances[None, :])
        / np.sqrt(2.0 * np.pi * variances[None, :]),
        dtype=np.float64,
    )


def _stationary_distribution(transition_matrix: FloatArray) -> FloatArray:
    """The ergodic (stationary) distribution ``pi`` solving ``pi P = pi``, ``sum pi = 1``."""
    k = transition_matrix.shape[0]
    system = np.vstack([transition_matrix.T - np.eye(k), np.ones(k)])
    rhs = np.append(np.zeros(k), 1.0)
    solution, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    return np.asarray(np.clip(solution, 0.0, None) / np.clip(solution, 0.0, None).sum(), np.float64)


@dataclass(frozen=True)
class HamiltonFilterResult:
    """The output of the Hamilton forward filter.

    Attributes
    ----------
    filtered : ndarray, shape (T, K)
        Filtered state probabilities :math:`P(s_t = k \\mid y_{1:t})`.
    predicted : ndarray, shape (T, K)
        One-step-ahead predicted probabilities :math:`P(s_t = k \\mid y_{1:t-1})`.
    loglikelihood : float
        The log-likelihood :math:`\\sum_t \\log p(y_t \\mid y_{1:t-1})`.
    """

    filtered: FloatArray
    predicted: FloatArray
    loglikelihood: float


def hamilton_filter(
    returns: FloatArray,
    means: FloatArray,
    variances: FloatArray,
    transition_matrix: FloatArray,
    *,
    initial: FloatArray | None = None,
) -> HamiltonFilterResult:
    r"""Hamilton (1989) forward filter for a Gaussian Markov-switching model.

    For each ``t`` the filter combines the one-step prediction with the new observation's
    likelihood under each regime, returning the filtered probabilities and accumulating the
    log-likelihood — the object EM maximises and the input to the smoother.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The observed series.
    means, variances : ndarray, shape (K,)
        The regime means and (strictly positive) variances.
    transition_matrix : ndarray, shape (K, K)
        Row-stochastic transition matrix ``P[i, j] = P(s_t = j | s_{t-1} = i)``.
    initial : ndarray, shape (K,), optional
        The initial state distribution at ``t = 0``. Defaults to the chain's stationary
        distribution.

    Returns
    -------
    HamiltonFilterResult
        The filtered and predicted probabilities and the log-likelihood.
    """
    y = np.asarray(returns, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    p_matrix = np.asarray(transition_matrix, dtype=np.float64)
    n, k = y.size, means.size

    densities = _gaussian_densities(y, means, variances)
    filtered = np.empty((n, k), dtype=np.float64)
    predicted = np.empty((n, k), dtype=np.float64)
    prior = (
        _stationary_distribution(p_matrix) if initial is None else np.asarray(initial, np.float64)
    )

    loglik = 0.0
    for t in range(n):
        predicted[t] = prior
        joint = prior * densities[t]
        marginal = joint.sum()
        loglik += np.log(max(marginal, _TINY))
        filtered[t] = joint / max(marginal, _TINY)
        prior = filtered[t] @ p_matrix
    return HamiltonFilterResult(filtered=filtered, predicted=predicted, loglikelihood=float(loglik))


def kim_smoother(
    filtered: FloatArray, predicted: FloatArray, transition_matrix: FloatArray
) -> FloatArray:
    r"""Kim (1994) backward smoother giving :math:`P(s_t = k \mid y_{1:T})`.

    Refines each filtered distribution using the whole sample by a single backward pass over the
    output of :func:`hamilton_filter`.

    Parameters
    ----------
    filtered, predicted : ndarray, shape (T, K)
        The filtered and predicted probabilities from :func:`hamilton_filter`.
    transition_matrix : ndarray, shape (K, K)
        The same row-stochastic transition matrix used in the filter.

    Returns
    -------
    ndarray, shape (T, K)
        The smoothed state probabilities.
    """
    filt = np.asarray(filtered, dtype=np.float64)
    pred = np.asarray(predicted, dtype=np.float64)
    p_matrix = np.asarray(transition_matrix, dtype=np.float64)
    n = filt.shape[0]

    smoothed = np.empty_like(filt)
    smoothed[-1] = filt[-1]
    for t in range(n - 2, -1, -1):
        ratio = smoothed[t + 1] / np.maximum(pred[t + 1], _TINY)
        smoothed[t] = filt[t] * (p_matrix @ ratio)
        smoothed[t] /= smoothed[t].sum()
    return smoothed


@dataclass(frozen=True)
class MarkovSwitchingResult:
    """A fitted ``K``-state Gaussian Markov-switching model, regimes ordered by variance.

    Regimes are sorted by variance ascending, so regime ``0`` is the lowest-variance (*calm*)
    state and regime ``K-1`` the
    highest-variance (*crisis*) state, which removes the label-switching ambiguity.

    Attributes
    ----------
    means, variances : ndarray, shape (K,)
        The fitted regime means and variances (variances sorted ascending).
    transition_matrix : ndarray, shape (K, K)
        The fitted row-stochastic transition matrix.
    initial_distribution : ndarray, shape (K,)
        The fitted initial (t=0) state distribution.
    stationary_distribution : ndarray, shape (K,)
        The ergodic distribution implied by ``transition_matrix`` (the unconditional regime
        frequencies).
    filtered_probabilities, smoothed_probabilities : ndarray, shape (T, K)
        The filtered and smoothed state probabilities at the fitted parameters.
    loglikelihood : float
        The maximised log-likelihood.
    loglikelihood_history : ndarray
        The log-likelihood at each EM iteration of the best start (non-decreasing).
    n_iter : int
        EM iterations taken by the best start.
    converged : bool
        Whether the best start met the tolerance before the iteration cap.
    """

    means: FloatArray
    variances: FloatArray
    transition_matrix: FloatArray
    initial_distribution: FloatArray
    stationary_distribution: FloatArray
    filtered_probabilities: FloatArray
    smoothed_probabilities: FloatArray
    loglikelihood: float
    loglikelihood_history: FloatArray
    n_iter: int
    converged: bool

    def most_likely_states(self) -> FloatArray:
        """The most probable regime for each observation (the smoothed-probability argmax)."""
        return np.asarray(self.smoothed_probabilities.argmax(axis=1), dtype=np.intp)

    def expected_durations(self) -> FloatArray:
        """Expected regime durations :math:`1 / (1 - P_{kk})` (persistence, in observations)."""
        diag = np.diag(self.transition_matrix)
        return np.asarray(1.0 / np.maximum(1.0 - diag, _TINY), dtype=np.float64)


def _em_once(
    y: FloatArray,
    k: int,
    *,
    switching_mean: bool,
    max_iter: int,
    tol: float,
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, list[float]]:
    """Run EM from one random start; return (means, variances, P, initial, loglik history)."""
    # Random start: means spread across the sample, variances jittered, a persistent transition.
    means = np.sort(rng.normal(y.mean(), y.std(), size=k))
    variances = y.var() * rng.uniform(0.25, 1.75, size=k)
    p_matrix = np.eye(k) * rng.uniform(0.80, 0.98, size=k)[:, None]
    p_matrix += (1.0 - np.diag(p_matrix))[:, None] * rng.dirichlet(np.ones(k), size=k)
    p_matrix /= p_matrix.sum(axis=1, keepdims=True)
    initial = np.full(k, 1.0 / k)

    history: list[float] = []
    for _ in range(max_iter):
        filter_result = hamilton_filter(y, means, variances, p_matrix, initial=initial)
        smoothed = kim_smoother(filter_result.filtered, filter_result.predicted, p_matrix)
        history.append(filter_result.loglikelihood)

        # E-step expected transition counts: xi[i, j] = sum_t P(s_t=i, s_{t+1}=j | Y).
        ratio = smoothed[1:] / np.maximum(filter_result.predicted[1:], _TINY)
        xi = np.einsum("ti,ij,tj->ij", filter_result.filtered[:-1], p_matrix, ratio)

        # M-step (all closed form for the Gaussian model).
        p_matrix = xi / xi.sum(axis=1, keepdims=True)
        initial = smoothed[0].copy()
        weight = smoothed.sum(axis=0)
        if switching_mean:
            means = (smoothed * y[:, None]).sum(axis=0) / weight
        else:
            means = np.full(k, float(y.mean()))
        resid = y[:, None] - means[None, :]
        variances = np.maximum((smoothed * resid * resid).sum(axis=0) / weight, _VARIANCE_FLOOR)

        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break
    return means, variances, p_matrix, initial, history


def fit_markov_switching(
    returns: FloatArray,
    k_regimes: int = 2,
    *,
    switching_mean: bool = True,
    max_iter: int = 500,
    tol: float = 1e-8,
    n_starts: int = 10,
    rng: np.random.Generator,
) -> MarkovSwitchingResult:
    r"""Fit a ``K``-state Gaussian Markov-switching model by EM (Baum--Welch).

    Estimates the regime means, variances, transition matrix and initial distribution by
    expectation-maximisation, running from ``n_starts`` random starts and keeping the highest
    log-likelihood (regime models are riddled with local optima). The fitted regimes are ordered by
    variance ascending, so regime 0 is the calm state and the last is the crisis state — resolving
    label switching.

    Parameters
    ----------
    returns : ndarray, shape (T,)
        The observed series (e.g. asset returns).
    k_regimes : int, optional
        Number of hidden states (default 2, calm vs crisis).
    switching_mean : bool, optional
        If ``True`` (default), each regime has its own mean; if ``False``, the mean is common
        across regimes and only the variance switches (the canonical equity-volatility model).
    max_iter : int, optional
        Maximum EM iterations per start (default 500).
    tol : float, optional
        EM stops when the log-likelihood improves by less than ``tol`` (default ``1e-8``).
    n_starts : int, optional
        Number of random starts (default 10); the best by log-likelihood is returned.
    rng : numpy.random.Generator
        Seeded generator for the starts (keyword-only).

    Returns
    -------
    MarkovSwitchingResult
        The fitted parameters, filtered/smoothed probabilities and the EM diagnostics.

    Raises
    ------
    ValueError
        If ``k_regimes < 2`` or ``n_starts < 1``.
    """
    if k_regimes < 2:
        raise ValueError("k_regimes must be at least 2")
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1")

    y = np.asarray(returns, dtype=np.float64)
    best: tuple[FloatArray, FloatArray, FloatArray, FloatArray, list[float]] | None = None
    for _ in range(n_starts):
        candidate = _em_once(
            y, k_regimes, switching_mean=switching_mean, max_iter=max_iter, tol=tol, rng=rng
        )
        if best is None or candidate[4][-1] > best[4][-1]:
            best = candidate
    assert best is not None  # n_starts >= 1
    means, variances, p_matrix, initial, history = best

    order = np.argsort(variances)  # calm (low variance) first, resolving label switching
    means, variances = means[order], variances[order]
    p_matrix = p_matrix[np.ix_(order, order)]
    initial = initial[order]

    final = hamilton_filter(y, means, variances, p_matrix, initial=initial)
    smoothed = kim_smoother(final.filtered, final.predicted, p_matrix)
    return MarkovSwitchingResult(
        means=means,
        variances=variances,
        transition_matrix=p_matrix,
        initial_distribution=initial,
        stationary_distribution=_stationary_distribution(p_matrix),
        filtered_probabilities=final.filtered,
        smoothed_probabilities=smoothed,
        loglikelihood=float(final.loglikelihood),
        loglikelihood_history=np.asarray(history, dtype=np.float64),
        n_iter=len(history),
        converged=len(history) < max_iter,
    )
