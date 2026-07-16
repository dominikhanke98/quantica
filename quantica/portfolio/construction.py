r"""Constrained portfolio construction — mean-variance, min-variance, risk-parity.

Three classical portfolio rules, each solved as a convex program with **`cvxpy`**
(CLAUDE.md §3 — lean on the solver, do not hand-roll the QP). The covariance they
consume comes from a :class:`~quantica.factor.estimators.CovarianceEstimator`, so the
stage-2 estimator comparison plugs straight in: the optimiser is only as good as the
:math:`\Sigma` it is handed, and inverting a noisy sample covariance is Michaud's
"error maximiser" — which is exactly why the estimator choice is evidence-backed.

The rules:

* :func:`minimum_variance_weights` — :math:`\min_w w^\top \Sigma w`. With only the
  full-investment budget it is the classic global minimum-variance portfolio, and it
  reduces **exactly** to the closed form
  :func:`~quantica.factor.estimators.min_variance_weights`
  (:math:`w \propto \Sigma^{-1}\mathbf 1`) — the validation anchor for the solver.
* :func:`mean_variance_weights` — Markowitz
  :math:`\max_w \mu^\top w - \tfrac{\gamma}{2} w^\top \Sigma w`.
* :func:`risk_parity_weights` — the equal-risk-contribution portfolio via Spinu's
  (2013) convex log-barrier reformulation (long-only by construction).

Realistic constraints live in :class:`PortfolioConstraints`: a long-only option,
per-name position limits, and an L1 **turnover budget** relative to the current
holdings (so a backtest can cap trading). Turnover is a linear constraint, so the
problem stays a convex QP the solver handles directly.

References
----------
Markowitz, H. (1952), "Portfolio selection", *Journal of Finance*.
Maillard, S., Roncalli, T. & Teïletche, J. (2010), "The properties of equally
weighted risk contribution portfolios", *Journal of Portfolio Management*.
Spinu, F. (2013), "An algorithm for computing risk parity weights", SSRN 2297383.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "PortfolioConstraints",
    "mean_variance_weights",
    "minimum_variance_weights",
    "risk_parity_weights",
]

# Solver statuses cvxpy reports for a usable solution (the second is a looser
# convergence that we accept but that callers can detect via a re-solve if needed).
_OK_STATUS = ("optimal", "optimal_inaccurate")


@dataclass(frozen=True)
class PortfolioConstraints:
    """Realistic construction constraints, applied as linear cvxpy constraints.

    Attributes
    ----------
    long_only : bool
        If true, forbid shorts (``w >= 0``). Default ``False`` (long-short allowed).
    max_position : float or None
        Upper bound on each weight (e.g. ``0.10`` caps any name at 10%). ``None``
        leaves it unbounded above.
    min_position : float or None
        Lower bound on each weight. ``None`` leaves it unbounded below; combined with
        ``long_only`` the effective floor is ``max(0, min_position)``.
    max_turnover : float or None
        Budget on one-way L1 turnover :math:`\\lVert w - w_{\\text{prev}}\\rVert_1`
        relative to the supplied current holdings. ``None`` means unconstrained
        trading. Ignored when no previous weights are passed to the constructor.
    full_investment : bool
        If true (default), require the weights to sum to one (fully invested, no
        cash). Set ``False`` for a budget-free formulation.
    """

    long_only: bool = False
    max_position: float | None = None
    min_position: float | None = None
    max_turnover: float | None = None
    full_investment: bool = True


def _cvxpy() -> Any:
    """Lazily import cvxpy (a heavy import kept out of package import time)."""
    import cvxpy as cp

    return cp


def _linear_constraints(
    cp: Any,
    w: Any,
    constraints: PortfolioConstraints,
    w_prev: FloatArray | None,
) -> list[Any]:
    """Translate :class:`PortfolioConstraints` into a list of cvxpy constraints."""
    cons = []
    if constraints.full_investment:
        cons.append(cp.sum(w) == 1.0)
    if constraints.long_only:
        cons.append(w >= 0.0)
    if constraints.min_position is not None:
        cons.append(w >= constraints.min_position)
    if constraints.max_position is not None:
        cons.append(w <= constraints.max_position)
    if constraints.max_turnover is not None and w_prev is not None:
        cons.append(cp.norm1(w - np.asarray(w_prev, dtype=np.float64)) <= constraints.max_turnover)
    return cons


def _solve(cp: Any, problem: Any, w: Any) -> FloatArray:
    """Solve a cvxpy problem and return the weight vector, raising on failure."""
    problem.solve()
    if problem.status not in _OK_STATUS or w.value is None:
        raise RuntimeError(f"portfolio optimisation failed (solver status: {problem.status})")
    return np.asarray(w.value, dtype=np.float64)


def _validate_cov(cov: FloatArray) -> FloatArray:
    sigma = np.asarray(cov, dtype=np.float64)
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"cov must be a square 2-D matrix, got shape {sigma.shape}")
    return sigma


def minimum_variance_weights(
    cov: FloatArray,
    constraints: PortfolioConstraints | None = None,
    w_prev: FloatArray | None = None,
) -> FloatArray:
    r"""Minimum-variance portfolio :math:`\min_w w^\top \Sigma w` under constraints.

    With only the full-investment budget this is the global minimum-variance
    portfolio and equals the closed form :math:`w \propto \Sigma^{-1}\mathbf 1`
    (:func:`quantica.factor.estimators.min_variance_weights`); adding a long-only,
    position-limit or turnover constraint turns it into an inequality-constrained QP
    that cvxpy solves.

    Parameters
    ----------
    cov : ndarray, shape (n, n)
        The asset covariance, typically from a
        :class:`~quantica.factor.estimators.CovarianceEstimator`.
    constraints : PortfolioConstraints, optional
        Defaults to the plain full-investment budget (long-short GMV).
    w_prev : ndarray, shape (n,), optional
        Current holdings, used only by a turnover constraint.
    """
    cp = _cvxpy()
    sigma = _validate_cov(cov)
    constraints = constraints or PortfolioConstraints()
    w = cp.Variable(sigma.shape[0])
    objective = cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma)))
    problem = cp.Problem(objective, _linear_constraints(cp, w, constraints, w_prev))
    return _solve(cp, problem, w)


def mean_variance_weights(
    expected_returns: FloatArray,
    cov: FloatArray,
    risk_aversion: float,
    constraints: PortfolioConstraints | None = None,
    w_prev: FloatArray | None = None,
) -> FloatArray:
    r"""Markowitz mean-variance portfolio under constraints.

    Maximises the mean-variance utility
    :math:`\mu^\top w - \tfrac{\gamma}{2}\, w^\top \Sigma w` for risk-aversion
    :math:`\gamma > 0`. As :math:`\gamma \to \infty` the mean term vanishes and the
    solution approaches the minimum-variance portfolio; small :math:`\gamma` chases
    the expected-return signal harder.

    Parameters
    ----------
    expected_returns : ndarray, shape (n,)
        The expected-return (alpha) signal :math:`\mu`.
    cov : ndarray, shape (n, n)
        The asset covariance.
    risk_aversion : float
        The risk-aversion coefficient :math:`\gamma` (must be positive).
    constraints : PortfolioConstraints, optional
        Defaults to the plain full-investment budget.
    w_prev : ndarray, shape (n,), optional
        Current holdings, used only by a turnover constraint.
    """
    if risk_aversion <= 0.0:
        raise ValueError(f"risk_aversion must be positive, got {risk_aversion}")
    cp = _cvxpy()
    sigma = _validate_cov(cov)
    mu = np.asarray(expected_returns, dtype=np.float64)
    if mu.shape != (sigma.shape[0],):
        raise ValueError(f"expected_returns must have shape ({sigma.shape[0]},), got {mu.shape}")
    constraints = constraints or PortfolioConstraints()
    w = cp.Variable(sigma.shape[0])
    utility = mu @ w - 0.5 * risk_aversion * cp.quad_form(w, cp.psd_wrap(sigma))
    problem = cp.Problem(cp.Maximize(utility), _linear_constraints(cp, w, constraints, w_prev))
    return _solve(cp, problem, w)


def risk_parity_weights(cov: FloatArray) -> FloatArray:
    r"""Equal-risk-contribution (risk-parity) portfolio via Spinu's convex program.

    The ERC portfolio equalises each asset's contribution to total risk,
    :math:`w_i (\Sigma w)_i = w_j (\Sigma w)_j` for all :math:`i, j`. Directly this
    is a non-convex system, but Spinu (2013) showed the minimiser of the convex
    log-barrier objective

    .. math::

        f(y) = \tfrac12\, y^\top \Sigma y - \tfrac1n \sum_i \log y_i,
        \qquad y > 0,

    once renormalised :math:`w = y / \mathbf 1^\top y`, is exactly the (long-only,
    fully-invested) equal-risk-contribution portfolio. Because renormalisation is
    intrinsic, risk parity does not take the general
    :class:`PortfolioConstraints` — it is long-only and fully invested by
    construction.

    Parameters
    ----------
    cov : ndarray, shape (n, n)
        The asset covariance (must be positive definite for a unique solution).
    """
    cp = _cvxpy()
    sigma = _validate_cov(cov)
    n = sigma.shape[0]
    y = cp.Variable(n, pos=True)
    objective = cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(sigma)) - cp.sum(cp.log(y)) / n)
    problem = cp.Problem(objective)
    problem.solve()
    if problem.status not in _OK_STATUS or y.value is None:
        raise RuntimeError(f"risk-parity optimisation failed (solver status: {problem.status})")
    raw = np.asarray(y.value, dtype=np.float64)
    return np.asarray(raw / raw.sum(), dtype=np.float64)
