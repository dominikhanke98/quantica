r"""Walk-forward backtest engine with transaction costs and turnover accounting.

A rebalanced long-only (or long-short) backtest built on the **tested no-lookahead
window machinery** from the factor step
(:func:`quantica.factor.evaluation.walk_forward_windows`): each window's trailing
training slice fits the strategy, and the resulting weights are held over the
*following*, non-overlapping slice — so no future data can leak into a weight.

The mechanics are deliberately exact rather than clever, because the whole point of
this repo is that the numbers reconcile:

* **Turnover** at each rebalance is the one-way L1 distance
  :math:`\lVert w_{\text{target}} - w_{\text{drift}} \rVert_1` between the target
  weights and the weights the previous holding *drifted* to (weights move with the
  assets between rebalances; that drift is tracked, not assumed away).
* **Costs** are a return drag of ``cost_model.cost(turnover)`` charged in the first
  period of each holding window, so ``gross - net`` sums to total cost *exactly*.
* With a zero-cost model, ``net`` equals ``gross`` to the last bit.

The portfolio is assumed **self-financing and fully invested** (weights sum to one),
which makes the drift update exact: if end-of-period grown weights are
:math:`\tilde w_i = w_i(1+r_i)`, the portfolio gross return is
:math:`\mathbf 1^\top \tilde w - 1` and the drifted weights renormalise to
:math:`\tilde w / \mathbf 1^\top \tilde w`.

This engine produces the per-period **net return series** that the backtest-validity
layer (:mod:`quantica.portfolio.overfitting`) then interrogates — the return series
is the seam, exactly as the risk pillar's P&L series was.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from quantica.factor.evaluation import walk_forward_windows

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "BacktestResult",
    "ProportionalCosts",
    "Strategy",
    "TransactionCostModel",
    "walk_forward_backtest",
]


@runtime_checkable
class TransactionCostModel(Protocol):
    """Maps one-way turnover to a proportional return drag."""

    def cost(self, turnover: float) -> float:
        """Cost (as a fraction of portfolio value) of trading ``turnover`` one-way."""
        ...


@dataclass(frozen=True)
class ProportionalCosts:
    """Linear proportional costs: ``cost = rate * turnover``.

    ``rate`` is the one-way cost per unit of turnover in decimal (e.g. ``0.001`` for
    10 bps). One-way turnover of 1.0 means the entire portfolio was traded.
    """

    rate: float

    def cost(self, turnover: float) -> float:
        return self.rate * turnover


@runtime_checkable
class Strategy(Protocol):
    """A rule turning a training window into target weights (the backtest seam)."""

    @property
    def name(self) -> str:
        """Short label for tables."""
        ...

    def target_weights(
        self,
        asset_returns: FloatArray,
        factor_returns: FloatArray | None,
        w_prev: FloatArray | None,
    ) -> FloatArray:
        """Target weights from the trailing ``(train, n)`` returns and current holdings."""
        ...


@dataclass(frozen=True)
class BacktestResult:
    """The output of a walk-forward backtest: net/gross series and trade accounting.

    Attributes
    ----------
    gross_returns : ndarray, shape (n_test_periods,)
        Per-period portfolio return before costs.
    net_returns : ndarray, shape (n_test_periods,)
        Per-period return after the rebalance cost drag.
    turnover : ndarray, shape (n_rebalances,)
        One-way turnover at each rebalance.
    costs : ndarray, shape (n_rebalances,)
        Cost drag applied at each rebalance (fraction of portfolio value).
    weights : ndarray, shape (n_rebalances, n_assets)
        Target weights set at each rebalance.
    rebalance_starts : ndarray, shape (n_rebalances,)
        Index (into the return panel) of the first period of each holding window.
    """

    gross_returns: FloatArray
    net_returns: FloatArray
    turnover: FloatArray
    costs: FloatArray
    weights: FloatArray
    rebalance_starts: FloatArray

    @property
    def total_cost(self) -> float:
        """Total cost paid over the backtest (sum of per-rebalance drags)."""
        return float(np.sum(self.costs))

    @property
    def average_turnover(self) -> float:
        """Mean one-way turnover per rebalance."""
        return float(np.mean(self.turnover))

    def sharpe_ratio(self, periods_per_year: int = 12, *, gross: bool = False) -> float:
        r"""Annualised Sharpe ratio of the (net by default) return series."""
        series = self.gross_returns if gross else self.net_returns
        mu = float(np.mean(series))
        sd = float(np.std(series, ddof=1))
        if sd == 0.0:
            return 0.0
        return float(mu / sd * np.sqrt(periods_per_year))

    def cumulative_return(self, *, gross: bool = False) -> float:
        r"""Total compounded return over the backtest, :math:`\prod_t (1+r_t) - 1`."""
        series = self.gross_returns if gross else self.net_returns
        return float(np.prod(1.0 + series) - 1.0)


def walk_forward_backtest(
    asset_returns: FloatArray,
    strategy: Strategy,
    *,
    train_window: int,
    rebalance_every: int,
    cost_model: TransactionCostModel | None = None,
    factor_returns: FloatArray | None = None,
) -> BacktestResult:
    r"""Run a rolling walk-forward backtest of ``strategy`` on ``asset_returns``.

    At each rebalance the strategy is fitted on the trailing ``train_window`` periods
    and its target weights are held for the next ``rebalance_every`` periods, during
    which the weights drift with the assets. Turnover is measured against the drifted
    weights and costs are charged in the first held period.

    Parameters
    ----------
    asset_returns : ndarray, shape (T, n)
        Simple (not log) asset returns.
    strategy : Strategy
        The construction rule (bundles a covariance estimator + optimiser).
    train_window : int
        Number of trailing periods used to fit the strategy at each rebalance.
    rebalance_every : int
        Holding-period length; weights are refreshed this often (the test-window
        length passed to :func:`walk_forward_windows`).
    cost_model : TransactionCostModel, optional
        Turnover→cost map; defaults to zero costs (gross == net).
    factor_returns : ndarray, shape (T, k), optional
        Factor returns aligned to ``asset_returns``, for factor-model estimators.
    """
    r = np.asarray(asset_returns, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError("asset_returns must be 2-D (T, n)")
    f = None if factor_returns is None else np.asarray(factor_returns, dtype=np.float64)
    if f is not None and f.shape[0] != r.shape[0]:
        raise ValueError("factor_returns must have the same number of rows as asset_returns")
    n_assets = r.shape[1]
    costs_model = cost_model or ProportionalCosts(0.0)
    windows = walk_forward_windows(r.shape[0], train_window, rebalance_every)

    gross: list[float] = []
    net: list[float] = []
    turnover: list[float] = []
    costs: list[float] = []
    weights: list[FloatArray] = []
    starts: list[int] = []

    w_drift = np.zeros(n_assets)  # start in cash: first rebalance trades from flat
    for window in windows:
        r_train = r[window.train_start : window.train_end]
        f_train = None if f is None else f[window.train_start : window.train_end]
        w_target = strategy.target_weights(r_train, f_train, w_drift)

        one_way_turnover = float(np.sum(np.abs(w_target - w_drift)))
        drag = costs_model.cost(one_way_turnover)
        turnover.append(one_way_turnover)
        costs.append(drag)
        weights.append(w_target)
        starts.append(window.test_start)

        w = w_target.copy()
        r_test = r[window.test_start : window.test_end]
        for t in range(r_test.shape[0]):
            grown = w * (1.0 + r_test[t])
            portfolio_gross = float(np.sum(grown) - np.sum(w))  # w may not sum to 1 if cash
            gross.append(portfolio_gross)
            net.append(portfolio_gross - (drag if t == 0 else 0.0))
            total = float(np.sum(grown))
            w = grown / total if total != 0.0 else grown
        w_drift = w

    return BacktestResult(
        gross_returns=np.array(gross, dtype=np.float64),
        net_returns=np.array(net, dtype=np.float64),
        turnover=np.array(turnover, dtype=np.float64),
        costs=np.array(costs, dtype=np.float64),
        weights=np.array(weights, dtype=np.float64),
        rebalance_starts=np.array(starts, dtype=np.float64),
    )
