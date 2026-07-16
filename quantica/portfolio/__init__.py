r"""Systematic portfolio management — construction, backtest, and validity.

The third pillar of ``quantica``. The framing that keeps it in this repo's identity:
*everyone ships a backtester; this ships the test of whether the backtest means
anything.* Three layers:

* **Construction** (:mod:`~quantica.portfolio.construction`) — mean-variance,
  minimum-variance and risk-parity portfolios via ``cvxpy``, consuming a
  :class:`~quantica.factor.estimators.CovarianceEstimator` from the factor step so the
  stage-2 estimator comparison plugs straight in.
* **Backtest** (:mod:`~quantica.portfolio.backtest`) — a walk-forward engine with
  transaction costs and exact turnover accounting, built on the tested no-lookahead
  window machinery, plus the strategies (:mod:`~quantica.portfolio.strategy`) that
  bind an estimator to a constructor.
"""

from __future__ import annotations

from quantica.portfolio.backtest import (
    BacktestResult,
    ProportionalCosts,
    Strategy,
    TransactionCostModel,
    walk_forward_backtest,
)
from quantica.portfolio.construction import (
    PortfolioConstraints,
    mean_variance_weights,
    minimum_variance_weights,
    risk_parity_weights,
)
from quantica.portfolio.strategy import (
    MeanVarianceStrategy,
    MinimumVarianceStrategy,
    RiskParityStrategy,
    Signal,
    historical_mean_signal,
)

__all__ = [
    "BacktestResult",
    "MeanVarianceStrategy",
    "MinimumVarianceStrategy",
    "PortfolioConstraints",
    "ProportionalCosts",
    "RiskParityStrategy",
    "Signal",
    "Strategy",
    "TransactionCostModel",
    "historical_mean_signal",
    "mean_variance_weights",
    "minimum_variance_weights",
    "risk_parity_weights",
    "walk_forward_backtest",
]
