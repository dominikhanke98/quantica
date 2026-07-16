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
* **Backtest-validity layer** (the headline) — the model-validation discipline applied
  to strategy backtests: the deflated Sharpe ratio and probability of backtest
  overfitting (:mod:`~quantica.portfolio.overfitting`), plus purged/embargoed
  cross-validation (:mod:`~quantica.portfolio.cv`).
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
from quantica.portfolio.cv import PurgedFold, purged_kfold_indices
from quantica.portfolio.data import TrialReturns, generate_trial_returns
from quantica.portfolio.overfitting import (
    DeflatedSharpeResult,
    PBOResult,
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_trials,
    expected_maximum_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
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
    "DeflatedSharpeResult",
    "MeanVarianceStrategy",
    "MinimumVarianceStrategy",
    "PBOResult",
    "PortfolioConstraints",
    "ProportionalCosts",
    "PurgedFold",
    "RiskParityStrategy",
    "Signal",
    "Strategy",
    "TransactionCostModel",
    "TrialReturns",
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_trials",
    "expected_maximum_sharpe",
    "generate_trial_returns",
    "historical_mean_signal",
    "mean_variance_weights",
    "minimum_track_record_length",
    "minimum_variance_weights",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "purged_kfold_indices",
    "risk_parity_weights",
    "sharpe_ratio",
    "walk_forward_backtest",
]
