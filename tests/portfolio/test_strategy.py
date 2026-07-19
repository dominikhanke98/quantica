"""Integration of the estimator → construction → backtest pipeline.

These tests wire the factor-step covariance estimators into the portfolio strategies
and run them through the walk-forward engine, confirming the seam works end to end and
that the constructions behave as advertised out of sample: minimum-variance really
does realise lower volatility than equal-weight, and costs strictly reduce the net
return.
"""

from __future__ import annotations

import numpy as np
from quantica.factor.data import generate_factor_data
from quantica.factor.estimators import (
    FactorCovariance,
    LedoitWolfCovariance,
    SampleCovariance,
)
from quantica.portfolio.backtest import ProportionalCosts, walk_forward_backtest
from quantica.portfolio.construction import PortfolioConstraints
from quantica.portfolio.strategy import (
    BlackLittermanStrategy,
    HRPStrategy,
    MeanVarianceStrategy,
    MinimumVarianceStrategy,
    RiskParityStrategy,
)


def _factor_panel(seed: int, n_periods: int = 240, n_assets: int = 12):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    betas = rng.uniform(0.2, 1.4, size=(n_assets, 4))
    data = generate_factor_data(n_periods, betas, rng, specific_vols=0.05)
    return data.asset_returns, data.factor_returns


def test_all_strategies_run_through_the_backtest() -> None:
    assets, factors = _factor_panel(seed=1)
    long_only = PortfolioConstraints(long_only=True)
    strategies = [
        MinimumVarianceStrategy(LedoitWolfCovariance(), long_only),
        RiskParityStrategy(SampleCovariance()),
        MeanVarianceStrategy(LedoitWolfCovariance(), risk_aversion=5.0, constraints=long_only),
        HRPStrategy(SampleCovariance()),
        BlackLittermanStrategy(LedoitWolfCovariance(), risk_aversion=3.0, constraints=long_only),
    ]
    for strategy in strategies:
        result = walk_forward_backtest(
            assets, strategy, train_window=60, rebalance_every=12, factor_returns=factors
        )
        assert result.net_returns.size > 0
        assert np.all(np.isfinite(result.net_returns))


def test_factor_estimator_seam_receives_factor_returns() -> None:
    """A FactorCovariance strategy needs the factor returns threaded through — it runs."""
    assets, factors = _factor_panel(seed=2)
    strategy = MinimumVarianceStrategy(FactorCovariance(), PortfolioConstraints(long_only=True))
    result = walk_forward_backtest(
        assets, strategy, train_window=60, rebalance_every=12, factor_returns=factors
    )
    assert np.all(np.isfinite(result.net_returns))


def test_minimum_variance_realises_lower_vol_than_equal_weight() -> None:
    """The whole point of min-variance: lower OOS volatility than 1/N."""
    assets, factors = _factor_panel(seed=3, n_periods=360, n_assets=15)

    class _EqualWeight:
        name = "equal-weight"

        def target_weights(self, asset_returns, factor_returns, w_prev):  # type: ignore[no-untyped-def]
            n = asset_returns.shape[1]
            return np.full(n, 1.0 / n)

    min_var = MinimumVarianceStrategy(LedoitWolfCovariance(), PortfolioConstraints(long_only=True))
    mv_result = walk_forward_backtest(
        assets, min_var, train_window=120, rebalance_every=12, factor_returns=factors
    )
    ew_result = walk_forward_backtest(
        assets, _EqualWeight(), train_window=120, rebalance_every=12, factor_returns=factors
    )
    assert np.std(mv_result.gross_returns) < np.std(ew_result.gross_returns)


def test_costs_reduce_net_return() -> None:
    assets, factors = _factor_panel(seed=4)
    strategy = MinimumVarianceStrategy(LedoitWolfCovariance(), PortfolioConstraints(long_only=True))
    free = walk_forward_backtest(
        assets, strategy, train_window=60, rebalance_every=6, factor_returns=factors
    )
    costed = walk_forward_backtest(
        assets,
        strategy,
        train_window=60,
        rebalance_every=6,
        cost_model=ProportionalCosts(0.002),
        factor_returns=factors,
    )
    assert costed.cumulative_return() < free.cumulative_return()
    assert costed.total_cost > 0.0
    # Gross series is identical; only the cost drag differs.
    assert np.allclose(free.gross_returns, costed.gross_returns, atol=1e-15)
