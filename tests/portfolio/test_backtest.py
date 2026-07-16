"""Validation of the walk-forward backtest engine.

The engine has no free parameters to converge — its correctness is *exactness*: with
zero costs the net series equals the gross series to the last bit; the cost drag and
the measured turnover reconcile to a hand computation; weights drift with the assets
exactly; and no rebalance can see data beyond its training slice (no lookahead).
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.portfolio.backtest import (
    ProportionalCosts,
    walk_forward_backtest,
)


class _FixedWeights:
    """A strategy that always returns the same target weights (for hand checks)."""

    def __init__(self, weights: np.ndarray, name: str = "fixed") -> None:
        self._w = np.asarray(weights, dtype=np.float64)
        self.name = name

    def target_weights(self, asset_returns, factor_returns, w_prev):  # type: ignore[no-untyped-def]
        return self._w.copy()


class _InverseVolWeights:
    """A data-dependent strategy: weights proportional to 1/vol over the training slice.

    Because the weights are a function of the training returns, it is a genuine test
    of no-lookahead — corrupting future returns must not move a past rebalance.
    """

    name = "inv-vol"

    def target_weights(self, asset_returns, factor_returns, w_prev):  # type: ignore[no-untyped-def]
        vol = np.std(np.asarray(asset_returns, dtype=np.float64), axis=0, ddof=1)
        inv = 1.0 / np.where(vol > 0.0, vol, np.inf)
        return inv / inv.sum()


class _SpyStrategy:
    """Records how many rows each fit saw — to prove the training slice is bounded."""

    def __init__(self, n_assets: int) -> None:
        self._w = np.full(n_assets, 1.0 / n_assets)
        self.name = "spy"
        self.rows_seen: list[int] = []
        self.factor_rows_seen: list[int] = []

    def target_weights(self, asset_returns, factor_returns, w_prev):  # type: ignore[no-untyped-def]
        self.rows_seen.append(asset_returns.shape[0])
        if factor_returns is not None:
            self.factor_rows_seen.append(factor_returns.shape[0])
        return self._w.copy()


def _returns(t: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.02, size=(t, n))


# --------------------------------------------------------------------------- #
# Exactness / reconciliation
# --------------------------------------------------------------------------- #


def test_zero_cost_net_equals_gross() -> None:
    r = _returns(60, 4, seed=1)
    result = walk_forward_backtest(
        r, _FixedWeights(np.full(4, 0.25)), train_window=12, rebalance_every=3
    )
    assert np.array_equal(result.net_returns, result.gross_returns)
    assert result.total_cost == 0.0


def test_cost_and_turnover_reconcile_exactly() -> None:
    r = _returns(60, 4, seed=2)
    rate = 0.001
    result = walk_forward_backtest(
        r,
        _FixedWeights(np.full(4, 0.25)),
        train_window=12,
        rebalance_every=3,
        cost_model=ProportionalCosts(rate),
    )
    # gross - net summed over all periods equals the total cost paid.
    assert np.isclose(
        np.sum(result.gross_returns) - np.sum(result.net_returns),
        result.total_cost,
        atol=1e-15,
    )
    # Each rebalance cost is exactly rate * that rebalance's turnover.
    assert np.allclose(result.costs, rate * result.turnover, atol=1e-15)


def test_first_rebalance_trades_from_cash() -> None:
    """The opening turnover of a fully-invested long book is exactly 1.0."""
    r = _returns(40, 5, seed=3)
    result = walk_forward_backtest(
        r, _FixedWeights(np.full(5, 0.2)), train_window=10, rebalance_every=5
    )
    assert np.isclose(result.turnover[0], 1.0, atol=1e-12)


def test_gross_return_is_weighted_asset_return_in_first_period() -> None:
    r = _returns(30, 3, seed=4)
    w = np.array([0.5, 0.3, 0.2])
    result = walk_forward_backtest(r, _FixedWeights(w), train_window=10, rebalance_every=5)
    first_test_period = int(result.rebalance_starts[0])
    expected = float(w @ r[first_test_period])
    assert np.isclose(result.gross_returns[0], expected, atol=1e-15)


def test_weights_drift_with_assets() -> None:
    """After one period the tracked drift matches the analytic renormalised weights."""
    r = np.zeros((22, 2))
    r[10] = [0.10, -0.05]  # the single test period of the first window
    w0 = np.array([0.5, 0.5])
    result = walk_forward_backtest(r, _FixedWeights(w0), train_window=10, rebalance_every=1)
    # After period 10, weights should drift to grown/sum(grown).
    grown = w0 * (1.0 + r[10])
    expected_drift = grown / grown.sum()
    # The second rebalance's turnover is measured against this drift.
    second_turnover = float(np.sum(np.abs(w0 - expected_drift)))
    assert np.isclose(result.turnover[1], second_turnover, atol=1e-15)


def test_equal_weight_reproduces_cross_sectional_mean() -> None:
    r = _returns(40, 4, seed=6)
    result = walk_forward_backtest(
        r, _FixedWeights(np.full(4, 0.25)), train_window=10, rebalance_every=1
    )
    # With rebalancing every period, each gross return is the equal-weight mean.
    starts = result.rebalance_starts.astype(int)
    expected = r[starts].mean(axis=1)
    assert np.allclose(result.gross_returns, expected, atol=1e-14)


# --------------------------------------------------------------------------- #
# No lookahead
# --------------------------------------------------------------------------- #


def test_no_lookahead_training_slice_is_bounded() -> None:
    r = _returns(50, 3, seed=7)
    spy = _SpyStrategy(3)
    walk_forward_backtest(r, spy, train_window=12, rebalance_every=4)
    # Every fit saw exactly train_window rows — never more (no peeking forward).
    assert spy.rows_seen
    assert all(rows == 12 for rows in spy.rows_seen)


def test_factor_returns_are_sliced_consistently() -> None:
    r = _returns(50, 3, seed=8)
    f = _returns(50, 2, seed=9)
    spy = _SpyStrategy(3)
    walk_forward_backtest(r, spy, train_window=12, rebalance_every=4, factor_returns=f)
    assert spy.factor_rows_seen == spy.rows_seen


def test_future_returns_do_not_change_past_weights() -> None:
    """Corrupting returns after a rebalance leaves that rebalance's weights intact."""
    r = _returns(50, 3, seed=10)
    strat = _InverseVolWeights()
    base = walk_forward_backtest(r, strat, train_window=12, rebalance_every=4)
    corrupted = r.copy()
    corrupted[30:] += 5.0  # wreck the future
    after = walk_forward_backtest(corrupted, strat, train_window=12, rebalance_every=4)
    # Rebalances whose training window ended at or before period 30 are unaffected.
    unaffected = np.where(base.rebalance_starts <= 30)[0]
    assert unaffected.size > 0
    assert np.allclose(base.weights[unaffected], after.weights[unaffected], atol=1e-15)


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_rejects_one_dimensional_returns() -> None:
    with pytest.raises(ValueError, match="2-D"):
        walk_forward_backtest(
            np.zeros(20), _FixedWeights(np.ones(1)), train_window=5, rebalance_every=2
        )


def test_rejects_misaligned_factor_returns() -> None:
    r = _returns(30, 3, seed=11)
    f = _returns(25, 2, seed=12)
    with pytest.raises(ValueError, match="same number of rows"):
        walk_forward_backtest(
            r, _FixedWeights(np.ones(3)), train_window=6, rebalance_every=2, factor_returns=f
        )
