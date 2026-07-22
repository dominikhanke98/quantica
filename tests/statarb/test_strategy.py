"""Validation of the pairs strategy + overfitting-aware backtest (numerical-validation skill).

The headline is the marriage of the two guards: mining many candidate pairs finds a
spurious in-sample winner by chance, and the Deflated Sharpe Ratio / Probability of Backtest
Overfitting (reused from the portfolio pillar) correctly flag it — while a genuinely
cointegrated pair survives. Cointegration guards the *signal*; DSR/PBO guard the *backtest*.
The rest pins the mechanics: profitable on a real mean-reverting pair, costs and turnover
reconcile, the backtest is causal (no look-ahead), the results are deterministic, and the
realised holding period is consistent with the spread's half-life.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.portfolio import (
    deflated_sharpe_ratio_from_trials,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from quantica.statarb import (
    PairsStrategyConfig,
    engle_granger,
    estimate_ou_process,
    generate_cointegrated_pair,
    generate_independent_random_walks,
    generate_time_varying_pair,
    pairs_backtest,
    pairs_return_matrix,
)

_ANNUALISE = np.sqrt(252.0)


# --------------------------------------------------------------------------- #
# Known-truth: profitable on a real pair, both spread methods
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("method", ["static", "kalman"])
def test_profitable_on_cointegrated_pair(method: str) -> None:
    """On a genuinely cointegrated pair the strategy makes money before and after costs."""
    y, x = generate_cointegrated_pair(1200, np.random.default_rng(0), beta=1.2, spread_kappa=0.1)
    result = pairs_backtest(y, x, method=method, train_window=250)
    assert result.sharpe_ratio(gross=True) > 0.5  # a real edge before costs
    assert result.sharpe_ratio() > 0.0  # survives costs
    assert result.n_trades > 5
    assert 0.0 <= result.hit_rate <= 1.0


def test_holding_period_consistent_with_half_life() -> None:
    """The realised average holding period is a small multiple of the spread half-life."""
    y, x = generate_cointegrated_pair(1500, np.random.default_rng(1), beta=1.2, spread_kappa=0.1)
    half_life = estimate_ou_process(engle_granger(y[:250], x[:250]).spread).half_life
    result = pairs_backtest(y, x, method="static", train_window=250)
    # Hold roughly until reversion (a few half-lives) — the same order of magnitude.
    assert 0.5 * half_life < result.avg_holding_period < 6.0 * half_life


# --------------------------------------------------------------------------- #
# The headline: DSR/PBO catch a mined spurious winner; the genuine pair survives
# --------------------------------------------------------------------------- #


def test_mining_many_pairs_is_flagged_while_genuine_pair_survives() -> None:
    """Best-of-many spurious pairs is flagged by DSR/PBO; a genuine pair passes PSR.

    Twenty-four independent random walks give 276 spurious pairs. The best in-sample Sharpe
    looks tradeable, but deflated for the 276 trials it is **not** significant and the
    probability of backtest overfitting is high. A single, economically pre-selected
    cointegrated pair (one trial) clears the probabilistic Sharpe bar. Two levels of guard,
    working together.
    """
    rng = np.random.default_rng(7)
    prices = np.cumsum(rng.standard_normal((1400, 24)), axis=0)  # 24 independent walks
    pairs = [(i, j) for i in range(24) for j in range(i + 1, 24)]  # 276 spurious pairs
    matrix = pairs_return_matrix(prices, pairs, method="static", train_window=250)

    best_sharpe = max(sharpe_ratio(matrix[:, k]) for k in range(matrix.shape[1]))
    dsr = deflated_sharpe_ratio_from_trials(matrix)
    pbo = probability_of_backtest_overfitting(matrix, n_splits=10)
    assert best_sharpe * _ANNUALISE > 1.0  # the mined winner *looks* tradeable
    assert not dsr.is_significant  # ... but deflated for 190 trials it is not
    assert pbo.pbo > 0.3  # and the selection does not hold out of sample

    # The genuine, pre-selected pair (one trial) clears the probabilistic Sharpe bar.
    y, x = generate_cointegrated_pair(1400, np.random.default_rng(0), beta=1.2, spread_kappa=0.1)
    genuine = pairs_backtest(y, x, method="static", train_window=250).net_returns
    assert probabilistic_sharpe_ratio(sharpe_ratio(genuine), genuine.size) > 0.95


def test_spurious_single_pairs_rarely_significant() -> None:
    """A lone random-walk pair clears PSR only about as often as the nominal 5% level."""
    significant = 0
    for seed in range(20):
        walks = generate_independent_random_walks(1200, 2, np.random.default_rng(100 + seed))
        net = pairs_backtest(
            walks[:, 0], walks[:, 1], method="static", train_window=250
        ).net_returns
        significant += probabilistic_sharpe_ratio(sharpe_ratio(net), net.size) > 0.95
    assert significant <= 3  # ~ nominal size, not a systematic edge on noise


# --------------------------------------------------------------------------- #
# Mechanics: costs, no look-ahead, determinism
# --------------------------------------------------------------------------- #


def test_costs_reduce_returns_and_reconcile() -> None:
    """Net = gross minus costs to the last bit, and trading incurs a positive total cost."""
    y, x = generate_cointegrated_pair(1000, np.random.default_rng(2), beta=1.2, spread_kappa=0.1)
    result = pairs_backtest(
        y, x, PairsStrategyConfig(cost_rate=0.001), method="static", train_window=250
    )
    assert result.total_cost > 0.0
    assert np.all(result.net_returns <= result.gross_returns + 1e-15)
    assert np.isclose(
        (result.gross_returns - result.net_returns).sum(), result.total_cost, atol=1e-12
    )


def test_no_lookahead() -> None:
    """Corrupting the future leaves the earlier out-of-sample returns bit-identical."""
    y, x = generate_cointegrated_pair(1000, np.random.default_rng(3), beta=1.2, spread_kappa=0.1)
    base = pairs_backtest(y, x, method="static", train_window=250)
    corrupt_from = 700  # scramble everything from here on
    y2, x2 = y.copy(), x.copy()
    rng = np.random.default_rng(99)
    y2[corrupt_from:] += rng.standard_normal(y.size - corrupt_from) * 50.0
    x2[corrupt_from:] += rng.standard_normal(x.size - corrupt_from) * 50.0
    perturbed = pairs_backtest(y2, x2, method="static", train_window=250)
    # OOS index j corresponds to original index 250 + j; only indices < corrupt_from are safe.
    safe = corrupt_from - 250 - 1
    assert np.allclose(base.net_returns[:safe], perturbed.net_returns[:safe], atol=1e-12)


def test_seeded_determinism() -> None:
    """The backtest is a pure function of its inputs — identical runs match exactly."""
    y, x = generate_cointegrated_pair(800, np.random.default_rng(4), beta=1.2, spread_kappa=0.1)
    a = pairs_backtest(y, x, method="kalman", train_window=200)
    b = pairs_backtest(y, x, method="kalman", train_window=200)
    assert np.array_equal(a.net_returns, b.net_returns)
    assert a.n_trades == b.n_trades


def test_kalman_helps_on_a_drifting_pair() -> None:
    """When the true hedge ratio drifts, the dynamic Kalman spread beats the static one."""
    true = np.linspace(1.0, 2.2, 1500)  # the hedge ratio drifts over the sample
    y, x = generate_time_varying_pair(true, np.random.default_rng(5), alpha=2.0, obs_vol=1.0)
    static = pairs_backtest(y, x, method="static", train_window=300).sharpe_ratio()
    kalman = pairs_backtest(y, x, method="kalman", train_window=300).sharpe_ratio()
    assert kalman > static  # the adapting hedge ratio tracks the drift the static one misses


# --------------------------------------------------------------------------- #
# Input / config validation
# --------------------------------------------------------------------------- #


def test_config_and_input_validation() -> None:
    """Bad thresholds, unknown methods and impossible windows are rejected."""
    rng = np.random.default_rng(0)
    y, x = generate_cointegrated_pair(300, rng, beta=1.2)
    with pytest.raises(ValueError, match="exit_z < entry_z < stop_z"):
        PairsStrategyConfig(entry_z=3.0, exit_z=0.5, stop_z=2.0)
    with pytest.raises(ValueError, match="method"):
        pairs_backtest(y, x, method="lstm", train_window=100)
    with pytest.raises(ValueError, match="train_window"):
        pairs_backtest(y, x, method="static", train_window=299)
