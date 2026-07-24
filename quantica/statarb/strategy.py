r"""Pairs-trading strategy and its overfitting-aware backtest — the signal→backtest arc.

This closes the statistical-arbitrage arc and ties it back to the portfolio pillar's
identity. The cointegration test (step 1) says *whether* a spread is real, the Kalman filter
(step 2) gives the *hedge ratio*; here that spread becomes a **trading strategy**, run
walk-forward with costs, and — the whole point — subjected to the **backtest-validity layer**
(Deflated Sharpe Ratio, Probability of Backtest Overfitting) from
:mod:`quantica.portfolio.overfitting`.

Statistical arbitrage is the strategy class *most* prone to backtest overfitting: mining many
candidate pairs finds spurious winners by chance (multiple testing). So the two guards work at
two levels — **cointegration guards against a spurious signal, DSR/PBO guard against a
spurious backtest**. Demonstrating that the pillar catches its own field's endemic failure
mode is the deliverable.

Scope discipline (CLAUDE.md §3): the transaction-cost model
(:class:`~quantica.portfolio.backtest.ProportionalCosts`), the no-lookahead walk-forward
windows (:func:`~quantica.factor.evaluation.walk_forward_windows`) and the whole DSR/PBO/CV
layer are **reused**, not rebuilt. The pairs backtest is deliberately *event-driven* (the
position flips on z-score crossings, not on a fixed rebalance schedule), so it does not use the
portfolio weights-rebalance engine directly — but it reuses that engine's cost model and the
validity layer, which operate on the return series. The new part is the strategy logic
(signal → positions).

The strategy
------------
A textbook mean-reversion rule on the spread's z-score: enter when the spread is
``entry_z`` standard deviations from its mean (short the rich leg / long the cheap leg via the
hedge ratio), exit on reversion inside ``exit_z``, and stop out beyond ``stop_z`` (a diverging
spread — the cointegration may have broken). The spread is the level spread
:math:`y_t - \beta_t x_t`, z-scored on a trailing window; ``static`` holds the hedge ratio
:math:`\beta` fixed (the Engle--Granger coefficient from the training window), while
``kalman`` lets it drift (the causal filtered estimate), so a relationship that moves is still
hedged well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from quantica.statarb.cointegration import engle_granger
from quantica.statarb.kalman import kalman_hedge_ratio

if TYPE_CHECKING:
    from quantica.core.types import FloatArray

__all__ = [
    "PairsBacktestResult",
    "PairsStrategyConfig",
    "pairs_backtest",
    "pairs_return_matrix",
]

_MIN_ZSCORE_WINDOW = 20  # minimum trailing points before a static z-score is trusted


@dataclass(frozen=True)
class PairsStrategyConfig:
    """Thresholds and costs for the z-score mean-reversion rule.

    Attributes
    ----------
    entry_z : float
        Enter when ``|z| >= entry_z`` (default 2.0): the spread is far from its mean.
    exit_z : float
        Exit when ``|z| <= exit_z`` (default 0.5): the spread has reverted.
    stop_z : float
        Stop out when ``|z| >= stop_z`` (default 4.0): the spread is diverging, not
        reverting — the relationship may have broken.
    zscore_window : int
        Trailing window (periods) for the **static** spread's rolling mean/std (default 60);
        ignored for the Kalman spread, which standardises by the filter's innovation variance.
    cost_rate : float
        One-way proportional trading cost per unit of leg turnover (default 0.0005 = 5 bps).
    """

    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    zscore_window: int = 60
    cost_rate: float = 0.0005

    def __post_init__(self) -> None:
        """Validate the threshold ordering and cost."""
        if not 0.0 <= self.exit_z < self.entry_z < self.stop_z:
            raise ValueError("require 0 <= exit_z < entry_z < stop_z")
        if self.zscore_window < _MIN_ZSCORE_WINDOW:
            raise ValueError(f"zscore_window must be >= {_MIN_ZSCORE_WINDOW}")
        if self.cost_rate < 0.0:
            raise ValueError(f"cost_rate must be non-negative, got {self.cost_rate}")


@dataclass(frozen=True)
class PairsBacktestResult:
    """Output of a pairs backtest: the out-of-sample return series and trade accounting.

    Attributes
    ----------
    net_returns : ndarray, shape (T_oos,)
        Per-period return after trading costs (the series fed to the validity layer).
    gross_returns : ndarray, shape (T_oos,)
        Per-period return before costs.
    positions : ndarray, shape (T_oos,)
        The held position each period: ``+1`` long the spread, ``-1`` short, ``0`` flat.
    spread : ndarray, shape (T_oos,)
        The level trading spread ``y - beta*x`` (the hedge ratio is constant for ``static``,
        drifting for ``kalman``).
    zscore : ndarray, shape (T_oos,)
        The spread z-score that drives entries and exits.
    hedge_ratio : ndarray, shape (T_oos,)
        The hedge ratio used each period (constant for ``static``, drifting for ``kalman``).
    n_trades : int
        Number of completed round-trips (entries).
    avg_holding_period : float
        Mean number of periods a position is held (compare against the spread half-life).
    hit_rate : float
        Fraction of completed round-trips that were profitable.
    total_cost : float
        Total trading cost paid over the backtest.
    """

    net_returns: FloatArray
    gross_returns: FloatArray
    positions: FloatArray
    spread: FloatArray
    zscore: FloatArray
    hedge_ratio: FloatArray
    n_trades: int
    avg_holding_period: float
    hit_rate: float
    total_cost: float

    def sharpe_ratio(self, periods_per_year: int = 252, *, gross: bool = False) -> float:
        r"""Annualised Sharpe ratio of the (net by default) out-of-sample return series."""
        series = self.gross_returns if gross else self.net_returns
        sd = float(np.std(series, ddof=1))
        if sd == 0.0:
            return 0.0
        return float(np.mean(series) / sd * np.sqrt(periods_per_year))


def pairs_backtest(
    y: FloatArray,
    x: FloatArray,
    config: PairsStrategyConfig | None = None,
    *,
    method: str = "static",
    train_window: int,
    hedge_ratio: float | None = None,
    process_var: float | None = None,
    obs_var: float | None = None,
) -> PairsBacktestResult:
    r"""Backtest the z-score mean-reversion rule on a pair, out of sample and net of costs.

    The hedge ratio and spread statistics are formed causally: the static hedge ratio is
    estimated on the first ``train_window`` observations, the Kalman hedge ratio is the
    filter's causal time-varying estimate, and the z-score uses a trailing window in both
    cases. Trading happens only *after* ``train_window`` (the estimation / filter burn-in), so
    there is no look-ahead. The per-period P&L of the dollar-neutral position is
    :math:`\text{pos}_{t-1}\,(\Delta y_t - \beta_{t-1}\,\Delta x_t)`, less
    ``cost_rate * |\Delta\text{pos}| * (1 + |\beta|)`` when the position changes.

    Parameters
    ----------
    y, x : ndarray, shape (T,)
        The two (log-)price series; ``y`` is traded against ``x`` via the hedge ratio.
    config : PairsStrategyConfig, optional
        Entry/exit/stop thresholds and cost (default :class:`PairsStrategyConfig`).
    method : {"static", "kalman"}, optional
        ``"static"`` uses the fixed OLS/Engle--Granger hedge ratio; ``"kalman"`` uses the
        drifting filtered hedge ratio in the level spread. Both z-score on a trailing window.
    train_window : int
        Estimation / burn-in length; trading starts after it (keyword-only).
    hedge_ratio : float, optional
        Override the static hedge ratio (else estimated by Engle--Granger on the train slice).
    process_var, obs_var : float, optional
        Kalman noise parameters; ``obs_var`` defaults to the training-spread variance and
        ``process_var`` to ``1e-4 * obs_var`` (a slowly drifting hedge ratio).

    Returns
    -------
    PairsBacktestResult
        The out-of-sample net/gross return series, positions, spread, and trade statistics.

    Raises
    ------
    ValueError
        If the inputs are not 1-D of equal length, ``train_window`` leaves too little data, or
        ``method`` is unknown.
    """
    cfg = config if config is not None else PairsStrategyConfig()
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    if y_arr.ndim != 1 or x_arr.ndim != 1 or y_arr.shape != x_arr.shape:
        raise ValueError("y and x must be 1-D series of equal length")
    n = y_arr.shape[0]
    if not 0 < train_window < n - 1:
        raise ValueError(f"train_window must be in (0, {n - 1}), got {train_window}")

    spread, zscore, beta_prev = _pairs_signal(
        y_arr, x_arr, cfg, method, train_window, hedge_ratio, process_var, obs_var
    )
    positions = _positions_from_zscore(zscore, cfg, train_window)

    # Dollar-neutral P&L: pos_{t-1} * (Δy - β_{t-1} Δx); costs on position changes.
    dy = np.diff(y_arr, prepend=y_arr[0])
    dx = np.diff(x_arr, prepend=x_arr[0])
    gross = np.zeros(n, dtype=np.float64)
    gross[1:] = positions[:-1] * (dy[1:] - beta_prev[:-1] * dx[1:])
    turnover = np.abs(np.diff(positions, prepend=0.0))
    costs = cfg.cost_rate * turnover * (1.0 + np.abs(beta_prev))
    net = gross - costs

    oos = slice(train_window, n)
    n_trades, avg_hold, hit_rate = _trade_stats(positions[oos], net[oos])
    return PairsBacktestResult(
        net_returns=net[oos],
        gross_returns=gross[oos],
        positions=positions[oos],
        spread=spread[oos],
        zscore=zscore[oos],
        hedge_ratio=beta_prev[oos],
        n_trades=n_trades,
        avg_holding_period=avg_hold,
        hit_rate=hit_rate,
        total_cost=float(costs[oos].sum()),
    )


def pairs_return_matrix(
    prices: FloatArray,
    pairs: list[tuple[int, int]],
    config: PairsStrategyConfig | None = None,
    *,
    method: str = "static",
    train_window: int,
) -> FloatArray:
    r"""Stack the out-of-sample net returns of many candidate pairs into a ``(T_oos, N)`` matrix.

    Runs :func:`pairs_backtest` for each ``(i, j)`` column pair of ``prices`` and stacks the
    resulting net-return series. This is the input to the overfitting layer
    (:func:`~quantica.portfolio.overfitting.deflated_sharpe_ratio_from_trials` /
    :func:`~quantica.portfolio.overfitting.probability_of_backtest_overfitting`) — mining many
    pairs is exactly the multiple-testing setting DSR/PBO exist to police.

    Parameters
    ----------
    prices : ndarray, shape (T, n_assets)
        The (log-)price panel.
    pairs : list of (int, int)
        Column-index pairs to backtest.
    config : PairsStrategyConfig, optional
        Strategy configuration (default :class:`PairsStrategyConfig`).
    method : {"static", "kalman"}, optional
        The spread construction (default ``"static"``).
    train_window : int
        Estimation / burn-in length (keyword-only).

    Returns
    -------
    ndarray, shape (T_oos, len(pairs))
        Column ``k`` is the OOS net-return series of ``pairs[k]``.
    """
    panel = np.asarray(prices, dtype=np.float64)
    if panel.ndim != 2:
        raise ValueError("prices must be a 2-D (T, n_assets) panel")
    columns = [
        pairs_backtest(
            panel[:, i], panel[:, j], config, method=method, train_window=train_window
        ).net_returns
        for i, j in pairs
    ]
    return np.column_stack(columns)


def _pairs_signal(
    y: FloatArray,
    x: FloatArray,
    cfg: PairsStrategyConfig,
    method: str,
    train_window: int,
    hedge_ratio: float | None,
    process_var: float | None,
    obs_var: float | None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the ``(spread, zscore, beta_prev)`` arrays for the requested spread method."""
    if method == "static":
        beta = (
            hedge_ratio
            if hedge_ratio is not None
            else engle_granger(y[:train_window], x[:train_window]).hedge_ratio
        )
        spread = y - beta * x
        zscore = _rolling_zscore(spread, cfg.zscore_window)
        beta_prev = np.full(y.shape[0], float(beta), dtype=np.float64)
        return spread, zscore, beta_prev
    if method == "kalman":
        train_spread = (
            y[:train_window]
            - engle_granger(y[:train_window], x[:train_window]).hedge_ratio * x[:train_window]
        )
        r = float(np.var(train_spread, ddof=1)) if obs_var is None else obs_var
        q = 1e-4 * r if process_var is None else process_var
        filtered = kalman_hedge_ratio(y, x, process_var=q, obs_var=r)
        # The dynamic *level* spread using the filtered (contemporaneous, causal) hedge
        # ratio — the tradeable deviation from equilibrium. The filter's raw innovation is
        # white by construction, so it carries no mean-reversion level to trade; the level
        # spread does, now with a hedge ratio that adapts as the relationship drifts.
        spread = y - filtered.hedge_ratio * x - filtered.intercept
        zscore = _rolling_zscore(spread, cfg.zscore_window)
        beta_prev = np.empty_like(filtered.hedge_ratio)
        beta_prev[0] = filtered.hedge_ratio[0]
        beta_prev[1:] = filtered.hedge_ratio[:-1]  # β held over [t-1, t] for the P&L
        return spread, zscore, beta_prev
    raise ValueError(f"method must be 'static' or 'kalman', got {method!r}")


def _rolling_zscore(series: FloatArray, window: int) -> FloatArray:
    """Trailing-window z-score (causal): standardise each point by its own past window."""
    n = series.shape[0]
    z = np.zeros(n, dtype=np.float64)
    for t in range(n):
        lo = max(0, t - window + 1)
        past = series[lo : t + 1]
        if past.shape[0] >= _MIN_ZSCORE_WINDOW:
            sd = float(np.std(past, ddof=1))
            if sd > 0.0:
                z[t] = (series[t] - float(np.mean(past))) / sd
    return z


def _positions_from_zscore(
    zscore: FloatArray, cfg: PairsStrategyConfig, train_window: int
) -> FloatArray:
    """Run the entry/exit/stop state machine over the z-score; no trading before burn-in."""
    n = zscore.shape[0]
    positions = np.zeros(n, dtype=np.float64)
    state = 0
    for t in range(train_window, n):
        if state == 0:
            if zscore[t] <= -cfg.entry_z:
                state = 1  # spread cheap -> long the spread
            elif zscore[t] >= cfg.entry_z:
                state = -1  # spread rich -> short the spread
        elif abs(zscore[t]) <= cfg.exit_z or abs(zscore[t]) >= cfg.stop_z:
            state = 0  # reverted (take profit) or diverged (stop out)
        positions[t] = state
    return positions


def _trade_stats(positions: FloatArray, net: FloatArray) -> tuple[int, float, float]:
    """Count round-trips and compute the mean holding period and hit rate."""
    n_trades = 0
    holding_periods: list[int] = []
    trade_pnls: list[float] = []
    in_trade = False
    length = 0
    pnl = 0.0
    for t in range(positions.shape[0]):
        active = positions[t] != 0.0
        if active and not in_trade:  # a new position opens
            n_trades += 1
            in_trade = True
            length = 0
            pnl = 0.0
        if in_trade:
            length += 1
            pnl += float(net[t])
            if not active:  # position just closed
                holding_periods.append(length)
                trade_pnls.append(pnl)
                in_trade = False
    if in_trade:  # a position still open at the end
        holding_periods.append(length)
        trade_pnls.append(pnl)
    avg_hold = float(np.mean(holding_periods)) if holding_periods else 0.0
    hit_rate = float(np.mean([p > 0.0 for p in trade_pnls])) if trade_pnls else 0.0
    return n_trades, avg_hold, hit_rate
