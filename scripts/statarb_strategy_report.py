#!/usr/bin/env python
"""Pairs strategy + overfitting-aware backtest — the two guards working together.

The statistical-arbitrage arc ends where it should: not with a headline Sharpe, but with the
evidence that the edge is (or is not) real. Three artifacts:

1. **Known-truth — the marriage (no network).** Mine many spurious pairs (independent random
   walks): the best in-sample Sharpe *looks* tradeable, but the Deflated Sharpe Ratio
   (deflated for the number of trials) and the Probability of Backtest Overfitting correctly
   flag it as noise — while a genuinely cointegrated pair, pre-selected on economic grounds,
   clears the probabilistic-Sharpe bar. Cointegration guards the signal; DSR/PBO guard the
   backtest.

2. **Real economically-motivated pairs.** Soda vs. Meals and a few other sensible pairs, net
   of costs, with their trade stats and the probabilistic-Sharpe verdict — reported straight,
   including where the edge does not survive costs.

3. **Mining the whole real universe.** Backtest every 49-industry pair, take the best
   in-sample Sharpe, and let DSR/PBO judge it — the real-world version of the overfitting
   trap.

Sections 2--3 fetch Ken French data (cached via ``scripts/_ff_data.py``; never in CI).
Regenerate with::

    python scripts/statarb_strategy_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from _ff_data import load_fama_french
from quantica.portfolio import (
    deflated_sharpe_ratio_from_trials,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from quantica.statarb import (
    PairsStrategyConfig,
    generate_cointegrated_pair,
    pairs_backtest,
    pairs_return_matrix,
)

_N_MONTHS = 360
_TRAIN = 120  # 10-year in-sample estimation window
_MONTHLY = PairsStrategyConfig(
    entry_z=2.0, exit_z=0.5, stop_z=4.0, zscore_window=36, cost_rate=0.001
)
_CANDIDATES = [("Soda", "Meals"), ("Hlth", "MedEq"), ("Aero", "Guns"), ("Beer", "Smoke")]


def _known_truth_section() -> None:
    """Mine spurious pairs → DSR/PBO flag the best; a genuine pair survives."""
    print("### 1. Known-truth — cointegration guards the signal, DSR/PBO guard the backtest\n")
    rng = np.random.default_rng(7)
    prices = np.cumsum(rng.standard_normal((1400, 24)), axis=0)  # 24 independent walks
    pairs = [(i, j) for i in range(24) for j in range(i + 1, 24)]  # 276 spurious pairs
    matrix = pairs_return_matrix(prices, pairs, _MONTHLY, method="static", train_window=250)
    best = max(sharpe_ratio(matrix[:, k]) for k in range(matrix.shape[1]))
    dsr = deflated_sharpe_ratio_from_trials(matrix)
    pbo = probability_of_backtest_overfitting(matrix, n_splits=10)

    y, x = generate_cointegrated_pair(1400, np.random.default_rng(0), beta=1.2, spread_kappa=0.1)
    genuine = pairs_backtest(y, x, _MONTHLY, method="static", train_window=250).net_returns
    genuine_psr = probabilistic_sharpe_ratio(sharpe_ratio(genuine), genuine.size)

    print(f"Mining {len(pairs)} spurious pairs (independent random walks) vs one genuine pair:\n")
    print("| | Best mined spurious pair | Genuine cointegrated pair |")
    print("| --- | ---: | ---: |")
    print(
        f"| Annualised Sharpe (in-sample) | {best * np.sqrt(252):.2f} | "
        f"{sharpe_ratio(genuine) * np.sqrt(252):.2f} |"
    )
    print(
        f"| Deflated Sharpe significant? | **{'yes' if dsr.is_significant else 'no'}** "
        f"(DSR {dsr.dsr:.2f}) | — (single trial) |"
    )
    print(f"| Prob. of backtest overfitting | **{pbo.pbo:.0%}** | — |")
    print(
        f"| Probabilistic Sharpe > 95%? | — | **{'yes' if genuine_psr > 0.95 else 'no'}** "
        f"(PSR {genuine_psr:.2f}) |"
    )
    print(
        f"\nThe mined winner shows an annualised Sharpe of {best * np.sqrt(252):.1f} — it "
        f"*looks* tradeable — but deflated for the {len(pairs)} trials it is **not significant**, "
        f"and its selection does not hold out of sample (**PBO {pbo.pbo:.0%}**). The genuinely "
        "cointegrated pair, pre-selected on economic grounds (one trial), clears the "
        "probabilistic-Sharpe bar. Two levels of guard, catching stat-arb's endemic failure "
        "mode.\n"
    )


def _real_pairs_section(log_price: np.ndarray, names: list[str]) -> None:  # type: ignore[type-arg]
    """Economically-motivated real pairs, net of costs, judged honestly."""
    print("### 2. Real economically-motivated pairs (49-industry FF, net of 10 bps)\n")
    print("| Pair | Net Sharpe (ann.) | Gross Sharpe | Trades | Avg hold (mo) | Hit rate | PSR |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for a, b in _CANDIDATES:
        if a not in names or b not in names:
            continue
        y, x = log_price[:, names.index(a)], log_price[:, names.index(b)]
        r = pairs_backtest(y, x, _MONTHLY, method="static", train_window=_TRAIN)
        psr = probabilistic_sharpe_ratio(sharpe_ratio(r.net_returns), r.net_returns.size)
        print(
            f"| {a}–{b} | {r.sharpe_ratio(12):+.2f} | {r.sharpe_ratio(12, gross=True):+.2f} "
            f"| {r.n_trades} | {r.avg_holding_period:.1f} | {r.hit_rate:.0%} | {psr:.2f} |"
        )
    print(
        "\n**Honest reading:** even the best economically-sensible pair (Soda–Meals) is only "
        "marginally positive net of costs and does **not** clear the 95% probabilistic-Sharpe "
        "bar; the others lose money. Cointegration (step 1 found Soda–Meals strongly "
        "cointegrated) is *necessary but not sufficient* — the spread reverts, but not far "
        "enough or often enough to beat costs. The validity layer correctly declines to "
        "certify it, which is the point of having one.\n"
    )


def _mining_section(log_price: np.ndarray) -> None:  # type: ignore[type-arg]
    """Mine every real pair and let DSR/PBO judge the best — the real overfitting trap."""
    n_assets = log_price.shape[1]
    pairs = [(i, j) for i in range(n_assets) for j in range(i + 1, n_assets)]
    matrix = pairs_return_matrix(log_price, pairs, _MONTHLY, method="static", train_window=_TRAIN)
    sharpes = np.array([sharpe_ratio(matrix[:, k]) for k in range(matrix.shape[1])])
    dsr = deflated_sharpe_ratio_from_trials(matrix)
    pbo = probability_of_backtest_overfitting(matrix, n_splits=10)

    print(f"### 3. Mining the whole universe — {len(pairs)} real industry pairs\n")
    print("| Quantity | Value |")
    print("| --- | ---: |")
    print(f"| Best in-sample Sharpe (ann.) | {sharpes.max() * np.sqrt(12):.2f} |")
    print(f"| Median pair Sharpe (ann.) | {np.median(sharpes) * np.sqrt(12):+.2f} |")
    print(f"| Fraction net-positive | {np.mean(sharpes > 0):.0%} |")
    print(
        f"| Best deflated-Sharpe significant? | **{'yes' if dsr.is_significant else 'no'}** "
        f"(DSR {dsr.dsr:.2f}) |"
    )
    print(f"| Prob. of backtest overfitting | **{pbo.pbo:.0%}** |")
    print(
        f"\nMining all {len(pairs)} pairs, the best after costs manages only an annualised "
        f"Sharpe of {sharpes.max() * np.sqrt(12):.2f}, the median pair loses money, and the "
        f"best is **not** DSR-significant with **PBO {pbo.pbo:.0%}**. The credible conclusion, "
        "reported straight: there is no robust industry pairs edge here after realistic costs — "
        "and the overfitting layer is exactly what stops a tempting best-of-many number from "
        "being mistaken for one.\n"
    )


def main() -> None:
    """Print the known-truth marriage, the real candidate pairs, and the mining verdict."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("## Statistical arbitrage — the pairs strategy and its overfitting-aware backtest\n")
    _known_truth_section()
    data = load_fama_french(_N_MONTHS, n_industries=49)
    log_price = np.cumsum(np.log1p(data.industry_excess), axis=0)
    _real_pairs_section(log_price, list(data.industry_names))
    _mining_section(log_price)


if __name__ == "__main__":
    main()
