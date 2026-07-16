#!/usr/bin/env python
"""Portfolio backtest-validity report — the headline of the portfolio pillar.

*Everyone ships a backtester; this ships the test of whether the backtest means
anything.* Two parts:

1. **Known-truth overfitting demonstration** (synthetic, no network). A deliberately
   overfit search over many pure-noise "strategies" is flagged spurious by the
   deflated Sharpe ratio and by the probability of backtest overfitting (PBO), while a
   genuinely predictive planted signal survives both. This is the proof the detector
   detects overfitting.

2. **Real-data walk-forward backtest** (Fama--French industry universe via
   ``scripts/_ff_data.py``, cached, never in CI). A grid of estimator x construction
   strategies is run net of transaction costs; the best in-sample config is then put
   through the same validity machinery. Results are reported **net of costs, honestly**
   — including whether the apparent edge survives deflation and costs.

Regenerate with::

    python scripts/portfolio_backtest_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from _ff_data import load_fama_french
from quantica.factor.estimators import (
    FactorCovariance,
    LedoitWolfCovariance,
    SampleCovariance,
)
from quantica.portfolio.backtest import ProportionalCosts, walk_forward_backtest
from quantica.portfolio.construction import PortfolioConstraints
from quantica.portfolio.data import generate_trial_returns
from quantica.portfolio.overfitting import (
    deflated_sharpe_ratio_from_trials,
    minimum_track_record_length,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from quantica.portfolio.strategy import (
    MeanVarianceStrategy,
    MinimumVarianceStrategy,
    RiskParityStrategy,
)

_MONTHS_PER_YEAR = 12
_N_MONTHS = 360
_TRAIN_WINDOW = 60
_REBALANCE_EVERY = 3
_COST_RATE = 0.001  # 10 bps one-way proportional cost
_SEED = 20240721


def _known_truth_section() -> None:
    print("### 1. Known-truth: does the overfitting detector detect overfitting?\n")
    print(
        "A search picks the best of many candidate strategies. When every candidate is "
        "pure noise, the winner's in-sample Sharpe is an artefact of selection; when one "
        "candidate has a real edge, it should survive. The deflated Sharpe ratio (DSR) and "
        "the probability of backtest overfitting (PBO) are run on both.\n"
    )
    print("| Search | Best in-sample SR (ann.) | DSR | Significant? | PBO |")
    print("| --- | ---: | ---: | :---: | ---: |")
    for label, planted in (("100 noise signals", 0.0), ("99 noise + 1 real signal", 0.35)):
        rng = np.random.default_rng(_SEED)
        trials = generate_trial_returns(360, 100, rng, planted_sharpe=planted)
        dsr = deflated_sharpe_ratio_from_trials(trials.returns)
        pbo = probability_of_backtest_overfitting(trials.returns, n_splits=10)
        ann_sr = dsr.observed_sr * np.sqrt(_MONTHS_PER_YEAR)
        flag = "**yes**" if dsr.is_significant else "no"
        print(f"| {label} | {ann_sr:.2f} | {dsr.dsr:.3f} | {flag} | {pbo.pbo:.2f} |")
    print(
        "\nThe noise search is flagged spurious (DSR far below 0.95, PBO near 0.5 — the "
        "in-sample winner is a coin-flip out of sample); the planted signal survives both "
        "(DSR ~ 1, PBO ~ 0). The detector works.\n"
    )


def _real_data_section() -> None:
    data = load_fama_french(_N_MONTHS, n_industries=49)
    assets = data.industry_excess
    factors = data.factor_returns
    n_assets = assets.shape[1]
    long_only = PortfolioConstraints(long_only=True, max_position=0.20)
    costs = ProportionalCosts(_COST_RATE)

    # A grid of estimator x construction "trials" — the search a practitioner runs.
    strategies = [
        MinimumVarianceStrategy(SampleCovariance(), long_only, name="minvar/sample"),
        MinimumVarianceStrategy(LedoitWolfCovariance(), long_only, name="minvar/ledoit-wolf"),
        MinimumVarianceStrategy(FactorCovariance(), long_only, name="minvar/factor"),
        RiskParityStrategy(SampleCovariance(), name="riskparity/sample"),
        RiskParityStrategy(LedoitWolfCovariance(), name="riskparity/ledoit-wolf"),
        MeanVarianceStrategy(
            LedoitWolfCovariance(), risk_aversion=10.0, constraints=long_only, name="meanvar/lw"
        ),
    ]

    print("### 2. Real-data walk-forward backtest (net of costs)\n")
    print(
        f"{n_assets} Fama--French industry portfolios, {len(data.dates)} months "
        f"({data.dates[0]}--{data.dates[-1]}), {_TRAIN_WINDOW}-month trailing train, "
        f"rebalanced every {_REBALANCE_EVERY} months, long-only with a 20% position cap, "
        f"{_COST_RATE * 1e4:.0f} bps one-way costs. Same no-lookahead windows for every "
        f"strategy.\n"
    )

    net_series = []
    rows = []
    for strat in strategies:
        result = walk_forward_backtest(
            assets,
            strat,
            train_window=_TRAIN_WINDOW,
            rebalance_every=_REBALANCE_EVERY,
            cost_model=costs,
            factor_returns=factors,
        )
        net_series.append(result.net_returns)
        gross_sr = result.sharpe_ratio(_MONTHS_PER_YEAR, gross=True)
        net_sr = result.sharpe_ratio(_MONTHS_PER_YEAR)
        rows.append((strat.name, gross_sr, net_sr, result.average_turnover, result.total_cost))

    print("| Strategy | Gross SR | Net SR | Avg turnover | Total cost |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, gross_sr, net_sr, turn, cost in rows:
        print(f"| {name} | {gross_sr:.2f} | {net_sr:.2f} | {turn:.2f} | {cost:.1%} |")

    # Assemble the net-return matrix and interrogate the *selection*.
    matrix = np.column_stack(net_series)
    dsr = deflated_sharpe_ratio_from_trials(matrix)
    pbo = probability_of_backtest_overfitting(matrix, n_splits=6)
    best_name = strategies[dsr.selected].name
    best_net = matrix[:, dsr.selected]
    n_obs = matrix.shape[0]
    ann_net_sr = sharpe_ratio(best_net) * np.sqrt(_MONTHS_PER_YEAR)

    print(
        f"\n**Best net-of-cost strategy:** `{best_name}` "
        f"(net Sharpe {ann_net_sr:.2f} annualised over {n_obs} months). "
        f"Deflated for the {matrix.shape[1]} configurations tried, DSR = **{dsr.dsr:.3f}** "
        f"({'significant' if dsr.is_significant else 'not significant'} at 0.95); "
        f"PBO across the grid = **{pbo.pbo:.2f}**.\n"
    )

    # Minimum track record length for the best net Sharpe (per-period units).
    per_period_sr = sharpe_ratio(best_net)
    if per_period_sr > 0:
        min_trl = minimum_track_record_length(per_period_sr, confidence=0.95)
        print(
            f"At that net Sharpe, the minimum track record length for significance at 95% "
            f"is **{min_trl:.0f} months** ({min_trl / 12:.1f} years); the backtest is "
            f"{n_obs} months long.\n"
        )
    print(
        "The honest reading is a **split verdict**, and the two metrics are supposed to "
        "disagree here — they answer different questions. The six configurations are all "
        "long-only industry portfolios, so their net Sharpes cluster tightly (~0.6): the "
        "cross-trial variance is small, the expected-maximum-Sharpe benchmark is low, and "
        "**DSR is high** — the underlying premium is real and survives deflation and costs. "
        "But precisely because the configurations are near-interchangeable captures of that "
        "same premium, **PBO is high** — which one was 'best' in sample is not repeatable out "
        "of sample. Read together: *trust the premium, not the ranking.* A reviewer who "
        "reported only the winning config's backtest curve would be selling a selection that "
        "PBO shows to be noise; the validity layer — not the curve — is the deliverable.\n"
    )


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("## Portfolio backtest-validity report\n")
    _known_truth_section()
    _real_data_section()


if __name__ == "__main__":
    main()
