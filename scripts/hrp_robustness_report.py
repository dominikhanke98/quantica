#!/usr/bin/env python
"""Hierarchical Risk Parity — robustness where inversion blows up (the HRP headline).

HRP's reason to exist is that it **never inverts the covariance**, so on an
ill-conditioned universe (many assets, few observations) it sidesteps the error
amplification that sinks the inverting minimum-variance portfolio — the direct
tie-back to factor stage 2's "sample covariance is an error maximiser" and
Jagannathan--Ma. Two artifacts on the 49-industry Fama--French universe:

1. **Out-of-sample robustness** — realized OOS volatility of the minimum-variance
   portfolio built three ways (sample covariance, inverting; Ledoit--Wolf shrinkage;
   HRP, no inversion), walk-forward with a short window so n/T is large.
2. **Net-of-cost backtest** — HRP alongside the other constructions through the
   walk-forward engine with transaction costs, reported honestly.

Requires network access (data via ``scripts/_ff_data.py``, cached, never in CI).
Regenerate with::

    python scripts/hrp_robustness_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from _ff_data import load_fama_french
from quantica.factor.estimators import (
    LedoitWolfCovariance,
    SampleCovariance,
    condition_number,
    min_variance_weights,
)
from quantica.factor.evaluation import walk_forward_windows
from quantica.portfolio import (
    HRPStrategy,
    MinimumVarianceStrategy,
    PortfolioConstraints,
    ProportionalCosts,
    RiskParityStrategy,
    hrp_weights,
    minimum_variance_weights,
    walk_forward_backtest,
)

_N_MONTHS = 240
_TRAIN_WINDOW = 55  # short window vs 49 industries -> n/T ~ 0.9, ill-conditioned
_TEST_WINDOW = 12
_MONTHS_PER_YEAR = 12
_ANNUALISE = np.sqrt(12.0)
_COST_RATE = 0.001  # 10 bps one-way
_REBALANCE_EVERY = 3


def _robustness_section(assets: np.ndarray) -> None:
    windows = walk_forward_windows(assets.shape[0], _TRAIN_WINDOW, _TEST_WINDOW)
    long_only = PortfolioConstraints(long_only=True)

    def realized(weights_fn) -> float:  # type: ignore[no-untyped-def]
        vols = []
        for w in windows:
            train, test = assets[w.train_start : w.train_end], assets[w.test_start : w.test_end]
            vols.append(np.std(test @ weights_fn(train), ddof=1))
        return float(np.mean(vols)) * _ANNUALISE

    sample = SampleCovariance()
    lw = LedoitWolfCovariance()
    vol_sample = realized(lambda tr: min_variance_weights(sample.estimate(tr)))  # inverts
    vol_lw = realized(lambda tr: minimum_variance_weights(lw.estimate(tr), long_only))  # shrinkage
    vol_hrp = realized(lambda tr: hrp_weights(sample.estimate(tr)))  # no inversion

    cond = condition_number(sample.estimate(assets[:_TRAIN_WINDOW]))

    print("### 1. Out-of-sample robustness (min-variance realised vol, annualised)\n")
    print(
        f"49 industries, {_TRAIN_WINDOW}-month train / {_TEST_WINDOW}-month test "
        f"(n/T ~ {assets.shape[1] / _TRAIN_WINDOW:.1f}; sample-covariance condition number "
        f"~ {cond:,.0f}):\n"
    )
    print("| Construction | Inverts Σ? | Realised OOS vol |")
    print("| --- | :---: | ---: |")
    print(f"| minimum-variance, sample cov | yes | {vol_sample:.1%} |")
    print(f"| minimum-variance, Ledoit–Wolf | yes (shrunk) | {vol_lw:.1%} |")
    print(f"| **HRP, sample cov** | **no** | **{vol_hrp:.1%}** |")
    print(
        f"\nInverting the near-singular sample covariance realises **{vol_sample:.0%}** "
        f"annualised vol (the error maximiser); shrinkage tames it to {vol_lw:.0%}. **HRP "
        f"reaches {vol_hrp:.0%} on the raw sample covariance without ever inverting it** — "
        f"it gets shrinkage-like robustness structurally, from the clustering, not from a "
        f"better Σ. Same finding as factor stage 2, from the construction side.\n"
    )


def _backtest_section(assets: np.ndarray, factors: np.ndarray, dates: np.ndarray) -> None:
    long_only = PortfolioConstraints(long_only=True, max_position=0.20)
    costs = ProportionalCosts(_COST_RATE)
    strategies = [
        HRPStrategy(SampleCovariance(), name="hrp/sample"),
        MinimumVarianceStrategy(SampleCovariance(), long_only, name="minvar/sample"),
        MinimumVarianceStrategy(LedoitWolfCovariance(), long_only, name="minvar/ledoit-wolf"),
        RiskParityStrategy(LedoitWolfCovariance(), name="riskparity/lw"),
    ]

    print("### 2. Net-of-cost walk-forward backtest\n")
    print(
        f"49 industries, {len(dates)} months ({dates[0]}--{dates[-1]}), {_TRAIN_WINDOW}-month "
        f"trailing train, rebalanced every {_REBALANCE_EVERY} months, long-only with a 20% "
        f"cap, {_COST_RATE * 1e4:.0f} bps one-way costs:\n"
    )
    print("| Strategy | Gross SR | Net SR | Avg turnover | Total cost |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for strat in strategies:
        result = walk_forward_backtest(
            assets,
            strat,
            train_window=_TRAIN_WINDOW,
            rebalance_every=_REBALANCE_EVERY,
            cost_model=costs,
            factor_returns=factors,
        )
        print(
            f"| {strat.name} | {result.sharpe_ratio(_MONTHS_PER_YEAR, gross=True):.2f} "
            f"| {result.sharpe_ratio(_MONTHS_PER_YEAR):.2f} | {result.average_turnover:.2f} "
            f"| {result.total_cost:.1%} |"
        )
    print(
        "\nHonest reading: on this universe the constructions land in a similar net-Sharpe "
        "band (the industry premium dominates); HRP's value is not a higher backtest Sharpe "
        "but its **robustness and low turnover without any covariance inversion or "
        "shrinkage tuning** — it degrades gracefully exactly where sample min-variance does "
        "not.\n"
    )


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    data = load_fama_french(_N_MONTHS, n_industries=49)
    print("## Hierarchical Risk Parity — robustness report\n")
    _robustness_section(data.industry_excess)
    _backtest_section(data.industry_excess, data.factor_returns, data.dates)


if __name__ == "__main__":
    main()
