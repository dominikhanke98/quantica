#!/usr/bin/env python
"""Black--Litterman — stabilising mean-variance (the BL headline).

Naive Markowitz treats noisy return estimates as truth and inverts the covariance, so
small changes in the estimates swing the weights wildly. Black--Litterman shrinks the
estimates toward the market-implied equilibrium, so the posterior — and the weights —
barely move. Two artifacts on the 49-industry Fama--French universe:

1. **Input-perturbation stability** — perturb the expected-return estimate and measure
   how far the naive mean-variance weights move versus the Black--Litterman weights (the
   classic reason BL exists).
2. **Net-of-cost backtest** — Black--Litterman versus naive mean-variance through the
   walk-forward engine with costs; BL's stability shows up as far lower turnover.

Requires network access (data via ``scripts/_ff_data.py``, cached, never in CI).
Regenerate with::

    python scripts/black_litterman_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from _ff_data import load_fama_french
from quantica.factor.estimators import LedoitWolfCovariance
from quantica.portfolio import (
    BlackLittermanStrategy,
    MeanVarianceStrategy,
    PortfolioConstraints,
    ProportionalCosts,
    black_litterman,
    implied_equilibrium_returns,
    mean_variance_weights,
    walk_forward_backtest,
)

_N_MONTHS = 240
_TRAIN_WINDOW = 60
_REBALANCE_EVERY = 3
_MONTHS_PER_YEAR = 12
_COST_RATE = 0.001  # 10 bps one-way
_RISK_AVERSION = 3.0
_PERTURBATION = 0.01  # 1% shock to the monthly return estimates
_N_TRIALS = 50
_SEED = 20240723


def _stability_section(assets: np.ndarray) -> None:
    # A well-conditioned covariance (Ledoit--Wolf on the full sample) isolates the
    # effect being demonstrated: sensitivity to the *return* estimate, not the covariance.
    cov = LedoitWolfCovariance().estimate(assets)
    n = cov.shape[0]
    market = np.full(n, 1.0 / n)  # equal-weight benchmark (no cap data for industries)
    equilibrium = implied_equilibrium_returns(cov, market, _RISK_AVERSION)
    # Unconstrained (budget-only) mean-variance, so the return-estimate sensitivity is
    # exposed: a long-only cap would itself stabilise the weights (Jagannathan--Ma) and
    # mask the effect BL is demonstrating.
    unconstrained = PortfolioConstraints()

    def naive_mv(mu: np.ndarray) -> np.ndarray:
        return mean_variance_weights(mu, cov, _RISK_AVERSION, unconstrained)

    def bl(view_shift: np.ndarray) -> np.ndarray:
        result = black_litterman(
            cov, market, _RISK_AVERSION, views_p=np.eye(n), views_q=equilibrium + view_shift
        )
        return mean_variance_weights(
            result.posterior_returns, result.posterior_cov, _RISK_AVERSION, unconstrained
        )

    rng = np.random.default_rng(_SEED)
    mv_swings, bl_swings = [], []
    for _ in range(_N_TRIALS):
        eps = rng.normal(0.0, _PERTURBATION, n)
        mv_swings.append(float(np.sum(np.abs(naive_mv(equilibrium + eps) - naive_mv(equilibrium)))))
        bl_swings.append(float(np.sum(np.abs(bl(eps) - bl(np.zeros(n))))))

    mv_mean, bl_mean = float(np.mean(mv_swings)), float(np.mean(bl_swings))
    print("### 1. Input-perturbation stability (mean L1 weight swing)\n")
    print(
        f"49 industries, unconstrained (budget-only) mean-variance, {_PERTURBATION:.0%} "
        f"perturbations of the expected-return estimate over {_N_TRIALS} seeded trials:\n"
    )
    print("| Construction | Mean L1 weight change |")
    print("| --- | ---: |")
    print(f"| naive mean-variance | {mv_mean:.3f} |")
    print(f"| **Black–Litterman** | **{bl_mean:.3f}** |")
    print(
        f"\nA {_PERTURBATION:.0%} wiggle in the return estimate moves the naive "
        f"mean-variance weights by **{mv_mean:.2f}** (L1) on average; the same shock, "
        f"expressed as Black–Litterman views, moves the weights only **{bl_mean:.2f}** — "
        f"**{mv_mean / bl_mean:.0f}× more stable**. That shrinkage toward equilibrium is "
        f"exactly why BL is the institutional workhorse.\n"
    )


def _backtest_section(assets: np.ndarray, factors: np.ndarray, dates: np.ndarray) -> None:
    long_only = PortfolioConstraints(long_only=True, max_position=0.20)
    costs = ProportionalCosts(_COST_RATE)
    strategies = [
        MeanVarianceStrategy(
            LedoitWolfCovariance(),
            risk_aversion=_RISK_AVERSION,
            constraints=long_only,
            name="naive mean-variance",
        ),
        BlackLittermanStrategy(
            LedoitWolfCovariance(),
            risk_aversion=_RISK_AVERSION,
            constraints=long_only,
            name="black-litterman",
        ),
    ]
    print("### 2. Net-of-cost walk-forward backtest\n")
    print(
        f"49 industries, {len(dates)} months ({dates[0]}--{dates[-1]}), {_TRAIN_WINDOW}-month "
        f"train, rebalanced every {_REBALANCE_EVERY} months, long-only 20% cap, "
        f"{_COST_RATE * 1e4:.0f} bps costs. Same covariance estimator; the only difference "
        f"is whether the return estimate is shrunk toward equilibrium:\n"
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
        "\nHonest reading: under a realistic long-only 20% cap the two are near-identical "
        "in Sharpe and turnover — because the constraint *itself* regularises the weights "
        "(the same Jagannathan–Ma effect the factor pillar measures), leaving little for the "
        "equilibrium anchor to add. **Black–Litterman's decisive advantage is in the "
        "unconstrained regime** (section 1, ~7× more stable): it is the shrinkage that lets "
        "you run mean-variance *without* leaning on hard constraints to tame it.\n"
    )


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    data = load_fama_french(_N_MONTHS, n_industries=49)
    print("## Black–Litterman — stabilising mean-variance report\n")
    _stability_section(data.industry_excess)
    _backtest_section(data.industry_excess, data.factor_returns, data.dates)


if __name__ == "__main__":
    main()
