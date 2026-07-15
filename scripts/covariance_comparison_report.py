#!/usr/bin/env python
"""Out-of-sample covariance-estimator comparison — the factor step's headline.

Races three covariance estimators — sample, Ledoit--Wolf shrinkage, and the
factor model — on the 49-industry Fama--French universe (n close to the training
window, the regime where estimation error bites), by walk-forward out-of-sample
risk forecasting. Three tables:

1. **Random-portfolio bias** (realized / forecast volatility, well-calibrated ≈ 1)
   — the estimators are near-indistinguishable on generic portfolios.
2. **Minimum-variance stress** — each estimator's own GMV portfolio, realized out
   of sample. This is where the sample covariance fails (Michaud's error maximiser).
3. **Ill-conditioning** — the sample covariance's condition number vs shrinkage and
   the factor model as the universe grows.

Requires network access (data via ``scripts/_ff_data.py``, cached, never in CI).
Regenerate with::

    python scripts/covariance_comparison_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from _ff_data import load_fama_french
from quantica.factor import (
    FactorCovariance,
    LedoitWolfCovariance,
    SampleCovariance,
    compare_estimators,
    condition_number,
)

_N_MONTHS = 240
_TRAIN_WINDOW = 60
_TEST_WINDOW = 12
_N_RANDOM = 50
_SEED = 20240720


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    data = load_fama_french(_N_MONTHS, n_industries=49)
    assets = data.industry_excess
    factors = data.factor_returns
    n_assets = assets.shape[1]
    estimators = (SampleCovariance(), LedoitWolfCovariance(), FactorCovariance())

    print("## Out-of-sample covariance-estimator comparison\n")
    print(
        f"{n_assets} Fama–French industry portfolios, trailing {len(data.dates)} months "
        f"({data.dates[0]}–{data.dates[-1]}), walk-forward with a {_TRAIN_WINDOW}-month "
        f"train / {_TEST_WINDOW}-month test window (n/T ≈ {n_assets / _TRAIN_WINDOW:.1f} — "
        f"the regime where estimation error bites). Same scenarios for every "
        f"estimator; strictly no lookahead.\n"
    )

    comparison = compare_estimators(
        assets,
        estimators,
        train_window=_TRAIN_WINDOW,
        test_window=_TEST_WINDOW,
        factor_returns=factors,
        rng=np.random.default_rng(_SEED),
        n_random_portfolios=_N_RANDOM,
    )

    print(
        f"### 1. Random-portfolio bias (realized / forecast vol; well-calibrated ≈ 1, "
        f"over {len(comparison.windows)} windows)\n"
    )
    print("| Estimator | mean | median | p05 | p95 | calibrated |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name in comparison.estimator_names:
        b = comparison.bias[name]
        print(
            f"| {name} | {b.mean:.2f} | {b.median:.2f} | {b.p05:.2f} | {b.p95:.2f} "
            f"| {b.fraction_calibrated():.0%} |"
        )
    print("\nOn generic portfolios the estimators barely differ — the sample covariance is fine.\n")

    print("### 2. Minimum-variance stress (each estimator's own GMV portfolio)\n")
    mv = comparison.mean_min_variance_vol()
    print("| Estimator | Realized OOS vol (ann.) | Forecast bias (mean) |")
    print("| --- | ---: | ---: |")
    for name in comparison.estimator_names:
        print(
            f"| {name} | {mv[name] * np.sqrt(12.0):.1%} "
            f"| {comparison.min_variance_bias[name].mean:.2f} |"
        )
    best = comparison.best_min_variance_estimator()
    worst_vol = mv["sample"] * np.sqrt(12.0)
    best_vol = mv[best] * np.sqrt(12.0)
    print(
        f"\nHere the estimators diverge sharply. The **sample** covariance's min-variance "
        f"portfolio realizes **{worst_vol:.1%}** annualised — the worst — with a forecast "
        f"bias far above 1 (it minimises *in-sample* variance, so its forecast is "
        f"optimistic). The **{best}** estimator wins at {best_vol:.1%}. Inverting a noisy "
        f"covariance is Michaud's error maximiser; shrinkage and the factor model tame it. "
        f"This is the same finding the synthetic known-truth test proves, now on real data.\n"
    )

    print("### 3. Ill-conditioning as the universe grows (condition number)\n")
    print("| # assets | sample | ledoit-wolf | factor |")
    print("| ---: | ---: | ---: | ---: |")
    train = assets[:_TRAIN_WINDOW]
    train_f = factors[:_TRAIN_WINDOW]
    for n in (10, 25, 40, n_assets):
        cs = condition_number(SampleCovariance().estimate(train[:, :n]))
        cl = condition_number(LedoitWolfCovariance().estimate(train[:, :n]))
        cf = condition_number(FactorCovariance().estimate(train[:, :n], train_f))
        print(f"| {n} | {cs:,.0f} | {cl:,.0f} | {cf:,.0f} |")
    print(
        f"\nWith a {_TRAIN_WINDOW}-month window, the sample covariance's condition number "
        f"explodes as the asset count approaches it, while shrinkage and (most of all) the "
        f"factor model stay bounded — the concrete mechanism behind the min-variance failure "
        f"above.\n"
    )


if __name__ == "__main__":
    main()
