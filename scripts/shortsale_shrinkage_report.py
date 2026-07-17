#!/usr/bin/env python
"""No-short-sale as covariance shrinkage — the cross-pillar resolution (Jagannathan-Ma).

Two of this repo's findings look contradictory:

* **Factor stage 2** — the sample covariance is the *worst* estimator under
  (unconstrained) minimum-variance optimisation: its GMV realises ~2x the OOS
  volatility of a shrunk estimator, on a 6x-optimistic forecast (Michaud's error
  maximiser).
* **Portfolio backtest** — the best net-of-cost strategy is `minvar/sample`,
  minimum-variance on the *same* sample covariance, but **long-only**.

Jagannathan & Ma (2003) resolve it: a no-short-sale constraint is *equivalent* to
shrinking the covariance, so the long-only constraint regularises exactly the
estimation error that sinks the unconstrained sample-covariance portfolio. This report
shows both the outcome (the 2x2 realised-vol table) and the exact mechanism (the
implied-shrinkage equivalence to machine precision) on the 49-industry Fama-French
universe.

Requires network access (data via ``scripts/_ff_data.py``, cached, never in CI).
Regenerate with::

    python scripts/shortsale_shrinkage_report.py

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
from quantica.portfolio.construction import PortfolioConstraints, minimum_variance_weights

_N_MONTHS = 240
_TRAIN_WINDOW = 60
_TEST_WINDOW = 12
_ANNUALISE = np.sqrt(12.0)


def _realized_vol(assets: np.ndarray, weights_fn) -> float:  # type: ignore[no-untyped-def]
    """Annualised realised OOS volatility of a walk-forward min-variance strategy.

    Averages each test window's realised volatility (matching the aggregation in
    ``covariance_comparison_report.py``, so the sample-covariance number reconciles
    with factor stage 2's).
    """
    windows = walk_forward_windows(assets.shape[0], _TRAIN_WINDOW, _TEST_WINDOW)
    window_vols = []
    for window in windows:
        train = assets[window.train_start : window.train_end]
        test = assets[window.test_start : window.test_end]
        window_vols.append(np.std(test @ weights_fn(train), ddof=1))
    return float(np.mean(window_vols)) * _ANNUALISE


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    data = load_fama_french(_N_MONTHS, n_industries=49)
    assets = data.industry_excess
    n_assets = assets.shape[1]
    sample, lw = SampleCovariance(), LedoitWolfCovariance()
    long_only = PortfolioConstraints(long_only=True)

    print("## No-short-sale is covariance shrinkage (Jagannathan-Ma 2003)\n")
    print(
        f"{n_assets} Fama-French industry portfolios, trailing {len(data.dates)} months "
        f"({data.dates[0]}-{data.dates[-1]}), walk-forward {_TRAIN_WINDOW}-month train / "
        f"{_TEST_WINDOW}-month test (n/T ~ {n_assets / _TRAIN_WINDOW:.1f}). Realised "
        f"out-of-sample volatility of the minimum-variance portfolio, annualised:\n"
    )

    uncon_sample = _realized_vol(assets, lambda tr: min_variance_weights(sample.estimate(tr)))
    lo_sample = _realized_vol(
        assets, lambda tr: minimum_variance_weights(sample.estimate(tr), long_only)
    )
    uncon_lw = _realized_vol(assets, lambda tr: min_variance_weights(lw.estimate(tr)))
    lo_lw = _realized_vol(assets, lambda tr: minimum_variance_weights(lw.estimate(tr), long_only))

    print("| Covariance | Unconstrained GMV | Long-only GMV |")
    print("| --- | ---: | ---: |")
    print(f"| sample | {uncon_sample:.1%} | {lo_sample:.1%} |")
    print(f"| ledoit-wolf | {uncon_lw:.1%} | {lo_lw:.1%} |")
    print(
        f"\nUnconstrained, the sample covariance is catastrophic (**{uncon_sample:.1%}** "
        f"realised) — factor stage 2's error-maximiser finding. **Long-only collapses it "
        f"to {lo_sample:.1%}**, essentially matching Ledoit-Wolf ({lo_lw:.1%}); and once "
        f"long-only, sample vs shrinkage barely differ. The constraint did the "
        f"regularising: that is why the backtest's best config is `minvar/sample`.\n"
    )

    # The exact mechanism on one real training window: long-only GMV == unconstrained
    # GMV of the Jagannathan-Ma shrunk covariance Sigma - (mu 1^T + 1 mu^T).
    train = assets[:_TRAIN_WINDOW]
    cov = sample.estimate(train)
    w_lo = minimum_variance_weights(cov, long_only)
    lam = float(w_lo @ cov @ w_lo)
    mu = cov @ w_lo - lam
    cov_tilde = cov - (np.outer(mu, np.ones_like(mu)) + np.outer(np.ones_like(mu), mu))
    recovery_err = float(np.max(np.abs(min_variance_weights(cov_tilde) - w_lo)))
    n_shorted = int(np.sum(min_variance_weights(cov) < 0.0))
    n_bound = int(np.sum(w_lo < 1e-6))

    print("### The exact mechanism\n")
    print(
        f"On the first training window, the unconstrained sample GMV shorts "
        f"**{n_shorted} of {n_assets}** industries; long-only pins **{n_bound}** at the "
        f"zero bound. Forming the Jagannathan-Ma shrunk covariance "
        f"Sigma - (mu 1' + 1 mu') from the KKT multiplier mu (mu > 0 exactly on the "
        f"bound-hitting assets, whose covariances are shrunk), its *unconstrained* GMV "
        f"reproduces the long-only weights to **{recovery_err:.1e}** — the equivalence is "
        f"exact, not analogical. Condition number: sample {condition_number(cov):,.0f}, "
        f"JM-shrunk {condition_number(cov_tilde):,.0f}.\n"
    )


if __name__ == "__main__":
    main()
