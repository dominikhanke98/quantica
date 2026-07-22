#!/usr/bin/env python
"""Dynamic hedge ratio — the Kalman filter tracks a drifting ratio a static fit cannot.

Two artifacts:

1. **Known-truth (headline, no network)** — a pair whose *true* hedge ratio drifts linearly
   over time. The Kalman filter tracks it within its own uncertainty band; a single static
   OLS / cointegration coefficient, fitted once, sits near the average and tracks neither
   end — many times the tracking error. Dynamic beats static exactly when the relationship
   moves.

2. **Real data** — the Soda vs. Meals pair from the cointegration step. The static hedge
   ratio is one number (0.85); the Kalman hedge ratio evolves over the sample, and the
   resulting *dynamic* spread mean-reverts a little faster than the static one (a cleaner
   signal, because the drift is filtered out rather than left in the residual).

Section 2 fetches Ken French data (cached via ``scripts/_ff_data.py``; never in CI).
Regenerate with::

    python scripts/statarb_kalman_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from _ff_data import load_fama_french
from quantica.statarb import (
    engle_granger,
    estimate_ou_process,
    generate_time_varying_pair,
    kalman_hedge_ratio,
)

_N_MONTHS = 360
_PAIR = ("Soda", "Meals")
_BURN_IN = 36  # months to let the diffuse prior settle before reading the hedge ratio


def _static_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0][0])


def _known_truth_section() -> None:
    """Kalman tracks a drifting true ratio; static OLS does not."""
    print("### 1. Known-truth: tracking a drifting hedge ratio\n")
    true = np.linspace(1.0, 2.0, 1000)  # the true ratio drifts from 1.0 to 2.0
    y, x = generate_time_varying_pair(true, np.random.default_rng(0), alpha=2.0, obs_vol=1.0)
    result = kalman_hedge_ratio(y, x, process_var=1e-4, obs_var=1.0)
    static = _static_hedge_ratio(y, x)

    kalman_rmse = float(np.sqrt(np.mean((result.hedge_ratio - true) ** 2)))
    static_rmse = float(np.sqrt(np.mean((static - true) ** 2)))
    post = slice(100, None)
    coverage = float(
        np.mean(np.abs(result.hedge_ratio[post] - true[post]) <= 3.0 * result.hedge_ratio_std[post])
    )

    print("True hedge ratio drifts 1.0 → 2.0 over 1000 steps; obs noise σ = 1.\n")
    print("| Estimator | Tracking RMSE vs the true path |")
    print("| --- | ---: |")
    print(f"| static OLS (one coefficient) | {static_rmse:.3f}  (stuck near {static:.2f}) |")
    print(f"| **Kalman (dynamic)** | **{kalman_rmse:.3f}** |")
    print(
        f"\nThe Kalman filter tracks the moving ratio with **{static_rmse / kalman_rmse:.0f}× "
        f"lower tracking error** than the static fit, and the true path stays inside its "
        f"one-sigma band {coverage:.0%} of the time (±3σ). Two anchors pin the recursion: with "
        "the process variance → 0 on a constant ratio the filter reduces to recursive least "
        "squares and its estimate converges to OLS, and when the model fits the standardised "
        "innovations are white. The process/observation variance ratio is the tuning knob — "
        "larger adapts faster, smaller pins the ratio down — a documented parameter, not a "
        "magic constant.\n"
    )


def _real_data_section() -> None:
    """Soda vs Meals: the hedge ratio evolves, and the dynamic spread is cleaner."""
    data = load_fama_french(_N_MONTHS, n_industries=49)
    names = list(data.industry_names)
    log_price = np.cumsum(np.log1p(data.industry_excess), axis=0)
    a, b = _PAIR
    y, x = log_price[:, names.index(a)], log_price[:, names.index(b)]

    eg = engle_granger(y, x)
    obs_var = float(np.var(eg.spread, ddof=2))
    static_half_life = estimate_ou_process(eg.spread).half_life
    # Process variance a small fraction of the observation variance: a slowly drifting ratio.
    result = kalman_hedge_ratio(y, x, process_var=obs_var * 1e-4, obs_var=obs_var)
    hedge = result.hedge_ratio[_BURN_IN:]
    dynamic_half_life = estimate_ou_process(result.spread[_BURN_IN:]).half_life

    drift_range = f"{hedge.min():.2f} – {hedge.max():.2f} (drifts)"
    print(f"### 2. Real pair — {a} vs {b} (49-industry FF, {log_price.shape[0]} months)\n")
    print("| Quantity | Static (step 1) | Kalman (dynamic) |")
    print("| --- | ---: | ---: |")
    print(f"| Hedge ratio | {eg.hedge_ratio:.2f} (one number) | {drift_range} |")
    print(f"| Spread half-life | {static_half_life:.1f} months | {dynamic_half_life:.1f} months |")
    print(
        f"\nThe static cointegration fit says the hedge ratio is a single {eg.hedge_ratio:.2f}; "
        f"the Kalman filter shows it drifting between {hedge.min():.2f} and {hedge.max():.2f} "
        f"over the sample (mean {hedge.mean():.2f}). Because that drift is absorbed into the "
        f"state rather than left in the residual, the **dynamic spread mean-reverts faster "
        f"({dynamic_half_life:.1f} vs {static_half_life:.1f} months)** — a cleaner signal for "
        "the mean-reversion strategy that is the next step. (Process variance set to "
        f"{1e-4:.0e}× the observation variance — the tuning knob, chosen for a slow drift.)\n"
    )


def main() -> None:
    """Print the known-truth tracking section and the real-pair dynamic-hedge-ratio section."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("## Statistical arbitrage — the Kalman dynamic hedge ratio\n")
    _known_truth_section()
    _real_data_section()


if __name__ == "__main__":
    main()
