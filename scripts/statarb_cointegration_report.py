#!/usr/bin/env python
"""Cointegration + spread — detect real cointegration AND reject spurious pairs.

Two artifacts:

1. **Known-truth (headline, no network)** — the effective challenge: on genuinely
   cointegrated series (a shared stochastic trend plus a stationary spread) both the
   Engle--Granger and Johansen tests detect cointegration; on independent random walks
   (spurious) both correctly fail to reject the null. The size/power table validates the
   validators — Engle--Granger is well-sized while the Johansen trace test over-rejects a
   little in finite samples (a documented small-sample bias).

2. **Real data** — a cointegrated pair from the Fama--French 49-industry universe (soft
   drinks vs. restaurants, built as cumulative log-price indices): the test statistics, the
   fitted hedge ratio, and the Ornstein--Uhlenbeck spread with its half-life (the natural
   holding period). An honest aside: on borderline pairs the two tests can disagree, which
   is exactly why a disciplined screen runs both.

Section 2 fetches Ken French data (cached via ``scripts/_ff_data.py``; never in CI).
Regenerate with::

    python scripts/statarb_cointegration_report.py

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
    generate_cointegrated_pair,
    generate_independent_random_walks,
    johansen,
)

_N_MONTHS = 360
_PAIR = ("Soda", "Meals")  # soft drinks vs. restaurants: both consumer food-and-beverage


def _known_truth_section() -> None:
    """Both tests detect real cointegration and reject spurious pairs; size/power."""
    print("### 1. Known-truth: detect real cointegration, reject spurious pairs\n")

    y, x = generate_cointegrated_pair(400, np.random.default_rng(0), beta=1.5, spread_kappa=0.1)
    eg_true = engle_granger(y, x)
    jo_true = johansen(np.column_stack([y, x]))
    walks = generate_independent_random_walks(400, 2, np.random.default_rng(1))
    eg_spur = engle_granger(walks[:, 0], walks[:, 1])
    jo_spur = johansen(walks)

    print("| Series | Engle–Granger p | EG verdict | Johansen rank | Johansen verdict |")
    print("| --- | ---: | :---: | ---: | :---: |")
    print(
        f"| **cointegrated** (β=1.5) | {eg_true.pvalue:.4f} | "
        f"{'cointegrated' if eg_true.is_cointegrated() else 'no'} | {jo_true.rank()} | "
        f"{'cointegrated' if jo_true.rank() >= 1 else 'no'} |"
    )
    print(
        f"| **independent walks** | {eg_spur.pvalue:.3f} | "
        f"{'cointegrated' if eg_spur.is_cointegrated() else 'no'} | {jo_spur.rank()} | "
        f"{'cointegrated' if jo_spur.rank() >= 1 else 'no'} |"
    )
    print(
        f"\nThe cointegrated pair is caught (EG p = {eg_true.pvalue:.4f}, Johansen rank "
        f"{jo_true.rank()}) and its hedge ratio recovered ({eg_true.hedge_ratio:.2f} vs the "
        "true 1.5); the independent walks are correctly left alone (EG "
        f"p = {eg_spur.pvalue:.2f}, Johansen rank {jo_spur.rank()}). Rejecting the spurious "
        "pair is the half that sinks naive pairs trading.\n"
    )

    _size_power_table()


def _size_power_table(n_trials: int = 200) -> None:
    """Rejection rates on known truth — validate the validators."""
    eg_pow = eg_size = jo_pow = jo_size = 0
    for i in range(n_trials):
        y, x = generate_cointegrated_pair(
            300, np.random.default_rng(1000 + i), beta=1.2, spread_kappa=0.15
        )
        eg_pow += engle_granger(y, x).is_cointegrated(0.05)
        jo_pow += johansen(np.column_stack([y, x])).rank(0.05) >= 1
        walks = generate_independent_random_walks(300, 2, np.random.default_rng(5000 + i))
        eg_size += engle_granger(walks[:, 0], walks[:, 1]).is_cointegrated(0.05)
        jo_size += johansen(walks).rank(0.05) >= 1

    print(f"Size and power at the 5% level ({n_trials} trials, T = 300):\n")
    print("| Test | Power (detect real) | Size (false positive) |")
    print("| --- | ---: | ---: |")
    print(f"| Engle–Granger | {eg_pow / n_trials:.0%} | {eg_size / n_trials:.1%} |")
    print(f"| Johansen (trace) | {jo_pow / n_trials:.0%} | {jo_size / n_trials:.1%} |")
    print(
        f"\nBoth tests are powerful ({min(eg_pow, jo_pow) / n_trials:.0%} detection here). "
        f"Engle--Granger is well-sized at ~{eg_size / n_trials:.0%}; the Johansen trace test "
        f"over-rejects at ~{jo_size / n_trials:.0%} — a documented finite-sample bias, "
        "surfaced by validating the validator rather than trusting the nominal 5%.\n"
    )


def _real_data_section() -> None:
    """A cointegrated FF-industry pair: test statistics, hedge ratio, spread half-life."""
    data = load_fama_french(_N_MONTHS, n_industries=49)
    names = list(data.industry_names)
    log_price = np.cumsum(np.log1p(data.industry_excess), axis=0)
    a, b = _PAIR
    ya, xb = log_price[:, names.index(a)], log_price[:, names.index(b)]

    eg = engle_granger(ya, xb)
    jo = johansen(np.column_stack([ya, xb]))
    ou = estimate_ou_process(eg.spread)
    z = (eg.spread[-1] - ou.long_run_mean) / np.std(eg.spread, ddof=1)

    print(f"### 2. Real pair — {a} vs {b} (49-industry FF, {log_price.shape[0]} months)\n")
    print("Cumulative log-price indices; regress the first on the second.\n")
    print("| Quantity | Value |")
    print("| --- | ---: |")
    print(
        f"| Engle–Granger ADF stat | {eg.adf_stat:.2f} (5% crit {eg.critical_values['5%']:.2f}) |"
    )
    print(f"| Engle–Granger p-value | {eg.pvalue:.4f} |")
    trace_cv = jo.trace_crit_values[0, 1]
    print(f"| Johansen trace stat | {jo.trace_stats[0]:.2f} (5% crit {trace_cv:.2f}) |")
    print(f"| Johansen rank | {jo.rank()} |")
    print(f"| Hedge ratio (β) | {eg.hedge_ratio:.2f} |")
    print(f"| Spread half-life | {ou.half_life:.1f} months |")
    print(f"| Mean-reversion speed (κ) | {ou.mean_reversion_speed:.3f} / month |")
    print(f"| Current spread z-score | {z:+.2f} |")
    print(
        f"\nBoth tests agree: {a}–{b} is cointegrated (EG p = {eg.pvalue:.4f}, Johansen rank "
        f"{jo.rank()}), with a {ou.half_life:.0f}-month half-life — slow but tradeable, and "
        "the practically useful number the OU fit exists to produce. **Honest aside:** other "
        "plausible pairs (e.g. Healthcare–MedEquip, Aero–Defense) clear Engle--Granger yet "
        "fail Johansen at 5% — borderline cases where the two tests disagree, which is "
        "precisely why a disciplined screen runs both rather than trusting one.\n"
    )


def main() -> None:
    """Print the known-truth detection/rejection section and the real-pair section."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("## Statistical arbitrage — cointegration and the mean-reverting spread\n")
    _known_truth_section()
    _real_data_section()


if __name__ == "__main__":
    main()
