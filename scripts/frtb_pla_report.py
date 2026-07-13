#!/usr/bin/env python
"""Generate the FRTB P&L-attribution report for the README.

Three desks are run through the PLA eligibility test over a 250-day window on the
*same* seeded market moves. Each desk pairs an option book with a risk model
(the RTPL method); the test compares the risk model's sensitivities P&L (RTPL)
against the full-revaluation hypothetical P&L (HPL) and assigns a green / amber /
red zone with its IMA consequence.

The point the table makes: PLA is the regulatory formalisation of the
"do the risk factors span the P&L?" question — a short-gamma desk whose risk
model carries only delta fails, while the same desk with a delta-gamma model
passes. This reuses the two-directional gamma divergence from the
derivatives-risk step, now as a pass/fail capital test.

Everything is seeded and deterministic; the README embeds this output verbatim.
Regenerate with::

    python scripts/frtb_pla_report.py
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BlackScholesProcess,
    EuropeanOption,
    OptionType,
)
from quantica.risk import (
    BookPosition,
    MarketScenarios,
    OptionBook,
    book_pla_test,
)
from quantica.risk.frtb import (
    KS_AMBER_THRESHOLD,
    KS_RED_THRESHOLD,
    SPEARMAN_AMBER_THRESHOLD,
    SPEARMAN_RED_THRESHOLD,
)

PROC = BlackScholesProcess(spot=100.0, rate=0.02, div=0.0, vol=0.2)
ENGINE = AnalyticEuropeanEngine()
CALL = EuropeanOption(100.0, 0.5, OptionType.CALL)
PUT = EuropeanOption(100.0, 0.5, OptionType.PUT)
N_DAYS = 250
DAILY_VOL = 0.05  # a stressed window, so curvature matters
SEED = 1


def straddle(quantity: float) -> OptionBook:
    return OptionBook(
        positions=(BookPosition(CALL, ENGINE, quantity), BookPosition(PUT, ENGINE, quantity)),
        process=PROC,
    )


def deep_itm_book() -> OptionBook:
    return OptionBook(
        positions=(BookPosition(EuropeanOption(60.0, 0.5, OptionType.CALL), ENGINE, 100.0),),
        process=PROC,
    )


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    scenarios = MarketScenarios.generate(N_DAYS, np.random.default_rng(SEED), spot_vol=DAILY_VOL)

    print("## FRTB P&L-attribution report — internal-models eligibility\n")
    print(
        f"{N_DAYS} daily moves (daily vol {DAILY_VOL:.0%}, a stressed window), same "
        f"scenarios for every desk. RTPL = the risk model's sensitivities P&L; "
        f"HPL = full revaluation (the pricing path). Thresholds are Basel FRTB "
        f"(BCBS d457, MAR33): Spearman green ≥ {SPEARMAN_AMBER_THRESHOLD:.2f} / "
        f"red < {SPEARMAN_RED_THRESHOLD:.2f}; KS green ≤ {KS_AMBER_THRESHOLD:.2f} / "
        f"red > {KS_RED_THRESHOLD:.2f}.\n"
    )

    desks = [
        ("Deep-ITM desk (near-linear)", deep_itm_book(), "delta-normal"),
        ("Short-straddle desk, delta+gamma model", straddle(-100.0), "delta-gamma"),
        ("Short-straddle desk, delta-only model", straddle(-100.0), "delta-normal"),
    ]

    print("| Desk (book + risk model) | Spearman | KS | Zone | Capital consequence |")
    print("| --- | ---: | ---: | --- | --- |")
    for name, book, method in desks:
        r = book_pla_test(book, scenarios, rtpl_method=method)  # type: ignore[arg-type]
        zone = str(r.zone).upper()
        print(
            f"| {name} | {r.spearman:.3f} ({r.spearman_zone}) "
            f"| {r.ks_statistic:.3f} ({r.ks_zone}) | **{zone}** "
            f"| {r.capital_consequence()} |"
        )

    print(
        "\nThe short-straddle desk fails PLA **only when its risk model omits gamma**: "
        "the delta-only model cannot reproduce the curvature in the true P&L, so RTPL "
        "and HPL diverge on both the ranking (Spearman) and the distribution (KS), and "
        "the desk drops out of the Internal Models Approach. Adding the gamma factor "
        "makes the risk model span the P&L again and the identical book passes green. "
        "PLA is exactly the 'do the risk factors span the P&L?' question the "
        "derivatives-risk integration explored, now as a capital-eligibility test.\n"
    )


if __name__ == "__main__":
    main()
