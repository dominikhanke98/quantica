#!/usr/bin/env python
"""Generate the derivatives-VaR report for the README — pricing meets risk.

Three option books (a near-linear deep-ITM position, a long-gamma straddle, a
short-gamma straddle) are risk-measured on the *same* seeded scenario set by three
revaluation methods — delta-normal, delta-gamma, and full revaluation through the
pricing engines — so every divergence in the table is approximation error, not
sampling noise. The model-validation question on display: **when is the fast
linear approximation safe, and which way does it break?**

The second table replays one book's *realized* full-revaluation P&L against each
method's static VaR forecast and lets the existing Kupiec backtest judge them —
the risk/backtest layer is reused unchanged on derivatives P&L.

Everything is seeded and deterministic; the README embeds this output verbatim.
Regenerate with::

    python scripts/derivatives_var_report.py
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
    book_var_es,
    kupiec_pof,
)

PROC = BlackScholesProcess(spot=100.0, rate=0.02, div=0.0, vol=0.2)
ENGINE = AnalyticEuropeanEngine()
CALL = EuropeanOption(100.0, 0.5, OptionType.CALL)
PUT = EuropeanOption(100.0, 0.5, OptionType.PUT)
DAILY_VOL = 0.0126  # ~ 20% annualised over one trading day
LEVEL = 0.99
N_SCENARIOS = 20_000
SCENARIO_SEED = 0
BACKTEST_SEED = 42
BACKTEST_DAYS = 750

METHODS = ("delta-normal", "delta-gamma", "full")


def books() -> dict[str, OptionBook]:
    itm = EuropeanOption(60.0, 0.5, OptionType.CALL)
    return {
        "Deep-ITM call (near-linear)": OptionBook(
            positions=(BookPosition(itm, ENGINE, 100.0),), process=PROC
        ),
        "Long ATM straddle (long gamma)": OptionBook(
            positions=(BookPosition(CALL, ENGINE, 100.0), BookPosition(PUT, ENGINE, 100.0)),
            process=PROC,
        ),
        "Short ATM straddle (short gamma)": OptionBook(
            positions=(BookPosition(CALL, ENGINE, -100.0), BookPosition(PUT, ENGINE, -100.0)),
            process=PROC,
        ),
    }


def divergence_table(scenarios: MarketScenarios) -> None:
    print("### 1. Delta-normal vs delta-gamma vs full revaluation (99% one-day VaR)\n")
    print(
        f"Same {N_SCENARIOS:,} seeded scenarios (daily vol {DAILY_VOL:.2%}) for every "
        "method, so divergence is approximation error, not noise:\n"
    )
    print("| Book | Delta-normal | Delta-gamma | Full revaluation | DN error | DG error |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, book in books().items():
        var = {m: book_var_es(book, scenarios, LEVEL, method=m).var for m in METHODS}
        full = var["full"]
        dn_err = (var["delta-normal"] - full) / full
        dg_err = (var["delta-gamma"] - full) / full
        print(
            f"| {name} | {var['delta-normal']:,.2f} | {var['delta-gamma']:,.2f} | "
            f"{full:,.2f} | {dn_err:+.1%} | {dg_err:+.1%} |"
        )
    print(
        "\nThe direction is exactly the omitted ½Γ·δS² term: for the **short-gamma** "
        "book it is a pure loss, so delta-normal *under*-states VaR; for the "
        "**long-gamma** book it cushions losses, so delta-normal *over*-states it; "
        "the near-linear book (Γ ≈ 0) agrees across methods. Delta-gamma repairs "
        "most of the curvature error at negligible cost.\n"
    )


def backtest_table(scenarios: MarketScenarios) -> None:
    book = books()["Short ATM straddle (short gamma)"]
    forecasts = {m: book_var_es(book, scenarios, LEVEL, method=m).var for m in METHODS}

    rng = np.random.default_rng(BACKTEST_SEED)
    realized = MarketScenarios.generate(BACKTEST_DAYS, rng, spot_vol=DAILY_VOL)
    losses = -book.full_revaluation_pnl(realized)
    n = losses.size

    print("### 2. The Kupiec backtest agrees (short-gamma book)\n")
    print(
        f"{BACKTEST_DAYS} days of realized full-revaluation P&L against each method's "
        f"static 99% VaR forecast — the existing backtest layer, reused unchanged:\n"
    )
    print("| VaR forecast | Exceptions | Expected | Kupiec p | Verdict |")
    print("| --- | ---: | ---: | ---: | --- |")
    for m in METHODS:
        x = int(np.sum(losses > forecasts[m]))
        kp = kupiec_pof(x, n, LEVEL)
        verdict = "**rejected**" if kp.reject() else "passes"
        label = "full revaluation" if m == "full" else m
        print(f"| {label} | {x} | {n * (1 - LEVEL):.1f} | {kp.p_value:.3f} | {verdict} |")
    print(
        "\nThe linear approximation's optimistic VaR takes far too many exceptions "
        "and is rejected; the full-revaluation (and delta-gamma) forecasts achieve "
        "the nominal coverage.\n"
    )


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("## Derivatives VaR report — pricing meets risk\n")
    scenarios = MarketScenarios.generate(
        N_SCENARIOS, np.random.default_rng(SCENARIO_SEED), spot_vol=DAILY_VOL
    )
    divergence_table(scenarios)
    backtest_table(scenarios)


if __name__ == "__main__":
    main()
