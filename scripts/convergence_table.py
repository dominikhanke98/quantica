#!/usr/bin/env python
"""Generate the European-option convergence table for the README.

Prices one canonical European call with the analytic Black--Scholes engine and
the CRR binomial engine at a range of step counts, and prints a GitHub-flavoured
Markdown table of prices and absolute errors against the analytic reference.

The computation is fully deterministic (no Monte Carlo / RNG), so it reproduces
byte-for-byte on every run; the README embeds this output verbatim. Regenerate
with::

    python scripts/convergence_table.py

Do not hand-edit the numbers in the README — rerun this script instead.
"""

from __future__ import annotations

from quantica.pricing import (
    AnalyticEuropeanEngine,
    BinomialEngine,
    BlackScholesProcess,
    EuropeanOption,
    OptionType,
)

# Canonical textbook contract: at-the-money call, the classic BS(100,100,5%,0,20%,1y).
SPOT = 100.0
RATE = 0.05
DIV = 0.0
VOL = 0.20
STRIKE = 100.0
EXPIRY = 1.0
STEP_COUNTS = (10, 50, 100, 500, 1000, 5000)


def main() -> None:
    process = BlackScholesProcess(spot=SPOT, rate=RATE, div=DIV, vol=VOL)
    option = EuropeanOption(strike=STRIKE, expiry=EXPIRY, option_type=OptionType.CALL)

    analytic = AnalyticEuropeanEngine().calculate(option, process)

    rows: list[tuple[str, float, str]] = [
        ("Black–Scholes (analytic)", analytic, "— (reference)"),
    ]
    for n in STEP_COUNTS:
        price = BinomialEngine(steps=n).calculate(option, process)
        rows.append((f"Binomial CRR (N={n})", price, f"{abs(price - analytic):.2e}"))

    print(
        f"European call — S={SPOT:g}, K={STRIKE:g}, r={RATE:g}, "
        f"q={DIV:g}, sigma={VOL:g}, T={EXPIRY:g}\n"
    )
    print("| Method | Price | Abs. error vs analytic |")
    print("| --- | ---: | ---: |")
    for name, price, err in rows:
        print(f"| {name} | {price:.8f} | {err} |")


if __name__ == "__main__":
    main()
