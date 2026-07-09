#!/usr/bin/env python
"""Generate the European-option convergence table for the README.

Prices one canonical European call three ways — the analytic Black--Scholes
engine, the CRR binomial engine at a range of step counts, and the Monte Carlo
engine (naive and with variance reduction) — and prints a GitHub-flavoured
Markdown table of prices and absolute errors against the analytic reference.

The Monte Carlo rows use an explicit, seeded ``numpy.random.Generator`` (never
the global RNG), so the whole table reproduces byte-for-byte on every run; the
README embeds this output verbatim. Regenerate with::

    python scripts/convergence_table.py

Do not hand-edit the numbers in the README — rerun this script instead.
"""

from __future__ import annotations

import numpy as np
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BinomialEngine,
    BlackScholesProcess,
    EuropeanOption,
    MonteCarloEngine,
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
MC_PATHS = 1_000_000
MC_SEED = 20240709

# (label, antithetic, control_variate)
MC_CONFIGS = (
    ("Monte Carlo (naive)", False, False),
    ("Monte Carlo (antithetic)", True, False),
    ("Monte Carlo (control variate)", False, True),
)


def main() -> None:
    process = BlackScholesProcess(spot=SPOT, rate=RATE, div=DIV, vol=VOL)
    option = EuropeanOption(strike=STRIKE, expiry=EXPIRY, option_type=OptionType.CALL)

    analytic = AnalyticEuropeanEngine().calculate(option, process)

    # (method, price, abs-error string, note)
    rows: list[tuple[str, float, str, str]] = [
        ("Black–Scholes (analytic)", analytic, "—", "reference"),
    ]
    for n in STEP_COUNTS:
        price = BinomialEngine(steps=n).calculate(option, process)
        rows.append((f"Binomial CRR (N={n})", price, f"{abs(price - analytic):.2e}", "O(1/N)"))

    naive_se: float | None = None
    for label, antithetic, control in MC_CONFIGS:
        engine = MonteCarloEngine(
            MC_PATHS,
            rng=np.random.default_rng(MC_SEED),
            antithetic=antithetic,
            control_variate=control,
        )
        result = engine.estimate(option, process)
        if naive_se is None:
            naive_se = result.std_error
        vrf = (naive_se / result.std_error) ** 2
        note = f"SE {result.std_error:.1e}, VRF {vrf:.1f}×"
        rows.append((label, result.price, f"{abs(result.price - analytic):.2e}", note))

    print(
        f"European call — S={SPOT:g}, K={STRIKE:g}, r={RATE:g}, "
        f"q={DIV:g}, sigma={VOL:g}, T={EXPIRY:g}"
    )
    print(f"(Monte Carlo: {MC_PATHS:,} paths, seed {MC_SEED})\n")
    print("| Method | Price | Abs. error vs analytic | Note |")
    print("| --- | ---: | ---: | --- |")
    for name, price, err, note in rows:
        print(f"| {name} | {price:.8f} | {err} | {note} |")


if __name__ == "__main__":
    main()
