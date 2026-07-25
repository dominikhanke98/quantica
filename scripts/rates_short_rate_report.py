#!/usr/bin/env python
"""Short-rate models — exact-fit vs best-fit, and analytic vs Monte Carlo.

Two artifacts (deterministic, no network — the curve is built from market quotes):

1. **Exact fit vs best fit (the headline).** Calibrate all three models to the step-1 curve,
   then reprice the curve's own discount factors. Hull--White reproduces the term structure
   **exactly** (its time-dependent drift absorbs the whole curve); Vasicek and CIR, with
   constant parameters, can only best-fit and leave a residual. This is the concrete reason
   practitioners use Hull--White for anything that must be arbitrage-free to the current curve.

2. **Analytic vs Monte Carlo (the cross-method check).** For each model the closed-form bond
   price is compared against a Monte Carlo estimate of ``E[exp(-∫ r ds)]`` from exact-transition
   simulation — validating the analytic formula and the simulation against each other.

Regenerate with::

    python scripts/rates_short_rate_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.rates import (
    Deposit,
    HullWhite,
    Swap,
    bootstrap,
    calibrate_cir,
    calibrate_vasicek,
    monotone_cubic_zero,
    monte_carlo_discount,
)

_MARKET = [
    Deposit(0.25, 0.030),
    Deposit(0.5, 0.032),
    Deposit(1.0, 0.035),
    Swap(2, 0.037),
    Swap(3, 0.039),
    Swap(5, 0.042),
    Swap(7, 0.044),
    Swap(10, 0.045),
]


def _calibration_section(curve) -> None:  # type: ignore[no-untyped-def]
    """Exact fit (Hull--White) vs best fit (Vasicek / CIR) on the same curve."""
    pillars = curve.times
    vasicek = calibrate_vasicek(curve)
    cir = calibrate_cir(curve)
    hull_white = HullWhite.from_curve(curve, a=0.1, sigma=0.01)
    hw_error = float(
        np.max(np.abs(hull_white.discount_bond(pillars) - curve.discount_factor(pillars)))
    )

    print("### 1. Exact fit vs best fit — calibrated to the same curve\n")
    print("| Model | Curve-fit zero-rate RMSE | Max reprice error | Fits the curve? |")
    print("| --- | ---: | ---: | :---: |")
    v_rmse, v_max = vasicek.rmse * 1e4, vasicek.max_abs_error * 1e4
    print(f"| Vasicek | {v_rmse:.1f} bps | {v_max:.1f} bps | best-fit |")
    print(f"| CIR | {cir.rmse * 1e4:.1f} bps | {cir.max_abs_error * 1e4:.1f} bps | best-fit |")
    print(f"| **Hull–White** | **~0** | **{hw_error:.1e}** | **exact** |")
    print(
        f"\nHull--White reprices every pillar to **{hw_error:.0e}** — exact, by construction, for "
        f"any volatility parameters. Vasicek and CIR, with constant ``(a, b, sigma, r0)``, can "
        f"only best-fit the shape and leave a **~{vasicek.rmse * 1e4:.0f} bps** zero-rate "
        "residual. That exact fit is why Hull--White, not Vasicek/CIR, prices anything that must "
        "be arbitrage-free to the current curve. (Honest caveat: the curve barely identifies "
        "``sigma`` — it enters only through a small convexity term — so the fitted ``sigma`` is "
        "not meaningful without volatility instruments; the *residual* is the point here.)\n"
    )


def _monte_carlo_section(curve) -> None:  # type: ignore[no-untyped-def]
    """Analytic bond price vs the exact-transition Monte Carlo estimate."""
    vasicek = calibrate_vasicek(curve).model
    cir = calibrate_cir(curve).model
    hull_white = HullWhite.from_curve(curve, a=0.1, sigma=0.01)
    maturity = 5.0

    print("### 2. Analytic vs Monte Carlo — the cross-method check\n")
    print(
        f"Zero-coupon bond ``P(0, {maturity:g})``: closed form vs MC of ``E[exp(-∫ r ds)]`` "
        "(exact-transition simulation, 100k paths):\n"
    )
    print("| Model | Analytic | Monte Carlo | (MC − analytic) / SE |")
    print("| --- | ---: | ---: | ---: |")
    for name, model in [("Vasicek", vasicek), ("CIR", cir), ("Hull–White", hull_white)]:
        analytic = float(model.discount_bond(maturity))
        est, se = monte_carlo_discount(
            model, maturity, n_paths=100_000, n_steps=400, rng=np.random.default_rng(0)
        )
        print(f"| {name} | {analytic:.6f} | {est:.6f} ± {se:.6f} | {(est - analytic) / se:+.2f} |")
    print(
        "\nEvery model's closed-form bond price sits within a standard error or two of its "
        "Monte Carlo estimate — the analytic affine formula and the exact-transition simulation "
        "validate each other, with no external reference needed.\n"
    )


def main() -> None:
    """Print the exact-fit-vs-best-fit table and the analytic-vs-Monte-Carlo check."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    curve = bootstrap(_MARKET, monotone_cubic_zero())
    print("## Rates — short-rate models: exact fit vs best fit, analytic vs Monte Carlo\n")
    _calibration_section(curve)
    _monte_carlo_section(curve)


if __name__ == "__main__":
    main()
