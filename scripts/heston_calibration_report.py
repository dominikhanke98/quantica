#!/usr/bin/env python
"""Generate the Heston calibration report for the README.

Three reproducible artifacts, printed as GitHub-flavoured Markdown:

1. **Synthetic recovery** (the rigorous, reference-free check) — a vol surface is
   generated from known Heston parameters, calibrated back, and the recovered
   parameters are tabulated against the truth. A noise-free surface is recovered
   to solver tolerance.
2. **Realistic-surface fit** (the compelling demo) — a hand-specified, *non-Heston*
   equity-index smile (downward skew flattening with maturity) is fitted. Heston
   can only approximate it, so the residuals are informative: the fit is good but
   *structured*, exposing the model's known short-maturity skew limitation. One
   maturity slice (the smile) is printed as market-vs-fitted implied vols.
3. **Identifiability** — the calibration objective is profiled along ``kappa`` and
   ``rho``; the relative width of each near-optimal valley shows ``kappa`` is far
   more weakly identified than ``rho``.

Everything is seeded, so the whole report reproduces byte-for-byte. The README
embeds this output verbatim. Regenerate with::

    python scripts/heston_calibration_report.py

Do not hand-edit the numbers in the README — rerun this script instead.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.pricing import (
    EuropeanOption,
    HestonFFTEngine,
    HestonParams,
    Market,
    OptionType,
    calibrate_heston,
    implied_volatility,
    profile_objective,
    vol_surface_from_grid,
)

MARKET = Market(spot=100.0, rate=0.03, div=0.01)
ENGINE = HestonFFTEngine()
_PARAM_NAMES = ("v0", "kappa", "theta", "xi", "rho")


def implied_surface(params: HestonParams, strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
    """The Black--Scholes implied-vol surface a Heston model implies (for synthetic data)."""
    process = params.to_process(MARKET)
    ivs = np.zeros((expiries.size, strikes.size))
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            kind = OptionType.CALL if MARKET.forward(float(T)) <= K else OptionType.PUT
            opt = EuropeanOption(float(K), float(T), kind)
            ivs[i, j] = implied_volatility(ENGINE.calculate(opt, process), opt, MARKET)
    return ivs


def realistic_surface(strikes: np.ndarray, expiries: np.ndarray) -> np.ndarray:
    """A hand-specified, non-Heston equity-index smile: downward skew, mild term structure.

    ``iv = atm(T) + skew(T)·k + curv·k²`` with ``k = ln(K/F)`` — a realistic
    parametric smile that Heston can only approximate, so the fit shows structure.
    """
    ivs = np.zeros((expiries.size, strikes.size))
    for i, T in enumerate(expiries):
        atm = 0.20 + 0.03 * np.sqrt(T)  # term structure of ATM vol
        skew = -0.06 / (0.3 + T)  # skew flattens with maturity
        forward = MARKET.forward(float(T))
        for j, K in enumerate(strikes):
            k = np.log(K / forward)
            ivs[i, j] = atm + skew * k + 0.10 * k * k
    return ivs


def synthetic_recovery() -> None:
    truth = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.3, rho=-0.7)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    expiries = np.array([0.25, 0.5, 1.0, 2.0])
    ivs = implied_surface(truth, strikes, expiries)
    quotes = vol_surface_from_grid(strikes, expiries, ivs)
    result = calibrate_heston(MARKET, quotes, engine=ENGINE)

    print("### 1. Synthetic recovery (reference-free)\n")
    print(
        f"Surface: {strikes.size}×{expiries.size} grid generated from known Heston "
        f"parameters, then calibrated back.\n"
    )
    print("| Parameter | Truth | Recovered | Abs. error |")
    print("| --- | ---: | ---: | ---: |")
    for name, tru, rec in zip(_PARAM_NAMES, truth, result.params, strict=True):
        print(f"| {name} | {tru:.4f} | {rec:.4f} | {abs(rec - tru):.2e} |")
    print(f"\nFit RMSE: {result.rmse_vol * 1e4:.1e} vol basis points (≈ machine zero). ", end="")
    print(f"Feller 2κθ ≥ ξ²: {'satisfied' if result.feller_satisfied else 'VIOLATED'}.\n")


def realistic_fit() -> None:
    strikes = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    expiries = np.array([0.1, 0.25, 0.5, 1.0, 2.0])
    ivs = realistic_surface(strikes, expiries)
    quotes = vol_surface_from_grid(strikes, expiries, ivs)
    result = calibrate_heston(
        MARKET, quotes, n_starts=12, rng=np.random.default_rng(2024), engine=ENGINE
    )

    print("### 2. Realistic-surface fit (non-Heston smile)\n")
    print(
        f"A hand-specified equity-index smile ({strikes.size}×{expiries.size}, downward "
        f"skew flattening with maturity) that Heston can only approximate.\n"
    )
    p = result.params
    print(
        f"Calibrated: v0={p.v0:.4f}, κ={p.kappa:.4f}, θ={p.theta:.4f}, "
        f"ξ={p.xi:.4f}, ρ={p.rho:+.4f}. "
        f"Feller 2κθ ≥ ξ²: {'satisfied' if result.feller_satisfied else 'VIOLATED'}."
    )
    print(
        f"Fit RMSE: {result.rmse_vol * 100:.3f} vol points "
        f"(worst quote {result.max_abs_vol_error * 100:.3f}).\n"
    )

    # Smile slice at T = 0.25 (row index 1): market vs fitted.
    row = 1
    model_ivs = result.model_ivs.reshape(expiries.size, strikes.size)
    print(f"Smile fit at T = {expiries[row]:g} (market vs fitted implied vol):\n")
    print("| Strike | Market IV | Fitted IV | Diff (vol pts) |")
    print("| --- | ---: | ---: | ---: |")
    for j, K in enumerate(strikes):
        mkt = ivs[row, j]
        mdl = model_ivs[row, j]
        print(f"| {K:g} | {mkt * 100:.2f}% | {mdl * 100:.2f}% | {(mdl - mkt) * 100:+.3f} |")
    print()


def identifiability() -> None:
    truth = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.3, rho=-0.7)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    expiries = np.array([0.25, 0.5, 1.0, 2.0])
    ivs = implied_surface(truth, strikes, expiries)
    quotes = vol_surface_from_grid(strikes, expiries, ivs)

    kappa_prof = profile_objective(
        MARKET, quotes, "kappa", np.linspace(0.5, 4.0, 15), anchor=truth, engine=ENGINE
    )
    rho_prof = profile_objective(
        MARKET, quotes, "rho", np.linspace(-0.9, -0.5, 15), anchor=truth, engine=ENGINE
    )

    def valley(prof, scale: float) -> tuple[float, float]:  # type: ignore[no-untyped-def]
        within = prof.values[prof.rmse_vol <= prof.rmse_vol.min() + 5e-4]
        width = float(within.max() - within.min())
        return width, width / abs(scale)

    kw, krel = valley(kappa_prof, truth.kappa)
    rw, rrel = valley(rho_prof, truth.rho)

    print("### 3. Identifiability (objective-valley width)\n")
    print(
        "Width of the near-optimal valley (RMSE within 5 basis points of the minimum) "
        "when each parameter is pinned and the others re-optimised:\n"
    )
    print("| Parameter | Valley width | Relative to value |")
    print("| --- | ---: | ---: |")
    print(f"| κ (mean-reversion) | {kw:.3f} | ±{krel * 50:.1f}% |")
    print(f"| ρ (skew) | {rw:.3f} | ±{rrel * 50:.1f}% |")
    print(
        f"\nκ's relative valley is ≈ {krel / rrel:.1f}× wider than ρ's — the surface "
        f"pins the skew (ρ) far more tightly than the mean-reversion speed (κ).\n"
    )


def main() -> None:
    # The report uses Greek letters and superscripts; rewrap stdout as UTF-8 so it
    # prints on a Windows console (cp1252) and when redirected into the README.
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("## Heston calibration report\n")
    print(f"Market: S={MARKET.spot:g}, r={MARKET.rate:g}, q={MARKET.div:g}.\n")
    synthetic_recovery()
    realistic_fit()
    identifiability()


if __name__ == "__main__":
    main()
