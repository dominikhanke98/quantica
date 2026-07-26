#!/usr/bin/env python
"""Interest-rate products — swap par rate, Hull--White vs Black-76, and vol identification.

Three artifacts (deterministic; the curve is bootstrapped from market quotes, no network):

1. **Swap par rate and the tie-back.** The par swap rate is read off the curve, and a swap
   struck at it values to **zero** — the products layer is self-consistent with the step-1
   bootstrap it prices on.

2. **Hull--White vs Black-76 (the headline consistency check).** The same caplet and swaption
   are priced two ways: the market-standard **Black-76** at a flat vol, and **analytically under
   Hull--White** (caplet as a put on a bond; swaption by Jamshidian). Each Hull--White price is
   cross-checked against an exact-transition **Monte Carlo** estimate (the model-independent
   oracle) and expressed as a Black **implied volatility**, so the two routes reconcile.

3. **Volatility identification (closing the step-2 finding).** Hull--White reprices the curve
   for *any* ``sigma`` — the curve carries no volatility information. Caps and swaptions do: a
   least-squares fit to option prices recovers a known ``sigma`` tightly, which is what pins the
   parameter the curve alone could not.

Regenerate with::

    python scripts/rates_products_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.rates import (
    Cap,
    Caplet,
    Deposit,
    HullWhite,
    Swap,
    Swaption,
    bootstrap,
    calibrate_hull_white_volatility,
    hull_white_caplet,
    hull_white_price,
    hull_white_price_mc,
    hull_white_swaption,
    monotone_cubic_zero,
    par_swap_rate,
    swap_value,
)
from scipy.optimize import brentq

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

_MEAN_REVERSION = 0.1
_SIGMA = 0.015


def _swap_section(curve) -> None:  # type: ignore[no-untyped-def]
    """Par swap rates off the curve, and the reprices-to-par tie-back."""
    print("### 1. Swaps — par rate and the reprices-to-par tie-back\n")
    print("| Maturity | Par swap rate | Value @ par | Value @ par+50bp |")
    print("| --- | ---: | ---: | ---: |")
    for maturity in (2.0, 5.0, 10.0):
        par = par_swap_rate(curve, maturity, frequency=1)
        at_par = swap_value(curve, maturity, par, frequency=1)
        off = swap_value(curve, maturity, par + 0.005, frequency=1)
        print(f"| {maturity:g}y | {par * 100:.3f}% | {at_par:+.1e} | {off:+.5f} |")
    print(
        "\nEach par rate is the fair fixed rate off the curve; struck at it, the swap values to "
        "**~0** (to machine precision) — the products layer is self-consistent with the bootstrap. "
        "The 5y and 10y par rates reproduce the 4.200% / 4.500% swap quotes the curve was built "
        "from. Paying 50bp over par turns the swap negative for the payer, as it must.\n"
    )


def _black_implied_vol(price: float, pricer, low: float = 1e-6, high: float = 5.0) -> float:  # type: ignore[no-untyped-def]
    """Back out the flat Black vol that reproduces ``price`` under ``pricer(vol)``."""
    return float(brentq(lambda v: pricer(v) - price, low, high, xtol=1e-14))


def _hw_vs_black_section(curve) -> None:  # type: ignore[no-untyped-def]
    """Caplet and swaption priced under Black-76 and Hull--White, cross-checked by MC."""
    model = HullWhite.from_curve(curve, a=_MEAN_REVERSION, sigma=_SIGMA)
    caplet = Caplet(1.0, 1.5, 0.04)
    swaption = Swaption(1.0, 4.0, 0.042, frequency=1, payer=True)

    hw_caplet = hull_white_caplet(model, caplet)
    hw_swaption = hull_white_swaption(model, swaption)
    mc_caplet, se_caplet = hull_white_price_mc(
        model, caplet, n_paths=200_000, n_steps=120, rng=np.random.default_rng(0)
    )
    mc_swaption, se_swaption = hull_white_price_mc(
        model, swaption, n_paths=200_000, n_steps=120, rng=np.random.default_rng(1)
    )
    iv_caplet = _black_implied_vol(hw_caplet, lambda v: caplet.black_price(curve, v))
    iv_swaption = _black_implied_vol(hw_swaption, lambda v: swaption.black_price(curve, v))

    hw_z_caplet = (mc_caplet - hw_caplet) / se_caplet
    hw_z_swaption = (mc_swaption - hw_swaption) / se_swaption
    print(f"### 2. Hull--White vs Black-76 (a={_MEAN_REVERSION:g}, sigma={_SIGMA:g})\n")
    print("| Instrument | Hull–White | Black-76 @ IV | Implied vol | Monte Carlo | (MC−HW)/SE |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    print(
        f"| Caplet 1y×0.5y K=4% | {hw_caplet:.6f} | {caplet.black_price(curve, iv_caplet):.6f} | "
        f"{iv_caplet * 100:.2f}% | {mc_caplet:.6f} ± {se_caplet:.6f} | {hw_z_caplet:+.2f} |"
    )
    print(
        f"| Payer swaption 1y→5y K=4.2% | {hw_swaption:.6f} | "
        f"{swaption.black_price(curve, iv_swaption):.6f} | {iv_swaption * 100:.2f}% | "
        f"{mc_swaption:.6f} ± {se_swaption:.6f} | {hw_z_swaption:+.2f} |"
    )
    print(
        "\nThe analytic Hull--White price, the Black-76 price at its own implied vol, and the "
        "independent Monte Carlo estimate all agree (MC within ~1 SE) — three routes to the same "
        "number. The Hull--White short-rate volatility ``sigma`` translates into a sensible flat "
        "Black **implied volatility** for each instrument, the language the market quotes in.\n"
    )


def _vol_identification_section(curve) -> None:  # type: ignore[no-untyped-def]
    """The curve gives no sigma info; option prices recover a known sigma tightly."""
    sigma_true = 0.012
    truth = HullWhite.from_curve(curve, a=_MEAN_REVERSION, sigma=sigma_true)
    caps = [Cap(0.5, m, 0.04, frequency=2) for m in (2.0, 3.0, 5.0)]
    quotes = [(cap, hull_white_price(truth, cap)) for cap in caps]
    fit = calibrate_hull_white_volatility(curve, quotes, a=_MEAN_REVERSION)

    low = HullWhite.from_curve(curve, a=_MEAN_REVERSION, sigma=0.005)
    high = HullWhite.from_curve(curve, a=_MEAN_REVERSION, sigma=0.025)
    pillars = curve.times
    curve_gap = float(np.max(np.abs(low.discount_bond(pillars) - high.discount_bond(pillars))))
    probe = Cap(0.5, 5.0, 0.04, frequency=2)

    print("### 3. Volatility identification — what the curve could not tell us\n")
    print(f"Calibrating ``sigma`` (a={_MEAN_REVERSION:g} fixed) to three cap prices:\n")
    print(
        f"- true sigma = **{sigma_true:.4f}**, recovered sigma = **{fit.sigma:.4f}** "
        f"(|error| = {abs(fit.sigma - sigma_true):.1e}), fit RMSE = {fit.rmse:.1e}\n"
    )
    print("Why the curve alone could not do this — two very different sigmas:\n")
    print("| sigma | Max curve reprice gap | 5y cap price |")
    print("| ---: | ---: | ---: |")
    cap_low = hull_white_price(low, probe)
    cap_high = hull_white_price(high, probe)
    print(f"| 0.005 | — | {cap_low:.6f} |")
    print(f"| 0.025 | — | {cap_high:.6f} |")
    print(f"| both | **{curve_gap:.1e}** | (differ **{cap_high / cap_low:.1f}×**) |")
    print(
        "\nHull--White reprices the curve identically (gap ~1e-16) for ``sigma`` = 0.005 or 0.025 "
        "— the curve carries **no** volatility information, exactly the step-2 finding. The cap "
        "price, by contrast, moves several-fold with ``sigma``, so calibrating to option prices "
        "**pins ``sigma`` down** — the vol surface identifies what the curve could not.\n"
    )


def main() -> None:
    """Print the swap-par, Hull--White-vs-Black-76, and volatility-identification artifacts."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    curve = bootstrap(_MARKET, monotone_cubic_zero())
    print("## Rates — interest-rate products: swaps, Hull--White vs Black-76, vol identification\n")
    _swap_section(curve)
    _hw_vs_black_section(curve)
    _vol_identification_section(curve)


if __name__ == "__main__":
    main()
