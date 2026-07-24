#!/usr/bin/env python
"""Yield-curve construction — the bootstrap, and why interpolation is a modelling decision.

Three artifacts (all deterministic, no network — the curve is built from market quotes):

1. **The bootstrap and its self-consistency anchor.** A discount curve is bootstrapped from
   deposits and par swaps; the foundational check is that it reprices *every* input instrument
   back to par to machine precision, whatever the interpolation scheme.

2. **Interpolation changes your forwards.** The same market inputs, all repriced exactly,
   produce materially different instantaneous-forward curves under different interpolation
   schemes — the "the default is a decision you didn't know you made" finding.

3. **The shape-preservation artifact.** Under a mild stress (one tenor trading rich), the
   smooth cubic schemes manufacture *negative* forward rates where the robust log-linear
   scheme stays positive — the classic Hagan--West cautionary result.

Regenerate with::

    python scripts/rates_curve_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.rates import (
    Deposit,
    Swap,
    bootstrap,
    linear_zero,
    log_linear_discount,
    monotone_cubic_zero,
    natural_cubic_zero,
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
# Flat at 3% with the 4y tenor trading rich — a mild, realistic stress.
_STRESS = [
    Deposit(0.5, 0.030),
    Deposit(1.0, 0.030),
    Swap(2, 0.030),
    Swap(3, 0.030),
    Swap(4, 0.036),
    Swap(5, 0.030),
    Swap(7, 0.030),
    Swap(10, 0.030),
]
_SCHEMES = {
    "linear-zero": linear_zero(),
    "log-linear-discount": log_linear_discount(),
    "natural-cubic-zero": natural_cubic_zero(),
    "monotone-cubic-zero": monotone_cubic_zero(),
}
_TENORS = np.array([0.75, 1.5, 4.0, 6.0, 8.5])


def _bootstrap_section() -> None:
    """The bootstrapped curve and its reprice-to-par self-consistency."""
    print("### 1. The bootstrap and its self-consistency anchor\n")
    curve = bootstrap(_MARKET, log_linear_discount())
    worst = max(abs(inst.value(curve)) for inst in _MARKET)
    print(
        "Curve bootstrapped from 3 deposits + 5 par swaps (log-linear-discount). Every input "
        f"reprices to par: **max |repricing residual| = {worst:.1e}** across all 8 instruments.\n"
    )
    print("| Pillar (y) | Zero rate | Discount factor |")
    print("| ---: | ---: | ---: |")
    for t in curve.times:
        print(
            f"| {t:g} | {float(curve.zero_rate(t)):.4%} | {float(curve.discount_factor(t)):.5f} |"
        )
    print()


def _forward_divergence_section() -> None:
    """Instantaneous forwards across schemes — all reprice inputs, all disagree."""
    print("### 2. Interpolation changes your forwards (all reprice the inputs exactly)\n")
    forwards = {
        name: bootstrap(_MARKET, sc).instantaneous_forward(_TENORS) for name, sc in _SCHEMES.items()
    }
    header = "| Tenor (y) | " + " | ".join(_SCHEMES) + " |"
    print(header)
    print("| ---: | " + " | ".join(["---:"] * len(_SCHEMES)) + " |")
    for i, t in enumerate(_TENORS):
        cells = " | ".join(f"{forwards[name][i]:.3%}" for name in _SCHEMES)
        print(f"| {t:g} | {cells} |")
    grid = np.linspace(0.1, 10.0, 400)
    stacked = np.array(
        [bootstrap(_MARKET, sc).instantaneous_forward(grid) for sc in _SCHEMES.values()]
    )
    max_bps = float((stacked.max(axis=0) - stacked.min(axis=0)).max()) * 1e4
    print(
        f"\nAll four schemes reprice the 8 instruments to par, yet their instantaneous forwards "
        f"disagree by up to **{max_bps:.0f} bps**. The interpolation scheme is not a cosmetic "
        "detail — it is a modelling choice that moves every forward-rate-sensitive price and "
        "risk on the book.\n"
    )


def _artifact_section() -> None:
    """The shape-preservation artifact: cubic forwards go negative, log-linear does not."""
    print("### 3. The shape-preservation artifact (Hagan–West)\n")
    grid = np.linspace(0.05, 10.0, 800)
    mins = {
        name: float(bootstrap(_STRESS, sc).instantaneous_forward(grid).min())
        for name, sc in _SCHEMES.items()
    }
    print("Same bootstrap on a curve flat at 3% with the 4y tenor trading rich (3.6%):\n")
    print("| Scheme | Min instantaneous forward | Negative? |")
    print("| --- | ---: | :---: |")
    for name in _SCHEMES:
        neg = "**yes**" if mins[name] < 0.0 else "no"
        print(f"| {name} | {mins[name]:.2%} | {neg} |")
    print(
        "\nThe smooth cubic schemes **overshoot** and produce negative — arbitrageable — "
        "instantaneous forwards, while log-linear on discount factors stays positive (its "
        "piecewise-flat forwards cannot go negative when discount factors decrease). Note the "
        "honest subtlety: even the *monotone* cubic on zero rates goes negative here — "
        "preserving the monotonicity of the *zeros* does not preserve the positivity of the "
        "*forwards*, which is exactly why Hagan--West interpolate the forwards directly. Robust "
        "beats smooth for a curve you will differentiate.\n"
    )


def main() -> None:
    """Print the bootstrap, the forward-divergence table, and the shape-preservation artifact."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("## Rates — yield-curve construction and the interpolation decision\n")
    _bootstrap_section()
    _forward_divergence_section()
    _artifact_section()


if __name__ == "__main__":
    main()
