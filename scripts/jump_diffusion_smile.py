#!/usr/bin/env python
"""Contrast the Merton (jump-diffusion) and Heston (stochastic-vol) smiles.

Both models are calibrated to the *same* baseline diffusion level (15% vol), so the
only difference on show is the *shape* of the implied-vol smile they generate. The
point: **jumps produce a steep short-dated smile that a pure diffusion cannot**.

At a short maturity a downward jump can move the spot a long way in little time, so
out-of-the-money puts (and, symmetrically, far OTM calls) are worth far more than
Black--Scholes says — a pronounced smile/skew. A stochastic-volatility diffusion
(Heston) needs *time* for the variance to spread the terminal distribution, so its
short-dated smile is comparatively flat and only bites at longer maturities. That
is exactly why Merton's smile is very steep at ``T = 0.1`` and flattens quickly,
while Heston's is milder and more persistent — the short-dated smile is the
fingerprint of jumps, and together the two models are the "why Black--Scholes
fails" story from two different mechanisms.

Reproduce with::

    python scripts/jump_diffusion_smile.py

The numbers are deterministic (Fourier pricing, no RNG); the README embeds this
output verbatim.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.pricing import (
    EuropeanOption,
    HestonFFTEngine,
    HestonProcess,
    Market,
    MertonFFTEngine,
    MertonProcess,
    OptionType,
    implied_volatility,
)

MARKET = Market(spot=100.0, rate=0.03, div=0.0)
_BASE_VOL = 0.15  # shared diffusion level so only the smile *shape* differs

# Merton: moderate jump activity with a downward bias (mu_j < 0).
MERTON = MertonProcess.from_market(MARKET, vol=_BASE_VOL, lam=1.0, mu_j=-0.12, sigma_j=0.15)
# Heston: same baseline variance (v0 = theta = 0.15^2), correlated stochastic vol.
HESTON = HestonProcess.from_market(
    MARKET, v0=_BASE_VOL**2, kappa=1.5, theta=_BASE_VOL**2, xi=0.4, rho=-0.6
)

STRIKES = np.array([85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0])
MATURITIES = (0.1, 1.0)


def smile(process: object, engine: object, expiry: float) -> np.ndarray:
    """Black--Scholes implied vols across the strikes for one model and maturity."""
    ivs = np.zeros(STRIKES.size)
    for j, K in enumerate(STRIKES):
        kind = OptionType.CALL if MARKET.forward(expiry) <= K else OptionType.PUT
        option = EuropeanOption(float(K), expiry, kind)
        price = engine.calculate(option, process)  # type: ignore[attr-defined]
        ivs[j] = implied_volatility(price, option, MARKET)
    return ivs


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    merton_fft = MertonFFTEngine()
    heston_fft = HestonFFTEngine()
    smiles = {("Merton", T): smile(MERTON, merton_fft, T) for T in MATURITIES} | {
        ("Heston", T): smile(HESTON, heston_fft, T) for T in MATURITIES
    }

    print("## Merton vs Heston — the short-dated smile is the fingerprint of jumps\n")
    print(
        f"Both models share the same baseline diffusion vol ({_BASE_VOL:.0%}); "
        f"S={MARKET.spot:g}, r={MARKET.rate:g}, q={MARKET.div:g}. "
        "Implied vol (%) by strike:\n"
    )
    header = "| Strike | Merton T=0.1 | Heston T=0.1 | Merton T=1.0 | Heston T=1.0 |"
    print(header)
    print("| ---: | ---: | ---: | ---: | ---: |")
    for j, K in enumerate(STRIKES):
        cells = [
            f"{smiles[('Merton', 0.1)][j] * 100:.2f}",
            f"{smiles[('Heston', 0.1)][j] * 100:.2f}",
            f"{smiles[('Merton', 1.0)][j] * 100:.2f}",
            f"{smiles[('Heston', 1.0)][j] * 100:.2f}",
        ]
        print(f"| {K:g} | {' | '.join(cells)} |")

    print("\nSmile steepness (widest minus narrowest implied vol across strikes):\n")
    print("| Model | T=0.1 | T=1.0 | short/long ratio |")
    print("| --- | ---: | ---: | ---: |")
    for model in ("Merton", "Heston"):
        short = np.ptp(smiles[(model, 0.1)]) * 100
        long = np.ptp(smiles[(model, 1.0)]) * 100
        print(f"| {model} | {short:.1f} pts | {long:.1f} pts | {short / long:.1f}× |")
    print(
        "\nMerton's smile is far steeper at the short maturity and flattens fast; "
        "Heston's is milder and persists — jumps dominate short-dated option prices, "
        "diffusion dominates long-dated ones."
    )


if __name__ == "__main__":
    main()
