#!/usr/bin/env python
"""Autocallable notes — flat vol misprices a short-skew structured product (the headline).

An autocallable pays a coupon while the underlying stays range-bound and puts the holder
short a down-and-in put if it falls through a barrier at maturity. That embedded short put
makes the note **short volatility and short skew**, so pricing it with a single flat
Black--Scholes vol — even one matched to the at-the-money implied vol — *misprices* it: the
flat model ignores the negative skew that makes the put dearer.

Two artifacts on a representative 3-year note (semi-annual observations, 100% autocall
barrier, 4% period coupon, 60% downside barrier):

1. **Flat-vol vs smile mispricing** — the note priced under a flat Black--Scholes vol
   matched to each smile model's ATM implied vol, versus the smile-consistent Heston
   (stochastic-vol) and Merton (jump-diffusion) prices. Flat vol *overprices*; the gap and
   the raised probability of loss are the short-skew signature.
2. **Autocall-probability-by-date** — where the note actually redeems under the realistic
   Heston dynamics, and the residual probability of surviving to (and losing at) maturity.

Everything reuses the package's validated path simulators and transform pricers; the note
payoff is the only new logic. Randomness is a seeded generator, so the numbers are
reproducible. Regenerate with::

    python scripts/autocallable_smile_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.pricing import (
    AutocallableMonteCarloEngine,
    AutocallableNote,
    BlackScholesProcess,
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
NOTE = AutocallableNote(
    maturity=3.0,
    n_observations=6,  # semi-annual observations
    autocall_barrier=1.0,  # 100% of the initial fixing
    coupon=0.04,  # 4% per period, accrued and paid on autocall
    downside_barrier=0.6,  # 60% barrier -> 40% buffer at maturity
)
_N_PATHS = 400_000
_HESTON_SUBSTEPS = 64
_SEED = 20240719

# Heston: steep negative skew (rho = -0.8, high vol-of-vol). Merton: downward jumps.
HESTON = HestonProcess.from_market(MARKET, v0=0.05, kappa=1.5, theta=0.05, xi=0.9, rho=-0.8)
MERTON = MertonProcess.from_market(MARKET, vol=0.18, lam=0.6, mu_j=-0.18, sigma_j=0.12)


def atm_implied_vol(process: object, engine: object) -> float:
    """The model's at-the-money implied vol at the note maturity — the flat-vol a desk would use."""
    atm = EuropeanOption(MARKET.spot, NOTE.maturity, OptionType.CALL)
    price = engine.calculate(atm, process)  # type: ignore[attr-defined]
    return implied_volatility(price, atm, MARKET)


def price_note(process: object, substeps: int = _HESTON_SUBSTEPS) -> object:
    """Monte Carlo valuation of the representative note under one process (fresh seed)."""
    engine = AutocallableMonteCarloEngine(
        _N_PATHS, rng=np.random.default_rng(_SEED), heston_substeps=substeps
    )
    return engine.estimate(NOTE, process)


def main() -> None:
    """Print the mispricing table and the Heston autocall-probability breakdown."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("## Autocallable — flat vol misprices a short-skew structured product\n")
    print(
        f"Representative note: {NOTE.maturity:g}y, {NOTE.n_observations} semi-annual "
        f"observations, {NOTE.autocall_barrier:.0%} autocall barrier, "
        f"{NOTE.coupon:.0%} period coupon, {NOTE.downside_barrier:.0%} downside barrier; "
        f"S={MARKET.spot:g}, r={MARKET.rate:g}. Prices per 1.0 notional, "
        f"{_N_PATHS:,} paths.\n"
    )

    print("### 1. Flat vol vs the smile (each flat model matched to the model's ATM vol)\n")
    print(
        "| Model | ATM vol | Flat-vol price | Smile price | Flat − smile | P(loss) flat → smile |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    results: dict[str, object] = {}
    for name, process, engine in (
        ("Heston (stoch vol)", HESTON, HestonFFTEngine()),
        ("Merton (jumps)", MERTON, MertonFFTEngine()),
    ):
        iv = atm_implied_vol(process, engine)
        flat = price_note(
            BlackScholesProcess(spot=MARKET.spot, rate=MARKET.rate, vol=iv), substeps=1
        )
        smile = price_note(process)
        results[name] = smile
        gap = flat.price - smile.price
        print(
            f"| {name} | {iv:.1%} | {flat.price:.4f} ± {flat.std_error:.4f} "
            f"| {smile.price:.4f} ± {smile.std_error:.4f} | **{gap:+.4f}** "
            f"| {flat.loss_probability:.1%} → {smile.loss_probability:.1%} |"
        )
    print(
        "\n**The headline is Heston.** Matching the flat Black--Scholes vol to Heston's ATM "
        "implied vol isolates the *skew*: the note is short the embedded down-and-in put, so "
        "the steep negative skew makes that put dearer and the flat model **overprices the "
        "note by ~0.85% of notional** while understating the probability of capital loss "
        "(3.8% → 4.9%). That is the money a flat-vol book does not see.\n"
    )
    print(
        "The Merton row is an honest contrast, not a repeat: a *jump* smile is roughly "
        "symmetric in the wings, so matching the flat vol to its (much higher) 24% ATM level "
        "fattens *both* tails. The upper tail then autocalls slightly more often — an effect "
        "that competes with, and here just outweighs, the dearer downside put, leaving a "
        "small opposite-signed gap. Skew, not tail-fatness per se, is what an autocallable is "
        "structurally short — which is exactly why the diffusive-skew (Heston) mispricing is "
        "the clean, material one.\n"
    )
    heston_result = results["Heston (stoch vol)"]

    assert heston_result is not None
    print("### 2. Autocall probability by observation date (Heston)\n")
    print("| Observation | Time (y) | First-autocall probability |")
    print("| ---: | ---: | ---: |")
    for i, (t, p) in enumerate(
        zip(NOTE.observation_times, heston_result.autocall_probabilities, strict=True), start=1
    ):
        print(f"| {i} | {t:.1f} | {p:.1%} |")
    print(
        f"| survive to maturity | {NOTE.maturity:.1f} | {heston_result.maturity_probability:.1%} |"
    )
    total = float(heston_result.autocall_probabilities.sum()) + heston_result.maturity_probability
    print(
        f"\nThe first-autocall probabilities plus the survival probability sum to "
        f"{total:.3f} — every path either redeems once or reaches maturity. Most of the mass "
        f"redeems at the first date (the note is at-the-money at inception); the "
        f"{heston_result.loss_probability:.1%} of paths that both survive and finish below the "
        f"60% barrier are where the capital loss — and the skew sensitivity — lives.\n"
    )


if __name__ == "__main__":
    main()
