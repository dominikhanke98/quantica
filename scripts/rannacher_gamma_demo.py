#!/usr/bin/env python
"""Rannacher start-up — the gamma-oscillation before/after (PDE-Greeks headline).

Crank--Nicolson is A-stable but **not L-stable**: it damps high-frequency error modes
only weakly, so the vanilla payoff's kink at the strike makes the *gamma* read off a raw
CN grid oscillate near the strike. **Rannacher start-up** — replacing the first couple of
CN steps nearest expiry with fully-implicit backward-Euler half-steps (L-stable) — damps
those modes and restores a smooth gamma at second order.

This script quantifies the fix. It sweeps a European call's gamma across spot near the
strike on a deliberately coarse-in-time grid (where CN rings), with Rannacher off (pure
CN) and on, and measures the oscillation by the **total variation of the gamma error**
against the analytic Black--Scholes gamma. Deterministic (a PDE solve, no RNG), so it
reproduces byte-for-byte; the README embeds a captured run and the printed profile is the
data for the before/after plot.

Regenerate with::

    python scripts/rannacher_gamma_demo.py
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BlackScholesProcess,
    EuropeanOption,
    FiniteDifferenceEngine,
    OptionType,
)

# A grid that is fine in space but coarse in time (25 steps) is exactly the regime that
# exposes the Crank--Nicolson weakness: little natural damping of the payoff-kink modes.
_SPACE_STEPS = 200
_TIME_STEPS = 25
_STRIKE = 100.0
_EXPIRY = 1.0
_RATE = 0.05
_DIV = 0.0
_VOL = 0.20
_SPOTS = np.linspace(90.0, 110.0, 81)


def _gamma_curve(rannacher_steps: int) -> np.ndarray:
    option = EuropeanOption(_STRIKE, _EXPIRY, OptionType.CALL)
    base = BlackScholesProcess(spot=_STRIKE, rate=_RATE, div=_DIV, vol=_VOL)
    engine = FiniteDifferenceEngine(
        space_steps=_SPACE_STEPS, time_steps=_TIME_STEPS, rannacher_steps=rannacher_steps
    )
    return np.array([engine.greeks(option, base.with_spot(float(s))).gamma for s in _SPOTS])


def _analytic_gamma() -> np.ndarray:
    option = EuropeanOption(_STRIKE, _EXPIRY, OptionType.CALL)
    base = BlackScholesProcess(spot=_STRIKE, rate=_RATE, div=_DIV, vol=_VOL)
    engine = AnalyticEuropeanEngine()
    return np.array([engine.greeks(option, base.with_spot(float(s))).gamma for s in _SPOTS])


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    analytic = _analytic_gamma()
    gamma_cn = _gamma_curve(0)  # pure Crank--Nicolson
    gamma_rann = _gamma_curve(2)  # Rannacher start-up

    err_cn = gamma_cn - analytic
    err_rann = gamma_rann - analytic
    tv_cn = float(np.sum(np.abs(np.diff(err_cn))))
    tv_rann = float(np.sum(np.abs(np.diff(err_rann))))
    max_cn = float(np.max(np.abs(err_cn)))
    max_rann = float(np.max(np.abs(err_rann)))

    print("## Rannacher start-up — gamma oscillation before/after\n")
    print(
        f"European call, K={_STRIKE:g}, T={_EXPIRY:g}, r={_RATE:.0%}, q={_DIV:.0%}, "
        f"σ={_VOL:.0%}. Gamma swept across spot on a {_SPACE_STEPS}×{_TIME_STEPS} "
        f"(space×time) grid — fine in space, coarse in time, the regime that exposes the "
        f"Crank–Nicolson payoff-kink oscillation.\n"
    )
    print("| Gamma-error metric (spot 90–110) | Pure CN | Rannacher | Improvement |")
    print("| --- | ---: | ---: | ---: |")
    tv_line = f"| Total variation of gamma error | {tv_cn:.2e} | {tv_rann:.2e} |"
    print(f"{tv_line} {tv_cn / tv_rann:.0f}× smoother |")
    max_line = f"| Max abs gamma error | {max_cn:.2e} | {max_rann:.2e} |"
    print(f"{max_line} {max_cn / max_rann:.0f}× |")
    print(
        "\nPure Crank–Nicolson rings at the strike (it is A-stable but not L-stable, so it "
        "fails to damp the high-frequency modes the payoff kink excites); Rannacher's "
        "backward-Euler half-steps annihilate them, leaving a smooth gamma — a "
        f"**{tv_cn / tv_rann:.0f}× reduction** in the total variation of the error.\n"
    )

    # The gamma profile near the strike — the data for the before/after plot (every
    # 5th point, to keep the embedded table short).
    print("Gamma profile near the strike (every 5th sampled spot):\n")
    print("| Spot | Pure CN | Rannacher | Analytic |")
    print("| ---: | ---: | ---: | ---: |")
    for i in range(0, len(_SPOTS), 5):
        print(f"| {_SPOTS[i]:.1f} | {gamma_cn[i]:.5f} | {gamma_rann[i]:.5f} | {analytic[i]:.5f} |")


if __name__ == "__main__":
    main()
