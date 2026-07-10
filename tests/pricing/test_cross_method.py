"""Four-way cross-method convergence — the effective-challenge centrepiece (skill §3).

Prices the *same* European option under all four independently-implemented
engines and asserts they agree, each anchored to the analytic closed form to a
tolerance appropriate to *its* error mechanism. Anchoring every numerical engine
to the analytic reference establishes mutual agreement transitively: if CRR and
PDE each sit within their discretisation bound of analytic and MC within its
statistical band, the four prices agree to the sum of those bounds.

The four methods rest on genuinely independent foundations — a closed-form
integral, a binomial lattice, a finite-difference PDE solve, and Monte Carlo
simulation — so their agreement is real cross-validation, not a shared bug.

Tolerances (explicit and justified per method, not one blanket number)
----------------------------------------------------------------------
* **Analytic** — the reference; exact to machine precision.
* **CRR binomial**, ``N = 2000`` steps — first-order ``O(1/N)`` discretisation.
  Measured error across this grid of contracts is <= ~1.3e-3; bounded at
  ``2e-3``.
* **Crank--Nicolson PDE**, ``500 x 500`` grid — second-order ``O(h^2)``.
  Measured error <= ~7.2e-4; bounded at ``1.5e-3``.
* **Monte Carlo**, ``500_000`` paths, antithetic + control variate, seeded — a
  random estimate, so the tolerance is statistical, ``|estimate - analytic| <=
  3 * standard_error`` (~99.7% band), *not* a fixed absolute number. Seeds are
  fixed for reproducibility.

Monte Carlo calibration and the 3-SE band
-----------------------------------------
``test_monte_carlo_estimator_is_calibrated`` re-runs the estimator over a fixed
sequence of seeds (0-19) on one representative contract and checks each estimate
against the 3-SE band. This verifies the *estimator's calibration across the
sampling distribution* — that the reported standard error is an honest gauge of
the estimator's dispersion — rather than merely that it happens to be correct on
a single draw. A biased estimator or an under-reported SE would push estimates
outside the band systematically across the seeds.

The band width follows from a false-failure-rate budget. For a correctly
calibrated estimator the studentised error is ~N(0, 1), so it lands outside
3 SE with probability ``2 * (1 - Phi(3)) ~= 0.27%``. Across the seeded Monte
Carlo assertions in this module — 20 calibration seeds plus the 18 contracts in
``test_four_methods_agree`` (38 total) — the expected number of spurious
failures is ``38 * 0.0027 ~= 0.10``, comfortably below 1. So with fixed seeds
the suite is deterministic *and* a correctly calibrated estimator is
overwhelmingly likely to pass, while a genuine mis-calibration would show up as
a systematic breach rather than a one-off.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BinomialEngine,
    BlackScholesProcess,
    EuropeanOption,
    FiniteDifferenceEngine,
    MonteCarloEngine,
    OptionType,
)

# Shared market with a non-zero dividend yield, so the carry terms (spot leg vs
# strike leg) are genuinely exercised in every engine.
MARKET = BlackScholesProcess(spot=100.0, rate=0.05, div=0.03, vol=0.2)

ANALYTIC = AnalyticEuropeanEngine()

# Grid resolutions and their justified tolerances (see module docstring).
CRR_STEPS = 2000
CRR_TOL = 2e-3
PDE_STEPS = 500
PDE_TOL = 1.5e-3
MC_PATHS = 500_000
MC_SIGMAS = 3.0
# Fixed seed sequence for the Monte Carlo calibration check (deterministic CI).
MC_CALIBRATION_SEEDS = tuple(range(20))

# Contract grid: three strikes (ITM / ATM / OTM) x three maturities x call/put.
_STRIKES = (80.0, 100.0, 120.0)
_EXPIRIES = (0.5, 1.0, 2.0)
_KINDS = (OptionType.CALL, OptionType.PUT)
_GRID = [(K, T, kind) for K in _STRIKES for T in _EXPIRIES for kind in _KINDS]
# Deterministic per-case Monte Carlo seed.
CASES = [(K, T, kind, 1000 + i) for i, (K, T, kind) in enumerate(_GRID)]


def _case_id(case: tuple[float, float, OptionType, int]) -> str:
    strike, expiry, kind, _ = case
    return f"K{strike:g}-T{expiry:g}-{kind}"


@pytest.mark.parametrize("strike, expiry, kind, seed", CASES, ids=[_case_id(c) for c in CASES])
def test_four_methods_agree(strike: float, expiry: float, kind: OptionType, seed: int) -> None:
    option = EuropeanOption(strike=strike, expiry=expiry, option_type=kind)

    # Analytic closed form: the reference all others are challenged against.
    reference = ANALYTIC.calculate(option, MARKET)

    # CRR binomial lattice — deterministic, O(1/N).
    crr = BinomialEngine(steps=CRR_STEPS).calculate(option, MARKET)
    assert abs(crr - reference) < CRR_TOL, f"CRR off by {abs(crr - reference):.2e}"

    # Crank--Nicolson PDE — deterministic, O(h^2).
    pde = FiniteDifferenceEngine(space_steps=PDE_STEPS, time_steps=PDE_STEPS).calculate(
        option, MARKET
    )
    assert abs(pde - reference) < PDE_TOL, f"PDE off by {abs(pde - reference):.2e}"

    # Monte Carlo — statistical, judged against its own standard error.
    mc = MonteCarloEngine(
        MC_PATHS,
        rng=np.random.default_rng(seed),
        antithetic=True,
        control_variate=True,
    ).estimate(option, MARKET)
    assert abs(mc.price - reference) <= MC_SIGMAS * mc.std_error, (
        f"MC off by {abs(mc.price - reference) / mc.std_error:.2f} SE"
    )


def test_all_four_mutually_agree_on_a_representative_contract() -> None:
    # A single explicit demonstration that the four prices land together, within
    # the loosest applicable tolerance (the MC 3-SE band dominates here).
    option = EuropeanOption(strike=105.0, expiry=1.0, option_type=OptionType.CALL)
    reference = ANALYTIC.calculate(option, MARKET)
    crr = BinomialEngine(steps=CRR_STEPS).calculate(option, MARKET)
    pde = FiniteDifferenceEngine(space_steps=PDE_STEPS, time_steps=PDE_STEPS).calculate(
        option, MARKET
    )
    mc = MonteCarloEngine(
        MC_PATHS, rng=np.random.default_rng(0), antithetic=True, control_variate=True
    ).estimate(option, MARKET)

    prices = {"analytic": reference, "crr": crr, "pde": pde, "mc": mc.price}
    spread = max(prices.values()) - min(prices.values())
    # All four within a band set by the coarsest error present (3 MC standard errors).
    assert spread <= 3.0 * mc.std_error + CRR_TOL + PDE_TOL


def test_monte_carlo_estimator_is_calibrated() -> None:
    # Re-run the estimator over a fixed sequence of seeds and check each estimate
    # against the 3-SE band. This probes calibration across the sampling
    # distribution (is the reported SE an honest gauge of dispersion?), not just
    # correctness on one lucky draw. See the module docstring for the
    # false-failure-rate budget that motivates the 3-SE choice.
    option = EuropeanOption(strike=105.0, expiry=1.0, option_type=OptionType.CALL)
    reference = ANALYTIC.calculate(option, MARKET)
    for seed in MC_CALIBRATION_SEEDS:
        mc = MonteCarloEngine(
            MC_PATHS,
            rng=np.random.default_rng(seed),
            antithetic=True,
            control_variate=True,
        ).estimate(option, MARKET)
        sigmas = abs(mc.price - reference) / mc.std_error
        assert sigmas <= MC_SIGMAS, f"seed {seed}: estimate {sigmas:.2f} SE from analytic"
