"""Validation of the Monte Carlo engine (numerical-validation skill §5).

Monte Carlo results are random variables, so they are tested statistically, not
against a fixed absolute tolerance:

* the estimate is within ~3 standard errors of the analytic price;
* variance reduction genuinely works — antithetic and control-variate SEs are
  materially below the naive SE at equal path count;
* the standard error scales as ~1/sqrt(n);
* the same seed reproduces the same result exactly.

All randomness comes from an injected, seeded ``numpy.random.Generator``; no test
touches the global ``numpy.random`` state.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BlackScholesProcess,
    EuropeanOption,
    MonteCarloEngine,
    OptionType,
)
from quantica.pricing.engines import GreeksEngine, PricingEngine

ANALYTIC = AnalyticEuropeanEngine()
PROC = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.2)
SEED = 12345
N = 200_000


def _option(kind: OptionType, strike: float = 100.0) -> EuropeanOption:
    return EuropeanOption(strike=strike, expiry=1.0, option_type=kind)


def _engine(
    *, antithetic: bool = False, control_variate: bool = False, n: int = N, seed: int = SEED
) -> MonteCarloEngine:
    return MonteCarloEngine(
        n,
        rng=np.random.default_rng(seed),
        antithetic=antithetic,
        control_variate=control_variate,
    )


# --------------------------------------------------------------------------- #
# Within 3 standard errors of the analytic price
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"antithetic": True},
        {"control_variate": True},
        {"antithetic": True, "control_variate": True},
    ],
    ids=["naive", "antithetic", "control", "anti+control"],
)
def test_estimate_within_three_standard_errors(kind: OptionType, config: dict[str, bool]) -> None:
    option = _option(kind)
    reference = ANALYTIC.calculate(option, PROC)
    result = _engine(**config).estimate(option, PROC)
    assert result.std_error > 0.0
    assert abs(result.price - reference) <= 3.0 * result.std_error


# --------------------------------------------------------------------------- #
# Variance reduction
# --------------------------------------------------------------------------- #


def test_antithetic_reduces_variance() -> None:
    option = _option(OptionType.CALL)
    se_naive = _engine().estimate(option, PROC).std_error
    se_anti = _engine(antithetic=True).estimate(option, PROC).std_error
    # Antithetic materially below naive (measured ratio ~0.75, VRF ~1.8x).
    assert se_anti < 0.85 * se_naive


def test_control_variate_reduces_variance() -> None:
    option = _option(OptionType.CALL)
    se_naive = _engine().estimate(option, PROC).std_error
    se_cv = _engine(control_variate=True).estimate(option, PROC).std_error
    # The discounted-spot control is highly correlated with a call payoff:
    # SE well under half the naive SE (measured ratio ~0.41, VRF ~6x).
    assert se_cv < 0.5 * se_naive
    variance_reduction_factor = (se_naive / se_cv) ** 2
    assert variance_reduction_factor > 4.0


def test_control_variate_degenerate_in_zero_vol_limit() -> None:
    # With sigma=0 the control (discounted spot) has zero variance and carries
    # no information: the engine falls back to the raw payoff, which is the
    # exact discounted intrinsic with zero standard error.
    proc0 = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.0)
    option = _option(OptionType.CALL)
    result = _engine(control_variate=True, seed=5).estimate(option, proc0)
    expected = max(100.0 * math.exp(-0.02) - 100.0 * math.exp(-0.05), 0.0)
    assert result.price == pytest.approx(expected, abs=1e-12)
    # SE is zero up to floating-point noise (all paths are identical).
    assert result.std_error < 1e-12


def test_standard_error_scales_as_inverse_sqrt_n() -> None:
    option = _option(OptionType.CALL)
    se_n = _engine(n=50_000, seed=1).estimate(option, PROC).std_error
    se_4n = _engine(n=200_000, seed=1).estimate(option, PROC).std_error
    # Quadrupling paths should roughly halve the SE.
    assert 1.7 < se_n / se_4n < 2.3


# --------------------------------------------------------------------------- #
# Determinism / reproducibility
# --------------------------------------------------------------------------- #


def test_same_seed_same_result() -> None:
    option = _option(OptionType.CALL)
    a = _engine(antithetic=True, control_variate=True, seed=7).estimate(option, PROC)
    b = _engine(antithetic=True, control_variate=True, seed=7).estimate(option, PROC)
    assert a == b


def test_different_seed_different_result() -> None:
    option = _option(OptionType.CALL)
    a = _engine(seed=1).estimate(option, PROC).price
    b = _engine(seed=2).estimate(option, PROC).price
    assert a != b


def test_calculate_matches_estimate_price() -> None:
    option = _option(OptionType.CALL)
    price = _engine(seed=99).calculate(option, PROC)
    estimate = _engine(seed=99).estimate(option, PROC).price
    assert price == estimate


# --------------------------------------------------------------------------- #
# Engine wiring
# --------------------------------------------------------------------------- #


def test_satisfies_pricing_protocol_only() -> None:
    engine = _engine()
    assert isinstance(engine, PricingEngine)
    assert not isinstance(engine, GreeksEngine)


def test_invalid_n_paths_raises() -> None:
    with pytest.raises(ValueError, match="n_paths must be at least 2"):
        MonteCarloEngine(1, rng=np.random.default_rng(0))


def test_prices_through_option_npv() -> None:
    option = _option(OptionType.CALL)
    reference = ANALYTIC.calculate(option, PROC)
    engine = _engine(control_variate=True, seed=3)
    option.set_engine(engine)
    # A single seeded draw: recompute the matching SE to bound the check.
    se = _engine(control_variate=True, seed=3).estimate(option, PROC).std_error
    assert abs(option.npv(PROC) - reference) <= 3.0 * se
