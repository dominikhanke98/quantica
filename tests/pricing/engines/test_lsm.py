"""Validation of the Longstaff--Schwartz engine (numerical-validation skill §5).

American LSM has no closed form, so it is validated against the American tree and
PDE prices already in place (themselves QuantLib-benchmarked), under Monte Carlo
discipline:

1. **Agreement** — LSM sits within a few standard errors of the tree/PDE price.
2. **Low-bias / lower-bound signature** — realized-cashflow LSM under a
   regression-estimated (sub-optimal) exercise policy is biased *low*: averaged
   over seeds it lands **at or just below** the reference. This is the expected
   correctness signature, not a failure.
3. **No-dividend call** — LSM on a no-dividend American call recovers the
   European price (early exercise never optimal), within statistical error.
4. **Standard-error discipline & determinism** (§5) — SE ~ 1/sqrt(n), seeded
   reproducibility.
5. **Sensitivity study** — a richer basis improves the policy and shrinks the
   downward bias.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.pricing import (
    AmericanOption,
    AnalyticEuropeanEngine,
    BinomialEngine,
    BlackScholesProcess,
    EuropeanOption,
    FiniteDifferenceEngine,
    LongstaffSchwartzEngine,
    MonteCarloEngine,
    OptionType,
)

# American put with dividends (a genuine early-exercise problem).
MARKET = BlackScholesProcess(spot=100.0, rate=0.05, div=0.04, vol=0.25)
N_PATHS = 100_000
EXERCISE_DATES = 50
BASIS_DEGREE = 3


def _lsm(seed: int, *, n: int = N_PATHS, degree: int = BASIS_DEGREE) -> LongstaffSchwartzEngine:
    return LongstaffSchwartzEngine(
        n,
        rng=np.random.default_rng(seed),
        exercise_dates=EXERCISE_DATES,
        basis_degree=degree,
        antithetic=True,
    )


_REFERENCE_CACHE: dict[float, float] = {}


def _reference(option: AmericanOption) -> float:
    """A near-exact American reference from the (independent) tree and PDE engines.

    Cached by strike (all references here use ``MARKET``) so the expensive American
    PDE solve runs once per contract across the test module.
    """
    if option.strike not in _REFERENCE_CACHE:
        tree = BinomialEngine(steps=4000).calculate(option, MARKET)
        pde = FiniteDifferenceEngine(space_steps=300, time_steps=300).calculate(option, MARKET)
        assert abs(tree - pde) < 3e-3  # the two references agree to their discretisation
        _REFERENCE_CACHE[option.strike] = 0.5 * (tree + pde)
    return _REFERENCE_CACHE[option.strike]


# --------------------------------------------------------------------------- #
# 1. Agreement with the tree/PDE reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strike, seed", [(90.0, 1), (100.0, 2), (110.0, 3)])
def test_lsm_agrees_with_reference_within_3_se(strike: float, seed: int) -> None:
    option = AmericanOption(strike=strike, expiry=1.0, option_type=OptionType.PUT)
    reference = _reference(option)
    result = _lsm(seed).estimate(option, MARKET)
    assert result.std_error > 0.0
    assert abs(result.price - reference) <= 3.0 * result.std_error


# --------------------------------------------------------------------------- #
# 2. Low-bias / lower-bound signature
# --------------------------------------------------------------------------- #


def test_lsm_is_a_low_biased_lower_bound() -> None:
    # Averaged over seeds, LSM lands at or just below the reference: a
    # regression-estimated (sub-optimal) exercise policy can only leave value on
    # the table. Below-reference is the expected signature, not a failure.
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    reference = _reference(option)
    estimates = np.array([_lsm(seed).calculate(option, MARKET) for seed in range(10)])
    mean_lsm = float(estimates.mean())
    assert mean_lsm <= reference  # lower bound (measured ~5e-3 below)
    assert mean_lsm > reference - 0.05  # ...but close, not broken


# --------------------------------------------------------------------------- #
# 3. No-dividend American call recovers the European price
# --------------------------------------------------------------------------- #


def test_no_dividend_american_call_recovers_european() -> None:
    market = BlackScholesProcess(spot=100.0, rate=0.05, div=0.0, vol=0.2)
    american = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    european = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    reference = AnalyticEuropeanEngine().calculate(european, market)
    result = _lsm(7).estimate(american, market)
    assert abs(result.price - reference) <= 3.0 * result.std_error


# --------------------------------------------------------------------------- #
# 4. Standard-error discipline and determinism
# --------------------------------------------------------------------------- #


def test_standard_error_scales_as_inverse_sqrt_n() -> None:
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    se_n = _lsm(3, n=25_000).estimate(option, MARKET).std_error
    se_4n = _lsm(3, n=100_000).estimate(option, MARKET).std_error
    assert 1.7 < se_n / se_4n < 2.3


def test_same_seed_same_result() -> None:
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    assert _lsm(11, n=10_000).estimate(option, MARKET) == _lsm(11, n=10_000).estimate(
        option, MARKET
    )


def test_different_seed_different_result() -> None:
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    assert _lsm(1, n=10_000).calculate(option, MARKET) != _lsm(2, n=10_000).calculate(
        option, MARKET
    )


# --------------------------------------------------------------------------- #
# 5. Sensitivity: a richer basis reduces the downward bias
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1])
def test_richer_basis_recovers_more_value(seed: int) -> None:
    # A degree-3 basis fits the continuation surface better than degree-1, so it
    # takes fewer sub-optimal exercise decisions and recovers more value (moves
    # up toward the reference from below).
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    reference = _reference(option)
    linear = _lsm(seed, degree=1).calculate(option, MARKET)
    cubic = _lsm(seed, degree=3).calculate(option, MARKET)
    assert cubic >= linear
    assert cubic <= reference + 3e-2  # still a lower bound (within noise)


# --------------------------------------------------------------------------- #
# Wiring / rejections
# --------------------------------------------------------------------------- #


def test_rejects_european_option() -> None:
    european = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    with pytest.raises(ValueError, match="American exercise only"):
        _lsm(0, n=1000).estimate(european, MARKET)


def test_invalid_parameters_raise() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="n_paths must be at least 2"):
        LongstaffSchwartzEngine(1, rng=rng)
    with pytest.raises(ValueError, match="exercise_dates must be at least 1"):
        LongstaffSchwartzEngine(1000, rng=rng, exercise_dates=0)
    with pytest.raises(ValueError, match="basis_degree must be at least 1"):
        LongstaffSchwartzEngine(1000, rng=rng, basis_degree=0)


def test_prices_through_option_npv() -> None:
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    engine = _lsm(5, n=20_000)
    reference = _reference(option)
    option.set_engine(engine)
    # npv delegates to calculate; check it lands near the reference (loose band).
    assert abs(option.npv(MARKET) - reference) < 0.2


def test_monte_carlo_engine_still_rejects_american() -> None:
    # The terminal-only engine is unchanged: it still declines American options.
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    engine = MonteCarloEngine(1000, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="European exercise only"):
        engine.estimate(option, MARKET)


def test_without_antithetic_variates() -> None:
    # Exercise the plain (non-antithetic) path-generation and sampling branches.
    option = AmericanOption(strike=100.0, expiry=1.0, option_type=OptionType.PUT)
    reference = _reference(option)
    engine = LongstaffSchwartzEngine(
        100_000, rng=np.random.default_rng(4), exercise_dates=EXERCISE_DATES, basis_degree=3
    )
    result = engine.estimate(option, MARKET)
    assert result.std_error > 0.0
    assert abs(result.price - reference) <= 3.0 * result.std_error


def test_deep_out_of_money_put_with_sparse_exercise() -> None:
    # A deep-OTM put with few paths and a rich basis leaves early exercise dates
    # with too few in-the-money paths to regress; the engine skips the fit there
    # and still returns a sane, small, non-negative price without erroring.
    option = AmericanOption(strike=60.0, expiry=1.0, option_type=OptionType.PUT)
    engine = LongstaffSchwartzEngine(
        2_000, rng=np.random.default_rng(0), exercise_dates=50, basis_degree=5
    )
    result = engine.estimate(option, MARKET)
    assert 0.0 <= result.price < 1.0
