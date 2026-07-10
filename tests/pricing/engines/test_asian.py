"""Validation of the Asian (average-price) engine (numerical-validation skill §5).

- The **geometric** Asian has a closed form (:func:`geometric_asian_price`), the
  analytic anchor: Monte Carlo over the same monitoring dates recovers it within
  a few standard errors.
- The **arithmetic** Asian has no closed form, so it is priced by Monte Carlo
  with the geometric Asian as a **control variate** — motivated by the near-perfect
  correlation between the two averages of a shared path. The variance-reduction
  factor is large and reported; the arithmetic price sits sensibly relative to
  the geometric one (arithmetic mean ≥ geometric mean).
- Structural sanity, standard-error discipline, and seeded determinism (§5).
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    AsianMonteCarloEngine,
    AsianOption,
    AveragingType,
    BlackScholesProcess,
    EuropeanOption,
    OptionType,
    geometric_asian_price,
)

MARKET = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.25)
N_DATES = 12
N_PATHS = 200_000


def _asian(strike: float, kind: OptionType, averaging: AveragingType) -> AsianOption:
    return AsianOption(
        strike=strike, expiry=1.0, option_type=kind, averaging=averaging, n_averaging_dates=N_DATES
    )


def _engine(seed: int, *, antithetic: bool = True, control_variate: bool = False, n: int = N_PATHS):
    return AsianMonteCarloEngine(
        n, rng=np.random.default_rng(seed), antithetic=antithetic, control_variate=control_variate
    )


# --------------------------------------------------------------------------- #
# Geometric Asian: Monte Carlo recovers the closed form
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "strike, kind", [(90.0, OptionType.CALL), (100.0, OptionType.CALL), (110.0, OptionType.PUT)]
)
def test_geometric_mc_recovers_closed_form(strike: float, kind: OptionType) -> None:
    option = _asian(strike, kind, AveragingType.GEOMETRIC)
    closed_form = geometric_asian_price(
        MARKET.spot, strike, MARKET.rate, MARKET.div, MARKET.vol, option.expiry, N_DATES, kind
    )
    result = _engine(1).estimate(option, MARKET)
    assert result.std_error > 0.0
    assert abs(result.price - closed_form) <= 3.0 * result.std_error


# --------------------------------------------------------------------------- #
# Arithmetic Asian: the geometric control variate (the highlight)
# --------------------------------------------------------------------------- #


def test_geometric_control_variate_slashes_variance() -> None:
    option = _asian(100.0, OptionType.CALL, AveragingType.ARITHMETIC)
    se_naive = _engine(2, antithetic=False).estimate(option, MARKET).std_error
    se_cv = _engine(2, antithetic=False, control_variate=True).estimate(option, MARKET).std_error
    variance_reduction_factor = (se_naive / se_cv) ** 2
    # Arithmetic and geometric averages of a shared path are almost perfectly
    # correlated: the VRF runs into the hundreds (measured ~880x).
    assert variance_reduction_factor > 50.0


def test_control_variate_and_naive_agree() -> None:
    # Both estimate the same arithmetic price; they must agree within their
    # combined standard error (the control variate only reduces variance, not
    # bias, since E[control] is the exact geometric price).
    option = _asian(100.0, OptionType.CALL, AveragingType.ARITHMETIC)
    naive = _engine(3, antithetic=False).estimate(option, MARKET)
    cv = _engine(4, antithetic=False, control_variate=True).estimate(option, MARKET)
    combined_se = np.hypot(naive.std_error, cv.std_error)
    assert abs(naive.price - cv.price) <= 3.0 * combined_se


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_arithmetic_vs_geometric_ordering(kind: OptionType) -> None:
    # The arithmetic mean dominates the geometric mean (AM >= GM), so an
    # arithmetic call >= geometric call and an arithmetic put <= geometric put.
    arithmetic = _engine(5, control_variate=True).calculate(
        _asian(100.0, kind, AveragingType.ARITHMETIC), MARKET
    )
    geometric = geometric_asian_price(
        MARKET.spot, 100.0, MARKET.rate, MARKET.div, MARKET.vol, 1.0, N_DATES, kind
    )
    if kind is OptionType.CALL:
        assert arithmetic >= geometric
    else:
        assert arithmetic <= geometric


def test_asian_cheaper_than_vanilla() -> None:
    # Averaging reduces the effective volatility, so an Asian call is worth less
    # than the otherwise-identical European call.
    asian = _engine(6, control_variate=True).calculate(
        _asian(100.0, OptionType.CALL, AveragingType.ARITHMETIC), MARKET
    )
    european = AnalyticEuropeanEngine().calculate(
        EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL), MARKET
    )
    assert asian < european


# --------------------------------------------------------------------------- #
# Discipline / wiring
# --------------------------------------------------------------------------- #


def test_standard_error_scales_as_inverse_sqrt_n() -> None:
    option = _asian(100.0, OptionType.CALL, AveragingType.ARITHMETIC)
    se_n = _engine(7, n=25_000).estimate(option, MARKET).std_error
    se_4n = _engine(7, n=100_000).estimate(option, MARKET).std_error
    assert 1.7 < se_n / se_4n < 2.3


def test_same_seed_same_result() -> None:
    option = _asian(100.0, OptionType.CALL, AveragingType.ARITHMETIC)
    assert _engine(8, n=10_000).estimate(option, MARKET) == _engine(8, n=10_000).estimate(
        option, MARKET
    )


def test_rejects_non_asian_option() -> None:
    european = EuropeanOption(strike=100.0, expiry=1.0, option_type=OptionType.CALL)
    with pytest.raises(TypeError, match="AsianOption"):
        _engine(0, n=1000).estimate(european, MARKET)  # type: ignore[arg-type]


def test_geometric_closed_form_zero_vol_limit() -> None:
    # sigma -> 0: the geometric average is deterministic; discounted intrinsic on
    # the geometric-forward.
    price = geometric_asian_price(100.0, 90.0, 0.05, 0.0, 0.0, 1.0, 12, OptionType.CALL)
    assert price > 0.0  # deep in the money, deterministic, positive


def test_asian_instrument_validation() -> None:
    with pytest.raises(ValueError, match="n_averaging_dates must be at least 1"):
        AsianOption(
            strike=100.0,
            expiry=1.0,
            option_type=OptionType.CALL,
            averaging=AveragingType.ARITHMETIC,
            n_averaging_dates=0,
        )


def test_asian_is_european_exercise() -> None:
    from quantica.core.types import ExerciseStyle

    assert (
        _asian(100.0, OptionType.CALL, AveragingType.ARITHMETIC).exercise is ExerciseStyle.EUROPEAN
    )


def test_engine_rejects_too_few_paths() -> None:
    with pytest.raises(ValueError, match="n_paths must be at least 2"):
        AsianMonteCarloEngine(1, rng=np.random.default_rng(0))
