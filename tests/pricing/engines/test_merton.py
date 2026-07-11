r"""Validation of the Merton jump-diffusion engines (numerical-validation skill).

- **Two methods against each other (the headline).** The closed-form Poisson sum
  and the CF/FFT engine price the same option; they agree to tight tolerance across
  strikes, maturities and call/put. This is a self-anchored cross-validation that
  needs no external reference — analogous to Heston's Black--Scholes-limit anchor.
- **Black--Scholes limit.** With ``lam = 0`` (no jumps) Merton collapses to
  Black--Scholes; both engines match ``AnalyticEuropeanEngine``.
- **Series-truncation convergence.** The Poisson sum converges (partial-sum error
  falls monotonically) and the truncation error stays below the stated tolerance.
- **Characteristic function** correct at the known points (``t = 0`` → ``S0^{iu}``;
  ``u = 0`` → 1).
- **Structural sanity** — put--call parity, arbitrage-free monotonicity, and the
  ``alpha``/grid stability of the FFT.
- **The jump smile** — downward jumps (``mu_j < 0``) produce a negatively-skewed
  implied-vol smile, i.e. the "why Black--Scholes fails" signature.

QuantLib exposes ``Merton76Process`` but not a wrapped jump-diffusion *engine* in
this build, so — per the validation protocol — the closed-form-vs-FFT agreement is
the rigorous check in place of a QuantLib benchmark.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from quantica.pricing import (
    AmericanOption,
    AnalyticEuropeanEngine,
    BlackScholesProcess,
    EuropeanOption,
    Market,
    MertonClosedFormEngine,
    MertonFFTEngine,
    MertonProcess,
    OptionType,
    implied_volatility,
    merton_characteristic_function,
    merton_jump_price,
)

MARKET = Market(spot=100.0, rate=0.05, div=0.02)
# A jumpy process with downward-skewed jumps (mu_j < 0) — the interesting case.
PROC = MertonProcess.from_market(MARKET, vol=0.2, lam=0.75, mu_j=-0.1, sigma_j=0.15)
CLOSED = MertonClosedFormEngine()
FFT = MertonFFTEngine()

_CF_PARAMS = {
    "rate": 0.05,
    "div": 0.02,
    "vol": 0.2,
    "lam": 0.75,
    "mu_j": -0.1,
    "sigma_j": 0.15,
    "spot": 100.0,
}


# --------------------------------------------------------------------------- #
# 1. Closed form vs FFT (the headline cross-validation)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("expiry", [0.1, 0.5, 1.0, 2.0])
def test_closed_form_matches_fft(kind: OptionType, strike: float, expiry: float) -> None:
    option = EuropeanOption(strike, expiry, kind)
    closed = CLOSED.calculate(option, PROC)
    fft = FFT.calculate(option, PROC)
    # Agreement is limited only by the FFT discretisation (~2e-7).
    assert closed == pytest.approx(fft, abs=1e-5)


# --------------------------------------------------------------------------- #
# 2. Black--Scholes limit (lam -> 0)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
def test_reduces_to_black_scholes_when_no_jumps(kind: OptionType, strike: float) -> None:
    no_jump = MertonProcess.from_market(MARKET, vol=0.25, lam=0.0, mu_j=0.0, sigma_j=0.0)
    bs = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=0.25)
    option = EuropeanOption(strike, 1.0, kind)
    reference = AnalyticEuropeanEngine().calculate(option, bs)
    assert CLOSED.calculate(option, no_jump) == pytest.approx(reference, abs=1e-10)
    assert FFT.calculate(option, no_jump) == pytest.approx(reference, abs=1e-5)


# --------------------------------------------------------------------------- #
# 3. Series-truncation convergence
# --------------------------------------------------------------------------- #


def test_poisson_series_converges_monotonically() -> None:
    # Heavy jump activity so many terms matter (lam*T = 2).
    proc = MertonProcess.from_market(MARKET, vol=0.2, lam=2.0, mu_j=-0.1, sigma_j=0.2)
    option = EuropeanOption(100.0, 1.0, OptionType.CALL)
    reference = merton_jump_price(option, proc, tol=0.0, max_terms=80)

    errors = [
        abs(merton_jump_price(option, proc, tol=0.0, max_terms=n) - reference) for n in range(15)
    ]
    # Every extra (non-negative) Poisson term strictly reduces the truncation error.
    assert all(errors[i + 1] < errors[i] for i in range(len(errors) - 1))
    assert errors[-1] < 1e-6


def test_truncation_error_below_stated_tolerance() -> None:
    proc = MertonProcess.from_market(MARKET, vol=0.2, lam=2.0, mu_j=-0.1, sigma_j=0.2)
    option = EuropeanOption(100.0, 1.0, OptionType.CALL)
    fine = merton_jump_price(option, proc, tol=0.0, max_terms=80)
    for tol in (1e-2, 1e-4, 1e-8):
        assert abs(merton_jump_price(option, proc, tol=tol) - fine) < tol


# --------------------------------------------------------------------------- #
# 4. Characteristic function at known points
# --------------------------------------------------------------------------- #


def test_cf_at_zero_maturity_is_spot_power() -> None:
    u = np.array([0.5, 1.0, 2.5, -1.3], dtype=np.complex128)
    cf = merton_characteristic_function(u, 0.0, **_CF_PARAMS)
    np.testing.assert_allclose(cf, np.exp(1j * u * math.log(100.0)), atol=1e-14)


def test_cf_at_zero_argument_is_one() -> None:
    cf = merton_characteristic_function(np.array([0j]), 1.5, **_CF_PARAMS)
    assert cf[0] == pytest.approx(1.0, abs=1e-14)


# --------------------------------------------------------------------------- #
# 5. Structural sanity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", [CLOSED, FFT])
@pytest.mark.parametrize("strike", [85.0, 100.0, 115.0])
def test_put_call_parity(engine: object, strike: float) -> None:
    T = 1.0
    call = engine.calculate(EuropeanOption(strike, T, OptionType.CALL), PROC)  # type: ignore[attr-defined]
    put = engine.calculate(EuropeanOption(strike, T, OptionType.PUT), PROC)  # type: ignore[attr-defined]
    rhs = MARKET.spot * math.exp(-MARKET.div * T) - strike * math.exp(-MARKET.rate * T)
    assert (call - put) == pytest.approx(rhs, abs=1e-9)


def test_prices_non_negative_and_monotone_in_strike() -> None:
    strikes = np.linspace(70.0, 140.0, 15)
    calls = [CLOSED.calculate(EuropeanOption(k, 1.0, OptionType.CALL), PROC) for k in strikes]
    puts = [CLOSED.calculate(EuropeanOption(k, 1.0, OptionType.PUT), PROC) for k in strikes]
    assert all(c >= -1e-12 for c in calls) and all(p >= -1e-12 for p in puts)
    assert np.all(np.diff(calls) < 0.0)  # call decreasing in strike
    assert np.all(np.diff(puts) > 0.0)  # put increasing in strike


def test_forward_is_unaffected_by_jumps() -> None:
    # The drift compensator keeps the spot a martingale: the forward is the plain
    # S0 e^{(r-q)T} regardless of the jump parameters.
    assert PROC.forward(1.0) == pytest.approx(100.0 * math.exp((0.05 - 0.02) * 1.0), abs=1e-12)
    assert PROC.compensator == pytest.approx(math.exp(-0.1 + 0.5 * 0.15**2) - 1.0, abs=1e-15)


# --------------------------------------------------------------------------- #
# 6. FFT numerical stability (alpha and grid are knobs, not magic numbers)
# --------------------------------------------------------------------------- #


def test_stable_across_damping_alpha() -> None:
    option = EuropeanOption(105.0, 1.0, OptionType.CALL)
    prices = [MertonFFTEngine(alpha=a).calculate(option, PROC) for a in (1.25, 1.5, 2.0)]
    assert max(prices) - min(prices) < 1e-4


def test_stable_across_fft_grid() -> None:
    option = EuropeanOption(105.0, 1.0, OptionType.CALL)
    prices = [
        MertonFFTEngine(n_fft=n, eta=eta).calculate(option, PROC)
        for n, eta in [(2048, 0.25), (4096, 0.25), (8192, 0.25), (4096, 0.1)]
    ]
    assert max(prices) - min(prices) < 1e-5


# --------------------------------------------------------------------------- #
# 7. The jump smile — why Black--Scholes fails
# --------------------------------------------------------------------------- #


def test_downward_jumps_produce_a_negative_skew() -> None:
    # Convert short-dated Merton prices to Black--Scholes implied vols; downward
    # jumps (mu_j < 0) make low strikes trade at a higher implied vol than high
    # strikes (negative skew), and the smile is not flat (jumps add convexity).
    T = 0.25
    strikes = [85.0, 100.0, 115.0]
    ivs = []
    for k in strikes:
        option = EuropeanOption(k, T, OptionType.CALL)
        price = FFT.calculate(option, PROC)
        ivs.append(implied_volatility(price, option, MARKET))
    low, atm, high = ivs
    assert low > atm > high  # negative skew
    assert low - high > 0.01  # a materially non-flat smile


# --------------------------------------------------------------------------- #
# 8. Wiring / validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("engine", [CLOSED, FFT])
def test_rejects_american_option(engine: object) -> None:
    with pytest.raises(ValueError, match="European exercise only"):
        engine.calculate(AmericanOption(100.0, 1.0, OptionType.PUT), PROC)  # type: ignore[attr-defined]


@pytest.mark.parametrize("engine", [CLOSED, FFT])
def test_rejects_non_merton_process(engine: object) -> None:
    bs = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    with pytest.raises(TypeError, match="MertonProcess"):
        engine.calculate(EuropeanOption(100.0, 1.0, OptionType.CALL), bs)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"n_fft": 1}, "n_fft must be at least 2"),
        ({"eta": 0.0}, "eta must be positive"),
    ],
)
def test_fft_engine_parameter_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MertonFFTEngine(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"tol": -1.0}, "tol must be non-negative"),
        ({"max_terms": -1}, "max_terms must be non-negative"),
    ],
)
def test_closed_form_engine_parameter_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MertonClosedFormEngine(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"spot": -1.0}, "spot must be positive"),
        ({"vol": -0.1}, "vol must be non-negative"),
        ({"lam": -0.1}, "lam must be non-negative"),
        ({"sigma_j": -0.1}, "sigma_j must be non-negative"),
    ],
)
def test_process_validation(kwargs: dict[str, float], match: str) -> None:
    base = {"spot": 100.0, "rate": 0.05, "vol": 0.2, "lam": 0.5, "mu_j": -0.1, "sigma_j": 0.15}
    with pytest.raises(ValueError, match=match):
        MertonProcess(**{**base, **kwargs})  # type: ignore[arg-type]
