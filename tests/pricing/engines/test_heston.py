"""Validation of the Heston FFT engine (numerical-validation skill).

- **Black--Scholes limit (the featured anchor).** With ``xi = 0`` (deterministic
  variance) and ``v0 = theta = sigma^2``, Heston collapses to Black--Scholes;
  the FFT price matches ``AnalyticEuropeanEngine`` to tight tolerance. This is the
  strongest correctness check available without an external reference.
- **Characteristic function** correct at the known points (``t = 0`` gives
  ``S0^{iu}``; ``u = 0`` gives 1) — validated directly, not only via the price.
- **Structural sanity** — put--call parity under Heston, prices non-negative and
  monotone in strike (arbitrage-free).
- **Numerical stability** — the price is stable across the damping factor ``alpha``
  and the FFT grid; ``alpha`` is a numerical knob, not a magic constant.

The QuantLib ``AnalyticHestonEngine`` benchmark lives in
``test_benchmark_quantlib.py``.
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
    HestonFFTEngine,
    HestonProcess,
    OptionType,
    heston_characteristic_function,
)

HESTON = HestonProcess(
    spot=100.0, rate=0.05, div=0.02, v0=0.04, kappa=1.5, theta=0.04, xi=0.4, rho=-0.7
)
ENGINE = HestonFFTEngine()

_CF_PARAMS = {
    "rate": 0.05,
    "div": 0.02,
    "v0": 0.04,
    "kappa": 1.5,
    "theta": 0.04,
    "xi": 0.3,
    "rho": -0.6,
    "spot": 100.0,
}


# --------------------------------------------------------------------------- #
# 1. Black--Scholes limit (featured)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
@pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
@pytest.mark.parametrize("expiry", [0.5, 1.0, 2.0])
def test_reduces_to_black_scholes_when_xi_zero(
    kind: OptionType, strike: float, expiry: float
) -> None:
    sigma = 0.25
    heston = HestonProcess(
        spot=100.0, rate=0.05, div=0.02, v0=sigma**2, kappa=1.5, theta=sigma**2, xi=0.0, rho=-0.5
    )
    black_scholes = BlackScholesProcess(spot=100.0, rate=0.05, div=0.02, vol=sigma)
    option = EuropeanOption(strike=strike, expiry=expiry, option_type=kind)
    assert ENGINE.calculate(option, heston) == pytest.approx(
        AnalyticEuropeanEngine().calculate(option, black_scholes), abs=1e-5
    )


# --------------------------------------------------------------------------- #
# 2. Characteristic function at known points
# --------------------------------------------------------------------------- #


def test_cf_at_zero_maturity_is_spot_power() -> None:
    # phi(u) at tau = 0 is E[e^{iu ln S_0}] = S_0^{iu} (deterministic).
    u = np.array([0.5, 1.0, 2.5, -1.3], dtype=np.complex128)
    cf = heston_characteristic_function(u, 0.0, **_CF_PARAMS)
    np.testing.assert_allclose(cf, np.exp(1j * u * math.log(100.0)), atol=1e-14)


def test_cf_at_zero_argument_is_one() -> None:
    # phi(0) = E[1] = 1 for any maturity.
    cf = heston_characteristic_function(np.array([0j]), 1.5, **_CF_PARAMS)
    assert cf[0] == pytest.approx(1.0, abs=1e-14)


# --------------------------------------------------------------------------- #
# 3. Structural sanity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("strike", [85.0, 100.0, 115.0])
def test_put_call_parity(strike: float) -> None:
    T = 1.0
    call = ENGINE.calculate(EuropeanOption(strike, T, OptionType.CALL), HESTON)
    put = ENGINE.calculate(EuropeanOption(strike, T, OptionType.PUT), HESTON)
    rhs = HESTON.spot * math.exp(-HESTON.div * T) - strike * math.exp(-HESTON.rate * T)
    assert (call - put) == pytest.approx(rhs, abs=1e-10)


def test_prices_non_negative_and_monotone_in_strike() -> None:
    strikes = np.linspace(70.0, 140.0, 15)
    calls = [ENGINE.calculate(EuropeanOption(k, 1.0, OptionType.CALL), HESTON) for k in strikes]
    puts = [ENGINE.calculate(EuropeanOption(k, 1.0, OptionType.PUT), HESTON) for k in strikes]
    assert all(c >= -1e-10 for c in calls) and all(p >= -1e-10 for p in puts)
    assert np.all(np.diff(calls) < 0.0)  # call decreasing in strike
    assert np.all(np.diff(puts) > 0.0)  # put increasing in strike


# --------------------------------------------------------------------------- #
# 4. Numerical stability (alpha and grid are knobs, not magic numbers)
# --------------------------------------------------------------------------- #


def test_stable_across_damping_alpha() -> None:
    # The price is theoretically independent of the damping alpha; across a
    # reasonable range it agrees to well under a cent. (Very small alpha near 1
    # degrades slightly, which is why the default is 1.5.)
    option = EuropeanOption(105.0, 1.0, OptionType.CALL)
    prices = [HestonFFTEngine(alpha=a).calculate(option, HESTON) for a in (1.25, 1.5, 2.0)]
    assert max(prices) - min(prices) < 1e-3


def test_stable_across_fft_grid() -> None:
    # Doubling N or refining eta leaves the price essentially unchanged (the
    # integral has converged). A too-coarse eta would not — hence eta is exposed.
    option = EuropeanOption(105.0, 1.0, OptionType.CALL)
    prices = [
        HestonFFTEngine(n_fft=n, eta=eta).calculate(option, HESTON)
        for n, eta in [(2048, 0.25), (4096, 0.25), (8192, 0.25), (4096, 0.1)]
    ]
    assert max(prices) - min(prices) < 1e-5


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_rejects_american_option() -> None:
    with pytest.raises(ValueError, match="European exercise only"):
        ENGINE.calculate(AmericanOption(100.0, 1.0, OptionType.PUT), HESTON)


def test_rejects_non_heston_process() -> None:
    bs = BlackScholesProcess(spot=100.0, rate=0.05, vol=0.2)
    with pytest.raises(TypeError, match="HestonProcess"):
        ENGINE.calculate(EuropeanOption(100.0, 1.0, OptionType.CALL), bs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"n_fft": 1}, "n_fft must be at least 2"),
        ({"eta": 0.0}, "eta must be positive"),
    ],
)
def test_engine_parameter_validation(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        HestonFFTEngine(**kwargs)  # type: ignore[arg-type]
