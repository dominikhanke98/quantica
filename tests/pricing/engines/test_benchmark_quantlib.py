"""QuantLib benchmarks for the pricing engines — the effective challenge (skill §4).

The analytic engine is re-priced against QuantLib's ``AnalyticEuropeanEngine``
(price and all Greeks), and the CRR tree against QuantLib's own CRR
``BinomialVanillaEngine``. We match QuantLib's analytic engine to ~1e-12 because
both use the same continuous-compounding Black--Scholes--Merton model; the only
care needed is aligning conventions so the year fraction is exactly ``T``:

* ``Actual365Fixed`` day count with expiry set ``round(365*T)`` days out, so
  ``yearFraction == T`` exactly (no day-count drift);
* ``NullCalendar`` (no holiday adjustment);
* ``FlatForward`` term structures, which are continuously compounded — matching
  our ``e^{-rT}`` / ``e^{-qT}`` discounting and continuous dividend yield.

QuantLib's Greek conventions match ours: ``vega``/``rho`` are per unit (not per
1%) and ``theta`` is per year (``VanillaOption.theta()``), so no rescaling. The
two CRR trees differ only by an ``O(1/N)`` lattice-placement discrepancy.

Run with ``pytest -m benchmark`` (needs the ``benchmark`` extra installed).
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BinomialEngine,
    BlackScholesProcess,
    EuropeanOption,
    OptionType,
)

ql = pytest.importorskip("QuantLib")  # skip cleanly if the benchmark extra is absent

pytestmark = pytest.mark.benchmark

# A single representative contract/market with non-zero dividends exercises
# every term (spot leg, strike leg, carry) in both price and Greeks.
SPOT, RATE, DIV, VOL = 100.0, 0.05, 0.02, 0.2
STRIKE, EXPIRY = 105.0, 1.0
_EVAL_DATE = (15, 6, 2020)  # arbitrary; only relative dates matter


def _ql_parts(kind: OptionType):  # type: ignore[no-untyped-def]
    """Build the shared QuantLib payoff, exercise, and process for our conventions."""
    today = ql.Date(*_EVAL_DATE)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    calendar = ql.NullCalendar()

    spot_h = ql.QuoteHandle(ql.SimpleQuote(SPOT))
    r_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, RATE, day_count))
    q_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, DIV, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, calendar, VOL, day_count))
    process = ql.BlackScholesMertonProcess(spot_h, q_ts, r_ts, vol_ts)

    ql_kind = ql.Option.Call if kind is OptionType.CALL else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(ql_kind, STRIKE)
    exercise = ql.EuropeanExercise(today + ql.Period(round(365 * EXPIRY), ql.Days))
    return payoff, exercise, process


def _ql_option(kind: OptionType):  # type: ignore[no-untyped-def]
    """The QuantLib analytic European option."""
    payoff, exercise, process = _ql_parts(kind)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
    return option


def _our_option(kind: OptionType) -> tuple[EuropeanOption, BlackScholesProcess]:
    proc = BlackScholesProcess(spot=SPOT, rate=RATE, div=DIV, vol=VOL)
    opt = EuropeanOption(strike=STRIKE, expiry=EXPIRY, option_type=kind)
    opt.set_engine(AnalyticEuropeanEngine())
    return opt, proc


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_price_matches_quantlib(kind: OptionType) -> None:
    opt, proc = _our_option(kind)
    assert np.isclose(opt.npv(proc), _ql_option(kind).NPV(), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_greeks_match_quantlib(kind: OptionType) -> None:
    opt, proc = _our_option(kind)
    ours = opt.greeks(proc)
    ql_opt = _ql_option(kind)
    # QuantLib conventions align with ours (per-unit vega/rho, per-year theta).
    assert np.isclose(ours.delta, ql_opt.delta(), rtol=1e-10)
    assert np.isclose(ours.gamma, ql_opt.gamma(), rtol=1e-10)
    assert np.isclose(ours.vega, ql_opt.vega(), rtol=1e-10)
    assert np.isclose(ours.rho, ql_opt.rho(), rtol=1e-10)
    assert np.isclose(ours.theta, ql_opt.theta(), rtol=1e-10)


_CRR_STEPS = 200


@pytest.mark.parametrize("kind", [OptionType.CALL, OptionType.PUT])
def test_crr_matches_quantlib(kind: OptionType) -> None:
    # Both are CRR trees converging to the same Black--Scholes price, but the
    # two libraries place their lattices slightly differently, so at finite N
    # they agree only to an O(1/N) grid difference (~8e-5 at N=200). Convergence
    # to the analytic price itself is covered in test_binomial.py.
    proc = BlackScholesProcess(spot=SPOT, rate=RATE, div=DIV, vol=VOL)
    option = EuropeanOption(strike=STRIKE, expiry=EXPIRY, option_type=kind)
    ours = BinomialEngine(steps=_CRR_STEPS).calculate(option, proc)

    payoff, exercise, process = _ql_parts(kind)
    ql_option = ql.VanillaOption(payoff, exercise)
    ql_option.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", _CRR_STEPS))
    theirs = ql_option.NPV()

    assert abs(ours - theirs) < 5e-4
