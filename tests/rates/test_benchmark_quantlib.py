"""QuantLib benchmark for the curve interpolation — the effective challenge (skill §4).

Our discount-curve **interpolation** is cross-checked against QuantLib's own interpolated
curves: log-linear on discount factors against ``ql.DiscountCurve`` and linear on zero rates
against ``ql.ZeroCurve``. With the time metric aligned (``Actual365Fixed`` and integer-day
pillars, so ``yearFraction`` equals our pillar times exactly) the two agree to machine
precision.

Scope note (logged as a tool-gap): a full QuantLib ``PiecewiseYieldCurve`` *bootstrap*
benchmark is **not** clean, because ``DepositRateHelper`` / ``SwapRateHelper`` impose market
conventions (calendars, business-day adjustment, float-index day counts and fixings) that this
deliberately-simplified single-curve model abstracts away — so the *bootstrap* is validated by
exact self-consistency plus a hand computation, and the *interpolation* is benchmarked here.
QuantLib's monotone cubic is a different (Hyman-filtered) algorithm, so our PCHIP monotone
scheme is validated against ``scipy`` instead.

For the **interest-rate products**, the market-standard **Black-76** formula is benchmarked
here against ``ql.blackFormula`` to machine precision. The **Hull--White** option prices are
*not* benchmarked against QuantLib's ``HullWhite`` engines (logged as a tool-gap): QuantLib's
term-structure handle uses its own interpolation convention, so the fitted :math:`\theta(t)`
drift — and hence every bond option built on it — differs from ours by a curve-convention
residual that swamps the option value. The Hull--White pricers are instead validated by rich
self-consistency (analytic vs Monte Carlo, consistency with Black-76 through the implied vol,
Jamshidian, and put-call parity) in ``test_hull_white_options.py``.

Run with ``pytest -m benchmark`` (needs the ``benchmark`` extra installed).
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.rates import CIR, DiscountCurve, Vasicek, black76, linear_zero, log_linear_discount

pytestmark = pytest.mark.benchmark


def _aligned_curve():  # type: ignore[no-untyped-def]
    """Pillars on integer-day dates so QuantLib's year fractions equal our pillar times."""
    ql = pytest.importorskip("QuantLib")
    today = ql.Date(15, 6, 2024)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    pillar_dates = [today + ql.Period(d, ql.Days) for d in (182, 365, 730, 1095, 1825, 2555, 3650)]
    times = np.array([day_count.yearFraction(today, d) for d in pillar_dates])
    zeros = np.array([0.030, 0.032, 0.035, 0.037, 0.040, 0.042, 0.043])
    dfs = np.exp(-zeros * times)
    query_dates = [today + ql.Period(d, ql.Days) for d in (90, 270, 500, 1400, 2200, 3100)]
    query_times = np.array([day_count.yearFraction(today, d) for d in query_dates])
    return ql, today, day_count, pillar_dates, times, zeros, dfs, query_dates, query_times


def test_log_linear_discount_matches_quantlib() -> None:
    """Our log-linear-on-discount curve matches QuantLib's ``DiscountCurve`` exactly."""
    ql, today, dc, pillar_dates, times, _zeros, dfs, q_dates, q_times = _aligned_curve()
    ql_curve = ql.DiscountCurve([today, *pillar_dates], [1.0, *dfs.tolist()], dc)
    mine = DiscountCurve(times, dfs, log_linear_discount())
    ql_dfs = np.array([ql_curve.discount(d) for d in q_dates])
    assert np.allclose(mine.discount_factor(q_times), ql_dfs, atol=1e-13)


def test_linear_zero_matches_quantlib() -> None:
    """Our linear-on-zero curve matches QuantLib's linear ``ZeroCurve`` to machine precision."""
    ql, today, dc, pillar_dates, times, zeros, dfs, q_dates, q_times = _aligned_curve()
    ql_curve = ql.ZeroCurve([today, *pillar_dates], [zeros[0], *zeros.tolist()], dc)
    mine = DiscountCurve(times, dfs, linear_zero())
    ql_zeros = np.array([ql_curve.zeroRate(d, dc, ql.Continuous).rate() for d in q_dates])
    assert np.allclose(mine.zero_rate(q_times), ql_zeros, atol=1e-13)


def test_vasicek_bond_matches_quantlib() -> None:
    """Our Vasicek zero-coupon bond matches ``ql.Vasicek.discountBond`` to machine precision."""
    ql = pytest.importorskip("QuantLib")
    a, b, sigma, r0 = 0.3, 0.04, 0.015, 0.03
    mine = Vasicek(a=a, b=b, sigma=sigma, r0=r0)
    ql_model = ql.Vasicek(r0, a, b, sigma)  # ql.Vasicek(r0, a, b, sigma)
    for maturity in (1.0, 5.0, 10.0):
        assert np.isclose(
            float(mine.zero_coupon_bond(0.0, maturity, r0)),
            ql_model.discountBond(0.0, maturity, r0),
            atol=1e-13,
        )


def test_cir_bond_matches_quantlib() -> None:
    """Our CIR zero-coupon bond matches ``ql.CoxIngersollRoss.discountBond`` exactly."""
    ql = pytest.importorskip("QuantLib")
    a, b, sigma, r0 = 0.5, 0.04, 0.05, 0.03
    mine = CIR(a=a, b=b, sigma=sigma, r0=r0)
    ql_model = ql.CoxIngersollRoss(r0, b, a, sigma)  # ql.CoxIngersollRoss(r0, theta=b, k=a, sigma)
    for maturity in (1.0, 5.0, 10.0):
        assert np.isclose(
            float(mine.zero_coupon_bond(0.0, maturity, r0)),
            ql_model.discountBond(0.0, maturity, r0),
            atol=1e-13,
        )


def test_black76_matches_quantlib_black_formula() -> None:
    """Our (undiscounted) Black-76 matches ``ql.blackFormula`` to machine precision."""
    ql = pytest.importorskip("QuantLib")
    forward, expiry = 0.04, 2.0
    for strike in (0.03, 0.04, 0.05):
        for vol in (0.15, 0.30):
            std = vol * np.sqrt(expiry)
            call = black76(forward, strike, vol, expiry, call=True)
            put = black76(forward, strike, vol, expiry, call=False)
            assert np.isclose(
                call, ql.blackFormula(ql.Option.Call, strike, forward, std), atol=1e-14
            )
            assert np.isclose(put, ql.blackFormula(ql.Option.Put, strike, forward, std), atol=1e-14)
