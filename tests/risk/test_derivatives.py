r"""Option-book P\&L — validating the pricing↔risk bridge (numerical-validation skill).

- **Consistency (no drift).** The book's value and scenario P\&L go through the
  pricing engines themselves, so they must equal the pricers' own outputs exactly.
- **Greeks-consistent P\&L.** Bump Greeks match the analytic Greeks for a European
  book; small-move full-revaluation P\&L matches the first-order approximation; a
  delta-hedged position exposes the residual :math:`\tfrac12\Gamma\,\delta S^2`.
- **The headline — where the approximations break.** On identical scenario sets:
  a near-linear (deep-ITM) book agrees across all three methods; a short-gamma
  book's delta-normal VaR *under*-states the full-revaluation VaR and a long-gamma
  book's *over*-states it (the signs follow from the sign of the omitted
  :math:`\tfrac12\Gamma\,\delta S^2` term); delta-gamma repairs most of the error.
- **Backtests reused unchanged.** Kupiec on realized full-revaluation P\&L rejects
  the delta-normal VaR of a short-gamma book and accepts the full-revaluation VaR.
- Seeded scenarios throughout; determinism asserted.
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
    OptionType,
)
from quantica.risk import (
    BookPosition,
    MarketScenarios,
    OptionBook,
    book_var_es,
    empirical_var_es,
    kupiec_pof,
)

PROC = BlackScholesProcess(spot=100.0, rate=0.02, div=0.0, vol=0.2)
ENGINE = AnalyticEuropeanEngine()
CALL = EuropeanOption(100.0, 0.5, OptionType.CALL)
PUT = EuropeanOption(100.0, 0.5, OptionType.PUT)
DAILY_VOL = 0.0126  # ~20% annual over one trading day

LEVEL = 0.99


def straddle(quantity: float) -> OptionBook:
    """An ATM straddle book (long gamma for positive quantity, short for negative)."""
    return OptionBook(
        positions=(BookPosition(CALL, ENGINE, quantity), BookPosition(PUT, ENGINE, quantity)),
        process=PROC,
    )


# --------------------------------------------------------------------------- #
# 1. Consistency — no drift between the risk path and the pricing path
# --------------------------------------------------------------------------- #


def test_book_value_equals_pricer_output() -> None:
    book = OptionBook(positions=(BookPosition(CALL, ENGINE, 3.0),), process=PROC)
    assert book.value() == pytest.approx(3.0 * ENGINE.calculate(CALL, PROC), abs=0.0)


def test_full_revaluation_matches_manual_repricing() -> None:
    scenarios = MarketScenarios(spot_returns=np.array([-0.04, 0.0, 0.03]))
    book = OptionBook(positions=(BookPosition(CALL, ENGINE, 2.0),), process=PROC)
    pnl = book.full_revaluation_pnl(scenarios)
    base = 2.0 * ENGINE.calculate(CALL, PROC)
    for i, r in enumerate(scenarios.spot_returns):
        manual = 2.0 * ENGINE.calculate(CALL, PROC.with_spot(100.0 * (1.0 + r))) - base
        assert pnl[i] == pytest.approx(manual, abs=0.0)
    assert pnl[1] == pytest.approx(0.0, abs=0.0)  # zero scenario -> zero P&L


def test_vol_scenarios_reprice_through_with_vol() -> None:
    scenarios = MarketScenarios(
        spot_returns=np.array([0.0, 0.01]), vol_shifts=np.array([0.05, -0.02])
    )
    book = OptionBook(positions=(BookPosition(CALL, ENGINE, 1.0),), process=PROC)
    pnl = book.full_revaluation_pnl(scenarios)
    base = ENGINE.calculate(CALL, PROC)
    manual0 = ENGINE.calculate(CALL, PROC.with_vol(0.25)) - base
    assert pnl[0] == pytest.approx(manual0, abs=0.0)


# --------------------------------------------------------------------------- #
# 2. Greeks-consistent P&L
# --------------------------------------------------------------------------- #


def test_bump_greeks_match_analytic_greeks() -> None:
    book = OptionBook(positions=(BookPosition(CALL, ENGINE, 1.0),), process=PROC)
    bumped = book.greeks()
    analytic = ENGINE.greeks(CALL, PROC)
    assert bumped.delta == pytest.approx(analytic.delta, rel=1e-6)
    assert bumped.gamma == pytest.approx(analytic.gamma, rel=1e-4)
    assert bumped.vega == pytest.approx(analytic.vega, rel=1e-6)


def test_underlying_leg_has_delta_one_and_no_gamma_vega() -> None:
    book = OptionBook(positions=(), process=PROC, underlying_quantity=5.0)
    g = book.greeks()
    assert g.delta == pytest.approx(5.0, rel=1e-10)
    assert g.gamma == pytest.approx(0.0, abs=1e-8)
    assert g.vega == pytest.approx(0.0, abs=1e-10)


def test_small_moves_full_matches_delta_normal() -> None:
    scenarios = MarketScenarios(spot_returns=np.linspace(-1e-4, 1e-4, 9))
    book = OptionBook(positions=(BookPosition(CALL, ENGINE, 100.0),), process=PROC)
    full = book.full_revaluation_pnl(scenarios)
    linear = book.delta_normal_pnl(scenarios)
    # First-order error is O(dS^2): with dS <= 1e-2, gamma ~ 2.8 -> error <= ~2e-2.
    np.testing.assert_allclose(full, linear, atol=2e-2)


def test_delta_hedged_book_isolates_gamma_pnl() -> None:
    # Long call, short delta units of the underlying: the linear P&L vanishes and
    # full revaluation shows the positive gamma residual ~ 0.5*gamma*dS^2 for
    # moves of BOTH signs.
    single = OptionBook(positions=(BookPosition(CALL, ENGINE, 1.0),), process=PROC)
    g = single.greeks()
    hedged = OptionBook(
        positions=(BookPosition(CALL, ENGINE, 1.0),),
        process=PROC,
        underlying_quantity=-g.delta,
    )
    scenarios = MarketScenarios(spot_returns=np.array([-0.05, -0.02, 0.02, 0.05]))
    linear = hedged.delta_normal_pnl(scenarios)
    full = hedged.full_revaluation_pnl(scenarios)
    gamma_term = 0.5 * g.gamma * (PROC.spot * scenarios.spot_returns) ** 2
    np.testing.assert_allclose(linear, 0.0, atol=1e-10)
    assert np.all(full > 0.0)  # long gamma profits from any move
    np.testing.assert_allclose(full, gamma_term, rtol=0.05)


# --------------------------------------------------------------------------- #
# 3. The headline — where delta-normal breaks, and which way
# --------------------------------------------------------------------------- #

RNG_SCEN = np.random.default_rng(0)
SCENARIOS = MarketScenarios.generate(8000, RNG_SCEN, spot_vol=DAILY_VOL)


def test_near_linear_book_all_methods_agree() -> None:
    itm = EuropeanOption(60.0, 0.5, OptionType.CALL)  # deep ITM: delta ~ 1, gamma ~ 0
    book = OptionBook(positions=(BookPosition(itm, ENGINE, 100.0),), process=PROC)
    full = book_var_es(book, SCENARIOS, LEVEL, method="full").var
    dn = book_var_es(book, SCENARIOS, LEVEL, method="delta-normal").var
    dg = book_var_es(book, SCENARIOS, LEVEL, method="delta-gamma").var
    assert dn == pytest.approx(full, rel=0.01)
    assert dg == pytest.approx(full, rel=0.01)


def test_short_gamma_delta_normal_underestimates_var() -> None:
    # Short straddle: the omitted -0.5*|gamma|*dS^2 term is a pure LOSS, so the
    # linear approximation understates tail risk; delta-gamma repairs it.
    book = straddle(-100.0)
    full = book_var_es(book, SCENARIOS, LEVEL, method="full").var
    dn = book_var_es(book, SCENARIOS, LEVEL, method="delta-normal").var
    dg = book_var_es(book, SCENARIOS, LEVEL, method="delta-gamma").var
    assert dn < 0.7 * full  # material underestimation
    assert abs(dg - full) < 0.2 * abs(dn - full)  # delta-gamma repairs most of it


def test_long_gamma_delta_normal_overestimates_var() -> None:
    # Long straddle: the omitted +0.5*gamma*dS^2 term CUSHIONS losses, so the
    # linear approximation overstates tail risk (conservative, but wrong).
    book = straddle(100.0)
    full = book_var_es(book, SCENARIOS, LEVEL, method="full").var
    dn = book_var_es(book, SCENARIOS, LEVEL, method="delta-normal").var
    dg = book_var_es(book, SCENARIOS, LEVEL, method="delta-gamma").var
    assert dn > 1.5 * full  # material overestimation
    assert abs(dg - full) < 0.2 * abs(dn - full)


def test_es_orders_the_same_way_as_var() -> None:
    book = straddle(-100.0)
    full = book_var_es(book, SCENARIOS, LEVEL, method="full")
    dn = book_var_es(book, SCENARIOS, LEVEL, method="delta-normal")
    assert full.es >= full.var and dn.es >= dn.var
    assert dn.es < full.es  # the ES tail is understated too


# --------------------------------------------------------------------------- #
# 4. Backtests reused unchanged on option-book P&L
# --------------------------------------------------------------------------- #


def test_kupiec_rejects_delta_normal_var_on_short_gamma_book() -> None:
    # One-day realized P&L over T days by full revaluation (the "truth"), against
    # static VaR forecasts from the same scenario distribution: the full-reval VaR
    # achieves ~1% coverage (Kupiec passes), the delta-normal VaR takes far too
    # many exceptions (Kupiec rejects). The backtest layer is untouched.
    book = straddle(-100.0)
    var_full = book_var_es(book, SCENARIOS, LEVEL, method="full").var
    var_dn = book_var_es(book, SCENARIOS, LEVEL, method="delta-normal").var

    rng = np.random.default_rng(42)
    realized_scenarios = MarketScenarios.generate(750, rng, spot_vol=DAILY_VOL)
    realized_losses = -book.full_revaluation_pnl(realized_scenarios)

    n = realized_losses.size
    x_full = int(np.sum(realized_losses > var_full))
    x_dn = int(np.sum(realized_losses > var_dn))
    assert not kupiec_pof(x_full, n, LEVEL).reject()
    assert kupiec_pof(x_dn, n, LEVEL).reject()
    assert x_dn > x_full


def test_book_pnl_feeds_empirical_var_es_directly() -> None:
    # The drop-in seam: losses = -pnl straight into the existing measure.
    book = straddle(-10.0)
    pnl = book.full_revaluation_pnl(SCENARIOS)
    direct = empirical_var_es(-pnl, LEVEL)
    via_adapter = book_var_es(book, SCENARIOS, LEVEL, method="full")
    assert direct.var == pytest.approx(via_adapter.var, abs=0.0)
    assert direct.es == pytest.approx(via_adapter.es, abs=0.0)


# --------------------------------------------------------------------------- #
# 5. Mixed engines, determinism, wiring
# --------------------------------------------------------------------------- #


def test_mixed_book_with_american_position() -> None:
    # An American put priced on the binomial lattice sits in the same book as an
    # analytic European call; full revaluation just works through both engines.
    american = AmericanOption(105.0, 0.5, OptionType.PUT)
    book = OptionBook(
        positions=(
            BookPosition(CALL, ENGINE, 50.0),
            BookPosition(american, BinomialEngine(steps=200), 50.0),
        ),
        process=PROC,
    )
    scenarios = MarketScenarios(spot_returns=np.array([-0.03, 0.0, 0.03]))
    pnl = book.full_revaluation_pnl(scenarios)
    assert pnl[1] == pytest.approx(0.0, abs=0.0)
    assert np.all(np.isfinite(pnl))
    # The book is long the call and long the put: it profits at both extremes less
    # than a pure straddle would, but the P&L must not be flat.
    assert pnl[0] != pytest.approx(pnl[2], rel=0.01)


def test_seeded_scenarios_are_deterministic() -> None:
    a = MarketScenarios.generate(1000, np.random.default_rng(7), spot_vol=0.01, vol_shift_vol=0.005)
    b = MarketScenarios.generate(1000, np.random.default_rng(7), spot_vol=0.01, vol_shift_vol=0.005)
    np.testing.assert_array_equal(a.spot_returns, b.spot_returns)
    assert a.vol_shifts is not None and b.vol_shifts is not None
    np.testing.assert_array_equal(a.vol_shifts, b.vol_shifts)
    book = straddle(-1.0)
    np.testing.assert_array_equal(book.full_revaluation_pnl(a), book.full_revaluation_pnl(b))


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="quantity must be finite and non-zero"):
        BookPosition(CALL, ENGINE, 0.0)
    with pytest.raises(ValueError, match="book is empty"):
        OptionBook(positions=(), process=PROC)
    with pytest.raises(ValueError, match="spot positive"):
        MarketScenarios(spot_returns=np.array([-1.0]))
    with pytest.raises(ValueError, match="non-empty 1-D"):
        MarketScenarios(spot_returns=np.array([[0.01]]))
    with pytest.raises(ValueError, match="vol_shifts must match"):
        MarketScenarios(spot_returns=np.array([0.01]), vol_shifts=np.array([0.01, 0.02]))
    with pytest.raises(ValueError, match="n_scenarios must be at least 1"):
        MarketScenarios.generate(0, np.random.default_rng(0), spot_vol=0.01)
    book = straddle(1.0)
    with pytest.raises(ValueError, match="method must be"):
        book_var_es(book, SCENARIOS, LEVEL, method="taylor")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bump sizes must be positive"):
        book.greeks(rel_spot_bump=0.0)
