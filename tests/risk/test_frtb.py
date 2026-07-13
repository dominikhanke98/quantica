r"""FRTB P\&L-attribution test — validation (numerical-validation skill).

- **Statistics anchored to scipy**: the hand-rolled Spearman and KS match
  ``scipy.stats.spearmanr`` / ``ks_2samp`` to machine precision, ties included.
- **Zone thresholds against the published FRTB breakpoints** (BCBS d457, MAR33):
  Spearman green ≥ 0.80 / red < 0.70; KS green ≤ 0.09 / red > 0.12; overall zone
  is the worse of the two.
- **Known-truth construction (the headline)**: RTPL and HPL are built — via real
  option books reusing the derivatives-risk gamma divergence — to agree (green)
  or diverge (amber/red) *by construction*, and PLA sorts them into the right
  zone. This proves PLA catches exactly the risk-factor-inadequacy it is designed
  to catch: a short-gamma desk whose risk model carries only delta fails because
  its factors cannot span the curved P\&L.
- **No drift**: the HPL used by ``book_pla_test`` is the full-revaluation pricing
  path itself.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BlackScholesProcess,
    EuropeanOption,
    OptionType,
)
from quantica.risk import (
    BookPosition,
    MarketScenarios,
    OptionBook,
    PLAZone,
    book_pla_test,
    ks_distance,
    pla_test,
    spearman_correlation,
)
from quantica.risk.frtb import (
    KS_AMBER_THRESHOLD,
    KS_RED_THRESHOLD,
    SPEARMAN_AMBER_THRESHOLD,
    SPEARMAN_RED_THRESHOLD,
)
from scipy.stats import ks_2samp, spearmanr

PROC = BlackScholesProcess(spot=100.0, rate=0.02, div=0.0, vol=0.2)
ENGINE = AnalyticEuropeanEngine()
CALL = EuropeanOption(100.0, 0.5, OptionType.CALL)
PUT = EuropeanOption(100.0, 0.5, OptionType.PUT)
N_DAYS = 250  # the FRTB observation window


def straddle(quantity: float) -> OptionBook:
    return OptionBook(
        positions=(BookPosition(CALL, ENGINE, quantity), BookPosition(PUT, ENGINE, quantity)),
        process=PROC,
    )


def deep_itm_book() -> OptionBook:
    itm = EuropeanOption(60.0, 0.5, OptionType.CALL)  # delta ~ 1, gamma ~ 0
    return OptionBook(positions=(BookPosition(itm, ENGINE, 100.0),), process=PROC)


# --------------------------------------------------------------------------- #
# 1. Statistics anchored to scipy
# --------------------------------------------------------------------------- #


def test_spearman_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        a = rng.normal(size=200)
        b = 0.7 * a + rng.normal(size=200)
        assert spearman_correlation(a, b) == pytest.approx(float(spearmanr(a, b).statistic))


def test_spearman_matches_scipy_with_ties() -> None:
    rng = np.random.default_rng(1)
    a = rng.integers(0, 5, 300).astype(float)  # heavy ties
    b = rng.integers(0, 5, 300).astype(float)
    assert spearman_correlation(a, b) == pytest.approx(float(spearmanr(a, b).statistic))


def test_ks_matches_scipy() -> None:
    rng = np.random.default_rng(2)
    for scale in (1.0, 1.5, 3.0):
        a = rng.normal(0, 1, 250)
        b = rng.normal(0, scale, 250)
        assert ks_distance(a, b) == pytest.approx(float(ks_2samp(a, b).statistic), abs=1e-12)


def test_constant_rtpl_gives_zero_correlation() -> None:
    # A perfectly delta-hedged book scored by a delta-only model: RTPL is constant,
    # rank correlation undefined -> reported 0.0 -> red zone.
    result = pla_test(np.zeros(50), np.random.default_rng(3).normal(size=50))
    assert result.spearman == 0.0
    assert result.spearman_zone is PLAZone.RED


# --------------------------------------------------------------------------- #
# 2. Zone thresholds against the published FRTB breakpoints
# --------------------------------------------------------------------------- #


def test_published_threshold_values() -> None:
    # BCBS d457, MAR33: cite the numbers so a change is a conscious edit.
    assert SPEARMAN_AMBER_THRESHOLD == 0.80
    assert SPEARMAN_RED_THRESHOLD == 0.70
    assert KS_AMBER_THRESHOLD == 0.09
    assert KS_RED_THRESHOLD == 0.12


def _series_with_spearman(target: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """Construct an (RTPL, HPL) pair with a chosen Spearman and a tiny KS."""
    rng = np.random.default_rng(7)
    hpl = np.sort(rng.normal(size=n))
    noise = rng.normal(size=n)
    for w in np.linspace(0.0, 1.0, 200):
        rtpl = np.sqrt(1 - w) * hpl + np.sqrt(w) * noise
        if spearman_correlation(rtpl, hpl) <= target:
            return rtpl, hpl
    return hpl.copy(), hpl


def test_spearman_zone_boundaries() -> None:
    # Green well above 0.80, red well below 0.70, amber in between — using nearly
    # identical distributions so KS stays green and the Spearman metric decides.
    for target, expected in [(0.95, PLAZone.GREEN), (0.75, PLAZone.AMBER), (0.5, PLAZone.RED)]:
        rtpl, hpl = _series_with_spearman(target)
        zone = pla_test(rtpl, hpl).spearman_zone
        assert zone is expected, f"spearman {spearman_correlation(rtpl, hpl):.3f} -> {zone}"


def test_ks_zone_classification_direct() -> None:
    from quantica.risk.frtb import _ks_zone, _spearman_zone

    assert _spearman_zone(0.80) is PLAZone.GREEN  # boundary is green
    assert _spearman_zone(0.79) is PLAZone.AMBER
    assert _spearman_zone(0.70) is PLAZone.AMBER  # boundary is amber
    assert _spearman_zone(0.699) is PLAZone.RED
    assert _ks_zone(0.09) is PLAZone.GREEN  # boundary is green
    assert _ks_zone(0.10) is PLAZone.AMBER
    assert _ks_zone(0.12) is PLAZone.AMBER  # boundary is amber
    assert _ks_zone(0.121) is PLAZone.RED


def test_overall_zone_is_the_worse_of_the_two() -> None:
    # Spearman green, KS red -> overall red (either-metric-red rule).
    rng = np.random.default_rng(9)
    hpl = rng.normal(0, 1, 400)
    rtpl = hpl + 5.0  # perfect ranks (Spearman 1) but shifted distribution (KS 1)
    result = pla_test(rtpl, hpl)
    assert result.spearman_zone is PLAZone.GREEN
    assert result.ks_zone is PLAZone.RED
    assert result.zone is PLAZone.RED
    assert not result.ima_eligible


# --------------------------------------------------------------------------- #
# 3. Known-truth books (the headline — reuse the gamma divergence)
# --------------------------------------------------------------------------- #


def test_near_linear_book_passes_green() -> None:
    scenarios = MarketScenarios.generate(N_DAYS, np.random.default_rng(1), spot_vol=0.0126)
    result = book_pla_test(deep_itm_book(), scenarios, rtpl_method="delta-normal")
    assert result.zone is PLAZone.GREEN
    assert result.ima_eligible
    assert "no PLA capital add-on" in result.capital_consequence()


def test_delta_gamma_model_spans_the_curvature_green() -> None:
    # A short-gamma book under LARGE moves, but the risk model carries gamma:
    # RTPL spans HPL and the desk passes.
    scenarios = MarketScenarios.generate(N_DAYS, np.random.default_rng(1), spot_vol=0.05)
    result = book_pla_test(straddle(-100.0), scenarios, rtpl_method="delta-gamma")
    assert result.zone is PLAZone.GREEN


def test_short_gamma_delta_only_desk_fails_red() -> None:
    # The textbook failure: a short-gamma desk whose risk model has only delta.
    # Under large moves the missing curvature makes RTPL fail to span HPL on BOTH
    # metrics -> red -> IMA ineligible.
    scenarios = MarketScenarios.generate(N_DAYS, np.random.default_rng(1), spot_vol=0.05)
    result = book_pla_test(straddle(-100.0), scenarios, rtpl_method="delta-normal")
    assert result.spearman_zone is PLAZone.RED
    assert result.ks_zone is PLAZone.RED
    assert result.zone is PLAZone.RED
    assert not result.ima_eligible
    assert "Standardised Approach" in result.capital_consequence()


def test_short_gamma_delta_only_small_moves_is_amber() -> None:
    # At small daily moves the ranking is still right (Spearman green) but the
    # distributions diverge enough for the KS metric to reach amber -> the desk
    # keeps IMA with a surcharge. The zone tightens as moves grow.
    scenarios = MarketScenarios.generate(N_DAYS, np.random.default_rng(1), spot_vol=0.007)
    result = book_pla_test(straddle(-100.0), scenarios, rtpl_method="delta-normal")
    assert result.spearman_zone is PLAZone.GREEN
    assert result.ks_zone is PLAZone.AMBER
    assert result.zone is PLAZone.AMBER
    assert result.ima_eligible


def test_pla_zone_worsens_monotonically_with_move_size() -> None:
    # The same short-gamma delta-only desk, larger daily moves -> weakly worse zone.
    severities = []
    for spot_vol in (0.004, 0.007, 0.02, 0.05):
        scenarios = MarketScenarios.generate(N_DAYS, np.random.default_rng(1), spot_vol=spot_vol)
        zone = book_pla_test(straddle(-100.0), scenarios, rtpl_method="delta-normal").zone
        severities.append({PLAZone.GREEN: 0, PLAZone.AMBER: 1, PLAZone.RED: 2}[zone])
    assert severities == sorted(severities)
    assert severities[0] == 0 and severities[-1] == 2  # spans green to red


# --------------------------------------------------------------------------- #
# 4. Consistency (no drift) and wiring
# --------------------------------------------------------------------------- #


def test_hpl_is_the_full_revaluation_pricing_path() -> None:
    scenarios = MarketScenarios.generate(50, np.random.default_rng(5), spot_vol=0.02)
    book = straddle(-100.0)
    # book_pla_test's HPL is exactly OptionBook.full_revaluation_pnl (no drift):
    hpl = book.full_revaluation_pnl(scenarios)
    rtpl = book.delta_gamma_pnl(scenarios)
    direct = pla_test(rtpl, hpl)
    via_book = book_pla_test(book, scenarios, rtpl_method="delta-gamma")
    assert via_book.spearman == direct.spearman
    assert via_book.ks_statistic == direct.ks_statistic


def test_result_reports_observation_count() -> None:
    scenarios = MarketScenarios.generate(N_DAYS, np.random.default_rng(1), spot_vol=0.0126)
    assert book_pla_test(deep_itm_book(), scenarios).n_observations == N_DAYS


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        pla_test(np.array([1.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="must match"):
        pla_test(np.zeros(5), np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        pla_test(np.array([1.0, np.nan, 2.0]), np.zeros(3))
    scenarios = MarketScenarios.generate(20, np.random.default_rng(1), spot_vol=0.02)
    with pytest.raises(ValueError, match="rtpl_method must be"):
        book_pla_test(straddle(-1.0), scenarios, rtpl_method="delta")  # type: ignore[arg-type]
