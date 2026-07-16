"""Validation of the backtest-validity layer (DSR / PSR / PBO / MinTRL).

Two kinds of check. **Closed-form anchors**: PSR, the expected-maximum-Sharpe formula
and MinTRL are each pinned to an independent computation (scipy's normal CDF/PPF), and
MinTRL is validated by an exact round-trip through PSR. **Known-truth discrimination**
(the headline): a deliberately overfit noise search is flagged spurious — the deflated
Sharpe never reaches significance and PBO sits near 0.5 — while a genuinely predictive
planted signal survives both (DSR ≈ 1, PBO ≈ 0). That is the proof the overfitting
detector detects overfitting.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.portfolio.data import generate_trial_returns
from quantica.portfolio.overfitting import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_trials,
    expected_maximum_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from scipy.stats import norm

_EULER = 0.5772156649015329


# --------------------------------------------------------------------------- #
# Sharpe ratio
# --------------------------------------------------------------------------- #


def test_sharpe_matches_definition() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.01, 0.05, size=200)
    expected = np.mean(r) / np.std(r, ddof=1)
    assert np.isclose(sharpe_ratio(r), expected, atol=1e-12)
    assert np.isclose(sharpe_ratio(r, periods_per_year=12), expected * np.sqrt(12), atol=1e-12)


def test_sharpe_zero_variance_is_zero() -> None:
    assert sharpe_ratio(np.full(10, 0.01)) == 0.0


# --------------------------------------------------------------------------- #
# Probabilistic Sharpe ratio
# --------------------------------------------------------------------------- #


def test_psr_equals_half_at_benchmark() -> None:
    """PSR is exactly 0.5 when observed equals the benchmark, for any n/moments."""
    assert np.isclose(probabilistic_sharpe_ratio(0.2, 500, benchmark_sr=0.2), 0.5, atol=1e-12)
    assert np.isclose(
        probabilistic_sharpe_ratio(0.2, 50, benchmark_sr=0.2, skew=-0.5, kurt=6.0), 0.5, atol=1e-12
    )


def test_psr_matches_closed_form_zscore() -> None:
    """PSR equals Φ of the independently computed z-score (the anchor)."""
    sr, n, skew, kurt = 0.1, 101, 0.0, 3.0
    var_factor = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr**2
    z = sr * np.sqrt(n - 1) / np.sqrt(var_factor)
    assert np.isclose(
        probabilistic_sharpe_ratio(sr, n, skew=skew, kurt=kurt), norm.cdf(z), atol=1e-12
    )


def test_psr_increases_with_track_record_and_sharpe() -> None:
    short = probabilistic_sharpe_ratio(0.15, 50)
    long = probabilistic_sharpe_ratio(0.15, 500)
    assert long > short  # more data → more confident
    higher = probabilistic_sharpe_ratio(0.30, 100)
    lower = probabilistic_sharpe_ratio(0.15, 100)
    assert higher > lower


def test_psr_penalises_negative_skew_and_fat_tails() -> None:
    """Non-normality inflates the Sharpe estimator's variance, lowering PSR."""
    normal = probabilistic_sharpe_ratio(0.2, 120, skew=0.0, kurt=3.0)
    ugly = probabilistic_sharpe_ratio(0.2, 120, skew=-1.0, kurt=8.0)
    assert ugly < normal


# --------------------------------------------------------------------------- #
# Expected maximum Sharpe / deflated Sharpe
# --------------------------------------------------------------------------- #


def test_expected_maximum_sharpe_matches_formula() -> None:
    n_trials, var = 20, 0.04
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected = np.sqrt(var) * ((1.0 - _EULER) * z1 + _EULER * z2)
    assert np.isclose(expected_maximum_sharpe(n_trials, var), expected, atol=1e-12)


def test_expected_maximum_sharpe_grows_with_trials() -> None:
    """More trials → higher luckiest-of-N Sharpe (the multiple-testing penalty)."""
    assert expected_maximum_sharpe(1000, 0.04) > expected_maximum_sharpe(10, 0.04)


def test_expected_maximum_sharpe_zero_variance_is_zero() -> None:
    assert expected_maximum_sharpe(50, 0.0) == 0.0


def test_deflated_sharpe_is_psr_at_expected_max() -> None:
    """DSR is exactly PSR evaluated at the expected-maximum-Sharpe benchmark."""
    benchmark = expected_maximum_sharpe(30, 0.03)
    direct = probabilistic_sharpe_ratio(0.5, 200, benchmark_sr=benchmark, skew=-0.3, kurt=5.0)
    dsr = deflated_sharpe_ratio(0.5, n_obs=200, n_trials=30, sr_variance=0.03, skew=-0.3, kurt=5.0)
    assert np.isclose(dsr, direct, atol=1e-12)


# --------------------------------------------------------------------------- #
# Minimum track record length
# --------------------------------------------------------------------------- #


def test_min_trl_round_trips_through_psr() -> None:
    """PSR evaluated at n = MinTRL returns exactly the target confidence."""
    sr, skew, kurt, conf = 0.15, -0.4, 5.0, 0.95
    n = minimum_track_record_length(sr, skew=skew, kurt=kurt, confidence=conf)
    # Evaluate PSR at the exact (fractional) MinTRL: it returns the target confidence.
    assert np.isclose(probabilistic_sharpe_ratio(sr, n, skew=skew, kurt=kurt), conf, atol=1e-9)


def test_min_trl_requires_edge_over_benchmark() -> None:
    with pytest.raises(ValueError, match="must exceed"):
        minimum_track_record_length(0.1, benchmark_sr=0.1)


def test_min_trl_grows_as_edge_shrinks() -> None:
    """A smaller Sharpe edge needs a longer track record to prove out."""
    assert minimum_track_record_length(0.05) > minimum_track_record_length(0.20)


# --------------------------------------------------------------------------- #
# Probability of backtest overfitting
# --------------------------------------------------------------------------- #


def test_pbo_rejects_odd_splits() -> None:
    rng = np.random.default_rng(0)
    r = generate_trial_returns(100, 10, rng).returns
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(r, n_splits=7)


def test_pbo_number_of_combinations() -> None:
    from math import comb

    rng = np.random.default_rng(0)
    r = generate_trial_returns(120, 8, rng).returns
    result = probability_of_backtest_overfitting(r, n_splits=8)
    assert result.n_combinations == comb(8, 4)


def test_pbo_near_half_for_pure_noise() -> None:
    """Averaged over realisations, the IS-best noise strategy is OOS-random (PBO≈0.5)."""
    pbos = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        r = generate_trial_returns(360, 50, rng, planted_sharpe=0.0).returns
        pbos.append(probability_of_backtest_overfitting(r, n_splits=10).pbo)
    assert 0.4 <= np.mean(pbos) <= 0.6


def test_pbo_near_zero_for_a_dominant_signal() -> None:
    rng = np.random.default_rng(1)
    r = generate_trial_returns(360, 50, rng, planted_sharpe=0.35).returns
    assert probability_of_backtest_overfitting(r, n_splits=10).pbo < 0.05


# --------------------------------------------------------------------------- #
# The headline: known-truth overfitting discrimination
# --------------------------------------------------------------------------- #


def test_noise_search_is_flagged_spurious() -> None:
    """The best of many pure-noise strategies is never significant after deflation."""
    for seed in range(6):
        rng = np.random.default_rng(seed)
        noise = generate_trial_returns(360, 50, rng, planted_sharpe=0.0)
        result = deflated_sharpe_ratio_from_trials(noise.returns)
        assert not result.is_significant  # DSR < 0.95 every time


def test_planted_signal_survives_both_detectors() -> None:
    """A genuine edge clears the deflated Sharpe and drives PBO to zero."""
    rng = np.random.default_rng(1)
    planted = generate_trial_returns(360, 50, rng, planted_sharpe=0.35)
    dsr = deflated_sharpe_ratio_from_trials(planted.returns)
    pbo = probability_of_backtest_overfitting(planted.returns, n_splits=10)
    assert dsr.selected == planted.planted_index  # the search finds the real one
    assert dsr.is_significant  # DSR >= 0.95
    assert pbo.pbo < 0.05


def test_signal_beats_noise_on_both_metrics() -> None:
    """The paired comparison: signal has higher DSR and lower PBO than noise."""
    rng_n = np.random.default_rng(3)
    rng_s = np.random.default_rng(3)
    noise = generate_trial_returns(360, 50, rng_n, planted_sharpe=0.0)
    signal = generate_trial_returns(360, 50, rng_s, planted_sharpe=0.35)
    dsr_noise = deflated_sharpe_ratio_from_trials(noise.returns)
    dsr_signal = deflated_sharpe_ratio_from_trials(signal.returns)
    pbo_noise = probability_of_backtest_overfitting(noise.returns, n_splits=10)
    pbo_signal = probability_of_backtest_overfitting(signal.returns, n_splits=10)
    assert dsr_signal.dsr > dsr_noise.dsr
    assert pbo_signal.pbo < pbo_noise.pbo
