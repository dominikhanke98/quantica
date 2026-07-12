r"""Calibration tests validated against hand computations and direct scipy calls.

The size/power meta-study lives in ``test_size_power.py``; this file anchors the
*wiring*: exact binomial tail probabilities, the Jeffreys posterior against a
direct ``scipy.stats.beta`` call, Hosmer--Lemeshow on a hand-groupable case, and
the per-grade table's internal consistency.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from quantica.risk.credit import (
    assign_grades,
    binomial_test,
    calibration_curve,
    grade_calibration,
    hosmer_lemeshow,
    jeffreys_test,
)
from scipy.stats import beta, binom

# --------------------------------------------------------------------------- #
# Binomial test — exact tails
# --------------------------------------------------------------------------- #


def test_binomial_greater_matches_exact_tail() -> None:
    # P(X >= 12 | n=500, p=0.02) with X ~ Bin: straight from the survival function.
    res = binomial_test(12, 500, 0.02)
    assert res.p_value == pytest.approx(float(binom.sf(11, 500, 0.02)), abs=1e-15)
    assert res.observed_rate == pytest.approx(12 / 500)


def test_binomial_alternatives_and_directions() -> None:
    # More defaults than expected: 'greater' small, 'less' large.
    high = binomial_test(25, 500, 0.02, alternative="greater")
    assert high.p_value < 0.001
    assert binomial_test(25, 500, 0.02, alternative="less").p_value > 0.999
    two = binomial_test(25, 500, 0.02, alternative="two-sided")
    assert two.p_value == pytest.approx(min(1.0, 2.0 * high.p_value), abs=1e-15)
    # Zero defaults with a material PD: 'greater' cannot reject (p = 1).
    assert binomial_test(0, 500, 0.02).p_value == pytest.approx(1.0)


def test_binomial_reject_threshold() -> None:
    res = binomial_test(20, 500, 0.02)  # twice the expected count
    assert res.reject(0.05)
    assert not binomial_test(10, 500, 0.02).reject(0.05)  # spot-on expectation


# --------------------------------------------------------------------------- #
# Jeffreys test — posterior against scipy directly
# --------------------------------------------------------------------------- #


def test_jeffreys_matches_direct_beta_posterior() -> None:
    d, n, pd = 12, 500, 0.02
    res = jeffreys_test(d, n, pd)
    assert res.p_value == pytest.approx(float(beta.cdf(pd, d + 0.5, n - d + 0.5)), abs=1e-15)
    assert res.posterior_mean == pytest.approx((d + 0.5) / (n + 1.0), abs=1e-15)


def test_jeffreys_direction_more_defaults_smaller_p() -> None:
    p_values = [jeffreys_test(d, 500, 0.02).p_value for d in (5, 10, 15, 20, 25)]
    assert all(a > b for a, b in pairwise(p_values))


def test_jeffreys_is_defined_at_zero_defaults() -> None:
    # The half prior keeps the zero-default grade testable (a known advantage
    # over approaches that degenerate on low-default portfolios).
    res = jeffreys_test(0, 200, 0.01)
    assert 0.0 < res.p_value < 1.0
    assert not res.reject()


# --------------------------------------------------------------------------- #
# Hosmer--Lemeshow
# --------------------------------------------------------------------------- #


def test_hosmer_lemeshow_statistic_matches_hand_computation() -> None:
    # Two clean groups by construction: low scores 0.1 (n=100, 15 defaults),
    # high scores 0.4 (n=100, 35 defaults).
    y = np.r_[np.ones(15), np.zeros(85), np.ones(35), np.zeros(65)]
    s = np.r_[np.full(100, 0.1), np.full(100, 0.4)]
    res = hosmer_lemeshow(y, s, n_groups=2, dof=2)
    expected = (15 - 10.0) ** 2 / (10.0 * 0.9) + (35 - 40.0) ** 2 / (40.0 * 0.6)
    assert res.statistic == pytest.approx(expected, abs=1e-12)
    assert res.n_groups == 2


def test_hosmer_lemeshow_accepts_truth_and_rejects_distortion() -> None:
    # With TRUE (non-estimated) PDs the null is chi^2_G -> dof=n_groups; see the
    # size study for the full calibration of both dof conventions.
    rng = np.random.default_rng(11)
    p = 1.0 / (1.0 + np.exp(-rng.normal(-3.0, 1.0, 6000)))
    y = (rng.random(6000) < p).astype(float)
    assert not hosmer_lemeshow(y, p, dof=10).reject()
    understated = np.clip(p * 0.5, 1e-9, 1.0)
    assert hosmer_lemeshow(y, understated, dof=10).reject()


# --------------------------------------------------------------------------- #
# Grades and the per-grade table
# --------------------------------------------------------------------------- #


def test_assign_grades_quantile_populations() -> None:
    rng = np.random.default_rng(12)
    scores = rng.random(7000)
    grades = assign_grades(scores, n_grades=7)
    counts = np.bincount(grades.astype(int))
    assert counts.size == 7
    assert counts.min() > 900  # near-equal populations on a continuous score


def test_grade_calibration_table_is_internally_consistent() -> None:
    rng = np.random.default_rng(13)
    pd_scores = 1.0 / (1.0 + np.exp(-rng.normal(-3.0, 1.2, 8000)))
    y = (rng.random(8000) < pd_scores).astype(float)
    rows = grade_calibration(y, pd_scores, n_grades=5)
    assert sum(r.n_obligors for r in rows) == 8000
    assert sum(r.n_defaults for r in rows) == int(y.sum())
    for r in rows:
        assert r.observed_rate == pytest.approx(r.n_defaults / r.n_obligors)
        assert r.binomial_p == pytest.approx(
            binomial_test(r.n_defaults, r.n_obligors, r.mean_pd).p_value
        )
        assert r.jeffreys_p == pytest.approx(
            jeffreys_test(r.n_defaults, r.n_obligors, r.mean_pd).p_value
        )
    # Grades are ordered by risk: mean PD increases across grades.
    mean_pds = [r.mean_pd for r in rows]
    assert all(a < b for a, b in pairwise(mean_pds))


def test_grade_calibration_flags_understated_grades() -> None:
    # Halve the reported PDs: most grades should be flagged by both tests.
    rng = np.random.default_rng(14)
    true_pd = 1.0 / (1.0 + np.exp(-rng.normal(-3.0, 1.0, 10000)))
    y = (rng.random(10000) < true_pd).astype(float)
    rows = grade_calibration(y, true_pd * 0.5, n_grades=5)
    assert sum(r.jeffreys_p < 0.05 for r in rows) >= 4


def test_calibration_curve_tracks_observed_rates() -> None:
    rng = np.random.default_rng(15)
    p = 1.0 / (1.0 + np.exp(-rng.normal(-2.5, 1.0, 12000)))
    y = (rng.random(12000) < p).astype(float)
    curve = calibration_curve(y, p, n_bins=8)
    assert curve.counts.sum() == 12000
    # Well-calibrated: observed ~ predicted in every well-populated bin.
    np.testing.assert_allclose(curve.observed_rate, curve.mean_predicted, atol=0.03)


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="n_defaults"):
        binomial_test(-1, 100, 0.02)
    with pytest.raises(ValueError, match="pd must be"):
        binomial_test(2, 100, 0.0)
    with pytest.raises(ValueError, match="alternative"):
        binomial_test(2, 100, 0.02, alternative="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_defaults"):
        jeffreys_test(101, 100, 0.02)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="n_groups"):
        hosmer_lemeshow(y, np.array([0.1, 0.2, 0.3, 0.4]), n_groups=1)
    with pytest.raises(ValueError, match="n_grades"):
        assign_grades(np.arange(10.0), n_grades=1)
    with pytest.raises(ValueError, match="grades shape"):
        grade_calibration(y, np.array([0.1, 0.2, 0.3, 0.4]), grades=np.zeros(3))
