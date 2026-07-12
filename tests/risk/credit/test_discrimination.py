r"""Discrimination metrics validated against independent computations and anchors.

- **AUC three independent ways** (the effective challenge): the Mann--Whitney rank
  form, trapezoidal integration of our own ROC curve, and scikit-learn's
  ``roc_auc_score`` — all three must agree to machine precision, ties included.
- **Analytic binormal anchor**: goods ~ N(0,1), bads ~ N(δ,1) gives
  AUC = Φ(δ/√2) and KS = 2Φ(δ/2) - 1 exactly; large-sample estimates must land
  within sampling error.
- **KS** cross-checked against ``scipy.stats.ks_2samp`` and a brute-force scan of
  the definition.
- **Bootstrap CIs**: stratified, seeded (deterministic), containing the point
  estimate, and shrinking with sample size.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.risk.credit import (
    auc,
    bootstrap_ci,
    discrimination_report,
    gini,
    ks_statistic,
    roc_curve,
)
from scipy.stats import ks_2samp, norm

sklearn_metrics = pytest.importorskip("sklearn.metrics")


def binormal_sample(
    delta: float, n_good: int, n_bad: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scores = np.r_[rng.normal(0.0, 1.0, n_good), rng.normal(delta, 1.0, n_bad)]
    y = np.r_[np.zeros(n_good), np.ones(n_bad)]
    return y, scores


# --------------------------------------------------------------------------- #
# AUC: three ways + analytic anchor
# --------------------------------------------------------------------------- #


def test_auc_three_independent_ways_agree() -> None:
    y, s = binormal_sample(1.2, 5000, 400, seed=0)
    rank_auc = auc(y, s)
    fpr, tpr = roc_curve(y, s)
    trapezoid_auc = float(np.trapezoid(tpr, fpr))
    sklearn_auc = float(sklearn_metrics.roc_auc_score(y, s))
    assert rank_auc == pytest.approx(trapezoid_auc, abs=1e-12)
    assert rank_auc == pytest.approx(sklearn_auc, abs=1e-12)


def test_auc_three_ways_agree_under_heavy_ties() -> None:
    # Integer-valued scores force massive tie groups; the tie-aware rank form,
    # our tie-merged ROC curve, and sklearn must still agree exactly.
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.1).astype(float)
    s = np.floor(rng.random(4000) * 8) + y  # 8 levels, defaulters shifted up one
    rank_auc = auc(y, s)
    fpr, tpr = roc_curve(y, s)
    assert rank_auc == pytest.approx(float(np.trapezoid(tpr, fpr)), abs=1e-12)
    assert rank_auc == pytest.approx(float(sklearn_metrics.roc_auc_score(y, s)), abs=1e-12)


def test_auc_matches_binormal_closed_form() -> None:
    delta = 1.2
    y, s = binormal_sample(delta, 40000, 2000, seed=2)
    # SE of AUC at these sizes is ~5e-3; allow ~3 SE.
    assert auc(y, s) == pytest.approx(float(norm.cdf(delta / np.sqrt(2.0))), abs=0.02)


def test_auc_known_edge_cases() -> None:
    y = np.array([0.0, 0.0, 1.0, 1.0])
    assert auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0  # perfect separation
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0  # perfectly inverted
    assert auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # constant score: ties
    # Hand-computed: pairs (bad, good): (.7>.1),(.7>.4),(.3>.1),(.3<.4) -> 3/4.
    assert auc(y, np.array([0.1, 0.4, 0.7, 0.3])) == pytest.approx(0.75)


def test_gini_is_two_auc_minus_one() -> None:
    y, s = binormal_sample(0.9, 3000, 300, seed=3)
    assert gini(y, s) == pytest.approx(2.0 * auc(y, s) - 1.0, abs=1e-14)


# --------------------------------------------------------------------------- #
# KS: brute force, scipy, analytic anchor
# --------------------------------------------------------------------------- #


def test_ks_matches_brute_force_definition() -> None:
    y, s = binormal_sample(1.0, 300, 60, seed=4)
    bad, good = s[y == 1.0], s[y == 0.0]
    brute = max(abs(float(np.mean(bad <= t)) - float(np.mean(good <= t))) for t in np.unique(s))
    assert ks_statistic(y, s) == pytest.approx(brute, abs=1e-14)


def test_ks_matches_scipy_including_ties() -> None:
    rng = np.random.default_rng(5)
    y = (rng.random(2000) < 0.15).astype(float)
    s = np.round(rng.normal(0, 1, 2000) + 0.8 * y, 1)  # rounded -> ties
    scipy_ks = float(ks_2samp(s[y == 1.0], s[y == 0.0], method="asymp").statistic)
    assert ks_statistic(y, s) == pytest.approx(scipy_ks, abs=1e-12)


def test_ks_matches_binormal_closed_form() -> None:
    delta = 1.2
    y, s = binormal_sample(delta, 40000, 2000, seed=6)
    # KS_true = 2*Phi(delta/2) - 1; the empirical max has a small positive bias,
    # so the tolerance is looser than the AUC anchor's.
    assert ks_statistic(y, s) == pytest.approx(2.0 * float(norm.cdf(delta / 2.0)) - 1.0, abs=0.03)


# --------------------------------------------------------------------------- #
# Bootstrap CIs
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_is_seeded_and_contains_point() -> None:
    y, s = binormal_sample(1.0, 1500, 150, seed=7)
    a = bootstrap_ci(y, s, auc, np.random.default_rng(0), n_boot=400)
    b = bootstrap_ci(y, s, auc, np.random.default_rng(0), n_boot=400)
    assert (a.lower, a.upper) == (b.lower, b.upper)  # deterministic
    assert a.lower < a.point < a.upper
    assert a.point == pytest.approx(auc(y, s), abs=1e-14)


def test_bootstrap_ci_narrows_with_sample_size() -> None:
    y_small, s_small = binormal_sample(1.0, 800, 80, seed=8)
    y_big, s_big = binormal_sample(1.0, 8000, 800, seed=9)
    ci_small = bootstrap_ci(y_small, s_small, auc, np.random.default_rng(1), n_boot=400)
    ci_big = bootstrap_ci(y_big, s_big, auc, np.random.default_rng(1), n_boot=400)
    assert (ci_big.upper - ci_big.lower) < 0.6 * (ci_small.upper - ci_small.lower)


def test_discrimination_report_bundles_all_three() -> None:
    y, s = binormal_sample(1.0, 1200, 120, seed=10)
    rep = discrimination_report(y, s, np.random.default_rng(2), n_boot=200)
    assert rep.n_obligors == 1320 and rep.n_defaults == 120
    assert rep.auc.point == pytest.approx(auc(y, s))
    assert rep.gini.point == pytest.approx(gini(y, s))
    assert rep.ks.point == pytest.approx(ks_statistic(y, s))


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_input_validation() -> None:
    ones = np.ones(4)
    with pytest.raises(ValueError, match="both classes"):
        auc(ones, np.arange(4.0))
    with pytest.raises(ValueError, match="only 0"):
        auc(np.array([0.0, 2.0]), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="shape"):
        auc(np.array([0.0, 1.0]), np.arange(3.0))
    with pytest.raises(ValueError, match="finite"):
        auc(np.array([0.0, 1.0]), np.array([0.1, np.nan]))
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="level must be"):
        bootstrap_ci(y, np.arange(4.0), auc, np.random.default_rng(0), level=1.0)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(y, np.arange(4.0), auc, np.random.default_rng(0), n_boot=1)
