"""Validation of the cointegration tests (numerical-validation skill).

The effective-challenge core: the tests must **detect genuine cointegration and reject
spurious pairs** (independent random walks), the exact failure mode that sinks naive pairs
trading. Beyond the single-sample known-truth checks, the *validator itself is validated* by
its size (false-positive rate on spurious pairs) and power (detection rate on real ones).
Both procedures are anchored to ``statsmodels`` to machine precision: Engle--Granger to
``coint`` (statistic and MacKinnon p-value), Johansen to ``coint_johansen`` (eigenvalues,
trace and maximum-eigenvalue statistics), with the embedded critical-value tables checked
against the library's own.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.statarb import (
    engle_granger,
    generate_cointegrated_pair,
    generate_independent_random_walks,
    johansen,
)

# --------------------------------------------------------------------------- #
# Engle--Granger
# --------------------------------------------------------------------------- #


def test_engle_granger_detects_cointegration_and_recovers_hedge_ratio() -> None:
    """On a genuinely cointegrated pair EG rejects the null and recovers the hedge ratio."""
    y, x = generate_cointegrated_pair(400, np.random.default_rng(0), beta=1.5, spread_kappa=0.1)
    result = engle_granger(y, x)
    assert result.is_cointegrated(0.05)
    assert result.pvalue < 0.05
    assert abs(result.hedge_ratio - 1.5) < 0.1  # true beta
    assert result.spread.shape == y.shape


def test_engle_granger_rejects_spurious_pair() -> None:
    """On independent random walks EG does not reject the null of no cointegration."""
    walks = generate_independent_random_walks(400, 2, np.random.default_rng(1))
    result = engle_granger(walks[:, 0], walks[:, 1])
    assert not result.is_cointegrated(0.05)


def test_engle_granger_anchors_to_statsmodels_coint() -> None:
    """The EG statistic and MacKinnon p-value match ``statsmodels.tsa.stattools.coint``."""
    coint = pytest.importorskip("statsmodels.tsa.stattools").coint
    y, x = generate_cointegrated_pair(300, np.random.default_rng(2))
    result = engle_granger(y, x, trend="c", autolag="AIC")
    sm_stat, sm_pvalue, _ = coint(y, x, trend="c", autolag="AIC")
    assert np.isclose(result.adf_stat, sm_stat, atol=1e-8)
    assert np.isclose(result.pvalue, sm_pvalue, atol=1e-8)


# --------------------------------------------------------------------------- #
# Johansen
# --------------------------------------------------------------------------- #


def test_johansen_detects_cointegration_rank() -> None:
    """A cointegrated pair has rank 1 under both the trace and max-eigenvalue tests."""
    y, x = generate_cointegrated_pair(400, np.random.default_rng(3), beta=1.2, spread_kappa=0.15)
    result = johansen(np.column_stack([y, x]))
    assert result.rank(0.05, statistic="trace") == 1
    assert result.rank(0.05, statistic="max_eig") == 1


def test_johansen_rejects_spurious_pair() -> None:
    """Independent random walks have inferred rank 0 (no cointegration)."""
    walks = generate_independent_random_walks(400, 2, np.random.default_rng(4))
    assert johansen(walks).rank(0.05) == 0


def test_johansen_worked_example_three_series_two_trends() -> None:
    """Three series driven by two common trends have exactly one cointegrating relation."""
    rng = np.random.default_rng(5)
    t1 = np.cumsum(rng.standard_normal(500))
    t2 = np.cumsum(rng.standard_normal(500))
    s1 = t1 + rng.standard_normal(500) * 0.3
    s2 = t1 + rng.standard_normal(500) * 0.3  # shares t1 with s1 -> one relation
    s3 = t2 + rng.standard_normal(500) * 0.3  # its own trend
    result = johansen(np.column_stack([s1, s2, s3]))
    assert result.rank(0.05) == 1  # n - (common trends) = 3 - 2


def test_johansen_anchors_to_statsmodels() -> None:
    """Eigenvalues, trace and max-eigenvalue statistics match ``coint_johansen`` exactly."""
    vecm = pytest.importorskip("statsmodels.tsa.vector_ar.vecm")
    y, x = generate_cointegrated_pair(350, np.random.default_rng(6))
    data = np.column_stack([y, x])
    for det_order in (-1, 0, 1):
        result = johansen(data, det_order=det_order, k_ar_diff=1)
        reference = vecm.coint_johansen(data, det_order, 1)
        assert np.allclose(result.eigenvalues, reference.eig, atol=1e-9)
        assert np.allclose(result.trace_stats, reference.lr1, atol=1e-8)
        assert np.allclose(result.max_eig_stats, reference.lr2, atol=1e-8)


def test_johansen_critical_values_match_published_tables() -> None:
    """The embedded Osterwald--Lenum tables agree with statsmodels' (transcription guard)."""
    vecm = pytest.importorskip("statsmodels.tsa.vector_ar.vecm")
    data = generate_independent_random_walks(400, 3, np.random.default_rng(7))
    for det_order in (-1, 0, 1):
        result = johansen(data, det_order=det_order)
        reference = vecm.coint_johansen(data, det_order, 1)
        # Embedded values are the published 2-decimal tables; statsmodels carries a
        # slightly higher-precision response surface, so they agree to ~0.01.
        assert np.allclose(result.trace_crit_values, reference.cvt, atol=0.02)
        assert np.allclose(result.max_eig_crit_values, reference.cvm, atol=0.02)


# --------------------------------------------------------------------------- #
# Validate the validator: size and power on known truth
# --------------------------------------------------------------------------- #


def test_size_and_power_on_known_truth() -> None:
    """EG and Johansen have high power on real pairs and controlled size on spurious ones.

    Power = detection rate when cointegration is present; size = false-positive rate when
    it is not. **Honest finding:** the Engle--Granger test is well-sized (~5-7% at the 5%
    level), while the Johansen trace test *over-rejects* in finite samples (~10-15%) — a
    documented small-sample bias (Reimers 1992), which is exactly the kind of thing a
    validate-the-validator study exists to surface.
    """
    n_trials = 100
    eg_power = eg_size = jo_power = jo_size = 0
    for i in range(n_trials):
        y, x = generate_cointegrated_pair(
            300, np.random.default_rng(1000 + i), beta=1.2, spread_kappa=0.15
        )
        if engle_granger(y, x).is_cointegrated(0.05):
            eg_power += 1
        if johansen(np.column_stack([y, x])).rank(0.05) >= 1:
            jo_power += 1
        walks = generate_independent_random_walks(300, 2, np.random.default_rng(5000 + i))
        if engle_granger(walks[:, 0], walks[:, 1]).is_cointegrated(0.05):
            eg_size += 1
        if johansen(walks).rank(0.05) >= 1:
            jo_size += 1

    assert eg_power / n_trials > 0.90  # high power
    assert jo_power / n_trials > 0.95
    assert eg_size / n_trials < 0.12  # well-sized around the nominal 5%
    assert jo_size / n_trials < 0.25  # over-rejects, but still mostly rejects the null


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #


def test_rejects_bad_inputs() -> None:
    """Mismatched lengths, wrong dimensions and unsupported options are rejected."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="equal length"):
        engle_granger(rng.standard_normal(100), rng.standard_normal(99))
    with pytest.raises(ValueError, match="trend"):
        engle_granger(rng.standard_normal(100), rng.standard_normal(100), trend="x")
    with pytest.raises(ValueError, match="at least two columns"):
        johansen(rng.standard_normal((100, 1)))
    with pytest.raises(ValueError, match="det_order"):
        johansen(rng.standard_normal((100, 2)), det_order=2)
