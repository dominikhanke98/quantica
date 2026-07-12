r"""Fairness checks validated on hand examples and the planted base-rate tension.

The centerpiece integration test uses the *true* generative PDs on a book with a
planted group base-rate difference: perfectly calibrated within each group by
construction, yet failing the four-fifths approval-rate convention — the
impossibility trade-off (Chouldechova 2017) demonstrated on known ground truth,
where it is unambiguous that the disparity is a base-rate fact and not a model
defect.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.risk.credit import generate_credit_portfolio
from quantica.risk.ml_validation import disparate_impact, group_calibration
from scipy.stats import beta

# --------------------------------------------------------------------------- #
# Disparate impact
# --------------------------------------------------------------------------- #


def test_disparate_impact_hand_example() -> None:
    # Protected group: 2 of 4 approved (PD <= 0.05); reference: 4 of 5 approved.
    scores = np.array([0.01, 0.02, 0.10, 0.20, 0.01, 0.02, 0.03, 0.04, 0.30])
    group = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    di = disparate_impact(scores, group, pd_threshold=0.05)
    assert di.protected_approval_rate == pytest.approx(0.5)
    assert di.reference_approval_rate == pytest.approx(0.8)
    assert di.ratio == pytest.approx(0.625)
    assert not di.passes_four_fifths
    assert di.n_protected == 4 and di.n_reference == 5


def test_disparate_impact_passes_when_rates_match() -> None:
    scores = np.array([0.01, 0.10, 0.01, 0.10])
    group = np.array([1.0, 1.0, 0.0, 0.0])
    di = disparate_impact(scores, group, pd_threshold=0.05)
    assert di.ratio == pytest.approx(1.0)
    assert di.passes_four_fifths


# --------------------------------------------------------------------------- #
# Calibration within group
# --------------------------------------------------------------------------- #


def test_group_calibration_matches_direct_beta_computation() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    group = (rng.random(n) < 0.4).astype(float)
    pd_scores = np.full(n, 0.05)
    y = (rng.random(n) < 0.05).astype(float)
    rows = group_calibration(y, pd_scores, group)
    assert len(rows) == 2
    for r in rows:
        lower = float(beta.cdf(r.mean_pd, r.n_defaults + 0.5, r.n_obligors - r.n_defaults + 0.5))
        expected = min(1.0, 2.0 * min(lower, 1.0 - lower))
        assert r.jeffreys_two_sided_p == pytest.approx(expected, abs=1e-15)
        # (No reject() assertion here: a correctly calibrated group is still
        # rejected ~5% of the time by construction — that is what size means.)


def test_group_calibration_flags_a_miscalibrated_group() -> None:
    rng = np.random.default_rng(1)
    n = 4000
    group = (rng.random(n) < 0.4).astype(float)
    true_pd = np.where(group == 1.0, 0.10, 0.05)  # protected group riskier ...
    y = (rng.random(n) < true_pd).astype(float)
    scores = np.full(n, 0.05)  # ... but scored as if not
    rows = {r.group: r for r in group_calibration(y, scores, group)}
    assert rows[1].reject()  # understated for the protected group
    assert not rows[0].reject()


def test_true_pds_are_calibrated_within_group_yet_fail_four_fifths() -> None:
    # The impossibility trade-off on known ground truth: with a planted group
    # base-rate difference, even the TRUE PDs are calibrated within each group
    # (they are the generative probabilities) yet fail the approval-rate ratio.
    # The disparity is a base-rate fact, not a model defect — which is exactly
    # why the fairness-metric choice must be documented, not defaulted.
    sample = generate_credit_portfolio(30_000, np.random.default_rng(42), group_effect=0.8)
    rows = group_calibration(sample.defaults, sample.true_pd, sample.group)
    assert all(r.jeffreys_two_sided_p > 0.2 for r in rows)  # calibrated in-group
    di = disparate_impact(sample.true_pd, sample.group, pd_threshold=0.05)
    assert di.ratio < 0.8  # yet the four-fifths convention fails
    # And the planted base-rate difference is real:
    by_group = {r.group: r.observed_rate for r in rows}
    assert by_group[1] > 2.0 * by_group[0]


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def test_input_validation() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="both groups"):
        disparate_impact(scores, np.zeros(4), pd_threshold=0.05)
    with pytest.raises(ValueError, match="pd_threshold"):
        disparate_impact(scores, np.array([0.0, 1.0, 0.0, 1.0]), pd_threshold=0.0)
    with pytest.raises(ValueError, match="matching 1-D"):
        disparate_impact(scores, np.array([0.0, 1.0]), pd_threshold=0.05)
    y = np.array([0.0, 1.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="group shape"):
        group_calibration(y, scores, np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="only 0 and 1"):
        group_calibration(y, scores, np.array([0.0, 1.0, 2.0, 1.0]))
