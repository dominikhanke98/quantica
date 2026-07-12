"""PSI / characteristic stability validated on hand computations and known drifts."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from quantica.risk.credit import StabilityBand, characteristic_stability, psi


def test_psi_matches_hand_computation() -> None:
    # Two bins split at the expected sample's median: expected 50/50, actual
    # 70/30 -> PSI = 0.2*ln(0.7/0.5) + (-0.2)*ln(0.3/0.5) = 0.169457...
    expected = np.r_[np.zeros(5), np.ones(5)]
    actual = np.r_[np.zeros(7), np.ones(3)]
    res = psi(expected, actual, n_bins=2)
    hand = 0.2 * np.log(0.7 / 0.5) - 0.2 * np.log(0.3 / 0.5)
    assert res.value == pytest.approx(float(hand), abs=1e-12)
    assert res.band is StabilityBand.MONITOR  # 0.10 <= 0.169 < 0.25


def test_psi_is_near_zero_for_identical_populations() -> None:
    rng = np.random.default_rng(0)
    expected = rng.normal(0, 1, 20000)
    actual = rng.normal(0, 1, 20000)
    res = psi(expected, actual)
    assert res.value < 0.01
    assert res.band is StabilityBand.STABLE


def test_psi_increases_monotonically_with_drift() -> None:
    rng = np.random.default_rng(1)
    expected = rng.normal(0, 1, 20000)
    values = [psi(expected, rng.normal(shift, 1, 20000)).value for shift in (0.0, 0.2, 0.5, 1.0)]
    assert all(a < b for a, b in pairwise(values))
    assert psi(expected, rng.normal(1.0, 1, 20000)).band is StabilityBand.SHIFTED


def test_psi_is_symmetric_in_form() -> None:
    # The PSI summand is symmetric in (p_e, p_a); with equal sample sizes and
    # swapped roles the value changes only through the bin edges (quantiles of
    # the expected sample), so a mild shift gives close-but-not-equal values.
    rng = np.random.default_rng(2)
    a = rng.normal(0, 1, 30000)
    b = rng.normal(0.3, 1, 30000)
    assert psi(a, b).value == pytest.approx(psi(b, a).value, rel=0.15)


def test_characteristic_stability_flags_only_the_drifted_feature() -> None:
    rng = np.random.default_rng(3)
    expected = rng.normal(0, 1, (15000, 3))
    actual = rng.normal(0, 1, (15000, 3))
    actual[:, 1] += 0.6  # drift the middle feature only
    rows = characteristic_stability(expected, actual, ("a", "b", "c"))
    assert rows[0].psi.band is StabilityBand.STABLE
    assert rows[1].psi.band is not StabilityBand.STABLE
    assert rows[2].psi.band is StabilityBand.STABLE
    assert rows[1].psi.value > 5.0 * max(rows[0].psi.value, rows[2].psi.value)


def test_input_validation() -> None:
    with pytest.raises(ValueError, match="non-empty 1-D"):
        psi(np.zeros((2, 2)), np.zeros(4))
    with pytest.raises(ValueError, match="n_bins"):
        psi(np.arange(10.0), np.arange(10.0), n_bins=1)
    with pytest.raises(ValueError, match="matching column"):
        characteristic_stability(np.zeros((5, 2)), np.zeros((5, 3)), ("a", "b"))
    with pytest.raises(ValueError, match="names"):
        characteristic_stability(np.zeros((5, 2)), np.zeros((5, 2)), ("a",))
