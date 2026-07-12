"""Backtest statistics — correctness on hand-checkable cases."""

from __future__ import annotations

import numpy as np
import pytest
from quantica.risk import (
    BaselZone,
    acerbi_szekely,
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    exceptions,
    kupiec_pof,
)
from scipy.stats import chi2, norm

# --------------------------------------------------------------------------- #
# Kupiec
# --------------------------------------------------------------------------- #


def test_kupiec_zero_exceptions_closed_form() -> None:
    # With x = 0 the LR reduces to -2 n ln(1-p); check that exact value.
    res = kupiec_pof(0, 250, 0.99)
    expected_lr = -2.0 * 250 * np.log(0.99)
    assert res.statistic == pytest.approx(expected_lr)
    assert res.p_value == pytest.approx(float(chi2.sf(expected_lr, 1)))


def test_kupiec_perfect_coverage_gives_zero_statistic() -> None:
    # Observed rate equal to the expected rate -> LR = 0, do not reject.
    res = kupiec_pof(10, 1000, 0.99)  # 1% of 1000
    assert res.statistic == pytest.approx(0.0, abs=1e-12)
    assert not res.reject()


def test_kupiec_too_many_exceptions_rejects() -> None:
    res = kupiec_pof(25, 250, 0.99)  # expected 2.5, observed 25
    assert res.observed_rate == pytest.approx(0.1)
    assert res.reject()


def test_kupiec_input_validation() -> None:
    with pytest.raises(ValueError, match="level must be in"):
        kupiec_pof(1, 10, 1.0)
    with pytest.raises(ValueError):
        kupiec_pof(11, 10, 0.99)


# --------------------------------------------------------------------------- #
# Christoffersen
# --------------------------------------------------------------------------- #


def test_christoffersen_independence_flags_clustering() -> None:
    # A clustered hit series (exceptions arrive back-to-back) should reject
    # independence; a spread-out one with the same count should not.
    clustered = np.zeros(200)
    clustered[50:60] = 1.0  # 10 consecutive exceptions
    spread = np.zeros(200)
    spread[::20] = 1.0  # 10 exceptions, evenly spaced
    assert christoffersen_independence(clustered).reject()
    assert not christoffersen_independence(spread).reject()


def test_christoffersen_cc_is_sum_of_uc_and_independence() -> None:
    rng = np.random.default_rng(0)
    hits = (rng.random(500) < 0.02).astype(float)
    uc = kupiec_pof(int(hits.sum()), hits.size, 0.99).statistic
    ind = christoffersen_independence(hits).statistic
    cc = christoffersen_cc(hits, 0.99)
    assert cc.statistic == pytest.approx(uc + ind)
    assert cc.dof == 2


# --------------------------------------------------------------------------- #
# Basel traffic light
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "x, zone, addon",
    [
        (2, BaselZone.GREEN, 0.0),
        (4, BaselZone.GREEN, 0.0),
        (5, BaselZone.YELLOW, 0.40),
        (7, BaselZone.YELLOW, 0.65),
        (9, BaselZone.YELLOW, 0.85),
        (12, BaselZone.RED, 1.0),
    ],
)
def test_basel_zones_250_days(x: int, zone: BaselZone, addon: float) -> None:
    res = basel_traffic_light(x, n_obs=250, level=0.99)
    assert res.zone is zone
    assert res.multiplier_addon == pytest.approx(addon)
    assert 0.0 <= res.cumulative_probability <= 1.0


# --------------------------------------------------------------------------- #
# Acerbi--Székely ES backtest
# --------------------------------------------------------------------------- #

_LEVEL = 0.975
_VAR = float(norm.ppf(_LEVEL))
_ES = float(norm.pdf(norm.ppf(_LEVEL)) / (1.0 - _LEVEL))


def test_acerbi_szekely_near_zero_under_correct_model() -> None:
    rng = np.random.default_rng(0)
    losses = rng.normal(0.0, 1.0, 200_000)  # correct model
    res = acerbi_szekely(losses, _VAR, _ES, _LEVEL, method="Z2")
    assert res.statistic == pytest.approx(0.0, abs=0.03)


def test_acerbi_szekely_positive_when_es_underestimated() -> None:
    # Heavier tail than the model, with the SAME VaR: the ES is understated, so Z2
    # is clearly positive even though a VaR count test would be fooled.
    rng = np.random.default_rng(1)
    from scipy.stats import t as student

    df = 3
    scale = _VAR / float(student.ppf(_LEVEL, df))
    losses = student.rvs(df, size=200_000, random_state=rng) * scale
    var_exceed_rate = float(np.mean(losses > _VAR))
    res = acerbi_szekely(losses, _VAR, _ES, _LEVEL, method="Z2")
    # VaR exception rate is ~ nominal (VaR looks fine) ...
    assert var_exceed_rate == pytest.approx(1.0 - _LEVEL, abs=0.003)
    # ... yet the ES statistic is materially positive (ES underestimated).
    assert res.statistic > 0.15


def test_acerbi_szekely_pvalue_small_when_underestimated() -> None:
    rng = np.random.default_rng(2)
    T, n_sims = 250, 2000
    null_losses = rng.normal(0.0, 1.0, (n_sims, T))
    bad = rng.normal(0.0, 1.4, T)  # both VaR and ES understated
    res = acerbi_szekely(bad, _VAR, _ES, _LEVEL, method="Z2", null_losses=null_losses)
    assert res.p_value is not None and res.p_value < 0.05


def test_acerbi_szekely_z1_nan_without_exceptions() -> None:
    losses = np.full(100, -5.0)  # no loss exceeds VaR
    res = acerbi_szekely(losses, _VAR, _ES, _LEVEL, method="Z1")
    assert np.isnan(res.statistic)
    assert res.n_exceptions == 0


def test_acerbi_szekely_rejects_bad_method() -> None:
    with pytest.raises(ValueError, match="method must be"):
        acerbi_szekely(np.zeros(10), _VAR, _ES, _LEVEL, method="Z3")


# --------------------------------------------------------------------------- #
# exceptions helper
# --------------------------------------------------------------------------- #


def test_exceptions_counts_hits() -> None:
    losses = np.array([1.0, 3.0, 0.5, 2.5])
    var = np.array([2.0, 2.0, 2.0, 2.0])
    hits = exceptions(losses, var)
    assert hits.tolist() == [0.0, 1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="shape mismatch"):
        exceptions(losses, var[:2])
