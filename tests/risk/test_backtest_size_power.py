r"""Validate the *backtests themselves* — their size and power (the meta-challenge).

The effective-challenge point (CLAUDE.md §1): running a backtest is not enough;
one must show the backtest *works* — that it rejects a correct model at about the
nominal rate (**size**) and rejects a mis-specified model often (**power**). These
are Monte-Carlo studies of the test statistics, with seeded generators, and they
document an honest finding: at 99% the exception-based tests are conservative
(under-sized) because exceptions are rare, while they retain strong power.

All rejection-rate bands below were calibrated empirically and carry Monte-Carlo
error ~ ``sqrt(r(1-r)/n_trials)``; they are deliberately loose enough to be robust
to the seed but tight enough to catch a broken statistic.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.risk import (
    acerbi_szekely,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
)
from scipy.stats import norm
from scipy.stats import t as student

_LEVEL = 0.99
_TAIL = 1.0 - _LEVEL
_SIZE = 0.05  # nominal test size


# --------------------------------------------------------------------------- #
# Kupiec: size and power
# --------------------------------------------------------------------------- #


def test_kupiec_has_correct_size() -> None:
    # Under a correct model, exceptions are i.i.d. Bernoulli(p); Kupiec should
    # reject at roughly the nominal 5% (a touch conservative from discreteness).
    rng = np.random.default_rng(0)
    n_trials, T = 3000, 750
    rejections = sum(
        kupiec_pof(int((rng.random(T) < _TAIL).sum()), T, _LEVEL).reject(_SIZE)
        for _ in range(n_trials)
    )
    rate = rejections / n_trials
    assert 0.02 < rate < 0.075


def test_kupiec_has_power_against_undercoverage() -> None:
    # True volatility 1.5x the model's: far too many exceptions -> reject nearly
    # always.
    rng = np.random.default_rng(1)
    n_trials, T = 1000, 750
    z = norm.ppf(_LEVEL)
    rejections = sum(
        kupiec_pof(int((rng.normal(0, 1.5, T) > z).sum()), T, _LEVEL).reject(_SIZE)
        for _ in range(n_trials)
    )
    assert rejections / n_trials > 0.9


# --------------------------------------------------------------------------- #
# Christoffersen: size and power (against clustering)
# --------------------------------------------------------------------------- #


def test_christoffersen_independence_size_is_not_oversized() -> None:
    # For rare 99% exceptions the independence LR is conservative (under-sized) —
    # an honest, documented limitation. We assert it does not *over*-reject.
    rng = np.random.default_rng(2)
    n_trials, T = 3000, 750
    rejections = sum(
        christoffersen_independence((rng.random(T) < _TAIL).astype(float)).reject(_SIZE)
        for _ in range(n_trials)
    )
    assert rejections / n_trials < 0.06


def test_christoffersen_independence_has_power_against_clustering() -> None:
    # A Markov exception process where a hit makes the next hit far more likely
    # (pi11 >> pi01) should be flagged as dependent.
    rng = np.random.default_rng(3)
    n_trials, T = 2000, 750
    rejections = 0
    for _ in range(n_trials):
        hits = np.zeros(T)
        state = 0
        for t in range(T):
            prob = 0.30 if state == 1 else 0.0072  # clustered, ~same overall rate
            state = 1 if rng.random() < prob else 0
            hits[t] = state
        if christoffersen_independence(hits).reject(_SIZE):
            rejections += 1
    assert rejections / n_trials > 0.5


def test_christoffersen_cc_has_power() -> None:
    # Conditional coverage should reject when the model badly under-covers.
    rng = np.random.default_rng(4)
    n_trials, T = 1000, 750
    z = norm.ppf(_LEVEL)
    rejections = sum(
        christoffersen_cc((rng.normal(0, 1.5, T) > z).astype(float), _LEVEL).reject(_SIZE)
        for _ in range(n_trials)
    )
    assert rejections / n_trials > 0.9


# --------------------------------------------------------------------------- #
# Acerbi--Székely ES backtest: size and power
# --------------------------------------------------------------------------- #

# ES tests use 97.5% (Basel FRTB's ES level), where the tail is less sparse.
_ES_LEVEL = 0.975
_ES_TAIL = 1.0 - _ES_LEVEL
_ES_VAR = float(norm.ppf(_ES_LEVEL))
_ES_ES = float(norm.pdf(norm.ppf(_ES_LEVEL)) / _ES_TAIL)


def _z2(losses: np.ndarray) -> np.ndarray:
    """Vectorised Z2 over the last axis, at the fixed model VaR/ES (for speed)."""
    hits = losses > _ES_VAR
    T = losses.shape[-1]
    return (losses * hits).sum(axis=-1) / _ES_ES / (T * _ES_TAIL) - 1.0


def test_acerbi_szekely_wiring_matches_reference() -> None:
    # The library statistic and p-value match the vectorised reference used below,
    # so the size/power study validates the shipped function.
    rng = np.random.default_rng(5)
    T, n_sims = 250, 500
    null = rng.normal(0, 1, (n_sims, T))
    realized = rng.normal(0, 1.3, T)
    res = acerbi_szekely(realized, _ES_VAR, _ES_ES, _ES_LEVEL, method="Z2", null_losses=null)
    assert res.statistic == pytest.approx(float(_z2(realized)))
    expected_p = float(np.mean(_z2(null) >= _z2(realized)))
    assert res.p_value == pytest.approx(expected_p)


def test_acerbi_szekely_has_correct_size() -> None:
    rng = np.random.default_rng(6)
    T, n_sims, n_trials = 250, 1500, 800
    null_stats = _z2(rng.normal(0, 1, (n_sims, T)))
    realized = _z2(rng.normal(0, 1, (n_trials, T)))
    p_values = np.mean(null_stats[None, :] >= realized[:, None], axis=1)
    rate = float(np.mean(p_values < _SIZE))
    assert 0.02 < rate < 0.08


def test_acerbi_szekely_has_power_against_es_underestimation() -> None:
    # Realised volatility 1.4x: the ES is badly understated, so Z2 rejects often.
    rng = np.random.default_rng(7)
    T, n_sims, n_trials = 250, 1500, 800
    null_stats = _z2(rng.normal(0, 1, (n_sims, T)))
    realized = _z2(rng.normal(0, 1.4, (n_trials, T)))
    p_values = np.mean(null_stats[None, :] >= realized[:, None], axis=1)
    assert float(np.mean(p_values < _SIZE)) > 0.8


def test_acerbi_szekely_detects_heavy_tail_that_var_count_misses() -> None:
    # The headline ES insight: a heavy tail with the SAME VaR passes a VaR count
    # test but fails the ES test. Average Z2 is clearly positive under the t-tail.
    rng = np.random.default_rng(8)
    T, n_trials = 250, 800
    df = 3
    scale = _ES_VAR / float(student.ppf(_ES_LEVEL, df))
    realized = student.rvs(df, size=(n_trials, T), random_state=rng) * scale
    # VaR exception rate ~ nominal (VaR is fine) ...
    assert float(np.mean(realized > _ES_VAR)) == pytest.approx(_ES_TAIL, abs=0.004)
    # ... but the mean ES statistic is positive (ES understated).
    assert float(np.mean(_z2(realized))) > 0.15
