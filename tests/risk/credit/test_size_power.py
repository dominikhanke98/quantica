r"""Validate the calibration tests themselves — size and power (the meta-challenge).

Exactly as for the market-risk backtests: running a calibration test is not
evidence; showing the test rejects a *correctly calibrated* model at about the
nominal rate (size) and catches an *understated* PD often (power) is. Simulated
rating grades with **known true PDs** make this measurable.

The honest findings these tests pin down (bands calibrated empirically, wide
enough for seed robustness, tight enough to catch a broken statistic):

* the **exact binomial test is conservative** — its discreteness pushes the size
  well below the nominal 5%, and the smaller the grade the worse it gets
  (~1.7% at n=150); its power suffers correspondingly;
* the **Jeffreys test holds near-nominal size** in the same grades and roughly
  **doubles the power** on the low-default grade — precisely the property for
  which the ECB instructions adopt it;
* **Hosmer--Lemeshow** has correct size with ``dof = G`` when the PDs are true
  (not estimated), while the textbook ``G - 2`` convention — meant for models
  fitted on the sample — visibly over-rejects in that setting (~11%): using the
  right degrees of freedom is part of validating the validator.
"""

from __future__ import annotations

import numpy as np
from quantica.risk.credit import binomial_test, hosmer_lemeshow, jeffreys_test

_SIZE = 0.05  # nominal test size throughout


def rejection_rate(defaults: np.ndarray, n_obligors: int, pd: float, test: str) -> float:
    """Rejection rate over simulated default counts, via a unique-value cache."""
    p_value = {
        int(d): (
            binomial_test(int(d), n_obligors, pd).p_value
            if test == "binomial"
            else jeffreys_test(int(d), n_obligors, pd).p_value
        )
        for d in np.unique(defaults)
    }
    return float(np.mean([p_value[int(d)] < _SIZE for d in defaults]))


# --------------------------------------------------------------------------- #
# Binomial vs Jeffreys: size
# --------------------------------------------------------------------------- #


def test_binomial_size_is_conservative() -> None:
    # A well-calibrated grade (n=800, PD=2%): the exact test's discreteness makes
    # it under-reject — measured ~2.8%, clearly below nominal. Honest finding.
    rng = np.random.default_rng(0)
    d = rng.binomial(800, 0.02, 5000)
    rate = rejection_rate(d, 800, 0.02, "binomial")
    assert 0.005 < rate < 0.045


def test_jeffreys_size_is_near_nominal() -> None:
    # Same grades: the Jeffreys test sits at ~5% — the reason ECB adopted it.
    rng = np.random.default_rng(1)
    d = rng.binomial(800, 0.02, 5000)
    rate = rejection_rate(d, 800, 0.02, "jeffreys")
    assert 0.03 < rate < 0.075


def test_conservatism_worsens_in_low_default_grades() -> None:
    # n=150, PD=1%: the binomial size collapses (~1.7%) while Jeffreys stays
    # in the nominal neighbourhood (~6%).
    rng = np.random.default_rng(2)
    d = rng.binomial(150, 0.01, 5000)
    binomial_size = rejection_rate(d, 150, 0.01, "binomial")
    jeffreys_size = rejection_rate(d, 150, 0.01, "jeffreys")
    assert binomial_size < 0.03
    assert 0.03 < jeffreys_size < 0.09
    assert jeffreys_size > 2.0 * binomial_size


# --------------------------------------------------------------------------- #
# Binomial vs Jeffreys: power against PD understatement
# --------------------------------------------------------------------------- #


def test_both_tests_have_power_in_a_large_grade() -> None:
    # True default rate double the assigned PD, n=800: both catch it.
    rng = np.random.default_rng(3)
    d = rng.binomial(800, 0.04, 5000)  # truth 4%, tested against PD 2%
    assert rejection_rate(d, 800, 0.02, "binomial") > 0.85
    assert rejection_rate(d, 800, 0.02, "jeffreys") > 0.90


def test_jeffreys_roughly_doubles_power_in_low_default_grade() -> None:
    # n=150, truth 2% vs assigned 1%: measured ~19% (binomial) vs ~36% (Jeffreys).
    # The conservatism of the exact test is not free — it costs detection.
    rng = np.random.default_rng(4)
    d = rng.binomial(150, 0.02, 5000)
    binomial_power = rejection_rate(d, 150, 0.01, "binomial")
    jeffreys_power = rejection_rate(d, 150, 0.01, "jeffreys")
    assert jeffreys_power > 1.4 * binomial_power
    assert jeffreys_power > 0.25


# --------------------------------------------------------------------------- #
# Hosmer--Lemeshow: size (both dof conventions) and power
# --------------------------------------------------------------------------- #


def _hl_trial(seed: int, scale: float) -> tuple[bool, bool, bool]:
    """One HL trial on true PDs: (reject@dof=G, reject@default G-2, reject scaled)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / (1.0 + np.exp(-rng.normal(-3.5, 1.0, 4000)))
    y = (rng.random(4000) < p).astype(float)
    understated = np.clip(p * scale, 1e-9, 1.0)
    return (
        hosmer_lemeshow(y, p, dof=10).reject(_SIZE),
        hosmer_lemeshow(y, p).reject(_SIZE),
        hosmer_lemeshow(y, understated, dof=10).reject(_SIZE),
    )


def test_hosmer_lemeshow_size_and_power() -> None:
    trials = [_hl_trial(1000 + i, scale=0.5) for i in range(800)]
    size_dof_g = float(np.mean([t[0] for t in trials]))
    size_default = float(np.mean([t[1] for t in trials]))
    power = float(np.mean([t[2] for t in trials]))
    # With TRUE PDs the statistic is chi^2_G: dof=G holds nominal size ...
    assert 0.02 < size_dof_g < 0.08
    # ... while the G-2 fitted-model convention over-rejects in this setting
    # (measured ~11%) — the documented reason hosmer_lemeshow exposes `dof`.
    assert size_default > 0.08
    # Power: PDs understated by half are caught essentially always.
    assert power > 0.95
