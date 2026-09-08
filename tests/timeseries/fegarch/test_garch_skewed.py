"""Validation of GARCH(1,1) under the four skewed distributions -- completing the set (Phase 5).

The Fernandez-Steel skew (``snorm``, ``sstd``, ``sged``, ``sald``) adds one parameter -- the
``skew`` -- on top of a symmetric base, split-and-restandardized so ``skew = 1`` recovers the base
(App. C.1 Eqs. 38--41, ``s = xi`` applied directly). The joint fit carries ``skew`` in the QMLE
vector alongside the base's own shape parameters; ``sald`` is the compound case (the ``P`` integer
grid profiled on the outside, ``skew`` fit jointly on the inside).

**xi <-> skew reconcile.** The internal FS wrapper's math variable is ``xi``, but fEGarch's argument
is ``skew`` and ``s = xi`` directly (the short-memory Phase-2 finding), so the public parameter is
**named ``skew``** -- ``get_distribution('snorm').param_names == ('skew',)`` -- as the fixtures.

**Identification.** On the symmetric Gaussian returns the skew is nonetheless *identified* (a finite
sample carries a detectable asymmetry signal), so snorm/sged/sald reproduce their fixtures -- skew
included -- to ~1e-6, and each sits legitimately at or above its symmetric base. The exception is
``sstd``: it inherits Student-t's flat ``df`` ridge (MLE ``df -> inf``), so its fixture (``df =
340.8, loglik 7600.93``) is **strictly dominated** -- below norm -- and our fit climbs ``df`` to it,
reaching ``snorm``'s optimum ``7601.21`` and dominating the fixture (the established std pattern).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import fit_garch, garch_recursion, garch_sim, get_distribution

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

# Skewed code -> (symmetric base code, base default shape params for the reduction check).
_SKEW_TO_BASE = {
    "snorm": ("norm", ()),
    "sstd": ("std", (8.0,)),
    "sged": ("ged", (1.5,)),
    "sald": ("ald", ()),
}


def _load(code: str) -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, a GARCH x <code> fit-params JSON, and its sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / f"fit_garch11_{code}_sigma.csv", skiprows=1)
    meta = json.loads(
        (_FIXTURE_DIR / f"fit_garch11_{code}_params.json").read_text(encoding="utf-8")
    )
    return returns, meta, sigma


def _loglik(code: str) -> float:
    """The GARCH x <code> fixture log-likelihood."""
    meta = json.loads(
        (_FIXTURE_DIR / f"fit_garch11_{code}_params.json").read_text(encoding="utf-8")
    )
    return float(meta["loglikelihood"])


# --------------------------------------------------------------------------- #
# The xi <-> skew reconcile and the skew = 1 -> base reduction (FS correctness)
# --------------------------------------------------------------------------- #


def test_skew_parameter_is_named_skew_on_the_public_boundary() -> None:
    """The FS wrapper exposes its parameter as ``skew`` (fEGarch's name), not internal ``xi``."""
    assert get_distribution("snorm").param_names == ("skew",)
    assert get_distribution("sstd").param_names == ("nu", "skew")
    assert get_distribution("sged").param_names == ("nu", "skew")
    assert get_distribution("sald").param_names == ("skew",)  # P is not a QMLE parameter


@pytest.mark.parametrize("skew_code", ["snorm", "sstd", "sged", "sald"])
def test_skew_one_recovers_the_symmetric_base(skew_code: str) -> None:
    """Reduction anchor: at ``skew = 1`` each skewed density equals its symmetric base, exactly.

    This is the Fernandez-Steel wrapper's correctness proof -- the split-and-restandardize collapses
    to the identity at ``s = 1``, so the log density matches the base to floating-point zero.
    """
    base_code, base_shape = _SKEW_TO_BASE[skew_code]
    skewed = get_distribution(skew_code)
    base = get_distribution(base_code)
    z = np.linspace(-5.0, 5.0, 401)
    dev = np.max(np.abs(skewed.logpdf(z, (*base_shape, 1.0)) - base.logpdf(z, base_shape or None)))
    assert dev < 1e-12


# --------------------------------------------------------------------------- #
# Identified fixtures: snorm / sged / sald match to machine order (skew included)
# --------------------------------------------------------------------------- #


def test_snorm_matches_fegarch_fixture() -> None:
    """snorm (skew alone, no base shape) reproduces every fEGarch param, loglik, AIC/BIC, sigma."""
    returns, meta, sigma_fix = _load("snorm")
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="snorm")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    assert abs(fit.params["omega"] - fx["omega"]) / fx["omega"] < 1e-3
    assert abs(fit.params["alpha"] - fx["phi1"]) / fx["phi1"] < 1e-3
    assert abs(fit.params["beta"] - fx["beta1"]) / fx["beta1"] < 1e-3
    assert abs(fit.params["skew"] - fx["skew"]) / fx["skew"] < 1e-3  # skew is identified here
    assert fit.params["skew"] < 1.0  # left-skew (< 1), the fixture's sign
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-5
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert _loglik("snorm") > _loglik("norm")  # a genuine improvement on norm, not a dominated stop
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_sged_matches_fegarch_fixture() -> None:
    """sged (shape + skew, both identified) reproduces the fixture to machine order, beats ged."""
    returns, meta, sigma_fix = _load("sged")
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="sged")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    assert abs(fit.params["omega"] - fx["omega"]) / fx["omega"] < 1e-3
    assert abs(fit.params["nu"] - fx["shape"]) / fx["shape"] < 1e-3  # our "nu" is fEGarch's "shape"
    assert abs(fit.params["skew"] - fx["skew"]) / fx["skew"] < 1e-3
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-5
    assert abs(fit.aic - meta["aic"]) < 1e-6  # k = 6 (shape + skew both count)
    assert _loglik("sged") > _loglik("ged") > _loglik("norm")  # shape then skew each improve
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 1e-5
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_sald_matches_fegarch_fixture_and_profiles_p() -> None:
    """sald is the compound case: the P grid profiles on the outside, skew fits jointly inside.

    Profiling selects P = 5 (as ald), the profile is monotone to the boundary, skew is fit at the
    winning P, and the AIC/BIC penalty counts *both* the profiled P and the skew (k = 6).
    """
    returns, meta, sigma_fix = _load("sald")
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="sald")

    assert fit.params["P"] == fx["P"] == 5  # same boundary selection as ald
    assert abs(fit.params["skew"] - fx["skew"]) / fx["skew"] < 1e-3
    assert abs(fit.params["beta"] - fx["beta1"]) / fx["beta1"] < 5e-3
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6  # k = 6 (P + skew)
    assert fit.profile is not None
    logliks = [ll for _p, ll in fit.profile]
    assert [p for p, _ll in fit.profile] == [1, 2, 3, 4, 5]
    assert all(logliks[i] < logliks[i + 1] for i in range(len(logliks) - 1))  # monotone to boundary
    assert _loglik("sald") > _loglik("ald")  # skew improves on the symmetric ald
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 5e-5


# --------------------------------------------------------------------------- #
# sstd: the dominated case (inherits Student-t's flat df ridge)
# --------------------------------------------------------------------------- #


def test_sstd_seam_is_exact_at_reported_params() -> None:
    """At fEGarch's sstd params the sigma series and sstd log-likelihood match to machine order.

    Proves the joint (df + skew) likelihood is correct even though our *fit* dominates the fixture:
    the seam is evaluated at the fixture's own parameters.
    """
    returns, meta, sigma_fix = _load("sstd")
    fx = meta["params"]
    var_params = np.array([fx["mu"], fx["omega"], fx["phi1"], fx["beta1"]])
    sigma = np.sqrt(garch_recursion(var_params, returns))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10

    sstd = get_distribution("sstd")
    z = (returns - fx["mu"]) / sigma
    loglik = float(np.sum(-np.log(sigma) + sstd.logpdf(z, (fx["df"], fx["skew"]))))
    assert abs(loglik - meta["loglikelihood"]) < 1e-6


def test_sstd_fit_dominates_the_sub_optimal_fixture() -> None:
    """sstd inherits std's flat df ridge, so its fixture is dominated; our fit climbs df past it.

    The fixture stopped at ``df = 340.8`` (loglik 7600.93, *below* norm 7601.11); sstd nests snorm
    as ``df -> inf``, so our Nelder-Mead fit reaches snorm's optimum 7601.21, dominating it. df is
    validated by regime (large), skew stays identified (close to the fixture's).
    """
    returns, meta, _sigma = _load("sstd")
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="sstd")

    assert meta["loglikelihood"] < _loglik("norm")  # the fixture is strictly dominated
    assert fit.loglikelihood > meta["loglikelihood"] + 0.1  # we strictly beat it
    assert (
        abs(fit.loglikelihood - _loglik("snorm")) < 1e-2
    )  # ... reaching the snorm (df->inf) optimum
    assert fit.params["nu"] > 100.0  # df by regime, not the dominated 340.8
    assert abs(fit.params["skew"] - fx["skew"]) < 1e-2  # skew stays identified


# --------------------------------------------------------------------------- #
# Known-truth: skew is recovered when the data is genuinely skewed
# --------------------------------------------------------------------------- #


def test_known_truth_sstd_recovers_skew_and_df() -> None:
    """GARCH x sstd at a genuinely skewed skew = 0.85 (df = 6) recovers both within ~1 SE.

    The positive control: on symmetric data the skew flirts with 1, but with a real left-skew it is
    sharply identified and the joint fit pins both skew and df.
    """
    true = {"mu": 0.0003, "omega": 3e-6, "alpha": 0.08, "beta": 0.90}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = garch_sim(
            12000, **true, cond_dist="sstd", dist_params=(6.0, 0.85), rng=np.random.default_rng(0)
        )
        fit = fit_garch(returns, cond_dist="sstd")
    se = fit.std_errors
    assert abs(fit.params["skew"] - 0.85) < 3.0 * se["skew"]
    assert abs(fit.params["nu"] - 6.0) < 3.0 * se["nu"]
    assert abs(fit.params["beta"] - true["beta"]) < 4.0 * se["beta"]
