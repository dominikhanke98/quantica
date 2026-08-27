"""GJR-GARCH / TGARCH / APARCH under the seven non-normal distributions (Phase 5).

The compressed distributions-breadth build for the asymmetric short-memory family: the three models
share the single APARCH power recursion (delta in {2, 1, free}) and inherit the GARCH distribution
machinery wholesale -- the shape parameters ride in the fitted vector, near-flat shape/skew ridges
Nelder-Mead, and the ALD's integer degree P is profiled over the grid. APARCH additionally carries a
jointly-estimated ``delta`` (the 7-parameter ``{..,gamma1,delta,df}`` and 8-parameter
``{..,gamma1,delta,df,skew}`` compounds).

Validation is identification-structured, exactly as GARCH x 8:

* **The reconstruction seam is machine-exact** at each fixture's own params, seeded from the
  fixture's sigma[0] (the asymmetric family's presample seed differs from fEGarch's by ~1e-7 -- the
  known per-recursion convention, identical for norm and non-norm -- so *fit-vs-fixture* sigma is
  ~1e-7, not machine, exactly as the norm fixtures; the recursion FORM + the distribution likelihood
  compose to ~1e-13, proving no unexpected interaction).
* **Identified fits** (ged/snorm/sged/sald, sharply identified even on Gaussian data) reproduce
  fixtures; **delta** recovers tight (~2.4, identified); **P** selection is exact (= 5).
* **Dominated ``std``/``sstd``** inherit Student-t's flat df ridge: their fixtures sit below the
  norm/snorm sibling, and our fit climbs df to the bound and dominates them (the std pattern).
* **skew -> 1** recovers the symmetric base; **aparch x sstd** known-truth is the 8-param positive
  control that delta, df and skew all recover when the data exercises every parameter.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    aparch_recursion,
    aparch_sim,
    fit_aparch,
    fit_gjr,
    fit_tgarch,
    get_distribution,
    gjr_recursion,
    tgarch_recursion,
)
from quantica.timeseries.fegarch.distributions import AverageLaplace, FernandezSteelSkew

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

# model code -> (fit function, recursion, fixed delta or None for the free-delta APARCH)
_MODELS = {
    "gjrgarch": (fit_gjr, gjr_recursion, 2.0),
    "tgarch": (fit_tgarch, tgarch_recursion, 1.0),
    "aparch": (fit_aparch, aparch_recursion, None),
}
_IDENTIFIED = ["ged", "ald", "snorm", "sged", "sald"]  # sharply identified on the Gaussian data
_DOMINATED = ["std", "sstd"]  # inherit Student-t's flat df ridge -> fixture strictly dominated
_NONNORM = ["std", "ged", "ald", "snorm", "sstd", "sged", "sald"]
_SKEW_TO_BASE = {"snorm": "norm", "sstd": "std", "sged": "ged", "sald": "ald"}


def _returns() -> np.ndarray:  # type: ignore[type-arg]
    """The committed synthetic return series."""
    return np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)


def _load(model: str, dist: str) -> tuple[dict, np.ndarray]:  # type: ignore[type-arg]
    """Load a fixture's params dict and sigma series."""
    meta = json.loads(
        (_FIXTURE_DIR / f"fit_{model}11_{dist}_params.json").read_text(encoding="utf-8")
    )
    sigma = np.loadtxt(_FIXTURE_DIR / f"fit_{model}11_{dist}_sigma.csv", skiprows=1)
    return meta, sigma


def _loglik(model: str, dist: str) -> float:
    """A fixture's log-likelihood."""
    return float(_load(model, dist)[0]["loglikelihood"])


def _dist_and_params(dist: str, fx: dict) -> tuple[object, tuple[float, ...]]:  # type: ignore[type-arg]
    """The distribution instance and its logpdf shape params for a fixture's reported values."""
    if dist == "std":
        return get_distribution("std"), (fx["df"],)
    if dist == "ged":
        return get_distribution("ged"), (fx["shape"],)
    if dist == "ald":
        return AverageLaplace(p=int(fx["P"])), ()
    if dist == "snorm":
        return get_distribution("snorm"), (fx["skew"],)
    if dist == "sstd":
        return get_distribution("sstd"), (fx["df"], fx["skew"])
    if dist == "sged":
        return get_distribution("sged"), (fx["shape"], fx["skew"])
    return FernandezSteelSkew(AverageLaplace(p=int(fx["P"]))), (fx["skew"],)  # sald


def _fixture_seeded_sigma(model: str, fx: dict, sigma_fix: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
    """Rebuild sigma from the params, seeding sigma[0] from the fixture (isolates the form).

    The asymmetric family's presample news-impact seed differs from fEGarch's by ~1e-7 (the known
    per-recursion convention), so seeding sigma[0] from the fixture is how the recursion FORM is
    checked machine-exactly -- exactly as the norm reconstruction test does.
    """
    returns = _returns()
    _fn, _rec, delta_fixed = _MODELS[model]
    d = fx["delta"] if delta_fixed is None else delta_fixed
    resid = returns - fx["mu"]
    sig_d = np.empty(returns.size)
    sig_d[0] = sigma_fix[0] ** d
    g = fx["gamma1"]
    for t in range(1, returns.size):
        kernel = (abs(resid[t - 1]) - g * resid[t - 1]) ** d
        sig_d[t] = fx["omega"] + fx["phi1"] * kernel + fx["beta1"] * sig_d[t - 1]
    return np.asarray(sig_d ** (1.0 / d), dtype=np.float64)


# --------------------------------------------------------------------------- #
# The reconstruction seam: recursion form + distribution likelihood, machine-exact
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", list(_MODELS))
@pytest.mark.parametrize("dist", _NONNORM)
def test_reconstruction_seam_is_machine_exact(model: str, dist: str) -> None:
    """At each fixture's params the recursion (fixture-seeded) + the distribution loglik are exact.

    This is the go/no-go the compressed build rests on: the recursions are proven on norm and the
    distribution likelihoods on GARCH x 8, so their composition must reproduce sigma[1:] to ~1e-13
    and the log-likelihood to ~1e-10 -- confirming no unexpected model x distribution interaction.
    """
    fx_meta, sigma_fix = _load(model, dist)
    fx = fx_meta["params"]
    sigma = _fixture_seeded_sigma(model, fx, sigma_fix)
    assert np.max(np.abs(sigma[1:] - sigma_fix[1:])) < 1e-11

    distribution, shape = _dist_and_params(dist, fx)
    z = (_returns() - fx["mu"]) / sigma
    loglik = float(np.sum(-np.log(sigma) + distribution.logpdf(z, shape or None)))
    assert abs(loglik - fx_meta["loglikelihood"]) < 1e-7


# --------------------------------------------------------------------------- #
# Identified fits reproduce their fixtures (delta tight, P exact)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", list(_MODELS))
@pytest.mark.parametrize("dist", _IDENTIFIED)
def test_identified_fit_matches_fixture(model: str, dist: str) -> None:
    """ged/ald/snorm/sged/sald reproduce loglik, sigma, AIC/BIC, gamma1, and shape/skew.

    delta (APARCH) recovers tight (~2.4, identified); P (ald/sald) selects the fixture's 5. The
    sigma tolerance is the ~1e-7 presample floor (as norm), not machine -- the seam test is the
    machine-exact check.
    """
    fit_fn, _rec, delta_fixed = _MODELS[model]
    fx_meta, sigma_fix = _load(model, dist)
    fx = fx_meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_fn(_returns(), cond_dist=dist)

    assert abs(fit.loglikelihood - fx_meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - fx_meta["aic"]) < 1e-6
    assert abs(fit.bic - fx_meta["bic"]) < 1e-6
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 1e-5
    assert abs(fit.params["gamma1"] - fx["gamma1"]) / abs(fx["gamma1"]) < 5e-3  # the asymmetry
    assert abs(fit.params["beta1"] - fx["beta1"]) / fx["beta1"] < 5e-3

    if "shape" in fx:  # ged/sged -> our "nu"
        assert abs(fit.params["nu"] - fx["shape"]) / fx["shape"] < 1e-2
    if "skew" in fx:
        assert abs(fit.params["skew"] - fx["skew"]) / fx["skew"] < 1e-2
    if "P" in fx:  # ald/sald
        assert fit.params["P"] == fx["P"] == 5
    if delta_fixed is None:  # APARCH free delta, jointly estimated with the shape
        assert abs(fit.params["delta"] - fx["delta"]) < 0.05
        assert 2.0 < fit.params["delta"] < 2.6  # identified, interior


# --------------------------------------------------------------------------- #
# Dominated std / sstd: our fit beats the sub-optimal fixture (flat df ridge)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", list(_MODELS))
@pytest.mark.parametrize("dist", _DOMINATED)
def test_dominated_std_sstd_fit_beats_fixture(model: str, dist: str) -> None:
    """std/sstd inherit Student-t's flat df ridge, so the fixture is dominated; our fit climbs df.

    Each fixture sits below its symmetric-tailed sibling (std < norm, sstd < snorm): the MLE is
    df -> inf; our Nelder-Mead fit reaches the sibling's optimum and dominates the fixture. df is
    validated by regime (large), never param-exact against the dominated df.
    """
    fit_fn, _rec, _delta = _MODELS[model]
    fx_meta, _sigma = _load(model, dist)
    sibling = "norm" if dist == "std" else "snorm"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_fn(_returns(), cond_dist=dist)

    assert fx_meta["loglikelihood"] < _loglik(model, sibling)  # fixture is strictly dominated
    assert fit.loglikelihood >= fx_meta["loglikelihood"] - 1e-4  # we reach at least it
    assert abs(fit.loglikelihood - _loglik(model, sibling)) < 1e-2  # ... and the sibling optimum
    assert fit.params["nu"] > 100.0  # df by regime, not the dominated fixture value


# --------------------------------------------------------------------------- #
# APARCH delta + shape co-estimation, P-profiling, skew reduction, known-truth
# --------------------------------------------------------------------------- #


def test_aparch_co_estimates_delta_and_shape() -> None:
    """APARCH x std is the 7-param {..,gamma1,delta,df}; x sstd the 8-param {..,delta,df,skew}."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seven = fit_aparch(_returns(), cond_dist="std")
        eight = fit_aparch(_returns(), cond_dist="sstd")
    # delta is interior (co-estimated, not pinned at a 1/2 grid seed) in both
    assert 2.0 < seven.params["delta"] < 2.6
    assert 2.0 < eight.params["delta"] < 2.6
    assert {"gamma1", "delta", "nu"} <= set(seven.params)  # 7-param compound
    assert {"gamma1", "delta", "nu", "skew"} <= set(eight.params)  # 8-param compound


@pytest.mark.parametrize("model", list(_MODELS))
def test_ald_sald_profile_selects_boundary_p(model: str) -> None:
    """The ALD P profile is monotone to the grid boundary and selects P = 5, for ald and sald."""
    fit_fn = _MODELS[model][0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for dist in ("ald", "sald"):
            fit = fit_fn(_returns(), cond_dist=dist)
            assert fit.params["P"] == 5
            assert fit.profile is not None
            logliks = [ll for _p, ll in fit.profile]
            assert [p for p, _ll in fit.profile] == [1, 2, 3, 4, 5]
            assert all(logliks[i] < logliks[i + 1] for i in range(len(logliks) - 1))


@pytest.mark.parametrize("model", list(_MODELS))
@pytest.mark.parametrize("skew_code", list(_SKEW_TO_BASE))
def test_skew_one_reduces_to_symmetric_base(model: str, skew_code: str) -> None:
    """Reduction anchor: at skew = 1 the skewed model's likelihood equals the symmetric base's.

    Composes the machine-exact Fernandez-Steel skew=1 reduction with each asymmetric recursion: at a
    common sigma path the skewed loglik at skew=1 matches the base loglik to floating-point order.
    """
    base_code = _SKEW_TO_BASE[skew_code]
    fx_meta, sigma_fix = _load(model, base_code)
    fx = fx_meta["params"]  # the base fixture: has the base shape (df/shape/P) but no skew
    sigma = _fixture_seeded_sigma(model, fx, sigma_fix)
    z = (_returns() - fx["mu"]) / sigma

    # Base shape params (norm: none; std: df; ged: shape; ald: none, P is construction-only).
    base_shape: tuple[float, ...] = {
        "norm": (),
        "std": (fx.get("df", 0.0),),
        "ged": (fx.get("shape", 0.0),),
        "ald": (),
    }[base_code]
    base = get_distribution(base_code) if base_code != "ald" else AverageLaplace(p=int(fx["P"]))
    # The skewed instance re-uses the base's construction (ALD degree P) but adds skew = 1 below.
    if skew_code == "sald":
        skewed: object = FernandezSteelSkew(AverageLaplace(p=int(fx["P"])))
    else:
        skewed = get_distribution(skew_code)
    ll_base = float(np.sum(-np.log(sigma) + base.logpdf(z, base_shape or None)))
    ll_skew = float(np.sum(-np.log(sigma) + skewed.logpdf(z, (*base_shape, 1.0))))
    assert abs(ll_skew - ll_base) < 1e-9


def test_aparch_sstd_known_truth_recovers_delta_df_skew() -> None:
    """The 8-param positive control: APARCH x sstd exercising delta, df and skew; recover all.

    Contrast the near-symmetric Gaussian fixture (where df flies to the bound and skew ~ 1): with a
    genuine identified delta = 1.6, fat tails df = 6 and left-skew 0.85, the 8-way joint fit pins
    every parameter to within a few standard errors.
    """
    true = {"mu": 0.0003, "omega": 8e-4, "phi1": 0.08, "beta1": 0.90, "gamma1": 0.15, "delta": 1.6}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = aparch_sim(
            14000, **true, cond_dist="sstd", dist_params=(6.0, 0.85), rng=np.random.default_rng(0)
        )
        fit = fit_aparch(returns, cond_dist="sstd")
    se = fit.std_errors
    assert abs(fit.params["delta"] - 1.6) < 3.0 * se["delta"]
    assert abs(fit.params["nu"] - 6.0) < 3.0 * se["nu"]
    assert abs(fit.params["skew"] - 0.85) < 3.0 * se["skew"]
    assert abs(fit.params["gamma1"] - 0.15) < 4.0 * se["gamma1"]
