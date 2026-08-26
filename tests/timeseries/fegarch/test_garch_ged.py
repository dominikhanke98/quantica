"""Validation of GARCH(1,1) under the GED — the joint shape-parameter path, sharply identified.

GARCH x ged reuses the same joint shape-parameter QMLE machinery as GARCH x std (the shape parameter
rides in the fitted vector, fit by derivative-free Nelder-Mead), but the outcome is the *opposite*
regime: on the synthetic returns the GED shape is **sharply identified** (the fixture's ``shape =
2.15`` sits at a genuine interior peak, SE ~0.10), so the whole fit -- including the shape --
reproduces the fEGarch fixture to machine order, exactly like the norm fit. There is no
dominated-reference issue here: the GED at ``shape = 2.15`` genuinely fits the finite-sample data
slightly better than the ``shape = 2`` normal, so its log-likelihood is *above* norm (contrast the
Student-t anomaly, where the fixture stopped 0.28 below norm). Our layer names the shape ``nu``; the
fEGarch fixture names it ``shape`` (same ``gennorm`` beta; ``shape = 2`` is the normal).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from quantica.timeseries.fegarch import fit_garch, garch_recursion, garch_sim, get_distribution

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"


def _load_ged() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the GARCH x ged fit-params JSON, and its sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_garch11_ged_sigma.csv", skiprows=1)
    meta = json.loads((_FIXTURE_DIR / "fit_garch11_ged_params.json").read_text(encoding="utf-8"))
    return returns, meta, sigma


def _norm_loglik() -> float:
    """The GARCH x norm fixture log-likelihood (for the above-norm effective-challenge note)."""
    meta = json.loads((_FIXTURE_DIR / "fit_garch11_norm_params.json").read_text(encoding="utf-8"))
    return float(meta["loglikelihood"])


def test_ged_recursion_and_likelihood_are_exact_at_reported_params() -> None:
    """At fEGarch's params the sigma series and the GED log-likelihood match to machine order."""
    returns, meta, sigma_fix = _load_ged()
    fx = meta["params"]
    var_params = np.array([fx["mu"], fx["omega"], fx["phi1"], fx["beta1"]])
    sigma = np.sqrt(garch_recursion(var_params, returns))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10

    ged = get_distribution("ged")
    z = (returns - fx["mu"]) / sigma
    loglik = float(np.sum(-np.log(sigma) + ged.logpdf(z, (fx["shape"],))))
    assert abs(loglik - meta["loglikelihood"]) < 1e-6


def test_garch11_ged_matches_fegarch_fixture() -> None:
    """fit_garch/ged reproduces every fEGarch parameter (the shape too), loglik, AIC/BIC and sigma.

    Unlike Student-t (flat nu ridge), the GED shape is sharply identified here, so it recovers to
    ~1e-6 relative -- the fit is as tight as the norm fit, on all five parameters.
    """
    returns, meta, sigma_fix = _load_ged()
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="ged")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    assert abs(fit.params["omega"] - fx["omega"]) / fx["omega"] < 1e-3
    assert abs(fit.params["alpha"] - fx["phi1"]) / fx["phi1"] < 1e-3
    assert abs(fit.params["beta"] - fx["beta1"]) / fx["beta1"] < 1e-3
    assert abs(fit.params["nu"] - fx["shape"]) / fx["shape"] < 1e-3  # our "nu" is fEGarch's "shape"
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-5
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_ged_fit_beats_norm_without_any_dominated_reference() -> None:
    """The GED fixture legitimately sits above norm — no strictly-dominated stop (contrast std).

    GED nests norm at ``shape = 2``; the fitted ``shape = 2.15`` is a real interior improvement on
    the finite-sample data, so the fixture log-likelihood is *above* the norm optimum. Our fit
    reaches at least the fixture, so there is nothing to challenge here -- just a confirmation.
    """
    returns, meta, _sigma = _load_ged()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="ged")
    assert meta["loglikelihood"] > _norm_loglik()  # the fixture is a genuine improvement on norm
    assert fit.loglikelihood >= meta["loglikelihood"] - 1e-5  # we reach it


def test_shape_2_recovers_the_normal_density() -> None:
    """Reduction anchor: the standardized GED density collapses to the normal at shape = 2."""
    ged = get_distribution("ged")
    norm = get_distribution("norm")
    z = np.linspace(-5.0, 5.0, 501)
    assert np.max(np.abs(ged.logpdf(z, (2.0,)) - norm.logpdf(z))) < 1e-12


def test_known_truth_ged_recovers_shape() -> None:
    """Simulating GARCH x ged at a fat-tailed shape = 1.2 recovers nu to a fraction of an SE."""
    true = {"mu": 0.0003, "omega": 3e-6, "alpha": 0.08, "beta": 0.90}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = garch_sim(
            9000, **true, cond_dist="ged", dist_params=(1.2,), rng=np.random.default_rng(0)
        )
        fit = fit_garch(returns, cond_dist="ged")
    se = fit.std_errors
    assert abs(fit.params["nu"] - 1.2) < 3.0 * se["nu"]
    for name in ("alpha", "beta"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]
