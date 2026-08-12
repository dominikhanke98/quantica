"""Validation of the fEGarch Log-GARCH(1,1) Type-II model (numerical-validation skill, Phase 2).

The headline is the **fEGarch-fit match**: fitting Log-GARCH(1,1)/normal to the committed synthetic
returns reproduces fEGarch's ``{mu, omega_sig, phi1, psi1}``, log-likelihood, information criteria
and the conditional-SD series. Type-II has **no asymmetry term** — the log-variance is an ARMA in
the log-square innovation ``xi = ln(eta^2) - E[ln eta^2]``:
``ln sigma^2_t = omega + phi1 ln sigma^2_{t-1} + (psi1 + phi1) xi_{t-1}``.

**Tolerances are structured by identification, not loosened uniformly.** The conditional variance,
the log-likelihood, the information criteria and the **combined loading ``(psi1 + phi1)``** are
asserted TIGHT (EGARCH tier); ``phi1`` and ``psi1`` *individually* are allowed a documented looser
tolerance, because ``phi1 ~ 0.989`` and ``psi1 ~ -0.954`` nearly cancel (near-common-root) so the
likelihood is flat/multimodal along the ``phi1 ~ -psi1`` ridge and the individual AR/MA coefficients
are weakly identified. The tolerance split is itself the diagnostic: a loose sigma-series would be a
real discrepancy, not the ridge. (Realized deviations are far tighter than the tolerances here —
from an fEGarch-basin start the individual coefficients match to ~1e-7 — but the split guards CI
against the ridge's start/platform sensitivity.)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    fit_loggarch,
    get_distribution,
    loggarch_recursion,
    loggarch_sim,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"
_MEAN_LOG_SQ_NORM = float(-np.euler_gamma - np.log(2.0))  # E[ln eta^2] for the standard normal


def _load() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit-params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_loggarch11_norm_sigma.csv", skiprows=1)
    meta = json.loads(
        (_FIXTURE_DIR / "fit_loggarch11_norm_params.json").read_text(encoding="utf-8")
    )
    return returns, meta, sigma


def test_mean_log_sq_norm_closed_form() -> None:
    """The normal's log-square moment is E[ln eta^2] = psi(1/2)+ln2 = -gamma_E-ln2 ~ -1.2703628."""
    value = get_distribution("norm").mean_log_sq()
    assert np.isclose(value, -1.2703628455, atol=1e-9)
    assert np.isclose(value, _MEAN_LOG_SQ_NORM, atol=1e-12)


# --------------------------------------------------------------------------- #
# Headline: match fEGarch's Log-GARCH(1,1)/norm fit (identification-structured tolerances)
# --------------------------------------------------------------------------- #


def test_loggarch11_norm_matches_fegarch_fixture() -> None:
    """fit_loggarch reproduces fEGarch: sigma / loglik / IC / (psi1+phi1) tight; phi1/psi1 loose."""
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    fit = fit_loggarch(returns, cond_dist="norm")

    # TIGHT (EGARCH tier): the well-identified quantities.
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5  # sigma-series: a loose value here would be a real discrepancy
    assert np.max(deviation / sigma_fix) < 1e-4
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-3
    assert abs(fit.aic - meta["aic"]) < 1e-5
    assert abs(fit.bic - meta["bic"]) < 1e-5
    combined_ours = fit.params["psi1"] + fit.params["phi1"]
    combined_fx = fx["psi1"] + fx["phi1"]
    assert abs(combined_ours - combined_fx) / abs(combined_fx) < 1e-3  # combined loading is pinned
    assert abs(fit.params["mu"] - fx["mu"]) < 1e-5
    assert abs(fit.params["omega_sig"] - fx["omega_sig"]) / abs(fx["omega_sig"]) < 1e-3

    # LOOSER (documented near-common-root ridge): phi1 and psi1 individually are weakly identified.
    assert abs(fit.params["phi1"] - fx["phi1"]) / abs(fx["phi1"]) < 5e-3
    assert abs(fit.params["psi1"] - fx["psi1"]) / abs(fx["psi1"]) < 5e-3

    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_recursion_reproduces_fixture_sigma_at_reported_params() -> None:
    """The recursion + pre-sample reproduce fEGarch's sigma at its own params (machine order).

    Isolates the recursion from the optimizer: ``ln sigma_0^2 = omega + phi1 ln Var(r)`` (ddof=1),
    zero xi history, loading ``(psi1+phi1)``, centering ``E[ln eta^2]=-gamma_E-ln2`` — matches to
    ~1e-15 (ddof=0 gives ~2e-6). This is the tight anchor even where the *fit* is ridge-limited.
    """
    returns, meta, sigma_fix = _load()
    p = meta["params"]
    params = np.array([p["mu"], p["omega_sig"], p["phi1"], p["psi1"]])
    sigma = np.sqrt(loggarch_recursion(params, returns, mean_log_sq=_MEAN_LOG_SQ_NORM))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10


# --------------------------------------------------------------------------- #
# Known-truth recovery + scale behaviour + simulation
# --------------------------------------------------------------------------- #


def test_known_truth_recovery() -> None:
    """QMLE recovers the combined loading (psi1+phi1) tight; individuals may be ridge-limited."""
    true = {"mu": 0.0003, "omega_sig": -8.7, "phi1": 0.95, "psi1": -0.90}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = loggarch_sim(12000, **true, rng=np.random.default_rng(0))
        fit = fit_loggarch(returns, cond_dist="norm")
    # The well-identified combined loading recovers tightly.
    combined = fit.params["psi1"] + fit.params["phi1"]
    assert abs(combined - (true["psi1"] + true["phi1"])) < 0.02
    # omega_sig (unconditional log-variance) recovers within a few SE.
    assert abs(fit.params["omega_sig"] - true["omega_sig"]) < 4.0 * fit.std_errors["omega_sig"]


def test_scale_behaviour_is_additive_in_omega_sig() -> None:
    """A return rescale shifts omega_sig by ln(scale^2); mu scales, phi1/psi1 invariant."""
    returns, _meta, _sigma = _load()
    c = 10.0
    base = fit_loggarch(returns, cond_dist="norm")
    scaled = fit_loggarch(returns * c, cond_dist="norm")
    assert abs((scaled.params["omega_sig"] - base.params["omega_sig"]) - np.log(c**2)) < 1e-4
    assert abs(scaled.params["mu"] / base.params["mu"] - c) < 1e-3
    assert abs(scaled.params["phi1"] - base.params["phi1"]) < 1e-4
    assert abs(scaled.params["psi1"] - base.params["psi1"]) < 1e-4


def test_loggarch_sim_is_stationary_and_positive() -> None:
    """Simulated conditional SDs are positive and the log-variance mean is near omega_sig."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, sigma = loggarch_sim(
            20000, mu=0.0, omega_sig=-8.8, phi1=0.9, psi1=-0.5, rng=np.random.default_rng(1)
        )
    assert np.all(sigma > 0.0)
    assert returns.shape == sigma.shape == (20000,)
    assert abs(np.mean(np.log(sigma**2)) - (-8.8)) < 0.1  # E[ln sigma^2] ~ omega_sig


def test_loggarch_sim_non_norm_is_deferred() -> None:
    """Simulation under non-norm is deferred: Type-II needs ``mean_log_sq`` (only ``norm``)."""
    with pytest.raises(NotImplementedError):
        loggarch_sim(
            3000,
            omega_sig=-8.8,
            phi1=0.9,
            psi1=-0.5,
            cond_dist="std",
            dist_params=(6.0,),
            rng=np.random.default_rng(2),
        )


def test_loggarch_sim_rejects_bad_inputs() -> None:
    """Simulation rejects a non-positive size and a non-stationary (|phi1| >= 1) spec."""
    with pytest.raises(ValueError, match="n must be positive"):
        loggarch_sim(0, omega_sig=-8.8, phi1=0.9, psi1=-0.5, rng=np.random.default_rng(3))
    with pytest.raises(ValueError, match=r"\|phi1\| < 1"):
        loggarch_sim(100, omega_sig=-8.8, phi1=1.0, psi1=-0.5, rng=np.random.default_rng(3))


def test_type_ii_mean_log_sq_deferred_for_non_norm() -> None:
    """Only ``norm`` exposes mean_log_sq; std/ged and the skewed variants are deferred."""
    for code in ("std", "ged", "snorm"):
        with pytest.raises(NotImplementedError):
            get_distribution(code).mean_log_sq(get_distribution(code).param_start)
