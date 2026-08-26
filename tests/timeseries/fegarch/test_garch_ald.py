"""Validation of GARCH(1,1) under the ALD — the P-profiling integer-grid fork.

The scaled average-Laplace (``ald``) is structurally different from the continuous-shape
distributions: its polynomial degree ``P`` is **not** a QMLE parameter but a discrete construction
attribute that fEGarch *profiles* over the integer grid ``Prange = c(1, 5)``. So ``fit_garch`` runs
an **outer grid search** -- for each ``P in {1, 2, 3, 4, 5}`` it fits the four continuous parameters
``(mu, omega, alpha, beta)`` at that fixed ``P`` (a fixed-shape ALD, so gradient-based L-BFGS-B),
then selects the ``P`` with the best log-likelihood. ``P`` never enters the continuous optimizer,
but it *is* counted in the AIC/BIC penalty (``k = 5``), matching fEGarch.

On the Gaussian synthetic returns the profile rises monotonically to the grid boundary -- the
fixture selected ``P = 5`` -- and even there the ALD underfits (loglik ~13 below norm): the ALD is a
fat-tailed family, the wrong shape for Gaussian data, so this is honest model misfit, not a defect.
The ``(P, loglik)`` profile is surfaced on the fit (``GarchFit.profile``). The spec's P-profiling
convention ("fit continuous at each integer P over Prange, keep the best loglik") is confirmed
*empirically*: our profile reproduces the fixture's ``P = 5`` selection and its loglik at ``P = 5``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from quantica.timeseries.fegarch import fit_garch, garch_recursion, get_distribution
from quantica.timeseries.fegarch.distributions import AverageLaplace

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"


def _load_ald() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the GARCH x ald fit-params JSON, and its sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_garch11_ald_sigma.csv", skiprows=1)
    meta = json.loads((_FIXTURE_DIR / "fit_garch11_ald_params.json").read_text(encoding="utf-8"))
    return returns, meta, sigma


def _norm_loglik() -> float:
    """The GARCH x norm fixture log-likelihood (for the honest ALD-underfit note)."""
    meta = json.loads((_FIXTURE_DIR / "fit_garch11_norm_params.json").read_text(encoding="utf-8"))
    return float(meta["loglikelihood"])


def _sim_garch_ald(
    n: int,
    *,
    mu: float,
    omega: float,
    alpha: float,
    beta: float,
    p: int,
    rng: np.random.Generator,
    n_burn: int = 500,
) -> np.ndarray:  # type: ignore[type-arg]
    """Simulate GARCH(1,1) with ALD(P) innovations (garch_sim only exposes the default-P ALD).

    A minimal local sim so the known-truth test can plant an *in-grid* ``P``: draw standardized
    ALD(``p``) innovations and run the identical GARCH(1,1) recursion.
    """
    total = n + n_burn
    innovations = AverageLaplace(p=p).sample(total, rng)
    sigma2 = np.empty(total, dtype=np.float64)
    eps = np.empty(total, dtype=np.float64)
    sigma2[0] = omega / (1.0 - alpha - beta)
    eps[0] = np.sqrt(sigma2[0]) * innovations[0]
    for t in range(1, total):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * innovations[t]
    return np.asarray(mu + eps[n_burn:], dtype=np.float64)


def test_ald_recursion_and_likelihood_are_exact_at_reported_params() -> None:
    """At fEGarch params (P = 5) the sigma series and ALD log-likelihood match to machine order."""
    returns, meta, sigma_fix = _load_ald()
    fx = meta["params"]
    var_params = np.array([fx["mu"], fx["omega"], fx["phi1"], fx["beta1"]])
    sigma = np.sqrt(garch_recursion(var_params, returns))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10

    ald = AverageLaplace(p=int(fx["P"]))
    z = (returns - fx["mu"]) / sigma
    loglik = float(np.sum(-np.log(sigma) + ald.logpdf(z)))
    assert abs(loglik - meta["loglikelihood"]) < 1e-6


def test_garch11_ald_matches_fegarch_fixture() -> None:
    """Profiling selects P = 5 and the continuous fit reproduces fEGarch's params, loglik and sigma.

    P is discrete (profiled, not optimized) so it has no standard error; the four continuous
    parameters' standard errors are finite. AIC/BIC count P in the penalty (k = 5).
    """
    returns, meta, sigma_fix = _load_ald()
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="ald")

    assert fit.params["P"] == fx["P"]  # profiling selects the fixture's P = 5
    assert abs(fit.params["mu"] - fx["mu"]) < 1e-5
    assert abs(fit.params["omega"] - fx["omega"]) / fx["omega"] < 1e-2
    assert abs(fit.params["alpha"] - fx["phi1"]) / fx["phi1"] < 5e-3
    assert abs(fit.params["beta"] - fx["beta1"]) / fx["beta1"] < 5e-3
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 5e-5
    assert np.max(deviation / sigma_fix) < 2e-3
    for name in ("mu", "omega", "alpha", "beta"):
        assert np.isfinite(fit.std_errors[name]) and fit.std_errors[name] > 0.0


def test_p_profile_is_monotone_and_selects_the_boundary() -> None:
    """The profiled log-likelihood rises monotonically across P = 1..5 and peaks at the boundary 5.

    On Gaussian data the ALD's best fit is its thinnest-tailed member (largest P in the grid), so
    the profile is increasing and the argmax sits at the ``Prange`` boundary -- reproducing the
    fixture's ``P = 5`` selection. (The known-truth test, with an in-grid truth, gives an interior
    max instead.)
    """
    returns, meta, _sigma = _load_ald()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_garch(returns, cond_dist="ald")
    assert fit.profile is not None
    grid = [p for p, _ll in fit.profile]
    logliks = [ll for _p, ll in fit.profile]
    assert grid == [1, 2, 3, 4, 5]
    assert all(logliks[i] < logliks[i + 1] for i in range(len(logliks) - 1))  # strictly increasing
    assert grid[int(np.argmax(logliks))] == 5
    assert abs(logliks[-1] - meta["loglikelihood"]) < 1e-4  # boundary loglik matches the fixture


def test_ald_underfits_gaussian_data_below_norm() -> None:
    """Honest misfit: even at best P the ALD is ~13 loglik below norm -- wrong family, not a bug.

    The ALD is fat-tailed (raw kurtosis 3 + 3/(P+1) > 3); Gaussian data has no excess kurtosis, so
    the ALD cannot match the normal even at its thinnest tail. This is genuine model
    misspecification, and it is why the profile pins at the boundary, not an interior optimum.
    """
    _returns, meta, _sigma = _load_ald()
    assert meta["params"]["P"] == 5  # thinnest-tailed member in the grid
    assert _norm_loglik() - meta["loglikelihood"] > 10.0  # ~13 below norm, a real gap


def test_ald_aic_counts_the_profiled_degree() -> None:
    """The AIC/BIC penalty counts P (k = 5) like fEGarch -- k = 4 would not match the fixture."""
    _returns, meta, _sigma = _load_ald()
    n = meta["n_obs"]
    loglik = meta["loglikelihood"]
    aic_k5 = (2.0 * 5 - 2.0 * loglik) / n
    aic_k4 = (2.0 * 4 - 2.0 * loglik) / n
    assert abs(aic_k5 - meta["aic"]) < 1e-9  # P is counted
    assert abs(aic_k4 - meta["aic"]) > 1e-4  # ... and k = 4 clearly does not match


def test_known_truth_ald_recovers_in_grid_p() -> None:
    """GARCH x ald at an in-grid P = 2 gives an interior profile max and recovers P and beta.

    With a genuine in-grid truth the profile peaks in the interior (not the boundary), so the grid
    search selects the true P, and the continuous parameters recover to within a few percent.
    """
    true = {"mu": 0.0003, "omega": 3e-6, "alpha": 0.08, "beta": 0.90}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns = _sim_garch_ald(12000, **true, p=2, rng=np.random.default_rng(1))
        fit = fit_garch(returns, cond_dist="ald")
    assert fit.params["P"] == 2  # the grid search recovers the planted in-grid degree
    logliks = [ll for _p, ll in fit.profile]  # type: ignore[union-attr]
    assert int(np.argmax(logliks)) not in (0, len(logliks) - 1)  # interior max, not a boundary
    assert abs(fit.params["beta"] - true["beta"]) < 0.02
    assert abs(fit.params["alpha"] - true["alpha"]) < 0.02


def test_get_distribution_ald_is_the_default_degree() -> None:
    """get_distribution('ald') is the default-P ALD; the profiling degree comes from fit_garch."""
    assert get_distribution("ald").name == "ald"
    assert get_distribution("ald").param_names == ()  # P is not a QMLE parameter
