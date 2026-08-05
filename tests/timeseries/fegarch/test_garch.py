"""Validation of the fEGarch GARCH(1,1) model (numerical-validation skill, Phase 1).

The headline is the **fEGarch-fit match**: fitting GARCH(1,1)/normal to the committed synthetic
returns reproduces fEGarch's reported parameters, log-likelihood, information criteria and the full
conditional-SD series to tolerance (these are *exact* fixtures — a real fit, not a noisy sampler, so
the agreement is machine-order, unlike the Monte-Carlo distribution fixtures). The pre-sample
convention (``sigma_0^2 = eps_0^2 = Var(r)``, unbiased) is isolated and confirmed against the
sigma-series fixture, a known-truth simulation recovers planted parameters, and an ``arch``
cross-check agrees modulo the documented convention differences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import fit_garch, garch_recursion, garch_sim

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"


def _load_garch_fixture() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_garch11_norm_sigma.csv", skiprows=1)
    meta = json.loads((_FIXTURE_DIR / "fit_garch11_norm_params.json").read_text(encoding="utf-8"))
    return returns, meta, sigma


# --------------------------------------------------------------------------- #
# Headline: match fEGarch's GARCH(1,1)/norm fit (exact output fixtures)
# --------------------------------------------------------------------------- #


def test_garch11_norm_matches_fegarch_fixture() -> None:
    """fit_garch reproduces fEGarch's params, log-likelihood, AIC/BIC and sigma series."""
    returns, meta, sigma_fix = _load_garch_fixture()
    fegarch = meta["params"]  # fEGarch names: mu, omega, phi1 (=alpha), beta1 (=beta)
    fit = fit_garch(returns, cond_dist="norm")

    # Parameters (fEGarch's phi1/beta1 are our alpha/beta) — a real fit, so agreement is tight.
    assert abs(fit.params["mu"] - fegarch["mu"]) < 1e-6
    assert abs(fit.params["omega"] - fegarch["omega"]) / fegarch["omega"] < 1e-3
    assert abs(fit.params["alpha"] - fegarch["phi1"]) / fegarch["phi1"] < 1e-3
    assert abs(fit.params["beta"] - fegarch["beta1"]) / fegarch["beta1"] < 1e-3
    # Log-likelihood and per-observation information criteria.
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-5
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    # The full conditional-SD series matches to well under a basis point, relative.
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 1e-4
    # Standard errors are finite and positive.
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_garch_recursion_reproduces_fixture_sigma_at_reported_params() -> None:
    """The recursion + pre-sample convention reproduce fEGarch's sigma series at its own params.

    This isolates the pre-sample choice from the optimizer: ``sigma_0^2 = eps_0^2 = Var(r)``
    (unbiased sample variance) matches fEGarch to machine precision; the biased ``mean(eps^2)`` and
    the unconditional ``omega/(1-alpha-beta)`` do not (documented in the module).
    """
    returns, meta, sigma_fix = _load_garch_fixture()
    p = meta["params"]
    params = np.array([p["mu"], p["omega"], p["phi1"], p["beta1"]])
    sigma = np.sqrt(garch_recursion(params, returns))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-6


def test_egarch11_matches_fegarch_fixture() -> None:
    """EGARCH(1,1) fit-match is Phase 2 (the log-variance recursion is not implemented yet)."""
    pytest.skip(
        "Phase 2: the fit fixture fit_egarch11_norm_* is committed, but matching it needs the "
        "EGARCH log-variance recursion (Phase 2 — this PR is Phase 1: GARCH(1,1))."
    )


# --------------------------------------------------------------------------- #
# Known-truth recovery + simulation
# --------------------------------------------------------------------------- #


def test_known_truth_recovery() -> None:
    """QMLE recovers planted (omega, alpha, beta) from a simulated GARCH(1,1) within a few SE."""
    true = {"mu": 0.0003, "omega": 3e-6, "alpha": 0.08, "beta": 0.90}
    returns, _sigma = garch_sim(8000, **true, rng=np.random.default_rng(0))
    fit = fit_garch(returns, cond_dist="norm")
    se = fit.std_errors
    assert abs(fit.params["alpha"] - true["alpha"]) < 4.0 * se["alpha"]
    assert abs(fit.params["beta"] - true["beta"]) < 4.0 * se["beta"]
    assert (
        abs(fit.params["omega"] - true["omega"]) / true["omega"] < 0.5
    )  # omega is weakly identified
    # persistence is the well-identified combination
    assert abs((fit.params["alpha"] + fit.params["beta"]) - 0.98) < 0.03


def test_garch_sim_is_stationary_and_positive() -> None:
    """Simulated conditional variances are positive and the series variance is near the target."""
    returns, sigma = garch_sim(
        20000, mu=0.0, omega=2e-6, alpha=0.05, beta=0.94, rng=np.random.default_rng(1)
    )
    assert np.all(sigma > 0.0)
    target_var = 2e-6 / (1.0 - 0.05 - 0.94)  # unconditional variance
    assert abs(np.var(returns) / target_var - 1.0) < 0.25


def test_garch_sim_supports_all_distributions() -> None:
    """Simulation routes through the Phase-0 distribution layer (e.g. a heavy-tailed Student-t)."""
    returns, sigma = garch_sim(
        3000,
        omega=3e-6,
        alpha=0.1,
        beta=0.85,
        cond_dist="std",
        dist_params=(6.0,),
        rng=np.random.default_rng(2),
    )
    assert returns.shape == sigma.shape == (3000,)
    assert np.all(sigma > 0.0)


def test_garch_sim_rejects_nonstationary_and_bad_size() -> None:
    """Simulation rejects a non-positive size and a non-stationary (alpha + beta >= 1) spec."""
    with pytest.raises(ValueError, match="n must be positive"):
        garch_sim(0, omega=1e-6, alpha=0.1, beta=0.8, rng=np.random.default_rng(3))
    with pytest.raises(ValueError, match="alpha \\+ beta < 1"):
        garch_sim(100, omega=1e-6, alpha=0.3, beta=0.75, rng=np.random.default_rng(3))


# --------------------------------------------------------------------------- #
# Cross-check vs the `arch` package (agreement modulo convention differences)
# --------------------------------------------------------------------------- #


def test_cross_check_against_arch() -> None:
    """Our GARCH(1,1)/norm agrees with ``arch`` on persistence, modulo pre-sample conventions.

    ``arch`` seeds its recursion with an exponentially-weighted backcast (not fEGarch's unbiased
    sample variance) and scales internally, so the individual coefficients differ slightly; the
    persistence ``alpha + beta`` and the broad fit agree. Skipped where ``arch`` (or its
    ``numba``/``llvmlite`` backend) cannot load in the environment; CI validates it.
    """
    try:
        from arch import arch_model
    except Exception as exc:  # numba/llvmlite can fail to load outside CI
        pytest.skip(f"arch unavailable in this environment: {exc}")

    returns, _meta, _sigma = _load_garch_fixture()
    ours = fit_garch(returns, cond_dist="norm")
    fitted = arch_model(returns * 100.0, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(
        disp="off"
    )
    arch_alpha = float(fitted.params["alpha[1]"])
    arch_beta = float(fitted.params["beta[1]"])
    assert abs(ours.params["alpha"] - arch_alpha) < 0.05
    assert abs(ours.params["beta"] - arch_beta) < 0.05
    assert abs((ours.params["alpha"] + ours.params["beta"]) - (arch_alpha + arch_beta)) < 0.02
