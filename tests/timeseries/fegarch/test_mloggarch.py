"""Validation of the fEGarch MLog-GARCH(1,1) Type-I model (numerical-validation skill, Phase 2).

MLog-GARCH is the ``(M_asy=1, p_asy=0; M_mag=1, p_mag=0)`` instance of the generalized Type-I EGF
recursion: the modulus-log transform ``zeta(eta) = sgn(eta)*ln(|eta|+1)`` in **both** terms — the
asymmetry ``zeta(eta)`` and the magnitude ``ln(|eta|+1)``. It shares EGARCH's parameter vector; its
magnitude centering is the new ``mean_log_modulus`` moment ``E[ln(|eta|+1)]`` (not ``E|eta|``), and
because the ``ln(|eta|+1)`` regressor is a compressed version of ``|eta|`` its fitted ``gamma`` is
markedly larger (~0.284 vs EGARCH/MEGARCH's ~0.158). Well-identified, so all params match tight.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    fit_mloggarch,
    get_distribution,
    mloggarch_recursion,
    mloggarch_sim,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"


def _mean_log_modulus_norm() -> float:
    """E[ln(|eta|+1)] for the standard normal, from the distribution layer."""
    return float(get_distribution("norm").mean_log_modulus())


def _load() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit-params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_mloggarch11_norm_sigma.csv", skiprows=1)
    meta = json.loads(
        (_FIXTURE_DIR / "fit_mloggarch11_norm_params.json").read_text(encoding="utf-8")
    )
    return returns, meta, sigma


def test_mean_log_modulus_norm_is_numerical_value() -> None:
    """The normal's modulus-log moment E[ln(|eta|+1)] ~ 0.5348223 (numerical, no closed form)."""
    assert np.isclose(_mean_log_modulus_norm(), 0.5348222957, atol=1e-8)


def test_mloggarch11_norm_matches_fegarch_fixture() -> None:
    """fit_mloggarch reproduces fEGarch's params, log-likelihood, AIC/BIC and sigma (tight)."""
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    fit = fit_mloggarch(returns, cond_dist="norm")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    for name in ("omega_sig", "phi1", "kappa", "gamma"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-3
    assert fit.params["kappa"] < 0.0 < fit.params["gamma"]
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_gamma_reflects_the_log_modulus_magnitude() -> None:
    """MLog-GARCH's log-modulus magnitude gives gamma ~ 0.28 (the parameterization tell).

    The ``ln(|eta|+1)`` regressor is compressed relative to ``|eta|``, so gamma is ~1.8x EGARCH's.
    If the magnitude were mis-wired as ``|eta|`` (EGARCH/MEGARCH's), gamma would return near 0.16 —
    this catches the mis-wired transformation (the ω-units role from the short-memory models).
    """
    returns, _meta, _sigma = _load()
    fit = fit_mloggarch(returns, cond_dist="norm")
    assert fit.params["gamma"] > 0.22  # decisively NOT ~0.16 (the |eta| magnitude)
    assert 0.26 < fit.params["gamma"] < 0.31


def test_recursion_reproduces_fixture_sigma_at_reported_params() -> None:
    """The recursion (log-modulus asymmetry + magnitude) reproduces the sigma series exactly.

    Isolates the g-transformation + the E[ln(|eta|+1)] centering from the optimizer: at fEGarch
    params the conditional-SD series matches to machine order (~1e-15).
    """
    returns, meta, sigma_fix = _load()
    p = meta["params"]
    params = np.array([p["mu"], p["omega_sig"], p["phi1"], p["kappa"], p["gamma"]])
    sigma = np.sqrt(mloggarch_recursion(params, returns, mean_log_modulus=_mean_log_modulus_norm()))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10


def test_known_truth_recovery() -> None:
    """QMLE recovers planted (omega_sig, phi1, kappa, gamma) from a simulation within a few SE."""
    true = {"mu": 0.0003, "omega_sig": -8.7, "phi1": 0.97, "kappa": -0.05, "gamma": 0.25}
    returns, _sigma = mloggarch_sim(12000, **true, rng=np.random.default_rng(0))
    fit = fit_mloggarch(returns, cond_dist="norm")
    se = fit.std_errors
    for name in ("omega_sig", "phi1", "kappa", "gamma"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]


def test_scale_behaviour_is_additive_in_omega_sig() -> None:
    """A return rescale shifts omega_sig additively by ln(scale^2); mu scales, others invariant."""
    returns, _meta, _sigma = _load()
    c = 10.0
    base = fit_mloggarch(returns, cond_dist="norm")
    scaled = fit_mloggarch(returns * c, cond_dist="norm")
    assert abs((scaled.params["omega_sig"] - base.params["omega_sig"]) - np.log(c**2)) < 1e-4
    assert abs(scaled.params["mu"] / base.params["mu"] - c) < 1e-3
    for name in ("phi1", "kappa", "gamma"):
        assert abs(scaled.params[name] - base.params[name]) < 1e-4


def test_mloggarch_sim_is_stationary_and_positive() -> None:
    """Simulated conditional SDs are positive and the log-variance mean is near omega_sig."""
    returns, sigma = mloggarch_sim(
        20000,
        mu=0.0,
        omega_sig=-8.8,
        phi1=0.95,
        kappa=-0.05,
        gamma=0.25,
        rng=np.random.default_rng(1),
    )
    assert np.all(sigma > 0.0)
    assert returns.shape == sigma.shape == (20000,)
    assert abs(np.mean(np.log(sigma**2)) - (-8.8)) < 0.1  # E[ln sigma^2] ~ omega_sig


def test_mloggarch_non_norm_supported() -> None:
    """MLog-GARCH under non-norm now runs: its ``mean_log_modulus`` centering is implemented (EGF).

    Structural inheritance of the EGF distribution infrastructure (fixture-validated on EGARCH).
    """
    returns, sigma = mloggarch_sim(
        100,
        omega_sig=-8.8,
        phi1=0.9,
        kappa=-0.05,
        gamma=0.25,
        cond_dist="std",
        dist_params=(6.0,),
        rng=np.random.default_rng(3),
    )
    assert returns.shape == sigma.shape == (100,)
    assert np.all(sigma > 0.0)
    assert np.isfinite(get_distribution("std").mean_log_modulus((6.0,)))  # centering available


def test_mloggarch_sim_rejects_bad_inputs() -> None:
    """Simulation rejects a non-positive size and a non-stationary (|phi1| >= 1) spec."""
    with pytest.raises(ValueError, match="n must be positive"):
        mloggarch_sim(
            0, omega_sig=-8.8, phi1=0.9, kappa=0.0, gamma=0.25, rng=np.random.default_rng(3)
        )
    with pytest.raises(ValueError, match=r"\|phi1\| < 1"):
        mloggarch_sim(
            100, omega_sig=-8.8, phi1=1.0, kappa=0.0, gamma=0.25, rng=np.random.default_rng(3)
        )
