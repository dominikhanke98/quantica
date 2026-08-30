"""Validation of the fEGarch EGARCH(1,1) model (numerical-validation skill, Phase 2).

The headline is the **fEGarch-fit match**: fitting EGARCH(1,1)/normal to the committed synthetic
returns reproduces fEGarch's reported parameters (``mu, omega_sig, phi1, kappa, gamma``),
log-likelihood, information criteria and the full conditional-SD series to tolerance. The recursion
is the Type-I EGF log-variance form ``ln sigma^2 = omega + g(eta) + phi1 ln sigma^2`` with
``g(eta) = kappa*eta + gamma*(|eta| - E|eta|)`` (kappa asymmetry, gamma magnitude — the WP171/WP173
orientation confirmed by the fixture signs). The pre-sample convention
(``ln sigma_0^2 = omega + phi1 ln Var(r)``, ``ddof=1``, zero news-impact history) is isolated and
confirmed against the sigma fixture; a kappa=0 symmetry reduction, a known-truth simulation and the
predicted additive-omega_sig scale behaviour round it out.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import egarch_recursion, egarch_sim, fit_egarch

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"
_E_ABS_NORM = float(np.sqrt(2.0 / np.pi))  # E|eta| for the standard normal


def _load() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit-params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_egarch11_norm_sigma.csv", skiprows=1)
    meta = json.loads((_FIXTURE_DIR / "fit_egarch11_norm_params.json").read_text(encoding="utf-8"))
    return returns, meta, sigma


# --------------------------------------------------------------------------- #
# Headline: match fEGarch's EGARCH(1,1)/norm fit
# --------------------------------------------------------------------------- #


def test_egarch11_norm_matches_fegarch_fixture() -> None:
    """fit_egarch reproduces fEGarch's params, log-likelihood, AIC/BIC and sigma series."""
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    fit = fit_egarch(returns, cond_dist="norm")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    for name in ("omega_sig", "phi1", "kappa", "gamma"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-3
    # Sign/orientation: kappa (asymmetry) < 0 (leverage), gamma (magnitude) > 0.
    assert fit.params["kappa"] < 0.0 < fit.params["gamma"]
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_recursion_reproduces_fixture_sigma_at_reported_params() -> None:
    """The recursion + pre-sample convention reproduce fEGarch's sigma series at its own params.

    Isolates the pre-sample from the optimizer: ``ln sigma_0^2 = omega + phi1 ln Var(r)`` (unbiased,
    ``ddof=1``) with a zero news-impact history matches fEGarch to machine precision; ``ddof=0``
    does not (documented in the module). Unlike the short-memory models, the whole series (not just
    sigma_0) is reproduced, since ln Var(r) is exactly fEGarch's seed.
    """
    returns, meta, sigma_fix = _load()
    p = meta["params"]
    params = np.array([p["mu"], p["omega_sig"], p["phi1"], p["kappa"], p["gamma"]])
    sigma = np.sqrt(egarch_recursion(params, returns, abs_moment=_E_ABS_NORM))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10


# --------------------------------------------------------------------------- #
# Reduction: kappa = 0 removes the leverage (g becomes symmetric in eta)
# --------------------------------------------------------------------------- #


def test_kappa_zero_is_symmetric() -> None:
    """With kappa=0 the news impact g(eta) is even, so reflecting residuals leaves sigma fixed."""
    rng = np.random.default_rng(3)
    returns, _sig = egarch_sim(
        2000, mu=0.0005, omega_sig=-8.8, phi1=0.95, kappa=0.0, gamma=0.15, rng=rng
    )
    mu = 0.0005
    sym = np.array([mu, -8.8, 0.95, 0.0, 0.15])
    s_a = egarch_recursion(sym, returns, abs_moment=_E_ABS_NORM)
    s_b = egarch_recursion(sym, 2.0 * mu - returns, abs_moment=_E_ABS_NORM)  # eta -> -eta
    assert np.max(np.abs(s_a - s_b)) < 1e-12
    # A non-zero kappa breaks the symmetry (guards against a vanishing asymmetry term).
    asy = np.array([mu, -8.8, 0.95, -0.1, 0.15])
    d_a = egarch_recursion(asy, returns, abs_moment=_E_ABS_NORM)
    d_b = egarch_recursion(asy, 2.0 * mu - returns, abs_moment=_E_ABS_NORM)
    assert np.max(np.abs(d_a - d_b)) > 1e-6


# --------------------------------------------------------------------------- #
# Known-truth recovery + scale behaviour + simulation
# --------------------------------------------------------------------------- #


def test_known_truth_recovery() -> None:
    """QMLE recovers planted (omega_sig, phi1, kappa, gamma) from a simulation within a few SE."""
    true = {"mu": 0.0003, "omega_sig": -8.7, "phi1": 0.97, "kappa": -0.08, "gamma": 0.15}
    returns, _sigma = egarch_sim(12000, **true, rng=np.random.default_rng(0))
    fit = fit_egarch(returns, cond_dist="norm")
    se = fit.std_errors
    for name in ("omega_sig", "phi1", "kappa", "gamma"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]


def test_scale_behaviour_is_additive_in_omega_sig() -> None:
    """A return rescale shifts omega_sig additively by ln(scale^2); mu scales, others invariant."""
    returns, _meta, _sigma = _load()
    c = 10.0
    base = fit_egarch(returns, cond_dist="norm")
    scaled = fit_egarch(returns * c, cond_dist="norm")
    assert abs((scaled.params["omega_sig"] - base.params["omega_sig"]) - np.log(c**2)) < 1e-4
    assert abs(scaled.params["mu"] / base.params["mu"] - c) < 1e-3
    for name in ("phi1", "kappa", "gamma"):
        assert abs(scaled.params[name] - base.params[name]) < 1e-4


def test_egarch_sim_is_stationary_and_positive() -> None:
    """Simulated conditional SDs are positive and the log-variance mean is near omega_sig."""
    returns, sigma = egarch_sim(
        20000,
        mu=0.0,
        omega_sig=-8.8,
        phi1=0.95,
        kappa=-0.05,
        gamma=0.15,
        rng=np.random.default_rng(1),
    )
    assert np.all(sigma > 0.0)
    assert returns.shape == sigma.shape == (20000,)
    assert abs(np.mean(np.log(sigma**2)) - (-8.8)) < 0.1  # E[ln sigma^2] ~ omega_sig


def test_egarch_sim_supports_all_distributions() -> None:
    """Simulation routes through the Phase-0 distribution layer (a heavy-tailed Student-t)."""
    returns, sigma = egarch_sim(
        3000,
        omega_sig=-8.8,
        phi1=0.95,
        kappa=-0.05,
        gamma=0.15,
        cond_dist="std",
        dist_params=(6.0,),
        rng=np.random.default_rng(2),
    )
    assert returns.shape == sigma.shape == (3000,)
    assert np.all(sigma > 0.0)


def test_egarch_sim_rejects_bad_inputs() -> None:
    """Simulation rejects a non-positive size and a non-stationary (|phi1| >= 1) spec."""
    with pytest.raises(ValueError, match="n must be positive"):
        egarch_sim(
            0, omega_sig=-8.8, phi1=0.95, kappa=0.0, gamma=0.15, rng=np.random.default_rng(3)
        )
    with pytest.raises(ValueError, match=r"\|phi1\| < 1"):
        egarch_sim(
            100, omega_sig=-8.8, phi1=1.0, kappa=0.0, gamma=0.15, rng=np.random.default_rng(3)
        )


def test_skewed_egarch_abs_moment_is_available() -> None:
    """The FS-skew ``abs_moment`` is now implemented (the Phase-5 EGF distribution infrastructure).

    It computes by quadrature over the skewed density; reduces to the normal base at ``skew = 1``.
    """
    from quantica.timeseries.fegarch import get_distribution

    snorm = get_distribution("snorm")
    assert np.isfinite(snorm.abs_moment((1.2,)))
    assert abs(snorm.abs_moment((1.0,)) - get_distribution("norm").abs_moment()) < 1e-9
