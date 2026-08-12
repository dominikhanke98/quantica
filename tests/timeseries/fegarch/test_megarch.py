"""Validation of the fEGarch MEGARCH(1,1) Type-I model (numerical-validation skill, Phase 2).

MEGARCH is the ``(M_asy=1, p_asy=0; M_mag=0, p_mag=1)`` instance of the generalized Type-I EGF
recursion: modulus-log asymmetry ``zeta(eta) = sgn(eta)*ln(|eta|+1)`` with the EGARCH magnitude
``|eta|``. It shares EGARCH's parameter vector ``{mu, omega_sig, phi1, kappa, gamma}`` and, since
it keeps the ``|eta|`` magnitude, its fitted ``gamma`` sits near EGARCH's (~0.158). The headline is
the fEGarch-fit match; the model is well-identified (not a ridge case), so all params match tight
(EGARCH tier).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    fit_megarch,
    get_distribution,
    megarch_recursion,
    megarch_sim,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"
_E_ABS_NORM = float(np.sqrt(2.0 / np.pi))  # E|eta| for the standard normal


def _load() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit-params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_megarch11_norm_sigma.csv", skiprows=1)
    meta = json.loads((_FIXTURE_DIR / "fit_megarch11_norm_params.json").read_text(encoding="utf-8"))
    return returns, meta, sigma


def test_megarch11_norm_matches_fegarch_fixture() -> None:
    """fit_megarch reproduces fEGarch's params, log-likelihood, AIC/BIC and sigma series (tight)."""
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    fit = fit_megarch(returns, cond_dist="norm")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    for name in ("omega_sig", "phi1", "kappa", "gamma"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-3
    # Orientation: kappa (asymmetry) < 0 (leverage), gamma (magnitude) > 0.
    assert fit.params["kappa"] < 0.0 < fit.params["gamma"]
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_gamma_reflects_the_absolute_value_magnitude() -> None:
    """MEGARCH keeps EGARCH's ``|eta|`` magnitude, so gamma ~ 0.158 (the parameterization tell).

    If the magnitude were mis-wired as the log-modulus ``ln(|eta|+1)`` (MLog-GARCH's), gamma would
    come back near 0.28 instead. This asserts the magnitude transform is ``|eta|``, not the log.
    """
    returns, _meta, _sigma = _load()
    fit = fit_megarch(returns, cond_dist="norm")
    assert 0.13 < fit.params["gamma"] < 0.19


def test_recursion_reproduces_fixture_sigma_at_reported_params() -> None:
    """The recursion (modulus-log asymmetry + |eta| magnitude) reproduces the sigma series exactly.

    Isolates the g-transformation + pre-sample from the optimizer: at fEGarch's own params the
    conditional-SD series matches to machine order (~1e-15).
    """
    returns, meta, sigma_fix = _load()
    p = meta["params"]
    params = np.array([p["mu"], p["omega_sig"], p["phi1"], p["kappa"], p["gamma"]])
    sigma = np.sqrt(megarch_recursion(params, returns, abs_moment=_E_ABS_NORM))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10


def test_known_truth_recovery() -> None:
    """QMLE recovers planted (omega_sig, phi1, kappa, gamma) from a simulation within a few SE."""
    true = {"mu": 0.0003, "omega_sig": -8.7, "phi1": 0.97, "kappa": -0.05, "gamma": 0.15}
    returns, _sigma = megarch_sim(12000, **true, rng=np.random.default_rng(0))
    fit = fit_megarch(returns, cond_dist="norm")
    se = fit.std_errors
    for name in ("omega_sig", "phi1", "kappa", "gamma"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]


def test_scale_behaviour_is_additive_in_omega_sig() -> None:
    """A return rescale shifts omega_sig additively by ln(scale^2); mu scales, others invariant."""
    returns, _meta, _sigma = _load()
    c = 10.0
    base = fit_megarch(returns, cond_dist="norm")
    scaled = fit_megarch(returns * c, cond_dist="norm")
    assert abs((scaled.params["omega_sig"] - base.params["omega_sig"]) - np.log(c**2)) < 1e-4
    assert abs(scaled.params["mu"] / base.params["mu"] - c) < 1e-3
    for name in ("phi1", "kappa", "gamma"):
        assert abs(scaled.params[name] - base.params[name]) < 1e-4


def test_megarch_sim_supports_symmetric_distributions() -> None:
    """MEGARCH's magnitude reuses abs_moment (present on std), so simulation routes through it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, sigma = megarch_sim(
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


def test_megarch_sim_rejects_bad_inputs() -> None:
    """Simulation rejects a non-positive size and a non-stationary (|phi1| >= 1) spec."""
    with pytest.raises(ValueError, match="n must be positive"):
        megarch_sim(
            0, omega_sig=-8.8, phi1=0.9, kappa=0.0, gamma=0.15, rng=np.random.default_rng(3)
        )
    with pytest.raises(ValueError, match=r"\|phi1\| < 1"):
        megarch_sim(
            100, omega_sig=-8.8, phi1=1.0, kappa=0.0, gamma=0.15, rng=np.random.default_rng(3)
        )


def test_egarch_regression_guard_via_shared_engine() -> None:
    """The generalized engine leaves EGARCH untouched: its (0,1,0,1) instance is bit-identical.

    Reconstructs EGARCH's sigma both through the shared recursion and a direct hand computation of
    ``g = kappa*eta + gamma*(|eta| - E|eta|)`` and asserts they are exactly equal — the refactor did
    not disturb the validated EGARCH path.
    """
    from quantica.timeseries.fegarch import egarch_recursion

    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    params = np.array([5e-4, -8.9, 0.98, -0.02, 0.15])
    via_engine = egarch_recursion(params, returns, abs_moment=_E_ABS_NORM)

    mu, omega_sig, phi1, kappa, gamma = (float(p) for p in params)
    resid = returns - mu
    omega = omega_sig * (1.0 - phi1)
    log_var = np.empty(returns.size)
    log_var[0] = omega + phi1 * np.log(np.var(returns, ddof=1))
    for t in range(1, returns.size):
        eta = resid[t - 1] / np.exp(log_var[t - 1] / 2.0)
        g = kappa * eta + gamma * (abs(eta) - _E_ABS_NORM)
        log_var[t] = omega + g + phi1 * log_var[t - 1]
    direct = np.exp(log_var)

    assert np.array_equal(via_engine, direct)  # bit-for-bit, not just close


def test_norm_moments_are_available() -> None:
    """The centering moments MEGARCH uses are exposed by the normal (magnitude = E|eta|)."""
    assert np.isclose(get_distribution("norm").abs_moment(), _E_ABS_NORM, atol=1e-12)
