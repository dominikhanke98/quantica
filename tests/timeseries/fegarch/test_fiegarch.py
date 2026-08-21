"""Validation of the fEGarch FIEGARCH(1,d,1) long-memory model (numerical-validation, Phase 4).

FIEGARCH is the first fractionally-integrated model, built by **composition** — it reuses the
Phase-2 Type-I news impact (``type1_news_impact`` at ``EGARCH_CONSTANTS``) and the Phase-3
``fracdiff_coeffs`` operator, adding the ``theta(B) = (1-phi1 B)^{-1}(1-B)^{-d}`` composition and
the truncated-MA(inf) recursion ``ln sigma^2_t = omega_sig + sum_i theta_i g(eta_{t-1-i})``. The
headline is the fEGarch-fit match (all six params incl. fractional ``d``); it is well-identified,
so it matches at EGARCH tier. The MA-form presample (intercept ``omega_sig`` directly,
``sigma[0]=exp(omega_sig/2)``, no ``ln Var`` seed) is the one divergence from EGARCH, a tell.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    fiegarch_recursion,
    fiegarch_sim,
    fit_fiegarch,
    get_distribution,
    theta_coefficients,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"
_E_ABS_NORM = float(np.sqrt(2.0 / np.pi))  # E|eta| for the standard normal


def _load() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit-params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_fiegarch11_norm_sigma.csv", skiprows=1)
    meta = json.loads(
        (_FIXTURE_DIR / "fit_fiegarch11_norm_params.json").read_text(encoding="utf-8")
    )
    return returns, meta, sigma


def _params(fx: dict) -> np.ndarray:  # type: ignore[type-arg]
    """The 6-vector (mu, omega_sig, phi1, kappa, gamma, d) from a fixture params dict."""
    return np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["kappa"], fx["gamma"], fx["d"]])


def test_fiegarch11_norm_matches_fegarch_fixture() -> None:
    """fit_fiegarch reproduces fEGarch's six params (incl. d), log-likelihood, AIC/BIC and sigma."""
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # transient exp-overflow at rejected optimizer points
        fit = fit_fiegarch(returns, cond_dist="norm")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    for name in ("omega_sig", "phi1", "kappa", "gamma", "d"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-3
    assert fit.params["kappa"] < 0.0 < fit.params["gamma"]
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 2e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_fitted_d_is_in_the_upper_long_memory_regime() -> None:
    """The fitted d ~ 0.744 is in (0.5, 1) — confirms the QMLE bound is NOT clamped at 0.5.

    The high-persistence GARCH-simulated series is captured by a large d and a small phi1 (~0.39):
    the persistence is reparameterized from the AR term into the slow theta tail.
    """
    _returns, meta, _sigma = _load()
    fx = meta["params"]
    assert 0.5 < fx["d"] < 1.0  # the fixture d itself is in the non-weakly-stationary regime
    assert fx["phi1"] < 0.5  # and phi1 dropped well below EGARCH's ~0.98


def test_recursion_reproduces_fixture_sigma_at_reported_params() -> None:
    """The theta(B) composition + MA(inf) recursion reproduce the sigma series to machine order.

    Isolates the whole spec (theta = phi^{-1} * (1-B)^{-d} via the Phase-3 engine, EGARCH g,
    omega_sig intercept, g-presample 0) from the optimizer: at fEGarch params it matches to ~1e-15.
    """
    returns, meta, sigma_fix = _load()
    sigma = np.sqrt(fiegarch_recursion(_params(meta["params"]), returns, abs_moment=_E_ABS_NORM))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10


def test_ma_presample_tell_sigma0_is_exp_half_omega_sig() -> None:
    """The MA(inf) presample gives sigma[0] = exp(omega_sig/2) exactly (no ln(Var) seed).

    This is the divergence from EGARCH (whose sigma[0] came from ln(Var(r))); the reconstruction
    confirmed the omega_sig intercept (~1e-16) vs the AR-form omega_sig(1-phi1) (rel 4.9).
    """
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    sigma = np.sqrt(fiegarch_recursion(_params(fx), returns, abs_moment=_E_ABS_NORM))
    assert np.isclose(sigma[0], np.exp(fx["omega_sig"] / 2.0), atol=1e-14)
    assert np.isclose(
        sigma_fix[0], np.exp(fx["omega_sig"] / 2.0), atol=1e-8
    )  # fEGarch's own sigma[0]


def test_d_zero_reduces_theta_to_egarch_geometric() -> None:
    """Reduction anchor: at d=0 the theta(B) composition collapses to EGARCH's geometric phi1^i.

    This is the clean check that the FI composition (phi^{-1}(B)(1-B)^{-d}) reduces correctly to
    short-memory EGARCH when the fractional operator is switched off.
    """
    for phi1 in (0.3, 0.55, 0.9):
        theta = theta_coefficients(phi1, 0.0, 40)
        geometric = phi1 ** np.arange(40, dtype=float)
        assert np.max(np.abs(theta - geometric)) < 1e-12


def test_scale_equivariance_is_exact_at_the_recursion_level() -> None:
    """A rescale r -> c*r with omega_sig -> omega_sig + ln(c^2), mu -> c*mu scales sigma by c.

    Tested on the recursion at fixed params (exact), since phi1/kappa/gamma/d and the theta tail are
    scale-invariant and omega_sig is a log-variance intercept (additive under rescaling).
    """
    returns, meta, _sigma = _load()
    fx = meta["params"]
    c = 10.0
    base = np.sqrt(fiegarch_recursion(_params(fx), returns, abs_moment=_E_ABS_NORM))
    shifted = _params(fx).copy()
    shifted[0] *= c  # mu
    shifted[1] += np.log(c**2)  # omega_sig
    scaled = np.sqrt(fiegarch_recursion(shifted, returns * c, abs_moment=_E_ABS_NORM))
    assert np.max(np.abs(scaled - c * base)) < 1e-10


def test_known_truth_recovery() -> None:
    """QMLE recovers planted (omega_sig, phi1, kappa, gamma, d) within a few SE (d's SE wider)."""
    true = {"mu": 0.0002, "omega_sig": -8.9, "phi1": 0.3, "kappa": -0.06, "gamma": 0.15, "d": 0.4}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = fiegarch_sim(9000, **true, rng=np.random.default_rng(0))
        fit = fit_fiegarch(returns, cond_dist="norm")
    se = fit.std_errors
    for name in ("omega_sig", "phi1", "kappa", "gamma", "d"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]


def test_fiegarch_sim_is_stationary_and_positive() -> None:
    """Simulated conditional SDs are positive and the log-variance mean is near omega_sig."""
    returns, sigma = fiegarch_sim(
        20000,
        mu=0.0,
        omega_sig=-8.8,
        phi1=0.3,
        kappa=-0.05,
        gamma=0.15,
        d=0.35,
        rng=np.random.default_rng(1),
    )
    assert np.all(sigma > 0.0)
    assert returns.shape == sigma.shape == (20000,)
    assert abs(np.mean(np.log(sigma**2)) - (-8.8)) < 0.2  # E[ln sigma^2] ~ omega_sig


def test_fiegarch_sim_supports_symmetric_distributions() -> None:
    """FIEGARCH's g reuses abs_moment (present on std), so simulation routes through it."""
    returns, sigma = fiegarch_sim(
        3000,
        omega_sig=-8.8,
        phi1=0.3,
        kappa=-0.05,
        gamma=0.15,
        d=0.35,
        cond_dist="std",
        dist_params=(6.0,),
        rng=np.random.default_rng(2),
    )
    assert returns.shape == sigma.shape == (3000,)
    assert np.all(sigma > 0.0)


def test_fiegarch_sim_rejects_bad_inputs() -> None:
    """Simulation rejects a non-positive size, |phi1| >= 1, and d outside [0, 1)."""
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="n must be positive"):
        fiegarch_sim(0, omega_sig=-8.8, phi1=0.3, kappa=0.0, gamma=0.15, d=0.3, rng=rng)
    with pytest.raises(ValueError, match=r"\|phi1\| < 1"):
        fiegarch_sim(100, omega_sig=-8.8, phi1=1.0, kappa=0.0, gamma=0.15, d=0.3, rng=rng)
    with pytest.raises(ValueError, match="0 <= d < 1"):
        fiegarch_sim(100, omega_sig=-8.8, phi1=0.3, kappa=0.0, gamma=0.15, d=1.0, rng=rng)


def test_norm_abs_moment_available() -> None:
    """The centering moment FIEGARCH uses (magnitude = E|eta|) is exposed by the normal."""
    assert np.isclose(get_distribution("norm").abs_moment(), _E_ABS_NORM, atol=1e-12)
