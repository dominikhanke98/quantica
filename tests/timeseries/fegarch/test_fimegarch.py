"""Validation of the fEGarch FIMEGARCH(1,d,1) long-memory model (numerical-validation, Phase 4).

FIMEGARCH is the fractionally-integrated MEGARCH — the exact long-memory analogue of MEGARCH,
sharing FIEGARCH's ``theta(B) = (1-phi1 B)^{-1}(1-B)^{-d}`` composition and MA(inf) pre-sample and
differing only in the Type-I constant-set: modulus-log asymmetry ``sgn(eta)*ln(|eta|+1)`` with the
EGARCH ``|eta|`` magnitude (``MEGARCH_CONSTANTS``, so ``abs_moment`` centering). Because it uses the
``|eta|`` magnitude, its fitted ``gamma`` sits near EGARCH/MEGARCH's (~0.17), decisively below
FIMLog-GARCH's (~0.32). Well-identified, so it matches the fEGarch fixture at FIEGARCH tier.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    fimegarch_recursion,
    fimegarch_sim,
    fit_fimegarch,
    get_distribution,
    megarch_recursion,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"
_E_ABS_NORM = float(np.sqrt(2.0 / np.pi))  # E|eta| for the standard normal


def _load() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit-params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_fimegarch11_norm_sigma.csv", skiprows=1)
    meta = json.loads(
        (_FIXTURE_DIR / "fit_fimegarch11_norm_params.json").read_text(encoding="utf-8")
    )
    return returns, meta, sigma


def _params(fx: dict) -> np.ndarray:  # type: ignore[type-arg]
    """The 6-vector (mu, omega_sig, phi1, kappa, gamma, d) from a fixture params dict."""
    return np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["kappa"], fx["gamma"], fx["d"]])


def test_fimegarch11_norm_matches_fegarch_fixture() -> None:
    """fit_fimegarch reproduces fEGarch's six params (incl. d), loglik, AIC/BIC and sigma."""
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # transient exp-overflow at rejected optimizer points
        fit = fit_fimegarch(returns, cond_dist="norm")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    for name in ("omega_sig", "phi1", "kappa", "gamma", "d"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-3
    assert fit.params["kappa"] < 0.0 < fit.params["gamma"]  # leverage asymmetry, positive magnitude
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_gamma_reflects_the_absolute_value_magnitude() -> None:
    """FIMEGARCH keeps the ``|eta|`` magnitude, so gamma ~ 0.17 — the constant-set tell.

    If the magnitude were mis-wired as the log-modulus ``ln(|eta|+1)`` (FIMLog-GARCH's), gamma would
    return near 0.32 instead. This catches a swapped constant-set / centering moment.
    """
    _returns, meta, _sigma = _load()
    assert meta["params"]["gamma"] < 0.22  # decisively NOT ~0.32 (the log-modulus magnitude)
    assert 0.14 < meta["params"]["gamma"] < 0.20


def test_fitted_d_is_in_the_upper_long_memory_regime() -> None:
    """The fitted d ~ 0.744 is in (0.5, 1) — confirms the QMLE bound is NOT clamped at 0.5."""
    _returns, meta, _sigma = _load()
    assert 0.5 < meta["params"]["d"] < 1.0
    assert meta["params"]["phi1"] < 0.5  # persistence reparameterized into the theta tail


def test_recursion_reproduces_fixture_sigma_at_reported_params() -> None:
    """The theta(B) composition + MEGARCH g reproduce the sigma series to machine order.

    Isolates the whole spec (theta via the Phase-3 engine, modulus-log asymmetry + |eta| magnitude,
    omega_sig intercept, g-presample 0) from the optimizer: at fEGarch params it matches to ~1e-15.
    """
    returns, meta, sigma_fix = _load()
    sigma = np.sqrt(fimegarch_recursion(_params(meta["params"]), returns, abs_moment=_E_ABS_NORM))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-10


def test_ma_presample_tell_sigma0_is_exp_half_omega_sig() -> None:
    """The MA(inf) presample gives sigma[0] = exp(omega_sig/2) exactly (no ln(Var) seed)."""
    returns, meta, _sigma = _load()
    fx = meta["params"]
    sigma = np.sqrt(fimegarch_recursion(_params(fx), returns, abs_moment=_E_ABS_NORM))
    assert np.isclose(sigma[0], np.exp(fx["omega_sig"] / 2.0), atol=1e-14)


def test_d_zero_reduces_to_short_memory_megarch() -> None:
    """Reduction anchor: at d->0 FIMEGARCH collapses to short-memory MEGARCH after the pre-sample.

    The FI MA(inf) form (theta_i -> phi1^i) and MEGARCH's AR form are the same stationary process
    with different pre-sample seeds, so they converge exponentially; after t=100 they agree to
    ~1e-9 (realized ~1e-11).
    """
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    p_short = np.array([3e-4, -8.9, 0.4, -0.05, 0.15])
    p_fi = np.array([3e-4, -8.9, 0.4, -0.05, 0.15, 1e-9])  # d ~ 0
    fi = np.sqrt(fimegarch_recursion(p_fi, returns, abs_moment=_E_ABS_NORM))
    short = np.sqrt(megarch_recursion(p_short, returns, abs_moment=_E_ABS_NORM))
    assert np.max(np.abs(fi[100:] - short[100:])) < 1e-9


def test_scale_equivariance_is_exact_at_the_recursion_level() -> None:
    """A rescale r -> c*r with omega_sig -> omega_sig + ln(c^2), mu -> c*mu scales sigma by c."""
    returns, meta, _sigma = _load()
    fx = meta["params"]
    c = 10.0
    base = np.sqrt(fimegarch_recursion(_params(fx), returns, abs_moment=_E_ABS_NORM))
    shifted = _params(fx).copy()
    shifted[0] *= c  # mu
    shifted[1] += np.log(c**2)  # omega_sig
    scaled = np.sqrt(fimegarch_recursion(shifted, returns * c, abs_moment=_E_ABS_NORM))
    assert np.max(np.abs(scaled - c * base)) < 1e-10


def test_known_truth_recovery() -> None:
    """QMLE recovers planted (omega_sig, phi1, kappa, gamma, d) within a few SE (d's SE wider)."""
    true = {"mu": 0.0002, "omega_sig": -8.9, "phi1": 0.3, "kappa": -0.06, "gamma": 0.15, "d": 0.4}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = fimegarch_sim(9000, **true, rng=np.random.default_rng(0))
        fit = fit_fimegarch(returns, cond_dist="norm")
    se = fit.std_errors
    for name in ("omega_sig", "phi1", "kappa", "gamma", "d"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]


def test_fimegarch_sim_supports_symmetric_distributions() -> None:
    """FIMEGARCH's magnitude reuses abs_moment (present on std), so simulation routes through it."""
    returns, sigma = fimegarch_sim(
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


def test_fimegarch_sim_rejects_bad_inputs() -> None:
    """Simulation rejects a non-positive size, |phi1| >= 1, and d outside [0, 1)."""
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="n must be positive"):
        fimegarch_sim(0, omega_sig=-8.8, phi1=0.3, kappa=0.0, gamma=0.15, d=0.3, rng=rng)
    with pytest.raises(ValueError, match=r"\|phi1\| < 1"):
        fimegarch_sim(100, omega_sig=-8.8, phi1=1.0, kappa=0.0, gamma=0.15, d=0.3, rng=rng)
    with pytest.raises(ValueError, match="0 <= d < 1"):
        fimegarch_sim(100, omega_sig=-8.8, phi1=0.3, kappa=0.0, gamma=0.15, d=1.0, rng=rng)


def test_norm_abs_moment_available() -> None:
    """The centering moment FIMEGARCH uses (magnitude = E|eta|) is exposed by the normal."""
    assert np.isclose(get_distribution("norm").abs_moment(), _E_ABS_NORM, atol=1e-12)
