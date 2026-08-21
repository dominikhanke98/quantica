"""Validation of the fEGarch FILog-GARCH(1,d,1) long-memory model (numerical-validation, Phase 4).

FILog-GARCH is the Type-II counterpart of FIEGARCH: the fractionally-integrated Log-GARCH. Its
truncated MA(inf) loads the log-square innovation ``xi = ln(eta^2) - E[ln eta^2]`` (Log-GARCH
news) through ``gamma(B) = (1-phi1 B)^{-1}(1-B)^{-d}(1+psi1 B) - 1`` (WP171 Eqs. 10-11), gamma_0=0.
It reuses FIEGARCH's ``theta_coefficients`` (fracdiff_coeffs(-d)) + the ``(1+psi1 B)`` MA factor,
the Log-GARCH ``mean_log_sq`` moment, and the FIEGARCH MA(inf) presample.

**An effective-challenge finding (validated honestly here).** fEGarch's committed FILog-GARCH fit on
this series is NON-CONVERGED: it reports ``mu=-0.003`` with ``loglik=7436``, but a
properly-converged clean-room fit finds ``mu`` near the data mean with ``loglik~7556`` (**+120
higher**). So we validate
in two parts: (1) the SEAM is machine-exact (the recursion at fEGarch's OWN params reproduces the
fixture sigma to ~1e-15 -- the model is correct); (2) our independent fit BEATS fEGarch's suboptimal
fixture. We do NOT assert a param-by-param match to the non-converged fixture -- that would
validate a bad optimum. This is a model-validation win, recorded, not hidden.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    filoggarch_gamma_coefficients,
    filoggarch_recursion,
    filoggarch_sim,
    fit_filoggarch,
    get_distribution,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"
_MEAN_LOG_SQ_NORM = -np.euler_gamma - np.log(2.0)  # E[ln eta^2] for the standard normal


def _load() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit-params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_filoggarch11_norm_sigma.csv", skiprows=1)
    meta = json.loads(
        (_FIXTURE_DIR / "fit_filoggarch11_norm_params.json").read_text(encoding="utf-8")
    )
    return returns, meta, sigma


def _params(fx: dict) -> np.ndarray:  # type: ignore[type-arg]
    """The 5-vector (mu, omega_sig, phi1, psi1, d) from a fixture params dict."""
    return np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["psi1"], fx["d"]])


def test_seam_is_machine_exact_at_fegarch_params() -> None:
    """The Type-II fractional recursion reproduces the fixture sigma to ~1e-15 at fEGarch's params.

    This is the core correctness proof: gamma(B) = (1-phi1 B)^{-1}(1-B)^{-d}(1+psi1 B)-1 with the xi
    log-square news, mean_log_sq centering, and the MA(inf) presample. It is independent of the
    optimizer (the fixture's own fit is non-converged -- see the fit-beats test).
    """
    returns, meta, sigma_fix = _load()
    sigma = np.sqrt(
        filoggarch_recursion(_params(meta["params"]), returns, mean_log_sq=_MEAN_LOG_SQ_NORM)
    )
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-12


def test_ma_presample_tell_sigma0_is_exp_half_omega_sig() -> None:
    """The MA(inf) presample gives sigma[0] = exp(omega_sig/2) exactly (zero pre-sample xi)."""
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    sigma = np.sqrt(filoggarch_recursion(_params(fx), returns, mean_log_sq=_MEAN_LOG_SQ_NORM))
    assert np.isclose(sigma[0], np.exp(fx["omega_sig"] / 2.0), atol=1e-14)
    assert np.isclose(
        sigma_fix[0], np.exp(fx["omega_sig"] / 2.0), atol=1e-8
    )  # fEGarch's own sigma[0]


def test_fit_beats_the_non_converged_fegarch_fixture() -> None:
    """Effective challenge: our converged fit BEATS fEGarch's non-converged fixture by ~+120 loglik.

    fEGarch's fixture reports mu=-0.003, loglik=7436 -- but its gradient in mu is far from zero, so
    it is not the ML optimum. A data-mean-started clean-room fit converges to mu near the data mean
    with loglik~7556 (+120 higher), at sensible params (d interior, coefficients separated). We
    assert our fit is strictly better and more sensible; we do NOT match the suboptimal fixture.
    """
    returns, meta, _sigma = _load()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_filoggarch(returns, cond_dist="norm")

    assert fit.converged
    # strictly higher likelihood than fEGarch's fixture (by a wide margin)
    assert fit.loglikelihood > meta["loglikelihood"] + 100.0
    # our mu is near the data mean; fEGarch's (-0.003) is far from it
    data_mean = float(np.mean(returns))
    assert abs(fit.params["mu"] - data_mean) < abs(meta["params"]["mu"] - data_mean)
    assert abs(fit.params["mu"] - data_mean) < 5e-4
    # d interior, coefficients well separated (the fractional d absorbed the persistence)
    assert 0.0 < fit.params["d"] < 0.5
    assert abs(fit.params["phi1"] + fit.params["psi1"]) > 0.1  # NOT the near-common-root ridge
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_gamma_composition() -> None:
    """gamma_0 = 0, and gamma_1 = d + phi1 + psi1 (the Type-II analogue of FIEGARCH's d+phi1)."""
    for phi1, psi1, d in ((0.3, -0.5, 0.29), (0.5, -0.2, 0.4)):
        gamma = filoggarch_gamma_coefficients(phi1, psi1, d, 10)
        assert gamma[0] == 0.0
        assert np.isclose(gamma[1], d + phi1 + psi1, atol=1e-12)


def test_d_zero_reduces_to_short_memory_loggarch() -> None:
    """Reduction anchor: at d=0 gamma(B) collapses to Log-GARCH's gamma_i = (psi1+phi1) phi1^{i-1}.

    The short-memory Log-GARCH ARMA (with MA(inf) coefficients (psi1+phi1) phi1^{i-1} on xi)
    has MA(inf) coefficients (psi1+phi1) phi1^{i-1}; switching off the fractional operator recovers
    exactly that.
    """
    for phi1, psi1 in ((0.5, -0.3), (0.9, -0.8)):
        gamma = filoggarch_gamma_coefficients(phi1, psi1, 0.0, 30)
        loggarch = np.zeros(30)
        loggarch[1:] = (psi1 + phi1) * phi1 ** np.arange(29)
        assert np.max(np.abs(gamma - loggarch)) < 1e-12


def test_scale_equivariance_is_exact_at_the_recursion_level() -> None:
    """A rescale r -> c*r with mu -> c*mu, omega_sig -> omega_sig + ln(c^2) scales sigma by c.

    xi is a function of the standardized eta = (r-mu)/sigma, hence scale-invariant; omega_sig is a
    log-variance intercept (additive shift); phi1/psi1/d are scale-invariant.
    """
    returns, meta, _sigma = _load()
    fx = meta["params"]
    c = 10.0
    base = np.sqrt(filoggarch_recursion(_params(fx), returns, mean_log_sq=_MEAN_LOG_SQ_NORM))
    shifted = _params(fx).copy()
    shifted[0] *= c  # mu
    shifted[1] += np.log(c**2)  # omega_sig
    scaled = np.sqrt(filoggarch_recursion(shifted, returns * c, mean_log_sq=_MEAN_LOG_SQ_NORM))
    assert np.max(np.abs(scaled - c * base)) < 1e-10


def test_known_truth_recovery() -> None:
    """QMLE recovers planted (omega_sig, phi1, psi1, d) within a few SE (interior d, no ridge)."""
    true = {"mu": 0.0003, "omega_sig": -8.8, "phi1": 0.4, "psi1": -0.5, "d": 0.3}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = filoggarch_sim(9000, **true, rng=np.random.default_rng(0))
        fit = fit_filoggarch(returns, cond_dist="norm")
    se = fit.std_errors
    for name in ("omega_sig", "phi1", "psi1", "d"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]


def test_filoggarch_sim_is_stationary_and_positive() -> None:
    """Simulated conditional SDs are positive and the log-variance mean is near omega_sig."""
    returns, sigma = filoggarch_sim(
        20000, mu=0.0, omega_sig=-8.8, phi1=0.4, psi1=-0.5, d=0.3, rng=np.random.default_rng(1)
    )
    assert np.all(sigma > 0.0)
    assert returns.shape == sigma.shape == (20000,)
    # E[ln sigma^2] = omega_sig; the sample mean mixes slowly at d=0.3 (long memory), so ~0.5 band.
    assert abs(np.mean(np.log(sigma**2)) - (-8.8)) < 0.5


def test_filoggarch_non_norm_is_deferred() -> None:
    """FILog-GARCH under non-norm is deferred: xi needs mean_log_sq (only norm, as MLog-GARCH)."""
    with pytest.raises(NotImplementedError):
        filoggarch_sim(
            100,
            omega_sig=-8.8,
            phi1=0.4,
            psi1=-0.5,
            d=0.3,
            cond_dist="std",
            dist_params=(6.0,),
            rng=np.random.default_rng(3),
        )
    with pytest.raises(NotImplementedError):
        get_distribution("std").mean_log_sq((6.0,))


def test_filoggarch_sim_rejects_bad_inputs() -> None:
    """Simulation rejects a non-positive size, |phi1| >= 1, and d outside [0, 1)."""
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="n must be positive"):
        filoggarch_sim(0, omega_sig=-8.8, phi1=0.4, psi1=-0.5, d=0.3, rng=rng)
    with pytest.raises(ValueError, match=r"\|phi1\| < 1"):
        filoggarch_sim(100, omega_sig=-8.8, phi1=1.0, psi1=-0.5, d=0.3, rng=rng)
    with pytest.raises(ValueError, match="0 <= d < 1"):
        filoggarch_sim(100, omega_sig=-8.8, phi1=0.4, psi1=-0.5, d=1.0, rng=rng)


def test_norm_mean_log_sq() -> None:
    """The log-square centering FILog-GARCH uses (E[ln eta^2] = -gamma_E - ln2) for the normal."""
    assert np.isclose(get_distribution("norm").mean_log_sq(), _MEAN_LOG_SQ_NORM, atol=1e-8)
    assert np.isclose(_MEAN_LOG_SQ_NORM, -1.2703628, atol=1e-6)
