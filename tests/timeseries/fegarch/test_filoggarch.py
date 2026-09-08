"""Validation of the fEGarch FILog-GARCH(1,d,1) long-memory model (numerical-validation, Phase 4).

FILog-GARCH is the Type-II counterpart of FIEGARCH: the fractionally-integrated Log-GARCH. Its
truncated MA(inf) loads the log-square innovation ``xi = ln(eta^2) - E[ln eta^2]`` (Log-GARCH
news) through ``gamma(B) = (1-phi1 B)^{-1}(1-B)^{-d}(1+psi1 B) - 1`` (WP171 Eqs. 10-11), gamma_0=0.
It reuses FIEGARCH's ``theta_coefficients`` (fracdiff_coeffs(-d)) + the ``(1+psi1 B)`` MA factor,
the Log-GARCH ``mean_log_sq`` moment, and the FIEGARCH MA(inf) presample.

**An effective-challenge finding (validated honestly here).** fEGarch's committed FILog-GARCH fit on
this series sits at a **strictly-dominated local optimum**: it reports ``mu=-0.003`` with
``loglik=7436``, but EVERY data-driven optimizer start -- the data mean, several random starts,
and even a start from fEGarch's OWN reported params -- escapes to a basin ``~+120`` log-likelihood
higher (``mu`` near the data mean); only an optimizer *placed* at ``mu=-0.003`` stays there.
So we validate in two parts: (1) the SEAM is machine-exact (the recursion at fEGarch's OWN params
reproduces the fixture sigma to ~1e-15 -- the model is correct); (2) our independent fit **strictly
dominates** fEGarch's local optimum. We do NOT match the dominated fixture params
-- that would validate a bad optimum. This is a model-validation win (fEGarch's optimizer, not the
model, landing poorly), recorded, not hidden.
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
    optimizer (the fixture's own fit is a dominated local optimum; see the strict-domination test).
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


def _loglik_from_start(
    returns: np.ndarray, tail: tuple[float, float, float]
) -> tuple[float, float]:  # type: ignore[type-arg]
    """Run the FILog-GARCH QMLE from a custom (phi1, psi1, d) start; return (loglik, mu) unscaled.

    Mirrors ``fit_filoggarch``'s scaling (mu at the data mean, omega_sig at the log sample variance)
    so different (phi1, psi1, d) starts probe the likelihood surface through the same engine.
    """
    from quantica.timeseries.fegarch.qmle import quasi_max_likelihood

    n = returns.size
    scale = float(np.std(returns))
    scaled = returns / scale
    dist = get_distribution("norm")
    mls = float(dist.mean_log_sq())
    var_start = (float(np.mean(scaled)), float(np.log(np.var(scaled, ddof=1))), *tail)
    bounds = ((-10.0, 10.0), (-50.0, 50.0), (-0.9999, 0.9999), (-0.9999, 0.9999), (1e-6, 0.9999))
    res = quasi_max_likelihood(
        scaled,
        lambda p, ret: filoggarch_recursion(p, ret, mean_log_sq=mls),
        dist,
        var_start=var_start,
        var_bounds=bounds,
        var_names=("mu", "omega_sig", "phi1", "psi1", "d"),
        mean=True,
    )
    return float(res.loglikelihood - n * np.log(scale)), float(res.params[0]) * scale


def test_fit_strictly_dominates_the_fegarch_local_optimum() -> None:
    """Effective challenge: EVERY sensible start beats fEGarch's dominated local-optimum fixture.

    fEGarch's fixture (mu=-0.003, loglik=7436) is a genuine but strictly-dominated local optimum:
    the data-mean start, several random starts, AND a start from fEGarch's own params all escape
    to a basin ~+120 log-likelihood higher (mu near the data mean); only an optimizer *pinned* at
    mu=-0.003 stays in the dominated basin. So it is fEGarch's optimizer, not the model (the seam is
    machine-exact), that lands poorly. We assert strict domination from every sensible start and do
    NOT match the suboptimal fixture params.
    """
    returns, meta, _sigma = _load()
    fixture_ll = meta["loglikelihood"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_filoggarch(returns, cond_dist="norm")  # the data-mean default start
        starts = {
            "random A": _loglik_from_start(returns, (0.5, -0.6, 0.4)),
            "random B": _loglik_from_start(returns, (0.1, -0.2, 0.2)),
            "fEGarch params": _loglik_from_start(
                returns, (meta["params"]["phi1"], meta["params"]["psi1"], meta["params"]["d"])
            ),
        }

    # the default (data-mean) fit strictly dominates by a wide margin, at sensible params
    assert fit.converged
    assert fit.loglikelihood > fixture_ll + 100.0
    data_mean = float(np.mean(returns))
    assert abs(fit.params["mu"] - data_mean) < abs(meta["params"]["mu"] - data_mean)
    assert abs(fit.params["mu"] - data_mean) < 5e-4
    assert 0.0 < fit.params["d"] < 0.5  # d interior
    assert (
        abs(fit.params["phi1"] + fit.params["psi1"]) > 0.1
    )  # coefficients separated, NOT the ridge
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())

    # EVERY other sensible start also strictly beats the fixture (unreachable from anywhere sane)
    for label, (ll, _mu) in starts.items():
        assert ll > fixture_ll + 100.0, f"{label} start ({ll:.2f}) did not dominate the fixture"


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


def test_filoggarch_non_norm_supported() -> None:
    """FILog-GARCH under non-norm now runs: its ``mean_log_sq`` centering is implemented (EGF).

    Structural inheritance of the EGF distribution infrastructure (fixture-validated on EGARCH).
    """
    returns, sigma = filoggarch_sim(
        100,
        omega_sig=-8.8,
        phi1=0.4,
        psi1=-0.5,
        d=0.3,
        cond_dist="std",
        dist_params=(6.0,),
        rng=np.random.default_rng(3),
    )
    assert returns.shape == sigma.shape == (100,)
    assert np.all(sigma > 0.0)
    assert np.isfinite(get_distribution("std").mean_log_sq((6.0,)))  # the centering is available


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
