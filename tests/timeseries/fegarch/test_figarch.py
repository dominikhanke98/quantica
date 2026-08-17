"""Validation of the fEGarch FIGARCH(1,d,1) long-memory model (numerical-validation, Phase 4).

FIGARCH is the first **variance-recursion** long-memory model — the fractional operator enters the
conditional-variance polynomial (WP175 Eq. 2.4), not the EGF log-variance. The recursion is
``sigma^2_t = omega + sum_i theta_i eps2_{t-i}``, ``theta(B) = 1-(1-phi1 B)(1-B)^d/(1-beta1 B)``,
the intercept ``omega`` used DIRECTLY (not BBM/arch's ``(1-beta1)^-1 omega``), and no feedback
(sigma^2 is a pure linear filter of observed eps2). The pre-sample is machine-resolved: the first 50
pre-sample eps2 are seeded at ``Var(r, ddof=1)`` (both the count 50 and ddof=1 are decisive). The
fixture reconstructs to 4.86e-17; the fit matches at ~1e-5.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    FIGARCH_PRESAMPLE,
    figarch_coefficients,
    figarch_recursion,
    figarch_sim,
    figarch_variance_filter,
    fit_figarch,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"


def _load() -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, the fEGarch fit-params JSON, and the fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / "fit_figarch11_norm_sigma.csv", skiprows=1)
    meta = json.loads((_FIXTURE_DIR / "fit_figarch11_norm_params.json").read_text(encoding="utf-8"))
    return returns, meta, sigma


def _params(fx: dict) -> np.ndarray:  # type: ignore[type-arg]
    """The 5-vector (mu, omega, phi1, beta1, d) from a fixture params dict."""
    return np.array([fx["mu"], fx["omega"], fx["phi1"], fx["beta1"], fx["d"]])


def test_figarch11_norm_matches_fegarch_fixture() -> None:
    """fit_figarch reproduces fEGarch's five params (incl. d), log-likelihood, AIC/BIC and sigma."""
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_figarch(returns, cond_dist="norm")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-6
    for name in ("omega", "phi1", "beta1", "d"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-3
    assert fit.params["omega"] > 0.0  # variance intercept is positive
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - meta["aic"]) < 1e-6
    assert abs(fit.bic - meta["bic"]) < 1e-6
    deviation = np.abs(fit.conditional_volatility - sigma_fix)
    assert deviation.max() < 1e-5
    assert np.max(deviation / sigma_fix) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_recursion_reproduces_fixture_sigma_at_reported_params() -> None:
    """The ω-direct ARCH(∞) filter reproduces the fixture sigma to machine precision (~4.86e-17).

    Isolates the whole spec (theta = fracdiff(+d) ⊛ ARMA, omega-direct intercept, 50-term Var
    pre-sample) from the optimizer: at fEGarch's own params it matches to floating-point noise.
    """
    returns, meta, sigma_fix = _load()
    sigma = np.sqrt(figarch_recursion(_params(meta["params"]), returns))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-12


def test_presample_convention_is_50_terms_var_ddof1() -> None:
    """The resolved pre-sample: exactly 50 terms at Var(ddof=1). Count and ddof are both decisive.

    50 terms + ddof=1 reconstructs to ~1e-16; ddof=0 misses at ~2e-6, and 49 or 51 terms miss at
    ~8e-6 — a sharp, unique convention, not an approximate warm-up.
    """
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    n = returns.size
    eps2 = (returns - fx["mu"]) ** 2
    theta = figarch_coefficients(fx["phi1"], fx["beta1"], fx["d"], n + 60)
    var1 = float(np.var(returns, ddof=1))
    var0 = float(np.var(returns, ddof=0))

    exact = figarch_variance_filter(eps2, theta, fx["omega"], var1, n_presample=50)
    assert np.max(np.abs(np.sqrt(exact) - sigma_fix)) < 1e-12  # 50 + ddof=1 -> machine precision

    ddof0 = figarch_variance_filter(eps2, theta, fx["omega"], var0, n_presample=50)
    assert np.max(np.abs(np.sqrt(ddof0) - sigma_fix)) > 1e-7  # ddof=0 decisively misses
    for npre in (49, 51):
        s = figarch_variance_filter(eps2, theta, fx["omega"], var1, n_presample=npre)
        assert np.max(np.abs(np.sqrt(s) - sigma_fix)) > 1e-7  # wrong count decisively misses


def test_theta_composition() -> None:
    """theta_0 = 0 (no contemporaneous term) and theta_1 = d + phi1 - beta1."""
    fx = _load()[1]["params"]
    theta = figarch_coefficients(fx["phi1"], fx["beta1"], fx["d"], 10)
    assert theta[0] == 0.0
    assert np.isclose(theta[1], fx["d"] + fx["phi1"] - fx["beta1"], atol=1e-12)


def test_d_zero_reduces_to_short_memory_garch() -> None:
    """Reduction anchor: at d=0 the ARCH(∞) collapses to GARCH's theta_i = (phi1-beta1) beta1^{i-1}.

    A check that the FI composition reduces correctly when the fractional operator is switched off
    (using phi1 > beta1 so the reduced GARCH ARCH coefficients are non-negative).
    """
    for phi1, beta1 in ((0.9, 0.6), (0.5, 0.3)):
        theta = figarch_coefficients(phi1, beta1, 0.0, 30)
        garch = np.zeros(30)
        garch[1:] = (phi1 - beta1) * beta1 ** np.arange(29)
        assert np.max(np.abs(theta - garch)) < 1e-12


def test_scale_equivariance_is_exact_at_the_recursion_level() -> None:
    """A rescale r -> c*r with mu -> c*mu and omega -> c^2*omega scales sigma by c exactly.

    omega is a VARIANCE intercept, so it scales multiplicatively by c^2 (unlike the EGF omega_sig,
    which shifts additively); phi1/beta1/d are scale-invariant.
    """
    returns, meta, _sigma = _load()
    fx = meta["params"]
    c = 10.0
    base = np.sqrt(figarch_recursion(_params(fx), returns))
    shifted = _params(fx).copy()
    shifted[0] *= c  # mu
    shifted[1] *= c**2  # omega (variance intercept)
    scaled = np.sqrt(figarch_recursion(shifted, returns * c))
    assert np.max(np.abs(scaled - c * base)) < 1e-10


def test_known_truth_recovery() -> None:
    """QMLE recovers planted (omega, phi1, beta1, d) from a simulation within a few SE."""
    true = {"mu": 0.0003, "omega": 1.5e-5, "phi1": 0.25, "beta1": 0.6, "d": 0.4}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = figarch_sim(6000, **true, rng=np.random.default_rng(0))
        fit = fit_figarch(returns, cond_dist="norm")
    se = fit.std_errors
    for name in ("omega", "phi1", "beta1", "d"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]


def test_arch_cross_check_documented_divergence() -> None:
    """arch's FIGARCH is a documented-divergence anchor (~2e-3), NOT a machine-precision benchmark.

    arch uses the BBM ``(1-beta)^-1 omega`` intercept and an EWMA backcast, so at the fEGarch params
    it diverges from the fixture at ~2e-3 — confirming the general BBM form while showing fEGarch's
    ω-direct + 50-term-Var convention is the distinguishing (and primary) anchor. Skip-safe.
    """
    arch_univariate = pytest.importorskip("arch.univariate")
    returns, meta, sigma_fix = _load()
    fx = meta["params"]
    resid = returns - fx["mu"]
    vol = arch_univariate.FIGARCH(p=1, q=1, truncation=1000)
    params = np.array(
        [fx["omega"], fx["phi1"], fx["d"], fx["beta1"]]
    )  # arch order [omega,phi,d,beta]
    sigma2 = np.zeros(returns.size)
    h = vol.compute_variance(params, resid, sigma2, vol.backcast(resid), vol.variance_bounds(resid))
    dev = np.max(np.abs(np.sqrt(h) - sigma_fix))
    assert 1e-4 < dev < 1e-2  # general BBM form agrees to ~2e-3; NOT machine precision


def test_figarch_sim_is_positive_and_shaped() -> None:
    """Simulated conditional SDs are positive and correctly shaped."""
    returns, sigma = figarch_sim(
        3000, mu=0.0, omega=1.2e-5, phi1=0.25, beta1=0.6, d=0.35, rng=np.random.default_rng(1)
    )
    assert returns.shape == sigma.shape == (3000,)
    assert np.all(sigma > 0.0)


def test_figarch_sim_supports_symmetric_distributions() -> None:
    """FIGARCH's variance recursion needs no distribution moment, so any innovation law works."""
    returns, sigma = figarch_sim(
        2000,
        omega=1.2e-5,
        phi1=0.25,
        beta1=0.6,
        d=0.35,
        cond_dist="std",
        dist_params=(6.0,),
        rng=np.random.default_rng(2),
    )
    assert returns.shape == sigma.shape == (2000,)
    assert np.all(sigma > 0.0)


def test_figarch_sim_rejects_bad_inputs() -> None:
    """Simulation rejects a non-positive size, omega <= 0, |beta1| >= 1, and d outside [0, 1)."""
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="n must be positive"):
        figarch_sim(0, omega=1e-5, phi1=0.25, beta1=0.6, d=0.3, rng=rng)
    with pytest.raises(ValueError, match="omega > 0"):
        figarch_sim(100, omega=0.0, phi1=0.25, beta1=0.6, d=0.3, rng=rng)
    with pytest.raises(ValueError, match=r"\|beta1\| < 1"):
        figarch_sim(100, omega=1e-5, phi1=0.25, beta1=1.0, d=0.3, rng=rng)
    with pytest.raises(ValueError, match="0 <= d < 1"):
        figarch_sim(100, omega=1e-5, phi1=0.25, beta1=0.6, d=1.0, rng=rng)


def test_presample_constant_is_50() -> None:
    """The exported FIGARCH_PRESAMPLE constant is fEGarch's presample=50."""
    assert FIGARCH_PRESAMPLE == 50
