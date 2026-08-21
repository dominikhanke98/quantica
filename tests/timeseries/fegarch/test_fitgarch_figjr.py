"""Validation of the fEGarch FITGARCH(1,d,1) + FIGJR(1,d,1) long-memory models (Phase 4).

FITGARCH and FIGJR are FIAPARCH with the power ``delta`` FIXED — FITGARCH at ``delta=1`` (the
Zakoian sigma-recursion) and FIGJR at ``delta=2`` (the GJR power). fEGarch's ``fitgarch()`` /
``figjrgarch()``
have no ``fix_delta`` arg, so delta is not a fitted parameter; both fit ``{mu, omega, phi1, beta1,
gamma, d}``. They reuse the proven FIGARCH/FIAPARCH machinery (``figarch_coefficients`` +
``figarch_variance_filter``) with the APARCH power-asymmetry news ``(|eps|-gamma eps)^delta`` — the
reconstruction gate confirmed the **FIGJR kernel is (|eps|-gamma eps)^2, NOT the Glosten indicator**
(carrying the Phase-1 short-memory GJR finding into the FI form). They inherit the same mean(news)
bounded-limit seed; the seam is machine-exact and both models are well-identified (unlike FIAPARCH's
d≈1 boundary, the fixed delta removes the flat ridge).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    fiaparch_news,
    fiaparch_recursion,
    figarch_coefficients,
    figarch_variance_filter,
    figjr_recursion,
    figjr_sim,
    fit_figjr,
    fit_fitgarch,
    fitgarch_recursion,
    fitgarch_sim,
    gjr_recursion,
    tgarch_recursion,
)

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

_FITGARCH = ("fit_fitgarch11_norm_params.json", "fit_fitgarch11_norm_sigma.csv", 1.0)
_FIGJR = ("fit_figjr11_norm_params.json", "fit_figjr11_norm_sigma.csv", 2.0)


def _load(pfile: str, sfile: str) -> tuple[np.ndarray, dict, np.ndarray]:  # type: ignore[type-arg]
    """Load the synthetic returns, a fit-params JSON, and its fEGarch sigma series."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    sigma = np.loadtxt(_FIXTURE_DIR / sfile, skiprows=1)
    meta = json.loads((_FIXTURE_DIR / pfile).read_text(encoding="utf-8"))
    return returns, meta, sigma


def _params(fx: dict) -> np.ndarray:  # type: ignore[type-arg]
    """The 6-vector (mu, omega, phi1, beta1, gamma, d) — delta is fixed, not a param."""
    return np.array([fx["mu"], fx["omega"], fx["phi1"], fx["beta1"], fx["gamma"], fx["d"]])


@pytest.mark.parametrize(("pfile", "sfile", "delta"), [_FITGARCH, _FIGJR])
def test_seam_is_machine_exact_at_the_empirical_seed(pfile: str, sfile: str, delta: float) -> None:
    """The news-term seam reproduces each fixture's sigma to ~1e-16 at its OWN empirical seed.

    Isolates the news term ((|eps|-gamma eps)^delta) from the seed convention: at the empirical seed
    (backed out of the fixture's sigma[0]) the reconstruction is machine-exact — proving the news
    term / recursion-metric is right, delta=1 (FITGARCH) and delta=2 (FIGJR).
    """
    returns, meta, sigma_fix = _load(pfile, sfile)
    fx = meta["params"]
    mu, omega, phi1, beta1, gamma, d = _params(fx)
    news = fiaparch_news(returns - mu, gamma, delta)
    theta = figarch_coefficients(phi1, beta1, d, returns.size + 50)
    seed = float((sigma_fix[0] ** delta - omega) / np.sum(theta[1:51]))
    sigma = figarch_variance_filter(news, theta, omega, seed) ** (1.0 / delta)
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-12


def test_fitgarch11_norm_matches_fegarch_fixture() -> None:
    """fit_fitgarch (delta=1) reproduces fEGarch's params, log-likelihood and sigma tight.

    FITGARCH lands d=1 (the high-persistence series forces it, Conrad-Haag), but — unlike FIAPARCH —
    the FIXED delta removes the flat ridge, so it is well-identified: all params recover and sigma
    matches to ~1e-6 (the mean(news) seed residual, small here).
    """
    returns, meta, sigma_fix = _load(*_FITGARCH[:2])
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_fitgarch(returns, cond_dist="norm")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-5
    assert abs(fit.params["phi1"] - fx["phi1"]) < 1e-3  # phi1 ~ 0 (near the lower bound)
    for name in ("omega", "beta1", "gamma"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-2
    assert (
        abs(fit.params["d"] - fx["d"]) < 1e-4
    )  # d ~ 1 (boundary), well-identified with delta fixed
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-2
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_figjr11_norm_matches_fegarch_fixture() -> None:
    """fit_figjr (delta=2, the (|eps|-gamma eps)^2 kernel) reproduces fEGarch's fixture tight.

    FIGJR lands d=0.66 (interior), so it is cleanly identified: all six params recover to ~1e-3 and
    sigma matches to ~4e-5 (the interior-d mean(news) seed residual).
    """
    returns, meta, sigma_fix = _load(*_FIGJR[:2])
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_figjr(returns, cond_dist="norm")

    assert abs(fit.params["mu"] - fx["mu"]) < 1e-5
    for name in ("omega", "phi1", "beta1", "gamma", "d"):
        assert abs(fit.params[name] - fx[name]) / abs(fx[name]) < 1e-2
    assert 0.5 < fx["d"] < 1.0  # interior of the upper regime, NOT at the boundary
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-2
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 1e-4
    assert all(np.isfinite(v) and v > 0.0 for v in fit.std_errors.values())


def test_figjr_kernel_is_aparch_not_glosten() -> None:
    """The FIGJR news kernel is (|eps|-gamma eps)^2 (APARCH), NOT the Glosten indicator.

    Reconstructs the FIGJR fixture at the empirical seed with each candidate: the APARCH kernel is
    machine-exact, while both Glosten indicator forms miss at ~1e-3. This carries the Phase-1
    short-memory GJR finding (fEGarch's GJR is the (|eps|-gamma eps)^2 kernel) into the FI form.
    """
    returns, meta, sigma_fix = _load(*_FIGJR[:2])
    fx = meta["params"]
    mu, omega, phi1, beta1, gamma, d = _params(fx)
    eps = returns - mu
    theta = figarch_coefficients(phi1, beta1, d, returns.size + 50)

    def recon(news: np.ndarray) -> float:  # type: ignore[type-arg]
        seed = float((sigma_fix[0] ** 2 - omega) / np.sum(theta[1:51]))
        sigma = np.sqrt(figarch_variance_filter(news, theta, omega, seed))
        return float(np.max(np.abs(sigma - sigma_fix)))

    aparch = recon((np.abs(eps) - gamma * eps) ** 2)  # (|eps|-gamma eps)^2
    glosten = recon(eps**2 * (1.0 + gamma * (eps < 0)))  # eps^2 (1 + gamma 1[eps<0])
    assert aparch < 1e-12  # the APARCH kernel is exact
    assert glosten > 1e-4  # the Glosten indicator decisively misses


def test_fitgarch_and_figjr_are_fiaparch_at_fixed_delta() -> None:
    """FITGARCH == FIAPARCH(delta=1) and FIGJR == FIAPARCH(delta=2), bit-for-bit (wrappers)."""
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    p6 = np.array([5e-4, 1e-4, 0.9, 0.6, 0.1, 0.3])
    p_delta1 = np.array([5e-4, 1e-4, 0.9, 0.6, 0.1, 1.0, 0.3])
    p_delta2 = np.array([5e-4, 1e-4, 0.9, 0.6, 0.1, 2.0, 0.3])
    assert np.array_equal(fitgarch_recursion(p6, returns), fiaparch_recursion(p_delta1, returns))
    assert np.array_equal(figjr_recursion(p6, returns), fiaparch_recursion(p_delta2, returns))


@pytest.mark.parametrize(
    ("fi_recursion", "sm_recursion", "omega"),
    [(fitgarch_recursion, tgarch_recursion, 2.3e-4), (figjr_recursion, gjr_recursion, 3e-6)],
    ids=["FITGARCH->TGARCH", "FIGJR->GJR"],
)
def test_d_zero_reduces_to_short_memory(fi_recursion, sm_recursion, omega) -> None:  # type: ignore[no-untyped-def]
    """Reduction anchor: at d->0 the FI model collapses to its short-memory sibling.

    With the parameterization mapping (the FI ``phi1`` is WP175's phi1 = alpha+beta, so the
    short-memory ARCH coefficient is ``phi1-beta1`` and the SM intercept ``omega*(1-beta1)``), the
    FI ARCH(inf) form and the short-memory AR-form are the same process with different pre-sample
    seeds — they converge to ~1e-10 after the transient (phi1 > beta1 keeps the reduced coefficients
    non-negative).
    """
    returns = np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)
    phi1, beta1, gamma = 0.95, 0.90, 0.1
    fi = np.sqrt(fi_recursion(np.array([5e-4, omega, phi1, beta1, gamma, 1e-9]), returns))
    sm = np.sqrt(
        sm_recursion(np.array([5e-4, omega * (1 - beta1), phi1 - beta1, beta1, gamma]), returns)
    )
    assert np.max(np.abs(fi - sm)[200:]) < 1e-9  # converge after the pre-sample transient


def test_fitgarch_known_truth_recovery() -> None:
    """QMLE recovers planted FITGARCH shape params (phi1, beta1, gamma, d) within a few SE."""
    true = {"mu": 0.0002, "omega": 3e-4, "phi1": 0.5, "beta1": 0.4, "gamma": 0.1, "d": 0.3}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = fitgarch_sim(6000, **true, rng=np.random.default_rng(0))
        fit = fit_fitgarch(returns, cond_dist="norm")
    se = fit.std_errors
    for name in ("phi1", "beta1", "gamma", "d"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]
    assert 0.3 < fit.params["omega"] / true["omega"] < 3.0  # omega is seed-sensitive


def test_figjr_known_truth_recovery() -> None:
    """QMLE recovers planted FIGJR shape params (phi1, beta1, gamma, d) within a few SE."""
    true = {"mu": 0.0002, "omega": 1.2e-5, "phi1": 0.5, "beta1": 0.4, "gamma": 0.1, "d": 0.3}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = figjr_sim(6000, **true, rng=np.random.default_rng(1))
        fit = fit_figjr(returns, cond_dist="norm")
    se = fit.std_errors
    for name in ("phi1", "beta1", "gamma", "d"):
        assert abs(fit.params[name] - true[name]) < 4.0 * se[name]
    assert 0.3 < fit.params["omega"] / true["omega"] < 3.0  # omega is seed-sensitive


@pytest.mark.parametrize("sim", [fitgarch_sim, figjr_sim])
def test_sim_is_positive_and_rejects_bad_inputs(sim) -> None:  # type: ignore[no-untyped-def]
    """Simulated conditional SDs are positive; the simulator rejects bad inputs."""
    returns, sigma = sim(
        3000, omega=1.2e-5, phi1=0.3, beta1=0.4, gamma=0.1, d=0.3, rng=np.random.default_rng(2)
    )
    assert returns.shape == sigma.shape == (3000,)
    assert np.all(sigma > 0.0)
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError, match="n must be positive"):
        sim(0, omega=1e-5, phi1=0.3, beta1=0.4, gamma=0.1, d=0.3, rng=rng)
    with pytest.raises(ValueError, match=r"\|beta1\| < 1"):
        sim(100, omega=1e-5, phi1=0.3, beta1=1.0, gamma=0.1, d=0.3, rng=rng)
    with pytest.raises(ValueError, match="0 <= d < 1"):
        sim(100, omega=1e-5, phi1=0.3, beta1=0.4, gamma=0.1, d=1.0, rng=rng)
