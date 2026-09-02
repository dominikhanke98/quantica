"""MEGARCH / MLog-GARCH / Log-GARCH under non-normal distributions — the EGF family (Phase 5).

The three remaining short-memory EGF models inherit the EGF distribution infrastructure proven on
EGARCH, plus **one new piece the EGARCH proof could not exercise**: the modulus-log **asymmetry**
centering ``E[sgn(eta) ln(|eta|+1)]`` (``mean_signed_log_modulus``, the 4th EGF moment). Odd, so
it vanishes for every symmetric innovation but is nonzero under skew -- why MEGARCH/MLog-GARCH
(whose ``g_asy`` is the modulus-log ``zeta(eta)``) missed under skew until wired, while EGARCH
(``g_asy = eta``, ``E[eta] = 0``) and Log-GARCH (Type-II, no asymmetry term) were already exact.

Validation: the reconstruction seam is machine-exact for all 16 converging fixtures (the 6 skewed
megarch/mloggarch now pin at ~4e-17). MEGARCH/MLog-GARCH are well identified and match tightly (the
gamma ~1.8x tell carries across distributions); Log-GARCH sits on its near-common-root ridge, so its
fit reaches-or-beats the fixture (often dominating), validated by the seam, not param-match. The
5 continuous-df cases fEGarch's optimizer failed (megarch/mloggarch std/sstd, loggarch std) are
validated by known-truth — our optimizer converges on all 5.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    fit_loggarch,
    fit_megarch,
    fit_mloggarch,
    get_distribution,
    loggarch_recursion,
    loggarch_sim,
    megarch_recursion,
    megarch_sim,
    mloggarch_recursion,
    mloggarch_sim,
)
from quantica.timeseries.fegarch.distributions import AverageLaplace, FernandezSteelSkew

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

# model -> distributions with a committed fixture (the ones that converged in fEGarch)
_TYPE1_DISTS = ["ged", "ald", "snorm", "sged", "sald"]  # megarch/mloggarch (std/sstd failed)
_LOGGARCH_DISTS = ["ged", "ald", "snorm", "sstd", "sged", "sald"]  # loggarch (only std failed)


def _returns() -> np.ndarray:  # type: ignore[type-arg]
    """The committed synthetic return series."""
    return np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)


def _load(model: str, dist: str) -> tuple[dict, np.ndarray]:  # type: ignore[type-arg]
    """Load a fixture's params dict and sigma series."""
    meta = json.loads(
        (_FIXTURE_DIR / f"fit_{model}11_{dist}_params.json").read_text(encoding="utf-8")
    )
    sigma = np.loadtxt(_FIXTURE_DIR / f"fit_{model}11_{dist}_sigma.csv", skiprows=1)
    return meta, sigma


def _dist_and_params(dist: str, fx: dict) -> tuple[object, tuple[float, ...] | None]:  # type: ignore[type-arg]
    """Distribution instance + shape params for a fixture's reported shape/skew."""
    if dist == "ged":
        return get_distribution("ged"), (fx["shape"],)
    if dist == "ald":
        return AverageLaplace(p=int(fx["P"])), None
    if dist == "snorm":
        return get_distribution("snorm"), (fx["skew"],)
    if dist == "sstd":
        return get_distribution("sstd"), (fx["df"], fx["skew"])
    if dist == "sged":
        return get_distribution("sged"), (fx["shape"], fx["skew"])
    return FernandezSteelSkew(AverageLaplace(p=int(fx["P"]))), (fx["skew"],)  # sald


# --------------------------------------------------------------------------- #
# The 4th EGF moment: signed modulus-log asymmetry centering
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("code", ["norm", "std", "ged", "ald"])
def test_signed_log_modulus_is_zero_for_symmetric_bases(code: str) -> None:
    """E[sgn(z) ln(|z|+1)] is identically 0 for every symmetric base (odd integrand, even f)."""
    dist = get_distribution(code) if code != "ald" else AverageLaplace(p=5)
    shape = {"norm": None, "std": (6.0,), "ged": (1.2,), "ald": None}[code]
    assert dist.mean_signed_log_modulus(shape) == 0.0


@pytest.mark.parametrize("skew_code", ["snorm", "sged", "sald"])
def test_signed_log_modulus_nonzero_under_skew_zero_at_unit_skew(skew_code: str) -> None:
    """The asymmetry centering is nonzero under skew, reduces to 0 at skew = 1 (odd-function)."""
    base_shape: tuple[float, ...] = {"snorm": (), "sged": (1.3,), "sald": ()}[skew_code]
    dist = (
        get_distribution(skew_code)
        if skew_code != "sald"
        else FernandezSteelSkew(AverageLaplace(p=5))
    )
    assert abs(dist.mean_signed_log_modulus((*base_shape, 0.85))) > 1e-4  # nonzero under skew
    assert abs(dist.mean_signed_log_modulus((*base_shape, 1.0))) < 1e-9  # -> 0 at skew = 1


# --------------------------------------------------------------------------- #
# Reconstruction seam: recursion + centering moments, machine-exact (all 16)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("dist", _TYPE1_DISTS)
def test_megarch_seam_is_machine_exact(dist: str) -> None:
    """MEGARCH seam: E|eta| magnitude + E[sgn(eta) ln(|eta|+1)] asymmetry reproduce sigma to ~1e-15.

    The skewed cases exercise the 4th EGF moment (the asymmetry centering) — the piece EGARCH's
    abs_moment-only proof did not cover.
    """
    fx_meta, sigma_fix = _load("megarch", dist)
    fx = fx_meta["params"]
    d, p = _dist_and_params(dist, fx)
    vp = np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["kappa"], fx["gamma"]])
    sigma = np.sqrt(
        megarch_recursion(
            vp,
            _returns(),
            abs_moment=d.abs_moment(p),
            mean_signed_log_modulus=d.mean_signed_log_modulus(p),
        )
    )
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-8


@pytest.mark.parametrize("dist", _TYPE1_DISTS)
def test_mloggarch_seam_is_machine_exact(dist: str) -> None:
    """MLog-GARCH seam: E[ln(|eta|+1)] magnitude + signed-log-modulus asymmetry reproduce sigma."""
    fx_meta, sigma_fix = _load("mloggarch", dist)
    fx = fx_meta["params"]
    d, p = _dist_and_params(dist, fx)
    vp = np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["kappa"], fx["gamma"]])
    sigma = np.sqrt(
        mloggarch_recursion(
            vp,
            _returns(),
            mean_log_modulus=d.mean_log_modulus(p),
            mean_signed_log_modulus=d.mean_signed_log_modulus(p),
        )
    )
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-8


@pytest.mark.parametrize("dist", _LOGGARCH_DISTS)
def test_loggarch_seam_is_machine_exact(dist: str) -> None:
    """Log-GARCH seam: the E[ln eta^2] centering reproduces sigma (Type-II, no asymmetry)."""
    fx_meta, sigma_fix = _load("loggarch", dist)
    fx = fx_meta["params"]
    d, p = _dist_and_params(dist, fx)
    vp = np.array([fx["mu"], fx["omega_sig"], fx["phi1"], fx["psi1"]])
    sigma = np.sqrt(loggarch_recursion(vp, _returns(), mean_log_sq=d.mean_log_sq(p)))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-8


# --------------------------------------------------------------------------- #
# Fixture-match: MEGARCH/MLog-GARCH tight; Log-GARCH ridge (reaches-or-beats)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model", ["megarch", "mloggarch"])
@pytest.mark.parametrize("dist", _TYPE1_DISTS)
def test_type1_fixture_match(model: str, dist: str) -> None:
    """MEGARCH/MLog-GARCH reproduce each fixture's loglik, sigma, gamma and (ald/sald) P.

    Both are well identified, so the match is tight (unlike Log-GARCH's ridge). Shape/skew are near
    their normal limits on the Gaussian data (by regime, not param-exact).
    """
    fit_fn = fit_megarch if model == "megarch" else fit_mloggarch
    fx_meta, sigma_fix = _load(model, dist)
    fx = fx_meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_fn(_returns(), cond_dist=dist)
    assert abs(fit.loglikelihood - fx_meta["loglikelihood"]) < 1e-4
    assert abs(fit.aic - fx_meta["aic"]) < 1e-6
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 1e-5
    assert abs(fit.params["gamma"] - fx["gamma"]) < 5e-3
    if "P" in fx:
        assert fit.params["P"] == fx["P"] == 5


@pytest.mark.parametrize("dist", _LOGGARCH_DISTS)
def test_loggarch_fixture_reaches_or_beats(dist: str) -> None:
    """Log-GARCH sits on the near-common-root ridge, so the fit reaches-or-beats fEGarch's fixture.

    The seam is machine-exact (correctness proven above); the fit is weakly identified along the
    phi1 ~ -psi1 ridge, so our optimizer finds a comparable-or-higher-likelihood point than fEGarch
    (often strictly dominating: ald +1.3, sstd +4.9). Validated by domination, not a param match.
    """
    fx_meta, _sigma = _load("loggarch", dist)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_loggarch(_returns(), cond_dist=dist)
    assert fit.converged
    assert (
        fit.loglikelihood >= fx_meta["loglikelihood"] - 1e-3
    )  # reaches or dominates the ridge point
    if "P" in fx_meta["params"]:
        assert fit.params["P"] == 5


def test_gamma_tell_mloggarch_is_1p8x_megarch() -> None:
    """The log-modulus magnitude tell holds across laws: MLog-GARCH gamma ~ 1.8x MEGARCH's.

    MEGARCH's |eta| magnitude gives gamma ~ 0.16; MLog-GARCH's compressed ln(|eta|+1) gives ~0.28, a
    ~1.8x ratio held across every converging law (read from the committed fixtures).
    """
    for dist in ["norm", *_TYPE1_DISTS]:
        g_me = _load("megarch", dist)[0]["params"]["gamma"]
        g_ml = _load("mloggarch", dist)[0]["params"]["gamma"]
        assert 1.75 < g_ml / g_me < 1.85


def test_norm_paths_stay_bit_identical() -> None:
    """The asymmetry-centering + per-iteration refactor leaves the three norm fits intact.

    norm has no shape parameter and (being symmetric) a zero asymmetry centering, so each keeps its
    precomputed-constant path and reproduces its norm fixture log-likelihood unchanged.
    """
    returns = _returns()
    for fit_fn, model in (
        (fit_megarch, "megarch"),
        (fit_mloggarch, "mloggarch"),
        (fit_loggarch, "loggarch"),
    ):
        meta, _sigma = _load(model, "norm")
        fit = fit_fn(returns, cond_dist="norm")
        assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-5


# --------------------------------------------------------------------------- #
# Known-truth for the 5 fEGarch-optimizer-failure cases (continuous df)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("model", "sim", "fit_fn", "kw"),
    [
        ("megarch", megarch_sim, fit_megarch, {"kappa": -0.05, "gamma": 0.15}),
        ("mloggarch", mloggarch_sim, fit_mloggarch, {"kappa": -0.05, "gamma": 0.25}),
    ],
)
@pytest.mark.parametrize("dist", ["std", "sstd"])
def test_type1_known_truth_where_fegarch_failed(model, sim, fit_fn, kw, dist) -> None:  # type: ignore[no-untyped-def]
    """megarch/mloggarch x std|sstd: recover df (+skew) where fEGarch's optimizer failed.

    The sstd case exercises the per-iteration asymmetry centering (nonzero under skew) in both the
    sim and the fit, so omega_sig and skew recover only if that 4th moment is wired consistently.
    """
    dist_params = (6.0,) if dist == "std" else (6.0, 0.85)
    true = {"mu": 0.0003, "omega_sig": -9.0, "phi1": 0.95, **kw}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = sim(
            9000, **true, cond_dist=dist, dist_params=dist_params, rng=np.random.default_rng(0)
        )
        fit = fit_fn(returns, cond_dist=dist)
    assert fit.converged
    assert abs(fit.params["nu"] - 6.0) < 3.0 * fit.std_errors["nu"]
    if dist == "sstd":
        assert abs(fit.params["skew"] - 0.85) < 3.0 * fit.std_errors["skew"]


def test_loggarch_std_known_truth_where_fegarch_failed() -> None:
    """Log-GARCH x std: recover df where fEGarch failed (its std failed, sstd converged --

    evidence these are flat-ridge starting-point solver failures, not model failures, which our
    Nelder-Mead + restart handles).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        returns, _sigma = loggarch_sim(
            9000,
            mu=0.0003,
            omega_sig=-9.0,
            phi1=0.4,
            psi1=-0.5,
            cond_dist="std",
            dist_params=(6.0,),
            rng=np.random.default_rng(0),
        )
        fit = fit_loggarch(returns, cond_dist="std")
    assert fit.converged
    assert abs(fit.params["nu"] - 6.0) < 3.0 * fit.std_errors["nu"]
