"""Phase 6 multi-step predict -- the n_ahead point forecast (distinct from predict_roll one-step).

Validated against the committed fixtures (``predict_garch11_norm_h{10,50}`` /
``predict_egarch11_norm_h{10,50}``). fEGarch's multi-step ``predict`` is the NAIVE ITERATED forecast
(future news = its expectation), reproduced clean-room (CLAUDE.md 12) from the recursion:

* GARCH is UNBIASED -- sigma^2_{n+h} = omega + (alpha+beta) sigma^2_{n+h-1} (h>=2; h=1 uses the
  realized eps_n), converging to the unconditional variance omega/(1-alpha-beta). Machine-exact
  (~1.7e-18) since the residual state is machine-exact.
* EGARCH is BIASED-BY-CONSTRUCTION -- iterate ln sigma^2 with future g(eta)=0 and report
  exp(ln sigma^2/2), converging to exp(omega_sig/2) = exp(E[ln sigma^2]/2) != E[sigma] (the Jensen
  gap). We reproduce fEGarch's biased output, NOT a bias-corrected one. Machine-exact (~1.7e-17).

Seam: predict h=1 sigma == predict_roll first sigma (the same one-step-ahead value).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import GarchFit, predict, predict_roll
from quantica.timeseries.fegarch.asymmetric import (
    aparch_recursion,
    gjr_recursion,
    tgarch_recursion,
)
from quantica.timeseries.fegarch.egarch import egarch_recursion
from quantica.timeseries.fegarch.garch import garch_recursion

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

_SIGMA_TOL = 1e-14  # residual state is machine-exact -> the iterate is too
# GARCH multi-step across laws: the variance iterate is distribution-independent, so sigma matches
# the fixture to machine precision -- symmetric laws ~1e-17, the skew laws ~7e-14 (the sigma_n
# training-end seed's reconstruction gap compounding over the 10-step iterate). Machine-precision.
_GARCH_PREDICT_TOL = 1e-12
# EGARCH non-norm: the h=1 news uses the law's E|eta| centering, so the band also carries the
# abs_moment-vs-fEGarch gap -- symmetric laws ~1e-15, the Fernandez-Steel-skew laws ~8e-13 (the
# skew-wrapper E|eta| is a touch looser than the symmetric closed forms). Calibrated against the
# fixtures (see test_egarch_predict_matches_fixture_all_dists).
_EGARCH_SIGMA_TOL = 5e-12


def _returns() -> np.ndarray:  # type: ignore[type-arg]
    """The committed synthetic return series (2500 obs)."""
    return np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)


def _garch_fit() -> GarchFit:
    """GARCH(1,1)/norm TRAINING fit (n_test=250) with machine-exact conditional volatility."""
    meta = json.loads((_FIXTURE_DIR / "predict_garch11_norm_meta.json").read_text(encoding="utf-8"))
    p = meta["fit_params"]
    params = {"mu": p["mu"], "omega": p["omega"], "alpha": p["phi1"], "beta": p["beta1"]}
    n_obs = int(meta["n_obs_train"])
    vec = np.array([params["mu"], params["omega"], params["alpha"], params["beta"]])
    cond_vol = np.sqrt(garch_recursion(vec, _returns()[:n_obs]))
    return GarchFit(
        cond_dist="norm",
        params=params,
        std_errors=dict.fromkeys(params, 0.0),
        loglikelihood=0.0,
        aic=0.0,
        bic=0.0,
        conditional_volatility=cond_vol,
        n_obs=n_obs,
        converged=True,
    )


def _egarch_fit() -> GarchFit:
    """EGARCH(1,1)/norm FULL-sample fit with machine-exact conditional volatility."""
    meta = json.loads(
        (_FIXTURE_DIR / "predict_egarch11_norm_meta.json").read_text(encoding="utf-8")
    )
    params = dict(meta["fit_params"])  # mu, omega_sig, phi1, kappa, gamma
    returns = _returns()
    vec = np.array([params[k] for k in ("mu", "omega_sig", "phi1", "kappa", "gamma")])
    e_abs = float(np.sqrt(2.0 / np.pi))  # E|z| for the standard normal
    cond_vol = np.sqrt(egarch_recursion(vec, returns, abs_moment=e_abs))
    return GarchFit(
        cond_dist="norm",
        params=params,
        std_errors=dict.fromkeys(params, 0.0),
        loglikelihood=0.0,
        aic=0.0,
        bic=0.0,
        conditional_volatility=cond_vol,
        n_obs=returns.size,
        converged=True,
    )


def _fix(name: str, h: int) -> np.ndarray:  # type: ignore[type-arg]
    """Load a predict fixture's sigma column for horizon ``h``."""
    return np.loadtxt(_FIXTURE_DIR / f"predict_{name}_h{h}.csv", delimiter=",", skiprows=1)[:, 1]


# --------------------------------------------------------------------------- #
# GARCH multi-step -- unbiased iterated forecast
# --------------------------------------------------------------------------- #


def test_garch_predict_matches_fixture() -> None:
    """GARCH sigma over h=10 and h=50 reproduces the fixture to machine precision."""
    fit, returns = _garch_fit(), _returns()
    for h in (10, 50):
        fc = predict(fit, returns, n_ahead=h)
        assert fc.n_ahead == h
        assert np.max(np.abs(fc.sigma - _fix("garch11_norm", h))) < _SIGMA_TOL


def test_garch_predict_converges_to_unconditional_variance() -> None:
    """GARCH multi-step sigma^2 climbs monotonically toward omega/(1-alpha-beta)."""
    meta = json.loads((_FIXTURE_DIR / "predict_garch11_norm_meta.json").read_text(encoding="utf-8"))
    uncond = meta["unconditional_variance"]
    fc = predict(_garch_fit(), _returns(), n_ahead=2000)
    var = fc.sigma**2
    assert np.all(
        np.diff(var) >= 0.0
    )  # monotone non-decreasing (plateaus once it reaches the limit)
    assert var[-1] > var[0]  # and it genuinely climbs from the 1-step forecast
    assert var[-1] == pytest.approx(uncond, rel=1e-6)  # far-horizon -> unconditional variance


def test_garch_predict_mean_constant() -> None:
    """The GARCH mean forecast is the constant fitted mu."""
    fit = _garch_fit()
    fc = predict(fit, _returns(), n_ahead=10)
    assert np.all(fc.cmeans == fit.params["mu"])


# --------------------------------------------------------------------------- #
# EGARCH multi-step -- biased-by-construction iterated forecast
# --------------------------------------------------------------------------- #


def test_egarch_predict_matches_fixture() -> None:
    """EGARCH sigma over h=10 and h=50 reproduces the fixture (the naive/biased iterate)."""
    fit, returns = _egarch_fit(), _returns()
    for h in (10, 50):
        fc = predict(fit, returns, n_ahead=h)
        assert np.max(np.abs(fc.sigma - _fix("egarch11_norm", h))) < 1e-13


def test_egarch_predict_converges_to_biased_limit() -> None:
    """EGARCH sigma converges to exp(omega_sig/2) = exp(E[ln s^2]/2), NOT E[sigma] (Jensen gap)."""
    meta = json.loads(
        (_FIXTURE_DIR / "predict_egarch11_norm_meta.json").read_text(encoding="utf-8")
    )
    omega_sig = meta["fit_params"]["omega_sig"]
    biased_limit = float(np.exp(omega_sig / 2.0))
    fc = predict(_egarch_fit(), _returns(), n_ahead=4000)
    assert fc.sigma[-1] == pytest.approx(biased_limit, rel=1e-6)
    # The biased limit is the exp of the mean log-variance -- documented as the left-in Jensen gap.
    assert biased_limit == pytest.approx(0.011349, abs=1e-5)


# --------------------------------------------------------------------------- #
# The predict / predict_roll h=1 seam
# --------------------------------------------------------------------------- #


def test_predict_h1_equals_predict_roll_first_sigma() -> None:
    """predict h=1 sigma == predict_roll's first one-step forecast (the same value)."""
    fit, returns = _garch_fit(), _returns()
    ph1 = predict(fit, returns, n_ahead=1).sigma[0]
    roll_first = predict_roll(fit, returns).sigma[0]
    assert (
        ph1 == roll_first
    )  # bit-for-bit: both are the one-step-ahead forecast from the same state
    # and both equal the committed rolling fixture's first value.
    roll_fix = np.loadtxt(_FIXTURE_DIR / "forecast_garch11_norm_roll_sigma.csv", skiprows=1)[0]
    assert abs(ph1 - roll_fix) < _SIGMA_TOL


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_predict_rejects_bad_horizon() -> None:
    """n_ahead must be positive."""
    with pytest.raises(ValueError, match="n_ahead"):
        predict(_garch_fit(), _returns(), n_ahead=0)


def test_predict_rejects_unsupported_family() -> None:
    """A fit whose params are neither GARCH nor EGARCH raises NotImplementedError."""
    fit = _garch_fit()
    fi_like = GarchFit(
        cond_dist="norm",
        params={"mu": 0.0, "omega": 1e-6, "phi1": 0.2, "beta1": 0.7, "d": 0.4},
        std_errors={},
        loglikelihood=0.0,
        aic=0.0,
        bic=0.0,
        conditional_volatility=fit.conditional_volatility,
        n_obs=fit.n_obs,
        converged=True,
    )
    with pytest.raises(NotImplementedError, match="multi-step predict"):
        predict(fi_like, _returns(), n_ahead=5)


# --------------------------------------------------------------------------- #
# distribution breadth -- GARCH + EGARCH multi-step under the non-norm laws
# --------------------------------------------------------------------------- #
#
# The engine is UNCHANGED. The multi-step iterate is distribution-independent for GARCH (variance
# recursion) and depends on the law ONLY through the EGARCH h=1 news centering E|eta|; both match
# the fixtures to machine precision under every converging law. The requested horizons h = 1, 5, 10
# are the first, fifth and tenth rows of the committed h10 path. GARCH covers all 7 non-norm laws;
# EGARCH covers the 5 that fEGarch's optimizer fits (std/sstd error out -- the continuous-df Type-I
# EGF robustness gap, docs/fegarch-spec-notes.md §20 -- so there is no fEGarch output to reproduce).

_GARCH_DISTS = ["std", "ged", "ald", "snorm", "sstd", "sged", "sald"]
_EGARCH_DISTS = ["ged", "ald", "snorm", "sged", "sald"]
_REQUESTED_HORIZONS = (1, 5, 10)


def _garch_params_from_json(p: dict, cond_dist: str) -> dict:  # type: ignore[type-arg]
    """R GARCH fit params (``mu, omega, phi1, beta1`` + shape/skew/P) -> quantica GarchFit names."""
    params = {"mu": p["mu"], "omega": p["omega"], "alpha": p["phi1"], "beta": p["beta1"]}
    if cond_dist in ("std", "sstd"):
        params["nu"] = p["df"]
    elif cond_dist in ("ged", "sged"):
        params["nu"] = p["shape"]
    if cond_dist in ("ald", "sald"):
        params["P"] = p["P"]
    if cond_dist in ("snorm", "sstd", "sged", "sald"):
        params["skew"] = p["skew"]
    return params


def _egarch_params_from_json(p: dict, cond_dist: str) -> dict:  # type: ignore[type-arg]
    """R EGARCH fit params (``mu, omega_sig, phi1, kappa, gamma`` + shape/skew/P) -> quantica."""
    params = {k: p[k] for k in ("mu", "omega_sig", "phi1", "kappa", "gamma")}
    if cond_dist in ("std", "sstd"):
        params["nu"] = p["df"]
    elif cond_dist in ("ged", "sged"):
        params["nu"] = p["shape"]
    if cond_dist in ("ald", "sald"):
        params["P"] = p["P"]
    if cond_dist in ("snorm", "sstd", "sged", "sald"):
        params["skew"] = p["skew"]
    return params


def _garch_fit_dist(dist: str) -> GarchFit:
    """GARCH(1,1) TRAINING fit (n_test=250) under ``dist``, with reconstructed conditional vol."""
    meta = json.loads(
        (_FIXTURE_DIR / f"predict_garch11_{dist}_meta.json").read_text(encoding="utf-8")
    )
    params = _garch_params_from_json(meta["fit_params"], dist)
    n_obs = int(meta["n_obs_train"])
    vec = np.array([params["mu"], params["omega"], params["alpha"], params["beta"]])
    cond_vol = np.sqrt(garch_recursion(vec, _returns()[:n_obs]))
    return GarchFit(
        cond_dist=dist,
        params=params,
        std_errors=dict.fromkeys(params, 0.0),
        loglikelihood=0.0,
        aic=0.0,
        bic=0.0,
        conditional_volatility=cond_vol,
        n_obs=n_obs,
        converged=True,
    )


def _egarch_fit_dist(dist: str) -> GarchFit:
    """EGARCH(1,1) FULL-sample fit under ``dist``, with conditional vol rebuilt at its E|eta|."""
    from quantica.timeseries.fegarch.forecast import _distribution_for_fit

    meta = json.loads(
        (_FIXTURE_DIR / f"predict_egarch11_{dist}_meta.json").read_text(encoding="utf-8")
    )
    params = _egarch_params_from_json(meta["fit_params"], dist)
    returns = _returns()
    vec = np.array([params[k] for k in ("mu", "omega_sig", "phi1", "kappa", "gamma")])
    # The log-variance recursion is centred on the law's E|eta| (distribution-dependent) -- rebuild
    # it exactly as predict does, so sigma_n (the h=1 seed) is machine-consistent with the fit.
    partial = GarchFit(
        cond_dist=dist,
        params=params,
        std_errors=dict.fromkeys(params, 0.0),
        loglikelihood=0.0,
        aic=0.0,
        bic=0.0,
        conditional_volatility=np.array([1.0]),
        n_obs=returns.size,
        converged=True,
    )
    dist_obj, shape = _distribution_for_fit(partial)
    e_abs = dist_obj.abs_moment(shape)
    cond_vol = np.sqrt(egarch_recursion(vec, returns, abs_moment=e_abs))
    return GarchFit(
        cond_dist=dist,
        params=params,
        std_errors=dict.fromkeys(params, 0.0),
        loglikelihood=0.0,
        aic=0.0,
        bic=0.0,
        conditional_volatility=cond_vol,
        n_obs=returns.size,
        converged=True,
    )


@pytest.mark.parametrize("dist", _GARCH_DISTS)
def test_garch_predict_matches_fixture_all_dists(dist: str) -> None:
    """GARCH multi-step sigma reproduces the h10 fixture (incl. h=1,5,10) under every law."""
    fit = _garch_fit_dist(dist)
    fc = predict(fit, _returns(), n_ahead=10)
    fix = _fix(f"garch11_{dist}", 10)
    assert fc.n_ahead == 10
    assert np.max(np.abs(fc.sigma - fix)) < _GARCH_PREDICT_TOL
    # The requested horizons h = 1, 5, 10 land on the committed path.
    for h in _REQUESTED_HORIZONS:
        assert abs(fc.sigma[h - 1] - fix[h - 1]) < _GARCH_PREDICT_TOL
    assert np.all(fc.cmeans == fit.params["mu"])


@pytest.mark.parametrize("dist", _GARCH_DISTS)
def test_garch_predict_h1_equals_roll_first_all_dists(dist: str) -> None:
    """The h=1 multi-step sigma equals the rolling forecast's first value (the one-step seam)."""
    fit = _garch_fit_dist(dist)
    ph1 = predict(fit, _returns(), n_ahead=1).sigma[0]
    roll_first = predict_roll(fit, _returns()).sigma[0]
    # The same one-step-ahead forecast from the same state -- equal to within a last-bit rounding
    # (the two paths reach eps_n/sigma_n by a different order of operations; <= 1 ULP apart).
    assert abs(ph1 - roll_first) < 1e-15
    roll_fix = np.loadtxt(_FIXTURE_DIR / f"forecast_garch11_{dist}_roll_sigma.csv", skiprows=1)[0]
    assert abs(ph1 - roll_fix) < _GARCH_PREDICT_TOL


@pytest.mark.parametrize("dist", _EGARCH_DISTS)
def test_egarch_predict_matches_fixture_all_dists(dist: str) -> None:
    """EGARCH multi-step sigma reproduces the naive/biased h10 fixture under each converging law.

    The h=1 news term uses the law's distribution-dependent E|eta| centering; h>=2 depends only on
    omega_sig / phi1. The known reproduced-not-corrected EGF bias is asserted as reproduced (we
    match fEGarch's naive iterate), never "fixed".
    """
    fit = _egarch_fit_dist(dist)
    fc = predict(fit, _returns(), n_ahead=10)
    fix = _fix(f"egarch11_{dist}", 10)
    assert np.max(np.abs(fc.sigma - fix)) < _EGARCH_SIGMA_TOL
    for h in _REQUESTED_HORIZONS:
        assert abs(fc.sigma[h - 1] - fix[h - 1]) < _EGARCH_SIGMA_TOL
    assert np.all(fc.cmeans == fit.params["mu"])


@pytest.mark.parametrize("dist", _EGARCH_DISTS)
def test_egarch_predict_reproduces_biased_limit_all_dists(dist: str) -> None:
    """Far-horizon EGARCH sigma converges to exp(omega_sig/2) -- the left-in Jensen gap, per law."""
    fit = _egarch_fit_dist(dist)
    biased_limit = float(np.exp(fit.params["omega_sig"] / 2.0))
    fc = predict(fit, _returns(), n_ahead=4000)
    assert fc.sigma[-1] == pytest.approx(biased_limit, rel=1e-6)


# --------------------------------------------------------------------------- #
# APARCH-family multi-step (forecast family 1) -- GJR / TGARCH / APARCH, all 8 laws
# --------------------------------------------------------------------------- #
#
# The APARCH-power multi-step forecast (spec-notes §33): m_h = E_t[sigma_{t+h}^delta] obeys the
# linear recursion m_h = omega + (phi1*kappa + beta1) m_{h-1} for h>=2, with the ASYMMETRY-CORRECTED
# news moment kappa = E[(|eta| - gamma1 eta)^delta] (full quadrature -- the gate showed fEGarch
# integrates the whole kernel even at delta=1,2, so the analytic closed forms diverge ~1e-13 while
# full quadrature is machine-exact), and sigma = (m_h)^(1/delta) the Jensen-biased plug-in
# (reproduced-not-corrected). GJR (delta=2) / TGARCH (delta=1) are the fixed-power specializations.

_APARCH_MODELS = ["aparch", "gjrgarch", "tgarch"]
_APARCH_DISTS = ["norm", "std", "ged", "ald", "snorm", "sstd", "sged", "sald"]
# 23/24 reproduce to machine precision (~1e-16); the one R<->SciPy kappa-quadrature-floor case is
# tgarch x sged at ~8.7e-12 (the skew-GED shape where R's and SciPy's adaptive integrals differ).
_APARCH_PREDICT_TOL = 1e-10


def _aparch_params_from_json(p: dict, model: str, dist: str) -> dict:  # type: ignore[type-arg]
    """R APARCH-family fit params -> quantica GarchFit names (incl. the power ``delta``)."""
    q = {
        "mu": p["mu"],
        "omega": p["omega"],
        "phi1": p["phi1"],
        "beta1": p["beta1"],
        "gamma1": p["gamma1"],
    }
    # delta is in the fitted vector for APARCH; fixed 2.0 / 1.0 for GJR / TGARCH.
    q["delta"] = p.get("delta", 2.0 if model == "gjrgarch" else 1.0)
    if dist in ("std", "sstd"):
        q["nu"] = p["df"]
    elif dist in ("ged", "sged"):
        q["nu"] = p["shape"]
    if dist in ("ald", "sald"):
        q["P"] = p["P"]
    if dist in ("snorm", "sstd", "sged", "sald"):
        q["skew"] = p["skew"]
    return q


def _aparch_family_cond_vol(q: dict, n: int) -> np.ndarray:  # type: ignore[type-arg]
    """Reconstruct conditional-SD over ``returns[:n]`` via the delta-matched recursion."""
    v = np.array([q["mu"], q["omega"], q["phi1"], q["beta1"], q["gamma1"]])
    y = _returns()[:n]
    if q["delta"] == 2.0:
        return np.sqrt(gjr_recursion(v, y))
    if q["delta"] == 1.0:
        return np.sqrt(tgarch_recursion(v, y))
    return np.sqrt(aparch_recursion(np.append(v, q["delta"]), y))


def _aparch_predict_fit(model: str, dist: str) -> GarchFit:
    """APARCH-family TRAINING fit (n_test=250) for ``(model, dist)`` with reconstructed cond vol."""
    meta = json.loads(
        (_FIXTURE_DIR / f"predict_{model}11_{dist}_meta.json").read_text(encoding="utf-8")
    )
    q = _aparch_params_from_json(meta["fit_params"], model, dist)
    n = int(meta["n_obs_train"])
    return GarchFit(
        cond_dist=dist,
        params=q,
        std_errors=dict.fromkeys(q, 0.0),
        loglikelihood=0.0,
        aic=0.0,
        bic=0.0,
        conditional_volatility=_aparch_family_cond_vol(q, n),
        n_obs=n,
        converged=True,
    )


@pytest.mark.parametrize("model", _APARCH_MODELS)
@pytest.mark.parametrize("dist", _APARCH_DISTS)
def test_aparch_family_predict_matches_fixture(model: str, dist: str) -> None:
    """APARCH/GJR/TGARCH multi-step sigma reproduces the h10 fixture (incl. h=1,5,10) per law.

    GJR/TGARCH flow through the same delta=2/delta=1 specialization of the APARCH path -- a clean
    inheritance (all land at machine precision bar the one documented quadrature-floor case).
    """
    fit = _aparch_predict_fit(model, dist)
    fc = predict(fit, _returns(), n_ahead=10)
    fix = _fix(f"{model}11_{dist}", 10)
    assert fc.n_ahead == 10
    assert np.max(np.abs(fc.sigma - fix)) < _APARCH_PREDICT_TOL
    for h in _REQUESTED_HORIZONS:
        assert abs(fc.sigma[h - 1] - fix[h - 1]) < _APARCH_PREDICT_TOL
    assert np.all(fc.cmeans == fit.params["mu"])


@pytest.mark.parametrize("model", _APARCH_MODELS)
@pytest.mark.parametrize("dist", _APARCH_DISTS)
def test_aparch_family_predict_h1_equals_roll_first(model: str, dist: str) -> None:
    """The h=1 multi-step sigma equals predict_roll's first value (one-step seam), per model/law."""
    fit = _aparch_predict_fit(model, dist)
    ph1 = predict(fit, _returns(), n_ahead=1).sigma[0]
    roll_first = predict_roll(fit, _returns()).sigma[0]
    assert abs(ph1 - roll_first) < 1e-15  # same one-step forecast from the same state (<=1 ULP)


def test_aparch_predict_converges_to_reduced_persistence_limit() -> None:
    """APARCH sigma^delta -> omega/(1 - (phi1*kappa + beta1)), the reduced-persistence limit."""
    from quantica.timeseries.fegarch.distributions import get_distribution

    fit = _aparch_predict_fit("aparch", "norm")
    delta, gamma1 = fit.params["delta"], fit.params["gamma1"]
    kappa = get_distribution("norm").news_moment(delta, gamma1)
    rho = fit.params["phi1"] * kappa + fit.params["beta1"]
    m_inf = fit.params["omega"] / (1.0 - rho)
    fc = predict(fit, _returns(), n_ahead=4000)
    assert (fc.sigma[-1] ** delta) == pytest.approx(m_inf, rel=1e-6)
    assert np.all(np.diff(fc.sigma**delta) >= -1e-18)  # monotone in sigma^delta
