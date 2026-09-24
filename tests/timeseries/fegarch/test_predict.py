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
from quantica.timeseries.fegarch.egarch import egarch_recursion
from quantica.timeseries.fegarch.garch import garch_recursion

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

_SIGMA_TOL = 1e-14  # residual state is machine-exact -> the iterate is too


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
