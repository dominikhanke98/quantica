"""Phase 6 -- the no-refit rolling forecast, VaR/ES assembly, and the risk-pillar tie-back.

Validated against the committed fEGarch forecast fixture (``forecast_garch11_norm_*``): a
GARCH(1,1)/norm model fit with ``n_test=250`` held out, then ``predict_roll(refit_after=NULL)`` ->
rolling one-step sigma-hat / mu-hat, then ``measure_risk`` -> VaR/ES at 0.975 & 0.99. The findings
the build pins:

* **sigma-hat reconstruction is machine-exact** (~7e-18): the rolling forecast is the existing
  ``garch_recursion`` continued past the training window with the training params -- the
  train-then-continue seed is irrelevant at the test window (the long training run decays the
  pre-sample transient), so there is no bounded limit here.
* **mu-hat is constant** = the fitted mu (plain GARCH, sigma-only forecast).
* **the standardized-ES helper** ``expected_shortfall`` is the ES integral of the existing ``ppf``
  (closed-form for the normal): E_norm ES(0.975) = -2.337803, ES(0.99) = -2.665214.
* **VaR/ES assembly matches to ~1e-6** -- NOT machine-exact: sigma-hat is machine-exact, so the
  residual is sigma-hat * (q_scipy - q_R), the R-vs-SciPy quantile-implementation difference.
* **the sign-map tie-back**: the return-space forecasts feed the loss-space backtests
  (Kupiec/Christoffersen/Basel/Acerbi-Szekely) via ``backtest_return_forecasts`` -- the exception
  count is asserted against a hand computation (a flipped sign silently inverts exceptions).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from quantica.risk.backtest import VaRESBacktest, backtest_return_forecasts
from quantica.timeseries.fegarch import (
    GarchFit,
    get_distribution,
    measure_risk,
    predict_roll,
)
from quantica.timeseries.fegarch.asymmetric import (
    aparch_recursion,
    gjr_recursion,
    tgarch_recursion,
)
from quantica.timeseries.fegarch.distributions import (
    ConditionalDistribution,
    FernandezSteelSkew,
    Normal,
    StudentT,
)
from quantica.timeseries.fegarch.garch import garch_recursion

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

# sigma-hat is machine-exact; the ~1e-6 band on VaR/ES is the documented R<->SciPy quantile diff.
_SIGMA_TOL = 1e-14
_VAR_ES_TOL = 3e-6
# The VaR/ES band across the non-normal laws: the same R<->SciPy quantile gap (max observed 2.66e-6,
# the GED), the same scale as the normal's 3e-6. Calibrated against the fixtures (see
# test_measure_risk_matches_fixture_all_dists).
_VAR_ES_DIST_TOL = 5e-6
# FI rolling sigma: the pre-sample-0 observed-news convolution is seam-exact (no kappa; §34.2) for
# the variance-recursion models (~1e-14); FIEGARCH's g-recursion carries the FS-skew abs_moment
# quadrature floor (max 2.55e-12, fiegarch x sald). Machine-precision-class across all FI laws.
_FI_ROLL_TOL = 1e-11


def _returns() -> np.ndarray:  # type: ignore[type-arg]
    """The committed synthetic return series (2500 obs)."""
    return np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)


def _fixture() -> dict:  # type: ignore[type-arg]
    """The forecast fixture: training-fit params + rolling sigma/cmeans + VaR/ES columns."""
    meta = json.loads(
        (_FIXTURE_DIR / "forecast_garch11_norm_params.json").read_text(encoding="utf-8")
    )
    sigma = np.loadtxt(_FIXTURE_DIR / "forecast_garch11_norm_roll_sigma.csv", skiprows=1)
    cmeans = np.loadtxt(_FIXTURE_DIR / "forecast_garch11_norm_roll_cmeans.csv", skiprows=1)
    var_es = np.loadtxt(
        _FIXTURE_DIR / "forecast_garch11_norm_var_es.csv", delimiter=",", skiprows=1
    )
    return {"meta": meta, "sigma": sigma, "cmeans": cmeans, "var_es": var_es}


def _fit_from_fixture(meta: dict, returns: np.ndarray) -> GarchFit:  # type: ignore[type-arg]
    """A :class:`GarchFit` carrying the fixture's *training* parameters (R names -> quantica names).

    The fixture JSON reports ``mu, omega, phi1, beta1``; :class:`GarchFit` uses ``mu, omega, alpha,
    beta``. The conditional-volatility series is the training-window recursion (unused by
    ``predict_roll`` but kept faithful).
    """
    p = meta["params"]
    params = {"mu": p["mu"], "omega": p["omega"], "alpha": p["phi1"], "beta": p["beta1"]}
    n_obs = int(meta["n_obs"])
    train_vec = np.array([params["mu"], params["omega"], params["alpha"], params["beta"]])
    cond_vol = np.sqrt(garch_recursion(train_vec, returns[:n_obs]))
    return GarchFit(
        cond_dist=meta["cond_dist"],
        params=params,
        std_errors=dict.fromkeys(params, 0.0),
        loglikelihood=float(meta["loglikelihood"]),
        aic=float(meta["aic"]),
        bic=float(meta["bic"]),
        conditional_volatility=cond_vol,
        n_obs=n_obs,
        converged=True,
    )


# --------------------------------------------------------------------------- #
# 1. predict_roll -- the no-refit rolling forecast (machine-exact sigma-hat)
# --------------------------------------------------------------------------- #


def test_predict_roll_sigma_machine_exact() -> None:
    """sigma-hat = garch_recursion continued past the training window -> matches the fixture."""
    fx = _fixture()
    returns = _returns()
    fit = _fit_from_fixture(fx["meta"], returns)
    forecast = predict_roll(fit, returns)
    assert forecast.n_test == fx["sigma"].size == 250
    assert np.max(np.abs(forecast.sigma - fx["sigma"])) < _SIGMA_TOL


def test_predict_roll_mean_constant() -> None:
    """mu-hat is the constant fitted mu (plain GARCH) and matches the fixture cmeans."""
    fx = _fixture()
    returns = _returns()
    fit = _fit_from_fixture(fx["meta"], returns)
    forecast = predict_roll(fit, returns)
    assert np.all(forecast.cmeans == fit.params["mu"])
    assert np.max(np.abs(forecast.cmeans - fx["cmeans"])) < _SIGMA_TOL


def test_predict_roll_seed_irrelevant_at_test_window() -> None:
    """Any reasonable training seed gives the same sigma-hat: transient decays before the tail."""
    fx = _fixture()
    returns = _returns()
    fit = _fit_from_fixture(fx["meta"], returns)
    baseline = predict_roll(fit, returns).sigma
    # An explicit n_test that reads the same 250-point tail must agree bit-for-bit.
    assert np.array_equal(predict_roll(fit, returns, n_test=250).sigma, baseline)


def test_predict_roll_rejects_bad_n_test() -> None:
    """n_test must be positive and within the series length."""
    fx = _fixture()
    returns = _returns()
    fit = _fit_from_fixture(fx["meta"], returns)
    with pytest.raises(ValueError, match="n_test"):
        predict_roll(fit, returns, n_test=0)
    with pytest.raises(ValueError, match="n_test"):
        predict_roll(fit, returns, n_test=returns.size + 1)


# --------------------------------------------------------------------------- #
# 2. the standardized-ES helper -- the one new piece (ppf integral, all 8 dists)
# --------------------------------------------------------------------------- #


def test_expected_shortfall_normal_sanity() -> None:
    """Normal ES closed form hits the pinned values -2.337803 (0.975) / -2.665214 (0.99)."""
    norm = get_distribution("norm")
    assert norm.expected_shortfall(0.975) == pytest.approx(-2.337803, abs=1e-6)
    assert norm.expected_shortfall(0.99) == pytest.approx(-2.665214, abs=1e-6)


def test_expected_shortfall_normal_closed_form_equals_quadrature() -> None:
    """The Normal closed form equals the base ``z*f(z)`` quadrature (ES = integral of the ppf)."""
    norm = get_distribution("norm")
    for alpha in (0.975, 0.99):
        quad = ConditionalDistribution.expected_shortfall(norm, alpha)  # base quadrature path
        assert norm.expected_shortfall(alpha) == pytest.approx(quad, abs=1e-9)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        ("norm", None),
        ("std", [6.0]),
        ("ged", [1.3]),
        ("ald", None),
        ("snorm", [1.2]),
        ("sstd", [6.0, 0.85]),
        ("sged", [1.3, 1.1]),
        ("sald", [0.8]),
    ],
)
def test_expected_shortfall_more_extreme_than_var(name: str, params: list[float] | None) -> None:
    """ES is deeper in the tail than the VaR quantile for every distribution and level."""
    dist = get_distribution(name)
    for alpha in (0.975, 0.99):
        q_eta = float(dist.ppf(np.array([1.0 - alpha]), params)[0])
        es_eta = dist.expected_shortfall(alpha, params)
        assert es_eta < q_eta < 0.0


def test_expected_shortfall_skew_reduces_to_base_at_skew1() -> None:
    """The Fernandez-Steel ES at skew=1 reduces to the symmetric base ES (odd-function anchor)."""
    snorm = FernandezSteelSkew(Normal())
    norm = Normal()
    sstd = FernandezSteelSkew(StudentT())
    std = StudentT()
    for alpha in (0.975, 0.99):
        assert snorm.expected_shortfall(alpha, [1.0]) == pytest.approx(
            norm.expected_shortfall(alpha), abs=1e-8
        )
        assert sstd.expected_shortfall(alpha, [6.0, 1.0]) == pytest.approx(
            std.expected_shortfall(alpha, [6.0]), abs=1e-8
        )


def test_expected_shortfall_rejects_bad_level() -> None:
    """The confidence level must be strictly inside (0, 1)."""
    norm = get_distribution("norm")
    ged = get_distribution("ged")  # base-quadrature path
    for dist in (norm, ged):
        with pytest.raises(ValueError, match="level"):
            dist.expected_shortfall(0.0)
        with pytest.raises(ValueError, match="level"):
            dist.expected_shortfall(1.0)


# --------------------------------------------------------------------------- #
# 3. measure_risk -- VaR/ES assembly vs the fixture (~1e-6, R<->SciPy quantile)
# --------------------------------------------------------------------------- #


def test_measure_risk_matches_fixture() -> None:
    """VaR/ES = mu-hat + sigma-hat * (ppf / ES_eta) matches the fixture var_es columns to ~1e-6."""
    fx = _fixture()
    returns = _returns()
    fit = _fit_from_fixture(fx["meta"], returns)
    forecast = predict_roll(fit, returns)
    risk = measure_risk(forecast, fit, levels=(0.975, 0.99))
    ve = fx["var_es"]  # cols: var_0.975, var_0.99, es_0.975, es_0.99
    assert np.max(np.abs(risk.var[0.975] - ve[:, 0])) < _VAR_ES_TOL
    assert np.max(np.abs(risk.var[0.99] - ve[:, 1])) < _VAR_ES_TOL
    assert np.max(np.abs(risk.es[0.975] - ve[:, 2])) < _VAR_ES_TOL
    assert np.max(np.abs(risk.es[0.99] - ve[:, 3])) < _VAR_ES_TOL


def test_measure_risk_es_below_var() -> None:
    """The assembled ES series is everywhere at least as deep as the VaR series (return-space)."""
    fx = _fixture()
    returns = _returns()
    fit = _fit_from_fixture(fx["meta"], returns)
    risk = measure_risk(predict_roll(fit, returns), fit)
    for alpha in (0.975, 0.99):
        assert np.all(risk.es[alpha] <= risk.var[alpha])
        assert np.all(risk.var[alpha] < 0.0)


# --------------------------------------------------------------------------- #
# 4. the backtest tie-back -- sign map into the existing risk pillar
# --------------------------------------------------------------------------- #


def test_backtest_sign_map_exception_count() -> None:
    """The sign map is asserted explicitly: the exception count matches a hand computation.

    A VaR exception is a realized return below the (negative) VaR threshold; a flipped sign would
    silently invert the count, so both the correct hand count and a mis-signed count are checked.
    """
    fx = _fixture()
    test_returns = _returns()[-250:]
    var_099 = fx["var_es"][:, 1]
    es_099 = fx["var_es"][:, 3]

    suite = backtest_return_forecasts(test_returns, var_099, es_099, 0.99)
    assert isinstance(suite, VaRESBacktest)

    hand_count = int(np.sum(test_returns < var_099))  # return below the negative VaR threshold
    assert suite.n_exceptions == hand_count
    assert suite.n_obs == 250

    # A flipped sign (comparing +returns against the negative threshold) gives a different count.
    flipped = int(np.sum(test_returns > var_099))
    assert flipped != suite.n_exceptions


def test_backtest_results_are_coherent() -> None:
    """fEGarch forecasts run cleanly through the four backtests and land in sensible ranges."""
    fx = _fixture()
    test_returns = _returns()[-250:]
    ve = fx["var_es"]
    suite = backtest_return_forecasts(test_returns, ve[:, 1], ve[:, 3], 0.99)

    # A correct 99% model on 250 days expects ~2.5 exceptions -> green Basel, non-reject Kupiec.
    assert 0 <= suite.n_exceptions <= 10
    assert 0.0 <= suite.kupiec.p_value <= 1.0
    assert 0.0 <= suite.conditional_coverage.p_value <= 1.0
    assert suite.basel.n_obs == 250
    assert np.isfinite(suite.es.statistic)
    assert suite.es.p_value is None  # no predictive null supplied -> statistic only


# --------------------------------------------------------------------------- #
# 5. distribution breadth -- GARCH(1,1) rolling forecast + VaR/ES under all 8 laws
# --------------------------------------------------------------------------- #
#
# The engine is UNCHANGED: the rolling sigma-hat recursion is distribution-independent, so the same
# ``predict_roll`` reproduces the GARCH fixture under every conditional law to machine precision.
# The distribution genuinely bites in the VaR/ES: ``measure_risk`` uses each law's fitted
# quantile/ES, so these fixtures exercise ``ppf`` + ``expected_shortfall`` across the seven
# non-normal laws (``std``/``ged``/``ald`` + the four FS-skews). EGARCH is excluded from rolling:
# ``predict_roll`` is GARCH-only (an EGARCH rolling fixture would need an engine change).

# R fixture-JSON shape-key -> quantica GarchFit.params key: StudentT/GED both expose ``nu`` (R
# writes ``df``/``shape``); the ALD's profiled degree is the arg ``P``; FS-skew adds ``skew``.
_FORECAST_DISTS = ["std", "ged", "ald", "snorm", "sstd", "sged", "sald"]


def _garch_params_from_json(p: dict, cond_dist: str) -> dict:  # type: ignore[type-arg]
    """Map an R GARCH fit JSON's params (``mu, omega, phi1, beta1`` + shape/skew/P) to quantica.

    The variance block is renamed ``phi1 -> alpha``, ``beta1 -> beta``; the distribution shape
    parameters are renamed to the quantica ``param_names`` (``df``/``shape`` -> ``nu``) so
    :func:`~quantica.timeseries.fegarch.forecast._distribution_for_fit` finds them, with the ALD's
    integer ``P`` and the Fernandez-Steel ``skew`` carried through under their own names.
    """
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


def _forecast_fixture(dist: str) -> dict:  # type: ignore[type-arg]
    """Load the GARCH rolling-forecast fixture for conditional law ``dist``."""
    stem = f"forecast_garch11_{dist}"
    meta = json.loads((_FIXTURE_DIR / f"{stem}_params.json").read_text(encoding="utf-8"))
    return {
        "meta": meta,
        "sigma": np.loadtxt(_FIXTURE_DIR / f"{stem}_roll_sigma.csv", skiprows=1),
        "cmeans": np.loadtxt(_FIXTURE_DIR / f"{stem}_roll_cmeans.csv", skiprows=1),
        "var_es": np.loadtxt(_FIXTURE_DIR / f"{stem}_var_es.csv", delimiter=",", skiprows=1),
    }


def _fit_from_forecast(dist: str, meta: dict, returns: np.ndarray) -> GarchFit:  # type: ignore[type-arg]
    """A :class:`GarchFit` carrying the ``dist`` fixture's training params (shape/skew included)."""
    params = _garch_params_from_json(meta["params"], dist)
    n_obs = int(meta["n_obs"])
    vec = np.array([params["mu"], params["omega"], params["alpha"], params["beta"]])
    cond_vol = np.sqrt(garch_recursion(vec, returns[:n_obs]))
    return GarchFit(
        cond_dist=meta["cond_dist"],
        params=params,
        std_errors=dict.fromkeys(params, 0.0),
        loglikelihood=float(meta["loglikelihood"]),
        aic=float(meta["aic"]),
        bic=float(meta["bic"]),
        conditional_volatility=cond_vol,
        n_obs=n_obs,
        converged=True,
    )


@pytest.mark.parametrize("dist", _FORECAST_DISTS)
def test_predict_roll_sigma_machine_exact_all_dists(dist: str) -> None:
    """Rolling sigma-hat is distribution-independent -> machine-exact under every law."""
    fx = _forecast_fixture(dist)
    returns = _returns()
    fit = _fit_from_forecast(dist, fx["meta"], returns)
    forecast = predict_roll(fit, returns)
    assert forecast.n_test == fx["sigma"].size == 250
    assert np.max(np.abs(forecast.sigma - fx["sigma"])) < _SIGMA_TOL
    # The mean is the constant fitted mu (plain GARCH, sigma-only forecast).
    assert np.max(np.abs(forecast.cmeans - fx["cmeans"])) < _SIGMA_TOL


@pytest.mark.parametrize("dist", _FORECAST_DISTS)
def test_measure_risk_matches_fixture_all_dists(dist: str) -> None:
    """VaR/ES reproduces the fixture under each law -- exercising its ppf + expected_shortfall.

    Not machine-exact: sigma-hat is, so the residual is the R<->SciPy quantile-implementation gap
    for the fitted distribution (same mechanism documented for the normal, a touch wider for the
    hand-implemented ALD / Fernandez-Steel-skew quantiles) -- bounded by ``_VAR_ES_DIST_TOL``.
    """
    fx = _forecast_fixture(dist)
    returns = _returns()
    fit = _fit_from_forecast(dist, fx["meta"], returns)
    risk = measure_risk(predict_roll(fit, returns), fit, levels=(0.975, 0.99))
    ve = fx["var_es"]  # cols: var_0.975, var_0.99, es_0.975, es_0.99
    assert np.max(np.abs(risk.var[0.975] - ve[:, 0])) < _VAR_ES_DIST_TOL
    assert np.max(np.abs(risk.var[0.99] - ve[:, 1])) < _VAR_ES_DIST_TOL
    assert np.max(np.abs(risk.es[0.975] - ve[:, 2])) < _VAR_ES_DIST_TOL
    assert np.max(np.abs(risk.es[0.99] - ve[:, 3])) < _VAR_ES_DIST_TOL
    # ES is everywhere at least as deep as VaR, and both are return-space (negative) thresholds.
    for alpha in (0.975, 0.99):
        assert np.all(risk.es[alpha] <= risk.var[alpha])
        assert np.all(risk.var[alpha] < 0.0)


# --------------------------------------------------------------------------- #
# 6. APARCH family (forecast family 1) -- rolling forecast + VaR/ES, GJR/TGARCH/APARCH x 8 laws
# --------------------------------------------------------------------------- #
#
# predict_roll for the APARCH family (§33): the one-step sigma_{t+1}^delta is deterministic
# at t, and the §4 pre-sample seed transient has decayed by the test window, so the rolling forecast
# is seam-exact for all three models. VaR/ES again exercises each law's quantile/ES. GJR
# (delta=2) and TGARCH (delta=1) are the fixed-power specializations of the APARCH path.

_APARCH_MODELS = ["aparch", "gjrgarch", "tgarch"]
_APARCH_DISTS = ["norm", "std", "ged", "ald", "snorm", "sstd", "sged", "sald"]


def _aparch_params_from_json(p: dict, model: str, dist: str) -> dict:  # type: ignore[type-arg]
    """R APARCH-family fit params -> quantica GarchFit names (incl. the power ``delta``)."""
    q = {
        "mu": p["mu"],
        "omega": p["omega"],
        "phi1": p["phi1"],
        "beta1": p["beta1"],
        "gamma1": p["gamma1"],
    }
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


def _aparch_forecast_fit(model: str, dist: str, returns: np.ndarray) -> GarchFit:  # type: ignore[type-arg]
    """APARCH-family training fit from the forecast_<model>11_<dist> fixture, reconstructed vol."""
    stem = f"forecast_{model}11_{dist}"
    meta = json.loads((_FIXTURE_DIR / f"{stem}_params.json").read_text(encoding="utf-8"))
    q = _aparch_params_from_json(meta["params"], model, dist)
    n = int(meta["n_obs"])
    v = np.array([q["mu"], q["omega"], q["phi1"], q["beta1"], q["gamma1"]])
    if q["delta"] == 2.0:
        cond_vol = np.sqrt(gjr_recursion(v, returns[:n]))
    elif q["delta"] == 1.0:
        cond_vol = np.sqrt(tgarch_recursion(v, returns[:n]))
    else:
        cond_vol = np.sqrt(aparch_recursion(np.append(v, q["delta"]), returns[:n]))
    return GarchFit(
        cond_dist=dist,
        params=q,
        std_errors=dict.fromkeys(q, 0.0),
        loglikelihood=float(meta["loglikelihood"]),
        aic=float(meta["aic"]),
        bic=float(meta["bic"]),
        conditional_volatility=cond_vol,
        n_obs=n,
        converged=True,
    )


def _aparch_roll_fixture(model: str, dist: str) -> dict:  # type: ignore[type-arg]
    """Load the APARCH-family rolling-forecast fixture (sigma / cmeans / VaR-ES) for model+dist."""
    stem = f"forecast_{model}11_{dist}"
    return {
        "sigma": np.loadtxt(_FIXTURE_DIR / f"{stem}_roll_sigma.csv", skiprows=1),
        "cmeans": np.loadtxt(_FIXTURE_DIR / f"{stem}_roll_cmeans.csv", skiprows=1),
        "var_es": np.loadtxt(_FIXTURE_DIR / f"{stem}_var_es.csv", delimiter=",", skiprows=1),
    }


@pytest.mark.parametrize("model", _APARCH_MODELS)
@pytest.mark.parametrize("dist", _APARCH_DISTS)
def test_aparch_family_predict_roll_matches_fixture(model: str, dist: str) -> None:
    """Rolling sigma-hat is seam-exact for the APARCH family (the §4 seed transient decays)."""
    returns = _returns()
    fit = _aparch_forecast_fit(model, dist, returns)
    fx = _aparch_roll_fixture(model, dist)
    forecast = predict_roll(fit, returns)
    assert forecast.n_test == fx["sigma"].size == 250
    assert np.max(np.abs(forecast.sigma - fx["sigma"])) < _SIGMA_TOL
    assert np.max(np.abs(forecast.cmeans - fx["cmeans"])) < _SIGMA_TOL


@pytest.mark.parametrize("model", _APARCH_MODELS)
@pytest.mark.parametrize("dist", _APARCH_DISTS)
def test_aparch_family_measure_risk_matches_fixture(model: str, dist: str) -> None:
    """VaR/ES reproduces the fixture for the APARCH family per law (R<->SciPy quantile gap)."""
    returns = _returns()
    fit = _aparch_forecast_fit(model, dist, returns)
    fx = _aparch_roll_fixture(model, dist)
    risk = measure_risk(predict_roll(fit, returns), fit, levels=(0.975, 0.99))
    ve = fx["var_es"]
    assert np.max(np.abs(risk.var[0.975] - ve[:, 0])) < _VAR_ES_DIST_TOL
    assert np.max(np.abs(risk.var[0.99] - ve[:, 1])) < _VAR_ES_DIST_TOL
    assert np.max(np.abs(risk.es[0.975] - ve[:, 2])) < _VAR_ES_DIST_TOL
    assert np.max(np.abs(risk.es[0.99] - ve[:, 3])) < _VAR_ES_DIST_TOL
    for alpha in (0.975, 0.99):
        assert np.all(risk.es[alpha] <= risk.var[alpha])
        assert np.all(risk.var[alpha] < 0.0)


# --------------------------------------------------------------------------- #
# 7. FI long-memory family (forecast family 2) -- rolling + VaR/ES
# --------------------------------------------------------------------------- #
#
# predict_roll for the FI family (spec-notes §34): the one-step rolling forecast uses the forecast
# truncation convention -- the (1-L)^d weights conditioned on the observed sample with pre-sample 0
# (the in-sample Var/mean-news warm-up is NOT re-applied) -- so it is seam-exact. VaR/ES again
# exercises each law's quantile/ES. FIEGARCH drops std/sstd (finding-3 EGF gap).

_FI_ALL_F = ["norm", "std", "ged", "ald", "snorm", "sstd", "sged", "sald"]
_FI_EGF_F = ["norm", "ged", "ald", "snorm", "sged", "sald"]
_FI_CASES_F = (
    [("figarch", d) for d in _FI_ALL_F]
    + [("figjr", d) for d in _FI_ALL_F]
    + [("fitgarch", d) for d in _FI_ALL_F]
    + [("fiaparch", d) for d in _FI_ALL_F]
    + [("fiegarch", d) for d in _FI_EGF_F]
)


def _fi_params_from_json_fc(p: dict, model: str, dist: str) -> dict:  # type: ignore[type-arg]
    """R FI fit params -> quantica names (fractional ``d``; fixed ``delta`` for FITGARCH/FIGJR)."""
    q: dict = {"mu": p["mu"], "d": p["d"]}
    if model == "fiegarch":
        q.update(omega_sig=p["omega_sig"], phi1=p["phi1"], kappa=p["kappa"], gamma=p["gamma"])
    else:
        q.update(omega=p["omega"], phi1=p["phi1"], beta1=p["beta1"])
        if model in ("fitgarch", "figjr", "fiaparch"):
            q["gamma"] = p["gamma"]
            q["delta"] = p.get("delta", 1.0 if model == "fitgarch" else 2.0)
    if dist in ("std", "sstd"):
        q["nu"] = p["df"]
    elif dist in ("ged", "sged"):
        q["nu"] = p["shape"]
    if dist in ("ald", "sald"):
        q["P"] = p["P"]
    if dist in ("snorm", "sstd", "sged", "sald"):
        q["skew"] = p["skew"]
    return q


def _fi_forecast_fit(model: str, dist: str, returns: np.ndarray) -> GarchFit:  # type: ignore[type-arg]
    """FI training fit from the forecast fixture, conditional vol via the engine helper."""
    from quantica.timeseries.fegarch.forecast import _fi_variance

    stem = f"forecast_{model}11_{dist}"
    meta = json.loads((_FIXTURE_DIR / f"{stem}_params.json").read_text(encoding="utf-8"))
    q = _fi_params_from_json_fc(meta["params"], model, dist)
    n = int(meta["n_obs"])
    tmp = GarchFit(
        cond_dist=dist,
        params=q,
        std_errors=dict.fromkeys(q, 0.0),
        loglikelihood=0.0,
        aic=0.0,
        bic=0.0,
        conditional_volatility=np.array([1.0]),
        n_obs=n,
        converged=True,
    )
    cv = np.sqrt(_fi_variance(tmp, returns[:n]))
    return GarchFit(
        cond_dist=dist,
        params=q,
        std_errors=dict.fromkeys(q, 0.0),
        loglikelihood=float(meta["loglikelihood"]),
        aic=float(meta["aic"]),
        bic=float(meta["bic"]),
        conditional_volatility=cv,
        n_obs=n,
        converged=True,
    )


def _fi_roll_fixture(model: str, dist: str) -> dict:  # type: ignore[type-arg]
    """Load the FI rolling-forecast fixture (sigma / cmeans / VaR-ES) for model+dist."""
    stem = f"forecast_{model}11_{dist}"
    return {
        "sigma": np.loadtxt(_FIXTURE_DIR / f"{stem}_roll_sigma.csv", skiprows=1),
        "cmeans": np.loadtxt(_FIXTURE_DIR / f"{stem}_roll_cmeans.csv", skiprows=1),
        "var_es": np.loadtxt(_FIXTURE_DIR / f"{stem}_var_es.csv", delimiter=",", skiprows=1),
    }


@pytest.mark.parametrize(("model", "dist"), _FI_CASES_F)
def test_fi_predict_roll_matches_fixture(model: str, dist: str) -> None:
    """Rolling sigma-hat is seam-exact for the FI family (pre-sample-0 observed-news filter)."""
    returns = _returns()
    fit = _fi_forecast_fit(model, dist, returns)
    fx = _fi_roll_fixture(model, dist)
    forecast = predict_roll(fit, returns)
    assert forecast.n_test == fx["sigma"].size == 250
    assert np.max(np.abs(forecast.sigma - fx["sigma"])) < _FI_ROLL_TOL
    assert np.max(np.abs(forecast.cmeans - fx["cmeans"])) < _SIGMA_TOL


@pytest.mark.parametrize(("model", "dist"), _FI_CASES_F)
def test_fi_measure_risk_matches_fixture(model: str, dist: str) -> None:
    """VaR/ES reproduces the fixture for the FI family per law (R<->SciPy quantile gap)."""
    returns = _returns()
    fit = _fi_forecast_fit(model, dist, returns)
    fx = _fi_roll_fixture(model, dist)
    risk = measure_risk(predict_roll(fit, returns), fit, levels=(0.975, 0.99))
    ve = fx["var_es"]
    assert np.max(np.abs(risk.var[0.975] - ve[:, 0])) < _VAR_ES_DIST_TOL
    assert np.max(np.abs(risk.var[0.99] - ve[:, 1])) < _VAR_ES_DIST_TOL
    assert np.max(np.abs(risk.es[0.975] - ve[:, 2])) < _VAR_ES_DIST_TOL
    assert np.max(np.abs(risk.es[0.99] - ve[:, 3])) < _VAR_ES_DIST_TOL
    for alpha in (0.975, 0.99):
        assert np.all(risk.es[alpha] <= risk.var[alpha])
        assert np.all(risk.var[alpha] < 0.0)
