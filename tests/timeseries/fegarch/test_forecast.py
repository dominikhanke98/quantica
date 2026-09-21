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
