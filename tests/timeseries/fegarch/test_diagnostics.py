"""Phase 6 diagnostics -- weighted Ljung-Box, Engle-Ng sign-bias, and PIT goodness-of-fit.

Validated against the committed fEGarch diagnostic fixtures (``diag_garch11_norm_*``) from
``fit_test_suite`` on the full-sample GARCH(1,1)/norm fit. fEGarch exposes ONLY p-values (LB adds
the lag m; GoF adds n_bins/df) -- never the raw statistics -- so validation reproduces p-values. The
standardized residuals are machine-exact (the GARCH recursion, Phase 1), so the p-value match tests
the null-distribution computation itself; all three reproduce to ~1e-14 (scipy's Gamma / chi2 /
Student-t CDFs match R's to machine precision, well inside any R<->scipy dist-function band).

The pinned conventions (the review items):

* Ljung-Box is the WEIGHTED portmanteau (Fisher-Gallagher 2012), linearly-decreasing weights
  (m+1-k)/m, GAMMA-approx null (NOT the classic equal-weight chi^2). Simple (level) + squared
  (volatility); adj_df 0 here (constant mean, no ARMA).
* Sign-bias regresses z^2 on {1, S-, S-*eps, S+*eps} -- the SIZE regressors use the RAW residual eps
  (not standardized z); individual OLS t-tests; JOINT is the LM statistic n*R^2 ~ chi^2_3 (NOT the
  F-test -- the fixture p-value distinguishes them: LM 0.6232 vs F 0.6236).
* GoF is the PIT u = F_eta(z) (Phi for norm) binned into equal-probability bins, Pearson chi^2,
  df = n_bins - 1 (estimated params NOT subtracted; the 20->19 fixture row confirms it).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import (
    GarchFit,
    fit_test_suite,
    get_distribution,
    goodness_of_fit_test,
    sign_bias_test,
    standardized_residuals,
    weighted_ljung_box,
)
from quantica.timeseries.fegarch.garch import garch_recursion

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

# The residuals are machine-exact, so the p-value reproduction is bounded only by the R<->scipy
# distribution-function agreement -- which is itself machine precision here.
_PVAL_TOL = 1e-10


def _returns() -> np.ndarray:  # type: ignore[type-arg]
    """The committed synthetic return series (2500 obs)."""
    return np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)


def _load(kind: str) -> dict:  # type: ignore[type-arg]
    """Load a diagnostic fixture JSON (``ljungbox`` / ``signbias`` / ``gof``)."""
    return json.loads((_FIXTURE_DIR / f"diag_garch11_norm_{kind}.json").read_text(encoding="utf-8"))


def _fit_from_fixture(meta: dict, returns: np.ndarray) -> GarchFit:  # type: ignore[type-arg]
    """A :class:`GarchFit` with the fixture's params + machine-exact conditional volatility."""
    p = meta["fit_params"]
    params = {"mu": p["mu"], "omega": p["omega"], "alpha": p["phi1"], "beta": p["beta1"]}
    vec = np.array([params["mu"], params["omega"], params["alpha"], params["beta"]])
    cond_vol = np.sqrt(garch_recursion(vec, returns))
    return GarchFit(
        cond_dist=meta["cond_dist"],
        params=params,
        std_errors=dict.fromkeys(params, 0.0),
        loglikelihood=0.0,
        aic=0.0,
        bic=0.0,
        conditional_volatility=cond_vol,
        n_obs=returns.size,
        converged=True,
    )


# --------------------------------------------------------------------------- #
# Weighted Ljung-Box (Fisher-Gallagher 2012, Gamma-approx)
# --------------------------------------------------------------------------- #


def test_ljung_box_reproduces_fixture_pvalues() -> None:
    """Weighted LB p-values (simple + squared, lags 1..20) match the fixture to ~1e-14."""
    fx = _load("ljungbox")
    returns = _returns()
    fit = _fit_from_fixture(fx, returns)
    z = standardized_residuals(fit, returns)

    lb_simple = weighted_ljung_box(z, m_max=20, fitdf=0)
    lb_squared = weighted_ljung_box(z * z, m_max=20, fitdf=0)

    fix_simple = np.array([row["pval"] for row in fx["simple"]])
    fix_squared = np.array([row["pval"] for row in fx["squared"]])
    assert np.array_equal(lb_simple.lags, np.arange(1, 21))
    assert np.max(np.abs(lb_simple.pvalue - fix_simple)) < _PVAL_TOL
    assert np.max(np.abs(lb_squared.pvalue - fix_squared)) < _PVAL_TOL


def test_ljung_box_is_weighted_not_classic() -> None:
    """The weighted Gamma form differs materially from the classic equal-weight chi^2 Ljung-Box."""
    fx = _load("ljungbox")
    returns = _returns()
    z = standardized_residuals(_fit_from_fixture(fx, returns), returns)
    from scipy import stats

    n = z.size
    zc = z - z.mean()
    g0 = np.mean(zc * zc)
    rho = np.array([np.sum(zc[: n - k] * zc[k:]) / n / g0 for k in range(1, 21)])
    # Classic Ljung-Box at m=20: chi^2_20 -- should NOT match the weighted fixture p-value.
    stat = n * (n + 2) * np.sum(rho**2 / (n - np.arange(1, 21)))
    classic_p = float(stats.chi2.sf(stat, 20))
    assert abs(classic_p - fx["simple"][-1]["pval"]) > 1e-3


def test_ljung_box_rejects_bad_args() -> None:
    """m_max must be positive; an over-large fitdf gives a non-positive Gamma parameter."""
    z = standardized_residuals(_fit_from_fixture(_load("ljungbox"), _returns()), _returns())
    with pytest.raises(ValueError, match="m_max"):
        weighted_ljung_box(z, m_max=0)
    with pytest.raises(ValueError, match="fitdf"):
        weighted_ljung_box(z, m_max=5, fitdf=100)


# --------------------------------------------------------------------------- #
# Sign-bias (Engle-Ng 1993)
# --------------------------------------------------------------------------- #


def test_sign_bias_reproduces_fixture_pvalues() -> None:
    """The four sign-bias p-values match the fixture (raw-eps size terms; joint = LM chi^2_3)."""
    fx = _load("signbias")
    returns = _returns()
    fit = _fit_from_fixture(fx, returns)
    z = standardized_residuals(fit, returns)
    eps = returns - fit.params["mu"]

    res = sign_bias_test(z, eps)
    rows = {row["name"]: row["pval"] for row in fx["rows"]}
    assert res.sign_bias_pvalue == pytest.approx(rows["Sign bias"], abs=_PVAL_TOL)
    assert res.negative_size_pvalue == pytest.approx(rows["Negative sign bias"], abs=_PVAL_TOL)
    assert res.positive_size_pvalue == pytest.approx(rows["Positive sign bias"], abs=_PVAL_TOL)
    assert res.joint_pvalue == pytest.approx(rows["Joint"], abs=_PVAL_TOL)


def test_sign_bias_joint_is_lm_not_f() -> None:
    """The joint test is LM (n*R^2 ~ chi^2_3), not the OLS F-test -- the fixture distinguishes."""
    fx = _load("signbias")
    returns = _returns()
    fit = _fit_from_fixture(fx, returns)
    z = standardized_residuals(fit, returns)
    eps = returns - fit.params["mu"]
    joint_fix = next(r["pval"] for r in fx["rows"] if r["name"] == "Joint")

    # Reconstruct the OLS F-test p-value and confirm it is the WRONG one (distinct from fixture).
    from scipy import stats

    zl, el = z[:-1], eps[:-1]
    y = z[1:] ** 2
    s_neg = (zl < 0).astype(float)
    design = np.column_stack([np.ones_like(zl), s_neg, s_neg * el, (1 - s_neg) * el])
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    r = y - design @ beta
    m = y.size
    rss, tot = r @ r, np.sum((y - y.mean()) ** 2)
    f_stat = ((tot - rss) / 3) / (rss / (m - 4))
    f_p = float(stats.f.sf(f_stat, 3, m - 4))
    assert sign_bias_test(z, eps).joint_pvalue == pytest.approx(joint_fix, abs=_PVAL_TOL)
    assert abs(f_p - joint_fix) > 1e-4  # the F-test would be ~0.6236 vs the LM fixture ~0.6232


def test_sign_bias_size_terms_use_raw_residuals() -> None:
    """Using standardized z (not raw eps) for the size regressors does NOT match the fixture."""
    fx = _load("signbias")
    returns = _returns()
    fit = _fit_from_fixture(fx, returns)
    z = standardized_residuals(fit, returns)
    # Feeding z as the "residuals" (wrong: standardized size) breaks the neg/pos match.
    wrong = sign_bias_test(z, z)
    rows = {row["name"]: row["pval"] for row in fx["rows"]}
    assert abs(wrong.negative_size_pvalue - rows["Negative sign bias"]) > 1e-2


# --------------------------------------------------------------------------- #
# Goodness-of-fit (PIT Pearson chi^2)
# --------------------------------------------------------------------------- #


def test_goodness_of_fit_reproduces_fixture_pvalues() -> None:
    """PIT Pearson chi^2 p-values (n_bins 20/30/40/50, df=n_bins-1) match the fixture ~1e-14."""
    fx = _load("gof")
    returns = _returns()
    fit = _fit_from_fixture(fx, returns)
    z = standardized_residuals(fit, returns)

    res = goodness_of_fit_test(z, get_distribution("norm"), n_bins=(20, 30, 40, 50))
    fix_nbins = np.array([row["n_bins"] for row in fx["rows"]])
    fix_df = np.array([row["df"] for row in fx["rows"]])
    fix_p = np.array([row["pval"] for row in fx["rows"]])
    assert np.array_equal(res.n_bins, fix_nbins)
    assert np.array_equal(res.dof, fix_df)  # df = n_bins - 1 (no param subtraction)
    assert np.max(np.abs(res.pvalue - fix_p)) < _PVAL_TOL


# --------------------------------------------------------------------------- #
# The bundled suite (mirrors fEGarch's fit_test_suite)
# --------------------------------------------------------------------------- #


def test_fit_test_suite_matches_all_fixtures() -> None:
    """The fit-level suite reproduces every diagnostic fixture in one call from a GarchFit."""
    lb, sb, gof = _load("ljungbox"), _load("signbias"), _load("gof")
    returns = _returns()
    fit = _fit_from_fixture(lb, returns)
    suite = fit_test_suite(fit, returns)

    assert (
        np.max(np.abs(suite.ljung_box_simple.pvalue - [r["pval"] for r in lb["simple"]]))
        < _PVAL_TOL
    )
    assert (
        np.max(np.abs(suite.ljung_box_squared.pvalue - [r["pval"] for r in lb["squared"]]))
        < _PVAL_TOL
    )
    rows = {row["name"]: row["pval"] for row in sb["rows"]}
    assert suite.sign_bias.sign_bias_pvalue == pytest.approx(rows["Sign bias"], abs=_PVAL_TOL)
    assert suite.sign_bias.joint_pvalue == pytest.approx(rows["Joint"], abs=_PVAL_TOL)
    assert (
        np.max(np.abs(suite.goodness_of_fit.pvalue - [r["pval"] for r in gof["rows"]])) < _PVAL_TOL
    )
    # Constant-mean GARCH: the simple-residual df correction is 0 (no ARMA mean params).
    assert suite.ljung_box_simple.fitdf == 0


def test_standardized_residuals_length_guard() -> None:
    """A returns series of the wrong length is rejected."""
    fit = _fit_from_fixture(_load("gof"), _returns())
    with pytest.raises(ValueError, match="length"):
        standardized_residuals(fit, _returns()[:-1])
