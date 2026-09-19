"""FARIMA-in-mean x GARCH(1,1) dual fits -- the D != 0 fractional mean (Phase 5).

The fractional-mean extension of the ARMA-in-mean work: the demeaned series is fractionally
differenced by D via the Phase-3 fracdiff operator (1-B)^D (full history, L=n-1) before the ARMA
recursion, then the mean-residuals drive the existing GARCH variance. Validation follows the
ARMA-in-mean §23 pattern -- the seam (fracdiff + ARMA recursion + coupling) is asserted
machine-exact separately from the fitted match; the seed is the ARMA-in-mean bounded-limit.

* **Seam machine-exact**: at the fixture's params + its implied sigma_0^2 the fractional-mean
  recursion reproduces sigma to ~1e-15 (6.9e-18 / 1.4e-17).
* **Fitted match**: farima0d0 (pure fractional mean) is identified -- loglik/variance/D tight;
  farima1d1's ARMA terms match the fixture (both at the near-common-root dominating optimum -- the
  corroboration that fEGarch's FARIMA optimizer reached the point its plain-ARMA optimizer missed).
* **Reductions**: (1-B)^0 = identity, so FARIMA(D=0) == ARMA-in-mean exactly; FARIMA(0,d,0) at D=0
  is the constant mean (y - mu).
* **Known-truth**: a simulated FARIMA(0,d,0)-GARCH with identified D=0.3 recovers D -- the positive
  control that fractional-mean estimation works when the data has genuine mean long-memory.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from quantica.timeseries.fegarch import farima_mean_residuals, fit_farima_garch
from quantica.timeseries.fegarch.fracdiff import fracdiff_coeffs
from quantica.timeseries.fegarch.garch import _garch_variance
from quantica.timeseries.fegarch.mean import arma_mean_residuals

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "fegarch"

# tag -> (P, Q); the committed FARIMA fixtures are norm-only. D is always present.
_ORDERS = {"farima0d0": (0, 0), "farima1d1": (1, 1)}


def _returns() -> np.ndarray:  # type: ignore[type-arg]
    """The committed synthetic return series."""
    return np.loadtxt(_FIXTURE_DIR / "synthetic_returns.csv", skiprows=1)


def _load(tag: str) -> tuple[dict, np.ndarray]:  # type: ignore[type-arg]
    """Load a FARIMA fixture's params dict and sigma series."""
    meta = json.loads(
        (_FIXTURE_DIR / f"fit_{tag}_garch11_norm_params.json").read_text(encoding="utf-8")
    )
    sigma = np.loadtxt(_FIXTURE_DIR / f"fit_{tag}_garch11_norm_sigma.csv", skiprows=1)
    return meta, sigma


# --------------------------------------------------------------------------- #
# The fractional-mean recursion: reductions
# --------------------------------------------------------------------------- #


def test_farima_reduces_to_arma_and_constant_at_D0() -> None:
    """(1-B)^0 = identity, so FARIMA(D=0) == ARMA-in-mean, and FARIMA(0,d,0) at D=0 == y - mu."""
    y = _returns()
    assert np.array_equal(fracdiff_coeffs(0.0, 5), np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
    # FARIMA(1,d,1) at D=0 == ARMA(1,1)-in-mean, exactly.
    r_farima = farima_mean_residuals(y, 0.0005, 0.3, -0.2, 0.0)
    r_arma = arma_mean_residuals(y, 0.0005, 0.3, -0.2)
    assert np.max(np.abs(r_farima - r_arma)) == 0.0
    # FARIMA(0,d,0) at D=0 (no AR/MA) == the constant-mean residuals y - mu.
    assert np.max(np.abs(farima_mean_residuals(y, 0.0005, 0.0, 0.0, 0.0) - (y - 0.0005))) == 0.0


# --------------------------------------------------------------------------- #
# Seam machine-exact (fixture params + implied seed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tag", ["farima0d0", "farima1d1"])
def test_seam_machine_exact(tag: str) -> None:
    """At the fixture params + its implied sigma_0^2, the fractional-mean recursion rebuilds sigma.

    The fracdiff (1-B)^D of the demeaned series + the ARMA recursion + the existing GARCH coupling,
    seeded from the fixture's own sigma_0^2, pin the whole series to machine order -- proof the
    fractional-mean recursion and coupling are exactly fEGarch's.
    """
    meta, sigma_fix = _load(tag)
    p = meta["params"]
    r = farima_mean_residuals(_returns(), p["mu"], p.get("ar1", 0.0), p.get("ma1", 0.0), p["D"])
    implied = (sigma_fix[0] ** 2 - p["omega"]) / (p["phi1"] + p["beta1"])
    sigma = np.sqrt(_garch_variance(r, p["omega"], p["phi1"], p["beta1"], implied))
    assert np.max(np.abs(sigma - sigma_fix)) < 1e-14


@pytest.mark.parametrize("tag", ["farima0d0", "farima1d1"])
def test_bounded_seed_transient(tag: str) -> None:
    """The Var(r,ddof=1) bounded seed leaves only a decaying transient (<1e-5, machine-0 tail).

    Same bounded-limit situation as ARMA-in-mean (fEGarch's exact dual seed is an internal
    preliminary-residual quantity, no published-math form; §12). The Var(r) analog's seed error is a
    bounded, geometrically-decaying transient.
    """
    meta, sigma_fix = _load(tag)
    p = meta["params"]
    r = farima_mean_residuals(_returns(), p["mu"], p.get("ar1", 0.0), p.get("ma1", 0.0), p["D"])
    seed = p["omega"] + (p["phi1"] + p["beta1"]) * float(np.var(r, ddof=1))
    dev = np.abs(np.sqrt(_garch_variance(r, p["omega"], p["phi1"], p["beta1"], seed)) - sigma_fix)
    assert dev.max() < 1e-5
    assert dev[500:].max() < 1e-12


# --------------------------------------------------------------------------- #
# Fitted match (bounded seed)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tag", ["farima0d0", "farima1d1"])
def test_farima_fitted_match(tag: str) -> None:
    """fit_farima_garch reproduces the fixture: loglik, variance block, sigma, and the mean D tight.

    farima0d0's pure fractional D is identified (recovers the fixture's D); farima1d1's ARMA terms
    match the fixture (both at the near-common-root dominating optimum), with D near-zero.
    Nelder-Mead navigates the stiff fractional-D ridge that defeats L-BFGS-B.
    """
    p_order, q_order = _ORDERS[tag]
    meta, sigma_fix = _load(tag)
    fx = meta["params"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_farima_garch(_returns(), mean_orders=(p_order, q_order), cond_dist="norm")
    assert fit.converged
    assert abs(fit.loglikelihood - meta["loglikelihood"]) < 1e-3  # bounded-seed loglik effect
    assert abs(fit.params["alpha"] - fx["phi1"]) / fx["phi1"] < 5e-3  # variance block tight
    assert abs(fit.params["beta"] - fx["beta1"]) / fx["beta1"] < 5e-3
    assert np.max(np.abs(fit.conditional_volatility - sigma_fix)) < 1e-5  # bounded seed
    assert abs(fit.params["D"] - fx["D"]) < 1e-3  # D recovers the fixture (identified or near-zero)
    if p_order:  # farima1d1: ARMA terms match the dominating optimum (unlike the arma11 fixture)
        assert abs(fit.params["ar1"] - fx["ar1"]) < 5e-3
    if q_order:
        assert abs(fit.params["ma1"] - fx["ma1"]) < 5e-3


# --------------------------------------------------------------------------- #
# Known-truth
# --------------------------------------------------------------------------- #


def _sim_farima00_garch(  # type: ignore[no-untyped-def]
    n, *, mu, d_mean, omega, alpha, beta, rng, n_burn=2000
):
    """Simulate FARIMA(0,d,0)-GARCH: GARCH residuals r_t, then y_t = mu + (1-B)^{-D} r_t."""
    total = n + n_burn
    z = rng.standard_normal(total)
    r = np.empty(total)
    s2 = np.empty(total)
    uncond = omega / (1.0 - alpha - beta)
    for t in range(total):
        prev_r2 = r[t - 1] ** 2 if t else uncond
        prev_s2 = s2[t - 1] if t else uncond
        s2[t] = omega + alpha * prev_r2 + beta * prev_s2
        r[t] = np.sqrt(s2[t]) * z[t]
    integ = fracdiff_coeffs(-d_mean, total)  # (1-B)^{-D} fractional integration of the residuals
    y = mu + np.convolve(integ, r)[:total]
    return np.asarray(y[n_burn:], dtype=np.float64)


def test_known_truth_recovers_fractional_mean() -> None:
    """A simulated FARIMA(0,d,0)-GARCH with identified D=0.3 recovers D, alpha, beta in a few SE.

    The positive control (separate from the near-zero flat-ridge fixtures): with genuine mean
    long-memory, the joint QMLE recovers the fractional mean order and the variance block.
    """
    true = {"mu": 0.0003, "d_mean": 0.3, "omega": 3e-6, "alpha": 0.08, "beta": 0.90}
    y = _sim_farima00_garch(8000, **true, rng=np.random.default_rng(0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = fit_farima_garch(y, mean_orders=(0, 0), cond_dist="norm")
    assert fit.converged
    se = fit.std_errors
    assert abs(fit.params["D"] - 0.3) < 4.0 * se["D"]
    assert abs(fit.params["alpha"] - 0.08) < 4.0 * se["alpha"]
    assert abs(fit.params["beta"] - 0.90) < 4.0 * se["beta"]


# --------------------------------------------------------------------------- #
# API contract
# --------------------------------------------------------------------------- #


def test_invalid_mean_orders_and_unsupported_dist() -> None:
    """mean_orders outside {0,1} raise; ald/sald (discrete-P profiling) are not yet supported."""
    y = _returns()
    with pytest.raises(ValueError, match="must each be 0 or 1"):
        fit_farima_garch(y, mean_orders=(2, 0))
    with pytest.raises(NotImplementedError, match="discrete-P profiling"):
        fit_farima_garch(y, mean_orders=(1, 1), cond_dist="ald")
