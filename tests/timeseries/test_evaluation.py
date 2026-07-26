"""Validation of the forecast-evaluation layer (numerical-validation skill, forecasting flavour).

The headline is *validate-the-validator*: on data with a known truth, the Diebold--Mariano test
must have the right **size** (reject equal accuracy at ~the nominal rate when two forecasts are
truly equally good) and **power** (reject when one is genuinely better). The subtlety the whole
pillar exists to demonstrate is that this holds **only with** the HAC/Newey--West variance
correction — when loss differentials are serially correlated (as volatility forecast errors are),
the naive-variance test badly over-rejects. A known-truth oracle check confirms QLIKE ranks the
true variance best, and hand-computation anchors pin the DM statistic, the Mincer--Zarnowitz
regression and the loss formulas to their definitions.
"""

from __future__ import annotations

import numpy as np
from quantica.timeseries import (
    diebold_mariano,
    mincer_zarnowitz,
    mse_loss,
    qlike_loss,
    simulate_garch,
    simulate_loss_differential,
)


def _rejection_rates(
    mean: float, phi: float, *, reps: int, n: int, seed: int
) -> tuple[float, float]:
    """Empirical rejection rate at the 5% level, (naive, HAC), over ``reps`` simulated diffs."""
    rng = np.random.default_rng(seed)
    zero = np.zeros(n)
    naive = np.empty(reps)
    hac = np.empty(reps)
    for i in range(reps):
        d = simulate_loss_differential(n, mean=mean, phi=phi, rng=rng)
        naive[i] = diebold_mariano(d, zero, hac=False).p_value < 0.05
        hac[i] = diebold_mariano(d, zero, hac=True).p_value < 0.05
    return float(naive.mean()), float(hac.mean())


# --------------------------------------------------------------------------- #
# The headline: validate the Diebold--Mariano test's size and power
# --------------------------------------------------------------------------- #


def test_hac_restores_size_where_naive_over_rejects() -> None:
    """With serially-correlated equal-accuracy losses, naive DM over-rejects; HAC restores size."""
    naive, hac = _rejection_rates(mean=0.0, phi=0.5, reps=400, n=500, seed=0)
    # Nominal size is 5%. The naive test badly over-rejects (positive autocorrelation inflates
    # its statistic); the HAC-corrected test sits near nominal.
    assert naive > 0.15  # over-sized (empirically ~0.26)
    assert hac < 0.13  # near nominal (empirically ~0.09)
    assert naive > hac + 0.05  # the correction visibly matters


def test_no_correction_needed_without_serial_correlation() -> None:
    """With iid loss differentials (phi=0) both naive and HAC are correctly sized."""
    naive, hac = _rejection_rates(mean=0.0, phi=0.0, reps=400, n=500, seed=1)
    assert naive < 0.12
    assert hac < 0.12  # HAC does no harm when it is not needed


def test_diebold_mariano_has_power_against_a_worse_forecast() -> None:
    """When one forecast is genuinely worse (mean>0), the HAC-corrected test rejects often."""
    _naive, hac = _rejection_rates(mean=0.15, phi=0.5, reps=400, n=500, seed=2)
    assert hac > 0.30  # real power (empirically ~0.48)


# --------------------------------------------------------------------------- #
# Known truth: the oracle (true variance) is the best forecast
# --------------------------------------------------------------------------- #


def test_oracle_variance_beats_a_constant_forecast() -> None:
    """QLIKE ranks the true conditional variance below a constant; DM detects it decisively."""
    rng = np.random.default_rng(3)
    returns, variance = simulate_garch(
        4000, omega=0.05, alpha=0.10, beta=0.88, model="GARCH", rng=rng, return_variance=True
    )
    proxy = returns**2
    constant = np.full_like(variance, variance.mean())  # unconditional (misspecified) forecast
    oracle_loss = qlike_loss(variance, proxy)
    constant_loss = qlike_loss(constant, proxy)
    assert oracle_loss.mean() < constant_loss.mean()  # oracle is better (Patton optimality)
    dm = diebold_mariano(oracle_loss, constant_loss)
    assert dm.statistic < -3.0  # oracle decisively better; A(oracle) worse would be positive
    assert dm.p_value < 0.01


# --------------------------------------------------------------------------- #
# Anchors: statistics against their definitions / direct computations
# --------------------------------------------------------------------------- #


def test_loss_functions_match_their_formulas() -> None:
    """MSE and QLIKE equal their direct definitions element-wise."""
    rng = np.random.default_rng(4)
    forecast = rng.uniform(0.5, 2.0, size=50)
    proxy = rng.uniform(0.0, 3.0, size=50)
    assert np.allclose(mse_loss(forecast, proxy), (proxy - forecast) ** 2)
    assert np.allclose(qlike_loss(forecast, proxy), proxy / forecast + np.log(forecast))


def test_diebold_mariano_statistic_matches_hand_newey_west() -> None:
    """The DM statistic equals a hand Newey--West computation at a fixed lag."""
    rng = np.random.default_rng(5)
    d = simulate_loss_differential(300, mean=0.05, phi=0.4, rng=rng)
    lags = 4
    result = diebold_mariano(d, np.zeros_like(d), hac=True, lags=lags)

    n = d.size
    centered = d - d.mean()
    lrv = float(np.mean(centered**2))
    for k in range(1, lags + 1):
        lrv += 2.0 * (1.0 - k / (lags + 1)) * float(np.mean(centered[k:] * centered[:-k]))
    expected = d.mean() / np.sqrt(lrv / n)
    assert np.isclose(result.statistic, expected, atol=1e-12)
    assert result.lags == lags


def test_naive_diebold_mariano_uses_plain_variance() -> None:
    """Without HAC the DM statistic uses the ordinary sample variance of the differential."""
    rng = np.random.default_rng(6)
    d = simulate_loss_differential(200, mean=0.1, phi=0.0, rng=rng)
    result = diebold_mariano(d, np.zeros_like(d), hac=False)
    expected = d.mean() / np.sqrt(np.var(d) / d.size)  # population variance (ddof=0)
    assert np.isclose(result.statistic, expected, atol=1e-12)
    assert result.lags == 0


def test_mincer_zarnowitz_matches_direct_ols_wald() -> None:
    """The MZ regression and joint test match a direct statsmodels OLS + Wald computation."""
    import statsmodels.api as sm

    rng = np.random.default_rng(7)
    forecast = rng.uniform(0.5, 2.0, size=400)
    proxy = 0.1 + 0.9 * forecast + rng.normal(scale=0.3, size=400)  # nearly efficient
    result = mincer_zarnowitz(proxy, forecast, hac_lags=6)

    x = sm.add_constant(forecast)
    fit = sm.OLS(proxy, x).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
    wald = fit.wald_test((np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 1.0])), scalar=True)
    assert np.isclose(result.intercept, fit.params[0], atol=1e-12)
    assert np.isclose(result.slope, fit.params[1], atol=1e-12)
    assert np.isclose(result.joint_p_value, float(wald.pvalue), atol=1e-12)
    assert np.isclose(result.r_squared, fit.rsquared, atol=1e-12)


def test_qlike_is_finite_with_a_zero_proxy() -> None:
    """The ranking-form QLIKE stays finite when the squared-return proxy is exactly zero."""
    forecast = np.array([0.5, 1.0, 2.0])
    proxy = np.array([0.0, 1.0, 0.0])  # flat days -> zero squared return
    loss = qlike_loss(forecast, proxy)
    assert np.all(np.isfinite(loss))
