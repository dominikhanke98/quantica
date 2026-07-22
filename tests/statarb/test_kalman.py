"""Validation of the Kalman dynamic hedge ratio (numerical-validation skill).

The point of the filter is that **dynamic beats static when the relationship moves**. The
headline known-truth simulates a pair whose true hedge ratio drifts (and, separately, steps)
and confirms the Kalman estimate tracks it — within its own uncertainty bands, and with far
lower tracking error than a static OLS/cointegration coefficient fitted once. Two anchors pin
the recursion: in the zero-process-noise limit on a constant ratio the filter reduces to
recursive least squares and its estimate converges to OLS; and when the model is
well-specified the standardised innovations are white.
"""

from __future__ import annotations

import numpy as np
import pytest
from quantica.statarb import (
    generate_time_varying_pair,
    kalman_hedge_ratio,
)


def _static_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """The single OLS coefficient a static cointegration fit would use."""
    design = np.column_stack([x, np.ones_like(x)])
    return float(np.linalg.lstsq(design, y, rcond=None)[0][0])


# --------------------------------------------------------------------------- #
# Headline: tracking a moving hedge ratio, and beating static
# --------------------------------------------------------------------------- #


def test_tracks_drifting_ratio_and_beats_static() -> None:
    """On a linearly drifting true ratio the Kalman filter tracks it far better than OLS."""
    true = np.linspace(1.0, 2.0, 1000)
    y, x = generate_time_varying_pair(true, np.random.default_rng(0), alpha=2.0, obs_vol=1.0)
    result = kalman_hedge_ratio(y, x, process_var=1e-4, obs_var=1.0)

    kalman_rmse = float(np.sqrt(np.mean((result.hedge_ratio - true) ** 2)))
    static_rmse = float(np.sqrt(np.mean((_static_hedge_ratio(y, x) - true) ** 2)))
    assert kalman_rmse < static_rmse  # dynamic beats static when the relationship moves
    assert kalman_rmse < 0.2  # and tracks the truth closely
    # The true path stays inside the filter's own uncertainty band (post burn-in).
    post = slice(100, None)
    inside = np.abs(result.hedge_ratio[post] - true[post]) <= 3.0 * result.hedge_ratio_std[post]
    assert np.mean(inside) > 0.90


def test_tracks_step_change() -> None:
    """A regime shift in the true ratio is followed by the filter but missed by static OLS."""
    true = np.where(np.arange(800) < 400, 1.0, 2.0)
    y, x = generate_time_varying_pair(true, np.random.default_rng(1), alpha=2.0, obs_vol=1.0)
    result = kalman_hedge_ratio(y, x, process_var=1e-3, obs_var=1.0)

    assert abs(np.median(result.hedge_ratio[320:390]) - 1.0) < 0.2  # pre-step level
    assert abs(np.median(result.hedge_ratio[700:]) - 2.0) < 0.2  # post-step level recovered
    kalman_rmse = float(np.sqrt(np.mean((result.hedge_ratio - true) ** 2)))
    static_rmse = float(np.sqrt(np.mean((_static_hedge_ratio(y, x) - true) ** 2)))
    assert kalman_rmse < static_rmse


# --------------------------------------------------------------------------- #
# Anchors: the recursion is right
# --------------------------------------------------------------------------- #


def test_reduces_to_ols_in_zero_process_noise_limit() -> None:
    """With process_var -> 0 on a constant ratio, the filter converges to the OLS estimate.

    Zero state evolution makes the Kalman filter recursive least squares, whose estimate on
    the full sample equals the ordinary least-squares fit — a known limiting case.
    """
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.standard_normal(600)) + 50.0
    y = 3.0 + 1.7 * x + rng.standard_normal(600) * 0.5
    result = kalman_hedge_ratio(y, x, process_var=1e-12, obs_var=1.0)

    design = np.column_stack([x, np.ones_like(x)])
    ols_beta, ols_intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    assert np.isclose(result.hedge_ratio[-1], ols_beta, atol=1e-4)
    assert np.isclose(result.intercept[-1], ols_intercept, atol=1e-4)


def test_innovations_are_white_when_well_specified() -> None:
    """A well-specified constant-ratio fit yields near-white standardised innovations."""
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.standard_normal(1500)) + 50.0
    y = 1.5 * x + rng.standard_normal(1500) * 1.0
    result = kalman_hedge_ratio(y, x, process_var=1e-8, obs_var=1.0)

    standardised = result.spread[100:] / np.sqrt(result.innovation_variance[100:])
    lag1_autocorr = float(np.corrcoef(standardised[:-1], standardised[1:])[0, 1])
    assert abs(lag1_autocorr) < 0.1  # approximately white


# --------------------------------------------------------------------------- #
# The tuning knob: process/observation variance ratio controls adaptation speed
# --------------------------------------------------------------------------- #


def test_larger_process_variance_adapts_faster() -> None:
    """A too-small process variance under-adapts to a drift; increasing it tracks better."""
    true = np.linspace(1.0, 2.0, 1000)
    y, x = generate_time_varying_pair(true, np.random.default_rng(4), alpha=2.0)
    slow = kalman_hedge_ratio(y, x, process_var=1e-7, obs_var=1.0)  # nearly static
    fast = kalman_hedge_ratio(y, x, process_var=1e-4, obs_var=1.0)  # adapts

    slow_rmse = np.sqrt(np.mean((slow.hedge_ratio - true) ** 2))
    fast_rmse = np.sqrt(np.mean((fast.hedge_ratio - true) ** 2))
    assert fast_rmse < slow_rmse  # the adapting filter tracks the moving ratio better
    assert fast.process_var == 1e-4  # the knob is exposed, not hard-coded


def test_rejects_bad_inputs() -> None:
    """Mismatched lengths and invalid noise parameters are rejected."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="equal length"):
        kalman_hedge_ratio(
            rng.standard_normal(100), rng.standard_normal(99), process_var=1e-4, obs_var=1.0
        )
    with pytest.raises(ValueError, match="process_var"):
        kalman_hedge_ratio(
            rng.standard_normal(100), rng.standard_normal(100), process_var=-1.0, obs_var=1.0
        )
    with pytest.raises(ValueError, match="obs_var"):
        kalman_hedge_ratio(
            rng.standard_normal(100), rng.standard_normal(100), process_var=1e-4, obs_var=0.0
        )
    with pytest.raises(ValueError, match="initial_state"):
        kalman_hedge_ratio(
            rng.standard_normal(100),
            rng.standard_normal(100),
            process_var=1e-4,
            obs_var=1.0,
            initial_state=np.zeros(3),
        )
