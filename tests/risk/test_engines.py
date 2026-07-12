"""VaR/ES engines — analytic anchors and cross-method agreement."""

from __future__ import annotations

import numpy as np
import pytest
from quantica.risk import (
    FilteredHistoricalSimulationVaR,
    HistoricalSimulationVaR,
    MonteCarloVaR,
    ParametricVaR,
    Portfolio,
    normal_var_es,
)


def test_portfolio_pnl_and_losses() -> None:
    pf = Portfolio(weights=np.array([0.6, 0.4]), value=1_000_000.0)
    returns = np.array([[0.01, -0.02], [0.0, 0.0]])
    pnl = pf.pnl(returns)
    assert pnl[0] == pytest.approx(1_000_000.0 * (0.6 * 0.01 + 0.4 * -0.02))
    assert pf.losses(returns)[0] == pytest.approx(-pnl[0])


def test_portfolio_validation() -> None:
    with pytest.raises(ValueError, match="non-empty 1-D"):
        Portfolio(weights=np.array([[1.0]]))
    with pytest.raises(ValueError, match="value must be positive"):
        Portfolio(weights=np.array([1.0]), value=0.0)
    with pytest.raises(ValueError, match="shape"):
        Portfolio(weights=np.array([1.0, 0.0])).pnl(np.zeros((10, 3)))


def test_parametric_equals_closed_form_on_sample_moments() -> None:
    # The variance--covariance engine IS the Gaussian closed form evaluated at the
    # sample mean vector and covariance matrix — assert it to machine precision.
    rng = np.random.default_rng(0)
    R = rng.normal(0.001, 0.02, (1000, 3))
    w = np.array([0.5, 0.3, 0.2])
    pf = Portfolio(weights=w, value=2_000_000.0)
    est = ParametricVaR().estimate(R, pf, level=0.99)

    mean_p = pf.value * float(w @ R.mean(axis=0))
    sigma_p = pf.value * float(np.sqrt(w @ np.cov(R, rowvar=False) @ w))
    ref = normal_var_es(mean_p, sigma_p, 0.99)
    assert est.var == pytest.approx(ref.var)
    assert est.es == pytest.approx(ref.es)


def test_monte_carlo_converges_to_parametric() -> None:
    # With a normal fit, MC VaR/ES converge to the parametric closed form — the
    # deliberate cross-check between the two engines.
    rng = np.random.default_rng(1)
    R = rng.normal(0.0005, 0.015, (750, 2))
    pf = Portfolio(weights=np.array([0.7, 0.3]), value=1_000_000.0)
    para = ParametricVaR().estimate(R, pf, level=0.99)
    mc = MonteCarloVaR(400_000, rng=np.random.default_rng(2)).estimate(R, pf, level=0.99)
    assert mc.var == pytest.approx(para.var, rel=0.02)
    assert mc.es == pytest.approx(para.es, rel=0.03)


def test_historical_close_to_parametric_on_large_normal_sample() -> None:
    rng = np.random.default_rng(3)
    R = rng.normal(0.0, 0.01, (20_000, 1))
    pf = Portfolio(weights=np.array([1.0]), value=1_000_000.0)
    hs = HistoricalSimulationVaR().estimate(R, pf, level=0.99)
    para = ParametricVaR().estimate(R, pf, level=0.99)
    assert hs.var == pytest.approx(para.var, rel=0.06)
    assert hs.es == pytest.approx(para.es, rel=0.08)


def test_monte_carlo_rejects_too_few_sims() -> None:
    with pytest.raises(ValueError, match="n_sims must be at least 2"):
        MonteCarloVaR(1, rng=np.random.default_rng(0))


@pytest.mark.filterwarnings("ignore")
def test_fhs_reacts_to_a_recent_volatility_spike() -> None:
    # Filtered HS should report a larger VaR when the *recent* conditional volatility
    # is high (GARCH picks up clustering) than when the tail end is calm, even with
    # the same unconditional variance over the window.
    rng = np.random.default_rng(4)
    pf = Portfolio(weights=np.array([1.0]), value=1_000_000.0)
    n = 1000
    calm_tail = np.concatenate([rng.normal(0, 0.03, n - 100), rng.normal(0, 0.005, 100)])
    vol_tail = np.concatenate([rng.normal(0, 0.005, n - 100), rng.normal(0, 0.03, 100)])
    engine = FilteredHistoricalSimulationVaR()
    calm = engine.estimate(calm_tail[:, None], pf, level=0.99)
    spiked = engine.estimate(vol_tail[:, None], pf, level=0.99)
    assert spiked.var > calm.var
    assert calm.var > 0.0 and spiked.es >= spiked.var
