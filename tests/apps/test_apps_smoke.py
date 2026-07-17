"""Smoke tests for the apps' compute layer — CI catches a break without a UI.

The apps are a thin presentation layer, not the tested core, so these do not repeat
the numerical battery. They assert only that each compute module imports and its
data-prep functions return sane, plot-ready shapes — which is enough to catch an API
drift between ``quantica`` and the UI (a renamed argument, a changed return type). The
compute modules are deliberately Streamlit-free, so this runs under the plain ``dev``
install with no UI dependency.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from apps import _capital, _data, _derivatives, _risk

warnings.filterwarnings("ignore")  # arch's rolling-GARCH convergence chatter


# --------------------------------------------------------------------------- #
# Bundled data
# --------------------------------------------------------------------------- #


def test_ff_sample_loads_with_sane_shapes() -> None:
    sample = _data.load_ff_sample()
    assert sample.industry_excess.shape == (sample.n_months, sample.n_industries)
    assert sample.factor_returns.shape == (sample.n_months, 4)
    assert len(sample.industry_names) == sample.n_industries
    assert sample.n_months > 100 and sample.n_industries > 10
    assert np.all(np.isfinite(sample.industry_excess))
    assert sample.equal_weight_portfolio().shape == (sample.n_months,)


# --------------------------------------------------------------------------- #
# Derivatives compute
# --------------------------------------------------------------------------- #


def test_price_and_greeks_returns_all_fields() -> None:
    pg = _derivatives.price_and_greeks(100.0, 100.0, 0.05, 0.0, 0.2, 1.0, "call")
    assert set(pg) == {"price", "delta", "gamma", "vega", "theta", "rho"}
    assert pg["price"] > 0.0 and 0.0 < pg["delta"] < 1.0 and pg["gamma"] > 0.0


def test_greek_profiles_shape() -> None:
    grid = np.linspace(60.0, 140.0, 25)
    df = _derivatives.greek_profiles(100.0, 0.05, 0.0, 0.2, 1.0, "call", grid)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 25
    assert {"spot", "price", "delta", "gamma", "vega", "theta", "rho"} <= set(df.columns)
    assert df["delta"].is_monotonic_increasing  # call delta rises with spot


def test_convergence_table_converges() -> None:
    df = _derivatives.convergence_table()
    assert {"method", "price", "abs_error", "note"} <= set(df.columns)
    analytic = df.iloc[0]["price"]
    assert abs(analytic - 10.4506) < 1e-3  # the canonical ATM call
    # The finest CRR/PDE rows land close to the analytic reference.
    assert df["abs_error"].min() < 1e-2


def test_smiles_and_surface_shapes() -> None:
    moneyness = np.linspace(0.8, 1.2, 9)
    heston = _derivatives.heston_vs_bs_smile(
        100, 0.03, 0.0, 0.04, 1.5, 0.04, 0.5, -0.7, 0.5, moneyness
    )
    assert len(heston) == 9 and {"moneyness", "heston_iv", "bs_iv"} <= set(heston.columns)
    merton = _derivatives.merton_smile(100, 0.03, 0.0, 0.15, 1.0, -0.1, 0.15, 0.5, moneyness)
    assert len(merton) == 9 and np.isfinite(merton["merton_iv"]).any()
    surf = _derivatives.heston_implied_vol_surface(
        100, 0.03, 0.0, 0.04, 1.5, 0.04, 0.5, -0.7, moneyness, np.array([0.25, 0.5, 1.0])
    )
    assert surf["iv"].shape == (3, 9)


# --------------------------------------------------------------------------- #
# Risk compute
# --------------------------------------------------------------------------- #


def test_gamma_divergence_direction() -> None:
    g = _risk.gamma_divergence("Short ATM straddle (short gamma)", n_scenarios=4000)
    assert set(g["var"]) == {"delta-normal", "delta-gamma", "full"}  # type: ignore[arg-type]
    assert g["pnl_full"].shape == (4000,)  # type: ignore[union-attr]
    # Short gamma → delta-normal under-states VaR (negative error), delta-gamma repairs it.
    assert g["dn_error"] < 0.0  # type: ignore[operator]
    assert abs(g["dg_error"]) < abs(g["dn_error"])  # type: ignore[arg-type]


def test_frtb_verdict_gamma_decides_eligibility() -> None:
    delta_only = _risk.frtb_verdict("Short ATM straddle (short gamma)", "delta-normal")
    delta_gamma = _risk.frtb_verdict("Short ATM straddle (short gamma)", "delta-gamma")
    assert delta_only["zone"] == "RED" and delta_only["ima_eligible"] is False
    assert delta_gamma["ima_eligible"] is True


def test_var_engine_backtest_table() -> None:
    df = _risk.var_engine_backtest(level=0.95, window=180)
    assert len(df) == 4  # four engines
    assert {"engine", "exceptions", "kupiec_p", "basel_zone", "as_z2"} <= set(df.columns)
    assert (df["exceptions"] >= 0).all()


# --------------------------------------------------------------------------- #
# Capital-markets compute
# --------------------------------------------------------------------------- #


def test_covariance_comparison_orders_estimators() -> None:
    df = _capital.covariance_comparison()
    assert set(df["estimator"]) == {"sample", "ledoit-wolf", "factor"}
    sample_vol = float(df.loc[df["estimator"] == "sample", "min_var_vol"].iloc[0])
    lw_vol = float(df.loc[df["estimator"] == "ledoit-wolf", "min_var_vol"].iloc[0])
    assert sample_vol > lw_vol  # sample GMV is worst under inversion (the headline)


def test_jagannathan_ma_equivalence_is_exact() -> None:
    jm = _capital.jagannathan_ma()
    assert isinstance(jm["table"], pd.DataFrame)
    assert jm["recovery_error"] < 1e-8  # type: ignore[operator]
    assert 0 < jm["n_shorted"] <= jm["n_assets"]  # type: ignore[operator]
    # Long-only cuts the sample covariance's realised vol materially.
    table = jm["table"]
    sample_row = table[table["covariance"] == "sample"].iloc[0]  # type: ignore[index]
    assert sample_row["long_only"] < sample_row["unconstrained"]


def test_overfit_search_flags_noise_and_passes_signal() -> None:
    noise = _capital.overfit_search(planted_sharpe=0.0)
    signal = _capital.overfit_search(planted_sharpe=0.35)
    assert noise["dsr_significant"] is False  # noise winner is spurious
    assert signal["dsr_significant"] is True  # planted signal survives
    assert signal["pbo"] < noise["pbo"]  # and is more repeatable OOS
    assert signal["trial_sharpes"].shape == (100,)  # type: ignore[union-attr]
