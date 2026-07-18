"""quantica — an interactive tour of the three pillars (Streamlit + Plotly).

Run with::

    pip install -e ".[app]"
    streamlit run apps/quantica_app.py

This module is **presentation only**: widgets, caching, and Plotly rendering. Every
number it shows comes from the ``quantica`` library via the Streamlit-free compute
modules in this package (CLAUDE.md §2 — zero quant logic in ``apps/``). Heavy compute
(the convergence table, the Heston surface, the covariance race, the rolling VaR
backtest) is wrapped in ``st.cache_data`` so slider moves stay responsive.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable so `from apps import ...` resolves under `streamlit run`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from apps import _capital, _derivatives, _risk

warnings.filterwarnings("ignore")  # quieten arch's rolling-GARCH convergence chatter

st.set_page_config(page_title="quantica", page_icon="📈", layout="wide")

_ZONE_COLOR = {"GREEN": "#1a9850", "AMBER": "#f0a202", "RED": "#d73027"}


# --------------------------------------------------------------------------- #
# Cached compute wrappers (the expensive artifacts)
# --------------------------------------------------------------------------- #


@st.cache_data
def _convergence_table():  # type: ignore[no-untyped-def]
    return _derivatives.convergence_table()


@st.cache_data
def _heston_surface(spot, rate, div, v0, kappa, theta, xi, rho):  # type: ignore[no-untyped-def]
    moneyness = np.linspace(0.8, 1.2, 17)
    maturities = np.array([0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    return _derivatives.heston_implied_vol_surface(
        spot, rate, div, v0, kappa, theta, xi, rho, moneyness, maturities
    )


@st.cache_data
def _smiles(spot, rate, div, v0, kappa, theta, xi, rho, expiry, mvol, lam, mu_j, sigma_j):  # type: ignore[no-untyped-def]
    m = np.linspace(0.75, 1.25, 21)
    heston = _derivatives.heston_vs_bs_smile(spot, rate, div, v0, kappa, theta, xi, rho, expiry, m)
    merton = _derivatives.merton_smile(spot, rate, div, mvol, lam, mu_j, sigma_j, expiry, m)
    return heston, merton


@st.cache_data
def _gamma(book, daily_vol, level):  # type: ignore[no-untyped-def]
    return _risk.gamma_divergence(book, daily_vol=daily_vol, level=level)


@st.cache_data
def _frtb(book, method, daily_vol):  # type: ignore[no-untyped-def]
    return _risk.frtb_verdict(book, method, daily_vol=daily_vol)


@st.cache_data
def _var_backtest(level, window):  # type: ignore[no-untyped-def]
    return _risk.var_engine_backtest(level=level, window=window)


@st.cache_data
def _cov_comparison(train_window, test_window):  # type: ignore[no-untyped-def]
    return _capital.covariance_comparison(train_window=train_window, test_window=test_window)


@st.cache_data
def _jm():  # type: ignore[no-untyped-def]
    return _capital.jagannathan_ma()


@st.cache_data
def _overfit(planted_sharpe):  # type: ignore[no-untyped-def]
    return _capital.overfit_search(planted_sharpe=planted_sharpe)


# --------------------------------------------------------------------------- #
# Derivatives tab
# --------------------------------------------------------------------------- #


def _derivatives_tab() -> None:
    st.header("Derivatives pricing")
    st.caption(
        "European options priced by the analytic Black–Scholes engine, with the "
        "four-way cross-method convergence table and stochastic-vol / jump smiles — "
        "every value from `quantica.pricing`."
    )

    with st.sidebar:
        st.subheader("Contract & market")
        spot = st.slider("Spot", 50.0, 150.0, 100.0, 1.0)
        strike = st.slider("Strike", 50.0, 150.0, 100.0, 1.0)
        vol = st.slider("Volatility σ", 0.05, 0.60, 0.20, 0.01)
        rate = st.slider("Rate r", 0.0, 0.10, 0.05, 0.005)
        div = st.slider("Dividend q", 0.0, 0.10, 0.0, 0.005)
        expiry = st.slider("Maturity T (yrs)", 0.05, 3.0, 1.0, 0.05)
        kind = st.radio("Type", ["call", "put"], horizontal=True)

    pg = _derivatives.price_and_greeks(spot, strike, rate, div, vol, expiry, kind)
    cols = st.columns(6)
    cols[0].metric("Price", f"{pg['price']:.4f}")
    for col, name in zip(cols[1:], ("delta", "gamma", "vega", "theta", "rho"), strict=True):
        col.metric(name.capitalize(), f"{pg[name]:.4f}")

    st.subheader("Greek profiles vs spot")
    greek = st.selectbox("Greek", ["price", "delta", "gamma", "vega", "theta", "rho"], index=1)
    grid = np.linspace(max(spot * 0.4, 1.0), spot * 1.6, 121)
    profile = _derivatives.greek_profiles(strike, rate, div, vol, expiry, kind, grid)
    fig = go.Figure(go.Scatter(x=profile["spot"], y=profile[greek], mode="lines", name=greek))
    fig.add_vline(x=strike, line_dash="dot", line_color="gray", annotation_text="strike")
    fig.update_layout(xaxis_title="Spot", yaxis_title=greek, height=360, margin={"t": 20})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Four engines, one price")
    st.caption(
        "The same option priced by the analytic formula, a CRR tree, a Crank–Nicolson "
        "PDE, and Monte Carlo — sharing no code path, converging to the same price."
    )
    table = _convergence_table()
    left, right = st.columns([3, 2])
    left.dataframe(
        table.style.format({"price": "{:.6f}", "abs_error": "{:.2e}"}),
        width="stretch",
        hide_index=True,
    )
    numeric = table[table["abs_error"] > 0]
    err_fig = go.Figure(go.Bar(x=numeric["method"], y=numeric["abs_error"]))
    err_fig.update_layout(
        yaxis_type="log",
        yaxis_title="abs error vs analytic",
        height=360,
        margin={"t": 20},
        xaxis_tickangle=-40,
    )
    right.plotly_chart(err_fig, width="stretch")

    st.subheader("Implied-vol surface (Heston) and model smiles")
    with st.sidebar:
        st.subheader("Heston / Merton")
        v0 = st.slider("Heston v0 (variance)", 0.01, 0.16, 0.04, 0.01)
        kappa = st.slider("Heston κ (mean reversion)", 0.1, 5.0, 1.5, 0.1)
        theta = st.slider("Heston θ (long-run var)", 0.01, 0.16, 0.04, 0.01)
        xi = st.slider("Heston ξ (vol of vol)", 0.05, 1.0, 0.5, 0.05)
        heston_rho = st.slider("Heston ρ (spot/var corr)", -0.95, 0.0, -0.7, 0.05)
        smile_expiry = st.slider("Smile maturity (yrs)", 0.1, 2.0, 0.5, 0.1)
        mvol = st.slider("Merton σ (diffusion)", 0.05, 0.4, 0.15, 0.01)
        lam = st.slider("Merton λ (jumps/yr)", 0.0, 3.0, 1.0, 0.1)
        mu_j = st.slider("Merton μ_J (mean jump)", -0.4, 0.2, -0.1, 0.02)
        sigma_j = st.slider("Merton σ_J (jump vol)", 0.01, 0.4, 0.15, 0.01)

    surface = _heston_surface(spot, rate, div, v0, kappa, theta, xi, heston_rho)
    surf_fig = go.Figure(
        go.Surface(
            x=surface["strikes"],
            y=surface["maturities"],
            z=surface["iv"],
            colorscale="Viridis",
            colorbar={"title": "IV"},
        )
    )
    surf_fig.update_layout(
        height=480,
        margin={"t": 20, "l": 0, "r": 0, "b": 0},
        scene={"xaxis_title": "strike", "yaxis_title": "maturity", "zaxis_title": "implied vol"},
    )
    st.plotly_chart(surf_fig, width="stretch")

    heston_smile, merton_smile = _smiles(
        spot, rate, div, v0, kappa, theta, xi, heston_rho, smile_expiry, mvol, lam, mu_j, sigma_j
    )
    c1, c2 = st.columns(2)
    hf = go.Figure()
    hf.add_scatter(
        x=heston_smile["moneyness"],
        y=heston_smile["heston_iv"],
        mode="lines+markers",
        name="Heston",
    )
    hf.add_scatter(
        x=heston_smile["moneyness"],
        y=heston_smile["bs_iv"],
        mode="lines",
        name="flat BS",
        line_dash="dash",
    )
    hf.update_layout(
        title="Heston vs Black–Scholes smile",
        xaxis_title="moneyness K/S",
        yaxis_title="implied vol",
        height=360,
        margin={"t": 40},
    )
    c1.plotly_chart(hf, width="stretch")

    mf = go.Figure()
    mf.add_scatter(
        x=merton_smile["moneyness"],
        y=merton_smile["merton_iv"],
        mode="lines+markers",
        name="Merton",
    )
    mf.add_scatter(
        x=merton_smile["moneyness"],
        y=merton_smile["bs_iv"],
        mode="lines",
        name="flat BS",
        line_dash="dash",
    )
    mf.update_layout(
        title="Merton jump smile",
        xaxis_title="moneyness K/S",
        yaxis_title="implied vol",
        height=360,
        margin={"t": 40},
    )
    c2.plotly_chart(mf, width="stretch")


# --------------------------------------------------------------------------- #
# Risk tab
# --------------------------------------------------------------------------- #


def _zone_badge(label: str, zone: str) -> str:
    style = f"background:{_ZONE_COLOR[zone]};color:white;padding:2px 10px;border-radius:6px"
    return f"<span style='{style}'>{label}: {zone}</span>"


def _risk_tab() -> None:
    st.header("Risk & model validation")
    st.caption(
        "The headline: whether a risk model carries **gamma** decides a short-gamma "
        "desk's fate — the delta-normal-vs-full-revaluation divergence and its FRTB "
        "P&L-attribution verdict, all from `quantica.risk`."
    )

    book = st.selectbox("Option book", _risk.BOOK_NAMES, index=2)
    c = st.columns(2)
    daily_vol = c[0].slider("Daily move size (vol)", 0.005, 0.05, 0.0126, 0.001)
    level = c[1].slider("VaR confidence", 0.90, 0.99, 0.99, 0.01)

    g = _gamma(book, daily_vol, level)
    var = g["var"]
    m = st.columns(5)
    m[0].metric("Delta-normal VaR", f"{var['delta-normal']:,.2f}")
    m[1].metric("Delta-gamma VaR", f"{var['delta-gamma']:,.2f}")
    m[2].metric("Full-reval VaR", f"{var['full']:,.2f}")
    m[3].metric("Delta-normal error", f"{g['dn_error']:+.1%}")
    m[4].metric("Delta-gamma error", f"{g['dg_error']:+.1%}")

    hist = go.Figure()
    hist.add_histogram(x=g["pnl_full"], name="full revaluation", opacity=0.6, nbinsx=80)
    hist.add_histogram(x=g["pnl_delta_normal"], name="delta-normal", opacity=0.6, nbinsx=80)
    hist.update_layout(
        barmode="overlay",
        title="Scenario P&L: full revaluation vs delta-normal",
        xaxis_title="P&L",
        yaxis_title="count",
        height=380,
        margin={"t": 40},
    )
    st.plotly_chart(hist, width="stretch")
    st.caption(
        "For a short-gamma book the omitted ½·Γ·δS² term is a pure loss, so the "
        "delta-normal P&L misses the left tail and *under*-states VaR — dangerous."
    )

    st.subheader("FRTB P&L attribution — does the risk model keep its internal-model status?")
    d_only = _frtb(book, "delta-normal", daily_vol)
    d_gamma = _frtb(book, "delta-gamma", daily_vol)
    cc = st.columns(2)
    for col, title, res in (
        (cc[0], "Delta-only risk model", d_only),
        (cc[1], "Delta-gamma risk model", d_gamma),
    ):
        with col:
            st.markdown(f"**{title}**")
            st.markdown(
                _zone_badge("Spearman", res["spearman_zone"])
                + " "
                + _zone_badge("KS", res["ks_zone"])
                + " "
                + _zone_badge("Overall", res["zone"]),
                unsafe_allow_html=True,
            )
            st.write(f"Spearman ρ = {res['spearman']:.3f}, KS = {res['ks']:.3f}")
            st.write(res["consequence"])

    st.subheader("VaR/ES engines rolled out-of-sample (bundled FF portfolio)")
    bt_level = st.slider("Backtest confidence", 0.90, 0.99, 0.95, 0.01, key="bt_level")
    with st.spinner("Rolling the four engines (GARCH refit each step)…"):
        bt = _var_backtest(bt_level, 120)
    st.dataframe(
        bt.style.format({"kupiec_p": "{:.3f}", "as_z2": "{:+.3f}"}),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "The equal-weight 49-industry portfolio, monthly, 120-month rolling window — "
        "illustrative (monthly data is milder than the README's daily fat-tailed stress)."
    )


# --------------------------------------------------------------------------- #
# Capital-markets tab
# --------------------------------------------------------------------------- #


def _capital_tab() -> None:
    st.header("Capital markets — factor risk & portfolios")
    st.caption(
        "The out-of-sample covariance study, the Jagannathan–Ma no-short-sale result, "
        "and DSR/PBO overfit detection — from `quantica.factor` and `quantica.portfolio` "
        "on the bundled 49-industry sample."
    )

    st.subheader("Which covariance estimator to trust (out-of-sample)")
    cov = _cov_comparison(60, 12)
    left, right = st.columns([2, 3])
    left.dataframe(
        cov.style.format(
            {
                "random_bias": "{:.2f}",
                "calibrated": "{:.0%}",
                "min_var_vol": "{:.1%}",
                "min_var_bias": "{:.2f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    bar = go.Figure(
        go.Bar(
            x=cov["estimator"], y=cov["min_var_vol"], text=[f"{v:.1%}" for v in cov["min_var_vol"]]
        )
    )
    bar.update_layout(
        title="Min-variance realised OOS vol (lower = better)",
        yaxis_tickformat=".0%",
        height=340,
        margin={"t": 40},
    )
    right.plotly_chart(bar, width="stretch")
    st.caption(
        "On random portfolios the estimators barely differ; the sample covariance only "
        "fails when a portfolio *inverts* it (Michaud's error maximiser) — its GMV "
        "realises ~2× the volatility on a badly optimistic forecast (bias ≫ 1)."
    )

    st.subheader("No-short-sale is covariance shrinkage (Jagannathan–Ma 2003)")
    jm = _jm()
    st.dataframe(
        jm["table"].style.format({"unconstrained": "{:.1%}", "long_only": "{:.1%}"}),
        width="stretch",
        hide_index=True,
    )
    st.markdown(
        f"On the first window the unconstrained sample GMV shorts **{jm['n_shorted']} of "
        f"{jm['n_assets']}** industries. The long-only weights equal the *unconstrained* "
        f"GMV of the Jagannathan–Ma shrunk covariance to **{jm['recovery_error']:.1e}** — "
        f"the constraint **is** shrinkage. Condition number "
        f"{jm['cond_sample']:,.0f} → {jm['cond_shrunk']:,.0f}."
    )

    st.subheader("Overfitting, detected (Deflated Sharpe & PBO)")
    st.caption(
        "Pick the best of 100 candidate strategies. With no real signal the winner is "
        "spurious; plant one genuine signal and it survives. Drag the slider from 0."
    )
    planted = st.slider("Planted signal Sharpe (0 = pure noise)", 0.0, 0.5, 0.0, 0.05)
    res = _overfit(planted)
    k = st.columns(4)
    k[0].metric("Best Sharpe (ann.)", f"{res['best_sharpe_ann']:.2f}")
    k[1].metric(
        "Deflated Sharpe",
        f"{res['dsr']:.3f}",
        "significant" if res["dsr_significant"] else "not significant",
    )
    k[2].metric("PBO", f"{res['pbo']:.2f}")
    k[3].metric("Expected-max benchmark", f"{res['benchmark_sharpe_ann']:.2f}")

    sharpes = res["trial_sharpes"]
    hist = go.Figure(go.Histogram(x=sharpes, nbinsx=30, name="trial Sharpes"))
    hist.add_vline(
        x=float(sharpes[res["selected"]]),
        line_color="#d73027",
        annotation_text="selected (best in-sample)",
    )
    hist.update_layout(
        title="Annualised Sharpe across the strategy search",
        xaxis_title="Sharpe (ann.)",
        yaxis_title="count",
        height=360,
        margin={"t": 40},
    )
    st.plotly_chart(hist, width="stretch")
    if planted == 0.0:
        st.info(
            "All noise: the winner's Sharpe is what the luckiest of 100 coin-flips looks "
            "like — DSR at chance, PBO ≈ 0.5."
        )
    else:
        st.success(
            "A real edge survives deflation (DSR → 1) and the selection is repeatable out "
            "of sample (PBO → 0)."
        )


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #


_PILLARS = {
    "Derivatives pricing": _derivatives_tab,
    "Risk & model validation": _risk_tab,
    "Capital markets": _capital_tab,
}


def main() -> None:
    """Render the app: the title, the sidebar pillar selector, and the chosen pillar."""
    st.title("📈 quantica")
    st.markdown(
        "A **validation-first** quantitative-finance library — the deliverable is the "
        "*evidence* each model is correct. This is a thin UI over the tested core; every "
        "number is computed live by `quantica`."
    )
    # A sidebar selector rather than `st.tabs`: Streamlit executes every tab body on each
    # run, so tabs would recompute all three pillars on any interaction. Rendering only
    # the selected pillar keeps each interaction to a single pillar's (cached) compute.
    pillar = st.sidebar.radio("Pillar", list(_PILLARS), index=0)
    st.sidebar.divider()
    _PILLARS[pillar]()


main()
