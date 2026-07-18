"""Compute for the derivatives view — pure orchestration of the pricing engines.

Every number here comes from a ``quantica`` pricing engine; this module only builds
grids, calls the engines, and shapes the results into DataFrames/arrays for plotting.
No pricing mathematics lives here (CLAUDE.md §2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quantica.core.types import FloatArray
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BinomialEngine,
    BlackScholesProcess,
    EuropeanOption,
    FiniteDifferenceEngine,
    HestonFFTEngine,
    HestonProcess,
    Market,
    MertonClosedFormEngine,
    MertonProcess,
    MonteCarloEngine,
    OptionType,
    implied_volatility,
)

_ANALYTIC = AnalyticEuropeanEngine()


def _option_type(kind: str) -> OptionType:
    return OptionType.CALL if kind.lower() == "call" else OptionType.PUT


def price_and_greeks(
    spot: float,
    strike: float,
    rate: float,
    div: float,
    vol: float,
    expiry: float,
    kind: str,
) -> dict[str, float]:
    """Analytic Black--Scholes price and the five Greeks for one contract."""
    option = EuropeanOption(strike, expiry, _option_type(kind))
    process = BlackScholesProcess(spot=spot, rate=rate, div=div, vol=vol)
    greeks = _ANALYTIC.greeks(option, process)
    return {
        "price": _ANALYTIC.calculate(option, process),
        "delta": greeks.delta,
        "gamma": greeks.gamma,
        "vega": greeks.vega,
        "theta": greeks.theta,
        "rho": greeks.rho,
    }


def greek_profiles(
    strike: float,
    rate: float,
    div: float,
    vol: float,
    expiry: float,
    kind: str,
    spot_grid: FloatArray,
) -> pd.DataFrame:
    """Price and Greeks across a grid of spot prices (for profile plots)."""
    option = EuropeanOption(strike, expiry, _option_type(kind))
    rows = []
    for spot in spot_grid:
        process = BlackScholesProcess(spot=float(spot), rate=rate, div=div, vol=vol)
        g = _ANALYTIC.greeks(option, process)
        rows.append(
            {
                "spot": float(spot),
                "price": _ANALYTIC.calculate(option, process),
                "delta": g.delta,
                "gamma": g.gamma,
                "vega": g.vega,
                "theta": g.theta,
                "rho": g.rho,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Four-way convergence table (mirrors scripts/convergence_table.py exactly)
# --------------------------------------------------------------------------- #

_CONV_SPOT = 100.0
_CONV_RATE = 0.05
_CONV_DIV = 0.0
_CONV_VOL = 0.20
_CONV_STRIKE = 100.0
_CONV_EXPIRY = 1.0
_CONV_STEPS = (10, 50, 100, 500, 1000, 5000)
_CONV_PDE_GRIDS = (50, 100, 200, 400)
_CONV_MC_PATHS = 200_000
_CONV_MC_SEED = 20240709
_CONV_MC_CONFIGS = (
    ("Monte Carlo (naive)", False, False),
    ("Monte Carlo (antithetic)", True, False),
    ("Monte Carlo (control variate)", False, True),
)


def convergence_table() -> pd.DataFrame:
    """The four-way cross-method convergence table for the canonical ATM call.

    Same contract and construction as ``scripts/convergence_table.py`` (fewer Monte
    Carlo paths to stay interactive; the seed keeps it reproducible). Columns:
    ``method``, ``price``, ``abs_error`` (vs analytic), ``note``.
    """
    process = BlackScholesProcess(spot=_CONV_SPOT, rate=_CONV_RATE, div=_CONV_DIV, vol=_CONV_VOL)
    option = EuropeanOption(_CONV_STRIKE, _CONV_EXPIRY, OptionType.CALL)
    analytic = _ANALYTIC.calculate(option, process)

    rows: list[dict[str, object]] = [
        {
            "method": "Black–Scholes (analytic)",
            "price": analytic,
            "abs_error": 0.0,
            "note": "reference",
        }
    ]
    for n in _CONV_STEPS:
        price = BinomialEngine(steps=n).calculate(option, process)
        rows.append(
            {
                "method": f"Binomial CRR (N={n})",
                "price": price,
                "abs_error": abs(price - analytic),
                "note": "O(1/N)",
            }
        )
    for m in _CONV_PDE_GRIDS:
        price = FiniteDifferenceEngine(space_steps=m, time_steps=m).calculate(option, process)
        rows.append(
            {
                "method": f"Crank–Nicolson PDE ({m}×{m})",
                "price": price,
                "abs_error": abs(price - analytic),
                "note": "O(h²)",
            }
        )
    naive_se: float | None = None
    for label, antithetic, control in _CONV_MC_CONFIGS:
        engine = MonteCarloEngine(
            _CONV_MC_PATHS,
            rng=np.random.default_rng(_CONV_MC_SEED),
            antithetic=antithetic,
            control_variate=control,
        )
        result = engine.estimate(option, process)
        if naive_se is None:
            naive_se = result.std_error
        vrf = (naive_se / result.std_error) ** 2
        rows.append(
            {
                "method": label,
                "price": result.price,
                "abs_error": abs(result.price - analytic),
                "note": f"SE {result.std_error:.1e}, VRF {vrf:.1f}×",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Implied-vol surface and smiles (stochastic-vol / jump models vs flat BS)
# --------------------------------------------------------------------------- #


def _bs_implied_vol(
    price: float, spot: float, strike: float, rate: float, div: float, expiry: float, kind: str
) -> float:
    option = EuropeanOption(strike, expiry, _option_type(kind))
    market = Market(spot=spot, rate=rate, div=div)
    try:
        return implied_volatility(price, option, market)
    except ValueError:
        return float("nan")


def heston_implied_vol_surface(
    spot: float,
    rate: float,
    div: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    moneyness: FloatArray,
    maturities: FloatArray,
) -> dict[str, FloatArray]:
    """Black--Scholes implied-vol surface backed out of Heston prices.

    Prices the OTM option at each (strike, maturity) with the Heston FFT engine and
    inverts to a BS implied vol, producing the smile/skew surface. Returns
    ``strikes`` (from ``moneyness × spot``), ``maturities``, and ``iv`` (a
    ``len(maturities) × len(moneyness)`` grid).
    """
    process = HestonProcess(
        spot=spot, rate=rate, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, div=div
    )
    engine = HestonFFTEngine()
    strikes = np.asarray(moneyness, dtype=np.float64) * spot
    iv = np.full((len(maturities), len(strikes)), np.nan)
    for i, expiry in enumerate(maturities):
        for j, strike in enumerate(strikes):
            kind = "call" if strike >= spot else "put"
            option = EuropeanOption(float(strike), float(expiry), _option_type(kind))
            price = engine.calculate(option, process)
            iv[i, j] = _bs_implied_vol(price, spot, float(strike), rate, div, float(expiry), kind)
    return {"strikes": strikes, "maturities": np.asarray(maturities, dtype=np.float64), "iv": iv}


def heston_vs_bs_smile(
    spot: float,
    rate: float,
    div: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    expiry: float,
    moneyness: FloatArray,
) -> pd.DataFrame:
    """One-maturity Heston smile (implied vol vs strike) against the flat BS line."""
    surface = heston_implied_vol_surface(
        spot, rate, div, v0, kappa, theta, xi, rho, moneyness, np.array([expiry])
    )
    flat = float(np.sqrt(v0))
    return pd.DataFrame(
        {
            "moneyness": np.asarray(moneyness, dtype=np.float64),
            "strike": surface["strikes"],
            "heston_iv": surface["iv"][0],
            "bs_iv": np.full(len(moneyness), flat),
        }
    )


def merton_smile(
    spot: float,
    rate: float,
    div: float,
    vol: float,
    lam: float,
    mu_j: float,
    sigma_j: float,
    expiry: float,
    moneyness: FloatArray,
) -> pd.DataFrame:
    """Merton jump-diffusion smile (implied vol vs strike) against the flat BS line."""
    process = MertonProcess(
        spot=spot, rate=rate, vol=vol, lam=lam, mu_j=mu_j, sigma_j=sigma_j, div=div
    )
    engine = MertonClosedFormEngine()
    strikes = np.asarray(moneyness, dtype=np.float64) * spot
    ivs = np.full(len(strikes), np.nan)
    for j, strike in enumerate(strikes):
        kind = "call" if strike >= spot else "put"
        option = EuropeanOption(float(strike), expiry, _option_type(kind))
        price = engine.calculate(option, process)
        ivs[j] = _bs_implied_vol(price, spot, float(strike), rate, div, expiry, kind)
    return pd.DataFrame(
        {
            "moneyness": np.asarray(moneyness, dtype=np.float64),
            "strike": strikes,
            "merton_iv": ivs,
            "bs_iv": np.full(len(moneyness), vol),
        }
    )
