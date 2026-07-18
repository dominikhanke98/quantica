"""Compute for the risk view — VaR/ES engines, the gamma divergence, and FRTB PLA.

Pure orchestration of ``quantica.risk``: it builds option books and scenario sets,
calls the risk engines / backtests / PLA test, and returns plot-ready data. No risk
mathematics lives here (CLAUDE.md §2).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from quantica.core.types import FloatArray
from quantica.pricing import (
    AnalyticEuropeanEngine,
    BlackScholesProcess,
    EuropeanOption,
    OptionType,
)
from quantica.risk import (
    BookPosition,
    FilteredHistoricalSimulationVaR,
    HistoricalSimulationVaR,
    MarketScenarios,
    MonteCarloVaR,
    OptionBook,
    ParametricVaR,
    Portfolio,
    acerbi_szekely,
    basel_traffic_light,
    book_pla_test,
    book_var_es,
    christoffersen_independence,
    exceptions,
    kupiec_pof,
    rolling_var_forecasts,
)

from apps._data import load_ff_sample

_PROC = BlackScholesProcess(spot=100.0, rate=0.02, div=0.0, vol=0.2)
_ENGINE = AnalyticEuropeanEngine()
_CALL = EuropeanOption(100.0, 0.5, OptionType.CALL)
_PUT = EuropeanOption(100.0, 0.5, OptionType.PUT)

BOOK_NAMES = (
    "Deep-ITM call (near-linear)",
    "Long ATM straddle (long gamma)",
    "Short ATM straddle (short gamma)",
)
_PNL_METHODS = ("delta-normal", "delta-gamma", "full")


def _book(name: str) -> OptionBook:
    if name == "Deep-ITM call (near-linear)":
        itm = EuropeanOption(60.0, 0.5, OptionType.CALL)
        return OptionBook(positions=(BookPosition(itm, _ENGINE, 100.0),), process=_PROC)
    if name == "Long ATM straddle (long gamma)":
        return OptionBook(
            positions=(BookPosition(_CALL, _ENGINE, 100.0), BookPosition(_PUT, _ENGINE, 100.0)),
            process=_PROC,
        )
    if name == "Short ATM straddle (short gamma)":
        return OptionBook(
            positions=(BookPosition(_CALL, _ENGINE, -100.0), BookPosition(_PUT, _ENGINE, -100.0)),
            process=_PROC,
        )
    raise ValueError(f"unknown book {name!r}")


def gamma_divergence(
    book_name: str,
    *,
    daily_vol: float = 0.0126,
    level: float = 0.99,
    n_scenarios: int = 20_000,
    seed: int = 0,
) -> dict[str, object]:
    """Delta-normal / delta-gamma / full-revaluation VaR and P&L for one book.

    All three methods run on the *same* seeded scenario set, so any divergence is
    approximation error rather than sampling noise.

    Parameters
    ----------
    book_name : str
        One of :data:`BOOK_NAMES`.
    daily_vol : float, optional
        Daily spot-move volatility of the scenarios (default 0.0126, ~20% annualised).
    level : float, optional
        VaR confidence level in ``(0, 1)`` (default 0.99).
    n_scenarios : int, optional
        Number of seeded scenarios (default 20000).
    seed : int, optional
        Scenario-generator seed (default 0).

    Returns
    -------
    dict
        Keys: ``var`` (a dict of the delta-normal / delta-gamma / full VaR), ``dn_error``
        and ``dg_error`` (relative error of each approximation vs full revaluation), and
        ``pnl_full`` / ``pnl_delta_normal`` / ``pnl_delta_gamma`` (the scenario P&L arrays).
    """
    book = _book(book_name)
    scenarios = MarketScenarios.generate(
        n_scenarios, np.random.default_rng(seed), spot_vol=daily_vol
    )
    var = {m: book_var_es(book, scenarios, level, method=m).var for m in _PNL_METHODS}
    full = var["full"]
    return {
        "var": var,
        "dn_error": (var["delta-normal"] - full) / full,
        "dg_error": (var["delta-gamma"] - full) / full,
        "pnl_full": book.full_revaluation_pnl(scenarios),
        "pnl_delta_normal": book.delta_normal_pnl(scenarios),
        "pnl_delta_gamma": book.delta_gamma_pnl(scenarios),
    }


def frtb_verdict(
    book_name: str,
    rtpl_method: str,
    *,
    daily_vol: float = 0.0126,
    n_days: int = 250,
    seed: int = 7,
) -> dict[str, object]:
    """FRTB P&L-attribution verdict for a book under a delta-only or delta-gamma model.

    Parameters
    ----------
    book_name : str
        One of :data:`BOOK_NAMES`.
    rtpl_method : str
        The risk model whose RTPL is tested: ``"delta-normal"`` or ``"delta-gamma"``.
    daily_vol : float, optional
        Daily spot-move volatility of the scenarios (default 0.0126).
    n_days : int, optional
        Length of the PLA observation window in days (default 250).
    seed : int, optional
        Scenario-generator seed (default 7).

    Returns
    -------
    dict
        Keys: ``spearman`` and ``ks`` (the two PLA statistics), ``spearman_zone`` /
        ``ks_zone`` / ``zone`` (their green/amber/red zones and the overall worse-of),
        ``ima_eligible`` (bool), and ``consequence`` (the capital consequence string).
    """
    book = _book(book_name)
    scenarios = MarketScenarios.generate(n_days, np.random.default_rng(seed), spot_vol=daily_vol)
    result = book_pla_test(book, scenarios, rtpl_method=rtpl_method)  # type: ignore[arg-type]
    return {
        "spearman": result.spearman,
        "ks": result.ks_statistic,
        "spearman_zone": result.spearman_zone.name,
        "ks_zone": result.ks_zone.name,
        "zone": result.zone.name,
        "ima_eligible": result.ima_eligible,
        "consequence": result.capital_consequence(),
    }


def _ff_portfolio_returns() -> FloatArray:
    """The bundled equal-weight Fama--French industry portfolio, as a (T, 1) matrix."""
    return load_ff_sample().equal_weight_portfolio()[:, None]


def var_engine_backtest(
    *,
    level: float = 0.95,
    window: int = 120,
    mc_seed: int = 1,
) -> pd.DataFrame:
    """Roll the four VaR/ES engines out-of-sample over the bundled FF portfolio.

    Monthly data, so this is illustrative rather than the daily fat-tailed stress in
    the README's risk report.

    Parameters
    ----------
    level : float, optional
        VaR/ES confidence level in ``(0, 1)`` (default 0.95).
    window : int, optional
        Rolling estimation-window length in months (default 120).
    mc_seed : int, optional
        Seed for the Monte-Carlo VaR engine (default 1).

    Returns
    -------
    pandas.DataFrame
        One row per engine with columns ``engine``, ``exceptions``, ``expected``,
        ``kupiec_p``, ``basel_zone``, ``christoffersen`` (independent / clustered),
        and ``as_z2`` (the Acerbi--Székely Z2 ES statistic).
    """
    returns = _ff_portfolio_returns()
    portfolio = Portfolio(weights=np.array([1.0]), value=1_000_000.0)
    engines = {
        "Historical simulation": HistoricalSimulationVaR(),
        "Parametric (normal)": ParametricVaR(),
        "Filtered HS (GARCH)": FilteredHistoricalSimulationVaR(),
        "Monte Carlo (normal)": MonteCarloVaR(20_000, rng=np.random.default_rng(mc_seed)),
    }
    rows = []
    for name, engine in engines.items():
        var_f, es_f, losses = rolling_var_forecasts(
            engine, returns, portfolio, level=level, window=window
        )
        hits = exceptions(losses, var_f)
        x, n = int(hits.sum()), int(hits.size)
        kp = kupiec_pof(x, n, level)
        basel = basel_traffic_light(x, n_obs=n, level=level)
        az = acerbi_szekely(losses, var_f, es_f, level, method="Z2")
        christ = christoffersen_independence(hits)
        rows.append(
            {
                "engine": name,
                "exceptions": x,
                "expected": round(n * (1 - level), 1),
                "kupiec_p": kp.p_value,
                "basel_zone": str(basel.zone),
                "christoffersen": "clustered" if christ.reject() else "independent",
                "as_z2": az.statistic,
            }
        )
    return pd.DataFrame(rows)
