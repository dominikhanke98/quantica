"""Market-risk engines and the backtesting/validation layer (Phase 3).

The design intent (CLAUDE.md §1, effective-challenge identity) is that the
*backtesting* layer — not the VaR/ES numbers — is the deliverable: a risk model is
only as good as the evidence that it is adequately calibrated, and this package
ships both the standard VaR backtests and the harder-to-get Expected-Shortfall
backtest (Acerbi--Székely), with the tests validating the *backtests themselves*
(size and power) in ``tests/risk``.
"""

from __future__ import annotations

from quantica.risk.backtest import (
    AcerbiSzekelyResult,
    BaselResult,
    BaselZone,
    ChristoffersenResult,
    KupiecResult,
    acerbi_szekely,
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    exceptions,
    kupiec_pof,
    rolling_var_forecasts,
)
from quantica.risk.derivatives import (
    BookGreeks,
    BookPosition,
    MarketScenarios,
    OptionBook,
    book_var_es,
)
from quantica.risk.engines import (
    FilteredHistoricalSimulationVaR,
    HistoricalSimulationVaR,
    MonteCarloVaR,
    ParametricVaR,
    VaREngine,
)
from quantica.risk.measures import RiskEstimate, empirical_var_es, normal_var_es
from quantica.risk.portfolio import Portfolio

__all__ = [
    "AcerbiSzekelyResult",
    "BaselResult",
    "BaselZone",
    "BookGreeks",
    "BookPosition",
    "ChristoffersenResult",
    "FilteredHistoricalSimulationVaR",
    "HistoricalSimulationVaR",
    "KupiecResult",
    "MarketScenarios",
    "MonteCarloVaR",
    "OptionBook",
    "ParametricVaR",
    "Portfolio",
    "RiskEstimate",
    "VaREngine",
    "acerbi_szekely",
    "basel_traffic_light",
    "book_var_es",
    "christoffersen_cc",
    "christoffersen_independence",
    "empirical_var_es",
    "exceptions",
    "kupiec_pof",
    "normal_var_es",
    "rolling_var_forecasts",
]
