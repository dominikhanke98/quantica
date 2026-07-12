#!/usr/bin/env python
"""Generate the market-risk backtesting report for the README.

Two reproducible artifacts, printed as GitHub-flavoured Markdown:

1. **Validating the backtests themselves** (the headline) — a Monte-Carlo study of
   the *size* (rejection rate under a correct model, target ~5%) and *power*
   (rejection rate under a mis-specified model) of the Kupiec, Christoffersen and
   Acerbi--Székely tests. A backtest you have not size/power-checked is not
   evidence.
2. **A worked backtest** — a fat-tailed, volatility-clustered market (a GARCH(1,1)
   process with Student-t shocks) is risk-managed with all four engines rolled
   out-of-sample; the table shows the parametric-normal model is *optimistic*
   (too many exceptions, worse Basel zone) while the nonparametric / filtered
   methods hold up, and the Acerbi--Székely ES statistic flags the tail.

Everything is seeded, so the report reproduces byte-for-byte. Regenerate with::

    python scripts/risk_backtest_report.py
"""

from __future__ import annotations

import io
import sys

import numpy as np
from quantica.risk import (
    FilteredHistoricalSimulationVaR,
    HistoricalSimulationVaR,
    MonteCarloVaR,
    ParametricVaR,
    Portfolio,
    acerbi_szekely,
    basel_traffic_light,
    christoffersen_independence,
    exceptions,
    kupiec_pof,
    rolling_var_forecasts,
)
from scipy.stats import norm
from scipy.stats import t as student


def _garch_t_returns(n: int, rng: np.random.Generator, *, df: int = 5) -> np.ndarray:
    """Simulate a GARCH(1,1) return series with standardised Student-t shocks."""
    omega, alpha, beta = 1e-6, 0.08, 0.90
    scale = np.sqrt((df - 2) / df)  # standardise the t to unit variance
    var_t = omega / (1 - alpha - beta)
    out = np.empty(n)
    for i in range(n):
        z = student.rvs(df, random_state=rng) * scale
        r = np.sqrt(var_t) * z
        out[i] = r
        var_t = omega + alpha * r * r + beta * var_t
    return out


def size_power_table() -> None:
    rng = np.random.default_rng(2024)
    level, tail, size = 0.99, 0.01, 0.05
    n_trials, T = 2000, 750
    z = norm.ppf(level)

    # Kupiec / Christoffersen — size (correct model) and power (vol 1.5x).
    k_size = c_size = k_pow = c_pow = 0
    for _ in range(n_trials):
        good = (rng.random(T) < tail).astype(float)
        k_size += kupiec_pof(int(good.sum()), T, level).reject(size)
        c_size += christoffersen_independence(good).reject(size)
        bad = (rng.normal(0, 1.5, T) > z).astype(float)
        k_pow += kupiec_pof(int(bad.sum()), T, level).reject(size)
    # Christoffersen power against clustering (Markov exceptions).
    for _ in range(n_trials):
        hits = np.zeros(T)
        state = 0
        for t in range(T):
            state = 1 if rng.random() < (0.30 if state else 0.0072) else 0
            hits[t] = state
        c_pow += christoffersen_independence(hits).reject(size)

    # Acerbi--Székely Z2 at 97.5% — size and power (vol 1.4x), vectorised.
    es_level, es_tail = 0.975, 0.025
    var975, es975 = norm.ppf(es_level), norm.pdf(norm.ppf(es_level)) / es_tail

    def z2(losses: np.ndarray) -> np.ndarray:
        hits = losses > var975
        return (losses * hits).sum(-1) / es975 / (T * es_tail) - 1.0

    null = z2(rng.normal(0, 1, (2000, T)))
    good_p = np.mean(null[None] >= z2(rng.normal(0, 1, (n_trials, T)))[:, None], axis=1)
    bad_p = np.mean(null[None] >= z2(rng.normal(0, 1.4, (n_trials, T)))[:, None], axis=1)
    a_size = float(np.mean(good_p < size))
    a_pow = float(np.mean(bad_p < size))

    print("### 1. Validating the backtests themselves (size and power)\n")
    print(
        f"Monte-Carlo rejection rates at the 5% level over {n_trials} trials, "
        f"T={T} days (VaR tests at 99%, ES test at 97.5%):\n"
    )
    print("| Backtest | Size (target ≈ 5%) | Power (mis-specified) |")
    print("| --- | ---: | ---: |")
    print(f"| Kupiec POF (coverage) | {k_size / n_trials:.1%} | {k_pow / n_trials:.1%} |")
    print(f"| Christoffersen independence | {c_size / n_trials:.1%} | {c_pow / n_trials:.1%} |")
    print(f"| Acerbi–Székely ES (Z2) | {a_size:.1%} | {a_pow:.1%} |")
    print(
        "\nKupiec and Acerbi–Székely are correctly sized; the Christoffersen "
        "independence test is *conservative* at 99% (rare exceptions → few "
        "transitions) — an honest limitation — while retaining power against "
        "clustering. Power is strong throughout.\n"
    )


def worked_backtest() -> None:
    rng = np.random.default_rng(7)
    returns = _garch_t_returns(2000, rng, df=5)[:, None]
    pf = Portfolio(weights=np.array([1.0]), value=1_000_000.0)
    level, window = 0.99, 500

    engines = {
        "Historical simulation": HistoricalSimulationVaR(),
        "Parametric (normal)": ParametricVaR(),
        "Filtered HS (GARCH)": FilteredHistoricalSimulationVaR(),
        "Monte Carlo (normal)": MonteCarloVaR(50_000, rng=np.random.default_rng(1)),
    }

    print("### 2. A worked backtest — fat-tailed, volatility-clustered market\n")
    print(
        "A GARCH(1,1) market with Student-t(5) shocks, risk-managed at 99% VaR with "
        "a 500-day rolling window; each engine re-fit out-of-sample each day.\n"
    )
    print("| Engine | Exceptions | Expected | Kupiec p | Basel zone | AS Z2 (ES) |")
    print("| --- | ---: | ---: | ---: | --- | ---: |")
    for name, engine in engines.items():
        var_f, es_f, losses = rolling_var_forecasts(engine, returns, pf, level=level, window=window)
        hits = exceptions(losses, var_f)
        x, n = int(hits.sum()), hits.size
        kp = kupiec_pof(x, n, level)
        basel = basel_traffic_light(x, n_obs=n, level=level)
        az = acerbi_szekely(losses, var_f, es_f, level, method="Z2")
        print(
            f"| {name} | {x} | {n * (1 - level):.0f} | {kp.p_value:.3f} | "
            f"{basel.zone} | {az.statistic:+.3f} |"
        )
    print(
        "\nThe parametric-normal model takes the most exceptions and the worst "
        "Basel zone — its Gaussian tail understates the fat-tailed, clustered risk — "
        "while filtered historical simulation, which tracks the conditional "
        "volatility, fares best. A positive Acerbi–Székely Z2 flags ES "
        "under-estimation that the exception count alone can miss.\n"
    )


def main() -> None:
    import warnings

    # arch emits convergence/scaling chatter over the many rolling GARCH fits; the
    # results are deterministic, so quieten it for a clean, embeddable report.
    warnings.filterwarnings("ignore")
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    print("## Market-risk backtesting report\n")
    size_power_table()
    worked_backtest()


if __name__ == "__main__":
    main()
