#!/usr/bin/env python
"""Time series — GARCH-family forecasts and the forecast-evaluation layer that judges them.

Three artifacts (deterministic; the returns are the real daily S&P 500 series bundled with
:mod:`arch`, 1999--2018, so there is no network access):

1. **In-sample fit.** GARCH(1,1), GJR-GARCH and EGARCH on the full sample: the asymmetry/leverage
   term is **strongly significant in-sample** (the log-likelihood jumps, information criteria
   fall) — the standard reason practitioners reach for the asymmetric models.

2. **Out-of-sample verdict (the honest test).** Rolling one-step-ahead variance forecasts scored
   against the squared-return proxy by QLIKE and MSE, and compared with the **Diebold--Mariano**
   test (HAC-corrected). The honest question is whether the leverage term that helps *in-sample*
   actually improves *out-of-sample* forecasts — often it does not survive DM.

3. **Validate the validator (the headline).** On simulated equal-accuracy loss differentials with
   known serial correlation, the naive-variance DM test **over-rejects** while the HAC correction
   restores the correct size; both have power against a genuinely worse forecast. The evaluation
   tool is only trustworthy because of the HAC correction — that is the point of the pillar.

Regenerate with::

    python scripts/timeseries_garch_report.py

The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys
import warnings

import numpy as np
from quantica.timeseries import (
    diebold_mariano,
    fit_volatility_model,
    mse_loss,
    qlike_loss,
    rolling_forecast,
    simulate_loss_differential,
)

warnings.simplefilter("ignore")  # arch emits convergence/scale chatter on some windows

_FIRST_FORECAST_BACK = 750  # size of the out-of-sample tail
_REFIT = 25


def _returns() -> np.ndarray:  # type: ignore[type-arg]
    """Real daily S&P 500 percent log-returns from the bundled :mod:`arch` dataset."""
    import arch.data.sp500 as sp500

    price = sp500.load()["Adj Close"].to_numpy(dtype=np.float64)
    return 100.0 * np.diff(np.log(price))


def _insample_section(returns: np.ndarray) -> None:  # type: ignore[type-arg]
    """Full-sample fits: leverage is strongly significant in-sample."""
    print("### 1. In-sample fit — the leverage term is highly significant\n")
    print("| Model | log-lik | BIC | leverage (γ) |")
    print("| --- | ---: | ---: | ---: |")
    for model in ("GARCH", "GJR", "EGARCH"):
        fit = fit_volatility_model(returns, model)  # type: ignore[arg-type]
        gamma = fit.params.get("gamma[1]")
        gamma_str = "— (symmetric)" if gamma is None else f"{gamma:+.3f}"
        print(f"| {model} | {fit.loglikelihood:,.1f} | {fit.bic:,.1f} | {gamma_str} |")
    print(
        "\nBoth asymmetric models improve the log-likelihood by ~100+ points over plain GARCH and "
        "lower the BIC — the leverage effect (bad news raises volatility more than good news) is "
        "unmistakable *in-sample*. Whether it helps *out-of-sample* is a separate question.\n"
    )


def _oos_section(returns: np.ndarray) -> None:  # type: ignore[type-arg]
    """Rolling OOS forecasts, QLIKE/MSE, and the Diebold--Mariano verdict vs GARCH."""
    first = returns.size - _FIRST_FORECAST_BACK
    forecasts = {
        model: rolling_forecast(returns, model, first_forecast=first, refit=_REFIT)  # type: ignore[arg-type]
        for model in ("GARCH", "GJR", "EGARCH")
    }
    proxy = forecasts["GARCH"].realized_proxy
    losses = {m: qlike_loss(fc.variance_forecast, proxy) for m, fc in forecasts.items()}

    print(
        f"### 2. Out-of-sample verdict — last {_FIRST_FORECAST_BACK} days, refit every {_REFIT}\n"
    )
    print("| Model | QLIKE | MSE | DM vs GARCH (HAC) | p-value |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for model, fc in forecasts.items():
        qlike = losses[model].mean()
        mse = mse_loss(fc.variance_forecast, proxy).mean()
        if model == "GARCH":
            dm_cell, p_cell = "— (baseline)", "—"
        else:
            dm = diebold_mariano(losses[model], losses["GARCH"])
            dm_cell = f"{dm.statistic:+.2f} (L={dm.lags})"
            p_cell = f"{dm.p_value:.3f}"
        print(f"| {model} | {qlike:.4f} | {mse:.3f} | {dm_cell} | {p_cell} |")

    dm_gjr = diebold_mariano(losses["GJR"], losses["GARCH"])
    dm_naive = diebold_mariano(losses["GJR"], losses["GARCH"], hac=False)
    print(
        f"\n**The honest finding.** Despite the decisive in-sample win, the leverage models do "
        f"**not** significantly beat plain GARCH out-of-sample: the DM test of GJR vs GARCH gives "
        f"a p-value of {dm_gjr.p_value:.2f} (HAC), so equal accuracy cannot be rejected. Note the "
        f"HAC correction matters even here — the naive-variance statistic is "
        f"{dm_naive.statistic:+.2f} versus the HAC {dm_gjr.statistic:+.2f} — because the loss "
        "differentials are serially correlated. In-sample significance is not out-of-sample "
        "value, and the DM test is what tells them apart.\n"
    )


def _size_power_section() -> None:
    """Monte Carlo size and power of the DM test, with and without the HAC correction."""
    rng = np.random.default_rng(0)
    n, reps = 500, 2000

    def rates(mean: float, phi: float) -> tuple[float, float]:
        zero = np.zeros(n)
        naive = np.empty(reps)
        hac = np.empty(reps)
        for i in range(reps):
            d = simulate_loss_differential(n, mean=mean, phi=phi, rng=rng)
            naive[i] = diebold_mariano(d, zero, hac=False).p_value < 0.05
            hac[i] = diebold_mariano(d, zero, hac=True).p_value < 0.05
        return float(naive.mean()), float(hac.mean())

    print("### 3. Validate the validator — DM size and power (nominal 5%)\n")
    print(
        f"{reps:,} replications, T={n}. A positive φ is the realistic case for volatility losses.\n"
    )
    print("| Scenario | φ (serial corr.) | naive rej. rate | HAC rej. rate |")
    print("| --- | ---: | ---: | ---: |")
    for label, mean, phi in [
        ("Size (equal accuracy)", 0.0, 0.0),
        ("Size (equal accuracy)", 0.0, 0.5),
        ("Power (worse forecast)", 0.15, 0.5),
    ]:
        naive, hac = rates(mean, phi)
        print(f"| {label} | {phi:.1f} | {naive:.3f} | {hac:.3f} |")
    print(
        "\nWith no serial correlation (φ=0) both tests are correctly sized at ~5%. With positive "
        "serial correlation (φ=0.5) — the realistic case, since volatility forecast errors cluster "
        "— the **naive test over-rejects massively (~5×)** while the **HAC correction restores the "
        "nominal size**. Under a genuinely worse forecast both reject, but only the HAC test's "
        "rejections reflect real power rather than size distortion. This is why the HAC-corrected "
        "Diebold--Mariano test — not the textbook naive one — is the tool the pillar ships.\n"
    )


def main() -> None:
    """Print the in-sample, out-of-sample and size/power artifacts."""
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    returns = _returns()
    print(
        "## Time series — GARCH-family forecasts and the forecast-evaluation layer "
        f"(daily S&P 500, {returns.size:,} obs)\n"
    )
    _insample_section(returns)
    _oos_section(returns)
    _size_power_section()


if __name__ == "__main__":
    main()
