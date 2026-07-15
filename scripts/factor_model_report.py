#!/usr/bin/env python
"""Fetch Fama--French--Carhart factors + an asset universe and fit the risk model.

Pulls, via ``scripts/_ff_data.py``, the three-factor set (Mkt-RF, SMB, HML, RF),
the momentum factor (MOM), and the 10 industry portfolios that serve as a modest,
self-contained asset universe — all monthly, from Ken French's library, cached in
the OS temp directory (never committed) and **never run in CI** (the deterministic
tests use the synthetic generator instead).

It then fits a :class:`~quantica.factor.FactorRiskModel`, prints the estimated
factor exposures per industry, and shows an equal-weight portfolio's systematic /
specific risk decomposition. Regenerate with::

    python scripts/factor_model_report.py

Requires network access. The README embeds a captured run.
"""

from __future__ import annotations

import io
import sys

import numpy as np
from _ff_data import load_fama_french
from quantica.factor import FactorRiskModel

_N_MONTHS = 120  # trailing 10 years


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    data = load_fama_french(_N_MONTHS)
    model = FactorRiskModel.fit(
        data.industry_excess,
        data.factor_returns,
        asset_names=data.industry_names,
        factor_names=data.factor_names,
    )

    print("## Factor risk model — Fama--French--Carhart on 10 industry portfolios\n")
    print(
        f"Monthly data from Ken French's library, trailing {len(data.dates)} months "
        f"({data.dates[0]}–{data.dates[-1]}). Estimated factor exposures (betas), with "
        f"R² the systematic share of each industry's variance:\n"
    )
    header = "| Industry | " + " | ".join(data.factor_names) + " | R² | Specific vol (ann.) |"
    print(header)
    print("| --- | " + " | ".join("---:" for _ in data.factor_names) + " | ---: | ---: |")
    for name, exp in zip(data.industry_names, model.exposures, strict=True):
        betas = " | ".join(f"{b:+.2f}" for b in exp.betas)
        spec_vol_ann = np.sqrt(exp.specific_variance * 12.0)
        print(f"| {name} | {betas} | {exp.r_squared:.2f} | {spec_vol_ann:.1%} |")

    weights = np.full(len(data.industry_names), 1.0 / len(data.industry_names))
    dec = model.portfolio_risk_decomposition(weights)
    print(
        f"\nEqual-weight portfolio: annualised volatility "
        f"{dec.total_volatility * np.sqrt(12.0):.1%}, of which "
        f"**{dec.systematic_fraction:.0%} is systematic** (factor) risk and "
        f"{1 - dec.systematic_fraction:.0%} is specific — diversification across 10 "
        f"industries cancels much idiosyncratic risk, leaving a factor-dominated "
        f"portfolio, exactly as the decomposition should show. Net factor exposure "
        f"(Bᵀw): "
        + ", ".join(
            f"{n} {e:+.2f}" for n, e in zip(data.factor_names, dec.factor_exposure, strict=True)
        )
        + ".\n"
    )


if __name__ == "__main__":
    main()
