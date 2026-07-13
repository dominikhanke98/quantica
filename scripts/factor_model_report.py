#!/usr/bin/env python
"""Fetch Fama--French--Carhart factors + an asset universe and fit the risk model.

Pulls, from Ken French's data library (the canonical public source), the
three-factor set (Mkt-RF, SMB, HML, RF), the momentum factor (MOM), and the 10
industry portfolios that serve as a modest, self-contained asset universe — all
monthly, all from one provider, so the pull is fully scriptable. The downloads are
cached in the OS temp directory (never committed: no data in the repo, CLAUDE.md
§3), and **CI never runs this** — the deterministic tests use the synthetic
generator instead.

It then fits a :class:`~quantica.factor.FactorRiskModel`, prints the estimated
factor exposures per industry, and shows an equal-weight portfolio's systematic /
specific risk decomposition. Regenerate with::

    python scripts/factor_model_report.py

Requires network access. The README embeds a captured run.
"""

from __future__ import annotations

import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path
from tempfile import gettempdir

import numpy as np
from quantica.factor import FactorRiskModel

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FILES = {
    "factors": "F-F_Research_Data_Factors_CSV.zip",
    "momentum": "F-F_Momentum_Factor_CSV.zip",
    "industries": "10_Industry_Portfolios_CSV.zip",
}
_CACHE = Path(gettempdir()) / "quantica_ff_cache"
_N_MONTHS = 120  # trailing 10 years
_MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,")


def _download(name: str) -> bytes:
    _CACHE.mkdir(exist_ok=True)
    local = _CACHE / _FILES[name]
    if not local.exists():
        with urllib.request.urlopen(_BASE + _FILES[name], timeout=30) as response:
            local.write_bytes(response.read())
    with zipfile.ZipFile(io.BytesIO(local.read_bytes())) as archive:
        return archive.read(archive.namelist()[0])


def _parse_monthly(raw: bytes) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse the monthly block of a Ken French CSV into (dates, values, headers).

    The column header is the comma-separated line immediately preceding the first
    monthly row (Ken French files begin with multi-line prose that also contains
    commas, so "first comma line" is not reliable — "last comma line before the
    data" is).
    """
    lines = raw.decode("latin-1").splitlines()
    header: list[str] = []
    last_comma_line = ""
    dates: list[int] = []
    rows: list[list[float]] = []
    started = False
    for line in lines:
        m = _MONTH_ROW.match(line)
        if m and 190000 <= int(m.group(1)) <= 209912:  # monthly YYYYMM only
            started = True
            if not header:
                header = [h.strip() for h in last_comma_line.split(",")[1:] if h.strip()]
            fields = [f.strip() for f in line.split(",")]
            values = [float(v) for v in fields[1 : len(header) + 1]]
            if len(values) == len(header):
                dates.append(int(m.group(1)))
                rows.append(values)
        elif started:
            # These files hold several monthly blocks with the same date format
            # (value-weighted returns, then equal-weighted, firm counts, sizes,
            # annual, ...). Take only the first contiguous block: stop at its end.
            break
        elif "," in line:
            last_comma_line = line
    return np.array(dates), np.array(rows, dtype=np.float64), header


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    fdates, fvals, fhead = _parse_monthly(_download("factors"))
    mdates, mvals, _ = _parse_monthly(_download("momentum"))
    idates, ivals, ihead = _parse_monthly(_download("industries"))

    # Align on the common trailing months.
    common = np.intersect1d(np.intersect1d(fdates, mdates), idates)[-_N_MONTHS:]
    fi = np.searchsorted(fdates, common)
    mi = np.searchsorted(mdates, common)
    ii = np.searchsorted(idates, common)

    mkt_rf, smb, hml, rf = (fvals[fi, fhead.index(c)] for c in ("Mkt-RF", "SMB", "HML", "RF"))
    mom = mvals[mi, 0]
    factor_returns = np.column_stack([mkt_rf, smb, hml, mom]) / 100.0
    factor_names = ("Mkt-RF", "SMB", "HML", "MOM")

    # Industry excess returns (percent -> decimal, minus the risk-free rate).
    industries = ivals[ii] / 100.0 - (rf / 100.0)[:, None]

    model = FactorRiskModel.fit(
        industries, factor_returns, asset_names=tuple(ihead), factor_names=factor_names
    )

    print("## Factor risk model — Fama--French--Carhart on 10 industry portfolios\n")
    print(
        f"Monthly data from Ken French's library, trailing {len(common)} months "
        f"({common[0]}–{common[-1]}). Estimated factor exposures (betas), with R² "
        f"the systematic share of each industry's variance:\n"
    )
    header = "| Industry | " + " | ".join(factor_names) + " | R² | Specific vol (ann.) |"
    print(header)
    print("| --- | " + " | ".join("---:" for _ in factor_names) + " | ---: | ---: |")
    for name, exp in zip(ihead, model.exposures, strict=True):
        betas = " | ".join(f"{b:+.2f}" for b in exp.betas)
        spec_vol_ann = np.sqrt(exp.specific_variance * 12.0)
        print(f"| {name} | {betas} | {exp.r_squared:.2f} | {spec_vol_ann:.1%} |")

    weights = np.full(len(ihead), 1.0 / len(ihead))
    dec = model.portfolio_risk_decomposition(weights)
    print(
        f"\nEqual-weight portfolio: annualised volatility "
        f"{dec.total_volatility * np.sqrt(12.0):.1%}, of which "
        f"**{dec.systematic_fraction:.0%} is systematic** (factor) risk and "
        f"{1 - dec.systematic_fraction:.0%} is specific — diversification across 10 "
        f"industries cancels much idiosyncratic risk, leaving a factor-dominated "
        f"portfolio, exactly as the decomposition should show. Net factor exposure "
        f"(Bᵀw): "
        + ", ".join(f"{n} {e:+.2f}" for n, e in zip(factor_names, dec.factor_exposure, strict=True))
        + ".\n"
    )


if __name__ == "__main__":
    main()
