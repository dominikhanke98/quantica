"""Shared Fama--French data loader for the factor-model report scripts.

Pulls, from Ken French's data library, the three-factor set (Mkt-RF, SMB, HML,
RF), the momentum factor (MOM), and the 10 industry portfolios that serve as a
modest, self-contained asset universe — all monthly, all from one provider.
Downloads are cached in the OS temp directory (never committed; CLAUDE.md §3),
and **this is never run in CI** — the deterministic tests use synthetic data.

The parsing handles two quirks of these files (documented gaps): the multi-line
prose preambles that contain commas (so the header is the last comma line *before*
the first data row, not the first comma line), and the several monthly blocks that
share the same ``YYYYMM`` date format (returns, then firm counts, then dollar
sizes) — only the first contiguous block is the returns.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import gettempdir

import numpy as np

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
_FACTORS_FILE = "F-F_Research_Data_Factors_CSV.zip"
_MOMENTUM_FILE = "F-F_Momentum_Factor_CSV.zip"
_INDUSTRY_FILES = {
    10: "10_Industry_Portfolios_CSV.zip",
    49: "49_Industry_Portfolios_CSV.zip",
}
_CACHE = Path(gettempdir()) / "quantica_ff_cache"
_MONTH_ROW = re.compile(r"^\s*(\d{6})\s*,")
_MISSING = -99.0  # Ken French missing-value sentinel is -99.99

FACTOR_NAMES = ("Mkt-RF", "SMB", "HML", "MOM")


@dataclass(frozen=True)
class FamaFrenchData:
    """Aligned monthly Fama--French--Carhart factors and industry excess returns."""

    dates: np.ndarray  # (T,) YYYYMM
    factor_returns: np.ndarray  # (T, 4) decimals, Mkt-RF/SMB/HML/MOM
    industry_excess: np.ndarray  # (T, 10) decimals, excess of the risk-free rate
    factor_names: tuple[str, ...]
    industry_names: tuple[str, ...]


def _download(filename: str) -> bytes:
    _CACHE.mkdir(exist_ok=True)
    local = _CACHE / filename
    if not local.exists():
        with urllib.request.urlopen(_BASE + filename, timeout=30) as response:
            local.write_bytes(response.read())
    with zipfile.ZipFile(io.BytesIO(local.read_bytes())) as archive:
        return archive.read(archive.namelist()[0])


def _parse_monthly(raw: bytes) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse the first monthly block of a Ken French CSV into (dates, values, headers)."""
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
            break  # first contiguous monthly block only (see module docstring)
        elif "," in line:
            last_comma_line = line
    return np.array(dates), np.array(rows, dtype=np.float64), header


def load_fama_french(n_months: int, n_industries: int = 10) -> FamaFrenchData:
    """Fetch, cache, align and return the trailing ``n_months`` of FF data (decimals).

    ``n_industries`` selects the 10- or 49-industry universe. Industries with any
    missing value (``-99.99``) in the window are dropped, so the returned universe
    may be slightly smaller than requested (the 49-industry set stresses the
    covariance estimators; the 10-industry set is the readable exposures table).
    """
    fdates, fvals, fhead = _parse_monthly(_download(_FACTORS_FILE))
    mdates, mvals, _ = _parse_monthly(_download(_MOMENTUM_FILE))
    idates, ivals, ihead = _parse_monthly(_download(_INDUSTRY_FILES[n_industries]))

    common = np.intersect1d(np.intersect1d(fdates, mdates), idates)[-n_months:]
    fi = np.searchsorted(fdates, common)
    mi = np.searchsorted(mdates, common)
    ii = np.searchsorted(idates, common)

    mkt_rf, smb, hml, rf = (fvals[fi, fhead.index(c)] for c in ("Mkt-RF", "SMB", "HML", "RF"))
    factor_returns = np.column_stack([mkt_rf, smb, hml, mvals[mi, 0]]) / 100.0

    industries = ivals[ii]
    keep = ~np.any(industries <= _MISSING, axis=0)  # drop industries with any missing month
    industry_excess = industries[:, keep] / 100.0 - (rf / 100.0)[:, None]
    industry_names = tuple(name for name, k in zip(ihead, keep, strict=True) if k)
    return FamaFrenchData(
        dates=common,
        factor_returns=factor_returns,
        industry_excess=industry_excess,
        factor_names=FACTOR_NAMES,
        industry_names=industry_names,
    )
