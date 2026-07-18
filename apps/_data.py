"""Bundled sample data for the apps — loaded from disk, never fetched at runtime.

The capital-markets and risk views need a real return panel. To keep the apps fast
and fully offline (CLAUDE.md §3 — no network at runtime), a small Fama--French
industry sample is committed at ``apps/data/ff_sample.npz`` and loaded here. Rebuild
it with ``python apps/data/_build_sample.py`` (that builder is the only thing that
touches the network, and it is never run by the app).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from quantica.core.types import FloatArray

_SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "ff_sample.npz"


@dataclass(frozen=True)
class FamaFrenchSample:
    """A bundled monthly Fama--French industry panel (decimal excess returns)."""

    dates: FloatArray
    industry_excess: FloatArray  # (T, n_industries)
    factor_returns: FloatArray  # (T, 4) — Mkt-RF, SMB, HML, MOM
    industry_names: tuple[str, ...]
    factor_names: tuple[str, ...]

    @property
    def n_months(self) -> int:
        return int(self.industry_excess.shape[0])

    @property
    def n_industries(self) -> int:
        return int(self.industry_excess.shape[1])

    @property
    def date_range(self) -> tuple[int, int]:
        return int(self.dates[0]), int(self.dates[-1])

    def equal_weight_portfolio(self) -> FloatArray:
        """The equal-weight industry portfolio's monthly excess-return series."""
        return np.asarray(self.industry_excess.mean(axis=1), dtype=np.float64)


@lru_cache(maxsize=1)
def load_ff_sample() -> FamaFrenchSample:
    """Load the committed Fama--French sample (cached for the process lifetime)."""
    if not _SAMPLE_PATH.exists():  # pragma: no cover - guards a broken checkout
        raise FileNotFoundError(
            f"bundled sample missing at {_SAMPLE_PATH}; rebuild with "
            "`python apps/data/_build_sample.py`"
        )
    with np.load(_SAMPLE_PATH, allow_pickle=False) as npz:
        return FamaFrenchSample(
            dates=npz["dates"].astype(np.float64),
            industry_excess=npz["industry_excess"].astype(np.float64),
            factor_returns=npz["factor_returns"].astype(np.float64),
            industry_names=tuple(str(s) for s in npz["industry_names"]),
            factor_names=tuple(str(s) for s in npz["factor_names"]),
        )
