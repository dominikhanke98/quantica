r"""The position/portfolio interface that feeds the risk engines.

A :class:`Portfolio` is a set of weights on assets plus a money value; given a
matrix of asset returns it produces the portfolio **P\&L** (and **loss**) series
that every VaR/ES engine and backtest consumes.

Extension point
---------------
Deliberately the engines take a *P\&L / loss series*, not the asset matrix
directly (except where the covariance is needed). That keeps the risk layer
agnostic to *where* the P\&L comes from: today it is a linear asset-returns
portfolio, but the same series could later be produced by revaluing a book of
options through the pricers in :mod:`quantica.pricing` (a full-revaluation P\&L
vector). The risk and backtesting machinery does not change when that P\&L source
is swapped in.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantica.core.types import FloatArray

__all__ = ["Portfolio"]


@dataclass(frozen=True)
class Portfolio:
    r"""A linear portfolio: weights on assets, scaled by a money value.

    Parameters
    ----------
    weights : ndarray
        Portfolio weights :math:`w` on the :math:`N` assets. Not required to sum to
        one (a net/gross exposure is a modelling choice), but must be finite.
    value : float, optional
        Portfolio money value :math:`V` (default 1.0). P\&L is
        :math:`V \, (R w)` for an asset-return matrix :math:`R`.
    """

    weights: FloatArray
    value: float = 1.0
    _weights: FloatArray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0:
            raise ValueError("weights must be a non-empty 1-D array")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
        if self.value <= 0.0:
            raise ValueError(f"value must be positive, got {self.value}")
        object.__setattr__(self, "_weights", w)

    @classmethod
    def equal_weight(cls, n_assets: int, value: float = 1.0) -> Portfolio:
        """An equally-weighted portfolio over ``n_assets`` assets."""
        if n_assets < 1:
            raise ValueError(f"n_assets must be at least 1, got {n_assets}")
        return cls(weights=np.full(n_assets, 1.0 / n_assets), value=value)

    @property
    def n_assets(self) -> int:
        """Number of assets ``N``."""
        return int(self._weights.size)

    def portfolio_returns(self, asset_returns: FloatArray) -> FloatArray:
        r"""The portfolio return series :math:`R w` from an asset-return matrix.

        Parameters
        ----------
        asset_returns : ndarray
            Shape ``(T, N)`` (or ``(T,)`` for a single asset) of simple returns.
        """
        R = np.asarray(asset_returns, dtype=np.float64)
        if R.ndim == 1:
            R = R[:, None]
        if R.ndim != 2 or R.shape[1] != self.n_assets:
            raise ValueError(f"asset_returns must have shape (T, {self.n_assets}), got {R.shape}")
        return R @ self._weights

    def pnl(self, asset_returns: FloatArray) -> FloatArray:
        r"""The portfolio P\&L series :math:`V\,(R w)` in money units."""
        return self.value * self.portfolio_returns(asset_returns)

    def losses(self, asset_returns: FloatArray) -> FloatArray:
        r"""The portfolio loss series :math:`-V\,(R w)` (positive = loss)."""
        return -self.pnl(asset_returns)
