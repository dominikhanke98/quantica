r"""Synthetic strategy-trial data with a *known* truth — for the validity headline.

The backtest-validity layer is validated by known-truth construction, the same
discipline as the rest of the repo: generate a matrix of candidate-strategy return
series where the truth is planted, then confirm the detectors call it correctly.

* :func:`generate_trial_returns` — a ``(T, N)`` matrix of trial return series. By
  default *every* column is pure noise (no strategy has any edge), so the
  best-in-sample column is spurious: the deflated Sharpe ratio must flag it and PBO
  must be ≈ 0.5. Set ``planted_sharpe`` to give **one** column a genuine per-period
  Sharpe — it should then survive deflation and drive PBO toward 0.

Only synthetic data makes the known-truth check possible; CI never touches a network
fetch (the real-data run lives in ``scripts/portfolio_backtest_report.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantica.core.types import FloatArray

__all__ = [
    "TrialReturns",
    "generate_trial_returns",
]


@dataclass(frozen=True)
class TrialReturns:
    """A matrix of candidate-strategy returns with its planted ground truth.

    Attributes
    ----------
    returns : ndarray, shape (n_periods, n_trials)
        Per-period return series, one column per candidate strategy.
    planted_index : int or None
        Column index carrying the genuine signal, or ``None`` if all noise.
    planted_sharpe : float
        The per-period Sharpe planted into ``planted_index`` (0 if all noise).
    """

    returns: FloatArray
    planted_index: int | None
    planted_sharpe: float


def generate_trial_returns(
    n_periods: int,
    n_trials: int,
    rng: np.random.Generator,
    *,
    volatility: float = 0.02,
    planted_sharpe: float = 0.0,
) -> TrialReturns:
    r"""Draw a ``(n_periods, n_trials)`` matrix of candidate-strategy returns.

    Every column is i.i.d. Gaussian noise with the given per-period ``volatility`` and
    zero mean — *except* the planted column (index 0) when ``planted_sharpe`` is
    non-zero, which is given mean ``planted_sharpe * volatility`` so its true
    per-period Sharpe equals ``planted_sharpe``.

    Parameters
    ----------
    n_periods : int
        Number of return observations ``T``.
    n_trials : int
        Number of candidate strategies ``N`` (columns).
    rng : numpy.random.Generator
        Seeded generator, injected for reproducibility.
    volatility : float
        Per-period return volatility of every trial.
    planted_sharpe : float
        Per-period Sharpe of the planted signal; ``0`` leaves the matrix all-noise.
    """
    if n_periods < 2 or n_trials < 2:
        raise ValueError("need at least 2 periods and 2 trials")
    returns = rng.normal(0.0, volatility, size=(n_periods, n_trials))
    planted_index: int | None = None
    if planted_sharpe != 0.0:
        planted_index = 0
        returns[:, planted_index] += planted_sharpe * volatility
    return TrialReturns(
        returns=np.asarray(returns, dtype=np.float64),
        planted_index=planted_index,
        planted_sharpe=planted_sharpe,
    )
