r"""Statistical arbitrage — the signal-research stage feeding portfolio construction.

The portfolio pillar ships construction, backtesting and the backtest-validity layer, but
takes the *signal* as given. This package fills that gap with the classic
mean-reversion / pairs-trading pipeline, built validation-first: the deliverable is the
evidence that a candidate relationship is a genuine, tradeable one rather than a spurious
coincidence between two random walks.

This foundational step ships the **cointegration + spread** layer:

* **Cointegration testing** (:mod:`~quantica.statarb.cointegration`) — Engle--Granger (the
  two-step residual test, with the *correct* MacKinnon cointegration critical values) and
  Johansen (the multivariate reduced-rank trace / maximum-eigenvalue test for the number of
  cointegrating relations), each anchored to ``statsmodels``.
* **Spread modelling** (:mod:`~quantica.statarb.spread`) — the Ornstein--Uhlenbeck fit of a
  mean-reverting spread and its half-life, the practically useful holding-period summary.
* **Dynamic hedge ratio** (:mod:`~quantica.statarb.kalman`) — a Kalman filter estimating a
  *time-varying* hedge ratio (an evolving latent state), for relationships that drift away
  from the static cointegrating coefficient.
* **Synthetic data** (:mod:`~quantica.statarb.data`) — cointegrated pairs, time-varying-ratio
  pairs, spurious random walks, and known-parameter OU paths for the known-truth validation.

Later step (not yet built): the mean-reversion strategy run through the portfolio backtest
and its DSR/PBO validity layer.
"""

from __future__ import annotations

from quantica.statarb.cointegration import (
    EngleGrangerResult,
    JohansenResult,
    engle_granger,
    johansen,
)
from quantica.statarb.data import (
    generate_cointegrated_pair,
    generate_independent_random_walks,
    generate_time_varying_pair,
    simulate_ou_process,
)
from quantica.statarb.kalman import KalmanHedgeResult, kalman_hedge_ratio
from quantica.statarb.spread import OUProcess, estimate_ou_process, ou_half_life

__all__ = [
    "EngleGrangerResult",
    "JohansenResult",
    "KalmanHedgeResult",
    "OUProcess",
    "engle_granger",
    "estimate_ou_process",
    "generate_cointegrated_pair",
    "generate_independent_random_walks",
    "generate_time_varying_pair",
    "johansen",
    "kalman_hedge_ratio",
    "ou_half_life",
    "simulate_ou_process",
]
