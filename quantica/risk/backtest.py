r"""Backtesting VaR and ES — the real deliverable of the risk layer.

A risk *number* is cheap; the evidence that a risk model is adequate is a battery
of statistical backtests, and the evidence that *those* are trustworthy is that
their size and power have themselves been checked (see ``tests/risk``). This module
implements the standard VaR backtests and — the part most libraries omit — a
correct **Expected-Shortfall** backtest.

VaR backtests (on the exception sequence)
-----------------------------------------
An *exception* on day :math:`t` is a loss exceeding the VaR forecast,
:math:`L_t > \mathrm{VaR}_t`; under a correct model exceptions are i.i.d.
Bernoulli with rate :math:`p = 1-\alpha`.

* :func:`kupiec_pof` — Kupiec's proportion-of-failures test of *unconditional
  coverage*: are there the right **number** of exceptions? (LR :math:`\sim
  \chi^2_1`.)
* :func:`christoffersen_independence` — are exceptions **independent**, or do they
  cluster? (LR :math:`\sim \chi^2_1`.)
* :func:`christoffersen_cc` — the joint *conditional coverage* test (right number
  **and** independent; LR :math:`\sim \chi^2_2`).
* :func:`basel_traffic_light` — the Basel green/yellow/red zone and capital
  multiplier add-on from the exception count over 250 days.

ES backtest (the highlighted gap)
---------------------------------
ES is **not elicitable** (Gneiting, 2011): there is no scoring function whose
minimiser is the ES, so the naive "count and compare" approach used for VaR does
not transfer. :func:`acerbi_szekely` implements the Acerbi--Székely (2014) tests,
which sidestep elicitability by testing the *magnitude* of tail losses against the
predicted ES, with significance from a Monte-Carlo null. This is precisely the
"validated where others are opaque" capability.

References
----------
Kupiec (1995); Christoffersen (1998); Basel Committee (1996/2019); Acerbi &
Székely (2014), "Backtesting Expected Shortfall"; Gneiting (2011),
"Making and evaluating point forecasts".
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import xlogy
from scipy.stats import binom, chi2

from quantica.core.types import FloatArray

if TYPE_CHECKING:
    from quantica.risk.engines import VaREngine
    from quantica.risk.portfolio import Portfolio

__all__ = [
    "AcerbiSzekelyResult",
    "BaselResult",
    "BaselZone",
    "ChristoffersenResult",
    "KupiecResult",
    "acerbi_szekely",
    "basel_traffic_light",
    "christoffersen_cc",
    "christoffersen_independence",
    "exceptions",
    "kupiec_pof",
    "rolling_var_forecasts",
]


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #


def exceptions(realized_losses: FloatArray, var_forecasts: FloatArray) -> FloatArray:
    """The 0/1 exception sequence ``1{loss > VaR}`` (a *hit* series)."""
    losses = np.asarray(realized_losses, dtype=np.float64)
    var = np.asarray(var_forecasts, dtype=np.float64)
    if losses.shape != var.shape:
        raise ValueError(f"shape mismatch: losses {losses.shape} vs var {var.shape}")
    return (losses > var).astype(np.float64)


# --------------------------------------------------------------------------- #
# Kupiec unconditional coverage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KupiecResult:
    """Kupiec proportion-of-failures test outcome."""

    statistic: float
    p_value: float
    n_exceptions: int
    n_obs: int
    expected_rate: float
    observed_rate: float

    def reject(self, size: float = 0.05) -> bool:
        """Whether to reject correct unconditional coverage at the given test size."""
        return self.p_value < size


def kupiec_pof(n_exceptions: int, n_obs: int, level: float) -> KupiecResult:
    r"""Kupiec's POF test of unconditional coverage (LR :math:`\sim \chi^2_1`)."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_obs <= 0 or not 0 <= n_exceptions <= n_obs:
        raise ValueError(
            f"need 0 <= n_exceptions <= n_obs and n_obs > 0, got {n_exceptions}, {n_obs}"
        )
    p = 1.0 - level
    x, n = n_exceptions, n_obs
    pi_hat = x / n
    ll_null = xlogy(x, p) + xlogy(n - x, 1.0 - p)
    ll_alt = xlogy(x, pi_hat) + xlogy(n - x, 1.0 - pi_hat)
    lr = float(-2.0 * (ll_null - ll_alt))
    lr = max(lr, 0.0)  # guard tiny negative from round-off
    return KupiecResult(
        statistic=lr,
        p_value=float(chi2.sf(lr, 1)),
        n_exceptions=x,
        n_obs=n,
        expected_rate=p,
        observed_rate=pi_hat,
    )


# --------------------------------------------------------------------------- #
# Christoffersen independence and conditional coverage
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChristoffersenResult:
    """Christoffersen independence / conditional-coverage test outcome."""

    statistic: float
    p_value: float
    dof: int

    def reject(self, size: float = 0.05) -> bool:
        """Whether to reject at the given test size."""
        return self.p_value < size


def _transition_counts(hits: FloatArray) -> tuple[int, int, int, int]:
    """Counts (n00, n01, n10, n11) of consecutive-state transitions in a hit series."""
    v = np.asarray(hits, dtype=np.int64)
    prev, curr = v[:-1], v[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    return n00, n01, n10, n11


def christoffersen_independence(hits: FloatArray) -> ChristoffersenResult:
    r"""Christoffersen test that exceptions are serially independent (:math:`\chi^2_1`).

    A first-order Markov alternative (different exception probabilities after a hit
    vs a non-hit) is tested against the i.i.d. null; a large statistic means
    exceptions **cluster**.
    """
    n00, n01, n10, n11 = _transition_counts(hits)
    n0, n1 = n00 + n01, n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0.0
    ll_null = xlogy(n01 + n11, pi) + xlogy(n00 + n10, 1.0 - pi)
    ll_alt = xlogy(n01, pi01) + xlogy(n00, 1.0 - pi01) + xlogy(n11, pi11) + xlogy(n10, 1.0 - pi11)
    lr = float(-2.0 * (ll_null - ll_alt))
    lr = max(lr, 0.0)
    return ChristoffersenResult(statistic=lr, p_value=float(chi2.sf(lr, 1)), dof=1)


def christoffersen_cc(hits: FloatArray, level: float) -> ChristoffersenResult:
    r"""Christoffersen conditional-coverage test: coverage **and** independence.

    The statistic is Kupiec's LR plus the independence LR and is
    :math:`\sim \chi^2_2` (Christoffersen, 1998).
    """
    hits_arr = np.asarray(hits, dtype=np.float64)
    n = hits_arr.size
    x = int(hits_arr.sum())
    lr_uc = kupiec_pof(x, n, level).statistic
    lr_ind = christoffersen_independence(hits_arr).statistic
    lr_cc = lr_uc + lr_ind
    return ChristoffersenResult(statistic=lr_cc, p_value=float(chi2.sf(lr_cc, 2)), dof=2)


# --------------------------------------------------------------------------- #
# Basel traffic-light zones
# --------------------------------------------------------------------------- #


class BaselZone(Enum):
    """Basel supervisory backtesting zones."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class BaselResult:
    """Basel traffic-light outcome for a VaR model over a backtest window."""

    zone: BaselZone
    n_exceptions: int
    n_obs: int
    cumulative_probability: float
    multiplier_addon: float


# Basel plus-factor add-ons for the yellow zone at 250 days / 99% VaR.
_YELLOW_ADDON = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def basel_traffic_light(n_exceptions: int, n_obs: int = 250, level: float = 0.99) -> BaselResult:
    r"""Basel green/yellow/red zone and capital multiplier add-on.

    Zones follow the cumulative probability of observing up to ``n_exceptions``
    exceptions under the null rate :math:`1-\alpha`: **green** below 95%, **yellow**
    from 95% up to 99.99%, **red** at or above 99.99%. The multiplier add-on is the
    supervisory plus-factor (0 in green; the standard 0.40-0.85 table in yellow;
    1.00 in red), calibrated for the canonical 250-day / 99% window.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    p = 1.0 - level
    cum = float(binom.cdf(n_exceptions, n_obs, p))
    if cum < 0.95:
        zone, addon = BaselZone.GREEN, 0.0
    elif cum < 0.9999:
        zone, addon = BaselZone.YELLOW, _YELLOW_ADDON.get(n_exceptions, 0.85)
    else:
        zone, addon = BaselZone.RED, 1.0
    return BaselResult(
        zone=zone,
        n_exceptions=n_exceptions,
        n_obs=n_obs,
        cumulative_probability=cum,
        multiplier_addon=addon,
    )


# --------------------------------------------------------------------------- #
# Acerbi--Székely Expected-Shortfall backtest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AcerbiSzekelyResult:
    """Acerbi--Székely ES backtest outcome.

    ``statistic`` is positive when realised tail losses *exceed* the predicted ES
    (the model under-estimates risk). ``p_value`` — present only when a Monte-Carlo
    null is supplied — is the upper-tail probability of the statistic under a
    correct model; a small value rejects ES adequacy.
    """

    statistic: float
    p_value: float | None
    method: str
    n_exceptions: int


def _z1(losses: FloatArray, var: FloatArray, es: FloatArray) -> float:
    """Acerbi--Székely Test 1 statistic (conditional on the exceptions)."""
    hits = losses > var
    n_hits = int(hits.sum())
    if n_hits == 0:
        return float("nan")
    return float(np.sum(losses[hits] / es[hits]) / n_hits - 1.0)


def _z2(losses: FloatArray, var: FloatArray, es: FloatArray, level: float) -> float:
    """Acerbi--Székely Test 2 statistic (unconditional)."""
    hits = losses > var
    n = losses.size
    return float(np.sum(losses * hits / es) / (n * (1.0 - level)) - 1.0)


def acerbi_szekely(
    realized_losses: FloatArray,
    var_forecasts: FloatArray,
    es_forecasts: FloatArray,
    level: float,
    *,
    method: str = "Z2",
    null_losses: FloatArray | None = None,
) -> AcerbiSzekelyResult:
    r"""Acerbi--Székely (2014) Expected-Shortfall backtest.

    Parameters
    ----------
    realized_losses : ndarray
        Realised losses :math:`L_t`, shape ``(T,)``.
    var_forecasts, es_forecasts : ndarray
        The VaR and ES forecasts aligned with ``realized_losses`` (scalars are
        broadcast).
    level : float
        Confidence level :math:`\alpha`.
    method : {"Z1", "Z2"}
        ``"Z2"`` (unconditional, the default workhorse) averages the tail loss over
        *all* days; ``"Z1"`` (conditional) averages only over exception days.
    null_losses : ndarray, optional
        Shape ``(n_sims, T)`` of losses simulated under the *predictive* model
        (i.e. under :math:`H_0`). When given, the p-value is the fraction of null
        statistics at least as large as the realised one (ES not elicitable → the
        null is obtained by simulation, not a closed form).
    """
    losses = np.asarray(realized_losses, dtype=np.float64)
    var = np.broadcast_to(np.asarray(var_forecasts, dtype=np.float64), losses.shape)
    es = np.broadcast_to(np.asarray(es_forecasts, dtype=np.float64), losses.shape)
    if method not in ("Z1", "Z2"):
        raise ValueError(f"method must be 'Z1' or 'Z2', got {method!r}")

    def stat(sample: FloatArray) -> float:
        return _z1(sample, var, es) if method == "Z1" else _z2(sample, var, es, level)

    statistic = stat(losses)
    n_exceptions = int(np.sum(losses > var))

    p_value: float | None = None
    if null_losses is not None:
        null = np.asarray(null_losses, dtype=np.float64)
        if null.ndim != 2 or null.shape[1] != losses.size:
            raise ValueError(
                f"null_losses must have shape (n_sims, {losses.size}), got {null.shape}"
            )
        null_stats = np.array([stat(row) for row in null], dtype=np.float64)
        null_stats = null_stats[np.isfinite(null_stats)]
        # Upper tail: reject when realised tail losses exceed the ES prediction.
        p_value = float(np.mean(null_stats >= statistic)) if null_stats.size else float("nan")

    return AcerbiSzekelyResult(
        statistic=statistic, p_value=p_value, method=method, n_exceptions=n_exceptions
    )


# --------------------------------------------------------------------------- #
# Rolling out-of-sample forecasts (drives real backtests and the demo)
# --------------------------------------------------------------------------- #


def rolling_var_forecasts(
    engine: VaREngine,
    asset_returns: FloatArray,
    portfolio: Portfolio,
    *,
    level: float,
    window: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    r"""Roll ``engine`` over a moving window, one-step-ahead, out of sample.

    For each day ``t >= window`` the engine is refit on the trailing ``window`` days
    ``[t-window, t)`` and its VaR/ES forecast is compared with the *realised* loss
    on day ``t``.

    Returns
    -------
    var_forecasts, es_forecasts, realized_losses : ndarray
        Three aligned arrays of length ``T - window``.
    """
    R = np.asarray(asset_returns, dtype=np.float64)
    if R.ndim == 1:
        R = R[:, None]
    n_obs = R.shape[0]
    if window < 2 or window >= n_obs:
        raise ValueError(f"window must satisfy 2 <= window < n_obs ({n_obs}), got {window}")

    realized = portfolio.losses(R)
    var_f, es_f, loss_f = [], [], []
    for t in range(window, n_obs):
        est = engine.estimate(R[t - window : t], portfolio, level=level)
        var_f.append(est.var)
        es_f.append(est.es)
        loss_f.append(realized[t])
    return (
        np.array(var_f, dtype=np.float64),
        np.array(es_f, dtype=np.float64),
        np.array(loss_f, dtype=np.float64),
    )
