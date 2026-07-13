r"""FRTB P\&L Attribution (PLA) — the internal-models eligibility test.

Under the Basel *Fundamental Review of the Trading Book* (FRTB), a trading desk may
only use its own internal model (the Internal Models Approach, IMA) to compute
capital if that model's risk factors demonstrably **span the desk's actual P\&L**.
The P\&L Attribution test formalises "span the P\&L" by comparing two daily P\&L
series over a ~250-day window:

* **HPL** — *hypothetical* P\&L: the full-revaluation P\&L of the book with
  positions held fixed, marked on the day's real market moves. This is exactly what
  :meth:`OptionBook.full_revaluation_pnl` produces — the pricing path itself.
* **RTPL** — *risk-theoretical* P\&L: the P\&L the risk model predicts from its
  risk factors and sensitivities. For an option desk that is the delta (or
  delta-gamma) approximation — exactly :meth:`OptionBook.delta_normal_pnl` /
  :meth:`~OptionBook.delta_gamma_pnl`.

So **PLA is the same full-revaluation-vs-sensitivities comparison built in
:mod:`quantica.risk.derivatives`, elevated to a regulatory pass/fail test.** The
question the derivatives-risk step explored informally — *when do the risk model's
factors fail to span the book's P\&L?* — is precisely what PLA scores, and a
short-gamma desk whose risk model carries only delta is the textbook failure: its
delta factor cannot reproduce the curvature in the true P\&L, so RTPL and HPL
diverge and the desk loses IMA eligibility.

The two PLA metrics (Basel MAR33)
---------------------------------
* **Spearman rank correlation** of the paired ``(RTPL_t, HPL_t)`` — does the risk
  model *order* the P\&L moves correctly?
* **Kolmogorov--Smirnov distance** between the RTPL and HPL *distributions* — do
  the two P\&L distributions agree in shape?

Each metric maps to a green / amber / red zone at published breakpoints, and the
desk's overall zone is the worse of the two (see :func:`pla_test`). The capital
consequence: **green** keeps IMA with no add-on; **amber** keeps IMA with a capital
surcharge; **red** makes the desk *ineligible* for IMA — it must fall back to the
Standardised Approach, typically a materially higher charge.

The zone thresholds below are the Basel FRTB values (BCBS d457, "Minimum capital
requirements for market risk", Jan 2019, MAR33) and are cited, not invented.

References
----------
Basel Committee on Banking Supervision, d457 (2019), MAR33 (P\&L attribution test).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import rankdata

if TYPE_CHECKING:
    from quantica.core.types import FloatArray
    from quantica.risk.derivatives import MarketScenarios, OptionBook

__all__ = [
    "KS_AMBER_THRESHOLD",
    "KS_RED_THRESHOLD",
    "SPEARMAN_AMBER_THRESHOLD",
    "SPEARMAN_RED_THRESHOLD",
    "PLAResult",
    "PLAZone",
    "book_pla_test",
    "ks_distance",
    "pla_test",
    "spearman_correlation",
]

# --------------------------------------------------------------------------- #
# Basel FRTB PLA thresholds (BCBS d457, MAR33) — published breakpoints.
# --------------------------------------------------------------------------- #
#: Spearman at or above this is green; below the red threshold is red.
SPEARMAN_AMBER_THRESHOLD = 0.80
#: Spearman below this is red.
SPEARMAN_RED_THRESHOLD = 0.70
#: KS at or below this is green; above the red threshold is red.
KS_AMBER_THRESHOLD = 0.09
#: KS above this is red.
KS_RED_THRESHOLD = 0.12

RTPLMethod = Literal["delta-normal", "delta-gamma"]


class PLAZone(Enum):
    """FRTB P&L-attribution zone (worsening left to right)."""

    GREEN = "green"
    AMBER = "amber"
    RED = "red"

    def __str__(self) -> str:
        return self.value


def spearman_correlation(rtpl: FloatArray, hpl: FloatArray) -> float:
    """Spearman rank correlation of paired ``(RTPL, HPL)`` observations.

    Computed as the Pearson correlation of the (tie-averaged) ranks — the
    definition of Spearman's rho. Matches :func:`scipy.stats.spearmanr` (asserted
    in the tests). If either series is constant (no rank information — e.g. a
    perfectly delta-hedged book scored by a delta-only risk model), the
    correlation is undefined and reported as ``0.0``, which lands in the red zone.
    """
    r, h = _validate_pair(rtpl, hpl)
    rank_r = rankdata(r)
    rank_h = rankdata(h)
    if rank_r.std() == 0.0 or rank_h.std() == 0.0:
        return 0.0
    return float(np.corrcoef(rank_r, rank_h)[0, 1])


def ks_distance(rtpl: FloatArray, hpl: FloatArray) -> float:
    r"""Two-sample Kolmogorov--Smirnov distance :math:`\sup_x |F_R(x) - F_H(x)|`.

    The empirical CDFs of the two (unpaired) distributions are compared at every
    pooled observation. Matches :func:`scipy.stats.ks_2samp` (asserted in the
    tests).
    """
    r, h = _validate_pair(rtpl, hpl)
    pooled = np.concatenate([r, h])
    cdf_r = np.searchsorted(np.sort(r), pooled, side="right") / r.size
    cdf_h = np.searchsorted(np.sort(h), pooled, side="right") / h.size
    return float(np.max(np.abs(cdf_r - cdf_h)))


def _spearman_zone(rho: float) -> PLAZone:
    if rho >= SPEARMAN_AMBER_THRESHOLD:
        return PLAZone.GREEN
    if rho < SPEARMAN_RED_THRESHOLD:
        return PLAZone.RED
    return PLAZone.AMBER


def _ks_zone(ks: float) -> PLAZone:
    if ks <= KS_AMBER_THRESHOLD:
        return PLAZone.GREEN
    if ks > KS_RED_THRESHOLD:
        return PLAZone.RED
    return PLAZone.AMBER


@dataclass(frozen=True)
class PLAResult:
    """Outcome of the FRTB P&L-attribution test for one desk."""

    spearman: float
    ks_statistic: float
    spearman_zone: PLAZone
    ks_zone: PLAZone
    zone: PLAZone
    n_observations: int

    @property
    def ima_eligible(self) -> bool:
        """Whether the desk may still use the Internal Models Approach (not red)."""
        return self.zone is not PLAZone.RED

    def capital_consequence(self) -> str:
        """A one-line description of the regulatory consequence of the zone."""
        if self.zone is PLAZone.GREEN:
            return "IMA-eligible, no PLA capital add-on"
        if self.zone is PLAZone.AMBER:
            return "IMA-eligible with a PLA capital surcharge (add-on)"
        return "IMA-ineligible — desk falls back to the Standardised Approach"


def pla_test(rtpl: FloatArray, hpl: FloatArray) -> PLAResult:
    """Run the FRTB P&L-attribution test on aligned RTPL and HPL series.

    The desk's overall zone is the **worse** of the two per-metric zones: green
    only if both metrics are green, red if either metric is red, amber otherwise
    (BCBS d457, MAR33.43--33.46).
    """
    rho = spearman_correlation(rtpl, hpl)
    ks = ks_distance(rtpl, hpl)
    spearman_zone = _spearman_zone(rho)
    ks_zone = _ks_zone(ks)
    overall = max(spearman_zone, ks_zone, key=_severity)
    return PLAResult(
        spearman=rho,
        ks_statistic=ks,
        spearman_zone=spearman_zone,
        ks_zone=ks_zone,
        zone=overall,
        n_observations=int(np.asarray(rtpl).size),
    )


def book_pla_test(
    book: OptionBook,
    scenarios: MarketScenarios,
    *,
    rtpl_method: RTPLMethod = "delta-gamma",
) -> PLAResult:
    """Run PLA on an option book: HPL = full revaluation, RTPL = sensitivities P&L.

    Makes the pricing/risk connection concrete — the HPL is the pricing path
    (:meth:`OptionBook.full_revaluation_pnl`, no drift from the pricers) and the
    RTPL is the risk model's sensitivities P&L on the *same* seeded scenarios.
    A ``"delta-gamma"`` risk model spans an option book's curvature; a
    ``"delta-normal"`` (delta-only) model does not, and short-gamma desks fail.
    """
    hpl = book.full_revaluation_pnl(scenarios)
    if rtpl_method == "delta-normal":
        rtpl = book.delta_normal_pnl(scenarios)
    elif rtpl_method == "delta-gamma":
        rtpl = book.delta_gamma_pnl(scenarios)
    else:
        raise ValueError(
            f"rtpl_method must be 'delta-normal' or 'delta-gamma', got {rtpl_method!r}"
        )
    return pla_test(rtpl, hpl)


def _severity(zone: PLAZone) -> int:
    return {PLAZone.GREEN: 0, PLAZone.AMBER: 1, PLAZone.RED: 2}[zone]


def _validate_pair(rtpl: FloatArray, hpl: FloatArray) -> tuple[FloatArray, FloatArray]:
    r = np.asarray(rtpl, dtype=np.float64)
    h = np.asarray(hpl, dtype=np.float64)
    if r.ndim != 1 or r.size < 2:
        raise ValueError("rtpl must be a 1-D array of at least 2 observations")
    if h.shape != r.shape:
        raise ValueError(f"hpl shape {h.shape} must match rtpl shape {r.shape}")
    if not (np.all(np.isfinite(r)) and np.all(np.isfinite(h))):
        raise ValueError("rtpl and hpl must be finite")
    return r, h
