r"""Fairness / disparate impact — with the metric choice stated, not smuggled.

Fairness metrics conflict by theorem: when base rates differ between groups, a
model cannot simultaneously be calibrated within each group and equalise
error/approval rates across groups (Kleinberg--Mullainathan--Raghavan 2016;
Chouldechova 2017). Any fairness assessment therefore *chooses* a definition,
and the choice is a modelling decision that belongs in the validation report.
This module implements the two sides of that trade-off:

* :func:`disparate_impact` — the **approval-rate ratio** between the protected
  and reference groups, with the four-fifths (0.8) threshold labelled as what it
  is: the US EEOC *convention* for adverse impact, not a statistical result.
* :func:`group_calibration` — **calibration within group**: does the assigned PD
  match the observed default rate *inside each group*? Tested two-sided (either
  direction of misestimation is a fairness problem: understating a group's risk
  misprices it, overstating it unfairly penalises it) via the Jeffreys posterior,
  consistent with the credit package's calibration battery.

A model can pass one and fail the other on the same data; reporting both makes
the trade-off visible instead of resolving it silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import beta

from quantica.core.types import FloatArray
from quantica.risk.credit._common import validate_binary_scores

__all__ = [
    "DisparateImpact",
    "GroupCalibration",
    "disparate_impact",
    "group_calibration",
]

#: The four-fifths rule threshold (US EEOC convention for adverse impact).
FOUR_FIFTHS = 0.8


@dataclass(frozen=True)
class DisparateImpact:
    """Approval-rate comparison between protected and reference groups."""

    protected_approval_rate: float
    reference_approval_rate: float
    ratio: float
    passes_four_fifths: bool
    threshold: float
    n_protected: int
    n_reference: int


def disparate_impact(
    pd_scores: FloatArray,
    group: FloatArray,
    *,
    pd_threshold: float,
) -> DisparateImpact:
    """Approval-rate ratio when approving obligors with PD at or below a cutoff.

    ``group`` is 0/1 with 1 the protected group. The ratio is
    protected-approval-rate over reference-approval-rate; below 0.8 fails the
    four-fifths convention.
    """
    scores = np.asarray(pd_scores, dtype=np.float64)
    g = np.asarray(group, dtype=np.float64)
    if scores.ndim != 1 or scores.shape != g.shape:
        raise ValueError("pd_scores and group must be matching 1-D arrays")
    if not np.all(np.isin(g, (0.0, 1.0))) or not (np.any(g == 1.0) and np.any(g == 0.0)):
        raise ValueError("group must be 0/1 with both groups present")
    if not 0.0 < pd_threshold < 1.0:
        raise ValueError(f"pd_threshold must be in (0, 1), got {pd_threshold}")
    approved = scores <= pd_threshold
    rate_protected = float(approved[g == 1.0].mean())
    rate_reference = float(approved[g == 0.0].mean())
    ratio = rate_protected / rate_reference if rate_reference > 0.0 else float("inf")
    return DisparateImpact(
        protected_approval_rate=rate_protected,
        reference_approval_rate=rate_reference,
        ratio=ratio,
        passes_four_fifths=ratio >= FOUR_FIFTHS,
        threshold=pd_threshold,
        n_protected=int((g == 1.0).sum()),
        n_reference=int((g == 0.0).sum()),
    )


@dataclass(frozen=True)
class GroupCalibration:
    """Within-group calibration row (two-sided Jeffreys test)."""

    group: int
    n_obligors: int
    n_defaults: int
    mean_pd: float
    observed_rate: float
    jeffreys_two_sided_p: float

    def reject(self, size: float = 0.05) -> bool:
        """Whether the group's PDs are flagged as miscalibrated (either direction)."""
        return self.jeffreys_two_sided_p < size


def group_calibration(
    y: FloatArray, pd_scores: FloatArray, group: FloatArray
) -> tuple[GroupCalibration, ...]:
    r"""Calibration within each group: pooled mean PD vs observed default rate.

    The two-sided p-value doubles the smaller tail of the Jeffreys
    :math:`\mathrm{Beta}(d+\tfrac12,\,n-d+\tfrac12)` posterior around the mean
    assigned PD (capped at 1) — either direction of group-level misestimation is
    a finding.
    """
    y_arr, s_arr = validate_binary_scores(y, pd_scores)
    g = np.asarray(group, dtype=np.float64)
    if g.shape != y_arr.shape:
        raise ValueError(f"group shape {g.shape} must match y shape {y_arr.shape}")
    if not np.all(np.isin(g, (0.0, 1.0))):
        raise ValueError("group must contain only 0 and 1")
    rows = []
    for label in np.unique(g):
        mask = g == label
        n = int(mask.sum())
        d = int(y_arr[mask].sum())
        mean_pd = float(np.clip(np.mean(s_arr[mask]), 1e-12, 1.0 - 1e-12))
        lower_tail = float(beta.cdf(mean_pd, d + 0.5, n - d + 0.5))
        p_two_sided = min(1.0, 2.0 * min(lower_tail, 1.0 - lower_tail))
        rows.append(
            GroupCalibration(
                group=int(label),
                n_obligors=n,
                n_defaults=d,
                mean_pd=mean_pd,
                observed_rate=d / n,
                jeffreys_two_sided_p=p_two_sided,
            )
        )
    return tuple(rows)
