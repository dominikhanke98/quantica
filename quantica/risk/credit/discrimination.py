r"""Discrimination — does the PD model *rank* obligors correctly?

Discrimination is the first of the three PD-validation dimensions (with
calibration and stability): a model discriminates well when defaulters receive
systematically higher risk scores than performers, regardless of whether the PD
*levels* are right. The standard measures:

* :func:`auc` — the area under the ROC curve, computed via its **Mann--Whitney
  rank identity** :math:`\mathrm{AUC} = \Pr(s_D > s_N) + \tfrac12\Pr(s_D = s_N)`
  (average ranks handle ties exactly). The tests recompute it two independent
  ways — trapezoidal integration of :func:`roc_curve` and scikit-learn's
  ``roc_auc_score`` — and against the analytic binormal anchor
  :math:`\Phi(\delta/\sqrt2)`.
* :func:`gini` — the accuracy ratio :math:`2\,\mathrm{AUC} - 1`, the banking
  industry's preferred rescaling.
* :func:`ks_statistic` — the Kolmogorov--Smirnov separation
  :math:`\max_t |F_D(t) - F_N(t)|` between the two score distributions.
* :func:`bootstrap_ci` / :func:`discrimination_report` — **stratified** bootstrap
  confidence intervals (resampling within each class keeps every resample
  non-degenerate), seeded via an injected ``Generator``.

Scores are oriented *higher = riskier* (a PD is a valid score).

References
----------
ECB, "Instructions for reporting the validation results of internal models"
(2019); Basel Committee WP 14, "Studies on the validation of internal rating
systems" (2005); Hanley & McNeil (1982) for the AUC--rank identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import rankdata

from quantica.risk.credit._common import validate_binary_scores

if TYPE_CHECKING:
    from collections.abc import Callable

    from quantica.core.types import FloatArray

__all__ = [
    "ConfidenceInterval",
    "DiscriminationReport",
    "auc",
    "bootstrap_ci",
    "discrimination_report",
    "gini",
    "ks_statistic",
    "roc_curve",
]

_DEFAULT_N_BOOT = 1000
_DEFAULT_CI_LEVEL = 0.95


def auc(y: FloatArray, scores: FloatArray) -> float:
    r"""Area under the ROC curve via the Mann--Whitney rank identity.

    With :math:`R_1` the sum of (average, tie-aware) ranks of the defaulters,

    .. math:: \mathrm{AUC} = \frac{R_1 - n_1(n_1+1)/2}{n_0\,n_1},

    which equals :math:`\Pr(s_D > s_N) + \tfrac12 \Pr(s_D = s_N)` exactly —
    including the tie term, so a constant score gives 0.5 by construction.
    """
    y_arr, s_arr = validate_binary_scores(y, scores)
    n1 = float(y_arr.sum())
    n0 = float(y_arr.size - n1)
    ranks = rankdata(s_arr)  # average ranks: exact tie handling
    r1 = float(ranks[y_arr == 1.0].sum())
    return (r1 - n1 * (n1 + 1.0) / 2.0) / (n0 * n1)


def gini(y: FloatArray, scores: FloatArray) -> float:
    r"""The Gini coefficient / accuracy ratio :math:`2\,\mathrm{AUC} - 1`."""
    return 2.0 * auc(y, scores) - 1.0


def ks_statistic(y: FloatArray, scores: FloatArray) -> float:
    r"""Kolmogorov--Smirnov separation between defaulter and performer scores.

    :math:`\mathrm{KS} = \max_t |F_D(t) - F_N(t)|`, evaluated at the distinct
    score values (the only points where the empirical CDFs move; evaluating
    inside a tie group would misstate both CDFs).
    """
    y_arr, s_arr = validate_binary_scores(y, scores)
    n1 = float(y_arr.sum())
    n0 = float(y_arr.size - n1)
    order = np.argsort(s_arr, kind="mergesort")
    y_sorted = y_arr[order]
    s_sorted = s_arr[order]
    cum_bad = np.cumsum(y_sorted) / n1
    cum_good = np.cumsum(1.0 - y_sorted) / n0
    # Only the last position of each tie group is a valid CDF evaluation point.
    last_of_value = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    return float(np.max(np.abs(cum_bad[last_of_value] - cum_good[last_of_value])))


def roc_curve(y: FloatArray, scores: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The ROC curve (FPR, TPR) points, one per distinct score threshold.

    Thresholds sweep from strictest (nothing flagged) to loosest (everything
    flagged); tie groups collapse to a single point, so trapezoidal integration
    of this curve reproduces :func:`auc` exactly, ties included — that identity
    is asserted in the tests as a two-independent-ways anchor.
    """
    y_arr, s_arr = validate_binary_scores(y, scores)
    n1 = float(y_arr.sum())
    n0 = float(y_arr.size - n1)
    order = np.argsort(-s_arr, kind="mergesort")  # descending score
    y_desc = y_arr[order]
    s_desc = s_arr[order]
    last_of_value = np.r_[s_desc[1:] != s_desc[:-1], True]
    tpr = np.cumsum(y_desc)[last_of_value] / n1
    fpr = np.cumsum(1.0 - y_desc)[last_of_value] / n0
    return np.r_[0.0, fpr], np.r_[0.0, tpr]


@dataclass(frozen=True)
class ConfidenceInterval:
    """A bootstrap percentile interval around a point estimate."""

    point: float
    lower: float
    upper: float
    level: float
    n_boot: int


def bootstrap_ci(
    y: FloatArray,
    scores: FloatArray,
    statistic: Callable[[FloatArray, FloatArray], float],
    rng: np.random.Generator,
    *,
    n_boot: int = _DEFAULT_N_BOOT,
    level: float = _DEFAULT_CI_LEVEL,
) -> ConfidenceInterval:
    """Stratified-bootstrap percentile CI for a discrimination statistic.

    Defaulters and performers are resampled *within class* (stratified), which
    preserves the class counts and rules out degenerate one-class resamples that
    a naive bootstrap produces for low-default portfolios. Seeded via the
    injected ``rng``.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_boot < 2:
        raise ValueError(f"n_boot must be at least 2, got {n_boot}")
    y_arr, s_arr = validate_binary_scores(y, scores)
    idx_bad = np.flatnonzero(y_arr == 1.0)
    idx_good = np.flatnonzero(y_arr == 0.0)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        take = np.r_[
            rng.choice(idx_bad, size=idx_bad.size, replace=True),
            rng.choice(idx_good, size=idx_good.size, replace=True),
        ]
        stats[b] = statistic(y_arr[take], s_arr[take])
    alpha = 1.0 - level
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return ConfidenceInterval(
        point=statistic(y_arr, s_arr),
        lower=float(lo),
        upper=float(hi),
        level=level,
        n_boot=n_boot,
    )


@dataclass(frozen=True)
class DiscriminationReport:
    """AUC / Gini / KS with bootstrap confidence intervals for one model."""

    auc: ConfidenceInterval
    gini: ConfidenceInterval
    ks: ConfidenceInterval
    n_obligors: int
    n_defaults: int


def discrimination_report(
    y: FloatArray,
    scores: FloatArray,
    rng: np.random.Generator,
    *,
    n_boot: int = _DEFAULT_N_BOOT,
    level: float = _DEFAULT_CI_LEVEL,
) -> DiscriminationReport:
    """The full discrimination battery (AUC, Gini, KS) with bootstrap CIs."""
    y_arr, s_arr = validate_binary_scores(y, scores)
    return DiscriminationReport(
        auc=bootstrap_ci(y_arr, s_arr, auc, rng, n_boot=n_boot, level=level),
        gini=bootstrap_ci(y_arr, s_arr, gini, rng, n_boot=n_boot, level=level),
        ks=bootstrap_ci(y_arr, s_arr, ks_statistic, rng, n_boot=n_boot, level=level),
        n_obligors=int(y_arr.size),
        n_defaults=int(y_arr.sum()),
    )
