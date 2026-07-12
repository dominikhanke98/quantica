r"""Calibration — are the PD *levels* right? This is where PD validation lives.

A model can rank obligors perfectly and still carry PDs that are half the true
default rates — discrimination cannot see that, calibration can. Regulatory PD
validation is therefore organised **per rating grade**: pool the obligors of a
grade, compare the observed default rate with the grade's assigned PD, and test
the difference.

* :func:`binomial_test` — the exact binomial test of a grade's default count
  against its PD. One-sided by default (``alternative="greater"``): the
  prudential question is whether the PD *under*-states the risk. Being an exact
  discrete test it is **conservative** for low-default grades (documented and
  measured in the size study).
* :func:`jeffreys_test` — the ECB-instruction test: with a Jeffreys
  :math:`\mathrm{Beta}(\tfrac12,\tfrac12)` prior the posterior of the true rate
  is :math:`\mathrm{Beta}(d+\tfrac12,\,n-d+\tfrac12)`, and the p-value is the
  posterior probability that the true rate lies at or below the assigned PD.
  Less conservative than the exact binomial in small grades — precisely why the
  ECB instructions adopt it (and our size study quantifies the difference).
* :func:`hosmer_lemeshow` — the portfolio-level :math:`\chi^2` goodness-of-fit
  over score deciles; the classic whole-curve companion to the per-grade tests.
* :func:`grade_calibration` — the per-grade validation table (n, defaults, mean
  PD, observed rate, both p-values) a bank's validation report actually shows.
* :func:`calibration_curve` — binned mean-predicted vs observed rates for the
  reliability diagram.

References
----------
ECB, "Instructions for reporting the validation results of internal models"
(2019), §2.5.3.1 (Jeffreys test); Basel Committee WP 14 (2005); Hosmer &
Lemeshow (1980); Tasche (2008), "Validation of internal rating systems and PD
estimates".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import beta, binom, chi2

from quantica.core.types import FloatArray
from quantica.risk.credit._common import validate_binary_scores

__all__ = [
    "BinomialTestResult",
    "CalibrationCurve",
    "GradeCalibration",
    "HosmerLemeshowResult",
    "JeffreysResult",
    "assign_grades",
    "binomial_test",
    "calibration_curve",
    "grade_calibration",
    "hosmer_lemeshow",
]

BinomialAlternative = Literal["greater", "less", "two-sided"]

_DEFAULT_N_GRADES = 7
_DEFAULT_HL_GROUPS = 10


# --------------------------------------------------------------------------- #
# Per-grade tests
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BinomialTestResult:
    """Exact binomial test of a grade's default count against its assigned PD."""

    p_value: float
    n_defaults: int
    n_obligors: int
    pd: float
    observed_rate: float
    alternative: BinomialAlternative

    def reject(self, size: float = 0.05) -> bool:
        """Whether to reject PD adequacy at the given test size."""
        return self.p_value < size


def binomial_test(
    n_defaults: int,
    n_obligors: int,
    pd: float,
    *,
    alternative: BinomialAlternative = "greater",
) -> BinomialTestResult:
    r"""Exact binomial test of ``n_defaults`` out of ``n_obligors`` against ``pd``.

    ``alternative="greater"`` (default) tests the prudential direction — too many
    defaults for the assigned PD, i.e. PD *under*-estimation:
    :math:`p = \Pr(X \ge d)` under :math:`X \sim \mathrm{Bin}(n, \mathrm{PD})`.
    ``"less"`` tests over-estimation; ``"two-sided"`` doubles the smaller tail
    (capped at 1) — the simple and transparent two-sided convention.
    """
    _check_grade_inputs(n_defaults, n_obligors, pd)
    d, n = n_defaults, n_obligors
    upper = float(binom.sf(d - 1, n, pd))  # P(X >= d)
    lower = float(binom.cdf(d, n, pd))  # P(X <= d)
    if alternative == "greater":
        p_value = upper
    elif alternative == "less":
        p_value = lower
    elif alternative == "two-sided":
        p_value = min(1.0, 2.0 * min(upper, lower))
    else:
        raise ValueError(
            f"alternative must be 'greater', 'less' or 'two-sided', got {alternative!r}"
        )
    return BinomialTestResult(
        p_value=p_value,
        n_defaults=d,
        n_obligors=n,
        pd=pd,
        observed_rate=d / n,
        alternative=alternative,
    )


@dataclass(frozen=True)
class JeffreysResult:
    r"""Jeffreys-prior calibration test (ECB instructions) for one grade.

    ``p_value`` is the posterior probability :math:`\Pr(\theta \le \mathrm{PD})`
    under the :math:`\mathrm{Beta}(d+\tfrac12,\,n-d+\tfrac12)` posterior: small
    means the true default rate most likely *exceeds* the assigned PD.
    """

    p_value: float
    n_defaults: int
    n_obligors: int
    pd: float
    observed_rate: float
    posterior_mean: float

    def reject(self, size: float = 0.05) -> bool:
        """Whether to flag the PD as under-estimating at the given test size."""
        return self.p_value < size


def jeffreys_test(n_defaults: int, n_obligors: int, pd: float) -> JeffreysResult:
    r"""ECB Jeffreys test of a grade PD against its observed default count.

    The posterior with the Jeffreys prior is
    :math:`\theta \mid d \sim \mathrm{Beta}(d+\tfrac12,\, n-d+\tfrac12)` and the
    reported p-value is :math:`\Pr(\theta \le \mathrm{PD})` — one-sided in the
    prudential direction, matching the ECB validation-reporting instructions.
    """
    _check_grade_inputs(n_defaults, n_obligors, pd)
    a = n_defaults + 0.5
    b = n_obligors - n_defaults + 0.5
    return JeffreysResult(
        p_value=float(beta.cdf(pd, a, b)),
        n_defaults=n_defaults,
        n_obligors=n_obligors,
        pd=pd,
        observed_rate=n_defaults / n_obligors,
        posterior_mean=a / (a + b),
    )


def _check_grade_inputs(n_defaults: int, n_obligors: int, pd: float) -> None:
    if n_obligors <= 0 or not 0 <= n_defaults <= n_obligors:
        raise ValueError(
            f"need 0 <= n_defaults <= n_obligors and n_obligors > 0, got {n_defaults}, {n_obligors}"
        )
    if not 0.0 < pd < 1.0:
        raise ValueError(f"pd must be in (0, 1), got {pd}")


# --------------------------------------------------------------------------- #
# Grades and the per-grade validation table
# --------------------------------------------------------------------------- #


def assign_grades(pd_scores: FloatArray, n_grades: int = _DEFAULT_N_GRADES) -> FloatArray:
    """Quantile-based rating grades (0 = safest) from PD scores.

    A pragmatic master scale for validation exercises: grade edges are the score
    quantiles, so grades are (near-)equally populated. Tied edges are merged, so
    heavily discrete scores may yield fewer than ``n_grades`` distinct grades.
    """
    scores = np.asarray(pd_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("pd_scores must be a non-empty 1-D array")
    if n_grades < 2:
        raise ValueError(f"n_grades must be at least 2, got {n_grades}")
    inner = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_grades + 1)[1:-1]))
    return np.digitize(scores, inner, right=True).astype(np.float64)


@dataclass(frozen=True)
class GradeCalibration:
    """One row of the per-grade calibration table."""

    grade: int
    n_obligors: int
    n_defaults: int
    mean_pd: float
    observed_rate: float
    binomial_p: float
    jeffreys_p: float


def grade_calibration(
    y: FloatArray,
    pd_scores: FloatArray,
    grades: FloatArray | None = None,
    *,
    n_grades: int = _DEFAULT_N_GRADES,
) -> tuple[GradeCalibration, ...]:
    """The per-grade calibration table: both tests on every rating grade.

    Each grade's assigned PD is taken as the mean predicted PD of its obligors
    (the pooled-calibration convention); ``grades`` may be supplied explicitly
    (e.g. a bank master scale) or defaults to :func:`assign_grades`.
    """
    y_arr, s_arr = validate_binary_scores(y, pd_scores)
    g = (
        np.asarray(grades, dtype=np.float64)
        if grades is not None
        else assign_grades(s_arr, n_grades)
    )
    if g.shape != y_arr.shape:
        raise ValueError(f"grades shape {g.shape} must match y shape {y_arr.shape}")
    rows = []
    for grade in np.unique(g):
        mask = g == grade
        n = int(mask.sum())
        d = int(y_arr[mask].sum())
        mean_pd = float(np.mean(s_arr[mask]))
        mean_pd = min(max(mean_pd, 1e-12), 1.0 - 1e-12)  # keep the tests defined
        rows.append(
            GradeCalibration(
                grade=int(grade),
                n_obligors=n,
                n_defaults=d,
                mean_pd=mean_pd,
                observed_rate=d / n,
                binomial_p=binomial_test(d, n, mean_pd).p_value,
                jeffreys_p=jeffreys_test(d, n, mean_pd).p_value,
            )
        )
    return tuple(rows)


# --------------------------------------------------------------------------- #
# Hosmer--Lemeshow and the calibration curve
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HosmerLemeshowResult:
    """Hosmer--Lemeshow goodness-of-fit outcome."""

    statistic: float
    p_value: float
    dof: int
    n_groups: int

    def reject(self, size: float = 0.05) -> bool:
        """Whether to reject calibration adequacy at the given test size."""
        return self.p_value < size


def hosmer_lemeshow(
    y: FloatArray,
    pd_scores: FloatArray,
    *,
    n_groups: int = _DEFAULT_HL_GROUPS,
    dof: int | None = None,
) -> HosmerLemeshowResult:
    r"""Hosmer--Lemeshow :math:`\chi^2` test over score-decile groups.

    .. math::

        H = \sum_g \frac{(O_g - E_g)^2}{E_g\,(1 - \bar p_g)},

    with :math:`E_g = \sum_{i \in g} p_i` and :math:`\bar p_g = E_g/n_g`.
    ``dof`` defaults to ``n_groups - 2`` — the convention for a model *fitted on
    this data* (Hosmer & Lemeshow, 1980). When the PDs were **not** estimated on
    the sample (e.g. testing known/true PDs, or strict out-of-sample validation),
    the statistic is :math:`\chi^2_{G}`; pass ``dof=n_groups``. The size study in
    the tests verifies both calibrations of the null.
    """
    y_arr, s_arr = validate_binary_scores(y, pd_scores)
    if n_groups < 2:
        raise ValueError(f"n_groups must be at least 2, got {n_groups}")
    groups = _quantile_groups(s_arr, n_groups)
    stat = 0.0
    n_used = 0
    for grp in np.unique(groups):
        mask = groups == grp
        n_g = float(mask.sum())
        e_g = float(np.sum(s_arr[mask]))
        o_g = float(np.sum(y_arr[mask]))
        p_bar = e_g / n_g
        if e_g <= 0.0 or p_bar >= 1.0:
            continue  # a degenerate all-0/all-1 group carries no information
        stat += (o_g - e_g) ** 2 / (e_g * (1.0 - p_bar))
        n_used += 1
    df = dof if dof is not None else max(n_used - 2, 1)
    return HosmerLemeshowResult(
        statistic=float(stat),
        p_value=float(chi2.sf(stat, df)),
        dof=df,
        n_groups=n_used,
    )


@dataclass(frozen=True)
class CalibrationCurve:
    """Reliability-diagram data: binned mean predicted PD vs observed rate."""

    mean_predicted: FloatArray
    observed_rate: FloatArray
    counts: FloatArray


def calibration_curve(
    y: FloatArray, pd_scores: FloatArray, *, n_bins: int = _DEFAULT_HL_GROUPS
) -> CalibrationCurve:
    """Binned (mean predicted PD, observed default rate) pairs on score quantiles."""
    y_arr, s_arr = validate_binary_scores(y, pd_scores)
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}")
    groups = _quantile_groups(s_arr, n_bins)
    uniq = np.unique(groups)
    mean_pred = np.array([np.mean(s_arr[groups == u]) for u in uniq])
    observed = np.array([np.mean(y_arr[groups == u]) for u in uniq])
    counts = np.array([np.sum(groups == u) for u in uniq], dtype=np.float64)
    return CalibrationCurve(mean_predicted=mean_pred, observed_rate=observed, counts=counts)


def _quantile_groups(scores: FloatArray, n_groups: int) -> FloatArray:
    """Quantile-bin group labels (tie-merged, so possibly fewer distinct groups)."""
    inner = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_groups + 1)[1:-1]))
    return np.digitize(scores, inner, right=True).astype(np.float64)
