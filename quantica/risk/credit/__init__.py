"""Credit-risk / PD model validation (Phase 3, second model family).

The package is organised along the three regulatory validation dimensions —
**discrimination** (AUC/Gini/KS with bootstrap CIs), **calibration** (per-grade
binomial and ECB Jeffreys tests, Hosmer--Lemeshow — the centerpiece: PD
validation lives in calibration), and **stability** (PSI / characteristic
stability) — plus a seeded synthetic-portfolio generator whose *known true PDs*
make the meta-validation possible (size and power of the calibration tests
themselves, measured in ``tests/risk/credit``).

Deliberately model-agnostic: every validator consumes model *outputs* (default
indicators and PD scores), never a fitted model object, so the package needs only
numpy/scipy — champion/challenger fitting (scikit-learn) lives in scripts and
tests, not in the library.
"""

from __future__ import annotations

from quantica.risk.credit.calibration import (
    BinomialTestResult,
    CalibrationCurve,
    GradeCalibration,
    HosmerLemeshowResult,
    JeffreysResult,
    assign_grades,
    binomial_test,
    calibration_curve,
    grade_calibration,
    hosmer_lemeshow,
    jeffreys_test,
)
from quantica.risk.credit.data import CreditSample, generate_credit_portfolio
from quantica.risk.credit.discrimination import (
    ConfidenceInterval,
    DiscriminationReport,
    auc,
    bootstrap_ci,
    discrimination_report,
    gini,
    ks_statistic,
    roc_curve,
)
from quantica.risk.credit.stability import (
    CharacteristicStability,
    PSIResult,
    StabilityBand,
    characteristic_stability,
    psi,
)

__all__ = [
    "BinomialTestResult",
    "CalibrationCurve",
    "CharacteristicStability",
    "ConfidenceInterval",
    "CreditSample",
    "DiscriminationReport",
    "GradeCalibration",
    "HosmerLemeshowResult",
    "JeffreysResult",
    "PSIResult",
    "StabilityBand",
    "assign_grades",
    "auc",
    "binomial_test",
    "bootstrap_ci",
    "calibration_curve",
    "characteristic_stability",
    "discrimination_report",
    "generate_credit_portfolio",
    "gini",
    "grade_calibration",
    "hosmer_lemeshow",
    "jeffreys_test",
    "ks_statistic",
    "psi",
    "roc_curve",
]
