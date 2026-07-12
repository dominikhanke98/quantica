"""ML-model validation under SR 11-7 (Phase 3, the risk pillar's third family).

The conceptual-soundness and robustness review a bank's model-validation
function runs on a machine-learning model, organised as:

- **explainability** — checks *on* SHAP output: local accuracy (the explainer's
  own consistency property), global importance and driver recovery against a
  known data-generating process, explanation-rank stability, and
  attribution-vs-feature direction.
- **robustness** — prediction stability under seeded input perturbation, and
  discrimination/calibration degradation under covariate shift.
- **fairness** — disparate impact (four-fifths convention) and
  calibration-within-group, with the metric choice and its impossibility
  trade-offs documented rather than hidden.
- **soundness** — the structured SR 11-7 effective-challenge review object that
  integrates the evidence into an explicit approve / approve-with-conditions /
  reject recommendation via a transparent rule.

Same model-agnostic posture as ``quantica.risk.credit``: functions consume model
*outputs* (SHAP matrices, PD scores, predictions) or a bare ``predict`` callable
— never model internals — so the package needs only numpy/scipy. Computing SHAP
values (the ``shap`` library) and fitting models (scikit-learn) happen in
scripts and tests as dev extras.
"""

from __future__ import annotations

from quantica.risk.ml_validation.explainability import (
    DriverRecovery,
    FeatureImportance,
    LocalAccuracy,
    RankStability,
    attribution_direction,
    check_local_accuracy,
    driver_recovery,
    global_importance,
    rank_stability,
)
from quantica.risk.ml_validation.fairness import (
    DisparateImpact,
    GroupCalibration,
    disparate_impact,
    group_calibration,
)
from quantica.risk.ml_validation.robustness import (
    PredictionStability,
    ShiftDegradation,
    performance_under_shift,
    prediction_stability,
)
from quantica.risk.ml_validation.soundness import (
    ConceptualSoundnessReview,
    Recommendation,
    SoundnessComponent,
    Verdict,
)

__all__ = [
    "ConceptualSoundnessReview",
    "DisparateImpact",
    "DriverRecovery",
    "FeatureImportance",
    "GroupCalibration",
    "LocalAccuracy",
    "PredictionStability",
    "RankStability",
    "Recommendation",
    "ShiftDegradation",
    "SoundnessComponent",
    "Verdict",
    "attribution_direction",
    "check_local_accuracy",
    "disparate_impact",
    "driver_recovery",
    "global_importance",
    "group_calibration",
    "performance_under_shift",
    "prediction_stability",
    "rank_stability",
]
