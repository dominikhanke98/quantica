"""The conceptual-soundness review structure: the aggregation rule is a rule."""

from __future__ import annotations

import pytest
from quantica.risk.ml_validation import (
    ConceptualSoundnessReview,
    Recommendation,
    SoundnessComponent,
    Verdict,
)


def component(verdict: Verdict, dim: str = "dim") -> SoundnessComponent:
    condition = "fix it" if verdict is Verdict.CONDITIONAL else ""
    return SoundnessComponent(dimension=dim, verdict=verdict, finding="f", condition=condition)


def test_all_pass_gives_approve() -> None:
    review = ConceptualSoundnessReview.from_components(
        "m", (component(Verdict.PASS, "a"), component(Verdict.PASS, "b"))
    )
    assert review.recommendation is Recommendation.APPROVE
    assert review.conditions() == ()


def test_any_conditional_gives_approve_with_conditions() -> None:
    review = ConceptualSoundnessReview.from_components(
        "m", (component(Verdict.PASS, "a"), component(Verdict.CONDITIONAL, "b"))
    )
    assert review.recommendation is Recommendation.APPROVE_WITH_CONDITIONS
    assert review.conditions() == ("fix it",)


def test_any_fail_gives_reject_even_with_conditionals() -> None:
    review = ConceptualSoundnessReview.from_components(
        "m",
        (
            component(Verdict.PASS, "a"),
            component(Verdict.CONDITIONAL, "b"),
            component(Verdict.FAIL, "c"),
        ),
    )
    assert review.recommendation is Recommendation.REJECT


def test_summary_renders_verdicts_and_recommendation() -> None:
    review = ConceptualSoundnessReview.from_components(
        "challenger GBM", (component(Verdict.CONDITIONAL, "calibration"),)
    )
    text = review.summary()
    assert "challenger GBM" in text
    assert "[CONDITIONAL] calibration" in text
    assert "condition: fix it" in text
    assert "APPROVE WITH CONDITIONS" in text


def test_validation() -> None:
    with pytest.raises(ValueError, match="at least one component"):
        ConceptualSoundnessReview.from_components("m", ())
    with pytest.raises(ValueError, match="must state its condition"):
        SoundnessComponent(dimension="d", verdict=Verdict.CONDITIONAL, finding="f")
    with pytest.raises(ValueError, match="non-empty"):
        SoundnessComponent(dimension="", verdict=Verdict.PASS, finding="f")
