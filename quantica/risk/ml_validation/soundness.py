"""The conceptual-soundness review — the SR 11-7 deliverable, as a structure.

SR 11-7 asks the model-validation function for *effective challenge*: an
evidence-based evaluation of conceptual soundness ending in a clear conclusion.
This module gives that write-up a structure with a **transparent aggregation
rule** — each dimension (discrimination, calibration, explainability,
robustness, fairness, stability, ...) enters as a component with a verdict and a
one-line evidential finding, and the overall recommendation follows
mechanically:

* any ``FAIL``                      → **REJECT**
* otherwise any ``CONDITIONAL``     → **APPROVE WITH CONDITIONS**
* all ``PASS``                      → **APPROVE**

The judgment lives where it belongs — in the per-dimension verdicts, each tied
to measured evidence — while the roll-up is a rule anyone can audit, not a vibe.
``summary()`` renders the review for a validation report.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ConceptualSoundnessReview",
    "Recommendation",
    "SoundnessComponent",
    "Verdict",
]


class Verdict(Enum):
    """Per-dimension outcome of the review."""

    PASS = "pass"
    CONDITIONAL = "conditional"
    FAIL = "fail"

    def __str__(self) -> str:
        return self.value


class Recommendation(Enum):
    """The overall model-validation recommendation."""

    APPROVE = "approve"
    APPROVE_WITH_CONDITIONS = "approve with conditions"
    REJECT = "reject"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class SoundnessComponent:
    """One review dimension: a verdict backed by a stated finding.

    ``condition`` states what must be done/monitored when the verdict is
    ``CONDITIONAL`` (required in that case, meaningless otherwise).
    """

    dimension: str
    verdict: Verdict
    finding: str
    condition: str = ""

    def __post_init__(self) -> None:
        if not self.dimension or not self.finding:
            raise ValueError("dimension and finding must be non-empty")
        if self.verdict is Verdict.CONDITIONAL and not self.condition:
            raise ValueError("a CONDITIONAL verdict must state its condition")


@dataclass(frozen=True)
class ConceptualSoundnessReview:
    """The assembled review: components plus the rule-derived recommendation."""

    model_name: str
    components: tuple[SoundnessComponent, ...]
    recommendation: Recommendation

    @classmethod
    def from_components(
        cls, model_name: str, components: tuple[SoundnessComponent, ...]
    ) -> ConceptualSoundnessReview:
        """Apply the transparent aggregation rule to the component verdicts."""
        if not components:
            raise ValueError("a review needs at least one component")
        verdicts = [c.verdict for c in components]
        if Verdict.FAIL in verdicts:
            recommendation = Recommendation.REJECT
        elif Verdict.CONDITIONAL in verdicts:
            recommendation = Recommendation.APPROVE_WITH_CONDITIONS
        else:
            recommendation = Recommendation.APPROVE
        return cls(model_name=model_name, components=components, recommendation=recommendation)

    def conditions(self) -> tuple[str, ...]:
        """The conditions attached to every CONDITIONAL component."""
        return tuple(c.condition for c in self.components if c.verdict is Verdict.CONDITIONAL)

    def summary(self) -> str:
        """Render the review as report text (markdown-friendly)."""
        lines = [f"Conceptual-soundness review — {self.model_name}", ""]
        for c in self.components:
            lines.append(f"- [{str(c.verdict).upper()}] {c.dimension}: {c.finding}")
            if c.verdict is Verdict.CONDITIONAL:
                lines.append(f"    condition: {c.condition}")
        lines.append("")
        lines.append(f"Recommendation: **{str(self.recommendation).upper()}**")
        return "\n".join(lines)
