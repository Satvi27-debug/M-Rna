"""Pydantic report schema and the deterministic safety validation layer.

Why safety rules live in code and not in the prompt
---------------------------------------------------
PRD Section 8.2 puts four safety constraints in the system prompt. Three of them
guide the model's behaviour, but constraint 1 - never recommend a dose that
contradicts the retrieved guideline text - is **also** enforced here in
deterministic Python, because FR-4.4 requires a test case with an
above-threshold dose to produce ``Eligible_For_Vaccine == false``. A prompt
instruction cannot be relied on to satisfy an acceptance test; a function can.

So the architecture is: the prompt asks the model to comply, and this module
verifies that it did, overriding it when it did not. That is also why safety
validation is kept out of :mod:`src.llm.report_generator`, which is the
presentation layer.

The five required fields (FR-4.1)
--------------------------------
``Eligible_For_Vaccine``, ``Recommended_Dose_mg``, ``Reasoning_Steps``,
``FDA_Safety_Warnings_Cited``, ``Disclaimer``.

The first four are mandatory. ``Disclaimer`` carries a default so a hand-written
report is valid, but :func:`enforce_safety_constraints` overwrites it with
``config.REQUIRED_DISCLAIMER`` unconditionally, which is what FR-4.5 demands
("without exception") regardless of what the model wrote.

Status
------
The Pydantic model is defined verbatim from PRD Section 8.3 because the schema
*is* the specified interface. The validation functions are contracts only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from src import config


# ---------------------------------------------------------------------------
# FR-4.1 - the required output schema
# ---------------------------------------------------------------------------


class ClinicalSafetyReport(BaseModel):
    """Structured clinical safety summary returned by Module 4.

    Defined verbatim from PRD Section 8.3. The four non-default fields are
    required, so instantiating the model without one raises ``pydantic.
    ValidationError`` - the FR-4.1 acceptance criterion.
    """

    Eligible_For_Vaccine: bool = Field(
        ..., description="Whether the patient is eligible given risk and guidelines"
    )
    Recommended_Dose_mg: float = Field(
        ..., description="Recommended dose in mg, must respect retrieved guideline bounds"
    )
    Reasoning_Steps: list[str] = Field(
        ..., description="Ordered, human-readable reasoning steps behind the decision"
    )
    FDA_Safety_Warnings_Cited: list[str] = Field(
        ..., description="Exact guideline snippets the recommendation relies on"
    )
    Disclaimer: str = Field(
        default="Student engineering project output — not real clinical guidance.",
        description="Must always be present in the final report",
    )


#: The exact field set FR-4.1 requires, used by ``tests/test_llm_schema`` and by
#: the completeness check in :func:`validate_report`.
REQUIRED_REPORT_FIELDS: tuple[str, ...] = (
    "Eligible_For_Vaccine",
    "Recommended_Dose_mg",
    "Reasoning_Steps",
    "FDA_Safety_Warnings_Cited",
    "Disclaimer",
)


# ---------------------------------------------------------------------------
# Deterministic dose bounds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseBound:
    """A dose ceiling read out of the retrieved guideline chunks (FR-4.4).

    Attributes
    ----------
    max_mcg:
        Maximum dose permitted for this patient profile. FR-4.4 fires when a
        recommended dose exceeds it.
    source_text:
        The guideline sentence the bound was derived from, so the rejection can
        cite its own evidence.
    unit:
        ``'mcg'`` or ``'mg'``. The guideline document is in micrograms;
        :attr:`max_mg` converts so the report's ``Recommended_Dose_mg`` field can
        be compared against it without unit confusion.
    """

    max_mcg: float
    source_text: str = ""
    unit: str = "mcg"

    @property
    def max_mg(self) -> float:
        """The same ceiling expressed in milligrams for ``Recommended_Dose_mg``."""
        return self.max_mcg / 1000.0

    def exceeds(self, dose_mcg: float) -> bool:
        """True when ``dose_mcg`` is above the bound (FR-4.4)."""
        return float(dose_mcg) > self.max_mcg

    @classmethod
    def hard_ceiling(cls) -> "DoseBound":
        """Bound derived from ``config.GUIDELINE_HARD_DOSE_CEILING_MCG``.

        The guideline document states that doses above 100 mcg are
        high-reactogenicity and require secondary clinician review, so 100 mcg is
        the unconditional ceiling. This is the fallback when no patient-specific
        bound was extracted, so the check always has something to enforce.
        """
        return cls(
            max_mcg=config.GUIDELINE_HARD_DOSE_CEILING_MCG,
            source_text=(
                "Doses above 100 micrograms are considered high-reactogenicity "
                "doses and require secondary clinician review before administration."
            ),
        )


@dataclass
class SafetyViolation:
    """A single failed safety rule, recorded rather than raised.

    Collecting violations instead of raising on the first one lets the report
    still be persisted with all problems annotated, which is more useful during
    development than losing the output entirely.
    """

    rule: str
    detail: str
    requirement: str

    def as_dict(self) -> dict[str, str]:
        return {"rule": self.rule, "detail": self.detail, "requirement": self.requirement}


@dataclass
class SafetyOutcome:
    """Result of applying the safety rules to a validated report."""

    report: ClinicalSafetyReport
    violations: list[SafetyViolation] = field(default_factory=list)
    overridden: bool = False

    @property
    def is_compliant(self) -> bool:
        return not self.violations

    def as_dict(self) -> dict[str, Any]:
        return {
            "compliant": self.is_compliant,
            "overridden": self.overridden,
            "violations": [v.as_dict() for v in self.violations],
            "report": self.report.model_dump(),
        }


# ---------------------------------------------------------------------------
# Validation entry points
# ---------------------------------------------------------------------------


def validate_report(payload: dict[str, Any]) -> ClinicalSafetyReport:
    """Validate raw LLM output against the Pydantic schema (FR-4.3 first gate).

    Parameters
    ----------
    payload:
        The parsed JSON object returned by the model.

    Returns
    -------
    ClinicalSafetyReport

    Raises
    ------
    pydantic.ValidationError
        If a required field is missing or mistyped. The caller catches this and
        retries exactly once, per FR-4.3.
    """
    raise NotImplementedError


def enforce_safety_constraints(
    report: ClinicalSafetyReport, bound: DoseBound | None = None
) -> SafetyOutcome:
    """Apply the deterministic safety rules to a schema-valid report.

    Enforced here, in code, so FR-4.4 and FR-4.5 hold regardless of what the
    model produced:

    1. **Dose bound (FR-4.4).** If ``Recommended_Dose_mg`` exceeds
       ``bound.max_mg``, the report is overridden: eligibility is forced to
       ``False``, the offending dose is clamped to the bound, and a violation
       naming the guideline source is recorded.
    2. **Disclaimer (FR-4.5).** ``Disclaimer`` is unconditionally replaced with
       ``config.REQUIRED_DISCLAIMER`` if it is empty or does not carry the
       required notice.
    3. **Citations (FR-4.2).** ``FDA_Safety_Warnings_Cited`` must be non-empty;
       an empty list means the recommendation is ungrounded and is recorded as a
       violation.
    4. **Negative dose (FR-4.1).** A negative recommended dose is a sign of
       malformed generation and is recorded.

    Parameters
    ----------
    report:
        A report that has already passed :func:`validate_report`.
    bound:
        Patient-specific dose ceiling. Defaults to :meth:`DoseBound.hard_ceiling`.

    Returns
    -------
    SafetyOutcome
        The possibly-corrected report plus every violation found.

    Notes
    -----
    The correction is a hard override, not a suggestion. Prompt instructions
    shape behaviour; this function guarantees the invariant.
    """
    raise NotImplementedError


def extract_dose_bound(
    guideline_chunks: list[str], default: DoseBound | None = None
) -> DoseBound:
    """Derive a dose ceiling from retrieved guideline text (FR-4.4 input).

    The guideline document states in Section 2.1 that primary-series doses fall
    between 25 and 100 mcg and that doses above 100 mcg are high-reactogenicity
    requiring secondary review, and in Section 2.2 that age bands 12-64 use that
    standard adult range. This function locates those statements in the
    retrieved chunks and returns the ceiling that applies.

    Implementation requirement
    ---------------------------
    Matching must be traceable to the actual retrieved text, not hard-coded, so
    the report can cite its own evidence. When no explicit numeric bound is
    found, ``default`` (or :meth:`DoseBound.hard_ceiling`) is returned and the
    fallback is recorded rather than silently assumed.
    """
    raise NotImplementedError


def parse_llm_json(raw_response: str) -> dict[str, Any]:
    """Extract the JSON object from a raw model response.

    Models wrap structured output in Markdown fences or surrounding prose even
    when given format instructions. This tolerates a ```json fence and leading or
    trailing prose, and returns the first balanced JSON object it finds.

    Raises
    ------
    ReportParseError
        If no JSON object can be located, which the caller treats as a validation
        failure and retries once per FR-4.3.
    """
    raise NotImplementedError


class ReportParseError(ValueError):
    """Raised when a raw LLM response contains no parsable JSON object."""