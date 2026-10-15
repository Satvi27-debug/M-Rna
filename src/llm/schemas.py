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
Fully implemented. :func:`validate_report`, :func:`parse_llm_json`,
:func:`extract_dose_bound` and :func:`enforce_safety_constraints` are all real.
Run ``python -m pytest tests/test_report.py`` to exercise them.
"""

from __future__ import annotations

import json
import re
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
class DoseReviewFinding:
    """The documented secondary-review requirement for a submitted dose.

    Created when a dose exceeds the threshold the guideline actually states.
    This is the record that satisfies "preserve the original submitted dose" -
    the submitted value is stored here untouched, so clamping a recommendation
    downstream never destroys what the caller originally asked.

    Attributes
    ----------
    submitted_dose_mcg:
        The dose as submitted, in the unit it was given. Never overwritten by a
        clamped or corrected value.
    unit:
        Unit of ``submitted_dose_mcg``, preserved verbatim from the input record.
    threshold_mcg:
        The threshold that was crossed.
    review_required:
        Whether the source attaches a consequence to crossing it. ``True`` for the
        "requires secondary clinician review" statement.
    threshold_kind:
        What the figure actually is - ``'secondary_clinician_review'``. Never
        ``'maximum_safe_dose'``, because the supplied document does not state one.
    citation:
        The verbatim guideline sentence the threshold came from, or ``""`` when
        the threshold was defaulted rather than read from the retrieved text.
    """

    submitted_dose_mcg: float
    unit: str
    threshold_mcg: float
    review_required: bool
    threshold_kind: str = "secondary_clinician_review"
    citation: str = ""

    @property
    def exceeds_threshold(self) -> bool:
        return self.submitted_dose_mcg > self.threshold_mcg

    def as_dict(self) -> dict[str, Any]:
        return {
            "submitted_dose_mcg": self.submitted_dose_mcg,
            "submitted_dose_preserved": True,
            "unit": self.unit,
            "threshold_mcg": self.threshold_mcg,
            "review_required": self.review_required,
            "threshold_kind": self.threshold_kind,
            "clinical_maximum_established": False,
            "citation": self.citation,
        }


@dataclass(frozen=True)
class DoseBound:
    """A dose threshold read out of the retrieved guideline chunks (FR-4.4).

    Attributes
    ----------
    max_mcg:
        The threshold that was found. FR-4.4 fires when a dose exceeds it.
    source_text:
        The guideline sentence the bound was derived from, so the finding can cite
        its own evidence. Empty when the bound was **defaulted** rather than read
        from the retrieved text, which is how a caller tells the two apart.
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
        """The same threshold expressed in milligrams for ``Recommended_Dose_mg``."""
        return self.max_mcg / 1000.0

    @property
    def derived_from_guideline(self) -> bool:
        """Whether this threshold was read from the retrieved text."""
        return bool(self.source_text)

    @property
    def clinical_maximum_established(self) -> bool:
        """Always ``False``.

        The supplied guideline establishes a *secondary clinician review*
        threshold, not a clinically validated maximum safe dose. No such maximum
        is stated anywhere in the source, so this property exists to make that
        absence impossible to overlook when the value is rendered or asserted on.
        """
        return False

    @property
    def threshold_kind(self) -> str:
        """What this figure is, in words safe to show a reader."""
        return "secondary_clinician_review" if self.derived_from_guideline else "defaulted_threshold"

    def exceeds(self, dose_mcg: float) -> bool:
        """True when ``dose_mcg`` is above the bound (FR-4.4)."""
        return float(dose_mcg) > self.max_mcg

    def review_for(self, dose_mcg: float, unit: str = "mcg") -> DoseReviewFinding:
        """Build the review finding for a submitted dose.

        Records the dose **as submitted**. The caller may separately clamp a
        recommendation; that must not rewrite this value.
        """
        return DoseReviewFinding(
            submitted_dose_mcg=float(dose_mcg),
            unit=unit,
            threshold_mcg=self.max_mcg,
            review_required=self.exceeds(float(dose_mcg)),
            threshold_kind=self.threshold_kind,
            citation=self.source_text,
        )

    @classmethod
    def hard_ceiling(cls) -> "DoseBound":
        """Bound derived from ``config.GUIDELINE_REVIEW_THRESHOLD_MCG``.

        The guideline document states that doses above 100 mcg are
        high-reactogenicity and require secondary clinician review, so 100 mcg is
        the unconditional review threshold. This is the fallback when no bound was
        extracted from the retrieved chunks, so the check always has something to
        enforce.

        Named ``hard_ceiling`` for continuity with existing callers. Read it as
        "the project's enforced review threshold", not as a clinical maximum -
        see :attr:`clinical_maximum_established`.
        """
        return cls(
            max_mcg=config.GUIDELINE_REVIEW_THRESHOLD_MCG,
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
    #: Set when a submitted or recommended dose crossed the guideline's review
    #: threshold. Carries the dose **as submitted**, so a clamped recommendation
    #: can never be mistaken for the original input.
    dose_review: DoseReviewFinding | None = None

    @property
    def is_compliant(self) -> bool:
        return not self.violations

    @property
    def clinical_maximum_established(self) -> bool:
        """Whether a clinical maximum safe dose was established. Always ``False``.

        Exposed on the outcome so no consumer - report, dashboard or pipeline - can
        read an enforced threshold and conclude the source set a clinical maximum.
        """
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "compliant": self.is_compliant,
            "overridden": self.overridden,
            "clinical_maximum_established": self.clinical_maximum_established,
            "dose_review": self.dose_review.as_dict() if self.dose_review else None,
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
    ReportParseError
        If ``payload`` is not a JSON object, e.g. the model returned a bare list or
        a string. That is a shape failure the caller retries once, like a
        validation error.
    pydantic.ValidationError
        If a required field is missing or mistyped. The caller catches this and
        retries exactly once, per FR-4.3.
    """
    if not isinstance(payload, dict):
        raise ReportParseError(
            f"Expected a JSON object at the top level of the report, got "
            f"{type(payload).__name__}. The model must return an object with the "
            f"fields {list(REQUIRED_REPORT_FIELDS)}."
        )
    return ClinicalSafetyReport(**payload)


def enforce_safety_constraints(
    report: ClinicalSafetyReport,
    bound: DoseBound | None = None,
    submitted_dose_mcg: float | None = None,
    submitted_unit: str = "mcg",
) -> SafetyOutcome:
    """Apply the deterministic safety rules to a schema-valid report.

    Enforced here, in code, so FR-4.4 and FR-4.5 hold regardless of what the
    model produced:

    1. **Dose threshold (FR-4.4).** If ``Recommended_Dose_mg`` exceeds
       ``bound.max_mg``, the report is overridden: eligibility is forced to
       ``False``, the offending dose is clamped to the threshold, and a violation
       naming the guideline source is recorded.
    2. **Submitted dose (FR-4.4).** When ``submitted_dose_mcg`` is supplied, it is
       checked **independently of anything the model said**. This is what stops an
       LLM returning ``Eligible_For_Vaccine=True`` for a dose the guideline flags
       for review: the model is never asked about the submitted dose at all, and
       the finding survives whatever it produced.
    3. **Disclaimer (FR-4.5).** ``Disclaimer`` is unconditionally replaced with
       ``config.REQUIRED_DISCLAIMER`` if it is empty or does not carry the
       required notice.
    4. **Citations (FR-4.2).** ``FDA_Safety_Warnings_Cited`` must be non-empty;
       an empty list means the recommendation is ungrounded and is recorded as a
       violation.
    5. **Negative dose (FR-4.1).** A negative recommended dose is a sign of
       malformed generation and is recorded.

    Parameters
    ----------
    report:
        A report that has already passed :func:`validate_report`.
    bound:
        Threshold to enforce. Defaults to :meth:`DoseBound.hard_ceiling`.
    submitted_dose_mcg:
        The dose the caller actually supplied, in ``submitted_unit``. Checked
        against ``bound`` on its own terms. ``None`` skips the check - which is
        correct for a report generated without a patient record behind it, and is
        the only way a submitted dose goes unexamined.
    submitted_unit:
        Unit of ``submitted_dose_mcg``, preserved in the finding.

    Returns
    -------
    SafetyOutcome
        The possibly-corrected report, every violation found, and a
        :class:`DoseReviewFinding` when a dose crossed the threshold. The finding
        always carries the dose **as submitted**; clamping affects only the
        report's recommended value.

    Notes
    -----
    The correction is a hard override, not a suggestion. Prompt instructions
    shape behaviour; this function guarantees the invariant.

    The input report is never mutated: a corrected copy is returned, so a caller
    holding the pre-enforcement report still has the model's original output for
    auditing. Which is why :class:`SafetyOutcome` carries both.
    """
    if bound is None:
        bound = DoseBound.hard_ceiling()

    violations: list[SafetyViolation] = []
    updates: dict[str, Any] = {}

    # --- Rule 0: the submitted dose, checked without reference to the model ---
    # Deliberately first and deliberately independent. The model's opinion is not
    # an input here, so it cannot influence the outcome.
    dose_review: DoseReviewFinding | None = None
    if submitted_dose_mcg is not None:
        dose_review = bound.review_for(submitted_dose_mcg, unit=submitted_unit)
        if dose_review.review_required:
            violations.append(
                SafetyViolation(
                    rule="submitted_dose_requires_clinician_review",
                    detail=(
                        f"Submitted dose {dose_review.submitted_dose_mcg} "
                        f"{dose_review.unit} is above the "
                        f"{bound.max_mcg} mcg threshold at which the guideline "
                        "requires secondary clinician review. The guideline states "
                        "a review threshold, not a clinically established maximum "
                        "safe dose; this project's software eligibility is not a "
                        "clinical approval. Submitted value preserved unchanged."
                    ),
                    requirement="FR-4.4",
                )
            )
            # The verdict is set regardless of what the model claimed.
            if report.Eligible_For_Vaccine:
                updates["Eligible_For_Vaccine"] = False

    # --- Rule 1: dose bound (FR-4.4) -------------------------------------
    dose_mg = float(report.Recommended_Dose_mg)
    bound_mg = bound.max_mg

    if dose_mg < 0:
        # A negative dose is malformed output, not a regulatory judgement, but it
        # fails the same deterministic dose rule so it is reported under FR-4.4 -
        # the only dose-related requirement label the contract defines.
        violations.append(
            SafetyViolation(
                rule="dose_non_negative",
                detail=(
                    f"Recommended_Dose_mg was {dose_mg}, which is negative. The "
                    "value is not a usable dose and was clamped to 0."
                ),
                requirement="FR-4.4",
            )
        )
        updates["Recommended_Dose_mg"] = 0.0

    elif bound.exceeds(dose_mg * 1000.0):
        source = bound.source_text or (
            f"Review threshold of {bound.max_mcg} {bound.unit}; no explicit figure "
            "was found in the retrieved text, so the configured default was used."
        )
        violations.append(
            SafetyViolation(
                rule="dose_exceeds_review_threshold",
                detail=(
                    f"Recommended_Dose_mg was {dose_mg} mg "
                    f"({round(dose_mg * 1000.0, 6)} mcg), above the "
                    f"{bound.max_mcg} mcg threshold. Eligibility was set to false "
                    "and the recommendation clamped to "
                    f"{bound_mg} mg. This threshold is a secondary clinician "
                    "review trigger, not a clinically established maximum safe "
                    f"dose. Guideline evidence: {source}"
                ),
                requirement="FR-4.4",
            )
        )
        updates["Recommended_Dose_mg"] = bound_mg
        updates["Eligible_For_Vaccine"] = False

    # --- Rule 2: disclaimer (FR-4.5) --------------------------------------
    if not _carries_required_disclaimer(report.Disclaimer):
        updates["Disclaimer"] = config.REQUIRED_DISCLAIMER
        violations.append(
            SafetyViolation(
                rule="disclaimer_required",
                detail=(
                    "Disclaimer did not carry the required student-project notice "
                    "and was replaced with it verbatim."
                ),
                requirement="FR-4.5",
            )
        )

    # --- Rule 3: grounding (FR-4.2) ---------------------------------------
    if not [entry for entry in report.FDA_Safety_Warnings_Cited if entry.strip()]:
        violations.append(
            SafetyViolation(
                rule="citations_required",
                detail=(
                    "FDA_Safety_Warnings_Cited was empty, so the recommendation "
                    "rests on no retrieved guideline excerpt. Recorded as "
                    "ungrounded rather than silently accepted."
                ),
                requirement="FR-4.2",
            )
        )

    corrected = report.model_copy(update=updates) if updates else report
    return SafetyOutcome(
        report=corrected,
        violations=violations,
        overridden=bool(updates),
        dose_review=dose_review,
    )


def _carries_required_disclaimer(text: str) -> bool:
    """Whether ``text`` already carries the required notice (FR-4.5).

    Compared on whitespace-normalised, casefolded text so a model that
    re-wrapped or re-cased the sentence is accepted, while a different notice
    (e.g. "consult a doctor") is replaced with the exact required wording.
    """
    if not text or not text.strip():
        return False
    normalise = lambda value: " ".join(value.split()).casefold()
    return normalise(config.REQUIRED_DISCLAIMER) in normalise(text)


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

    What counts as an applicable maximum
    -------------------------------------
    Two statements in Section 2.1 jointly establish the ceiling:

    * ``primary-series doses commonly fall between 25 and 100 micrograms`` - the
      stated upper end of the adult reference range.
    * ``Doses above 100 micrograms are considered high-reactogenicity doses and
      require secondary clinician review`` - an explicit threshold.

    Only a **threshold that states both a number and the consequence of crossing
    it** is accepted as an enforceable ceiling. A bare mention of a number is not
    enough, because "100 micrograms" also appears in contexts that are not dose
    ceilings. When no such statement is present in the retrieved text, this
    function refuses to invent one and returns the configured fallback with an
    empty ``source_text``, which is how a caller can tell an extracted bound from
    a defaulted one.
    """
    ceiling_mcg: float | None = None
    evidence = ""

    for chunk in guideline_chunks or []:
        for sentence in _sentences(chunk):
            figures = _microgram_figures(sentence)
            if not figures:
                continue
            # Only a sentence that states a consequence for exceeding a figure
            # establishes a ceiling; a bare number does not.
            if not _states_threshold_consequence(sentence):
                continue
            candidate = max(figures)
            if ceiling_mcg is None or candidate > ceiling_mcg:
                ceiling_mcg = candidate
                evidence = sentence.strip()

    if ceiling_mcg is not None:
        return DoseBound(max_mcg=ceiling_mcg, source_text=evidence)

    fallback = default if default is not None else DoseBound.hard_ceiling()
    # Deliberately drops the fallback's own source_text. That text is a genuine
    # quotation, but if the bound is being defaulted it was NOT among the chunks
    # this run retrieved, and presenting it as the retrieved evidence would
    # misrepresent where the number came from. An empty source_text is the
    # signal that no applicable maximum was established from the supplied text.
    return DoseBound(max_mcg=fallback.max_mcg, source_text="", unit=fallback.unit)


def _sentences(text: str) -> list[str]:
    """Split a chunk into sentences, keeping a prepended heading line separate."""
    lines = (text or "").split("\n")
    heading = lines[0].strip() if lines else ""
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""

    parts: list[str] = [heading] if heading else []
    import re as _re

    _re.sub(r"\n\s*\n+", "\n\n", body)
    parts.extend(_re.split(r"(?<=[.;:!?])\s+|\n\n", body))
    return [part.strip() for part in parts if part.strip()]


def _microgram_figures(sentence: str) -> list[float]:
    """Microgram quantities stated in ``sentence``.

    Accepts both range separators the guideline uses ("between 25 and 100
    micrograms" and "25 to 50 mcg"), matching the reader in
    :mod:`src.rag.retrieve`. Non-dose sentences return an empty list, so a
    temperature or a duration is never mistaken for a dose.
    """
    import re as _re

    separator = r"(?:to|and|\u2013|-)"
    pattern = (
        r"(\d+(?:\.\d+)?)\s*(?:" + separator + r"\s*(\d+(?:\.\d+)?)\s*)?"
        r"(?:micrograms?|mcg)\b"
    )
    figures: list[float] = []
    for low, high in _re.findall(pattern, sentence, flags=_re.IGNORECASE):
        figures.append(float(low))
        if high:
            figures.append(float(high))
    return figures


def _states_threshold_consequence(sentence: str) -> bool:
    """Whether the sentence says what happens above a figure.

    Required so a ceiling is only derived from a statement that actually imposes
    one. "Doses above 100 micrograms ... require secondary clinician review"
    qualifies; a passing mention of a microgram value does not.
    """
    lowered = sentence.lower()
    markers = (
        "require",
        "requires",
        "required",
        "review",
        "prohibit",
        "not exceed",
        "maximum",
        "ceiling",
        "must not",
        "high-reactogenicity",
        "contraindicat",
    )
    return any(marker in lowered for marker in markers)


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

    Notes
    -----
    Scans for a *balanced* object rather than regex-matching to the first closing
    brace, because a guideline excerpt nested inside ``Reasoning_Steps`` will
    itself contain braces. A brace-counting walk also handles the common case of
    a fenced block being present alongside prose.
    """
    text = (raw_response or "").strip()
    if not text:
        raise ReportParseError("The model returned an empty response.")

    # A fenced ```json block is the most reliable source when present.
    fenced = _FENCE_RE.search(text)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    # Otherwise walk the text for the first balanced top-level object.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : position + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break  # not valid JSON; stop scanning this object
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)

    raise ReportParseError(
        "No parsable JSON object was found in the model response. First 200 "
        f"characters were: {text[:200]!r}"
    )


#: Matches a ```json fenced block, tolerating a bare ``` fence.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


class ReportParseError(ValueError):
    """Raised when a raw LLM response contains no parsable JSON object."""