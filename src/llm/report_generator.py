"""LLM call, validation retry and report persistence.

Requirements owned here
-----------------------
FR-4.2  Build the prompt using only the Module 1 risk output and the Module 3
        retrieved chunks as factual grounding.
FR-4.3  Validate every LLM response against the Pydantic schema before accepting
        it, and retry generation exactly once on validation failure.
FR-4.6  Persist every validated report as both a ``.json`` file and a
        human-readable ``.md`` file.

The two layers, and why they are separate
-----------------------------------------
* **Grounding.** :func:`build_prompt` assembles the messages. The PRD's
  FR-4.2 acceptance criterion is that manual inspection reveals no clinical
  facts beyond what Modules 1 and 3 supplied, so the patient summary is a
  formatted restatement of the patient's own record and the guideline section is
  only the retrieved chunks. No clinical value is hard-coded into the prompt.
* **Verification.** :func:`generate_report` owns the schema gate, the single
  retry, and the deterministic safety pass from
  :func:`src.llm.schemas.enforce_safety_constraints`. The safety rules are not
  restated in the prompt's language because code, not prose, is what FR-4.4 and
  FR-4.5 are tested against.

Credential handling
-------------------
``LLM_API_KEY`` is read from the environment at call time via ``python-dotenv``,
never at import time and never from source control. Importing this module in a
test suite requires no key; only calling :func:`generate_report` does.

Status
------
Structure and contracts only. Bodies raise ``NotImplementedError``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config


#: The four REQUIRED SAFETY CONSTRAINTS from PRD Section 8.2, verbatim in
#: substance. These shape model behaviour; the deterministic half of constraints
#: 1 and 4 is additionally enforced in :mod:`src.llm.schemas`.
SYSTEM_PROMPT = """You are a clinical safety reporting assistant for a student
engineering project. You must ONLY use the risk score and guideline excerpts
provided below. Never invent clinical facts. Never recommend a dose outside
the bounds stated in the guideline excerpts. If information is insufficient,
default to NOT eligible and explain why. Always include the disclaimer field.

Required safety constraints:
1) Never recommend a dose that contradicts the retrieved guideline text.
2) Never invent a clinical metric, statistic, or guideline that was not present
   in the retrieved chunks.
3) If the retrieved chunks do not contain enough information to answer
   confidently, set Eligible_For_Vaccine to a conservative default and explain
   why in Reasoning_Steps rather than guessing.
4) Every generated report must include a visible disclaimer that this output is
   a student engineering project and not real clinical guidance.
"""


#: Shown to the LLM when the retrieved chunks do not settle the question.
#: Implements constraint 3: refuse to be confidently wrong.
INSUFFICIENT_EVIDENCE_TEMPLATE = (
    "The retrieved guideline excerpts do not contain enough information to "
    "determine eligibility for this patient profile. Set "
    "Eligible_For_Vaccine to false and explain in Reasoning_Steps that the "
    "evidence is insufficient rather than inferring a clinical fact."
)


@dataclass(frozen=True)
class GenerationAttempt:
    """Record of one LLM call, kept so FR-4.3's retry is auditable."""

    attempt_number: int
    succeeded: bool
    failure_reason: str = ""
    latency_s: float = 0.0


@dataclass
class ReportRun:
    """Everything produced by one end-to-end report generation.

    ``attempts`` is how FR-4.3's "retry exactly once" is verified: a successful
    first call yields one entry, a successful retry yields two, and a run that
    never validates yields two and raises.
    """

    report: Any
    safety: Any
    attempts: list[GenerationAttempt] = field(default_factory=list)
    dose_bound: Any = None
    json_path: Path | None = None
    markdown_path: Path | None = None

    @property
    def total_latency_s(self) -> float:
        """Seconds spent in the LLM, for dashboard Panel C."""
        return sum(a.latency_s for a in self.attempts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "succeeded": a.succeeded,
                    "failure_reason": a.failure_reason,
                    "latency_s": round(a.latency_s, 4),
                }
                for a in self.attempts
            ],
            "total_latency_s": round(self.total_latency_s, 4),
            "json_path": str(self.json_path) if self.json_path else None,
            "markdown_path": str(self.markdown_path) if self.markdown_path else None,
        }


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    """Read ``LLM_API_KEY`` from the environment.

    Loads ``.env`` first, then falls back to ``OPENAI_API_KEY`` so an existing
    local setup keeps working.

    Returns
    -------
    str
        The API key.

    Raises
    ------
    RuntimeError
        If no key is present, with a message telling the user to copy
        ``.env.example`` to ``.env``. The key value itself is never logged.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 1 - prompt assembly (FR-4.2)
# ---------------------------------------------------------------------------


def summarise_patient(patient_record: dict[str, Any]) -> str:
    """Restate the patient's own record as a short factual sentence.

    Every value comes from ``patient_record``; nothing clinical is inferred or
    added. This is the Module 1 input side of FR-4.2.
    """
    raise NotImplementedError


def build_prompt(
    risk_label: str,
    risk_probability: float,
    guideline_chunks: list[str],
    patient_summary: str,
) -> list[tuple[str, str]]:
    """Assemble the system and user messages (FR-4.2).

    The user message carries three things and nothing else: the patient's
    summary, the Module 1 risk output, and the Module 3 retrieved chunks joined by
    a separator. The output-parser format instructions are appended to the system
    message by the caller.

    Parameters
    ----------
    risk_label:
        ``'High'`` or ``'Low'`` from :class:`src.ml.predict.RiskPrediction`.
    risk_probability:
        ``P(Reaction_Score = 1)``.
    guideline_chunks:
        Retrieved chunk texts. Must be non-empty; an empty list means the
        grounding is missing and the caller should use
        :data:`INSUFFICIENT_EVIDENCE_TEMPLATE`.
    patient_summary:
        Output of :func:`summarise_patient`.

    Returns
    -------
    list of (role, content)
        Ready for ``ChatPromptTemplate.from_messages``.

    Raises
    ------
    ValueError
        If ``guideline_chunks`` is empty, because generating a report with no
        grounding would violate FR-4.2 outright.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 2 - generate, validate, retry exactly once (FR-4.3)
# ---------------------------------------------------------------------------


def generate_report(
    risk_label: str,
    risk_probability: float,
    guideline_chunks: list[str],
    patient_summary: str,
    max_retries: int = config.LLM_MAX_RETRIES,
) -> ReportRun:
    """Call the LLM, validate the response, and retry once on failure.

    The loop, per FR-4.3:

    1. Call the LLM with the assembled prompt.
    2. Parse the JSON payload and validate it against
       :class:`~src.llm.schemas.ClinicalSafetyReport`.
    3. On a parse or validation failure, retry **exactly once**.
    4. If the retry also fails, raise ``ReportGenerationError`` carrying both
       failure reasons - no third attempt.

    After a report validates, :func:`src.llm.schemas.enforce_safety_constraints`
    runs to apply the dose bound (FR-4.4) and force the disclaimer (FR-4.5).
    Those overrides are deterministic and are not retried.

    Parameters
    ----------
    max_retries:
        Defaults to ``config.LLM_MAX_RETRIES`` (1). Exceeding 1 is rejected,
        since FR-4.3 specifies exactly one retry.

    Returns
    -------
    ReportRun
        The validated, safety-checked report plus the attempt log.

    Raises
    ------
    ReportGenerationError
        After both attempts fail.
    ValueError
        If ``max_retries`` is greater than 1.
    """
    raise NotImplementedError


class ReportGenerationError(RuntimeError):
    """Raised when both the initial call and the single retry fail validation."""

    def __init__(self, attempts: list[GenerationAttempt]) -> None:
        self.attempts = list(attempts)
        reasons = " | ".join(
            f"attempt {a.attempt_number}: {a.failure_reason}" for a in self.attempts
        )
        super().__init__(
            f"Report generation failed after {len(self.attempts)} attempt(s) "
            f"(FR-4.3 permits one retry). Causes - {reasons}"
        )


# ---------------------------------------------------------------------------
# The single LLM call - the one seam the retry loop depends on
# ---------------------------------------------------------------------------


def invoke_llm(messages: list[tuple[str, str]], model: str = config.LLM_MODEL) -> str:
    """Send one prompt to the LLM and return the raw text response.

    This is the *only* place a network call is made, and it is deliberately a
    single function. :func:`generate_report` calls it, and ``tests/test_report.py``
    patches it to exercise the FR-4.3 retry path without spending tokens or
    requiring an API key. Keeping the seam narrow means the retry logic can be
    tested exhaustively and for free.

    Parameters
    ----------
    messages:
        ``[(role, content), ...]`` from :func:`build_prompt`, with the parser's
        format instructions already appended to the system message.
    model:
        Model identifier.

    Returns
    -------
    str
        The raw response text, which :func:`src.llm.schemas.parse_llm_json`
        then extracts JSON from.

    Raises
    ------
    RuntimeError
        If no API key is configured, or the provider call fails.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 3 - persistence and presentation (FR-4.6)
# ---------------------------------------------------------------------------


def render_markdown(run: ReportRun) -> str:
    """Render the report as a human-readable Markdown summary.

    Presentation only. This function reads the already-validated report and
    contains no safety logic, which is what keeps safety validation and
    presentation separate (architecture rule 7).

    Includes the eligibility verdict, recommended dose, numbered reasoning steps,
    the cited guideline excerpts, any safety violations found during
    enforcement, and the disclaimer rendered as a blockquote so it is impossible
    to miss.
    """
    raise NotImplementedError


def persist_report(
    run: ReportRun, reports_dir: Any = None, basename: str | None = None
) -> tuple[Path, Path]:
    """Write the report as a matching ``.json`` / ``.md`` pair (FR-4.6).

    Both files share a basename so the pair is obvious in ``outputs/reports/``.
    The ``.md`` carries the rendered summary; the ``.json`` carries the schema
    dump together with the safety outcome and attempt log, so a reader can see
    exactly what was enforced.

    Parameters
    ----------
    run:
        Output of :func:`generate_report`.
    reports_dir:
        Destination. Defaults to ``config.REPORTS_DIR``.
    basename:
        File stem. Defaults to the patient id, or ``report`` when unknown.

    Returns
    -------
    tuple of pathlib.Path
        ``(json_path, markdown_path)``, both recorded on ``run``.

    Raises
    ------
    OSError
        If the reports directory cannot be created or written.
    """
    raise NotImplementedError


def record_generation_latency(run: ReportRun, path: Any = None) -> Any:
    """Append this run's latency to ``outputs/logs/latency_log.json``.

    Feeds the LLM half of dashboard Panel C, which compares retrieval latency
    against report-generation latency (PRD Section 8.4). Called by
    :mod:`src.pipeline`, not by the dashboard.
    """
    raise NotImplementedError