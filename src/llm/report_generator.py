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
Fully implemented. Run ``python -m pytest tests/test_report.py``. Every unit test
patches :func:`invoke_llm`, so the suite needs no API key and no network access.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src import config
from src.llm import schemas


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
    #: Optional patient identifier, used as the report filename stem. Added
    #: because :func:`persist_report` documents "defaults to the patient id" and
    #: there was no field to read it from. It is optional, so every existing
    #: construction of ``ReportRun`` remains valid.
    patient_id: str | None = None

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
    MissingAPIKeyError
        If no key is present. The message names the environment variables to set
        and how to create ``.env``. The key value itself is never logged, and
        never appears in the exception.
    """
    # Loaded here rather than at import time so that importing this module - for
    # tests, or for the dashboard - never requires a credential.
    _load_dotenv_once()

    for variable in ("LLM_API_KEY", "OPENAI_API_KEY"):
        value = os.getenv(variable, "").strip()
        if value:
            return value

    raise MissingAPIKeyError(
        "No LLM credential is configured. Report generation requires an API key.\n"
        "Set one of these environment variables:\n"
        "  LLM_API_KEY    (preferred - the name .env.example declares)\n"
        "  OPENAI_API_KEY (accepted as a fallback)\n"
        "Either export it in your shell, or create a .env file at the project "
        f"root:\n  {Path(config.PROJECT_ROOT) / '.env'}\n"
        "containing the line LLM_API_KEY=<your key>. Copy .env.example to get "
        "started.\n"
        "The key is read only at call time and is never written to logs or "
        "reports."
    )


class MissingAPIKeyError(RuntimeError):
    """Raised when no LLM API key is configured.

    A dedicated type so a caller can distinguish "you have not configured
    credentials" from "the provider rejected the request". The two need opposite
    responses, and conflating them would either send the user chasing a network
    problem or hide a real provider failure.
    """


def _load_dotenv_once() -> None:
    """Load ``.env`` into the environment exactly once per process."""
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    env_file = Path(config.PROJECT_ROOT) / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv

        # override=False so a variable already exported in the shell wins over
        # the file. The shell is the more specific instruction.
        load_dotenv(dotenv_path=env_file, override=False)
    except ImportError:
        # python-dotenv is declared in requirements.txt, but a missing key is
        # already handled below, so this degrades rather than breaking.
        pass


_DOTENV_LOADED = False


# ---------------------------------------------------------------------------
# Step 1 - prompt assembly (FR-4.2)
# ---------------------------------------------------------------------------


#: Fields restated into the prompt, in a stable order.
#:
#: ``config.REQUIRED_COLUMNS`` is deliberately *not* used: it includes
#: ``Reaction_Score``, which is the classification target. Putting the answer into
#: the prompt would let the model copy a label it was supposed to predict, which
#: would make every report look confident and every evaluation meaningless. Only
#: the model inputs plus the identifier are restated.
PATIENT_SUMMARY_FIELDS: tuple[str, ...] = ("Patient_ID",) + config.FEATURE_COLS


def summarise_patient(patient_record: dict[str, Any]) -> str:
    """Restate the patient's own record as a short factual sentence.

    Every value comes from ``patient_record``; nothing clinical is inferred or
    added. This is the Module 1 input side of FR-4.2.

    Missing keys are described as absent rather than filled with a default, so a
    prompt can never carry an invented value that the record did not supply.
    """
    if not isinstance(patient_record, dict):
        raise ValueError(
            f"patient_record must be a dict, got {type(patient_record).__name__}."
        )

    clauses: list[str] = []
    for field in PATIENT_SUMMARY_FIELDS:
        if field not in patient_record:
            continue
        value = patient_record[field]
        if value is None:
            continue
        label = field.replace("_", " ")
        if isinstance(value, float):
            clauses.append(f"{label}={value:g}")
        else:
            clauses.append(f"{label}={value}")

    if not clauses:
        return (
            "No patient record fields were supplied, so no patient profile can "
            "be stated. Treat the evidence as insufficient."
        )

    joined = "; ".join(clauses)
    return f"Patient record fields as supplied by the dataset - {joined}."


#: Separator between retrieved chunks in the user prompt. Deliberately loud so a
#: reader of the assembled prompt can see exactly where Module 3 output begins
#: and ends, and cannot mistake it for text the model generated.
CHUNK_SEPARATOR = "\n\n===== RETRIEVED GUIDELINE CHUNK =====\n\n"


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

    The two blocks are explicitly labelled so the model can tell them apart. FR-4.2
    is judged by inspecting this prompt, and the requirement that retrieved
    evidence be distinguishable from model-generated interpretation is only
    demonstrable if the prompt itself marks the boundary.

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
    usable = [chunk for chunk in (guideline_chunks or []) if chunk and chunk.strip()]
    if not usable:
        raise ValueError(
            "build_prompt requires at least one retrieved guideline chunk. "
            "FR-4.2 grounds the report in Module 3 output, so an empty chunk list "
            "cannot produce a compliant prompt. Retrieve evidence first, or use "
            "INSUFFICIENT_EVIDENCE_TEMPLATE to record that none was available."
        )

    user = (
        "PATIENT CONTEXT (from the patient dataset; not generated by you):\n"
        f"{patient_summary}\n\n"
        "MODEL RISK OUTPUT (from the trained classifier; a statistical "
        "prediction, NOT a clinical outcome and NOT a confirmation of any "
        "medical fact):\n"
        f"Risk band: {risk_label}\n"
        f"P(Reaction_Score = 1): {risk_probability}\n\n"
        "RETRIEVED GUIDELINE EVIDENCE (verbatim excerpts from "
        f"{config.GUIDELINE_DOCUMENT_NAME}; quote from these only, and do not "
        "add facts from outside them):\n"
        f"{CHUNK_SEPARATOR.join(chunk.strip() for chunk in usable)}\n\n"
        "Produce the JSON object now, using only the material above. Every entry "
        "in FDA_Safety_Warnings_Cited must be copied from the retrieved evidence "
        "above."
    )

    return [("system", SYSTEM_PROMPT), ("user", user)]


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
        After both attempts fail schema validation.
    LLMCallError
        If the provider call itself failed. Deliberately **not** retried: FR-4.3's
        single retry exists to recover from malformed model output, and spending
        it on a credential or network fault would mask the real problem behind a
        misleading "validation failed" message.
    ValueError
        If ``max_retries`` is greater than 1.
    """
    # Checked before anything else, including the API key. A spec violation in the
    # caller's own arguments must not be reported as a credential problem.
    if max_retries > config.LLM_MAX_RETRIES:
        raise ValueError(
            f"max_retries={max_retries} exceeds config.LLM_MAX_RETRIES="
            f"{config.LLM_MAX_RETRIES}. FR-4.3 permits exactly one retry, so a "
            "larger budget would be a specification violation rather than a "
            "configuration choice."
        )

    messages = build_prompt(
        risk_label, risk_probability, guideline_chunks, patient_summary
    )
    system, user = messages[0][1], messages[1][1]

    attempts: list[GenerationAttempt] = []
    max_attempts = max_retries + 1
    report = None

    for attempt_number in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            raw = invoke_llm([("system", system), ("user", user)])
        except MissingAPIKeyError:
            raise
        except Exception as exc:  # noqa: BLE001 - provider faults are re-raised
            attempts.append(
                GenerationAttempt(
                    attempt_number=attempt_number,
                    succeeded=False,
                    failure_reason=f"provider call failed: {type(exc).__name__}: {exc}",
                    latency_s=time.perf_counter() - started,
                )
            )
            raise LLMCallError(
                f"The LLM provider call failed on attempt {attempt_number} of "
                f"{max_attempts}: {type(exc).__name__}: {exc}. This is a provider "
                "or credential fault, not a schema validation failure, so it was "
                "not retried and no report was produced."
            ) from exc

        latency = time.perf_counter() - started

        try:
            payload = schemas.parse_llm_json(raw)
            report = schemas.validate_report(payload)
        except (schemas.ReportParseError, ValidationError) as exc:
            attempts.append(
                GenerationAttempt(
                    attempt_number=attempt_number,
                    succeeded=False,
                    failure_reason=_summarise_validation_failure(exc),
                    latency_s=latency,
                )
            )
            continue  # FR-4.3: exactly one retry, then stop.

        attempts.append(
            GenerationAttempt(
                attempt_number=attempt_number,
                succeeded=True,
                latency_s=latency,
            )
        )
        break

    if report is None:
        raise ReportGenerationError(attempts)

    # Deterministic enforcement happens after validation and is never retried: a
    # dose bound or a missing disclaimer is a property of the report, not
    # something a second generation attempt could change.
    bound = schemas.extract_dose_bound(list(guideline_chunks))
    safety = schemas.enforce_safety_constraints(report, bound)

    return ReportRun(
        report=safety.report,
        safety=safety,
        attempts=attempts,
        dose_bound=bound,
    )


def _summarise_validation_failure(exc: Exception) -> str:
    """A short, actionable description of why a response was rejected.

    The retry prompt feeds this back to the model, so it must name the fields
    that were wrong rather than dumping a stack trace. Pydantic errors carry a
    structured ``errors()`` list, which is used where available.
    """
    if isinstance(exc, ValidationError):
        parts = []
        for error in exc.errors():
            location = ".".join(str(item) for item in error.get("loc", ())) or "<root>"
            parts.append(f"{location}: {error.get('msg', 'invalid')}")
        detail = "; ".join(parts[:6])
        if len(parts) > 6:
            detail += f"; (+{len(parts) - 6} more)"
        return f"schema validation failed - {detail}"
    return f"response could not be parsed - {exc}"


class LLMCallError(RuntimeError):
    """Raised when the provider call itself fails.

    Separate from :class:`ReportGenerationError` so a caller can tell "the model
    produced output I refused" from "the API was unreachable or rejected the
    credential". Both are failures, but they need different remediation.
    """


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
    api_key = load_api_key()

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": model,
        "api_key": api_key,
        "temperature": config.LLM_TEMPERATURE,
    }
    base_url = os.getenv("LLM_BASE_URL", "").strip()
    if base_url:
        # Optional OpenAI-compatible endpoint (Azure, a local gateway). Declared
        # in .env.example; absent by default.
        kwargs["base_url"] = base_url

    # Keyed access so an unexpected provider response surfaces as a caller error
    # rather than an opaque AttributeError.
    client = ChatOpenAI(**kwargs)

    chat = [SystemMessage(content=messages[0][1])] + [
        HumanMessage(content=content) for role, content in messages[1:] if role == "user"
    ]

    try:
        response = client.invoke(chat)
    except Exception as exc:  # noqa: BLE001 - surfaced by generate_report as LLMCallError
        # The credential is never included: some providers echo it in error text.
        raise RuntimeError(
            f"{config.LLM_PROVIDER} request to model '{model}' failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    # Some providers return a list of content blocks; join the text parts.
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        joined = "".join(part for part in parts if part)
        if joined:
            return joined

    raise RuntimeError(
        f"The model '{model}' returned no textual content (response type "
        f"{type(response).__name__})."
    )


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
    report = run.report
    lines: list[str] = [
        "# Clinical Safety Report",
        "",
        f"_Generated by {config.PROJECT_NAME} - a student engineering project. "
        f"Model: `{config.LLM_MODEL}`._",
        "",
        "## Decision",
        "",
        f"- **Eligible for vaccine:** {'Yes' if report.Eligible_For_Vaccine else 'No'}",
        f"- **Recommended dose:** {report.Recommended_Dose_mg} mg",
        "",
        "## Reasoning steps",
        "",
    ]
    if report.Reasoning_Steps:
        lines += [f"{index}. {step}" for index, step in enumerate(report.Reasoning_Steps, 1)]
    else:
        lines.append("_No reasoning steps were supplied._")
    lines += ["", "## Retrieved guideline evidence cited", ""]
    if report.FDA_Safety_Warnings_Cited:
        lines += [f"> {snippet}" for snippet in report.FDA_Safety_Warnings_Cited]
    else:
        lines.append(
            "_No guideline excerpt was cited, so this report is **ungrounded**._"
        )

    safety = getattr(run, "safety", None)
    violations = list(getattr(safety, "violations", []) or [])
    lines += ["", "## Safety checks applied", ""]
    if violations:
        lines.append(
            "Deterministic checks overrode or flagged parts of the model's output:"
        )
        lines.append("")
        for violation in violations:
            lines.append(f"- **{violation.requirement} / {violation.rule}** - {violation.detail}")
    else:
        lines.append(
            "All deterministic checks passed: the recommended dose was within the "
            "guideline bound, a guideline excerpt was cited, and the required "
            "disclaimer was present."
        )

    bound = getattr(run, "dose_bound", None)
    if bound is not None:
        lines += [
            "",
            f"- **Dose bound applied:** {bound.max_mcg} {bound.unit}"
            + (f" - source: \"{bound.source_text}\"" if bound.source_text
               else " - defaulted; no explicit ceiling was found in the retrieved text"),
        ]

    lines += [
        "",
        "## Generation detail",
        "",
        f"- Attempts: {len(run.attempts)} (FR-4.3 permits one retry)",
        f"- Total LLM latency: {run.total_latency_s:.3f} s",
    ]
    for attempt in run.attempts:
        status = "validated" if attempt.succeeded else f"failed - {attempt.failure_reason}"
        lines.append(
            f"  - attempt {attempt.attempt_number}: {status} ({attempt.latency_s:.3f} s)"
        )

    # The disclaimer is last and rendered as a blockquote so it cannot be missed
    # when the file is skimmed. FR-4.5.
    lines += [
        "",
        "---",
        "",
        f"> **{config.REQUIRED_DISCLAIMER}**",
        "",
        "The risk band above is a statistical prediction from a classifier trained "
        "on this project's dataset. It is not a clinical outcome, and a similarity "
        "score from the retrieval stage is not evidence that a statement is "
        "clinically correct.",
        "",
    ]
    return "\n".join(lines)


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
    destination = Path(reports_dir) if reports_dir else config.REPORTS_DIR
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Could not create the reports directory {destination}: {exc}"
        ) from exc

    stem = _unique_report_stem(destination, basename or run.patient_id)

    markdown = render_markdown(run)
    payload = {
        "report": run.report.model_dump(),
        "safety": run.safety.as_dict() if run.safety is not None else None,
        "generation": run.as_dict(),
        "dose_bound": _dose_bound_as_dict(run.dose_bound),
        "disclaimer": config.REQUIRED_DISCLAIMER,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    json_path = destination / f"{stem}.json"
    markdown_path = destination / f"{stem}.md"
    try:
        json_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        markdown_path.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        raise OSError(
            f"Could not write the report pair to {destination}: {exc}"
        ) from exc

    run.json_path = json_path
    run.markdown_path = markdown_path
    return json_path, markdown_path


#: Characters permitted in a report filename stem. Patient identifiers come from
#: a dataset, so they are sanitised rather than trusted as path components.
_STEM_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Runs of dots collapse to a single underscore. Without this, an identifier like
#: ``../../escape`` would sanitise to ``.._.._escape``, which is a harmless
#: filename but still reads as a traversal attempt and could confuse tooling that
#: inspects stems.
_DOT_RUNS = re.compile(r"\.{2,}")


def _unique_report_stem(directory: Path, basename: str | None) -> str:
    """Build a filename stem that cannot overwrite an earlier report.

    The stem is ``<base>-<UTC timestamp>``. If that name is already taken - two
    runs inside the same second - a ``-2``, ``-3`` suffix is appended, so
    overwriting is impossible rather than merely unlikely.
    """
    cleaned = _STEM_UNSAFE.sub("_", (basename or "").strip())
    cleaned = _DOT_RUNS.sub("_", cleaned).strip("_")
    base = cleaned or "report"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    stem = f"{base}-{stamp}"
    counter = 2
    while (directory / f"{stem}.json").exists() or (directory / f"{stem}.md").exists():
        stem = f"{base}-{stamp}-{counter}"
        counter += 1
    return stem


def _dose_bound_as_dict(bound: Any) -> Any:
    """Serialise a :class:`~src.llm.schemas.DoseBound`, or pass ``None`` through."""
    if bound is None:
        return None
    return {
        "max_mcg": getattr(bound, "max_mcg", None),
        "max_mg": getattr(bound, "max_mg", None),
        "unit": getattr(bound, "unit", None),
        "source_text": getattr(bound, "source_text", ""),
        "derived_from_guideline": bool(getattr(bound, "source_text", "")),
    }


def record_generation_latency(run: ReportRun, path: Any = None) -> Any:
    """Append this run's latency to ``outputs/logs/latency_log.json``.

    Feeds the LLM half of dashboard Panel C, which compares retrieval latency
    against report-generation latency (PRD Section 8.4). Called by
    :mod:`src.pipeline`, not by the dashboard.

    Two files are written because two different readers exist:

    * ``generation_log.json`` - the file ``load_latency_panel_data`` reads for its
      ``LLM Report`` bar. Entries carry ``latency_s``, appended so repeated runs
      accumulate.
    * ``latency_log.json`` - the per-stage pipeline timing file, where this run is
      recorded as one more ``stages`` entry alongside the other modules.

    Neither is required for the report itself, so a failure to write is raised
    rather than swallowed: a latency that was not recorded is a silent regression
    in Panel C's usefulness.

    Parameters
    ----------
    run:
        Output of :func:`generate_report`.
    path:
        Overrides the ``generation_log.json`` destination, for tests.

    Returns
    -------
    pathlib.Path
        The generation log that was appended to.
    """
    generation_path = Path(path) if path else config.GENERATION_LOG_JSON_PATH

    entry = {
        "attempts": len(run.attempts),
        "latency_s": round(run.total_latency_s, 4),
        "model": config.LLM_MODEL,
        "eligible": bool(getattr(run.report, "Eligible_For_Vaccine", None)),
        "compliant": bool(getattr(run.safety, "is_compliant", False)),
        "violations": len(getattr(run.safety, "violations", []) or []),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _append_json_entries(generation_path, entry, key="entries")

    _append_json_entries(
        config.LATENCY_LOG_JSON_PATH,
        {
            "stage": "llm_report",
            "seconds": round(run.total_latency_s, 4),
            "ok": True,
            "attempts": len(run.attempts),
            "recorded_at": entry["recorded_at"],
        },
        key="stages",
    )
    return generation_path


def _append_json_entries(path: Path, entry: dict[str, Any], key: str) -> None:
    """Append ``entry`` to ``path`` under ``key``, creating the file if absent.

    An existing file that cannot be parsed is replaced rather than appended to
    blindly, because failing here would abort a report run that already succeeded.
    The prior contents are preserved when they are readable.
    """
    payload: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except (json.JSONDecodeError, OSError):
            payload = {}

    existing = payload.get(key)
    payload[key] = list(existing) + [entry] if isinstance(existing, list) else [entry]
    payload.setdefault("disclaimer", config.REQUIRED_DISCLAIMER)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")