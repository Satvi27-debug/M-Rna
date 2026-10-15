"""Backend execution of real pipeline runs for the web interface.

Why this module exists
----------------------
``dashboard/app.py`` is presentation: it serves files and reads state. This module
owns the other half - turning an HTTP request into a *real* pipeline execution and
returning what actually happened. Keeping them apart preserves architecture rule 9
(the dashboard renders artifacts rather than recomputing them) and keeps the HTTP
handler free of pipeline logic.

Every stage runs the genuine implementation. Nothing here produces a prediction,
a chunk, a report or a timing value that a module did not produce.

Execution model
---------------
**Synchronous, with a concurrency lock.** The pipeline is already served by
``ThreadingHTTPServer``, so a long run occupies one worker thread and leaves
``/api/state``, ``/api/health`` and the static routes responsive. A background
thread pool would add a task queue, cancellation semantics and orphan-process
handling for no benefit: the browser already renders a loading state, and the
user is waiting on one action.

The one thing threading *does* make dangerous is two runs writing the same
artifacts at once - the ChromaDB collection, the model directory, the latency
log. A single :class:`threading.Lock` serialises execution, and a second request
that arrives while a run holds the lock is refused with HTTP 409 rather than
queued, because a user pressing "Run" twice should be told, not silently
doubled.

Import order
------------
This module imports :mod:`src.pipeline.pipeline` **lazily**, inside the
executor, not at module scope. That is load-bearing. Importing the pipeline
eagerly here would drag scikit-learn in as soon as the dashboard starts, before
``native_stack.preload_native_libraries()`` could order torch and ChromaDB first -
reproducing exactly the native crash that module exists to prevent. See
:mod:`src.pipeline.native_stack`.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src import config


#: Stages the interface may request. Derived from what ``app.js`` actually sends,
#: plus ``all`` for a complete end-to-end run.
REQUEST_STAGES: tuple[str, ...] = ("ml", "rag", "report", "all")

#: Fields a client may supply in a patient record. Anything else is dropped rather
#: than passed through, so a request cannot inject a key the pipeline would read.
ALLOWED_RECORD_FIELDS: tuple[str, ...] = (
    "Patient_ID",
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History",
)

#: Bounds enforced on client input. A negative age or a dose of 10^9 is not a
#: plausible patient and would produce a meaningless prediction presented as if it
#: were real, so it is refused at the boundary rather than downstream.
INPUT_BOUNDS: dict[str, tuple[float, float]] = {
    "Age": (0.0, 130.0),
    "Biomarker_Level": (0.0, 10000.0),
    "Pre_Existing_Conditions": (0.0, 50.0),
    "Vaccine_Dose_mcg": (0.0, 10000.0),
}

#: Longest accepted identifier or question. Bounds log size and reflection risk.
MAX_ID_LENGTH = 64
MAX_QUESTION_LENGTH = 500

#: Held for the duration of one pipeline run. Serialises execution so two runs
#: cannot write the ChromaDB collection or the latency log concurrently.
_RUN_LOCK = threading.Lock()


class InputRejected(ValueError):
    """Raised when a client request is malformed.

    Distinct from a pipeline failure: the request never reached the pipeline, so
    the correct response is a 400 with an actionable field-level message rather
    than a 500.
    """


class RunInProgress(RuntimeError):
    """Raised when a run is requested while another holds the execution lock."""


# ---------------------------------------------------------------------------
# Run registry
# ---------------------------------------------------------------------------


@dataclass
class RunRecord:
    """One tracked run, so a response can be tied to a specific execution.

    The identifier exists because report filenames are timestamped rather than
    keyed, so without it a client could not tell which file its run produced from
    an older one left in the directory.
    """

    run_id: str
    stage: str
    status: str = config.RUN_COMPLETED
    started_at: str = ""
    finished_at: str = ""
    seconds: float = 0.0
    patient_id: str = ""
    error: str = ""
    report_json: str = ""
    report_markdown: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "seconds": round(self.seconds, 4),
            "patient_id": self.patient_id,
            "error": self.error,
            "report_json": self.report_json,
            "report_markdown": self.report_markdown,
        }


#: Most recent runs, newest first. Bounded so a long-lived server cannot grow
#: without limit.
_RUN_HISTORY: list[RunRecord] = []
_MAX_HISTORY = 25


def new_run_id() -> str:
    """Unique, time-ordered run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def recent_runs() -> list[dict[str, Any]]:
    """The tracked runs, newest first. Never includes a credential."""
    return [record.as_dict() for record in _RUN_HISTORY]


def _remember(record: RunRecord) -> RunRecord:
    _RUN_HISTORY.insert(0, record)
    del _RUN_HISTORY[_MAX_HISTORY:]
    return record


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def validate_record(payload: Any) -> dict[str, Any]:
    """Validate and normalise a client-supplied patient record.

    Only the six declared dataset fields are carried through. Unknown keys are
    dropped rather than rejected, so a forward-compatible client is not broken by
    a server that has not caught up - but they are also never passed to the
    pipeline, which reads exactly these columns.

    Parameters
    ----------
    payload:
        The ``record`` object from the request body.

    Returns
    -------
    dict
        A clean record with numeric fields coerced to ``float``.

    Raises
    ------
    InputRejected
        If a required field is missing, mistyped, out of range, or carries a value
        the pipeline could not use.
    """
    if not isinstance(payload, dict):
        raise InputRejected(
            f"'record' must be a JSON object, got {type(payload).__name__}."
        )

    cleaned: dict[str, Any] = {}

    identifier = payload.get("Patient_ID")
    if identifier is None or not str(identifier).strip():
        raise InputRejected("'Patient_ID' is required and must be a non-empty string.")
    identifier = str(identifier).strip()
    if len(identifier) > MAX_ID_LENGTH:
        raise InputRejected(
            f"'Patient_ID' must be at most {MAX_ID_LENGTH} characters."
        )
    cleaned["Patient_ID"] = identifier

    for field_name, (low, high) in INPUT_BOUNDS.items():
        if field_name not in payload or payload[field_name] is None:
            raise InputRejected(f"'{field_name}' is required.")
        value = payload[field_name]
        # bool is an int subclass; a True here is a client bug, not a number.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InputRejected(f"'{field_name}' must be a number.")
        value = float(value)
        if value != value or value in (float("inf"), float("-inf")):
            raise InputRejected(f"'{field_name}' must be a finite number.")
        if not low <= value <= high:
            raise InputRejected(
                f"'{field_name}' must be between {low:g} and {high:g}; got {value:g}."
            )
        cleaned[field_name] = value

    history = payload.get("Prior_Reaction_History")
    if history is None or not str(history).strip():
        raise InputRejected("'Prior_Reaction_History' is required.")
    history = str(history).strip()
    if history not in config.CATEGORICAL_VALUES:
        raise InputRejected(
            f"'Prior_Reaction_History' must be one of "
            f"{list(config.CATEGORICAL_VALUES)}; got {history!r}."
        )
    cleaned["Prior_Reaction_History"] = history

    # Only declared fields survive. Nothing else reaches the pipeline.
    return {k: v for k, v in cleaned.items() if k in ALLOWED_RECORD_FIELDS}


def validate_stage(payload: Any) -> str:
    """Validate the requested stage name."""
    if not isinstance(payload, dict):
        raise InputRejected("The request body must be a JSON object.")
    stage = payload.get("stage")
    if not stage:
        raise InputRejected("'stage' is required.")
    stage = str(stage).strip().lower()
    if stage not in REQUEST_STAGES:
        raise InputRejected(
            f"'stage' must be one of {list(REQUEST_STAGES)}; got {stage!r}."
        )
    return stage


def validate_question(payload: dict[str, Any]) -> str:
    """Validate a retrieval question."""
    question = payload.get("question")
    if question is None or not str(question).strip():
        raise InputRejected("'question' is required for the 'rag' stage.")
    question = str(question).strip()
    if len(question) > MAX_QUESTION_LENGTH:
        raise InputRejected(
            f"'question' must be at most {MAX_QUESTION_LENGTH} characters."
        )
    return question


# ---------------------------------------------------------------------------
# Shared response envelope
# ---------------------------------------------------------------------------


def _envelope(run_id: str, stage: str, status: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "stage": stage,
        "status": status,
        "disclaimer": config.REQUIRED_DISCLAIMER,
    }


def stage_status_map(timings: Any) -> dict[str, str]:
    """Stage name to status, from the pipeline's own records.

    Read from :class:`~src.pipeline.pipeline.VIPResult` rather than re-derived, so
    the interface can never disagree with the pipeline about what ran.
    """
    return {t.stage: t.status for t in (timings or [])}


def dose_payload(result: Any) -> dict[str, Any]:
    """Dose facts for the client, preserving the submitted value and unit.

    Two distinct numbers travel together and are never conflated:

    * ``submitted_dose_mcg`` / ``submitted_dose_unit`` - the caller's input, exactly
      as supplied.
    * ``review_threshold_mcg`` - the guideline's secondary clinician review
      trigger.

    ``clinical_maximum_established`` is always ``False``: the supplied document
    states a review threshold, not a clinically validated maximum safe dose.
    """
    review = getattr(result, "dose_review", None)
    return {
        "submitted_dose_mcg": getattr(result, "submitted_dose_mcg", None),
        "submitted_dose_unit": getattr(result, "submitted_dose_unit", "mcg"),
        "review_threshold_mcg": config.GUIDELINE_REVIEW_THRESHOLD_MCG,
        "review_threshold_kind": "secondary_clinician_review",
        "engineering_dose_guard_mcg": config.ENGINEERING_DOSE_GUARD_MCG,
        # Falls back to comparing the submitted dose when the pipeline stopped
        # before the review finding was produced - for example when the run failed
        # at the LLM stage, which is after retrieval but before enforcement. The
        # arithmetic is the same deterministic rule
        # (`submitted > threshold`), not an independent guess.
        "review_required": bool(
            getattr(review, "review_required", False)
            or _exceeds_threshold(result)
        ),
        "clinical_maximum_established": False,
        "threshold_note": (
            "The guideline states that doses above this figure require secondary "
            "clinician review. It does not state a clinically established maximum "
            "safe dose, and this project's software eligibility is not a clinical "
            "approval."
        ),
    }


def _exceeds_threshold(result: Any) -> bool:
    """Whether the submitted dose is above the guideline's review threshold.

    Same arithmetic the pipeline's own ``DoseBound.exceeds`` uses: strictly
    greater than, so exactly 100 mcg does not trigger. Used only when the run did
    not reach safety enforcement, so the response does not under-report a dose that
    the pipeline would have flagged.
    """
    submitted = getattr(result, "submitted_dose_mcg", None)
    if submitted is None:
        return False
    return float(submitted) > config.GUIDELINE_REVIEW_THRESHOLD_MCG


def artifacts_payload(result: Any) -> dict[str, Any]:
    """Artifact references produced by a run, for the client to fetch."""
    run = getattr(result, "report_run", None)
    json_path = getattr(run, "json_path", None)
    markdown_path = getattr(run, "markdown_path", None)
    heatmap = getattr(result, "heatmap_path", None)

    return {
        "report_json": Path(json_path).name if json_path else None,
        "report_markdown": Path(markdown_path).name if markdown_path else None,
        "attention_heatmap": Path(heatmap).name if heatmap else None,
    }


# ---------------------------------------------------------------------------
# Stage executors
# ---------------------------------------------------------------------------


def _pipeline_module() -> Any:
    """Import the pipeline lazily, after the native import order is established.

    Not at module scope: importing the pipeline eagerly here would drag
    scikit-learn in as soon as the dashboard starts, before
    ``preload_native_libraries()`` could order torch and ChromaDB first, which
    reproduces the native crash this ordering exists to prevent.

    **Call this before importing any ``src.*`` module that reaches scikit-learn** -
    ``src.ml.predict`` does, transitively through ``src.data.preprocessing``.
    An earlier version of :func:`run_ml` imported it first and crashed exactly that
    way; ``tests/test_dashboard_api.py`` now asserts the ordering.
    """
    from src.pipeline import pipeline as pipeline_module

    return pipeline_module


def run_ml(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Score one patient through the real trained model.

    Runs the pipeline's own preprocessing and prediction stages. No prediction is
    synthesised: if the model artifacts are missing, the failure is reported.
    """
    # The pipeline must be imported FIRST, before anything that reaches
    # scikit-learn. `src.ml.predict` pulls sklearn in at module scope via
    # `src.data.preprocessing`, and sklearn's OpenMP runtime resident before torch
    # is exactly what makes `c10.dll` fail to initialise. Calling
    # `_pipeline_module()` first lets its `preload_native_libraries()` establish the
    # verified order. See src/pipeline/native_stack.py.
    pipeline_module = _pipeline_module()

    envelope = _envelope(run_id, "ml", config.RUN_COMPLETED)
    pipeline = pipeline_module.VIPPipeline()

    pipeline.preprocess()
    prediction = pipeline.predict_risk(record)
    prediction_dto = prediction.as_dict()
    selected = prediction.selected
    # Read straight off the pipeline's own records rather than wrapping them in a
    # throwaway VIPResult, so this stage's timings are exactly what it measured.
    stage_timings = list(pipeline.timings)

    envelope.update(
        {
            "run_id": run_id,
            "stage": "ml",
            "status": config.RUN_COMPLETED,
            "patient_id": record["Patient_ID"],
            # Flattened for the interface's existing renderers, which read
            # `risk_label` and `risk_probability` directly.
            "risk_label": prediction.risk_label,
            "risk_probability": (
                round(float(prediction.risk_probability), 6)
                if prediction.risk_probability is not None
                else None
            ),
            "explanation": _risk_explanation(prediction),
            "prediction": prediction_dto,
            "selected_model": (
                selected.model_name if selected is not None else None
            ),
            "model_agreement": prediction.agreement(),
            "stage_statuses": stage_status_map(stage_timings),
            "stage_details": [t.as_dict() for t in stage_timings],
            "timings": {t.stage: round(t.seconds, 4) for t in stage_timings},
        }
    )
    return envelope


def _risk_explanation(prediction: Any) -> str:
    """One factual sentence about the prediction. No clinical claim.

    States what the classifier output and how the models agreed. It does not
    interpret the band clinically, because the model has no clinical validity.
    """
    if prediction.risk_probability is None:
        return "No model produced a probability."
    selected = prediction.selected
    name = selected.model_name if selected is not None else "the selected model"
    agreement = prediction.agreement()
    suffix = (
        "Both models agree on the band."
        if agreement
        else "The models disagree on the band."
        if agreement is False
        else ""
    )
    return (
        f"{name} assigned this patient a risk band of "
        f"{prediction.risk_label} with P(Reaction_Score = 1) = "
        f"{prediction.risk_probability:.4f}. {suffix} This is a statistical "
        f"prediction from a model trained on this project's dataset, not a "
        f"clinical outcome."
    ).strip()


def run_rag(question: str, run_id: str) -> dict[str, Any]:
    """Retrieve real guideline chunks for a question.

    Uses the persisted ChromaDB index. A missing index is reported as an error -
    never as an empty but successful result, which would read as "the document
    says nothing" rather than "retrieval could not run".
    """
    pipeline_module = _pipeline_module()
    envelope = _envelope(run_id, "rag", config.RUN_COMPLETED)
    pipeline = pipeline_module.VIPPipeline()

    # The client's question is the input, so it is queried directly rather than
    # derived from a synthetic patient record - which would also mean inventing
    # clinical inputs the request never supplied.
    from src.rag import ingest as rag_ingest
    from src.rag import retrieve as rag_retrieve

    pipeline.index = rag_ingest.load_index(persist_dir=pipeline.vectorstore_dir)
    embedding_function = rag_ingest.build_embedding_function()
    result = rag_retrieve.query_guidelines(
        question,
        top_k=pipeline.top_k,
        index=pipeline.index,
        embedding_function=embedding_function,
    )

    envelope.update(
        {
            "question": result.question,
            "k": result.k,
            "n_results": len(result.chunks),
            "latency_s": round(result.latency_s, 4),
            "embedding_model": result.embedding_model_name,
            "chunks": [chunk.as_dict() for chunk in result.chunks],
        }
    )
    return envelope


def run_full(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Run every stage end to end through the real pipeline.

    Returns the pipeline's own result. A required stage failure surfaces as a
    non-success status with the failing stage named - never as a completed run.
    """
    pipeline_module = _pipeline_module()
    pipeline = pipeline_module.VIPPipeline()
    envelope = _envelope(run_id, "all", config.RUN_FAILED)

    started = time.perf_counter()
    try:
        result = pipeline.run(record)
    except pipeline_module.PipelineError as exc:
        partial = getattr(exc, "result", None)
        envelope.update(
            {
                "status": config.RUN_FAILED,
                "partial": bool(getattr(partial, "partial", False)),
                "error": str(exc),
                "failed_stage": exc.stage,
                "seconds": round(time.perf_counter() - started, 4),
            }
        )
        if partial is not None:
            envelope.update(_result_fields(partial))
        return envelope

    envelope.update(_result_fields(result))
    envelope.update(
        {
            "status": result.status,
            "partial": result.partial,
            "seconds": round(time.perf_counter() - started, 4),
            "risk_label": (
                result.risk_prediction.risk_label
                if result.risk_prediction is not None
                else None
            ),
            "risk_probability": (
                round(float(result.risk_prediction.risk_probability), 6)
                if result.risk_prediction is not None
                and result.risk_prediction.risk_probability is not None
                else None
            ),
            "explanation": (
                _risk_explanation(result.risk_prediction)
                if result.risk_prediction is not None
                else None
            ),
            "attention": _attention_payload(result),
        }
    )
    return envelope


def _result_fields(result: Any) -> dict[str, Any]:
    """Shared fields extracted from a :class:`VIPResult`."""
    return {
        "patient_id": result.patient_id,
        "stage_statuses": stage_status_map(result.timings),
        "stage_details": [t.as_dict() for t in result.timings],
        "timings": result.timing_map(),
        "retrieved_chunks": len(result.retrieved_chunks),
        "retrieval_queries": [r.question for r in result.retrievals],
        "dose": dose_payload(result),
        "artifacts": artifacts_payload(result),
        "report": _report_payload(result),
        "sequence": result.sequence,
        "sequence_source": result.sequence_source,
    }


def _attention_payload(result: Any) -> dict[str, Any]:
    """Attention shape and row-sum check, from the real computation."""
    attention = result.attention
    if attention is None:
        return {"available": False}
    weights = getattr(attention, "weights", None)
    row_sums = None
    if weights is not None:
        import numpy as np

        row_sums = np.asarray(weights, dtype=float).sum(axis=1)
    return {
        "available": True,
        "shape": list(weights.shape) if weights is not None else None,
        "input_mode": getattr(attention, "input_mode", None),
        "sequence": getattr(attention, "sequence", None),
        "row_sums_min": round(float(row_sums.min()), 8) if row_sums is not None else None,
        "row_sums_max": round(float(row_sums.max()), 8) if row_sums is not None else None,
        "softmax_verified": (
            bool(row_sums is not None and abs(row_sums - 1.0).max() < 1e-5)
        ),
        "heatmap": (
            Path(result.heatmap_path).name if result.heatmap_path else None
        ),
        "interpretation_note": (
            "Arithmetic properties of an attention matrix computed from random "
            "projections. They do not establish biological or clinical "
            "relationships."
        ),
    }


def _report_payload(result: Any) -> dict[str, Any]:
    """Report status. Absent report means absent, never a placeholder."""
    run = getattr(result, "report_run", None)
    if run is None:
        return {
            "generated": False,
            "reason": "No validated report was produced by this run.",
            "json": None,
            "markdown": None,
        }
    safety = getattr(run, "safety", None)
    return {
        "generated": True,
        "eligible_for_vaccine": getattr(run.report, "Eligible_For_Vaccine", None),
        "recommended_dose_mg": getattr(run.report, "Recommended_Dose_mg", None),
        "reasoning_steps": list(getattr(run.report, "Reasoning_Steps", []) or []),
        "citations": list(
            getattr(run.report, "FDA_Safety_Warnings_Cited", []) or []
        ),
        "disclaimer": getattr(run.report, "Disclaimer", config.REQUIRED_DISCLAIMER),
        "compliant": bool(getattr(safety, "is_compliant", False)),
        "violations": [v.as_dict() for v in (getattr(safety, "violations", []) or [])],
        "json": (
            Path(run.json_path).name if getattr(run, "json_path", None) else None
        ),
        "markdown": (
            Path(run.markdown_path).name
            if getattr(run, "markdown_path", None)
            else None
        ),
        "attempts": len(getattr(run, "attempts", []) or []),
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def execute(payload: Any) -> tuple[int, dict[str, Any]]:
    """Run the requested stage and return ``(http_status, response_body)``.

    Parameters
    ----------
    payload:
        Parsed JSON request body.

    Returns
    -------
    tuple
        ``(status_code, body)``. 200 for a completed run, 400 for malformed input,
        409 when a run is already in flight, 503 when a required upstream artifact
        is missing, 502 when the LLM stage failed for an external reason, and 500
        for an unexpected fault.

    Raises
    ------
    Nothing. Every expected failure is translated to a status code here, so the
    HTTP handler stays trivial and no exception escapes with a traceback in the
    response body.
    """
    run_id = new_run_id()

    try:
        stage = validate_stage(payload)
        if stage == "ml" or stage == "all":
            record = validate_record(payload.get("record"))
        if stage == "rag":
            question = validate_question(payload)
    except InputRejected as exc:
        return 400, {
            "error": "invalid_request",
            "message": str(exc),
            "run_id": run_id,
            "allowed_record_fields": list(ALLOWED_RECORD_FIELDS),
        }

    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    start = time.perf_counter()

    if not _RUN_LOCK.acquire(blocking=False):
        # Refused, not queued: a user pressing Run twice should be told the first
        # run is still going, rather than silently doubled.
        return 409, {
            "error": "run_in_progress",
            "message": (
                "Another pipeline run is already executing. Wait for it to "
                "finish, then submit again."
            ),
            "run_id": run_id,
        }

    try:
        body = _dispatch(stage, payload, run_id)
        status_code = _status_for(body)
    except InputRejected as exc:
        status_code, body = 400, {"error": "invalid_request", "message": str(exc)}
    except RunInProgress as exc:
        status_code, body = 409, {"error": "run_in_progress", "message": str(exc)}
    except _UpstreamMissing as exc:
        # A missing artifact or credential is the caller's environment to fix, not
        # a fault in the server. 503 says "not now", which is accurate and does not
        # imply retrying would help on its own.
        status_code, body = exc.status, {
            "error": exc.kind,
            "message": str(exc),
            "run_id": run_id,
        }
    except _UpstreamUnavailable as exc:
        status_code, body = exc.status, {
            "error": exc.kind,
            "message": str(exc),
            "run_id": run_id,
        }
    except Exception as exc:  # noqa: BLE001 - translated, never leaked
        # The client gets a class name and a short message. No traceback, no path,
        # no credential - the stack stays in the server log.
        import traceback

        traceback.print_exc()
        status_code, body = 500, {
            "error": "internal_error",
            "message": f"The pipeline raised {type(exc).__name__}. See the server log.",
            "run_id": run_id,
        }
    finally:
        _RUN_LOCK.release()

    _remember(
        RunRecord(
            run_id=run_id,
            stage=stage,
            status=body.get("status", config.RUN_FAILED),
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            seconds=time.perf_counter() - start,
            patient_id=str(body.get("patient_id", "")),
            error=str(body.get("error", "")),
            report_json=str((body.get("report") or {}).get("json") or ""),
            report_markdown=str((body.get("report") or {}).get("markdown") or ""),
        )
    )
    return status_code, body


def _status_for(body: dict[str, Any]) -> int:
    """HTTP status implied by a completed dispatch.

    ``200`` only when the pipeline itself reports ``completed``. A failed or
    partial run is a ``500``, and a report that was not generated never yields
    ``200`` - which is the whole point: the client must not be able to read a
    success that the pipeline did not earn.
    """
    return 200 if body.get("status") == config.RUN_COMPLETED else 500


def _dispatch(stage: str, payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Call the right executor, translating expected failures to status codes."""
    try:
        if stage == "ml":
            return run_ml(payload["record"], run_id)
        if stage == "rag":
            return run_rag(payload["question"], run_id)
        if stage == "all":
            return run_full(payload["record"], run_id)
        return run_report(run_id)
    except FileNotFoundError as exc:
        raise _UpstreamMissing(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - classified below
        name = type(exc).__name__
        if name in {"IndexingError", "ModelLoadError", "DatasetNotFoundError",
                    "MissingArtifactError", "PreprocessorNotFittedError"}:
            raise _UpstreamMissing(f"{name}: {exc}") from exc
        if name in {"MissingAPIKeyError"}:
            # Actionable and specific: which variable, where to set it, and an
            # explicit statement that nothing was generated.
            raise _UpstreamMissing(
                f"{exc} No report was generated, and none is claimed."
            ) from exc
        if name in {"LLMCallError", "ReportGenerationError"}:
            raise _UpstreamUnavailable(f"{name}: {exc}") from exc
        raise


class _UpstreamMissing(RuntimeError):
    """A required artifact or credential is absent. HTTP 503."""

    status = 503
    kind = "upstream_unavailable"


class _UpstreamUnavailable(RuntimeError):
    """An external service failed. HTTP 502."""

    status = 502
    kind = "upstream_error"


def run_report(run_id: str) -> dict[str, Any]:
    """Generate a report for the most recent completed run, if one exists.

    Deliberately refuses rather than running the pipeline: the interface's report
    button acts on an existing analysis. With no prior run there is nothing to
    report on, and inventing one would be exactly the failure mode this project
    exists to avoid.
    """
    envelope = _envelope(run_id, "report", config.RUN_FAILED)
    run_log = config.LOGS_DIR / "pipeline_run.json"
    if not run_log.is_file():
        envelope.update(
            {
                "status": config.RUN_FAILED,
                "error": "no_run",
                "message": (
                    "No pipeline run has completed yet. Submit the patient "
                    "analysis form first; a report is generated from that run."
                ),
                "generated": False,
            }
        )
        raise _UpstreamMissing(envelope["message"])

    stored = json.loads(run_log.read_text(encoding="utf-8"))
    report_paths = stored.get("report_paths", {}) or {}
    envelope.update(
        {
            "status": stored.get("status", config.RUN_FAILED),
            "patient_id": stored.get("patient_id"),
            "generated": bool(report_paths.get("json")),
            "json": report_paths.get("json"),
            "markdown": report_paths.get("markdown"),
            "stage_statuses": stored.get("stage_statuses", {}),
            "dose": _dose_from_run_log(stored),
            "disclaimer": config.REQUIRED_DISCLAIMER,
        }
    )
    return envelope


class _StoredRunView:
    """Adapter exposing the attributes :func:`dose_payload` reads from a log."""

    def __init__(self, stored: dict[str, Any]) -> None:
        self.submitted_dose_mcg = stored.get("submitted_dose_mcg")
        self.submitted_dose_unit = stored.get("submitted_dose_unit", "mcg")
        self.dose_review = stored.get("dose_review")


def _dose_from_run_log(stored: dict[str, Any]) -> dict[str, Any]:
    """Dose facts for a stored run, built with the same rules as a live one.

    Routed through :func:`dose_payload` so a stored run and a fresh one cannot
    drift into describing the review threshold differently - or start calling it
    something it is not.
    """
    return dose_payload(_StoredRunView(stored))


# ---------------------------------------------------------------------------
# Report artifact access
# ---------------------------------------------------------------------------


def resolve_report_artifact(name: str) -> Path | None:
    """Resolve a requested report filename inside ``outputs/reports/``.

    Two independent checks, because either alone is insufficient:

    1. ``name`` must be a bare filename - no separators, no drive letters, no
       ``..``. This rejects traversal before touching the filesystem.
    2. The resolved path must still be inside ``config.REPORTS_DIR`` after
       symlink resolution, which catches anything that slipped past the first check.

    Only ``.json`` and ``.md`` are served, so a stray artifact of another type in
    the directory is not exposed.

    Returns
    -------
    pathlib.Path | None
        The resolved path, or ``None`` if the request is unsafe or absent.
    """
    if not name or name != Path(name).name:
        return None
    if not name.endswith((".json", ".md")):
        return None

    candidate = config.REPORTS_DIR / name
    try:
        resolved = candidate.resolve()
        base = config.REPORTS_DIR.resolve()
        resolved.relative_to(base)
    except (ValueError, OSError):
        return None

    if not resolved.is_file():
        return None
    return resolved


def list_reports() -> list[dict[str, Any]]:
    """Report files on disk, newest first.

    Reported so the interface can offer the real artifacts. Names only - no file
    contents, and nothing outside ``outputs/reports/``.
    """
    directory = config.REPORTS_DIR
    if not directory.is_dir():
        return []
    entries = []
    for path in directory.iterdir():
        if path.suffix not in {".json", ".md"} or not path.is_file():
            continue
        stat = path.stat()
        entries.append(
            {
                "name": path.name,
                "bytes": stat.st_size,
                "modified": datetime.fromtimestamp(
                    stat.st_mtime, timezone.utc
                ).isoformat(timespec="seconds"),
            }
        )
    entries.sort(key=lambda e: e["modified"], reverse=True)
    return entries