"""End-to-end orchestration of the mRNA VIP platform.

This module is the integration layer described in PRD Section 3.1 and
architecture rule 8. It owns no algorithm of its own: each stage delegates to the
package that implements it, and this file decides the order, the data contract
between stages, and the error handling.

Stage order
-----------
1. **Preprocessing** - :mod:`src.data.preprocessing`. Loads, validates, imputes,
   encodes, scales and splits the patient dataset (FR-1.1 ... FR-1.4).
2. **ML risk prediction** - :mod:`src.ml.predict`. Loads the serialized model
   and scores the patient under review to a risk band and probability
   (FR-1.5 ... FR-1.7).
3. **Attention** - :mod:`src.attention.self_attention`. Computes self-attention
   weights over the patient's mRNA sequence and saves the heatmap (FR-2.1 ...
   FR-2.5).
4. **RAG retrieval** - :mod:`src.rag.retrieve`. Retrieves the guideline chunks
   relevant to the patient's profile - age band, dose, biomarker, pre-existing
   conditions and prior reaction history (FR-3.4, FR-3.5).
5. **LLM report** - :mod:`src.llm.report_generator`. Builds a grounded prompt,
   validates the structured response against the Pydantic schema, applies the
   deterministic safety rules and persists the JSON/Markdown pair (FR-4.1 ...
   FR-4.6).
6. **Outputs** - reports, figures and logs are written for the dashboard to read
   back (FR-5.2).

Why retrieval queries are built from the patient profile
--------------------------------------------------------
Step 4 issues one query per clinical dimension (age band, dose band,
contraindications, biomarker, prior reaction, documentation). The patient is not
a single question: a 72-year-old with two pre-existing conditions matches two
different guideline passages, and a single combined query tends to retrieve only
the passage that dominates the wording. Separate queries make each citation
traceable to the clinical factor that triggered it.

Latency capture
---------------
Every stage is timed and recorded on :class:`VIPResult.timings`, because PRD
Section 8.4 makes per-stage latency the content of dashboard Panel C. Timing is
therefore part of the pipeline contract, not debug output.

Status
------
Fully implemented. Run ``python -m src.pipeline.pipeline --run``, or
``python -m src.pipeline.pipeline --all`` for the complete workflow including the
figure. Report generation needs ``LLM_API_KEY``; every other stage runs locally.

Import order in this file is load-bearing - see the comment above
:func:`src.pipeline.native_stack.preload_native_libraries`. Reordering the imports
here will crash the process natively rather than raise.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config

# ---------------------------------------------------------------------------
# NATIVE IMPORT ORDER - MUST NOT BE REORDERED
# ---------------------------------------------------------------------------
# scikit-learn pulls in an OpenMP/MKL runtime that breaks torch and ChromaDB if it
# is resident first: torch then fails with ``WinError 1114`` on ``c10.dll`` and
# ChromaDB's Rust HNSW extension dies with an uncatchable ``0xC0000005``.
#
# Loading torch and ChromaDB *before* scikit-learn makes all four modules coexist
# in one interpreter, verified across repeated fresh-process runs. See
# ``src/pipeline/native_stack.py`` for the full matrix and the diagnostic command.
#
# This call must precede EVERY import that reaches scikit-learn - including
# ``src.data.preprocessing``, which builds a ``ColumnTransformer``. If a linter or
# a well-meaning edit moves any of them above this line, the pipeline crashes
# natively instead of raising, so the ordering is load-bearing and commented as
# such. ``native_stack`` itself imports only ``src.config``, which is safe.
from src.pipeline.native_stack import preload_native_libraries

preload_native_libraries()

from src.data import preprocessing  # noqa: E402  - order is load-bearing
from src.ml import evaluate as ml_evaluate  # noqa: E402
from src.ml import predict as ml_predict  # noqa: E402
from src.ml import train as ml_train  # noqa: E402
from src.attention import self_attention  # noqa: E402
from src.rag import ingest as rag_ingest  # noqa: E402
from src.rag import retrieve as rag_retrieve  # noqa: E402
from src.llm import report_generator as llm_report  # noqa: E402
from src.llm import schemas as llm_schemas  # noqa: E402


class PipelineError(RuntimeError):
    """Raised when a stage cannot complete.

    The stage name is carried so a failure in a six-step flow is immediately
    attributable without reading a traceback.

    ``result`` carries the partially-populated :class:`VIPResult` when the failure
    happened mid-run, so the stages that *did* complete are still inspectable.
    Losing a real risk prediction because the LLM credential was absent would be a
    regression in usefulness - provided the caller can see the run was incomplete,
    which is why the result's status is ``failed``, never ``completed``.
    """

    def __init__(self, stage: str, cause: BaseException, result: Any = None) -> None:
        self.stage = stage
        self.cause = cause
        self.result = result
        super().__init__(f"Pipeline stage '{stage}' failed: {cause}")


#: Ordered stage names, used for the timing log and for the CLI summary.
PIPELINE_STAGES: tuple[str, ...] = (
    "preprocessing",
    "ml_prediction",
    "attention",
    "rag_retrieval",
    "llm_report",
    "persist",
)


@dataclass
class StageTiming:
    """Wall-clock duration and outcome of one stage.

    Attributes
    ----------
    stage:
        Stage name from :data:`PIPELINE_STAGES`.
    seconds:
        Wall-clock duration.
    ok:
        Kept for backward compatibility. ``True`` when ``status`` is
        ``completed``; the inverse of "failed" otherwise.
    detail:
        Human-readable note. On failure, the actionable message - never a
        credential and never a full patient record.
    status:
        One of :data:`config.STAGE_STATUSES`.
    error:
        The exception class name when the stage failed, so a caller can branch on
        the failure kind without parsing ``detail``.
    required:
        Whether the stage is required for a completed run. A failure in a
        required stage can never produce :data:`config.RUN_COMPLETED`.
    """

    stage: str
    seconds: float
    ok: bool = True
    detail: str = ""
    status: str = config.STAGE_COMPLETED
    error: str = ""
    required: bool = True

    def __post_init__(self) -> None:
        # Keep `ok` consistent with `status` without requiring callers to set both.
        self.ok = self.status == config.STAGE_COMPLETED

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "seconds": round(self.seconds, 4),
            "ok": self.ok,
            "status": self.status,
            "detail": self.detail,
            "error": self.error,
            "required": self.required,
        }


@dataclass
class VIPResult:
    """Everything one pipeline run produced.

    Attributes
    ----------
    patient_id:
        Identifier of the patient scored.
    risk_prediction:
        :class:`src.ml.predict.RiskPrediction` - the Module 1 output that the
        report prompt consumes (FR-4.2).
    attention:
        :class:`src.attention.self_attention.AttentionResult`.
    heatmap_path:
        Written attention heatmap (FR-2.5).
    retrievals:
        One :class:`src.rag.retrieve.RetrievalResult` per clinical dimension.
    retrieved_chunks:
        Flattened, de-duplicated chunk texts handed to the LLM as the only
        source of clinical facts (FR-4.2).
    report_run:
        :class:`src.llm.report_generator.ReportRun`.
    timings:
        Per-stage durations feeding dashboard Panel C.
    """

    patient_id: str
    risk_prediction: Any = None
    attention: Any = None
    heatmap_path: Path | None = None
    retrievals: list[Any] = field(default_factory=list)
    retrieved_chunks: list[str] = field(default_factory=list)
    report_run: Any = None
    timings: list[StageTiming] = field(default_factory=list)
    safety_outcome: Any = None

    #: Overall verdict, one of :data:`config.RUN_STATUSES`. Derived from the
    #: per-stage statuses by :func:`derive_run_status`; never set by hand, so a
    #: stage that failed cannot be reported as a successful run.
    status: str = config.RUN_FAILED

    #: True when at least one stage completed and another failed. Outputs from the
    #: completed stages are kept and marked partial rather than discarded, which is
    #: more useful than losing a real prediction because a credential was missing -
    #: provided the caller can see it is partial.
    partial: bool = False

    #: Sequence actually used by the attention stage, and whether it came from the
    #: patient record. The supplied dataset has no sequence column, so this is
    #: normally the PRD's declared sample sequence. Recorded so a report never
    #: implies a sequence originated from the patient.
    sequence: str | None = None
    sequence_source: str = "patient_record"

    #: The dose the caller supplied, in micrograms, preserved verbatim. Present so a
    #: clamped or recommended value can never be mistaken for the original input.
    submitted_dose_mcg: float | None = None
    submitted_dose_unit: str = "mcg"

    #: Secondary-review finding from the deterministic dose check (FR-4.4).
    dose_review: Any = None

    def stage_status(self, stage: str) -> str:
        """Status of one stage, or ``not_started`` if it never ran."""
        for timing in self.timings:
            if timing.stage == stage:
                return timing.status
        return config.STAGE_NOT_STARTED

    def stage(self, name: str) -> StageTiming | None:
        """The :class:`StageTiming` for ``name``, or ``None``."""
        for timing in self.timings:
            if timing.stage == name:
                return timing
        return None

    def sync_stage_records(self) -> "VIPResult":
        """Fill in a ``not_started`` record for every stage that has no timing.

        Stages after a failure never execute, so they record nothing. Leaving them
        absent would make "never ran" indistinguishable from "ran and succeeded"
        in the printed table and in :meth:`as_dict` - precisely the ambiguity that
        makes a failed pipeline look successful. Called once at the end of a run,
        after :func:`derive_run_status` has already judged the real records.

        Existing records are never touched, so a completed stage cannot be
        downgraded by this.
        """
        recorded = {t.stage for t in self.timings}
        for name in PIPELINE_STAGES:
            if name not in recorded:
                self.timings.append(
                    StageTiming(
                        stage=name,
                        seconds=0.0,
                        ok=False,
                        status=config.STAGE_NOT_STARTED,
                        detail="not reached",
                        required=True,
                    )
                )
        return self

    def timing_map(self) -> dict[str, float]:
        """Stage name to seconds, the shape dashboard Panel C plots."""
        return {t.stage: round(t.seconds, 4) for t in self.timings}

    @property
    def succeeded(self) -> bool:
        """Whether the run completed, i.e. every required stage completed."""
        return self.status == config.RUN_COMPLETED

    def failed_stages(self) -> list[str]:
        """Names of stages that failed. Empty on a completed run."""
        return [t.stage for t in self.timings if t.status == config.STAGE_FAILED]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary written to ``outputs/logs/``.

        Carries the per-stage statuses and the overall verdict, because a consumer
        - the dashboard in particular - must be able to tell a real successful run
        from a partial one without re-deriving it.
        """
        return {
            "patient_id": self.patient_id,
            "status": self.status,
            "partial": self.partial,
            "stages": [t.as_dict() for t in self.timings],
            "stage_statuses": {t.stage: t.status for t in self.timings},
            "risk_prediction": (
                self.risk_prediction.as_dict() if self.risk_prediction else None
            ),
            "attention": (
                {
                    "sequence": self.sequence,
                    "sequence_source": self.sequence_source,
                    "heatmap_path": str(self.heatmap_path) if self.heatmap_path else None,
                    "shape": list(self.attention.weights.shape)
                    if getattr(self.attention, "weights", None) is not None
                    else None,
                }
                if self.attention is not None
                else None
            ),
            "heatmap_path": str(self.heatmap_path) if self.heatmap_path else None,
            "n_retrieved_chunks": len(self.retrieved_chunks),
            "retrieval_queries": [r.question for r in self.retrievals],
            "submitted_dose_mcg": self.submitted_dose_mcg,
            "submitted_dose_unit": self.submitted_dose_unit,
            "dose_review": self.dose_review.as_dict() if self.dose_review else None,
            "clinical_maximum_established": False,
            "report_paths": {
                "json": str(self.report_run.json_path)
                if getattr(self.report_run, "json_path", None)
                else None,
                "markdown": str(self.report_run.markdown_path)
                if getattr(self.report_run, "markdown_path", None)
                else None,
            },
            "timings": self.timing_map(),
        }


def derive_run_status(timings: list[StageTiming]) -> str:
    """Overall verdict from the per-stage timings (architecture rule 8).

    The rules, in order:

    * Any required stage ``failed`` -> :data:`config.RUN_FAILED`.
    * Any required stage never started or blocked -> :data:`config.RUN_FAILED`,
      because a run that skipped its report did not complete.
    * Every required stage ``completed``, but an optional one did not ->
      :data:`config.RUN_COMPLETED`, because an optional stage is by definition not
      required. Its non-completion is still visible in the stage record.
    * All required stages completed -> :data:`config.RUN_COMPLETED`.

    A stage marked ``completed`` is never reclassified, so this cannot report
    success over a failure.
    """
    required = [t for t in timings if t.required]
    if not required:
        return config.RUN_FAILED

    for timing in required:
        if timing.status != config.STAGE_COMPLETED:
            return config.RUN_FAILED
    return config.RUN_COMPLETED


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class VIPPipeline:
    """Runs the five modules in order and returns a :class:`VIPResult`.

    Usage
    -----
    >>> pipeline = VIPPipeline()
    >>> result = pipeline.train()                      # Module 1, once
    >>> result = pipeline.run(patient_record)           # Modules 1-4
    >>> pipeline.build_dashboard()                      # Module 5

    Artifacts are loaded from disk on construction so a run is reproducible from
    a clean checkout once Module 1 and Module 3 have been executed.
    """

    def __init__(
        self,
        raw_csv: Any = None,
        guideline_docx: Any = None,
        models_dir: Any = None,
        vectorstore_dir: Any = None,
        top_k: int = config.RETRIEVAL_TOP_K,
        llm_model: str = config.LLM_MODEL,
    ) -> None:
        """Resolve paths and record settings. No I/O happens here.

        Parameters
        ----------
        raw_csv:
            Patient dataset. Defaults to ``config.RAW_PATIENT_CSV``.
        guideline_docx:
            Read-only guideline source. Defaults to ``config.GUIDELINE_DOCX``.
        models_dir, vectorstore_dir:
            Artifact locations. Default to the canonical ``models/`` and
            ``vectorstore/`` directories.
        top_k:
            Chunks per retrieval query. Defaults to 3 (FR-3.4).
        llm_model:
            Model used for report generation. Defaults to ``config.LLM_MODEL``.
        """
        self.raw_csv = Path(raw_csv) if raw_csv else config.RAW_PATIENT_CSV
        self.guideline_docx = (
            Path(guideline_docx) if guideline_docx else config.GUIDELINE_DOCX
        )
        self.models_dir = Path(models_dir) if models_dir else config.MODELS_DIR
        self.vectorstore_dir = (
            Path(vectorstore_dir) if vectorstore_dir else config.VECTORSTORE_DIR
        )
        self.top_k = int(top_k)
        self.llm_model = llm_model

        self.timings: list[StageTiming] = []
        self.split = None
        self.trained = None
        self.index = None
        self.embedding_function = None
        self.selection = None
        self.evaluations: dict[str, Any] = {}
        self.model_paths: dict[str, Path] = {}

        #: Resolved once so a run reuses one loaded index and one embedding model
        #: rather than re-loading per query. ChromaDB index loading is the slow
        #: part of stage 4.
        self._index_loaded = False

    # -- Module 1 -----------------------------------------------------------

    def train(self) -> "VIPPipeline":
        """Run the full Module 1 workflow: preprocess, train, evaluate, select.

        Executes FR-1.1 through FR-1.7 and writes the model artifacts to
        ``models/``. Must succeed before :meth:`run` can score a patient.

        Returns
        -------
        VIPPipeline
            Self, so calls chain.

        Raises
        ------
        PipelineError
            If any part of Module 1 fails.
        """
        started = time.perf_counter()
        try:
            split = self.preprocess()

            x_test, y_test = split.x_test, split.y_test
            trained = ml_train.train_models(split)
            written = ml_train.save_model_artifacts(trained)
            ml_train.save_training_metadata(trained, split)

            # The preprocessor is fitted on train rows only, so evaluation never
            # sees test-time imputation statistics (FR-1.4 / no leakage).
            evaluations = ml_evaluate.evaluate_all_models(
                trained, x_test, y_test, test_raw=split.test_raw
            )
            selection = ml_evaluate.select_best_model(evaluations)
            ml_evaluate.save_evaluation_artifacts(evaluations, selection, split)

            self.split = split
            self.trained = trained
            self.selection = selection
            self.evaluations = evaluations
            self.model_paths = written

        except PipelineError:
            raise
        except Exception as exc:  # noqa: BLE001 - attributed to the stage
            raise PipelineError("train", exc) from exc

        self.timings.append(
            StageTiming(
                stage="train",
                seconds=time.perf_counter() - started,
                status=config.STAGE_COMPLETED,
                detail=f"selected {selection.best_name} on {selection.metric}",
            )
        )
        return self

    # -- Stage 1 ------------------------------------------------------------

    def preprocess(self) -> Any:
        """Stage 1: load, validate, clean, encode, scale and split (FR-1.1-1.4).

        Returns
        -------
        SplitData
            The fitted-and-split dataset from
            :func:`src.data.preprocessing.stratified_train_test_split`.

        Raises
        ------
        PipelineError
            If the dataset is missing, fails validation, or cannot be split.
        """
        with Stopwatch("preprocessing") as watch:
            try:
                frame = preprocessing.load_raw_records(self.raw_csv)
                preprocessing.validate_dataset(frame)
                report = preprocessing.inspect_dataset(frame)
                medians, modes = preprocessing.fit_imputation_strategy(frame)
                cleaned = preprocessing.impute_missing_values(
                    frame, medians=medians, modes=modes, report=report
                )

                # ``stratified_train_test_split`` fits the preprocessor on the
                # training rows only, so no test-row imputation statistic or
                # scaler parameter leaks into the fitted transform (FR-1.4).
                preprocessor = preprocessing.build_preprocessing_pipeline(
                    numeric_medians=medians, categorical_modes=modes
                )
                split = preprocessing.stratified_train_test_split(
                    cleaned, preprocessor
                )
                preprocessing.save_clean_records(cleaned)
            except Exception as exc:  # noqa: BLE001 - attributed to the stage
                self.timings.append(watch.as_timing())
                raise PipelineError("preprocessing", exc) from exc

        self.split = split
        self.timings.append(watch.as_timing())
        return split

    # -- Stage 2 ------------------------------------------------------------

    def predict_risk(self, patient_record: dict[str, Any]) -> Any:
        """Stage 2: score one patient with the serialized risk model (FR-1.7).

        Parameters
        ----------
        patient_record:
            Raw patient features keyed by column name.

        Returns
        -------
        RiskPrediction
            Risk band and probability from the serialized FR-1.7 winner.

        Raises
        ------
        PipelineError
            If the record is missing required features, or the model artifacts are
            absent or unreadable. Never falls back to a fabricated probability.
        """
        with Stopwatch("ml_prediction") as watch:
            try:
                prediction = ml_predict.predict_patient(patient_record)
            except Exception as exc:  # noqa: BLE001 - attributed to the stage
                self.timings.append(watch.as_timing())
                raise PipelineError("ml_prediction", exc) from exc

        if prediction.risk_probability is None:
            watch.detail = "no model produced a probability"
            self.timings.append(watch.as_timing())
            raise PipelineError(
                "ml_prediction",
                ml_predict.PredictionError(
                    "No trained model is available, so no risk prediction could be "
                    f"produced. Train first: python -m src.ml"
                ),
            )

        self.timings.append(watch.as_timing())
        return prediction

    # -- Stage 3 ------------------------------------------------------------

    def run_attention(self, sequence: str) -> Any:
        """Stage 3: self-attention over the patient's mRNA sequence (FR-2.1-2.5).

        Parameters
        ----------
        sequence:
            Nucleotide string over {A, C, G, U}, length 5-50.

        Returns
        -------
        AttentionResult
            Weights, Q/K/V projections and the saved heatmap path.

        Raises
        ------
        PipelineError
            If the sequence is absent, not a valid nucleotide string, or outside
            the accepted length range. Attention is never skipped silently - a
            failed stage is reported as failed.
        """
        with Stopwatch("attention") as watch:
            try:
                validated = self_attention.validate_sequence(sequence)
                result = self_attention.run_self_attention(validated)
                written = self_attention.save_attention_artifacts(result)
            except Exception as exc:  # noqa: BLE001 - attributed to the stage
                self.timings.append(watch.as_timing())
                raise PipelineError("attention", exc) from exc

        self.timings.append(watch.as_timing())
        self._attention_artifacts = written
        return result

    # -- Stage 4 ------------------------------------------------------------

    def build_retrieval_queries(self, patient_record: dict[str, Any]) -> list[str]:
        """Derive one natural-language question per relevant clinical dimension.

        Each question maps to a known guideline section, so retrieval results can
        be attributed:

        ====================================== ==============================
        Patient attribute                       Guideline section targeted
        ====================================== ==============================
        ``Age``                                 2.2 age-banded reference doses
        ``Vaccine_Dose_mcg``                    2.1 general dosing principles
        ``Biomarker_Level``                     2.3 elevated-biomarker adjustment
        ``Pre_Existing_Conditions``             3.2 precautions needing review
        ``Prior_Reaction_History``              3.1 / 3.2 contraindications
        combination                             3.3 standard warnings
        ====================================== ==============================

        Returns
        -------
        list of str

        Notes
        -----
        Queries are phrased in the guideline's **own** vocabulary wherever the
        document supplies it. Measured, not stylistic: the paraphrase "How long
        must a patient be monitored after administration?" scores 0.497 on the
        relevant chunk, while the document's own phrasing ("standard warnings ...
        monitored for at least 15 minutes") scores 0.698. Where the document does
        not supply phrasing, the question falls back to the clinical dimension and
        the result is still real retrieved text - just less well targeted.
        """
        def band_for(age: Any) -> str:
            """Age band wording matching guideline Section 2.2."""
            try:
                years = float(age)
            except (TypeError, ValueError):
                return ""
            if years < 5:
                return "Ages 6 months to 4 years"
            if years < 12:
                return "Ages 5 to 11 years"
            if years < 65:
                return "Ages 12 to 64 years"
            return "Ages 65 and older"

        queries: list[str] = []

        age = patient_record.get("Age")
        if age is not None:
            queries.append(f"{band_for(age)}: age-banded reference doses")

        dose = patient_record.get("Vaccine_Dose_mcg")
        if dose is not None:
            queries.append(
                "general dosing principles primary series dose range micrograms"
            )

        biomarker = patient_record.get("Biomarker_Level")
        if biomarker is not None:
            queries.append("dose adjustment for elevated biomarker levels")

        conditions = patient_record.get("Pre_Existing_Conditions")
        if conditions is not None:
            queries.append("precautions requiring secondary clinician review")

        history = patient_record.get("Prior_Reaction_History")
        if history is not None:
            queries.append(
                "absolute contraindications known severe allergic reaction to a prior dose"
            )

        # Standard warnings cover observation after administration, which no single
        # column captures, so it is always asked.
        queries.append("standard warnings patients monitored for at least 15 minutes")
        return queries

    def retrieve_guidelines(
        self, patient_record: dict[str, Any]
    ) -> list[Any]:
        """Stage 4: retrieve top-k guideline chunks per clinical dimension.

        Returns
        -------
        list of RetrievalResult
            One per query, each carrying its own latency for FR-3.5 and Panel C.

        Raises
        ------
        PipelineError
            If the vector index is missing or unreadable. Never substitutes
            synthetic chunks: a retrieval failure must be visible, because
            fabricated evidence would flow straight into the report prompt.
        """
        queries = self.build_retrieval_queries(patient_record)
        with Stopwatch("rag_retrieval") as watch:
            try:
                if not self._index_loaded:
                    self.index = rag_ingest.load_index(
                        persist_dir=self.vectorstore_dir
                    )
                    self.embedding_function = rag_ingest.build_embedding_function()
                    self._index_loaded = True

                if self.index.count() < self.top_k:
                    raise rag_ingest.IndexingError(
                        f"The index holds {self.index.count()} chunk(s), fewer "
                        f"than top_k={self.top_k}. Rebuild it with: "
                        "python -m src.rag.ingest --reset"
                    )

                retrievals = [
                    rag_retrieve.query_guidelines(
                        question,
                        top_k=self.top_k,
                        index=self.index,
                        embedding_function=self.embedding_function,
                    )
                    for question in queries
                ]
            except Exception as exc:  # noqa: BLE001 - attributed to the stage
                self.timings.append(watch.as_timing())
                raise PipelineError("rag_retrieval", exc) from exc

        self.timings.append(watch.as_timing())
        return retrievals

    # -- Stage 5 ------------------------------------------------------------

    def generate_report(
        self, patient_record: dict[str, Any], retrievals: list[Any], risk: Any
    ) -> Any:
        """Stage 5: generate the structured clinical safety report (FR-4.1-4.6).

        Grounding is limited to the Module 1 risk output and the Module 3 chunks
        (FR-4.2). The retrieved text is also inspected for a dose ceiling, which
        is enforced deterministically in
        :func:`src.llm.schemas.enforce_safety_constraints` (FR-4.4).

        Returns
        -------
        ReportRun
            The validated, safety-checked report.

        Raises
        ------
        PipelineError
            If no retrieved evidence is available, no API key is configured, the
            provider call fails, or the response does not validate after the single
            permitted retry. A failure here means **no report artifact is
            written** - a report that did not validate is never persisted as a
            success.

        Notes
        -----
        The dose check is deliberately given the **submitted** dose as well as the
        retrieved chunks, so the deterministic validation of the patient's actual
        input does not depend on what the model said about it.
        """
        chunks = _unique_chunks(retrievals)
        if not chunks:
            self.timings.append(
                StageTiming(
                    stage="llm_report",
                    seconds=0.0,
                    status=config.STAGE_BLOCKED,
                    detail=(
                        "No retrieved guideline evidence is available, so a "
                        "grounded report cannot be produced (FR-4.2). Retrieval "
                        "must run first."
                    ),
                )
            )
            raise PipelineError(
                "llm_report",
                ValueError(
                    "Report generation requires at least one retrieved guideline "
                    "chunk. Retrieval produced none, and fabricating evidence "
                    "would violate FR-4.2."
                ),
            )

        submitted = patient_record.get("Vaccine_Dose_mcg")
        risk_label = risk.risk_label if risk is not None else "Unknown"
        risk_probability = (
            risk.risk_probability if risk is not None and risk.risk_probability is not None else 0.0
        )
        summary = llm_report.summarise_patient(patient_record)

        with Stopwatch("llm_report") as watch:
            try:
                run = llm_report.generate_report(
                    risk_label, risk_probability, chunks, summary
                )
            except llm_report.MissingAPIKeyError as exc:
                # Actionable, and never disguised as a success.
                self.timings.append(watch.as_timing())
                raise PipelineError("llm_report", exc) from exc
            except Exception as exc:  # noqa: BLE001 - attributed to the stage
                self.timings.append(watch.as_timing())
                raise PipelineError("llm_report", exc) from exc

        # Re-run the deterministic checks with the submitted dose attached, so the
        # patient's real input is validated independently of the model's output.
        bound = llm_schemas.extract_dose_bound(chunks)
        outcome = llm_schemas.enforce_safety_constraints(
            run.report,
            bound,
            submitted_dose_mcg=float(submitted) if submitted is not None else None,
            submitted_unit="mcg",
        )
        run.report = outcome.report
        run.safety = outcome
        run.dose_bound = bound
        run.patient_id = patient_record.get("Patient_ID")

        self.timings.append(watch.as_timing())
        return run

    # -- Stage 6 ------------------------------------------------------------

    def persist_outputs(self, result: VIPResult) -> VIPResult:
        """Stage 6: write reports, figures and logs.

        Writes the report pair to ``outputs/reports/`` and the consolidated
        result plus latency log to ``outputs/logs/``, which is what the dashboard
        reads back (FR-5.2).

        Returns
        -------
        VIPResult
            The result, with its overall status derived from the stage records and
            its report paths filled in.

        Raises
        ------
        PipelineError
            If the report could not be persisted. Persistence failing after a
            successful generation is a real failure, so it is not swallowed.
        """
        result.report_run = getattr(self, "_pending_run", None)

        with Stopwatch("persist") as watch:
            try:
                run = self._pending_run
                if run is None:
                    raise ValueError(
                        "There is no validated report to persist. Persistence "
                        "requires a report that passed schema validation."
                    )

                json_path, markdown_path = llm_report.persist_report(run)
                llm_report.record_generation_latency(run)
                self._write_run_log(result)

            except Exception as exc:  # noqa: BLE001 - attributed to the stage
                self.timings.append(watch.as_timing())
                raise PipelineError("persist", exc) from exc

        self.timings.append(watch.as_timing())
        result.status = derive_run_status(self.timings)
        result.partial = result.status == config.RUN_PARTIAL
        return result

    def _write_run_log(self, result: VIPResult) -> Path:
        """Write the consolidated run summary to ``outputs/logs/``.

        Marked ``partial`` explicitly when stages failed, so a reader of the file
        cannot mistake a run that stopped early for a successful one.
        """
        destination = config.LOGS_DIR / "pipeline_run.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        return destination

    # -- Full run -----------------------------------------------------------

    def run(self, patient_record: dict[str, Any]) -> VIPResult:
        """Execute stages 1-6 for one patient and return the assembled result.

        Parameters
        ----------
        patient_record:
            Raw patient features keyed by column name. ``Patient_ID`` is used
            for labelling.

        Returns
        -------
        VIPResult

        Raises
        ------
        VIPResult
            Always returned, even when a stage failed. Inspect
            :attr:`VIPResult.status` - it is :data:`config.RUN_FAILED` if any
            required stage did not complete.

        Raises
        ------
        PipelineError
            If a **required** stage fails. Stages after a failure are not run, and
            not run means ``not_started``, never ``completed``.

        Notes
        -----
        The sequence input is handled explicitly. The supplied dataset has no
        sequence column, so the PRD's declared sample sequence
        (:data:`config.PIPELINE_DEFAULT_SEQUENCE`) is used and the result records
        ``sequence_source='config_default'``. No patient sequence is invented, and
            the distinction is visible to any consumer of the result.

        Attention is a **required** stage. The PRD's Week 4 definition of done
        requires the heatmap as Panel B of the final figure, so a failure there
        fails the run rather than being skipped.

        When a later stage fails, the earlier stages' outputs are preserved on the
        returned result and marked partial. A real prediction is more useful than
        none - provided it cannot be mistaken for a complete run.
        """
        patient_id = str(patient_record.get("Patient_ID", "unknown"))
        result = VIPResult(patient_id=patient_id)
        result.submitted_dose_mcg = (
            float(patient_record["Vaccine_Dose_mcg"])
            if patient_record.get("Vaccine_Dose_mcg") is not None
            else None
        )
        result.submitted_dose_unit = "mcg"

        # Each stage is invoked through `_stage`, which attaches the partially
        # populated result to the PipelineError it raises. Without that, a failure
        # at stage 5 would discard a perfectly good stage-2 prediction.
        try:
            self._stage(result, "preprocessing", self.preprocess)
            result.risk_prediction = self._stage(
                result, "ml_prediction", lambda: self.predict_risk(patient_record)
            )

            sequence, source = self._resolve_sequence(patient_record)
            result.sequence = sequence
            result.sequence_source = source
            result.attention = self._stage(
                result, "attention", lambda: self.run_attention(sequence)
            )
            heatmap = getattr(self, "_attention_artifacts", {}).get("heatmap")
            result.heatmap_path = Path(heatmap) if heatmap else None

            result.retrievals = self._stage(
                result, "rag_retrieval",
                lambda: self.retrieve_guidelines(patient_record),
            )
            result.retrieved_chunks = _unique_chunks(result.retrievals)

            run = self._stage(
                result, "llm_report",
                lambda: self.generate_report(
                    patient_record, result.retrievals, result.risk_prediction
                ),
            )
            self._pending_run = run
            result.report_run = run
            result.safety_outcome = getattr(run, "safety", None)
            result.dose_review = getattr(result.safety_outcome, "dose_review", None)

            self._stage(result, "persist", lambda: self.persist_outputs(result))

        except PipelineError as exc:
            # Attach whatever completed, clearly marked incomplete. The stage
            # records live on the pipeline, so they are copied onto the result -
            # without this the caller sees a "partial" result with no evidence of
            # which stages actually ran.
            result.timings = list(self.timings)
            exc.result = result
            result.status = config.RUN_FAILED
            result.partial = any(
                t.status == config.STAGE_COMPLETED for t in self.timings
            )
            raise

        result.status = derive_run_status(self.timings)
        result.partial = result.status == config.RUN_PARTIAL
        # The stage records live on the pipeline; copy them so the returned result
        # is self-describing.
        result.timings = list(self.timings)
        return result

    def _stage(self, result: "VIPResult", name: str, call: Any) -> Any:
        """Invoke one stage, preserving partial results when it fails.

        Each stage method records its own :class:`StageTiming` on ``self.timings``
        before raising, so the stage that failed is always present in the
        partial result. This wrapper only attaches that result to the raised
        :class:`PipelineError`.

        Stages after a failure are never reached, because the first failure raises.
        Their recorded status stays ``not_started``, which is distinct from
        ``completed`` - so a skipped stage can never be read as a successful one.
        """
        try:
            return call()
        except PipelineError as exc:
            raise PipelineError(
                exc.stage or name, exc.cause, result=result
            ) from exc.cause
        except Exception as exc:  # noqa: BLE001 - attributed to the stage
            # A stage that raised before recording anything still gets a record,
            # so the failing stage is never invisible.
            if not any(t.stage == name for t in self.timings):
                self.timings.append(
                    StageTiming(
                        stage=name,
                        seconds=0.0,
                        status=config.STAGE_FAILED,
                        detail=str(exc),
                        error=type(exc).__name__,
                    )
                )
            raise PipelineError(name, exc, result=result) from exc

    def stage_timing(self, name: str) -> StageTiming | None:
        """The :class:`StageTiming` recorded on this pipeline for ``name``."""
        for timing in self.timings:
            if timing.stage == name:
                return timing
        return None

    def _resolve_sequence(self, patient_record: dict[str, Any]) -> tuple[str, str]:
        """Return the sequence to attend over, and where it came from.

        Prefers a sequence carried by the record. The supplied dataset has none, so
        this returns ``config.PIPELINE_DEFAULT_SEQUENCE`` - the PRD's declared
        sample sequence - and labels the source ``config_default`` so the result can
        never imply the sequence came from the patient.
        """
        for key in ("mRNA_Sequence", "Sequence", "sequence"):
            value = patient_record.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip(), "patient_record"

        # No sequence in the record: fall back to the PRD's declared sample
        # sequence. Labelled ``config_default`` so no consumer can mistake it for
        # patient data.
        return config.PIPELINE_DEFAULT_SEQUENCE, "config_default"

    # -- Module 5 -----------------------------------------------------------

    def build_dashboard(self, save_path: Any = None) -> Path:
        """Invoke the dashboard to render the three-panel figure (FR-5.1).

        Delegates to :mod:`dashboard.app` rather than duplicating plotting code
        here, so the UI layer stays the only place figures are composed
        (architecture rule 9).

        Returns
        -------
        pathlib.Path
            The saved figure.

        Raises
        ------
        PipelineError
            If the figure cannot be rendered. A panel whose artifact is missing
            renders an empty state rather than an invented chart, so a missing
            model still yields a valid - and honest - figure.
        """
        with Stopwatch("build_dashboard", required=False) as watch:
            try:
                from dashboard import app as dashboard_app

                path = dashboard_app.build_dashboard(save_path)
            except Exception as exc:  # noqa: BLE001 - attributed to the stage
                self.timings.append(watch.as_timing())
                raise PipelineError("build_dashboard", exc) from exc

        self.timings.append(watch.as_timing())
        return Path(path) if path else Path(save_path or "")


# ---------------------------------------------------------------------------
# Module-level helpers used by VIPPipeline
# ---------------------------------------------------------------------------


def _unique_chunks(retrievals: list[Any]) -> list[str]:
    """Flatten retrieval results into de-duplicated chunk texts, best first.

    Overlap is expected - several clinical queries legitimately retrieve the same
    passage - so duplicates are removed while preserving retrieval order. The
    result is what reaches the prompt as the *only* source of clinical facts
    (FR-4.2).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for result in retrievals or []:
        for text in getattr(result, "texts", []) or []:
            key = " ".join(text.split())
            if key and key not in seen:
                seen.add(key)
                ordered.append(text)
    return ordered


def schemas_dose_bound(chunks: list[str]) -> Any:
    """Derive the review threshold from retrieved chunks (FR-4.4 input)."""
    return llm_schemas.extract_dose_bound(chunks)


# ---------------------------------------------------------------------------
# Timing helpers (feed dashboard Panel C, PRD Section 8.4)
# ---------------------------------------------------------------------------


class Stopwatch:
    """Context manager that appends a :class:`StageTiming` on exit.

    Used as::

        with Stopwatch("attention") as watch:
            result = pipeline.run_attention(sequence)
        timings.extend(watch.record)
    """

    def __init__(self, stage: str, required: bool = True) -> None:
        self.stage = stage
        self.required = required
        self.start = 0.0
        self.elapsed_s = 0.0
        self.ok = True
        self.detail = ""
        self.error = ""

    def __enter__(self) -> "Stopwatch":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed_s = time.perf_counter() - self.start
        self.ok = exc_type is None
        self.error = "" if exc_type is None else type(exc).__name__
        self.detail = "" if exc_type is None else str(exc)
        return False

    def as_timing(self, required: bool = True) -> StageTiming:
        """Snapshot this measurement as a :class:`StageTiming`.

        ``required=False`` marks a stage whose failure does not invalidate the
        run, so :func:`derive_run_status` can distinguish an optional miss from a
        required one.
        """
        return StageTiming(
            stage=self.stage,
            seconds=self.elapsed_s,
            ok=self.ok,
            detail=self.detail,
            status=(
                config.STAGE_COMPLETED if self.ok
                else config.STAGE_FAILED
            ),
            error=self.error,
            required=required,
        )


def save_latency_log(
    timings: list[StageTiming], path: Any = None
) -> Path:
    """Write stage timings to ``outputs/logs/latency_log.json``.

    Panel C plots the retrieval and LLM latencies this file contains. Defaults
    to ``config.LATENCY_LOG_JSON_PATH``.

    Parameters
    ----------
    timings:
        Timings collected during the run.
    path:
        Override destination.

    Returns
    -------
    pathlib.Path
    """
    destination = Path(path) if path else config.LATENCY_LOG_JSON_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"stages": [t.__dict__ for t in timings]}, indent=2),
        encoding="utf-8",
    )
    return destination


# ---------------------------------------------------------------------------
# Functional entry point
# ---------------------------------------------------------------------------


def run_pipeline(patient_record: dict[str, Any]) -> VIPResult:
    """Run the whole platform for one patient and return the result.

    Thin wrapper around :meth:`VIPPipeline.run` for callers that do not need a
    configured instance, such as tests and one-off scripts.

    Raises
    ------
    PipelineError
        If a required stage fails.
    """
    pipeline = VIPPipeline()
    result = pipeline.run(patient_record)
    if result.status != config.RUN_COMPLETED:
        failed = ", ".join(result.failed_stages()) or "unknown"
        raise PipelineError(
            "run",
            RuntimeError(
                f"The pipeline did not complete. Status={result.status}. "
                f"Failing stage(s): {failed}. No report artifact was written."
            ),
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the pipeline.

    Usage
    -----
    ``python -m src.pipeline.pipeline --train``
        Execute Module 1 only and write the model artifacts.

    ``python -m src.pipeline.pipeline --index``
        Execute Module 3 ingest only and write the ChromaDB index.

    ``python -m src.pipeline.pipeline --run --patient-id P0001``
        Score one patient from the dataset through Modules 1-4 and write the
        report pair.

    ``python -m src.pipeline.pipeline --all``
        Train, ingest, run the sample patient, then build the dashboard.

    Arguments
    ---------
    --patient-id:
        Row identifier in the raw CSV. Defaults to the first record.
    --sequence:
        Overrides the nucleotide sequence attributed to the patient. The provided
        dataset has no sequence column, so the default is the PRD's declared sample
        sequence (``config.PIPELINE_DEFAULT_SEQUENCE``), recorded in the result with
        ``sequence_source='config_default'``. No patient sequence is invented.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline.pipeline",
        description="Run the mRNA VIP pipeline end to end.",
    )
    parser.add_argument("--train", action="store_true", help="Execute Module 1 only")
    parser.add_argument("--index", action="store_true", help="Execute Module 3 ingest only")
    parser.add_argument("--run", action="store_true", help="Score one patient end to end")
    parser.add_argument("--all", action="store_true", help="Train, index, run, dashboard")
    parser.add_argument("--dashboard", action="store_true", help="Build the figure only")
    parser.add_argument("--patient-id", default=None, help="Row identifier in the raw CSV")
    parser.add_argument("--sequence", default=None, help="Override the attention sequence")
    parser.add_argument("--json", action="store_true", help="Print the run summary as JSON")
    args = parser.parse_args(argv)

    if not any((args.train, args.index, args.run, args.all, args.dashboard)):
        parser.error("Choose at least one of --train, --index, --run, --all, --dashboard")

    pipeline = VIPPipeline()
    exit_code = 0

    try:
        if args.train or args.all:
            pipeline.train()
            print(f"[ok] Module 1 trained. Selected: {pipeline.selection.best_name}")

        if args.index or args.all:
            pipeline.index = rag_ingest.build_index(
                guideline_docx=pipeline.guideline_docx,
                persist_dir=pipeline.vectorstore_dir,
                reset=True,
            )
            pipeline._index_loaded = True
            print(f"[ok] Module 3 indexed. {pipeline.index.count()} chunks in "
                  f"{pipeline.index.collection_name}")

        if args.run or args.all:
            record = load_patient(args.patient_id)
            if args.sequence:
                record["mRNA_Sequence"] = args.sequence
            result = pipeline.run(record)

            if args.json:
                print(json.dumps(result.as_dict(), indent=2, default=str))
            else:
                print_summary(result)

            if result.status != config.RUN_COMPLETED:
                print(f"[fail] status={result.status} failed={result.failed_stages()}")
                exit_code = 1

        if args.dashboard or args.all:
            path = pipeline.build_dashboard()
            print(f"[ok] Dashboard written: {path}")

    except PipelineError as exc:
        # Honest failure: named stage, actionable cause, no fabricated success.
        # Stages that did complete are still reported, marked incomplete.
        print(f"[fail] {exc}")
        partial = getattr(exc, "result", None)
        if partial is not None:
            partial.sync_stage_records()
            print_summary(partial)
        return 1

    return exit_code


def load_patient(patient_id: str | None = None) -> dict[str, Any]:
    """Load one patient row from the raw dataset as a plain dict.

    Parameters
    ----------
    patient_id:
        ``Patient_ID`` to load. Defaults to the first row.

    Raises
    ------
    KeyError
        If ``patient_id`` is not in the dataset.
    """
    frame = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
    if patient_id is None:
        return frame.iloc[0].to_dict()
    matches = frame[frame["Patient_ID"].astype(str) == str(patient_id)]
    if matches.empty:
        raise KeyError(
            f"Patient '{patient_id}' is not in {config.RAW_PATIENT_CSV}. "
            f"Available ids begin: {list(frame['Patient_ID'].astype(str)[:5])}"
        )
    return matches.iloc[0].to_dict()


def print_summary(result: VIPResult) -> None:
    """Print a readable run summary.

    Prints the patient identifier and aggregate outputs only. No credential and no
    complete patient record reaches stdout, so the log is safe to share.
    """
    print()
    print("=" * 74)
    label = result.patient_id
    print(f"PIPELINE RUN  -  {label}  -  status: {result.status.upper()}")
    if result.partial:
        print("  (PARTIAL: some stages completed; this is NOT a successful run)")
    print("=" * 74)
    recorded = {t.stage: t for t in result.timings}
    marks = {"completed": "ok", "failed": "FAIL", "blocked": "BLOCKED",
             "not_started": "--", "skipped": "skip", "running": "..."}
    # Listed in canonical stage order rather than completion order, and every stage
    # appears exactly once - including the ones never reached. Silence would be
    # indistinguishable from success, which is exactly the confusion this pipeline
    # exists to avoid.
    for name in PIPELINE_STAGES:
        timing = recorded.get(name)
        if timing is None:
            print(f"  [{'--':<7}] {name:<16} {'-':>7}  not reached")
            continue
        mark = marks.get(timing.status, timing.status)
        print(f"  [{mark:<7}] {timing.stage:<16} {timing.seconds:>7.3f}s  "
              f"{timing.detail[:44]}")
    # Anything recorded outside the canonical list, e.g. train or build_dashboard.
    for name, timing in recorded.items():
        if name not in PIPELINE_STAGES:
            mark = marks.get(timing.status, timing.status)
            print(f"  [{mark:<7}] {timing.stage:<16} {timing.seconds:>7.3f}s  "
                  f"{timing.detail[:44]}")
    if result.risk_prediction is not None:
        prediction = result.risk_prediction
        print(f"\n  risk band      {prediction.risk_label}  "
              f"P(Reaction_Score=1)={prediction.risk_probability:.4f}")
    if result.sequence:
        print(f"  sequence       {result.sequence}  "
              f"(source: {result.sequence_source})")
    print(f"  retrieved      {len(result.retrieved_chunks)} unique chunks")

    review = result.dose_review
    if review is not None:
        print(f"\n  submitted dose   {review.submitted_dose_mcg} {review.unit} "
              "(preserved as supplied)")
        print(f"  review threshold {review.threshold_mcg} mcg  "
              f"kind={review.threshold_kind}")
        print(f"  review required  {review.review_required}")
    print("\n  clinical maximum established: False - the guideline states a "
          "secondary review\n  threshold, not a clinically established maximum "
          "safe dose.")
    print()
    print(f"  {config.REQUIRED_DISCLAIMER}")
    print()