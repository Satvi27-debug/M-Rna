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
Structure and contracts only. Bodies raise ``NotImplementedError``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config
from src.data import preprocessing
from src.ml import evaluate as ml_evaluate
from src.ml import predict as ml_predict
from src.ml import train as ml_train
from src.attention import self_attention
from src.rag import ingest as rag_ingest
from src.rag import retrieve as rag_retrieve
from src.llm import report_generator as llm_report
from src.llm import schemas as llm_schemas


class PipelineError(RuntimeError):
    """Raised when a stage cannot complete.

    The stage name is carried so a failure in a six-step flow is immediately
    attributable without reading a traceback.
    """

    def __init__(self, stage: str, cause: BaseException) -> None:
        self.stage = stage
        self.cause = cause
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
    """Wall-clock duration and outcome of one stage."""

    stage: str
    seconds: float
    ok: bool = True
    detail: str = ""


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

    def timing_map(self) -> dict[str, float]:
        """Stage name to seconds, the shape dashboard Panel C plots."""
        return {t.stage: round(t.seconds, 4) for t in self.timings}

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary written to ``outputs/logs/``."""
        return {
            "patient_id": self.patient_id,
            "risk_prediction": (
                self.risk_prediction.as_dict() if self.risk_prediction else None
            ),
            "heatmap_path": str(self.heatmap_path) if self.heatmap_path else None,
            "n_retrieved_chunks": len(self.retrieved_chunks),
            "retrieval_queries": [r.question for r in self.retrievals],
            "timings": self.timing_map(),
        }


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
        raise NotImplementedError

    # -- Stage 1 ------------------------------------------------------------

    def preprocess(self) -> Any:
        """Stage 1: load, validate, clean, encode, scale and split (FR-1.1-1.4).

        Returns
        -------
        SplitData
            The fitted-and-split dataset from
            :func:`src.data.preprocessing.stratified_train_test_split`.
        """
        raise NotImplementedError

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
            Risk band and probability. Trained on identical data to the Random
            Forest per FR-1.5.
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

    def retrieve_guidelines(
        self, patient_record: dict[str, Any]
    ) -> list[Any]:
        """Stage 4: retrieve top-k guideline chunks per clinical dimension.

        Returns
        -------
        list of RetrievalResult
            One per query, each carrying its own latency for FR-3.5 and Panel C.
        """
        raise NotImplementedError

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
        """
        raise NotImplementedError

    # -- Stage 6 ------------------------------------------------------------

    def persist_outputs(self, result: VIPResult) -> VIPResult:
        """Stage 6: write reports, figures and logs.

        Writes the report pair to ``outputs/reports/`` and the consolidated
        result plus latency log to ``outputs/logs/``, which is what the dashboard
        reads back (FR-5.2).

        Returns
        -------
        VIPResult
        """
        raise NotImplementedError

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
        PipelineError
            Wrapping the originating failure, tagged with the failing stage.
        """
        raise NotImplementedError

    # -- Module 5 -----------------------------------------------------------

    def build_dashboard(self, save_path: Any = None) -> Path:
        """Invoke the dashboard to render the three-panel figure (FR-5.1).

        Delegates to :mod:`dashboard.app` rather than duplicating plotting code
        here, so the UI layer stays the only place figures are composed
        (architecture rule 9).

        Returns
        -------
        pathlib.Path
        """
        raise NotImplementedError


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

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.start = 0.0
        self.elapsed_s = 0.0
        self.ok = True
        self.detail = ""

    def __enter__(self) -> "Stopwatch":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed_s = time.perf_counter() - self.start
        self.ok = exc_type is None
        self.detail = "" if exc_type is None else str(exc)
        return False

    def as_timing(self) -> StageTiming:
        return StageTiming(
            stage=self.stage,
            seconds=self.elapsed_s,
            ok=self.ok,
            detail=self.detail,
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
    """
    raise NotImplementedError


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
        Overrides the nucleotide sequence attributed to the patient. The
        provided dataset has no sequence column, so the default comes from a
        seeded generator to keep the run reproducible.
    """
    raise NotImplementedError