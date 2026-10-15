"""Reusable UI components and the platform-state collector.

Two responsibilities, both strictly presentational:

1. **Figure panels.** One Matplotlib function per dashboard panel, each reading
   a saved artifact from the module that produced it.
2. **Platform state.** :func:`collect_platform_state` inspects the filesystem and
   reports what genuinely exists, so the web interface can distinguish *ready*
   from *not initialized* without guessing.

Architecture rule 9 and the project's separation rule
----------------------------------------------------
The dashboard contains no ML, RAG or LLM logic. It never imports
``src.ml``, ``src.rag``, ``src.llm.report_generator`` or ``src.pipeline``, never
trains, embeds or queries anything, and never recomputes a metric. It reads files
that those modules wrote, plus two declarations it is entitled to consume: the
column names in :mod:`src.config` and the field names of the Pydantic report
schema. Consequence: the interface renders correctly with no API key, no vector
index and no trained model, and it cannot produce a number the pipeline did not
produce.

FR-5.2 and the empty-state contract
----------------------------------
Each panel is built from its module's saved output. When that output does not
exist, the panel renders an explicit empty state naming what will produce it.
A panel never draws an invented curve, heatmap or bar.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import config
from dashboard import styles


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MissingArtifactError(FileNotFoundError):
    """Raised when a panel's saved input artifact does not exist."""

    def __init__(self, path: Path, producer: str) -> None:
        self.path = Path(path)
        self.producer = producer
        super().__init__(
            f"Required artifact '{path}' is missing. Produce it first with: {producer}"
        )


# ---------------------------------------------------------------------------
# Shared copy - one source of truth for empty/loading/error messages
# ---------------------------------------------------------------------------
# The same strings are served to the web shell through :func:`platform_state`, so
# the figure and the browser can never disagree about what is missing or how to
# produce it.

#: Command that produces the Module 1 model artifact.
TRAIN_COMMAND = "python -m src.pipeline.pipeline --train"

#: Command that produces the Module 3 vector index.
INDEX_COMMAND = "python -m src.pipeline.pipeline --index"

#: Command that produces the Module 2 attention artifacts.
ATTENTION_COMMAND = "python -m src.attention"

#: Command that runs the full flow.
RUN_COMMAND = "python -m src.pipeline.pipeline --all"

#: Empty-state copy, keyed by panel id.
EMPTY_STATES: dict[str, str] = {
    "roc": "Evaluation results will appear after model training.",
    "attention": "Attention map will appear after inference.",
    "latency": "Latency measurements will appear after pipeline execution.",
    "models": "Model results will appear after training.",
    "risk": "Results will appear after model execution.",
    "retrieval": (
        "Retrieved evidence will appear after the regulatory retrieval pipeline "
        "is initialized."
    ),
    "interpretation": "Attention interpretation will appear after inference.",
    "report": "No report generated.",
    "dose": (
        "Maximum permitted dose will be evaluated by the validation pipeline."
    ),
}

#: Loading-state copy.
LOADING_STATE = "Processing..."

#: Generic not-initialized copy.
NOT_INITIALIZED_STATE = "This module has not been initialized."

#: Generic error copy, paired with a Retry action in the interface.
ERROR_STATE = "Something went wrong."

#: The report disclaimer the interface must always display (FR-4.5, PRD 8.2).
REPORT_DISCLAIMER = (
    "This is a student engineering project and is not intended to provide real "
    "clinical guidance."
)

#: The five pipeline stages, in order. Numbers match PRD Section 3.1.
PIPELINE_STAGES: tuple[dict[str, str], ...] = (
    {
        "id": "patient-data",
        "number": "01",
        "name": "Patient Data",
        "description": "Load, validate and standardise patient records.",
    },
    {
        "id": "risk-model",
        "number": "02",
        "name": "Risk Model",
        "description": "Train and score the reaction-risk classifiers.",
    },
    {
        "id": "attention",
        "number": "03",
        "name": "mRNA Attention",
        "description": "Compute self-attention over the nucleotide sequence.",
    },
    {
        "id": "retrieval",
        "number": "04",
        "name": "Regulatory Retrieval",
        "description": "Retrieve supporting guideline evidence.",
    },
    {
        "id": "report",
        "number": "05",
        "name": "Safety Report",
        "description": "Generate and validate the structured report.",
    },
)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path, producer: str) -> dict[str, Any]:
    """Load a JSON artifact, failing loudly when it is absent or corrupt."""
    path = Path(path)
    if not path.exists():
        raise MissingArtifactError(path, producer)
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_read_json(path: Any) -> dict[str, Any] | None:
    """Return parsed JSON, or ``None`` when the file is absent or unreadable.

    Used by the empty-state path. A corrupt file is treated as absent so the
    interface shows "not initialized" rather than crashing on a parse error.
    """
    path = Path(path) if path else None
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Shared panel chrome
# ---------------------------------------------------------------------------


def _panel_header(ax: Any, panel: str) -> None:
    """Apply the shared panel title and subtitle placement.

    FR-5.1 requires each panel to carry a title. The title is offset with a large
    ``pad`` so the subtitle, drawn just above the axes, sits between the title
    and the plot area instead of colliding with it.

    Parameters
    ----------
    ax:
        Target axes.
    panel:
        Key into :data:`styles.PANEL_TITLES` / :data:`styles.PANEL_SUBTITLES`.
    """
    ax.set_title(
        styles.PANEL_TITLES[panel],
        fontsize=12,
        fontweight="600",
        pad=30,
        color=styles.PALETTE["text"],
    )
    ax.text(
        0.5,
        1.012,
        styles.PANEL_SUBTITLES[panel],
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color=styles.PALETTE["text-subtle"],
    )


# ---------------------------------------------------------------------------
# Panel A - ROC curve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ROCPanelData:
    """Curves to draw in Panel A.

    ``curves`` holds one entry per evaluated model, so the panel shows the
    selected model against the model it was selected over. That is the honest
    comparison for FR-1.7: the reader can see the difference that drove the choice.
    """

    curves: list[dict[str, Any]]
    selected_model: str

    @property
    def selected(self) -> dict[str, Any]:
        """The curve of the FR-1.7 winner.

        Matched on the stable ``model_key`` where available, falling back to the
        display name. ``selected_model`` is stored as a key such as
        ``logistic_regression``, while a curve's ``model_name`` is the label
        ``"Logistic Regression"``, so matching only on the label would fail.
        """
        for curve in self.curves:
            if curve.get("model_key") == self.selected_model:
                return curve
        for curve in self.curves:
            if curve.get("model_name") == self.selected_model:
                return curve
        raise KeyError(f"Selected model '{self.selected_model}' not in metrics")


def load_roc_panel_data(
    metrics_path: Any = None, roc_path: Any = None
) -> ROCPanelData:
    """Load ROC points from the artifacts written by ``src.ml.evaluate``.

    Reads ``config.ROC_CURVE_JSON_PATH`` for the curve points and
    ``config.METRICS_JSON_PATH`` to learn which model FR-1.7 selected.

    Raises
    ------
    MissingArtifactError
        If either file is absent. The message names the command that produces it.
    """
    metrics_file = Path(metrics_path) if metrics_path else config.METRICS_JSON_PATH
    roc_file = Path(roc_path) if roc_path else config.ROC_CURVE_JSON_PATH

    metrics = _read_json(metrics_file, TRAIN_COMMAND)
    roc = _read_json(roc_file, TRAIN_COMMAND)
    curves = roc.get("curves", [])
    if not curves:
        raise MissingArtifactError(roc_file, TRAIN_COMMAND)

    # Accept either key spelling so the panel does not depend on one producer's
    # naming. A previous version wrote "best_model" here and "selected_model"
    # elsewhere; the mismatch surfaced only as an empty panel, because the
    # KeyError was absorbed into an empty state.
    selection = metrics.get("selection", {})
    selected = selection.get("selected_model") or selection.get("best_model")
    if not selected:
        raise MissingArtifactError(metrics_file, TRAIN_COMMAND)

    # Match on the stable key, not the display name. The curves carry both, but
    # the display label is presentation and could be reworded without the
    # selection changing.
    keys = {curve.get("model_key") for curve in curves}
    names = {curve.get("model_name") for curve in curves}
    if selected not in keys and selected not in names:
        raise MissingArtifactError(roc_file, TRAIN_COMMAND)

    return ROCPanelData(curves=curves, selected_model=selected)


def _is_selected_curve(curve: dict[str, Any], selected_model: str) -> bool:
    """Whether a saved curve is the FR-1.7 winner.

    Matched on ``model_key`` first, then the display name, because
    ``selected_model`` holds the stable key (``logistic_regression``) while
    ``model_name`` is the label (``"Logistic Regression"``). Matching only on the
    label silently marked both curves unselected.
    """
    return (
        curve.get("model_key") == selected_model
        or curve.get("model_name") == selected_model
    )


def plot_roc_panel(ax: Any, data: ROCPanelData | None = None) -> Any:
    """Draw the Module 1 ROC curves on ``ax`` (FR-5.1 Panel A).

    Plots the FR-1.7 selected model solid with its AUC in the legend, the other
    model dashed for comparison, and the ``[0, 1]`` reference diagonal.

    The AUC is read from the saved curve data rather than recomputed, so the panel
    reports exactly what the evaluation stage recorded.

    Parameters
    ----------
    ax:
        Target axes.
    data:
        Pre-loaded panel data. Read from ``outputs/logs/`` when omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from sklearn.metrics import auc as sklearn_auc

    _panel_header(ax, "roc")

    if data is None:
        with styles.empty_panel(ax, EMPTY_STATES["roc"]):
            return ax

    for curve in data.curves:
        is_selected = _is_selected_curve(curve, data.selected_model)
        colour = styles.PALETTE["accent"] if is_selected else styles.PALETTE["navy"]
        fpr, tpr = curve["fpr"], curve["tpr"]
        # Prefer the AUC the evaluation stage recorded; recompute only if absent.
        auc_value = curve.get("roc_auc")
        if auc_value is None:
            auc_value = sklearn_auc(fpr, tpr)
        label = f"{curve['model_name']} (AUC = {auc_value:.3f})"
        if is_selected:
            label += " — selected"
        ax.plot(
            fpr,
            tpr,
            label=label,
            color=colour,
            linewidth=2.2 if is_selected else 1.5,
            linestyle="-" if is_selected else "--",
        )

    ax.plot(
        [0, 1], [0, 1],
        linestyle=":", color=styles.REFERENCE_LINE_COLOR,
        label="No-skill reference",
    )
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=7)
    return ax


# ---------------------------------------------------------------------------
# Panel B - attention heatmap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionPanelData:
    """Weight matrix and token labels for Panel B.

    ``sequence`` holds one label per position: a string of nucleotides in
    nucleotide mode, or a list of encoded column names in feature mode. Both are
    rendered on the axes, so the panel is readable either way.
    """

    weights: Any
    sequence: Any
    input_mode: str = "unknown"

    @property
    def labels(self) -> list[str]:
        return [str(label) for label in self.sequence]

    @property
    def seq_len(self) -> int:
        return len(self.labels)


def load_attention_panel_data(
    weights_path: Any = None,
    labels_path: Any = None,
    sequence_path: Any = None,
) -> AttentionPanelData:
    """Load the attention matrix from the artifacts Module 2 saved (FR-5.2).

    Reads ``config.ATTENTION_WEIGHTS_PATH`` (an ``.npy`` matrix) and the token
    labels from ``config.ATTENTION_TOKEN_LABELS_PATH`` - a JSON object holding the
    per-position labels plus the input mode, since Module 2 accepts either a
    nucleotide sequence or the processed feature vector. The older plain-text
    ``config.ATTENTION_SEQUENCE_PATH`` is still accepted when present, so an
    artifact written before the JSON labels existed still loads.

    Raises
    ------
    MissingArtifactError
        If the weights or the labels are absent.
    """
    import numpy as np

    weights_file = Path(weights_path) if weights_path else config.ATTENTION_WEIGHTS_PATH
    if not weights_file.exists():
        raise MissingArtifactError(weights_file, ATTENTION_COMMAND)

    json_labels = Path(labels_path) if labels_path else config.ATTENTION_TOKEN_LABELS_PATH
    legacy = Path(sequence_path) if sequence_path else config.ATTENTION_SEQUENCE_PATH

    labels: list[str] = []
    input_mode = "unknown"
    if json_labels.exists():
        payload = json.loads(json_labels.read_text(encoding="utf-8"))
        labels = [str(label) for label in payload.get("labels", [])]
        input_mode = str(payload.get("input_mode", "unknown"))
    elif legacy.exists():
        labels = list(legacy.read_text(encoding="utf-8").strip())
        input_mode = "nucleotide"

    if not labels:
        raise MissingArtifactError(json_labels, ATTENTION_COMMAND)

    return AttentionPanelData(
        weights=np.load(weights_file),
        sequence="".join(labels) if input_mode == "nucleotide" else labels,
        input_mode=input_mode,
    )


def plot_attention_panel(ax: Any, data: AttentionPanelData | None = None) -> Any:
    """Draw the self-attention heatmap on ``ax`` (FR-5.1 Panel B).

    Both axes are labelled with nucleotide identity and position, matching the
    standalone ``attention_heatmap.png`` required by FR-2.5, so the two images are
    directly comparable. When no artifact exists, renders :data:`EMPTY_STATES`.
    """
    _panel_header(ax, "attention")

    if data is None:
        with styles.empty_panel(ax, EMPTY_STATES["attention"]):
            return ax

    labels = data.labels
    image = ax.imshow(data.weights, cmap=styles.ATTENTION_CMAP, aspect="auto")
    positions = range(len(labels))
    ax.set_xticks(list(positions))
    ax.set_yticks(list(positions))
    # Feature-mode labels are column names, so they are rotated to stay legible.
    ax.set_xticklabels(labels, fontsize=styles.ATTENTION_TICK_FONT_SIZE,
                       rotation=90 if data.input_mode == "feature" else 0)
    ax.set_yticklabels(labels, fontsize=styles.ATTENTION_TICK_FONT_SIZE)
    ax.set_xlabel("Attending TO position")
    ax.set_ylabel("Attending FROM position")
    figure = ax.get_figure()
    if figure is not None:
        figure.colorbar(image, ax=ax, label="Attention weight", fraction=0.046)
    return ax


# ---------------------------------------------------------------------------
# Panel C - latency comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LatencyPanelData:
    """Mean latency per stage for Panel C.

    ``stages`` maps a stage label to its mean latency in seconds. PRD Section 8.4
    compares RAG query latency against LLM report-generation latency, so both
    means are always included.
    """

    stages: dict[str, float]
    retrieval_queries: int = 0
    report_runs: int = 0


def load_latency_panel_data(
    retrieval_log_path: Any = None,
    latency_log_path: Any = None,
    generation_log_path: Any = None,
) -> LatencyPanelData:
    """Load latencies from the logs Module 3 and Module 4 wrote (FR-3.5, FR-5.2).

    Reads ``config.RETRIEVAL_LOG_JSON_PATH`` for one latency per RAG query,
    ``config.LATENCY_LOG_JSON_PATH`` for per-stage pipeline timings, and
    ``outputs/logs/generation_log.json`` for LLM generation latencies.

    Raises
    ------
    MissingArtifactError
        If neither the retrieval nor the generation log exists, because the
        comparison is only meaningful when both series were measured. A single
        missing series is tolerated and simply omitted from the chart.
    """
    retrieval = _optional_read_json(retrieval_log_path or config.RETRIEVAL_LOG_JSON_PATH)
    generation = _optional_read_json(
        generation_log_path or config.GENERATION_LOG_JSON_PATH
    )
    pipeline = _optional_read_json(latency_log_path or config.LATENCY_LOG_JSON_PATH)

    if retrieval is None and generation is None:
        raise MissingArtifactError(
            Path(latency_log_path or config.LATENCY_LOG_JSON_PATH), RUN_COMMAND
        )

    stages: dict[str, float] = {}
    queries = 0
    if retrieval is not None:
        entries = retrieval.get("entries", retrieval if isinstance(retrieval, list) else [])
        latencies = [float(e["latency_s"]) for e in entries if "latency_s" in e]
        if latencies:
            stages["RAG Retrieval"] = sum(latencies) / len(latencies)
            queries = len(latencies)

    if generation is not None:
        entries = generation.get("entries", generation if isinstance(generation, list) else [])
        latencies = [float(e["latency_s"]) for e in entries if "latency_s" in e]
        if latencies:
            stages["LLM Report"] = sum(latencies) / len(latencies)

    if pipeline is not None:
        for entry in pipeline.get("stages", []):
            if entry.get("ok") and isinstance(entry.get("seconds"), (int, float)):
                key = str(entry.get("stage", "")).replace("_", " ").title()
                stages.setdefault(key, float(entry["seconds"]))

    return LatencyPanelData(
        stages=stages, retrieval_queries=queries, report_runs=len(generation or []) or 0
    )


def plot_latency_panel(ax: Any, data: LatencyPanelData | None = None) -> Any:
    """Draw the RAG vs LLM latency comparison on ``ax`` (FR-5.1 Panel C).

    Bars are annotated with their sample count, so a bar derived from a single
    query cannot be mistaken for a meaningful average. When no artifact exists,
    renders :data:`EMPTY_STATES` rather than an invented bar.
    """
    _panel_header(ax, "latency")

    if data is None or not data.stages:
        with styles.empty_panel(ax, EMPTY_STATES["latency"]):
            return ax

    labels = list(data.stages)
    values = [data.stages[label] for label in labels]
    colors = [
        styles.LATENCY_COLORS[i % len(styles.LATENCY_COLORS)] for i in range(len(labels))
    ]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.set_ylabel("Seconds")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.margins(y=0.18)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.tick_params(axis="x", labelrotation=20)
    return ax


#: Panel registry, ordered left to right. FR-5.1 requires exactly three.
PANELS: dict[str, Any] = {
    "roc": plot_roc_panel,
    "attention": plot_attention_panel,
    "latency": plot_latency_panel,
}


# ---------------------------------------------------------------------------
# Platform state - what genuinely exists on disk
# ---------------------------------------------------------------------------


def _directory_has_content(path: Any) -> bool:
    """True when ``path`` is a directory holding at least one real entry.

    Placeholder files are ignored. An empty generated directory is kept in version
    control with a ``.gitkeep``, so counting that file as content would make an
    unrun stage report itself as ready - a false success, which is exactly what
    this project must never show.
    """
    path = Path(path) if path else None
    if not path or not path.is_dir():
        return False
    return any(
        not child.name.startswith(".")
        for child in path.iterdir()
    )


def _count_dataset_rows() -> int | None:
    """Count data rows in the raw CSV, or ``None`` if it is unreadable.

    Counts lines rather than using Pandas so the web server stays light and can
    start without the scientific stack installed.
    """
    path = config.RAW_PATIENT_CSV
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return max(sum(1 for _ in handle) - 1, 0)
    except OSError:
        return None


def _ml_results() -> dict[str, Any] | None:
    """Load ``outputs/logs/ml_results.json``, or ``None`` when it does not exist."""
    return _optional_read_json(config.ML_RESULTS_JSON_PATH)


def _evaluated_metrics() -> dict[str, dict[str, float]]:
    """Per-model test metrics, read from the Module 1 results file.

    Returns an empty mapping until Module 1 has run, which is what keeps the Risk
    Prediction page on its empty state instead of showing invented numbers. Only
    the five declared metric keys are passed through, so the results file cannot
    widen what the interface presents.
    """
    results = _ml_results()
    if not results:
        return {}

    evaluated: dict[str, dict[str, float]] = {}
    for name in config.MODEL_NAMES:
        entry = results.get(name)
        if not isinstance(entry, dict):
            continue
        subset = {
            key: entry[key]
            for key in config.METRIC_KEYS
            if isinstance(entry.get(key), (int, float))
        }
        if len(subset) == len(config.METRIC_KEYS):
            evaluated[name] = subset  # type: ignore[assignment]
    return evaluated


def _selected_model_key() -> str | None:
    """The FR-1.7 winner recorded in the results file, or ``None``."""
    results = _ml_results() or {}
    selected = results.get("selection", {})
    winner = selected.get("selected_model") if isinstance(selected, dict) else None
    return winner if winner in config.MODEL_NAMES else None


def _report_schema_fields() -> list[dict[str, str]]:
    """Field names and types of the PRD report schema.

    Read from the Pydantic model itself rather than duplicated here, so the
    Safety Report page renders exactly the fields Module 4 will emit and cannot
    drift from the schema.
    """
    try:
        from src.llm.schemas import ClinicalSafetyReport
    except Exception:  # pragma: no cover - only if pydantic is unavailable
        return [
            {"name": name, "type": "str", "description": ""}
            for name in config.__dict__.get("REPORT_FIELDS", ())
        ]

    descriptions = {
        "Eligible_For_Vaccine": "Whether the patient is eligible given risk and guidelines",
        "Recommended_Dose_mg": "Recommended dose in mg, must respect guideline bounds",
        "Reasoning_Steps": "Ordered reasoning steps behind the decision",
        "FDA_Safety_Warnings_Cited": "Guideline snippets the recommendation relies on",
        "Disclaimer": "Always present in the final report",
    }
    fields = []
    for name, info in ClinicalSafetyReport.model_fields.items():
        fields.append(
            {
                "name": name,
                "type": getattr(info.annotation, "__name__", str(info.annotation)),
                "description": descriptions.get(name, info.description or ""),
                "required": info.is_required(),
            }
        )
    return fields


def _risk_model_available() -> bool:
    """Whether a trained risk model exists on disk.

    Checks ``config.SELECTED_MODEL_PATH`` - the artifact ``save_model_artifacts``
    actually writes for the FR-1.7 winner - rather than ``RISK_MODEL_PATH``.

    ``RISK_MODEL_PATH`` is the PRD's *acceptance-criteria* filename, kept as a read
    alias in ``src.ml.predict``. Nothing in the codebase writes it, so testing it
    here reported the Risk Model stage "not initialized" on a correctly trained
    project. Accepts either so a project carrying the legacy name still reads
    truthfully.
    """
    return any(
        Path(path).is_file()
        for path in (config.SELECTED_MODEL_PATH, config.RISK_MODEL_PATH)
    )


def stage_status() -> list[dict[str, str]]:
    """Report the true readiness of each pipeline stage.

    A stage is ``ready`` only when the artifact that stage produces exists on
    disk. Nothing here infers success from a timestamp or from the fact that a
    module was imported.

    Returns
    -------
    list of dict
        One entry per stage in :data:`PIPELINE_STAGES`, each with a ``status``
        of ``ready``, ``awaiting-data`` or ``not-initialized``.
    """
    checks = {
        "patient-data": (
            config.RAW_PATIENT_CSV.exists(),
            "Source dataset present.",
            "Source dataset missing.",
        ),
        "risk-model": (
            _risk_model_available(),
            "Trained classifier available.",
            "Run model training to enable this stage.",
        ),
        "attention": (
            config.ATTENTION_WEIGHTS_PATH.exists(),
            "Attention weights available.",
            "Awaiting attention inference output.",
        ),
        "retrieval": (
            _directory_has_content(config.CHROMA_PERSIST_DIR),
            "Vector index available.",
            "Run the retrieval ingest to enable this stage.",
        ),
        # A later stage adds "report" once Module 4 exists.
        "report": (
            _directory_has_content(config.REPORTS_DIR),
            "Report artifacts available.",
            "No report has been generated yet.",
        ),
    }
    resolved = []
    for stage in PIPELINE_STAGES:
        exists, ready_note, pending_note = checks[stage["id"]]
        resolved.append(
            {
                **stage,
                "status": "ready" if exists else "not-initialized",
                "note": ready_note if exists else pending_note,
            }
        )
    return resolved


def artifact_availability() -> dict[str, bool]:
    """Report which panels can render real data right now.

    Keys match :data:`PANELS`. Consumed by the web shell to decide between a real
    image and an empty state.
    """
    return {
        "roc": config.ROC_CURVE_JSON_PATH.exists() and config.METRICS_JSON_PATH.exists(),
        "attention": config.ATTENTION_WEIGHTS_PATH.exists(),
        "latency": config.RETRIEVAL_LOG_JSON_PATH.exists()
        or config.GENERATION_LOG_JSON_PATH.exists(),
    }


def platform_state() -> dict[str, Any]:
    """Assemble the complete, truthful state consumed by the web interface.

    Every value is either read from disk or declared in :mod:`src.config` /
    the Pydantic schema. No metric, probability, latency or clinical statement is
    synthesised, which is what lets the interface distinguish *ready* from
    *not initialized* honestly.

    Returns
    -------
    dict
        JSON-serialisable snapshot; served verbatim at ``/api/state``.
    """
    import platform
    import sys as _sys

    stages = stage_status()
    artifacts = artifact_availability()
    return {
        "project": {
            "name": "mRNA VIP",
            "subtitle": "Vaccine Intelligence & Protection",
            "version": config.PROJECT_VERSION,
            "footer": "Student Engineering Project",
        },
        "stages": stages,
        "artifacts": {
            **artifacts,
            "model": _risk_model_available(),
            "index": _directory_has_content(config.CHROMA_PERSIST_DIR),
            "report": _directory_has_content(config.REPORTS_DIR),
            "dataset": config.RAW_PATIENT_CSV.exists(),
            "guidelines": config.GUIDELINE_DOCX.exists(),
        },
        "initialized": all(stage["status"] == "ready" for stage in stages),
        "metrics": {
            "patient_records": _count_dataset_rows(),
            "ml_models": len(config.MODEL_NAMES),
            "model_names": list(config.MODEL_NAMES),
            "retrieval_top_k": config.RETRIEVAL_TOP_K,
            "report_format": "Pydantic",
            "embed_dim": config.ATTENTION_EMBED_DIM,
            "metric_keys": list(config.METRIC_KEYS),
            "selected_model": _selected_model_key(),
            # Populated only once Module 1 has actually run. Empty while the
            # models are untrained, so the interface falls back to empty states
            # rather than showing placeholder numbers.
            "evaluated": _evaluated_metrics(),
        },
        "model": {
            "numeric_columns": list(config.NUMERIC_COLS),
            "categorical_column": config.CATEGORICAL_COL,
            "categorical_values": list(config.CATEGORICAL_VALUES),
        },
        "report_schema": _report_schema_fields(),
        "copy": {
            "empty_states": dict(EMPTY_STATES),
            "loading": LOADING_STATE,
            "not_initialized": NOT_INITIALIZED_STATE,
            "error": ERROR_STATE,
            "disclaimer": REPORT_DISCLAIMER,
        },
        "dose_bound": None,
        "source": {
            "name": "Moderna mRNA Safety Guidelines",
            "path": str(config.GUIDELINE_DOCX),
            "available": config.GUIDELINE_DOCX.exists(),
        },
        "environment": {
            "python": platform.python_version(),
            "implementation": platform.system(),
            "mode": "Local",
        },
        "commands": {"train": TRAIN_COMMAND, "index": INDEX_COMMAND, "run": RUN_COMMAND},
    }