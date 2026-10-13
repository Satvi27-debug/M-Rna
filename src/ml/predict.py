"""Inference for the adverse-reaction risk classifiers.

This module is the read-only half of Module 1. It loads serialized artifacts and
scores records; it has no training code path at all, so a bug here cannot
overwrite a deployed model (architecture rule 4: "Keep training and inference are
separate").

Each saved artifact is a complete ``sklearn.pipeline.Pipeline`` holding both the
preprocessing and the estimator, so loading one is sufficient to predict. There
is no separate scaler to keep in sync.

Consumed by the dashboard through ``src.pipeline`` in a later stage.

Contract
--------
:func:`predict_patient` returns **both** models' output, because the PRD does not
declare one model clinically better and the selection policy is a modelling
decision, not a medical one. The FR-1.7 winner is additionally exposed as the
*selected* model for downstream stages, but nothing here discards the runner-up.

The band names come straight from the PRD's own definition of the target
(Section 5.2): ``0 = Low reaction risk``, ``1 = High reaction risk``. They are
translations of the model's output, not new clinical judgements.

This is an engineering output from a student project. Nothing returned here is a
diagnosis, a recommendation, or a statement about the safety of a vaccine.

Status: fully implemented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config
from src.data import preprocessing


class ModelLoadError(RuntimeError):
    """Raised when a saved model artifact cannot be loaded."""


class PredictionError(RuntimeError):
    """Raised when inference fails on an otherwise valid record."""


#: Human-readable band names, keyed by the predicted integer label.
#: Transcribed from PRD Section 5.2.
RISK_LABELS: dict[int, str] = {0: "Low", 1: "High"}


@dataclass(frozen=True)
class ModelPrediction:
    """One model's output for one patient."""

    model_key: str
    model_name: str
    predicted_class: int
    risk_label: str
    probability: float

    @property
    def is_high_risk(self) -> bool:
        return self.predicted_class == 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "prediction": int(self.predicted_class),
            "risk_label": self.risk_label,
            "probability": round(float(self.probability), 6),
        }


@dataclass(frozen=True)
class RiskPrediction:
    """Structured risk output for one patient.

    Attributes
    ----------
    patient_id:
        Identifier carried through from the record, for labelling only. It is
        never a model feature.
    predictions:
        Per-model output for every model named in ``config.MODEL_NAMES``.
    selected_model_key:
        FR-1.7 winner. ``None`` when artifacts are absent, which is a legitimate
        state rather than an error.
    """

    patient_id: str
    predictions: dict[str, ModelPrediction] = field(default_factory=dict)
    selected_model_key: str | None = None
    model_version: str = config.PROJECT_VERSION

    @property
    def selected(self) -> ModelPrediction | None:
        """The FR-1.7 winner's output, if present."""
        if self.selected_model_key is None:
            return None
        return self.predictions.get(self.selected_model_key)

    @property
    def risk_label(self) -> str | None:
        selected = self.selected
        return selected.risk_label if selected else None

    @property
    def risk_probability(self) -> float | None:
        selected = self.selected
        return selected.probability if selected else None

    def agreement(self) -> bool | None:
        """Whether every model returned the same class.

        Reported because a disagreement between a linear and an ensemble model is
        genuinely informative about a borderline record, and hiding it would
        misrepresent the output.

        Returns
        -------
        bool | None
            ``True`` when all models agree, ``False`` when they differ, and
            ``None`` when fewer than two models ran - with one model there is
            nothing to agree or disagree with.
        """
        if len(self.predictions) < 2:
            return None
        classes = {p.predicted_class for p in self.predictions.values()}
        return len(classes) == 1

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable form, matching the shape the prompt specifies."""
        payload: dict[str, Any] = {
            "patient_id": self.patient_id,
            "model_version": self.model_version,
            "selected_model": self.selected_model_key,
            "agreement": self.agreement(),
        }
        for name in config.MODEL_NAMES:
            prediction = self.predictions.get(name)
            if prediction is not None:
                payload[name] = prediction.as_dict()
        return payload


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------


def load_model(model_name: str = "selected", path: Any = None) -> Any:
    """Load a fitted pipeline with ``joblib.load``.

    Parameters
    ----------
    model_name:
        A key from ``config.MODEL_NAMES``, or ``'selected'`` for the FR-1.7
        winner.
    path:
        Explicit artifact path, overriding ``model_name``.

    Returns
    -------
    sklearn.pipeline.Pipeline

    Raises
    ------
    ModelLoadError
        If the artifact is missing or unreadable. The message names the command
        that produces it, so a first-run failure is self-explanatory.
    """
    import joblib

    if path is not None:
        artifact = Path(path)
    elif model_name == "selected":
        artifact = config.SELECTED_MODEL_PATH
        if not artifact.exists():
            artifact = config.RISK_MODEL_PATH
    else:
        artifact = config.MODEL_PATHS.get(model_name, Path(model_name))

    if not Path(artifact).is_file():
        raise ModelLoadError(
            f"Model artifact not found at {artifact}. "
            "Train the models first: python -m src.ml.train"
        )
    try:
        return joblib.load(artifact)
    except Exception as exc:  # noqa: BLE001 - surfaced with actionable context
        raise ModelLoadError(
            f"Failed to load model artifact {artifact}: {exc}. "
            "The file may be corrupt or was written by an incompatible "
            "library version; re-run python -m src.ml.train."
        ) from exc


def load_risk_model(path: Any = None) -> Any:
    """Load the FR-1.7 selected model.

    Defaults to ``config.RISK_MODEL_PATH``, falling back to
  ``config.SELECTED_MODEL_PATH``.

    Raises
    ------
    ModelLoadError
        If neither artifact exists. The acceptance criterion for FR-1.7 is that
        this call reloads and reproduces the original predictions.
    """
    return load_model("selected", path)


def available_models() -> list[str]:
    """Names of the models whose artifacts currently exist on disk."""
    return [name for name, path in config.MODEL_PATHS.items() if path.is_file()]


def _selected_key_from_disk() -> str | None:
    """Read the FR-1.7 winner from the results file, without loading a model."""
    results = config.ML_RESULTS_JSON_PATH
    if not results.is_file():
        return None
    try:
        payload = json.loads(results.read_text(encoding="utf-8"))
        return payload.get("selection", {}).get("selected_model")
    except (json.JSONDecodeError, OSError):
        return None


def _load_available_models() -> dict[str, Any]:
    """Load every model artifact present on disk.

    Returns
    -------
    dict
        Name to fitted pipeline.

    Raises
    ------
    ModelLoadError
        If no artifact exists. An empty result would otherwise let
        :func:`predict_patient` return an empty prediction set, which reads like
        a successful call that happened to produce nothing.
    """
    names = available_models()
    if not names:
        raise ModelLoadError(
            "No model artifacts found. Train the models first: python -m src.ml"
        )
    return {name: load_model(name) for name in names}


def extract_preprocessor(pipeline: Any) -> preprocessing.Preprocessor:
    """Recover a :class:`Preprocessor` from a loaded pipeline.

    The pipeline stores the fitted ``ColumnTransformer``. This wraps it in the
    project dataclass with the feature names, so the inference path uses the same
    transformation objects as training rather than a parallel implementation.
    """
    transformer = None
    if hasattr(pipeline, "named_steps"):
        transformer = pipeline.named_steps.get("preprocess")
    elif hasattr(pipeline, "transform"):
        transformer = pipeline

    if transformer is None:
        raise PredictionError(
            "Loaded pipeline does not contain a 'preprocess' step, so a record "
            "cannot be transformed consistently with training."
        )

    try:
        names = [str(n) for n in transformer.get_feature_names_out()]
    except (AttributeError, ValueError):
        names = []

    return preprocessing.Preprocessor(
        column_transformer=transformer,
        feature_cols=tuple(config.FEATURE_COLS),
        numeric_cols=tuple(config.NUMERIC_COLS),
        categorical_cols=(config.CATEGORICAL_COL,),
        categorical_values={config.CATEGORICAL_COL: list(config.CATEGORICAL_VALUES)},
        encoded_feature_names=names,
    )


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_patient(
    patient_record: dict[str, Any],
    models: dict[str, Any] | None = None,
    preprocessors: dict[str, preprocessing.Preprocessor] | None = None,
) -> RiskPrediction:
    """Score one patient record with every available model.

    Parameters
    ----------
    patient_record:
        Raw feature values keyed by dataset column name, e.g.
        ``{'Patient_ID': 'P0001', 'Age': 58, 'Biomarker_Level': 11.2,
        'Pre_Existing_Conditions': 2, 'Vaccine_Dose_mcg': 120.0,
        'Prior_Reaction_History': 'Mild'}``. ``Patient_ID`` is used for
        labelling only and is never a model feature.
    models:
        Pre-loaded pipelines keyed by model name. Loaded from disk when omitted.
    preprocessors:
        Pre-loaded preprocessors keyed by model name, as returned by
        :func:`extract_preprocessor`. Derived from ``models`` when omitted.

    Returns
    -------
    RiskPrediction
        Both models' predictions plus the FR-1.7 winner when known.

    Raises
    ------
    MissingColumnsError
        If the record lacks a required feature field.
    ModelLoadError
        If no model artifact is available.
    PredictionError
        If the pipeline cannot transform the record.
    """
    if models is None:
        models = _load_available_models()
    elif not models:
        # An empty mapping is a caller error, not an instruction to load from
        # disk. Raising keeps a mistyped argument from silently succeeding.
        raise ModelLoadError(
            "An empty model mapping was supplied, so no prediction can be made. "
            "Pass loaded pipelines, or omit the argument to load from disk."
        )

    if preprocessors is None:
        preprocessors = {
            name: extract_preprocessor(pipeline)
            for name, pipeline in models.items()
        }

    predictions: dict[str, ModelPrediction] = {}
    for name, pipeline in models.items():
        preprocessor = preprocessors.get(name)
        if preprocessor is None:
            continue
        try:
            # Build a RAW one-row frame, because the pipeline's first step is the
            # ColumnTransformer and expects the original column names. Feeding it
            # the pre-encoded matrix would fail on 'Prior_Reaction_History', which
            # no longer exists once one-hot encoding has run.
            raw_row = preprocessing.build_single_record_frame(patient_record)
            predicted_class = int(pipeline.predict(raw_row)[0])
            probability = float(pipeline.predict_proba(raw_row)[0][1])
        except preprocessing.PatientDataValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced with actionable context
            raise PredictionError(
                f"Model '{name}' failed to score the record: {exc}"
            ) from exc

        predictions[name] = ModelPrediction(
            model_key=name,
            model_name=config.MODEL_DISPLAY_NAMES.get(name, name),
            predicted_class=predicted_class,
            risk_label=RISK_LABELS.get(predicted_class, "Unknown"),
            probability=probability,
        )

    return RiskPrediction(
        patient_id=str(patient_record.get(config.ID_COL, "unknown")),
        predictions=predictions,
        selected_model_key=_selected_key_from_disk(),
    )


def predict_risk(
    patient_record: dict[str, Any],
    model: Any,
    preprocessor: preprocessing.Preprocessor | None = None,
) -> RiskPrediction:
    """Score one patient with a single already-loaded model.

    A convenience wrapper for callers that hold one pipeline, e.g. a test checking
    that a reloaded artifact reproduces the original run (FR-1.7).

    Parameters
    ----------
    patient_record:
        Raw feature values keyed by dataset column name.
    model:
        A fitted pipeline.
    preprocessor:
        Pre-loaded preprocessor. Recovered from the pipeline when omitted.

    Returns
    -------
    RiskPrediction
        Carrying a single model's output, with ``selected_model_key`` set to it.
    """
    key = "selected"
    if preprocessor is None:
        preprocessor = extract_preprocessor(model)
    result = predict_patient(
        patient_record, models={key: model}, preprocessors={key: preprocessor}
    )
    return RiskPrediction(
        patient_id=result.patient_id,
        predictions=result.predictions,
        selected_model_key=key,
    )


def predict_patient_batch(
    records: list[dict[str, Any]],
    models: dict[str, Any] | None = None,
    preprocessors: dict[str, preprocessing.Preprocessor] | None = None,
) -> list[RiskPrediction]:
    """Score several records with every available model.

    Loads each artifact once and scores every record against it, rather than
    reloading a pipeline per record, so batch scoring of the full dataset stays
    fast.
    """
    if models is None:
        models = _load_available_models()

    if preprocessors is None:
        preprocessors = {
            name: extract_preprocessor(pipeline)
            for name, pipeline in models.items()
        }

    return [
        predict_patient(record, models=models, preprocessors=preprocessors)
        for record in records
    ]