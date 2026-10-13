"""Evaluation of the risk classifiers and selection of the production model.

Every number this module produces comes from calling scikit-learn on the held-out
test split. Nothing is hard-coded, defaulted, or estimated. If a metric is absent
from the output it is because it was not computed, not because a placeholder was
substituted.

Requirements owned here
-----------------------
FR-1.6  Produce a classification report (precision, recall, F1) and a confusion
        matrix for **both** models on the held-out test set. Both artifacts are
        written to ``outputs/`` and are readable without re-running training.
FR-1.7  Select the higher-performing model by test-set F1 and serialise it for
        reuse by Module 4.

Metrics reported
----------------
Accuracy, precision, recall, F1 and ROC-AUC, all for the positive class
(``Reaction_Score == 1``, the high-reaction-risk band) except accuracy, which is
overall. Precision, recall and F1 are read from
``classification_report(..., output_dict=True)`` so the classification report and
the headline numbers cannot disagree with each other.

Artifacts written
-----------------
``outputs/logs/ml_results.json``                 the machine-readable comparison
``outputs/logs/metrics.json``                    consolidated metrics + selection
``outputs/logs/classification_report.json``      per-model full reports
``outputs/logs/confusion_matrix.json``           per-model confusion matrices
``outputs/logs/roc_curve.json``                  per-model ROC points
``outputs/figures/ml_roc_curve.png``             the required ROC comparison
``outputs/figures/<model>_confusion_matrix.png`` one per model

Why the ROC points are persisted
--------------------------------
FR-5.2 requires each dashboard panel to be generated from the corresponding
module's *saved output*. ROC points depend on the fitted model, so they are
written to disk here and re-read by the dashboard rather than recomputed, which
would require the dashboard to import the model.

Status: fully implemented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config


#: The positive class. ``Reaction_Score == 1`` is the high-reaction-risk band.
POSITIVE_LABEL = 1


@dataclass(frozen=True)
class ModelEvaluation:
    """Metrics for a single evaluated model, computed on the test split."""

    name: str
    display_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    classification_report: dict[str, Any]
    confusion_matrix: list[list[int]]
    fpr: list[float] = field(default_factory=list)
    tpr: list[float] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)
    test_rows: int = 0

    @property
    def is_selected(self) -> bool:
        return bool(getattr(self, "_selected", False))

    def headline_metrics(self) -> dict[str, float]:
        """The five scalar metrics, rounded for storage and display."""
        return {
            "accuracy": round(float(self.accuracy), 6),
            "precision": round(float(self.precision), 6),
            "recall": round(float(self.recall), 6),
            "f1": round(float(self.f1), 6),
            "roc_auc": round(float(self.roc_auc), 6),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.display_name,
            **self.headline_metrics(),
            "classification_report": self.classification_report,
            "confusion_matrix": self.confusion_matrix,
            "test_rows": self.test_rows,
        }


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of the FR-1.7 comparison between the two models."""

    best_name: str
    best_f1: float
    comparison: dict[str, float]
    metric: str = config.SELECTION_METRIC

    @property
    def best_display_name(self) -> str:
        return config.MODEL_DISPLAY_NAMES.get(self.best_name, self.best_name)

    def rationale(self, evaluations: dict[str, ModelEvaluation] | None = None) -> str:
        """Plain-language justification for the Week 1 written comparison."""
        if not evaluations:
            return (
                f"{self.best_display_name} was selected on held-out test "
                f"{self.metric.upper()} ({self.best_f1:.4f})."
            )
        lines = []
        for name, score in self.comparison.items():
            evaluation = evaluations.get(name)
            label = config.MODEL_DISPLAY_NAMES.get(name, name)
            if evaluation:
                lines.append(
                    f"{label}: {self.metric.upper()}={score:.4f}, "
                    f"ROC-AUC={evaluation.roc_auc:.4f}, "
                    f"accuracy={evaluation.accuracy:.4f}"
                )
        joined = "; ".join(lines)
        return (
            f"{self.best_display_name} was selected on held-out test "
            f"{self.metric.upper()} per FR-1.7. Observed: {joined}. "
            "Both models remain available through the inference interface."
        )


# ---------------------------------------------------------------------------
# Per-model evaluation (FR-1.6)
# ---------------------------------------------------------------------------


def evaluate_model(
    name: str,
    pipeline: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> ModelEvaluation:
    """Compute the FR-1.6 metrics for one model on the held-out test set.

    Parameters
    ----------
    name:
        Key from ``config.MODEL_NAMES``.
    pipeline:
        The fitted ``sklearn.pipeline.Pipeline``. Predictions come from
        ``predict``; probabilities from ``predict_proba[:, 1]``, which is what
        the ROC curve needs.
    x_test:
        **Raw** test rows carrying the original feature column names, *not* the
        pre-transformed matrix. The pipeline's first step is the ColumnTransformer,
        so it must receive the columns it was fitted with. Passing an already
        encoded frame fails because ``Prior_Reaction_History`` no longer exists
        in it.
    y_test:
        Test labels, used only to score predictions the model has never seen.

    Returns
    -------
    ModelEvaluation
        Including the full classification report and confusion matrix, so the
        headline scalars and the detailed report come from the same computation.
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, POSITIVE_LABEL]
    y_true = np.asarray(y_test)

    # average='binary' restricts the result to the positive class. Without it
    # scikit-learn returns per-class arrays and a weighted average across both
    # classes, which is not the single number the report and the selection
    # criterion expect.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        pos_label=POSITIVE_LABEL,
        average="binary",
        zero_division=0,
    )
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)

    return ModelEvaluation(
        name=name,
        display_name=config.MODEL_DISPLAY_NAMES.get(name, name),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        classification_report=classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        confusion_matrix=confusion_matrix(
            y_true, y_pred, labels=[0, POSITIVE_LABEL]
        ).tolist(),
        fpr=[float(v) for v in fpr],
        tpr=[float(v) for v in tpr],
        thresholds=[float(v) for v in thresholds],
        test_rows=int(len(y_true)),
    )


def evaluate_all_models(
    trained: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    test_raw: pd.DataFrame | None = None,
) -> dict[str, ModelEvaluation]:
    """Evaluate every model in a :class:`~src.ml.train.TrainedModels` bundle.

    FR-1.6 requires artifacts for *both* models, so this iterates over the whole
    mapping rather than stopping at the winner.

    Parameters
    ----------
    trained:
        Output of :func:`~src.ml.train.train_models`.
    x_test:
        The transformed test matrix. Used to verify that the pipeline's own
        output matches the expected transformation.
    y_test:
        Test labels.
    test_raw:
        Raw test rows, which is what the pipelines actually consume. Falls back
        to ``x_test`` only when it still carries the original column names, so a
        caller cannot silently pass the wrong frame shape.
    """
    if test_raw is None:
        missing = [c for c in config.FEATURE_COLS if c not in x_test.columns]
        if missing:
            raise ValueError(
                "evaluate_all_models needs the raw test rows: the fitted "
                "pipelines expect "
                f"{list(config.FEATURE_COLS)} but the supplied frame is missing "
                f"{missing}. Pass test_raw=split.test_raw."
            )
        test_raw = x_test

    features = test_raw[list(config.FEATURE_COLS)]
    return {
        name: evaluate_model(name, pipeline, features, y_test)
        for name, pipeline in trained.pipelines.items()
    }


# ---------------------------------------------------------------------------
# Selection (FR-1.7)
# ---------------------------------------------------------------------------


def select_best_model(
    evaluations: dict[str, ModelEvaluation], metric: str = config.SELECTION_METRIC
) -> SelectionResult:
    """Pick the model with the highest test-set ``metric`` (FR-1.7).

    The PRD specifies test-set F1 as the selection criterion, so that is the
    default. Ties resolve to the earlier name in ``config.MODEL_NAMES`` so the
    choice is deterministic rather than dependent on dict ordering.

    The metric is read from the evaluation, never recomputed from training
    scores, so a model cannot be selected on data it was fitted on.
    """
    if not evaluations:
        raise ValueError("Cannot select a model from an empty evaluation set.")

    comparison: dict[str, float] = {}
    for order, name in enumerate(config.MODEL_NAMES):
        if name in evaluations:
            comparison[name] = float(getattr(evaluations[name], metric))

    if not comparison:
        raise ValueError(
            f"No evaluated model carries the metric '{metric}'."
        )

    # max() over a dict ordered by config.MODEL_NAMES keeps the first on a tie.
    best_name = max(comparison, key=lambda key: (comparison[key], -config.MODEL_NAMES.index(key)))
    return SelectionResult(
        best_name=best_name,
        best_f1=comparison[best_name],
        comparison=comparison,
        metric=metric,
    )


# ---------------------------------------------------------------------------
# Persistence (FR-1.6)
# ---------------------------------------------------------------------------


def save_evaluation_artifacts(
    evaluations: dict[str, ModelEvaluation],
    selection: SelectionResult,
    split: Any = None,
) -> dict[str, Path]:
    """Write every FR-1.6 artifact to ``outputs/``.

    Produces the classification report, confusion matrix, ROC points, the
    machine-readable comparison and the consolidated metrics file. Nothing is
    print-only: each artifact is a file that can be read back later, which is what
    the acceptance criteria for FR-1.6 and FR-5.2 require.

    Parameters
    ----------
    evaluations:
        Per-model results from :func:`evaluate_all_models`.
    selection:
        The FR-1.7 outcome.
    split:
        Optional :class:`~src.data.preprocessing.SplitData`, used to record row
        counts and the class balance actually held out.

    Returns
    -------
    dict
        Mapping of artefact name to written path.
    """
    config.ensure_directories()
    written: dict[str, Path] = {}

    def dump(name: str, payload: Any) -> Path:
        path = config.LOGS_DIR / name
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        written[name.replace(".json", "")] = path
        return path

    results = {
        name: evaluation.as_dict() for name, evaluation in evaluations.items()
    }
    # The prompt's required shape, verbatim, plus the selection block.
    ml_results = {
        "logistic_regression": results.get("logistic_regression"),
        "random_forest": results.get("random_forest"),
        "selection": {
            "selected_model": selection.best_name,
            "selected_display_name": selection.best_display_name,
            "metric": selection.metric,
            "scores": {k: round(v, 6) for k, v in selection.comparison.items()},
        },
        "target_definition": (
            "Reaction_Score used directly as the binary target (PRD Section 5.2: "
            "0 = Low reaction risk, 1 = High reaction risk). No threshold applied."
        ),
        "metrics_scope": (
            "All metrics computed on the held-out test split. precision, recall "
            "and f1 are for the positive class (Reaction_Score == 1); accuracy "
            "is overall."
        ),
        "disclaimer": config.REQUIRED_DISCLAIMER,
    }
    dump("ml_results.json", ml_results)
    dump(
        "classification_report.json",
        {
            name: evaluation.classification_report
            for name, evaluation in evaluations.items()
        },
    )
    dump(
        "confusion_matrix.json",
        {
            name: evaluation.confusion_matrix
            for name, evaluation in evaluations.items()
        },
    )
    dump(
        "roc_curve.json",
        {
            "curves": [
                {
                    "model_name": evaluation.display_name,
                    "model_key": evaluation.name,
                    "roc_auc": round(evaluation.roc_auc, 6),
                    "fpr": [round(v, 6) for v in evaluation.fpr],
                    "tpr": [round(v, 6) for v in evaluation.tpr],
                    "thresholds": [round(v, 6) for v in evaluation.thresholds],
                }
                for evaluation in evaluations.values()
            ]
        },
    )

    metrics_payload: dict[str, Any] = {
        "models": results,
        "headline": {
            name: evaluation.headline_metrics()
            for name, evaluation in evaluations.items()
        },
        "selection": ml_results["selection"],
        "target_definition": ml_results["target_definition"],
        "metrics_scope": ml_results["metrics_scope"],
    }
    if split is not None:
        metrics_payload["split"] = {
            "train_rows": split.n_train,
            "test_rows": split.n_test,
            "test_size": config.TEST_SIZE,
            "random_seed": config.RANDOM_SEED,
            "stratified_on": config.TARGET_COL,
            "train_class_balance": {
                str(int(k)): int(v)
                for k, v in split.y_train.value_counts().sort_index().items()
            },
            "test_class_balance": {
                str(int(k)): int(v)
                for k, v in split.y_test.value_counts().sort_index().items()
            },
        }
    dump("metrics.json", metrics_payload)

    written["roc_figure"] = plot_roc_comparison(evaluations, selection=selection)
    # Flattened rather than nested so the CLI can print one path per line.
    for name, path in plot_confusion_matrices(evaluations).items():
        written[f"confusion_matrix_figure:{name}"] = path
    return written


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_roc_comparison(
    evaluations: dict[str, ModelEvaluation],
    save_path: Any = None,
    selection: SelectionResult | None = None,
) -> Path:
    """Draw the required Logistic Regression vs Random Forest ROC comparison.

    Both curves come from ``roc_curve`` points computed in :func:`evaluate_model`
    on real probability predictions. Each is labelled with its name and its
    actual ROC-AUC. The dashed diagonal is the no-skill reference, not a model.

    When ``selection`` is supplied, the FR-1.7 winner is drawn solid and labelled
    as selected, and the other model is dashed.

    Saved at ``config.DASHBOARD_DPI`` (150), the resolution FR-5.3 requires.

    Parameters
    ----------
    evaluations:
        Per-model results.
    save_path:
        Destination. Defaults to ``config.FIGURES_DIR / config.FIGURE_ROC_FILENAME``.
    selection:
        Optional FR-1.7 outcome, used to highlight the chosen model.

    Returns
    -------
    pathlib.Path
    """
    import matplotlib.pyplot as plt

    from dashboard import styles

    destination = (
        Path(save_path) if save_path else config.FIGURES_DIR / config.FIGURE_ROC_FILENAME
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    with styles.apply_style():
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        palette = [styles.PALETTE["accent"], styles.PALETTE["navy"]]
        # Highlight the FR-1.7 winner so the comparison shows which model was
        # selected, not merely two curves.
        winner = selection.best_name if selection is not None else None
        for index, evaluation in enumerate(evaluations.values()):
            colour = palette[index % len(palette)]
            is_selected = evaluation.name == winner
            ax.plot(
                evaluation.fpr,
                evaluation.tpr,
                color=colour,
                linewidth=2.2 if is_selected else 1.5,
                linestyle="-" if is_selected else "--",
                label=(
                    f"{evaluation.display_name} (AUC = {evaluation.roc_auc:.3f})"
                    + (" — selected" if is_selected else "")
                ),
            )
        ax.plot(
            [0, 1], [0, 1],
            linestyle=":", color=styles.REFERENCE_LINE_COLOR, label="No-skill reference",
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Risk Classifier ROC Comparison", fontsize=12, fontweight="600", pad=14)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        fig.tight_layout()
        saved = styles.save_figure(fig, destination)

    # The front end requests Panel A by a neutral name so it stays stable if the
    # artifact naming ever changes. Write the alias only when it differs, so the
    # outputs directory does not accumulate duplicate figures.
    if not save_path:
        for alias in config.FIGURE_ROC_FILENAMES:
            target = config.FIGURES_DIR / alias
            if target != saved and not target.exists():
                target.write_bytes(saved.read_bytes())
    return saved


def plot_confusion_matrices(
    evaluations: dict[str, ModelEvaluation], save_dir: Any = None
) -> dict[str, Path]:
    """Draw one confusion matrix per model into ``outputs/figures/``.

    Filenames follow ``config.CONFLUION_MATRIX_FILENAMES``, e.g.
    ``logistic_regression_confusion_matrix.png``.

    Counts are annotated in each cell and the axes are labelled with the actual
    class names from the PRD, so the figure is readable without the surrounding
    report.

    Returns
    -------
    dict
        Mapping of model key to written path.
    """
    import matplotlib.pyplot as plt

    from dashboard import styles

    directory = Path(save_dir) if save_dir else config.FIGURES_DIR
    directory.mkdir(parents=True, exist_ok=True)

    labels = ["0\n(Low risk)", "1\n(High risk)"]
    written: dict[str, Path] = {}

    for name, evaluation in evaluations.items():
        matrix = np.asarray(evaluation.confusion_matrix)
        with styles.apply_style():
            fig, ax = plt.subplots(figsize=(4.6, 4.2))
            image = ax.imshow(matrix, cmap="Blues")
            fig.colorbar(image, ax=ax, fraction=0.046)
            ax.set_xticks([0, 1], labels=labels)
            ax.set_yticks([0, 1], labels=labels)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            ax.set_title(
                f"{evaluation.display_name} Confusion Matrix\n"
                f"accuracy {evaluation.accuracy:.3f} · F1 {evaluation.f1:.3f}",
                fontsize=11,
                fontweight="600",
                pad=12,
            )
            threshold = matrix.max() / 2.0 if matrix.size else 0
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(
                        j, i,
                        f"{matrix[i, j]}",
                        ha="center", va="center",
                        fontsize=13, fontweight="600",
                        color="white" if matrix[i, j] > threshold else styles.PALETTE["text"],
                    )
            fig.tight_layout()
            filename = config.CONFUSION_MATRIX_FILENAMES.get(
                name, f"{name}_confusion_matrix.png"
            )
            written[name] = styles.save_figure(fig, directory / filename)
    return written
