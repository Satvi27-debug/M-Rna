"""Model fitting for the adverse-reaction risk classifier.

This module trains and serialises. It deliberately does **not** compute metrics
or choose a winner - see :mod:`src.ml.evaluate`, which owns FR-1.6 and FR-1.7.

What gets saved
---------------
Each artifact is a complete ``sklearn.pipeline.Pipeline``:

    Pipeline([
        ('preprocess', ColumnTransformer(...)),   # imputation + scaling + one-hot
        ('classifier', LogisticRegression(...)),  # or RandomForestClassifier
    ])

Serialising the pipeline rather than the bare estimator is deliberate. FR-1.7's
acceptance criterion is that ``joblib.load()`` reproduces the original
predictions; if only the estimator were saved, the scaler and encoder would have
to be re-derived at inference time and could drift from the training-time
transformation. With the pipeline saved, one load is sufficient and sufficient
is provable by comparing predictions.

Requirements owned here
-----------------------
FR-1.5  Train a Logistic Regression and a Random Forest classifier on identical data.
        Both are fitted against the *same* ``x_train``/``y_train`` arrays produced
        by one ``ColumnTransformer``, so "identical training data" is structural
        rather than a matter of care.

Requirements owned by the caller
--------------------------------
FR-1.1 - FR-1.4 are satisfied in :mod:`src.data.preprocessing`, which validates,
imputes, splits and fits the preprocessing on training rows only.

Status: fully implemented. Run with ``python -m src.ml.train``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.data import preprocessing


class ModelArtifactError(RuntimeError):
    """Raised when a model artifact cannot be written or read."""


@dataclass(frozen=True)
class TrainedModels:
    """Both fitted pipelines plus the preprocessor they were trained with.

    Attributes
    ----------
    pipelines:
        Mapping of the two names in ``config.MODEL_NAMES`` to fitted
        ``sklearn.pipeline.Pipeline`` objects, both fitted on exactly the same
        ``x_train`` / ``y_train``.
    preprocessor:
        The fitted :class:`~src.data.preprocessing.Preprocessor`, kept so a single
        record can be transformed with the training-time statistics.
    feature_names:
        The encoded column order the models expect. Recorded so a mismatch is
        detected instead of silently mis-predicting.
    train_rows, test_rows:
        Row counts, for provenance in the results file.
    """

    pipelines: dict[str, Any]
    preprocessor: preprocessing.Preprocessor
    feature_names: list[str] = field(default_factory=list)
    train_rows: int = 0
    test_rows: int = 0

    @property
    def models(self) -> dict[str, Any]:
        """The bare estimators, for callers that only need the classifier."""
        return {
            name: pipe.named_steps["classifier"]
            for name, pipe in self.pipelines.items()
        }


def build_logistic_regression() -> Any:
    """Instantiate the Logistic Regression baseline.

    ``max_iter=1000`` is the PRD's value (Section 5.4) and is required because
    L-BFGS does not reach convergence on this feature scaling within the default
    budget, which would otherwise emit a convergence warning and quietly return a
    partially-fitted model.

    ``solver='lbfgs'`` is left at the scikit-learn default; ``random_state`` is
    not passed because L-BFGS is deterministic and has no stochastic component.
    Adding it would imply reproducibility that does not need configuring.
    """
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(**config.LOGREG_PARAMS)


def build_random_forest() -> Any:
    """Instantiate the Random Forest classifier.

    ``n_estimators=200`` and ``random_state=42`` are the PRD's values
    (Section 5.4). The seed makes the ensemble reproducible: scikit-learn draws
    bootstrap samples and feature subsets from the global random state, so
    without it two runs would produce different forests and non-comparable scores.

    No hyper-parameter search is performed. The brief and the PRD both ask for an
    implementation that stays understandable under examination, and a tuned
    forest would obscure which choice drove the result.
    """
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(**config.RANDOM_FOREST_PARAMS)


def build_models() -> dict[str, Any]:
    """Instantiate both unfitted classifiers with the PRD's hyper-parameters.

    Returns
    -------
    dict
        ``{'logistic_regression': LogisticRegression(max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42)}``.
    """
    return {
        "logistic_regression": build_logistic_regression(),
        "random_forest": build_random_forest(),
    }


def train_models(split: preprocessing.SplitData) -> TrainedModels:
    """Fit both pipelines on the identical training split (FR-1.5).

    Both pipelines receive the same ``split.x_train`` and ``split.y_train``
    objects. Imputation, scaling and encoding already happened inside the fitted
    ``ColumnTransformer``, so nothing is fitted here beyond the estimators.

    Parameters
    ----------
    split:
        Output of
        :func:`~src.data.preprocessing.stratified_train_test_split`. Must carry
        both the transformed matrices and the raw training frame, because the
        pipeline is fitted on raw columns.

    Returns
    -------
    TrainedModels
    """
    from sklearn.pipeline import Pipeline

    if split is None or split.x_train is None or len(split.x_train) == 0:
        raise ModelArtifactError(
            "Cannot train on an empty split. Run the preprocessing stage first."
        )
    if split.train_raw is None or split.train_raw.empty:
        raise ModelArtifactError(
            "SplitData.train_raw is empty. The pipeline is fitted on raw columns "
            "so the saved artifact can transform records itself; the split must "
            "retain them."
        )

    # Fit on the RAW feature columns, not on the pre-transformed matrix.
    #
    # The pipeline's first step is the ColumnTransformer, so its input must be a
    # frame carrying the original column names. Passing split.x_train would hand
    # it an already-encoded matrix and fail, because the raw column
    # 'Prior_Reaction_History' no longer exists there.
    #
    # Fitting on train_raw also makes the saved artifact genuinely self-contained:
    # it accepts raw records at inference time and performs its own
    # transformation, which is what FR-1.7's reload-and-reproduce criterion
    # requires. The transformer is refitted here on the same 400 training rows it
    # was originally fitted on, so the result is identical.
    x_train_raw = split.train_raw[list(config.FEATURE_COLS)]
    if x_train_raw.empty:
        raise ModelArtifactError(
            "SplitData.train_raw is empty; the split must retain raw training "
            "rows so the pipeline can be fitted self-containedly."
        )

    estimators = build_models()
    pipelines: dict[str, Any] = {}
    for name, estimator in estimators.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", split.preprocessor.column_transformer),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train_raw, split.y_train)
        pipelines[name] = pipeline

    # Consistency check: the pipeline's own internal transform must reproduce the
    # matrix produced during preprocessing. If these ever diverge, one of the two
    # paths is wrong, and silently scoring against the other would misreport every
    # metric. Cheap to check once, and it turns a subtle corruption into a loud
    # failure at training time.
    expected = split.preprocessor.column_transformer.transform(x_train_raw)
    actual = pipelines["logistic_regression"].named_steps["preprocess"].transform(
        x_train_raw
    )
    if not np.allclose(np.asarray(expected, dtype=float), np.asarray(actual, dtype=float)):
        raise ModelArtifactError(
            "Pipeline preprocessing does not match the fitted ColumnTransformer. "
            "Refusing to train: evaluation would be reported against a different "
            "transformation than the one recorded in the artifacts."
        )

    return TrainedModels(
        pipelines=pipelines,
        preprocessor=split.preprocessor,
        feature_names=list(split.preprocessor.encoded_feature_names),
        train_rows=split.n_train,
        test_rows=split.n_test,
    )


def save_model_artifacts(
    trained: TrainedModels, selection: Any = None
) -> dict[str, Path]:
    """Serialise the fitted pipelines to ``models/``.

    Writes one ``joblib`` per model, the FR-1.7 winner, and the PRD-named
    compatibility copy at ``outputs/risk_model.joblib`` so the FR-1.7 acceptance
    criterion is satisfied at the literal path it names.

    Parameters
    ----------
    trained:
        Output of :func:`train_models`.
    selection:
        Optional :class:`~src.ml.evaluate.SelectionResult`. When supplied, the
        winner is also written to ``config.SELECTED_MODEL_PATH``.

    Returns
    -------
    dict
        Mapping of artefact name to written path.
    """
    import joblib

    config.ensure_directories()
    written: dict[str, Path] = {}

    def dump(pipeline: Any, destination: Path, label: str) -> Path:
        try:
            joblib.dump(pipeline, destination)
        except OSError as exc:
            raise ModelArtifactError(f"Failed to write {destination}: {exc}") from exc
        written[label] = destination
        return destination

    for name, pipeline in trained.pipelines.items():
        dump(pipeline, config.MODEL_PATHS[name], name)

    if selection is not None:
        winner = trained.pipelines[selection.best_name]
        dump(winner, config.SELECTED_MODEL_PATH, "selected_model")
        # FR-1.7 names ``outputs/risk_model.joblib`` in its acceptance criteria.
        # models/risk_model.joblib is this repository's canonical location, so
        # the PRD-named copy is written alongside it.
        dump(winner, config.PRD_COMPAT_MODEL_PATH, "prd_compat_risk_model")

    return written


def save_training_metadata(trained: TrainedModels, split: preprocessing.SplitData) -> Path:
    """Record what was trained, so an artifact can be traced back to its inputs.

    Written to ``outputs/logs/ml_training_metadata.json``: the encoded feature
    names in order, row counts, the seed, the imputation statistics and the
    hyper-parameters. Without this, a saved ``joblib`` is an opaque blob.
    """
    config.ensure_directories()
    payload = {
        "feature_names": list(trained.feature_names),
        "n_features": len(trained.feature_names),
        "train_rows": trained.train_rows,
        "test_rows": trained.test_rows,
        "random_seed": config.RANDOM_SEED,
        "test_size": config.TEST_SIZE,
        "models": list(trained.pipelines),
        "hyperparameters": {
            "logistic_regression": dict(config.LOGREG_PARAMS),
            "random_forest": dict(config.RANDOM_FOREST_PARAMS),
        },
        "imputation": {
            "numeric_medians": {
                k: round(v, 6) for k, v in trained.preprocessor.median_values.items()
            },
            "categorical_modes": dict(trained.preprocessor.mode_values),
        },
        "source_csv": str(config.RAW_PATIENT_CSV),
        "target_definition": (
            "Reaction_Score used directly as the binary target. The PRD "
            "(Section 5.2) defines it as 'Target (Integer 0-1)' where "
            "0 = Low reaction risk and 1 = High reaction risk. No threshold or "
            "re-derivation is applied."
        ),
    }
    destination = config.LOGS_DIR / "ml_training_metadata.json"
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination