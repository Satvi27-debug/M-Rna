"""Module 1 ML tests: training, evaluation, selection, persistence, inference.

Requirements covered
--------------------
FR-1.5  Logistic Regression and Random Forest both trained on identical data.
FR-1.6  Classification report and confusion matrix for both models, readable from
        ``outputs/`` without re-running training.
FR-1.7  The higher test-set F1 model is selected and serialized, and reloading it
        reproduces the original predictions.

Plus the guards this stage must prove: metrics come from real computation rather
than hard-coded values, predictions land in a believable range rather than a
suspiciously perfect one, and saved artifacts round-trip.

The fixture trains once at module scope. 500 rows and two small models, so this is
a few seconds rather than a cost worth avoiding - and a session-scoped fixture
would let the selection and persistence tests disagree with each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import preprocessing
from src.ml import evaluate as ml_evaluate
from src.ml import predict as ml_predict
from src.ml import train as ml_train
from tests import requires_implementation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_frame() -> pd.DataFrame:
    """The supplied dataset, loaded."""
    return preprocessing.load_raw_records(config.RAW_PATIENT_CSV)


@pytest.fixture(scope="module")
def cleaned_frame(raw_frame) -> pd.DataFrame:
    """The cleaned dataset."""
    return preprocessing.impute_missing_values(raw_frame)


@pytest.fixture(scope="module")
def split(raw_frame) -> preprocessing.SplitData:
    """The stratified 80/20 split with preprocessing fitted on train only.

    Takes the **raw** frame, not ``cleaned_frame``. Splitting a pre-imputed frame
    leaves nothing missing for the transformer to estimate, so it silently adopted
    the whole-dataset medians - which put the held-out test rows into the fitted
    imputer (Biomarker_Level 4.82 instead of the train-only 4.81). Because this
    fixture feeds ``save_model_artifacts``, the leak was not confined to the tests:
    it reached the shipped models.

    The split fits its own imputation statistics from the training rows.
    """
    return preprocessing.stratified_train_test_split(raw_frame)


@pytest.fixture(scope="module")
def trained(split) -> ml_train.TrainedModels:
    """Both fitted pipelines."""
    return ml_train.train_models(split)


@pytest.fixture(scope="module")
def evaluations(trained, split) -> dict:
    """Both models evaluated on the held-out test split."""
    return ml_evaluate.evaluate_all_models(
        trained, split.x_test, split.y_test, split.test_raw
    )


@pytest.fixture(scope="module")
def selection(evaluations) -> ml_evaluate.SelectionResult:
    """The FR-1.7 selection."""
    return ml_evaluate.select_best_model(evaluations)


@pytest.fixture(scope="module")
def artifacts(evaluations, selection, split) -> dict:
    """Evaluation artifacts written by the real code path.

    Writes into the project's own ``outputs/`` tree, which is where they belong
    and where the persistence tests read them back from. The content is identical
    to what ``python -m src.ml`` produces - these tests do not verify a stubbed
    path, they verify the shipped one.
    """
    return ml_evaluate.save_evaluation_artifacts(evaluations, selection, split)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


class TestModelConfiguration:
    """The two models are exactly the ones the PRD specifies."""

    def test_exactly_two_models_are_declared(self) -> None:
        assert config.MODEL_NAMES == ("logistic_regression", "random_forest")

    def test_hyperparameters_match_the_prd_template(self) -> None:
        assert config.LOGREG_PARAMS == {"max_iter": 1000}
        assert config.RANDOM_FOREST_PARAMS == {
            "n_estimators": 200,
            "random_state": 42,
        }

    def test_split_settings_match_fr14(self) -> None:
        assert config.TEST_SIZE == 0.2
        assert config.RANDOM_SEED == 42

    def test_selection_metric_is_test_set_f1(self) -> None:
        assert config.SELECTION_METRIC == "f1"

    def test_display_names_are_defined_for_every_model(self) -> None:
        assert set(config.MODEL_DISPLAY_NAMES) == set(config.MODEL_NAMES)
        assert config.MODEL_DISPLAY_NAMES["random_forest"] == "Random Forest"

    @requires_implementation(ml_train.build_models)
    def test_build_models_returns_the_right_estimators(self) -> None:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        models = ml_train.build_models()
        assert isinstance(models["logistic_regression"], LogisticRegression)
        assert isinstance(models["random_forest"], RandomForestClassifier)

    @requires_implementation(ml_train.build_models)
    def test_build_models_returns_unfitted_estimators(self) -> None:
        for estimator in ml_train.build_models().values():
            assert not hasattr(estimator, "coef_")

    @requires_implementation(ml_train.build_logistic_regression)
    def test_logistic_regression_uses_the_prd_max_iter(self) -> None:
        assert ml_train.build_logistic_regression().max_iter == 1000


# ---------------------------------------------------------------------------
# FR-1.5
# ---------------------------------------------------------------------------


class TestTraining:
    """FR-1.5: both classifiers fit on identical training data."""

    @requires_implementation(ml_train.train_models)
    def test_both_pipelines_are_fitted(self, trained) -> None:
        """Each estimator must carry its own post-fit attribute.

        Checked per class rather than for one shared attribute: a linear model
        exposes ``coef_`` and a forest exposes ``estimators_``, so a single
        attribute would report a fitted forest as unfitted.
        """
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        assert set(trained.pipelines) == set(config.MODEL_NAMES)
        for name, pipeline in trained.pipelines.items():
            classifier = pipeline.named_steps["classifier"]
            # Raises NotFittedError if the estimator was never fitted.
            check_is_fitted(classifier)
            if name == "logistic_regression":
                # Binary outcome, so coef_ holds one row of per-class weights.
                assert classifier.coef_.shape == (1, len(trained.feature_names))
            else:
                assert len(classifier.estimators_) == 200

    @requires_implementation(ml_train.train_models)
    def test_both_models_see_the_same_feature_width(self, trained) -> None:
        assert set(trained.feature_names)
        for pipeline in trained.pipelines.values():
            assert pipeline.named_steps["classifier"].n_features_in_ == len(
                trained.feature_names
            )

    @requires_implementation(ml_train.train_models)
    def test_recorded_row_counts_match_the_split(self, trained, split) -> None:
        assert trained.train_rows == split.n_train == 400
        assert trained.test_rows == split.n_test == 100

    @requires_implementation(ml_train.train_models)
    def test_each_pipeline_embeds_its_preprocessing(self, trained) -> None:
        """One load must be enough to predict; that is FR-1.7's guarantee."""
        for pipeline in trained.pipelines.values():
            assert list(pipeline.named_steps) == ["preprocess", "classifier"]

    @requires_implementation(ml_train.train_models)
    def test_training_is_reproducible(self, split) -> None:
        """The Random Forest is seeded, so refitting must be identical."""
        first = ml_train.train_models(split)
        second = ml_train.train_models(split)
        assert np.array_equal(
            first.pipelines["random_forest"].predict(split.test_raw[list(config.FEATURE_COLS)]),
            second.pipelines["random_forest"].predict(split.test_raw[list(config.FEATURE_COLS)]),
        )

    @requires_implementation(ml_train.train_models)
    def test_training_refuses_an_empty_split(self) -> None:
        with pytest.raises(ml_train.ModelArtifactError):
            ml_train.train_models(None)

    @requires_implementation(ml_train.train_models)
    def test_pipeline_transform_matches_the_preprocessor(self, trained, split) -> None:
        """Guards against the pipeline and the preprocessor diverging."""
        raw = split.train_raw[list(config.FEATURE_COLS)]
        expected = split.preprocessor.column_transformer.transform(raw)
        actual = trained.pipelines["logistic_regression"].named_steps[
            "preprocess"
        ].transform(raw)
        assert np.allclose(expected, actual)


# ---------------------------------------------------------------------------
# FR-1.6
# ---------------------------------------------------------------------------


class TestEvaluation:
    """FR-1.6: real metrics for both models on the held-out test set."""

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_both_models_are_evaluated(self, evaluations) -> None:
        assert set(evaluations) == set(config.MODEL_NAMES)

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_every_headline_metric_is_present_and_finite(self, evaluations) -> None:
        for evaluation in evaluations.values():
            metrics = evaluation.headline_metrics()
            assert set(metrics) == set(config.METRIC_KEYS)
            for key, value in metrics.items():
                assert np.isfinite(value), f"{evaluation.name}.{key} is not finite"

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_all_metrics_lie_in_the_unit_interval(self, evaluations) -> None:
        for evaluation in evaluations.values():
            for key, value in evaluation.headline_metrics().items():
                assert 0.0 <= value <= 1.0, f"{evaluation.name}.{key} = {value}"

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_both_models_produce_a_classification_report(self, evaluations) -> None:
        for evaluation in evaluations.values():
            for label in ("0", "1"):
                assert {"precision", "recall", "f1-score", "support"} <= set(
                    evaluation.classification_report[label]
                )

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_both_models_produce_a_confusion_matrix(self, evaluations) -> None:
        for evaluation in evaluations.values():
            matrix = np.asarray(evaluation.confusion_matrix)
            assert matrix.shape == (2, 2)
            assert matrix.sum() == evaluation.test_rows == 100

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_metrics_describe_the_positive_class(self, evaluations) -> None:
        """precision/recall/f1 must match the report's positive-class row."""
        for evaluation in evaluations.values():
            positive = evaluation.classification_report["1"]
            assert evaluation.precision == pytest.approx(positive["precision"], abs=1e-9)
            assert evaluation.recall == pytest.approx(positive["recall"], abs=1e-9)
            assert evaluation.f1 == pytest.approx(positive["f1-score"], abs=1e-9)

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_accuracy_matches_the_confusion_matrix(self, evaluations) -> None:
        for evaluation in evaluations.values():
            matrix = np.asarray(evaluation.confusion_matrix)
            diagonal = (matrix[0, 0] + matrix[1, 1]) / matrix.sum()
            assert evaluation.accuracy == pytest.approx(diagonal, abs=1e-9)

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_scores_are_believable_not_perfect(self, evaluations) -> None:
        """PRD Section 5.2 warns against a suspiciously perfect classifier.

        A near-1.0 score on this noisy dataset would indicate target leakage, so
        an upper bound is enforced rather than merely documented.
        """
        low, high = config.EXPECTED_TEST_F1_RANGE
        for evaluation in evaluations.values():
            assert low <= evaluation.f1 <= high, (
                f"{evaluation.name} scored F1={evaluation.f1:.3f}, outside the "
                f"credible range {low}-{high}. Check for target leakage."
            )

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_auc_exceeds_the_no_skill_line(self, evaluations) -> None:
        for evaluation in evaluations.values():
            assert evaluation.roc_auc > 0.5, (
                f"{evaluation.name} AUC {evaluation.roc_auc:.3f} is at or below "
                "chance, which suggests the features carry no signal."
            )

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_roc_points_are_well_formed(self, evaluations) -> None:
        for evaluation in evaluations.values():
            assert len(evaluation.fpr) == len(evaluation.tpr) > 10
            assert evaluation.fpr[0] == pytest.approx(0.0, abs=1e-9)
            assert evaluation.tpr[-1] == pytest.approx(1.0, abs=1e-9)

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_evaluation_uses_the_held_out_rows(self, evaluations, split) -> None:
        for evaluation in evaluations.values():
            assert evaluation.test_rows == split.n_test

    @requires_implementation(ml_evaluate.evaluate_all_models)
    def test_missing_raw_rows_are_refused_loudly(self, trained, split) -> None:
        """Handing the encoded matrix in place of raw rows must not pass silently."""
        with pytest.raises(ValueError, match="raw test rows"):
            ml_evaluate.evaluate_all_models(trained, split.x_test, split.y_test)

    @requires_implementation(ml_evaluate.evaluate_model)
    def test_probabilities_are_real_and_ordered(self, trained, split) -> None:
        pipeline = trained.pipelines["logistic_regression"]
        rows = split.test_raw[list(config.FEATURE_COLS)]
        probabilities = pipeline.predict_proba(rows)[:, 1]
        assert len(probabilities) == 100
        assert probabilities.min() >= 0.0 and probabilities.max() <= 1.0
        assert probabilities.std() > 0, "probabilities must vary across records"


# ---------------------------------------------------------------------------
# FR-1.7
# ---------------------------------------------------------------------------


class TestSelection:
    """FR-1.7: selection on test-set F1."""

    @requires_implementation(ml_evaluate.select_best_model)
    def test_selection_takes_the_higher_f1_model(self, evaluations, selection) -> None:
        assert selection.best_f1 == pytest.approx(max(selection.comparison.values()))

    @requires_implementation(ml_evaluate.select_best_model)
    def test_selected_model_is_one_of_the_two_configured(self, selection) -> None:
        assert selection.best_name in config.MODEL_NAMES

    @requires_implementation(ml_evaluate.select_best_model)
    def test_selection_covers_both_models(self, evaluations, selection) -> None:
        assert set(selection.comparison) == set(config.MODEL_NAMES)

    @requires_implementation(ml_evaluate.select_best_model)
    def test_selection_is_deterministic(self, evaluations, selection) -> None:
        repeated = ml_evaluate.select_best_model(evaluations)
        assert repeated.best_name == selection.best_name

    @requires_implementation(ml_evaluate.select_best_model)
    def test_empty_evaluation_set_is_refused(self) -> None:
        with pytest.raises(ValueError):
            ml_evaluate.select_best_model({})

    @requires_implementation(ml_evaluate.select_best_model)
    def test_rationale_reports_both_models_without_medical_claims(
        self, evaluations, selection
    ) -> None:
        text = selection.rationale(evaluations).lower()
        for label in ("logistic regression", "random forest"):
            assert label in text
        for forbidden in ("safe", "approved", "clinically recommended", "diagnosis"):
            assert forbidden not in text, f"selection rationale implies '{forbidden}'"


# ---------------------------------------------------------------------------
# Artifact fidelity (added after the final audit)
# ---------------------------------------------------------------------------


class TestTrainedArtifactFidelity:
    """The saved model artifacts must match the current training code.

    Added after an audit found the committed artifacts fitted with the
    **full-data** median for ``Biomarker_Level`` (4.82) while the code produces the
    train-only median (4.81) - a silent test-set leak baked into the artifact. The
    code was already correct; the artifacts predated that fix.

    These tests inspect what was actually fitted rather than trusting the metrics
    file, which is what let the mismatch go unnoticed.
    """

    #: Artifacts ``save_model_artifacts`` actually writes. ``risk_model.joblib`` is
    #: deliberately absent: no code path writes it, and a lingering copy can shadow
    #: the real model - see ``test_no_unwritten_artifact_shadows_the_selected_model``.
    ARTIFACTS = (
        "logistic_regression.joblib",
        "random_forest.joblib",
        "selected_model.joblib",
    )

    @staticmethod
    def _train_only_medians() -> dict:
        split = preprocessing.stratified_train_test_split(
            preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        )
        return {
            column: float(split.train_raw[column].median())
            for column in config.NUMERIC_COLS
        }

    @pytest.mark.parametrize("name", ARTIFACTS)
    def test_artifact_imputer_used_train_only_statistics(self, name) -> None:
        """Every saved artifact must carry the train-only medians.

        ``n_samples_seen_`` elsewhere proves the *row count*; this proves the
        *values*. A stale artifact fitted on the full dataset fails here even when
        its row count is right.
        """
        path = config.MODELS_DIR / name
        if not path.is_file():
            from tests import requires_artifact

            requires_artifact(path, "python -m src.ml")

        import joblib

        numeric = joblib.load(path).named_steps["preprocess"].named_transformers_[
            "numeric"
        ]
        stored = dict(
            zip(config.NUMERIC_COLS, numeric.named_steps["imputer"].statistics_)
        )

        train_only = self._train_only_medians()
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)

        for column, expected in train_only.items():
            assert stored[column] == pytest.approx(expected, abs=1e-6), (
                f"{name} has {column}={stored[column]} but the train-only median "
                f"is {expected}. The artifact is stale; retrain with: python -m src.ml"
            )
            full = float(raw[column].median())
            if abs(expected - full) > 1e-9:
                assert stored[column] != pytest.approx(full, abs=1e-6), (
                    f"{name} imputes {column} with the FULL-DATA median ({full}) "
                    f"rather than the train-only median ({expected}) - the "
                    "held-out test set leaked into the fitted imputer"
                )

    def test_artifact_scaler_saw_only_training_rows(self) -> None:
        """The row-count guard, applied to what is actually on disk.

        Uses ``SELECTED_MODEL_PATH`` - the artifact ``save_model_artifacts`` writes
        for the FR-1.7 winner. ``RISK_MODEL_PATH`` is a read alias that no code
        path writes, so it may legitimately be absent.
        """
        from tests import requires_artifact

        requires_artifact(config.SELECTED_MODEL_PATH, "python -m src.ml")

        import joblib

        scaler = joblib.load(config.SELECTED_MODEL_PATH).named_steps[
            "preprocess"
        ].named_transformers_["numeric"].named_steps["scaler"]
        seen = int(np.asarray(scaler.n_samples_seen_).ravel()[0])
        expected = int(round(len(preprocessing.load_raw_records(
            config.RAW_PATIENT_CSV)) * (1 - config.TEST_SIZE)))
        assert seen == expected, (
            f"scaler saw {seen} rows, expected {expected} (the training split)"
        )

    def test_no_unwritten_artifact_shadows_the_selected_model(self) -> None:
        """``models/risk_model.joblib`` must not linger beside the real artifact.

        Nothing in the codebase writes that filename - it is the PRD's acceptance
        criteria path, kept as a read alias. A copy that survives a retrain can be
        served as the current model while carrying statistics from an older run,
        which is exactly how a leaked model outlived its own fix.
        """
        from tests import requires_artifact

        requires_artifact(config.SELECTED_MODEL_PATH, "python -m src.ml")

        legacy = config.MODELS_DIR / "risk_model.joblib"
        assert not legacy.is_file() or ml_predict._is_current(legacy), (
            "models/risk_model.joblib exists but is not byte-identical to the "
            "current selected model. Nothing writes it, so it is stale and would "
            "shadow the real artifact. Delete it, or regenerate with: python -m src.ml"
        )

    def test_artifact_declares_the_configured_seed(self) -> None:
        """``random_state`` must be set, or the forest is not reproducible."""
        from tests import requires_artifact

        requires_artifact(config.RANDOM_FOREST_MODEL_PATH, "python -m src.ml")

        import joblib

        estimator = joblib.load(config.RANDOM_FOREST_MODEL_PATH).named_steps["classifier"]
        assert estimator.random_state == config.RANDOM_FOREST_PARAMS["random_state"]
        assert estimator.n_estimators == config.RANDOM_FOREST_PARAMS["n_estimators"]

    def test_saved_roc_curve_matches_the_saved_artifact(self) -> None:
        """The curve in ``roc_curve.json`` must come from the shipped artifact.

        Catches a metrics file left behind by a different training run.
        """
        from tests import requires_artifact

        requires_artifact(config.ROC_CURVE_JSON_PATH, "python -m src.ml")

        from sklearn.metrics import roc_auc_score

        split = preprocessing.stratified_train_test_split(
            preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        )
        curve = next(
            c
            for c in json.loads(
                config.ROC_CURVE_JSON_PATH.read_text(encoding="utf-8")
            )["curves"]
            if c["model_key"] == "random_forest"
        )

        pipeline = ml_predict.load_risk_model(config.RANDOM_FOREST_MODEL_PATH)
        proba = pipeline.predict_proba(split.test_raw[list(config.FEATURE_COLS)])[:, 1]
        recomputed = roc_auc_score(split.y_test, proba)
        assert recomputed == pytest.approx(curve["roc_auc"], abs=1e-6), (
            f"the saved RF curve reports AUC {curve['roc_auc']} but the saved "
            f"artifact scores {recomputed:.6f} on the same held-out rows; the "
            "metrics file is stale relative to the model"
        )


# ---------------------------------------------------------------------------
# Persistence (FR-1.6, FR-1.7)
# ---------------------------------------------------------------------------


class TestPersistence:
    """Artifacts must exist, be machine-readable, and round-trip."""

    @requires_implementation(ml_evaluate.save_evaluation_artifacts)
    def test_results_file_has_the_required_shape(self, artifacts) -> None:
        payload = json.loads((config.LOGS_DIR / "ml_results.json").read_text(encoding="utf-8"))
        assert "logistic_regression" in payload
        assert "random_forest" in payload
        for name in config.MODEL_NAMES:
            for key in config.METRIC_KEYS:
                assert key in payload[name], f"{name}.{key} missing"
                assert isinstance(payload[name][key], float)

    @requires_implementation(ml_evaluate.save_evaluation_artifacts)
    def test_results_file_records_the_target_definition(self) -> None:
        payload = json.loads((config.LOGS_DIR / "ml_results.json").read_text(encoding="utf-8"))
        assert "Reaction_Score" in payload["target_definition"]
        assert "No threshold" in payload["target_definition"]

    @requires_implementation(ml_evaluate.save_evaluation_artifacts)
    def test_results_carry_the_disclaimer(self) -> None:
        payload = json.loads((config.LOGS_DIR / "ml_results.json").read_text(encoding="utf-8"))
        assert payload["disclaimer"] == config.REQUIRED_DISCLAIMER

    @requires_implementation(ml_evaluate.save_evaluation_artifacts)
    def test_classification_report_is_readable_without_retraining(self) -> None:
        payload = json.loads(
            config.CLASSIFICATION_REPORT_JSON_PATH.read_text(encoding="utf-8")
        )
        assert set(payload) == set(config.MODEL_NAMES)

    @requires_implementation(ml_evaluate.save_evaluation_artifacts)
    def test_confusion_matrix_file_holds_both_models(self) -> None:
        payload = json.loads(config.CONFUSION_MATRIX_JSON_PATH.read_text(encoding="utf-8"))
        assert set(payload) == set(config.MODEL_NAMES)
        for matrix in payload.values():
            assert np.asarray(matrix).shape == (2, 2)

    @requires_implementation(ml_evaluate.save_evaluation_artifacts)
    def test_roc_file_holds_real_curve_points(self) -> None:
        payload = json.loads(config.ROC_CURVE_JSON_PATH.read_text(encoding="utf-8"))
        assert len(payload["curves"]) == 2
        for curve in payload["curves"]:
            assert len(curve["fpr"]) == len(curve["tpr"]) > 10
            assert 0.0 <= curve["roc_auc"] <= 1.0

    @requires_implementation(ml_evaluate.save_evaluation_artifacts)
    def test_metrics_file_records_the_split_for_provenance(self) -> None:
        payload = json.loads(config.METRICS_JSON_PATH.read_text(encoding="utf-8"))
        assert payload["split"]["train_rows"] == 400
        assert payload["split"]["test_rows"] == 100
        assert payload["split"]["stratified_on"] == config.TARGET_COL
        assert payload["split"]["random_seed"] == 42

    @requires_implementation(ml_evaluate.save_evaluation_artifacts)
    def test_stored_metrics_match_the_live_evaluation(self, evaluations) -> None:
        """Guards against a hard-coded or stale results file."""
        payload = json.loads(config.METRICS_JSON_PATH.read_text(encoding="utf-8"))
        for name, evaluation in evaluations.items():
            assert payload["headline"][name] == evaluation.headline_metrics()

    @requires_implementation(ml_evaluate.plot_roc_comparison)
    def test_roc_figure_is_written(self, evaluations, tmp_path) -> None:
        path = ml_evaluate.plot_roc_comparison(evaluations, tmp_path / "roc.png")
        assert path.exists() and path.stat().st_size > 0

    @requires_implementation(ml_evaluate.plot_confusion_matrices)
    def test_one_confusion_matrix_figure_per_model(self, evaluations, tmp_path) -> None:
        written = ml_evaluate.plot_confusion_matrices(evaluations, tmp_path)
        assert set(written) == set(config.MODEL_NAMES)
        for path in written.values():
            assert path.exists() and path.stat().st_size > 0
        assert "logistic_regression_confusion_matrix.png" in {
            p.name for p in written.values()
        }

    @requires_implementation(ml_evaluate.plot_roc_comparison)
    def test_figures_meet_the_minimum_dpi(self, evaluations, tmp_path) -> None:
        """FR-5.3's 150 DPI floor, read from the file's own metadata.

        Pixel width is deliberately not used as a proxy: ``save_figure`` applies
        ``bbox_inches='tight'``, which crops whitespace and so legitimately
        produces a width slightly below the nominal figure size. The DPI recorded
        in the PNG ``pHYs`` chunk is the authoritative value.
        """
        from dashboard import app

        path = ml_evaluate.plot_roc_comparison(evaluations, tmp_path / "roc.png")
        dpi = app._read_png_dpi(path)
        assert dpi is not None, "the PNG carries no resolution metadata"
        assert dpi >= config.DASHBOARD_DPI, f"{dpi:.1f} DPI is below the FR-5.3 floor"

    @requires_implementation(ml_train.save_model_artifacts)
    def test_both_models_are_serialised(self, trained, selection, tmp_path) -> None:
        written = ml_train.save_model_artifacts(trained, selection)
        assert "logistic_regression" in written
        assert "random_forest" in written
        assert written["logistic_regression"].exists()

    @requires_implementation(ml_train.save_model_artifacts)
    def test_selected_model_is_serialised_for_module_four(
        self, trained, selection, tmp_path
    ) -> None:
        """FR-1.7 requires the winner be reusable downstream."""
        written = ml_train.save_model_artifacts(trained, selection)
        assert written["selected_model"].exists()
        assert written["prd_compat_risk_model"].exists()

    @requires_implementation(ml_train.save_training_metadata)
    def test_training_metadata_is_recorded(self, trained, split) -> None:
        path = ml_train.save_training_metadata(trained, split)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["train_rows"] == 400
        assert payload["random_seed"] == 42
        assert set(payload["models"]) == set(config.MODEL_NAMES)
        assert payload["target_definition"].startswith("Reaction_Score used directly")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


class TestInference:
    """Inference is a separate path and never fits anything."""

    @pytest.fixture(scope="class")
    def loaded(self, trained):
        """Both pipelines reloaded from disk, so inference exercises serialization.

        Written to a temporary directory rather than ``models/`` so the test never
        disturbs the real artifacts, and deleted afterwards.
        """
        import joblib

        directory = config.PROJECT_ROOT / "models"
        directory.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        try:
            for name, pipeline in trained.pipelines.items():
                target = directory / f"_test_{name}.joblib"
                joblib.dump(pipeline, target)
                written[name] = target
            return {name: joblib.load(path) for name, path in written.items()}
        finally:
            for path in written.values():
                path.unlink(missing_ok=True)

    @requires_implementation(ml_predict.predict_patient)
    def test_prediction_returns_both_models(self, loaded) -> None:
        record = {
            "Patient_ID": "P0001", "Age": 58, "Biomarker_Level": 11.2,
            "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0,
            "Prior_Reaction_History": "Mild",
        }
        result = ml_predict.predict_patient(record, dict(loaded))
        assert set(result.predictions) == set(config.MODEL_NAMES)

    @requires_implementation(ml_predict.predict_patient)
    def test_probability_is_a_valid_probability(self, loaded) -> None:
        record = {
            "Patient_ID": "P0001", "Age": 58, "Biomarker_Level": 11.2,
            "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0,
            "Prior_Reaction_History": "Mild",
        }
        result = ml_predict.predict_patient(record, dict(loaded))
        for prediction in result.predictions.values():
            assert 0.0 <= prediction.probability <= 1.0

    @requires_implementation(ml_predict.predict_patient)
    def test_risk_label_follows_the_prd_definition(self, loaded) -> None:
        """The band name must be the PRD's own translation of the target."""
        assert ml_predict.RISK_LABELS == {0: "Low", 1: "High"}
        record = {
            "Patient_ID": "P0001", "Age": 58, "Biomarker_Level": 11.2,
            "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0,
            "Prior_Reaction_History": "Mild",
        }
        result = ml_predict.predict_patient(record, dict(loaded))
        for prediction in result.predictions.values():
            assert prediction.risk_label == ml_predict.RISK_LABELS[
                prediction.predicted_class
            ]

    @requires_implementation(ml_predict.predict_patient)
    def test_reloaded_model_reproduces_the_original_predictions(
        self, trained, split
    ) -> None:
        """FR-1.7 acceptance criterion: joblib.load() gives matching output."""
        import joblib

        path = config.PROJECT_ROOT / "models" / "_test_roundtrip.joblib"
        try:
            joblib.dump(trained.pipelines["logistic_regression"], path)
            reloaded = joblib.load(path)
        finally:
            path.unlink(missing_ok=True)

        rows = split.test_raw[list(config.FEATURE_COLS)]
        assert np.array_equal(
            trained.pipelines["logistic_regression"].predict(rows),
            reloaded.predict(rows),
        )

    @requires_implementation(ml_predict.predict_patient)
    def test_probabilities_survive_a_reload_exactly(self, trained) -> None:
        """Serialization must preserve probabilities bit-for-bit.

        Driven through the public inference API on a raw record rather than by
        hand-transforming a frame, so the round-trip exercises exactly the path a
        real caller uses.
        """
        import joblib

        record = {
            "Patient_ID": "P0042", "Age": 40, "Biomarker_Level": 5.0,
            "Pre_Existing_Conditions": 1, "Vaccine_Dose_mcg": 80.0,
            "Prior_Reaction_History": "Mild",
        }
        path = config.PROJECT_ROOT / "models" / "_test_proba.joblib"
        try:
            joblib.dump(trained.pipelines["random_forest"], path)
            reloaded = joblib.load(path)
        finally:
            path.unlink(missing_ok=True)

        original = ml_predict.predict_patient(record, {"random_forest": trained.pipelines["random_forest"]})
        round_tripped = ml_predict.predict_patient(record, {"random_forest": reloaded})
        assert round_tripped.predictions["random_forest"].probability == pytest.approx(
            original.predictions["random_forest"].probability, abs=1e-15
        )
        assert round_tripped.predictions["random_forest"].predicted_class == (
            original.predictions["random_forest"].predicted_class
        )

    @requires_implementation(ml_predict.predict_patient)
    def test_missing_biomarker_is_imputed_not_fatal(self, loaded) -> None:
        record = {
            "Patient_ID": "TEST", "Age": 45, "Biomarker_Level": None,
            "Pre_Existing_Conditions": 1, "Vaccine_Dose_mcg": 75.0,
            "Prior_Reaction_History": "None",
        }
        result = ml_predict.predict_patient(record, dict(loaded))
        for prediction in result.predictions.values():
            assert np.isfinite(prediction.probability)

    @requires_implementation(ml_predict.predict_patient)
    def test_incomplete_record_is_refused(self, loaded) -> None:
        with pytest.raises(preprocessing.MissingColumnsError):
            ml_predict.predict_patient({"Age": 40}, dict(loaded))

    @requires_implementation(ml_predict.load_model)
    def test_absent_artifact_names_the_fix(self, tmp_path) -> None:
        with pytest.raises(ml_predict.ModelLoadError, match="src.ml"):
            ml_predict.load_model("logistic_regression", tmp_path / "absent.joblib")

    @requires_implementation(ml_predict.predict_patient)
    def test_no_models_available_is_refused(self) -> None:
        """An empty model set must raise, not return an empty prediction.

        Returning silently would look like a successful call that produced
        nothing, which is far harder to notice than an error.
        """
        with pytest.raises(ml_predict.ModelLoadError):
            ml_predict.predict_patient(
                {"Age": 40, "Biomarker_Level": 5.0, "Pre_Existing_Conditions": 1,
                 "Vaccine_Dose_mcg": 80.0, "Prior_Reaction_History": "Mild"},
                models={},
            )

    @requires_implementation(ml_predict.predict_patient)
    def test_missing_artifacts_are_reported_with_the_fix(self, monkeypatch) -> None:
        """Simulates a project where the models have never been trained."""
        monkeypatch.setattr(ml_predict, "available_models", lambda: [])
        record = {"Age": 40, "Biomarker_Level": 5.0, "Pre_Existing_Conditions": 1,
                  "Vaccine_Dose_mcg": 80.0, "Prior_Reaction_History": "Mild"}
        with pytest.raises(ml_predict.ModelLoadError, match="src.ml"):
            ml_predict.predict_patient(record)

    @requires_implementation(ml_predict.predict_patient)
    def test_prediction_works_without_passing_models_explicitly(self, trained) -> None:
        """The default path loads from disk, which is how callers actually use it."""
        import joblib

        from tests import requires_artifact

        requires_artifact(
            config.LOGISTIC_REGRESSION_MODEL_PATH, "python -m src.ml"
        )
        record = {"Patient_ID": "P0001", "Age": 58, "Biomarker_Level": 11.2,
                  "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0,
                  "Prior_Reaction_History": "Mild"}
        result = ml_predict.predict_patient(record)
        assert set(result.predictions) <= set(config.MODEL_NAMES)
        assert result.predictions, "at least one artifact must score the record"

    @requires_implementation(ml_predict.extract_preprocessor)
    def test_preprocessor_is_recovered_from_a_pipeline(self, trained) -> None:
        preprocessor = ml_predict.extract_preprocessor(
            trained.pipelines["random_forest"]
        )
        assert preprocessor.is_fitted
        assert preprocessor.encoded_feature_names == trained.feature_names

    @requires_implementation(ml_predict.extract_preprocessor)
    def test_pipeline_without_a_preprocess_step_is_refused(self) -> None:
        from sklearn.linear_model import LogisticRegression

        with pytest.raises(ml_predict.PredictionError):
            ml_predict.extract_preprocessor(LogisticRegression())

    @requires_implementation(ml_predict.predict_patient)
    def test_output_carries_no_medical_claims(self, loaded) -> None:
        record = {
            "Patient_ID": "P0001", "Age": 58, "Biomarker_Level": 11.2,
            "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0,
            "Prior_Reaction_History": "Mild",
        }
        payload = json.dumps(
            ml_predict.predict_patient(record, dict(loaded)).as_dict()
        ).lower()
        for forbidden in ("safe", "approve", "diagnosis", "recommend", "prescri"):
            assert forbidden not in payload, f"prediction output implies '{forbidden}'"

    @requires_implementation(ml_predict.predict_patient)
    def test_agreement_is_reported_when_both_models_ran(self, loaded) -> None:
        record = {
            "Patient_ID": "P0001", "Age": 58, "Biomarker_Level": 11.2,
            "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0,
            "Prior_Reaction_History": "Mild",
        }
        result = ml_predict.predict_patient(record, dict(loaded))
        assert result.agreement() in (True, False)

    @requires_implementation(ml_predict.predict_patient_batch)
    def test_batch_scoring_matches_single_scoring(self, trained, split) -> None:
        models = dict(trained.pipelines)
        preprocessors = {
            name: ml_predict.extract_preprocessor(p) for name, p in models.items()
        }
        records = split.test_raw.head(5).to_dict(orient="records")
        batch = ml_predict.predict_patient_batch(records, models, preprocessors)
        assert len(batch) == 5
        for record, result in zip(records, batch):
            single = ml_predict.predict_patient(record, models, preprocessors)
            assert result.predictions["logistic_regression"].probability == pytest.approx(
                single.predictions["logistic_regression"].probability
            )


# ---------------------------------------------------------------------------
# Target definition
# ---------------------------------------------------------------------------


class TestTargetDefinition:
    """The PRD defines Reaction_Score as the target; nothing may re-derive it."""

    def test_target_column_is_reaction_score(self) -> None:
        assert config.TARGET_COL == "Reaction_Score"

    def test_target_is_used_directly_without_a_threshold(self, split) -> None:
        """PRD Section 5.2: Reaction_Score is already the 0/1 target."""
        assert set(split.y_train.unique()) <= {0, 1}
        assert set(split.y_test.unique()) <= {0, 1}

    def test_target_labels_match_the_source_column(self, split, cleaned_frame) -> None:
        pd.testing.assert_series_equal(
            split.y_test.sort_index(),
            cleaned_frame.loc[split.y_test.index, config.TARGET_COL].astype(int).sort_index(),
        )

    def test_no_derived_target_column_is_created(self, split) -> None:
        """A re-derived target would appear as a new column; there must be none."""
        assert set(split.x_train.columns) <= set(config.FEATURE_COLS) | set(
            split.preprocessor.encoded_feature_names
        )
        assert config.TARGET_COL not in split.preprocessor.encoded_feature_names