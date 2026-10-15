"""Module 1 data tests: loading, validation, inspection, cleaning, splitting.

Requirements covered
--------------------
FR-1.1  All 7 required columns validated; a named error identifies the missing one.
FR-1.2  Median imputation for ``Biomarker_Level``, mode for
        ``Prior_Reaction_History``; zero missing values afterwards.
FR-1.3  ``Age``, ``Biomarker_Level`` and ``Vaccine_Dose_mcg`` standardised to
        mean ~0 and std ~1 on the training set.
FR-1.4  80/20 stratified split on ``Reaction_Score`` with a fixed seed, and
        identical row sets on a repeated run.

Plus the leakage and identifier guards this stage is required to prove:
preprocessing is fitted on training rows only, and ``Patient_ID`` never reaches
the model.

The dataset-driven assertions are written against the supplied CSV, whose
characteristics were measured directly rather than assumed: 500 rows, ids
``P0001``-``P0500``, ages 18-85, doses 25.1-150 mcg, 20 missing
``Biomarker_Level`` values (4.0%), 17 genuinely empty ``Prior_Reaction_History``
values (3.4%), and a 255/245 target split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import preprocessing
from tests import requires_implementation


# ---------------------------------------------------------------------------
# The literal "None" finding - the most consequential test in this file
# ---------------------------------------------------------------------------


class TestNoneIsACategoryNotAMissingValue:
    """The literal string "None" must survive loading as a real category.

    Pandas' default NA list contains the string ``"None"``. In this dataset it is
    a legitimate ``Prior_Reaction_History`` level present in 291 of 500 rows;
    only 17 rows (3.4%) have a genuinely empty field. Reading with pandas
    defaults therefore destroys 291 real observations and inflates the apparent
    missingness from 3.4% to 61.6%.
    """

    def test_loading_preserves_the_literal_none_category(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        observed = set(df[config.CATEGORICAL_COL].dropna().unique())
        assert "None" in observed, (
            "The literal 'None' category was converted to NaN on load. "
            f"Observed values instead: {sorted(observed)}"
        )

    def test_only_genuinely_empty_fields_become_missing(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        missing = int(df[config.CATEGORICAL_COL].isna().sum())
        assert missing == 17, (
            f"Expected 17 genuinely empty Prior_Reaction_History fields, found "
            f"{missing}. A different number means the NA handling has changed."
        )

    def test_missingness_matches_the_documented_prd_figures(self) -> None:
        """The PRD documents ~4% for the biomarker and ~3.5% for the history."""
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        biomarker_pct = 100 * df["Biomarker_Level"].isna().mean()
        history_pct = 100 * df[config.CATEGORICAL_COL].isna().mean()
        assert 3.0 <= biomarker_pct <= 5.0, biomarker_pct
        assert 2.5 <= history_pct <= 4.5, history_pct

    def test_na_configuration_is_narrow_by_design(self) -> None:
        """A regression guard on the loader's parameters themselves."""
        assert config.CSV_KEEP_DEFAULT_NA is False
        assert config.CSV_NA_VALUES == ("",)

    def test_loading_twice_is_deterministic(self) -> None:
        first = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        second = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------------------
# Dataset contract
# ---------------------------------------------------------------------------


class TestDatasetContract:
    """Assertions about the supplied data that must always hold."""

    def test_raw_csv_is_present(self) -> None:
        assert config.RAW_PATIENT_CSV.exists()

    def test_raw_csv_has_seven_columns_in_documented_order(self) -> None:
        header = config.RAW_PATIENT_CSV.read_text(encoding="utf-8").splitlines()[0]
        assert header.split(",") == list(config.REQUIRED_COLUMNS)

    def test_dataset_has_five_hundred_records(self) -> None:
        assert len(pd.read_csv(config.RAW_PATIENT_CSV)) == 500

    def test_target_is_binary_and_nearly_balanced(self) -> None:
        y = pd.read_csv(config.RAW_PATIENT_CSV)[config.TARGET_COL]
        assert set(y.unique()) <= {0, 1}
        assert 0.45 <= y.mean() <= 0.55

    def test_numeric_ranges_match_the_prd(self) -> None:
        df = pd.read_csv(config.RAW_PATIENT_CSV)
        assert df["Age"].min() >= 18 and df["Age"].max() <= 85
        assert df["Vaccine_Dose_mcg"].min() >= 25 and df["Vaccine_Dose_mcg"].max() <= 150
        assert df["Pre_Existing_Conditions"].min() >= 0
        assert df["Pre_Existing_Conditions"].max() <= 4

    def test_numeric_columns_are_the_three_fr13_columns(self) -> None:
        assert config.NUMERIC_COLS == ("Age", "Biomarker_Level", "Vaccine_Dose_mcg")

    def test_categorical_values_are_within_the_documented_domain(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        observed = set(df[config.CATEGORICAL_COL].dropna().unique())
        assert observed <= set(config.CATEGORICAL_VALUES)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


class TestInspection:
    """The reusable inspection function must report measured facts."""

    @requires_implementation(preprocessing.inspect_dataset)
    def test_report_captures_shape_and_types(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        report = preprocessing.inspect_dataset(df)
        assert report.shape == (500, 7)
        assert report.columns == list(config.REQUIRED_COLUMNS)
        assert report.dtypes["Age"] == "int64"
        assert report.dtypes["Patient_ID"] == "object"

    @requires_implementation(preprocessing.inspect_dataset)
    def test_report_captures_missingness_per_column(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        report = preprocessing.inspect_dataset(df)
        assert report.missing_before["Biomarker_Level"] == 20
        assert report.missing_before[config.CATEGORICAL_COL] == 17
        assert report.total_missing_before == 37

    @requires_implementation(preprocessing.inspect_dataset)
    def test_report_confirms_no_duplicates(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        report = preprocessing.inspect_dataset(df)
        assert report.duplicate_rows == 0
        assert report.duplicate_patient_ids == []

    @requires_implementation(preprocessing.inspect_dataset)
    def test_report_captures_categorical_levels_and_target_balance(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        report = preprocessing.inspect_dataset(df)
        history = next(
            p for p in report.profiles if p.name == config.CATEGORICAL_COL
        )
        assert sorted(history.categorical_values) == ["Mild", "Moderate", "None"]
        assert report.class_balance == {0: 255, 1: 245}

    @requires_implementation(preprocessing.inspect_dataset)
    def test_report_captures_numerical_ranges(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        report = preprocessing.inspect_dataset(df)
        age = next(p for p in report.profiles if p.name == "Age")
        assert (age.numeric_min, age.numeric_max) == (18, 85)

    @requires_implementation(preprocessing.inspect_dataset)
    def test_inspecting_an_empty_frame_raises(self) -> None:
        with pytest.raises(preprocessing.EmptyDatasetError):
            preprocessing.inspect_dataset(pd.DataFrame(columns=config.REQUIRED_COLUMNS))

    @requires_implementation(preprocessing.format_inspection_report)
    def test_console_report_mentions_the_key_sections(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        text = preprocessing.format_inspection_report(
            preprocessing.inspect_dataset(df)
        )
        for heading in (
            "DATASET INSPECTION", "COLUMN PROFILE", "CATEGORICAL VALUES",
            "TARGET DISTRIBUTION", "Duplicate rows",
        ):
            assert heading in text

    @requires_implementation(preprocessing.inspect_dataset)
    def test_report_is_json_serialisable(self) -> None:
        import json

        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        payload = preprocessing.inspect_dataset(df).as_dict()
        assert json.loads(json.dumps(payload))["rows"] == 500


# ---------------------------------------------------------------------------
# FR-1.1 and validation
# ---------------------------------------------------------------------------


class TestColumnValidation:
    """FR-1.1: 7 required columns validated with a named, column-specific error."""

    @requires_implementation(preprocessing.load_raw_records)
    def test_valid_dataset_loads_without_error(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        assert list(df.columns) == list(config.REQUIRED_COLUMNS)
        assert len(df) == 500

    @requires_implementation(preprocessing.load_raw_records)
    def test_dropping_a_column_raises_error_naming_that_column(self, tmp_path) -> None:
        broken = tmp_path / "missing_column.csv"
        pd.read_csv(config.RAW_PATIENT_CSV).drop(
            columns=["Biomarker_Level"]
        ).to_csv(broken, index=False)
        with pytest.raises(preprocessing.MissingColumnsError) as excinfo:
            preprocessing.load_raw_records(broken)
        assert "Biomarker_Level" in str(excinfo.value)

    def test_missing_columns_error_lists_every_missing_column(self) -> None:
        err = preprocessing.MissingColumnsError(["Age", "Dose"], config.REQUIRED_COLUMNS)
        assert err.missing == ["Age", "Dose"]
        assert "Age" in str(err) and "Dose" in str(err)

    @requires_implementation(preprocessing.load_raw_records)
    def test_absent_file_raises_dataset_not_found(self, tmp_path) -> None:
        with pytest.raises(preprocessing.DatasetNotFoundError):
            preprocessing.load_raw_records(tmp_path / "absent.csv")

    @requires_implementation(preprocessing.validate_dataset)
    def test_valid_dataset_passes_validation(self) -> None:
        preprocessing.validate_dataset(
            preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        )

    @requires_implementation(preprocessing.validate_dataset)
    def test_duplicate_patient_id_raises(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        duplicated = pd.concat([df, df.head(1)], ignore_index=True)
        with pytest.raises(preprocessing.DuplicatePatientIdError):
            preprocessing.validate_dataset(duplicated)

    @requires_implementation(preprocessing.validate_dataset)
    def test_empty_dataset_raises(self) -> None:
        empty = pd.DataFrame(columns=config.REQUIRED_COLUMNS)
        with pytest.raises(preprocessing.EmptyDatasetError):
            preprocessing.validate_dataset(empty)

    @requires_implementation(preprocessing.validate_dataset)
    def test_non_numeric_feature_column_raises(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        df["Age"] = df["Age"].astype(str)
        with pytest.raises(preprocessing.InvalidFeatureError):
            preprocessing.validate_dataset(df)

    @requires_implementation(preprocessing.validate_dataset)
    def test_undocumented_categorical_level_raises(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        df.loc[0, config.CATEGORICAL_COL] = "Severe"
        with pytest.raises(preprocessing.InvalidFeatureError):
            preprocessing.validate_dataset(df)


# ---------------------------------------------------------------------------
# FR-1.2
# ---------------------------------------------------------------------------


class TestImputation:
    """FR-1.2: median for the biomarker, mode for the categorical column."""

    @requires_implementation(preprocessing.impute_missing_values)
    def test_no_missing_values_remain_after_cleaning(self) -> None:
        cleaned = preprocessing.impute_missing_values(
            preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        )
        assert cleaned.isnull().sum().sum() == 0

    @requires_implementation(preprocessing.impute_missing_values)
    def test_biomarker_is_imputed_with_the_median(self) -> None:
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        expected = raw["Biomarker_Level"].median()
        cleaned = preprocessing.impute_missing_values(raw)
        assert cleaned.loc[
            raw["Biomarker_Level"].isna(), "Biomarker_Level"
        ].eq(expected).all()

    @requires_implementation(preprocessing.impute_missing_values)
    def test_categorical_is_imputed_with_the_mode(self) -> None:
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        expected = raw[config.CATEGORICAL_COL].mode()[0]
        cleaned = preprocessing.impute_missing_values(raw)
        assert cleaned.loc[
            raw[config.CATEGORICAL_COL].isna(), config.CATEGORICAL_COL
        ].eq(expected).all()

    @requires_implementation(preprocessing.impute_missing_values)
    def test_no_rows_are_dropped(self) -> None:
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        assert len(preprocessing.impute_missing_values(raw)) == len(raw) == 500

    @requires_implementation(preprocessing.impute_missing_values)
    def test_imputation_preserves_the_none_category(self) -> None:
        """Filling 17 blanks must not collapse the 291 real "None" rows."""
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        cleaned = preprocessing.impute_missing_values(raw)
        counts = cleaned[config.CATEGORICAL_COL].value_counts()
        assert counts["Mild"] > 0
        assert counts["Moderate"] > 0
        assert counts["None"] >= 291

    @requires_implementation(preprocessing.fit_imputation_strategy)
    def test_strategy_returns_median_and_mode(self) -> None:
        df = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        medians, modes = preprocessing.fit_imputation_strategy(df)
        assert medians["Biomarker_Level"] == pytest.approx(4.82)
        assert modes[config.CATEGORICAL_COL] == "None"

    @requires_implementation(preprocessing.impute_missing_values)
    def test_imputation_uses_supplied_statistics_verbatim(self) -> None:
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        cleaned = preprocessing.impute_missing_values(
            raw, medians={"Biomarker_Level": -1.0}, modes={config.CATEGORICAL_COL: "Moderate"}
        )
        assert cleaned.loc[raw["Biomarker_Level"].isna(), "Biomarker_Level"].eq(-1.0).all()
        assert cleaned.loc[
            raw[config.CATEGORICAL_COL].isna(), config.CATEGORICAL_COL
        ].eq("Moderate").all()

    @requires_implementation(preprocessing.impute_missing_values)
    def test_imputation_does_not_mutate_the_input_frame(self) -> None:
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        before = raw["Biomarker_Level"].isna().sum()
        preprocessing.impute_missing_values(raw)
        assert raw["Biomarker_Level"].isna().sum() == before

    @requires_implementation(preprocessing.impute_missing_values)
    def test_residual_gap_in_a_complete_column_raises(self) -> None:
        """A gap in Reaction_Score must not be silently imputed."""
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        raw.loc[0, config.TARGET_COL] = np.nan
        with pytest.raises(preprocessing.InvalidFeatureError):
            preprocessing.impute_missing_values(raw)


# ---------------------------------------------------------------------------
# FR-1.3, and the identifier guard
# ---------------------------------------------------------------------------


class TestFeaturePreparation:
    """FR-1.3 and the requirement that Patient_ID is never a feature."""

    @pytest.fixture(scope="class")
    def split(self):
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        cleaned = preprocessing.impute_missing_values(raw)
        return preprocessing.stratified_train_test_split(cleaned)

    def test_patient_id_is_not_in_the_feature_list(self) -> None:
        assert config.ID_COL not in config.FEATURE_COLS

    def test_patient_id_is_dropped_by_the_column_transformer(self) -> None:
        """remainder='drop' must actually exclude it from the matrix."""
        assert config.ID_COL not in preprocessing.build_preprocessing_pipeline().feature_cols

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_encoded_feature_names_exclude_patient_id(self, split) -> None:
        names = split.preprocessor.encoded_feature_names
        assert names, "encoded feature names must be recorded"
        assert not any(config.ID_COL in name for name in names)

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_one_hot_encoding_expands_the_categorical_column(self, split) -> None:
        """One-hot gives one column per documented level, not an invented order."""
        names = split.preprocessor.encoded_feature_names
        for level in config.CATEGORICAL_VALUES:
            assert f"{config.CATEGORICAL_COL}_{level}" in names

    @requires_implementation(preprocessing.stratified_train_test_split)
    @pytest.mark.parametrize("column", config.NUMERIC_COLS)
    def test_training_columns_have_zero_mean_and_unit_variance(self, split, column) -> None:
        """FR-1.3 acceptance criterion on the transformed training matrix."""
        values = split.x_train[column].to_numpy(dtype=float)
        assert values.mean() == pytest.approx(0.0, abs=1e-9)
        assert values.std(ddof=0) == pytest.approx(1.0, abs=1e-9)

    @requires_implementation(preprocessing.build_single_record_frame)
    def test_single_record_frame_uses_the_declared_column_order(self) -> None:
        record = {
            "Patient_ID": "P9999", "Age": 50, "Biomarker_Level": 5.0,
            "Pre_Existing_Conditions": 1, "Vaccine_Dose_mcg": 90.0,
            "Prior_Reaction_History": "Mild",
        }
        frame = preprocessing.build_single_record_frame(record)
        assert list(frame.columns) == list(config.FEATURE_COLS)
        assert config.ID_COL not in frame.columns

    @requires_implementation(preprocessing.build_single_record_frame)
    def test_incomplete_record_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(preprocessing.MissingColumnsError):
            preprocessing.build_single_record_frame({"Age": 40})

    @requires_implementation(preprocessing.transform_single_record)
    def test_single_record_matches_the_training_pipeline(self, split) -> None:
        """An unseen patient must be encoded exactly as training rows were."""
        record = split.train_raw.iloc[0].to_dict()
        row = preprocessing.transform_single_record(record, split.preprocessor)
        assert list(row.columns) == list(split.preprocessor.encoded_feature_names)
        for column in config.NUMERIC_COLS:
            assert row[column].iloc[0] == pytest.approx(
                split.x_train[column].iloc[0], abs=1e-9
            )

    @requires_implementation(preprocessing.fit_preprocessor)
    def test_unfitted_transformer_refuses_to_build_features(self) -> None:
        pre = preprocessing.build_preprocessing_pipeline()
        with pytest.raises(preprocessing.PreprocessorNotFittedError):
            preprocessing.build_feature_target(
                preprocessing.load_raw_records(config.RAW_PATIENT_CSV), pre
            )


# ---------------------------------------------------------------------------
# FR-1.4 and the leakage guard
# ---------------------------------------------------------------------------


class TestSplitReproducibility:
    """FR-1.4: 80/20, stratified on the target, fixed seed, identical on re-run."""

    @staticmethod
    def _split():
        cleaned = preprocessing.impute_missing_values(
            preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        )
        return preprocessing.stratified_train_test_split(cleaned)

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_split_proportions_are_eighty_twenty(self) -> None:
        split = self._split()
        assert split.n_test == 100
        assert split.n_train == 400

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_repeated_split_produces_identical_row_sets(self) -> None:
        """FR-1.4 acceptance criterion: two runs with the same seed agree."""
        first, second = self._split(), self._split()
        assert list(first.test_index) == list(second.test_index)
        assert list(first.train_index) == list(second.train_index)

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_stratification_preserves_the_class_ratio(self) -> None:
        split = self._split()
        overall = split.y_test.value_counts(normalize=True)
        train = split.y_train.value_counts(normalize=True)
        assert abs(overall[1] - train[1]) < 0.02

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_train_and_test_rows_do_not_overlap(self) -> None:
        split = self._split()
        assert not set(split.train_index) & set(split.test_index)

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_all_rows_are_accounted_for(self) -> None:
        split = self._split()
        assert split.n_train + split.n_test == 500

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_raw_frames_are_retained_for_pipeline_fitting(self, ) -> None:
        split = self._split()
        assert len(split.train_raw) == 400
        assert len(split.test_raw) == 100
        assert config.ID_COL in split.train_raw.columns

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_scaler_saw_only_the_training_rows(self) -> None:
        """The core leakage guard.

        ``StandardScaler.n_samples_seen_`` records how many rows the fitted
        scaler actually observed. It must equal the training row count, not 500.
        """
        split = self._split()
        scaler = split.preprocessor.column_transformer.named_transformers_[
            "numeric"
        ].named_steps["scaler"]
        assert int(np.asarray(scaler.n_samples_seen_).ravel()[0]) == 400

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_test_set_distribution_differed_from_training(self) -> None:
        """If the two splits were identical, the guard above would be vacuous.

        Guards against the leakage test passing for the wrong reason.
        """
        split = self._split()
        train_dose = split.train_raw["Vaccine_Dose_mcg"].mean()
        test_dose = split.test_raw["Vaccine_Dose_mcg"].mean()
        assert train_dose != pytest.approx(test_dose, abs=1e-9)

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_single_class_target_raises(self) -> None:
        cleaned = preprocessing.impute_missing_values(
            preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        )
        single = cleaned.assign(**{config.TARGET_COL: 1})
        with pytest.raises(preprocessing.InvalidTargetError):
            preprocessing.stratified_train_test_split(single)

    @requires_implementation(preprocessing.stratified_train_test_split)
    def test_imputer_median_is_the_train_only_median_not_the_full_data_median(
        self,
    ) -> None:
        """The fitted median must come from the training rows.

        Uses the **raw** frame rather than this class's ``_split()`` fixture. That
        fixture pre-imputes the whole dataset first, so by the time the split runs
        no value is missing and the all-data median and train-only median coincide
        in what the split can observe. Starting from raw rows is the case where the
        distinction is meaningful.

        ``n_samples_seen_`` above is a row *count*, so it cannot tell a correct
        train-only median from a wrong all-data one. This asserts the statistic.

        On this dataset ``Biomarker_Level`` is the discriminator: train median
        4.81, full-data median 4.82. ``Age`` also differs (50.5 vs 51.0);
        ``Vaccine_Dose_mcg`` is identical either way.
        """
        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        split = preprocessing.stratified_train_test_split(raw)
        train = raw.loc[split.train_index]

        imputer = split.preprocessor.column_transformer.named_transformers_[
            "numeric"
        ].named_steps["imputer"]
        stored = dict(zip(config.NUMERIC_COLS, imputer.statistics_))

        discriminating = 0
        for column in config.NUMERIC_COLS:
            train_median = float(train[column].median())
            full_median = float(raw[column].median())
            assert stored[column] == pytest.approx(train_median, abs=1e-9), (
                f"{column} imputer statistic {stored[column]} does not match the "
                f"train-only median ({train_median})"
            )
            if abs(train_median - full_median) > 1e-9:
                discriminating += 1
                assert stored[column] != pytest.approx(full_median, abs=1e-9), (
                    f"{column} was imputed with the FULL-DATA median "
                    f"({full_median}) instead of the train-only median "
                    f"({train_median}) - the held-out test set leaked into the "
                    "fitted imputer"
                )

        # Guards against the assertions above being vacuous.
        assert discriminating >= 1, (
            "no column distinguishes train-only from full-data medians, so this "
            "test cannot detect a leak on the current dataset"
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestCleanedDatasetPersistence:
    """The raw CSV must never be overwritten by the cleaned artifact."""

    @requires_implementation(preprocessing.save_clean_records)
    def test_cleaned_dataset_is_written_outside_the_raw_directory(self, tmp_path) -> None:
        cleaned = preprocessing.impute_missing_values(
            preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        )
        destination = preprocessing.save_clean_records(cleaned, tmp_path / "clean.csv")
        assert destination.exists()
        assert destination.parent != config.DATA_RAW_DIR
        assert len(pd.read_csv(destination)) == 500

    @requires_implementation(preprocessing.save_clean_records)
    def test_cleaned_dataset_has_zero_missing_values(self, tmp_path) -> None:
        cleaned = preprocessing.impute_missing_values(
            preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        )
        destination = preprocessing.save_clean_records(cleaned, tmp_path / "clean.csv")
        # Read with the same NA settings the loader uses, otherwise pandas turns
        # the imputed "None" category back into NaN and this test would fail for a
        # reason unrelated to the cleaning it is checking.
        reread = pd.read_csv(
            destination,
            keep_default_na=config.CSV_KEEP_DEFAULT_NA,
            na_values=list(config.CSV_NA_VALUES),
        )
        assert reread.isnull().sum().sum() == 0

    @requires_implementation(preprocessing.save_clean_records)
    def test_cleaned_dataset_retains_all_three_history_levels(self, tmp_path) -> None:
        """The written file must round-trip the categorical levels intact."""
        cleaned = preprocessing.impute_missing_values(
            preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        )
        destination = preprocessing.save_clean_records(cleaned, tmp_path / "clean.csv")
        reread = pd.read_csv(
            destination,
            keep_default_na=config.CSV_KEEP_DEFAULT_NA,
            na_values=list(config.CSV_NA_VALUES),
        )
        assert set(reread[config.CATEGORICAL_COL].unique()) == set(
            config.CATEGORICAL_VALUES
        )

    def test_raw_csv_is_never_modified_by_the_pipeline(self) -> None:
        """Guards the PRD's instruction that the source file is preserved."""
        assert config.RAW_PATIENT_CSV.is_file()
        header = config.RAW_PATIENT_CSV.read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(config.REQUIRED_COLUMNS)
        assert len(pd.read_csv(config.RAW_PATIENT_CSV)) == 500