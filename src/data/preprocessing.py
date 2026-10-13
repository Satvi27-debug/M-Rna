"""Loading, validation, inspection, cleaning and feature preparation.

The dataset is `data/raw/patient_records.csv`, supplied with the PRD and never
written to. This module is the only place allowed to read from `data/`.

Design decisions and the evidence behind them
---------------------------------------------
**1. The literal string "None" is a category, not a missing value.**
This is the single most consequential finding in the data stage, so it is
recorded in full. Reading the CSV with pandas' defaults yields:

    Prior_Reaction_History: {NaN: 308, 'Mild': 150, 'Moderate': 42}

which would suggest 61.6% of the column is missing. Reading the raw text of the
same file shows what is physically there:

    'None'     291 rows  (58.2%)
    'Mild'     150 rows  (30.0%)
    'Moderate'  42 rows  ( 8.4%)
    ''          17 rows  ( 3.4%)   <- genuinely absent

Pandas' default NA list contains the *string* ``"None"``, so 291 real
observations are silently converted to NaN. The PRD documents this column as
"None / Mild / Moderate" (Section 5.2) and states missing values are present in
"about 3.5% of rows" - which is 17 rows, not 308. So the PRD's own
specification confirms that "None" is a level, not an absence.

Treating it as missing would not be harmless. The class-conditional target rates
are cleanly ordinal:

    None -> 0.447    Mild -> 0.520    Moderate -> 0.667

Collapsing "None" into NaN and then applying FR-1.2's mode imputation would
overwrite all 291 rows with the majority category and destroy the strongest
categorical signal in the dataset. The loader therefore reads with
``keep_default_na=False`` and treats only the empty string as absent.

**2. Median for ``Biomarker_Level``.** It is a right-skewed concentration
(0.64 to 15.76, mean 5.28, median 4.82), so the median represents a typical
patient better than the mean and is not pulled up by the small number of
high-biomarker patients. 20 rows (4.0%) are absent, matching the PRD's "~4%".
FR-1.2.

**3. Mode for ``Prior_Reaction_History``.** Categorical with three levels;
imputing the most frequent level keeps the imputation inside the observed label
set instead of inventing a fourth one. The mode is "None". FR-1.2.

**4. No rows are dropped.** Both missing-value columns are clinically meaningful,
so removing the affected records would bias the sample toward healthier
patients. All 500 rows are retained.

**5. Preprocessing is fitted on the training split only.** Imputation statistics
and the scaler/encoder are all learned from the training rows and then applied
unchanged to the held-out test rows. Fitting anything before the split would leak
test-set information into the model, so the split happens before preprocessing.

**6. One-hot encoding rather than the PRD template's ``LabelEncoder``.**
PRD Section 5.4 sketches ``LabelEncoder``, which assigns integers by alphabetical
order and therefore yields ``Mild -> 0, Moderate -> 1, None -> 2``. That ordering
is semantically inverted, since the *absence* of prior reactions becomes the
highest numeric value and a linear model would learn a backwards relationship.
PRD Section 2 states requirements are testable statements "independent of exactly
how you implement it", and the requirement here is only that the column be
encoded. ``OneHotEncoder`` satisfies that without imposing a false order, so it is
used and the deviation from the sketch is documented.

Status: fully implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from src import config


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PatientDataValidationError(ValueError):
    """Base class for every data-layer validation failure."""


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the supplied dataset cannot be read."""


class MissingColumnsError(PatientDataValidationError):
    """Raised when required dataset columns are absent.

    The message lists the missing columns by name so the FR-1.1 acceptance test
    can assert on it directly.
    """

    def __init__(self, missing: list[str], required: tuple[str, ...]) -> None:
        self.missing = list(missing)
        self.required = list(required)
        super().__init__(
            "Missing required column(s): "
            f"{', '.join(self.missing)}. "
            f"Required columns are: {', '.join(self.required)}."
        )


class DuplicatePatientIdError(PatientDataValidationError):
    """Raised when ``Patient_ID`` is not unique, which would break row identity."""

    def __init__(self, duplicates: list[str]) -> None:
        self.duplicates = list(duplicates)
        super().__init__(
            f"Patient_ID must be unique. Duplicated value(s): "
            f"{', '.join(map(str, self.duplicates[:10]))}"
            + (" ..." if len(self.duplicates) > 10 else "")
        )


class InvalidTargetError(PatientDataValidationError):
    """Raised when ``Reaction_Score`` does not form a usable binary target."""


class InvalidFeatureError(PatientDataValidationError):
    """Raised when a feature column is not numeric or holds impossible values."""


class EmptyDatasetError(PatientDataValidationError):
    """Raised when the dataset has no rows."""


class PreprocessorNotFittedError(RuntimeError):
    """Raised when a preprocessor is used before ``fit`` has been called."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnProfile:
    """Per-column facts gathered by :func:`inspect_dataset`."""

    name: str
    dtype: str
    n_missing: int
    missing_pct: float
    n_unique: int
    is_numeric: bool
    numeric_min: float | None = None
    numeric_max: float | None = None
    numeric_mean: float | None = None
    categorical_values: list[Any] = field(default_factory=list)
    target_mean: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "n_missing": self.n_missing,
            "missing_pct": round(self.missing_pct, 4),
            "n_unique": self.n_unique,
            "is_numeric": self.is_numeric,
            "min": self.numeric_min,
            "max": self.numeric_max,
            "mean": self.numeric_mean,
            "values": self.categorical_values,
            "target_mean": self.target_mean,
        }


@dataclass(frozen=True)
class DataQualityReport:
    """Structured profile of the dataset, captured before cleaning.

    Kept as an object rather than printed text so the pipeline can log it and
    tests can assert on it without capturing stdout.
    """

    rows_in: int
    cols_in: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_before: dict[str, int]
    missing_after: dict[str, int]
    duplicate_rows: int
    duplicate_patient_ids: list[str]
    class_balance: dict[int, int]
    profiles: list[ColumnProfile] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows_in, self.cols_in)

    @property
    def total_missing_before(self) -> int:
        return sum(self.missing_before.values())

    @property
    def total_missing_after(self) -> int:
        """Must be 0 after cleaning - the FR-1.2 acceptance criterion."""
        return sum(self.missing_after.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows_in,
            "cols": self.cols_in,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "missing_before": self.missing_before,
            "missing_after": self.missing_after,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_patient_ids": self.duplicate_patient_ids,
            "class_balance": self.class_balance,
            "profiles": [p.as_dict() for p in self.profiles],
        }


@dataclass(frozen=True)
class Preprocessor:
    """Fitted preprocessing state, serialisable alongside the models.

    Wraps the ``ColumnTransformer`` that performs scaling and encoding, and
    records the imputation statistics it was fitted on so a single-row
    prediction can reproduce the training-time transformation exactly.
    """

    column_transformer: Any
    feature_cols: tuple[str, ...]
    numeric_cols: tuple[str, ...]
    categorical_cols: tuple[str, ...]
    categorical_values: dict[str, list[Any]] = field(default_factory=dict)
    median_values: dict[str, float] = field(default_factory=dict)
    mode_values: dict[str, str] = field(default_factory=dict)
    encoded_feature_names: list[str] = field(default_factory=list)

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer has actually been fitted.

        Checks scikit-learn's own fitted attributes rather than merely testing that
        the object exists. An unfitted ``ColumnTransformer`` is a real object, so
        ``column_transformer is not None`` would report ``True`` before ``fit`` and
        let a later ``transform`` fail with a confusing internal error instead of
        the intended named one.
        """
        if self.column_transformer is None:
            return False
        try:
            from sklearn.utils.validation import check_is_fitted

            check_is_fitted(self.column_transformer, attributes=["transformers_"])
            return True
        except (NotFittedError, TypeError, ImportError, AttributeError):
            return False

    @property
    def n_encoded_features(self) -> int:
        """Width of the matrix the models actually receive."""
        return len(self.encoded_feature_names)


@dataclass(frozen=True)
class SplitData:
    """Immutable 80/20 stratified split plus the fitted preprocessor.

    ``x_train`` and ``x_test`` are the *transformed* matrices the models consume,
    in ``encoded_feature_names`` order. The raw frames and the fitted
    preprocessor are retained so an unseen patient can be transformed identically.
    """

    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: Preprocessor
    train_index: pd.Index
    test_index: pd.Index
    train_raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_raw: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def n_train(self) -> int:
        return len(self.x_train)

    @property
    def n_test(self) -> int:
        return len(self.x_test)


# ---------------------------------------------------------------------------
# Step 1 - load and validate (FR-1.1)
# ---------------------------------------------------------------------------


def load_raw_records(path: Any = None) -> pd.DataFrame:
    """Read the patient CSV and verify it matches the documented schema.

    Parameters
    ----------
    path:
        CSV location. Defaults to ``config.RAW_PATIENT_CSV``.

    Returns
    -------
    pandas.DataFrame
        The raw dataset, unmodified.

    Raises
    ------
    DatasetNotFoundError
        If ``path`` does not exist.
    MissingColumnsError
        If any of the 7 required columns is absent (FR-1.1).
    """
    from pathlib import Path

    path = Path(path) if path else config.RAW_PATIENT_CSV
    if not path.is_file():
        raise DatasetNotFoundError(
            f"Patient dataset not found at {path}. "
            "Place the supplied patient_records.csv there."
        )

    # keep_default_na=False preserves the literal string "None" as a real
    # category; only a genuinely empty field becomes NaN. See module docstring.
    df = pd.read_csv(
        path,
        keep_default_na=config.CSV_KEEP_DEFAULT_NA,
        na_values=list(config.CSV_NA_VALUES),
    )

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MissingColumnsError(missing, config.REQUIRED_COLUMNS)
    return df


def validate_dataset(df: pd.DataFrame) -> None:
    """Check the dataset is structurally usable before any modelling work.

    Validates that the dataset is non-empty, that the 7 required columns are
    present (FR-1.1), that ``Patient_ID`` is unique, that the numeric feature
    columns are actually numeric, and that ``Reaction_Score`` forms a usable
    binary target.

    Raises
    ------
    EmptyDatasetError, MissingColumnsError, DuplicatePatientIdError,
    InvalidFeatureError, InvalidTargetError
    """
    if df is None or len(df) == 0:
        raise EmptyDatasetError("Dataset is empty; nothing to train on.")

    missing = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MissingColumnsError(missing, config.REQUIRED_COLUMNS)

    duplicated = df[config.ID_COL][df[config.ID_COL].duplicated()].unique().tolist()
    if duplicated:
        raise DuplicatePatientIdError(duplicated)

    for column in config.NUMERIC_COLS + (config.COUNT_COL,):
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise InvalidFeatureError(
                f"Column '{column}' must be numeric but has dtype {df[column].dtype}."
            )
        if df[column].dropna().lt(0).any():
            raise InvalidFeatureError(
                f"Column '{column}' contains negative values, which are not "
                "physically meaningful for this dataset."
            )

    observed = set(df[config.CATEGORICAL_COL].dropna().unique())
    unexpected = observed - set(config.CATEGORICAL_VALUES)
    if unexpected:
        raise InvalidFeatureError(
            f"Column '{config.CATEGORICAL_COL}' contains undocumented "
            f"value(s): {sorted(unexpected)}. "
            f"Expected only {list(config.CATEGORICAL_VALUES)}."
        )


def inspect_dataset(df: pd.DataFrame) -> DataQualityReport:
    """Profile the dataset: shape, dtypes, missingness, duplicates, ranges, target.

    Reusable and side-effect free. The prompt requires the implementation to
    work from the actual supplied CSV rather than from assumed characteristics,
    so this is the function that establishes the facts the rest of the module
    relies on.

    Parameters
    ----------
    df:
        Raw or cleaned records.

    Returns
    -------
    DataQualityReport
        Includes per-column profiles and the class balance of the target.
    """
    if df is None or len(df) == 0:
        raise EmptyDatasetError("Cannot inspect an empty dataset.")

    profiles: list[ColumnProfile] = []
    for column in df.columns:
        series = df[column]
        numeric = pd.api.types.is_numeric_dtype(series)
        target_mean = (
            float(series.mean())
            if column == config.TARGET_COL and numeric
            else None
        )
        profiles.append(
            ColumnProfile(
                name=column,
                dtype=str(series.dtype),
                n_missing=int(series.isna().sum()),
                missing_pct=round(100.0 * series.isna().mean(), 4),
                n_unique=int(series.nunique(dropna=True)),
                is_numeric=bool(numeric),
                numeric_min=float(series.min()) if numeric else None,
                numeric_max=float(series.max()) if numeric else None,
                numeric_mean=round(float(series.mean()), 6) if numeric else None,
                categorical_values=(
                    [] if numeric else sorted(series.dropna().unique().tolist())
                ),
                target_mean=target_mean,
            )
        )

    balance: dict[int, int] = {}
    if config.TARGET_COL in df.columns:
        balance = {
            int(k): int(v)
            for k, v in df[config.TARGET_COL].value_counts().sort_index().items()
        }

    return DataQualityReport(
        rows_in=int(len(df)),
        cols_in=int(len(df.columns)),
        columns=[str(c) for c in df.columns],
        dtypes={str(c): str(df[c].dtype) for c in df.columns},
        missing_before={str(c): int(df[c].isna().sum()) for c in df.columns},
        missing_after={str(c): int(df[c].isna().sum()) for c in df.columns},
        duplicate_rows=int(df.duplicated().sum()),
        duplicate_patient_ids=df[config.ID_COL][
            df[config.ID_COL].duplicated()
        ].unique().tolist(),
        class_balance=balance,
        profiles=profiles,
    )


def format_inspection_report(report: DataQualityReport) -> str:
    """Render :func:`inspect_dataset` output as the readable console summary."""
    lines: list[str] = []
    add = lines.append

    add("=" * 68)
    add("DATASET INSPECTION")
    add("=" * 68)
    add(f"Shape              : {report.rows_in} rows x {report.cols_in} columns")
    add(f"Duplicate rows     : {report.duplicate_rows}")
    add(f"Duplicate IDs      : {len(report.duplicate_patient_ids)}")
    add("")
    add("COLUMN PROFILE")
    add(f"{'column':<26}{'dtype':<10}{'missing':>9}{'unique':>9}{'min':>11}{'max':>11}")
    for profile in report.profiles:
        add(
            f"{profile.name:<26}{profile.dtype:<10}{profile.n_missing:>9}"
            f"{profile.n_unique:>9}"
            f"{('' if profile.numeric_min is None else f'{profile.numeric_min:g}'):>11}"
            f"{('' if profile.numeric_max is None else f'{profile.numeric_max:g}'):>11}"
        )
    add("")
    add("CATEGORICAL VALUES")
    for profile in report.profiles:
        if not profile.is_numeric:
            add(f"{profile.name:<26}{profile.categorical_values}")
    add("")
    add("TARGET DISTRIBUTION")
    total = sum(report.class_balance.values()) or 1
    for label, count in sorted(report.class_balance.items()):
        add(f"  Reaction_Score={label}  {count:>4}  ({100 * count / total:.1f}%)")
    add("")
    add(f"MISSING TOTALS       : {report.total_missing_before} cells")
    add("=" * 68)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 2 - imputation (FR-1.2)
# ---------------------------------------------------------------------------


def fit_imputation_strategy(df: pd.DataFrame) -> tuple[dict[str, float], dict[str, str]]:
    """Derive imputation statistics from a frame, without modifying it.

    Median for the numeric biomarker, mode for the categorical column, per FR-1.2.

    Parameters
    ----------
    df:
        The frame to learn from - **the training split**, so the statistics do not
        carry test-set information into training.

    Returns
    -------
    tuple
        ``(median_values, mode_values)``.
    """
    medians = {
        column: float(df[column].median())
        for column in config.NUMERIC_COLS
        if column in df and df[column].notna().any()
    }
    modes: dict[str, str] = {}
    if config.CATEGORICAL_COL in df:
        observed = df[config.CATEGORICAL_COL].dropna()
        if not observed.empty:
            modes[config.CATEGORICAL_COL] = str(observed.mode()[0])
    return medians, modes


def impute_missing_values(
    df: pd.DataFrame,
    medians: dict[str, float] | None = None,
    modes: dict[str, str] | None = None,
    report: DataQualityReport | None = None,
) -> pd.DataFrame:
    """Fill the two documented gaps without dropping any row (FR-1.2).

    ``Biomarker_Level`` receives the median and ``Prior_Reaction_History`` the
    mode. The remaining columns are complete in the supplied dataset; any gap
    there is a hard error rather than a silent fill, because guessing a target
    value would corrupt the labels.

    Parameters
    ----------
    df:
        Records to clean. Not modified - a new frame is returned.
    medians, modes:
        Statistics from :func:`fit_imputation_strategy`. Computed from ``df``
        itself when omitted, which is only appropriate for the standalone
        inspection path, never for the modelling path.
    report:
        Optional in-place :class:`DataQualityReport` to populate.

    Returns
    -------
    pandas.DataFrame
        A new frame with no missing values and the same row count.
    """
    cleaned = df.copy()

    for column in config.NUMERIC_COLS:
        if column not in cleaned.columns or not cleaned[column].isna().any():
            continue
        value = (medians or {}).get(column)
        if value is None:
            value = float(cleaned[column].median())
        cleaned[column] = cleaned[column].fillna(value)

    if config.CATEGORICAL_COL in cleaned.columns:
        if cleaned[config.CATEGORICAL_COL].isna().any():
            value = (modes or {}).get(config.CATEGORICAL_COL)
            if value is None:
                observed = cleaned[config.CATEGORICAL_COL].dropna()
                value = str(observed.mode()[0]) if not observed.empty else "None"
            cleaned[config.CATEGORICAL_COL] = cleaned[config.CATEGORICAL_COL].fillna(value)

    # Any residual gap in a column the PRD documents as complete is an error.
    residual = {c: int(n) for c, n in cleaned.isna().sum().items() if n}
    if residual:
        raise InvalidFeatureError(
            f"Unexpected missing values remain after imputation in: {residual}. "
            "These columns are documented as complete, so imputing them would "
            "conceal a data defect."
        )

    if report is not None:
        object.__setattr__(
            report, "missing_after", {str(c): int(cleaned[c].isna().sum()) for c in cleaned.columns}
        )
    return cleaned


# ---------------------------------------------------------------------------
# Step 3 - feature / target separation
# ---------------------------------------------------------------------------


def build_preprocessing_pipeline(
    numeric_medians: dict[str, float] | None = None,
    categorical_modes: dict[str, str] | None = None,
) -> Preprocessor:
    """Construct the unfitted ``ColumnTransformer`` used by both models.

    One pipeline object serves both classifiers, so they are guaranteed to see
    byte-identical input matrices - which is what FR-1.5's "on identical training
    data" requires.

    Numerical branch
        ``SimpleImputer(median)`` then ``StandardScaler``. Imputing inside the
        pipeline means the statistics are learned from the training rows only.

    Categorical branch
        ``SimpleImputer(most_frequent)`` then ``OneHotEncoder``. See the module
        docstring for why one-hot replaces the PRD template's ``LabelEncoder``.

    The ``Patient_ID`` column is deliberately not routed through either branch;
    it is an identifier, never a predictive feature.

    Parameters
    ----------
    numeric_medians, categorical_modes:
        Recorded on the returned :class:`Preprocessor` for single-row inference.
        The pipeline re-derives them from the training rows during ``fit``, so
        these are provenance rather than input.

    Returns
    -------
    Preprocessor
        With ``column_transformer`` unfitted. Call
        :func:`fit_preprocessor` before use.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_branch = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    categories=[list(config.CATEGORICAL_VALUES)],
                    drop=None,
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_branch, list(config.NUMERIC_COLS)),
            (
                "categorical",
                categorical_branch,
                [config.CATEGORICAL_COL],
            ),
            ("count", "passthrough", [config.COUNT_COL]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return Preprocessor(
        column_transformer=transformer,
        feature_cols=config.FEATURE_COLS,
        numeric_cols=config.NUMERIC_COLS,
        categorical_cols=(config.CATEGORICAL_COL,),
        median_values=dict(numeric_medians or {}),
        mode_values=dict(categorical_modes or {}),
    )


def fit_preprocessor(preprocessor: Preprocessor, df: pd.DataFrame) -> Preprocessor:
    """Fit the transformer on ``df`` and return a new, fitted ``Preprocessor``.

    Parameters
    ----------
    preprocessor:
        The unfitted pipeline from :func:`build_preprocessing_pipeline`.
    df:
        The **training** rows only. Fitting on the full dataset would leak the
        test distribution into the model.

    Returns
    -------
    Preprocessor
        Carries the fitted transformer and the encoded feature names.
    """
    # Deliberately not ``is_fitted``: that property reports whether the
    # transformer has been fitted, which is exactly the state this function
    # exists to change. What must exist is the transformer object itself.
    if preprocessor.column_transformer is None:
        raise PreprocessorNotFittedError(
            "No column transformer to fit. Build one with "
            "build_preprocessing_pipeline() first."
        )

    columns = list(config.FEATURE_COLS)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise MissingColumnsError(missing, tuple(columns))

    transformer = preprocessor.column_transformer
    transformer.fit(df[columns])

    names = [str(n) for n in transformer.get_feature_names_out()]
    return Preprocessor(
        column_transformer=transformer,
        feature_cols=tuple(columns),
        numeric_cols=preprocessor.numeric_cols,
        categorical_cols=preprocessor.categorical_cols,
        categorical_values={
            config.CATEGORICAL_COL: list(config.CATEGORICAL_VALUES)
        },
        median_values=dict(preprocessor.median_values),
        mode_values=dict(preprocessor.mode_values),
        encoded_feature_names=names,
    )


def build_feature_target(
    df: pd.DataFrame, preprocessor: Preprocessor
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the fitted transformer and return ``(X, y)``.

    ``X`` has one column per encoded feature, named by
    ``config.FEATURE_COLS`` expanded through the one-hot encoding.
    ``Patient_ID`` is never included.

    Raises
    ------
    PreprocessorNotFittedError
        If the transformer has not been fitted.
    """
    if not preprocessor.is_fitted:
        raise PreprocessorNotFittedError(
            "Preprocessor must be fitted before features are built. "
            "Call fit_preprocessor() on the training split first."
        )
    columns = list(config.FEATURE_COLS)
    transformed = preprocessor.column_transformer.transform(df[columns])
    names = preprocessor.encoded_feature_names or [
        f"f{i}" for i in range(transformed.shape[1])
    ]
    x = pd.DataFrame(
        np.asarray(transformed, dtype=float),
        columns=names,
        index=df.index,
    )
    y = df[config.TARGET_COL].astype(int)
    return x, y


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper: fit on ``df`` then transform ``df``.

    Intended for the standalone inspection path only. The modelling path calls
    :func:`stratified_train_test_split`, which fits on training rows exclusively.
    """
    medians, modes = fit_imputation_strategy(df)
    fitted = fit_preprocessor(build_preprocessing_pipeline(medians, modes), df)
    return build_feature_target(df, fitted)


# ---------------------------------------------------------------------------
# Step 4 - stratified split (FR-1.4)
# ---------------------------------------------------------------------------


def stratified_train_test_split(
    df: pd.DataFrame, preprocessor: Preprocessor | None = None
) -> SplitData:
    """Split 80/20 stratified on ``Reaction_Score``, then fit preprocessing on train.

    Order matters and follows the required workflow:

        validate -> feature/target separation -> split -> fit on train -> transform

    The split happens *before* the transformer is fitted, so the scaler and
    imputers see only training rows. Two calls with the same seed produce
    identical row sets (FR-1.4).

    Parameters
    ----------
    df:
        Cleaned records.
    preprocessor:
        Optional unfitted preprocessor. One is constructed when omitted.

    Returns
    -------
    SplitData
        Transformed ``x_train``/``x_test`` plus the fitted preprocessor and the
        raw frames, so an unseen patient can be transformed identically.

    Raises
    ------
    InvalidTargetError
        If a class has too few members for a stratified split.
    """
    from sklearn.model_selection import train_test_split

    if df is None or len(df) == 0:
        raise EmptyDatasetError("Cannot split an empty dataset.")

    target = df[config.TARGET_COL].astype(int)
    classes, counts = np.unique(target, return_counts=True)
    if len(classes) < 2:
        raise InvalidTargetError(
            f"Target must contain at least two classes to stratify; found {classes.tolist()}."
        )
    smallest = int(counts.min())
    n_test = int(round(len(df) * config.TEST_SIZE))
    if smallest < 2 or smallest < n_test:
        raise InvalidTargetError(
            f"Class with only {smallest} member(s) cannot support a stratified "
            f"{int((1 - config.TEST_SIZE) * 100)}/{int(config.TEST_SIZE * 100)} split."
        )

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=target,
    )
    train_raw = df.loc[train_idx].copy()
    test_raw = df.loc[test_idx].copy()

    # Fit imputation statistics and the transformer on training rows ONLY.
    medians, modes = fit_imputation_strategy(train_raw)
    if preprocessor is None:
        preprocessor = build_preprocessing_pipeline(medians, modes)
    fitted = fit_preprocessor(preprocessor, train_raw)

    x_train, y_train = build_feature_target(train_raw, fitted)
    x_test, y_test = build_feature_target(test_raw, fitted)

    return SplitData(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=fitted,
        train_index=pd.Index(sorted(train_idx)),
        test_index=pd.Index(sorted(test_idx)),
        train_raw=train_raw,
        test_raw=test_raw,
    )


# ---------------------------------------------------------------------------
# Inference-time helpers (used by src/ml/predict.py)
# ---------------------------------------------------------------------------


def build_single_record_frame(record: dict[str, Any]) -> pd.DataFrame:
    """Normalise a patient record into a one-row raw feature frame.

    This is what the saved pipelines consume. Each artifact is a
    ``Pipeline([ColumnTransformer, classifier])``, so inference hands it the
    original feature columns and lets the pipeline apply its own fitted
    imputation, scaling and encoding. Passing a pre-encoded matrix instead would
    fail, because ``Prior_Reaction_History`` no longer exists once one-hot
    encoding has run.

    Absent values are left as ``None`` rather than filled here. The pipeline's
    ``SimpleImputer`` steps were fitted on the training rows, so letting them
    apply their own recorded statistics keeps inference consistent with training
    even when a caller omits a field.

    Parameters
    ----------
    record:
        Mapping with at least the keys in ``config.FEATURE_COLS``. Extra keys such
        as ``Patient_ID`` are ignored - the identifier is never a feature.

    Returns
    -------
    pandas.DataFrame
        One row, one column per entry in ``config.FEATURE_COLS``.

    Raises
    ------
    MissingColumnsError
        If a required feature key is absent. An omitted field is an incomplete
        record, and silently defaulting it would misreport a prediction.
    """
    missing = [c for c in config.FEATURE_COLS if c not in record]
    if missing:
        raise MissingColumnsError(missing, config.FEATURE_COLS)

    row: dict[str, Any] = {}
    for column in config.FEATURE_COLS:
        value = record[column]
        # Normalise NaN to None so SimpleImputer recognises it as absent.
        if isinstance(value, float) and np.isnan(value):
            value = None
        row[column] = value
    return pd.DataFrame([row], columns=list(config.FEATURE_COLS))


def transform_single_record(
    record: dict[str, Any], preprocessor: Preprocessor
) -> pd.DataFrame:
    """Build a single-row **encoded** feature matrix for an unseen patient.

    Applies exactly the transformation the training rows received, including
    imputation, so an inference vector cannot silently differ from a training one.
    Use this when working with the encoded matrix directly; the model pipelines
    in :mod:`src.ml.predict` use :func:`build_single_record_frame` instead,
    since they perform the transform internally.

    Parameters
    ----------
    record:
        Mapping with the keys in ``config.FEATURE_COLS``. Missing numeric values
        fall back to the recorded training median.
    preprocessor:
        Fitted preprocessor from the saved model artifact.

    Returns
    -------
    pandas.DataFrame
        One row, one column per encoded feature name.

    Raises
    ------
    MissingColumnsError, PreprocessorNotFittedError
    """
    if not preprocessor.is_fitted:
        raise PreprocessorNotFittedError(
            "Cannot transform a record with an unfitted preprocessor."
        )

    frame = build_single_record_frame(record)
    medians = preprocessor.median_values or {}
    modes = preprocessor.mode_values or {}
    for column in config.NUMERIC_COLS:
        if frame[column].isna().all() and column in medians:
            frame[column] = medians[column]
    if frame[config.CATEGORICAL_COL].isna().all() and config.CATEGORICAL_COL in modes:
        frame[config.CATEGORICAL_COL] = modes[config.CATEGORICAL_COL]

    transformed = preprocessor.column_transformer.transform(frame)
    names = preprocessor.encoded_feature_names or [
        f"f{i}" for i in range(transformed.shape[1])
    ]
    return pd.DataFrame(np.asarray(transformed, dtype=float), columns=names)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_clean_records(df: pd.DataFrame, path: Any = None) -> Any:
    """Write the cleaned dataset to ``data/processed/``.

    Satisfies the Week 1 Definition of Done: the artifact exists and has zero
    missing values. The raw CSV is never touched.

    Parameters
    ----------
    df:
        Cleaned records.
    path:
        Destination. Defaults to ``config.PROCESSED_PATIENT_CSV``.

    Returns
    -------
    pathlib.Path
        The written path.
    """
    from pathlib import Path

    destination = Path(path) if path else config.PROCESSED_PATIENT_CSV
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return destination