"""Patient data subpackage: loading, validation, imputation and preparation.

Boundary rule: this package is the only place allowed to read from ``data/``.
Everything downstream receives an already-cleaned ``pandas.DataFrame`` plus a
fitted preprocessor.

Requirements owned here
-----------------------
FR-1.1  Validate that all 7 required columns are present; raise a named error
        naming the offending column(s).
FR-1.2  Impute missing ``Biomarker_Level`` with the column median and missing
        ``Prior_Reaction_History`` with the column mode.
FR-1.3  Standardise ``Age``, ``Biomarker_Level`` and ``Vaccine_Dose_mcg`` to zero
        mean and unit variance.
FR-1.4  80/20 train/test split stratified on ``Reaction_Score`` with a fixed seed.
"""

from src.data.preprocessing import (
    MissingColumnsError,
    PreprocessorNotFittedError,
    PatientDataValidationError,
)

__all__ = [
    "MissingColumnsError",
    "PatientDataValidationError",
    "PreprocessorNotFittedError",
]