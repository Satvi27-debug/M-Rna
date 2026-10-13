"""Staged command-line runner for the complete data + machine-learning stage.

Invoked as ``python -m src.ml.cli``, and re-exported from ``src.ml`` as
``run_training_stage``. It lives in its own module rather than in
``__main__.py`` because ``src/ml/__init__.py`` imports ``train``, and a
``__main__.py`` that imported ``train`` too would trigger a ``RuntimeWarning``
about the module already being in ``sys.modules``.

The stage it runs, with no dependency on the dashboard:

    1. Load the supplied dataset
    2. Inspect it and print the profile
    3. Validate its structure
    4. Clean it (impute the two documented gaps, keep every row)
    5. Split 80/20 stratified on Reaction_Score with a fixed seed
    6. Fit the preprocessing on the training rows only
    7. Train Logistic Regression and Random Forest on identical data
    8. Evaluate both on the held-out test set
    9. Generate the ROC comparison figure
   10. Generate one confusion matrix per model
   11. Save both model pipelines with joblib
   12. Save the evaluation results and a concise summary

Safety framing
--------------
The printed summary reports model metrics and nothing else. No output describes
a model as safe, unsafe, approved or clinically recommended. Every figure and
results file carries the student-project disclaimer, because this is an
engineering demonstration and not a clinical system.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from src import config
from src.data import preprocessing
from src.ml import evaluate as ml_evaluate
from src.ml import predict as ml_predict
from src.ml import train as ml_train


RULE = "=" * 72
THIN = "-" * 72


def _banner(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def run(verbose: bool = True) -> dict[str, Any]:
    """Execute the full stage and return the machine-readable summary.

    Parameters
    ----------
    verbose:
        Print the staged progress and the console summary.

    Returns
    -------
    dict
        The same content written to ``outputs/logs/ml_results.json``, plus the
        paths of everything written.
    """
    config.ensure_directories()

    # -- 1-2. load and inspect ------------------------------------------
    if verbose:
        _banner("STEP 1-2  LOAD AND INSPECT")
    raw = preprocessing.load_raw_records()
    report = preprocessing.inspect_dataset(raw)
    if verbose:
        print(preprocessing.format_inspection_report(report))

    # -- 3. validate ------------------------------------------------------
    preprocessing.validate_dataset(raw)
    if verbose:
        print("Validation: passed "
              "(required columns present, IDs unique, numerics numeric, "
              "target binary)")

    # -- 4. clean ---------------------------------------------------------
    medians, modes = preprocessing.fit_imputation_strategy(raw)
    cleaned = preprocessing.impute_missing_values(raw, medians, modes)
    report = preprocessing.inspect_dataset(cleaned)
    if verbose:
        print(f"\nCleaning: {report.total_missing_before} missing cells -> "
              f"{report.total_missing_after}. Rows retained: "
              f"{len(cleaned)}/{len(raw)}.")
        print(f"  Biomarker_Level median (imputed)     : {medians.get('Biomarker_Level')}")
        print(f"  Prior_Reaction_History mode (imputed): {modes.get('Prior_Reaction_History')}")

    # -- 5-6. split, then fit preprocessing on train only ------------------
    split = preprocessing.stratified_train_test_split(cleaned)
    if verbose:
        _banner("STEP 5-6  SPLIT AND FIT PREPROCESSING (TRAIN ONLY)")
        print(f"Train rows: {split.n_train}   Test rows: {split.n_test}   "
              f"(test_size={config.TEST_SIZE}, seed={config.RANDOM_SEED}, "
              f"stratified on {config.TARGET_COL})")
        print(f"Train balance: {split.y_train.value_counts().sort_index().to_dict()}")
        print(f"Test balance : {split.y_test.value_counts().sort_index().to_dict()}")
        print(f"Encoded features ({split.preprocessor.n_encoded_features}):")
        for name in split.preprocessor.encoded_feature_names:
            print(f"  - {name}")

    # -- 7. train ----------------------------------------------------------
    if verbose:
        _banner("STEP 7  TRAIN")
    trained = ml_train.train_models(split)
    if verbose:
        for name, pipeline in trained.pipelines.items():
            estimator = pipeline.named_steps["classifier"]
            print(f"  {config.MODEL_DISPLAY_NAMES[name]:<22} "
                  f"{type(estimator).__name__:<24} "
                  f"params={estimator.get_params().get('max_iter', estimator.get_params().get('n_estimators'))}")

    # -- 8. evaluate -------------------------------------------------------
    if verbose:
        _banner("STEP 8  EVALUATE ON HELD-OUT TEST SET")
    evaluations = ml_evaluate.evaluate_all_models(
        trained, split.x_test, split.y_test, split.test_raw
    )
    selection = ml_evaluate.select_best_model(evaluations)

    header = f"{'model':<24}{'accuracy':>10}{'precision':>11}{'recall':>10}{'f1':>10}{'roc_auc':>10}"
    print(header)
    print(THIN)
    for evaluation in evaluations.values():
        print(f"{evaluation.display_name:<24}{evaluation.accuracy:>10.4f}"
              f"{evaluation.precision:>11.4f}{evaluation.recall:>10.4f}"
              f"{evaluation.f1:>10.4f}{evaluation.roc_auc:>10.4f}")
    print()
    print(f"Selection (FR-1.7): {selection.best_display_name} on test "
          f"{selection.metric.upper()} = {selection.best_f1:.4f}")
    print(selection.rationale(evaluations))

    for name, evaluation in evaluations.items():
        matrix = evaluation.confusion_matrix
        print(f"\nConfusion matrix - {evaluation.display_name} "
              f"(rows = actual, cols = predicted)")
        print(f"               pred 0   pred 1")
        print(f"  actual 0   {matrix[0][0]:>6}   {matrix[0][1]:>6}")
        print(f"  actual 1   {matrix[1][0]:>6}   {matrix[1][1]:>6}")

    # -- 9-12. figures, artifacts, results ---------------------------------
    if verbose:
        _banner("STEP 9-12  FIGURES, ARTIFACTS, RESULTS")
    written = ml_evaluate.save_evaluation_artifacts(evaluations, selection, split)
    model_paths = ml_train.save_model_artifacts(trained, selection)
    metadata_path = ml_train.save_training_metadata(trained, split)
    cleaned_path = preprocessing.save_clean_records(cleaned)

    for label, path in {**written, **model_paths,
                        "training_metadata": metadata_path,
                        "cleaned_dataset": cleaned_path}.items():
        target = path if isinstance(path, str) else path
        print(f"  {label:<24} {target}")

    if verbose:
        _banner("SUMMARY")
    print(f"Target          : {config.TARGET_COL} used directly as the binary target")
    print(f"                 (PRD Section 5.2: 0 = Low reaction risk, 1 = High)")
    print(f"Selected model  : {selection.best_display_name}")
    print(f"Artifacts       : {config.MODELS_DIR}")
    print(f"Results         : {config.ML_RESULTS_JSON_PATH}")
    print(f"Disclaimer      : {config.REQUIRED_DISCLAIMER}")

    summary: dict[str, Any] = {
        "results": json.loads(config.ML_RESULTS_JSON_PATH.read_text(encoding="utf-8")),
        "artifacts": {
            **{k: str(v) for k, v in written.items()},
            **{k: str(v) for k, v in model_paths.items()},
            "training_metadata": str(metadata_path),
            "cleaned_dataset": str(cleaned_path),
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the stage."""
    parser = argparse.ArgumentParser(
        prog="python -m src.ml.cli",
        description=(
            "Train and evaluate the adverse-reaction risk classifiers. "
            "Outputs model artifacts, metrics and figures."
        ),
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress staged progress output"
    )
    args = parser.parse_args(argv)
    run(verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())