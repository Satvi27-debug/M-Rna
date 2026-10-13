"""Classical machine learning subpackage: training, evaluation and inference.

Boundary rules
--------------
* **Training and inference are separate entry points.** ``train.py`` fits models
  and writes artifacts; ``predict.py`` only reads artifacts and never trains.
  This is architecture rule 4 and it keeps the production path free of any
  capability to silently overwrite a deployed model.
* ``train.py`` does not compute metrics. ``evaluate.py`` does, because FR-1.7
  selects the winning model on test-set F1 and that decision belongs to the
  evaluation layer, not the training layer.
* No module here imports from ``src.rag``, ``src.llm`` or ``src.attention``.
  Dependencies point one way only, toward ``src.pipeline``.

Requirements owned here
-----------------------
FR-1.5  Train a Logistic Regression and a Random Forest classifier on identical data.
FR-1.6  Classification report and confusion matrix for both models.
FR-1.7  Select the higher test-set F1 model and serialise it.
"""

from src.ml.evaluate import (
    ModelEvaluation,
    SelectionResult,
    evaluate_all_models,
    evaluate_model,
    plot_confusion_matrices,
    plot_roc_comparison,
    save_evaluation_artifacts,
    select_best_model,
)
from src.ml.predict import (
    ModelPrediction,
    RiskPrediction,
    load_model,
    load_risk_model,
    predict_patient,
    predict_risk,
)
from src.ml.train import (
    TrainedModels,
    build_models,
    save_model_artifacts,
    train_models,
)

#: The staged CLI lives in ``src/ml/cli.py`` rather than ``__main__.py`` so that
#: ``python -m src.ml.train`` does not execute this module a second time as
#: ``__main__`` after ``src.ml.__init__`` has already imported it.
from src.ml.cli import main as training_cli_main, run as run_training_stage

__all__ = [
    "run_training_stage",
    "training_cli_main",
    "build_models",
    "train_models",
    "save_model_artifacts",
    "TrainedModels",
    "evaluate_model",
    "evaluate_all_models",
    "select_best_model",
    "save_evaluation_artifacts",
    "plot_roc_comparison",
    "plot_confusion_matrices",
    "ModelEvaluation",
    "SelectionResult",
    "load_model",
    "load_risk_model",
    "predict_patient",
    "predict_risk",
    "ModelPrediction",
    "RiskPrediction",
]
