"""Central configuration for the mRNA Vaccine Intelligence & Protection (VIP) platform.

Every path, column name, model hyper-parameter and threshold used anywhere in the
project is declared here exactly once. Modules import from this module instead of
hard-coding values, which keeps the pipeline reproducible and makes it possible to
audit one file when a specification changes.

Provenance of the values below
------------------------------
* Paths and the directory layout follow the project architecture rules for the
  ``mRNA_VIP`` repository.
* ``REQUIRED_COLUMNS``, ``TARGET_COL``, ``NUMERIC_COLS`` and ``CATEGORICAL_COL`` are
  taken verbatim from the dataset table and code template in the Product
  Requirements Document (PRD), Sections 5.2 and 5.4.
* Hyper-parameters (``RANDOM_SEED``, ``TEST_SIZE``, ``LOGREG_PARAMS``,
  ``RANDOM_FOREST_PARAMS``) are taken verbatim from the PRD Section 5.4 template.
* Chunking, embedding and retrieval constants follow PRD Sections 7.3-7.4 and
  requirements FR-3.1, FR-3.2 and FR-3.4.

No API keys or other secrets are stored in this module. Secrets are read from
environment variables by the module that needs them (see ``.env.example``).
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------

PROJECT_NAME = "mRNA_VIP"
PROJECT_VERSION = "0.1.0"

#: Repository root. ``src/config.py`` -> parent is ``src``, grandparent is the root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Directory layout (architecture rules 1, 2, 10, 11, 12)
# ---------------------------------------------------------------------------

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"

#: Read-only regulatory / safety source document supplied with the project.
#: Architecture rule 2: never modify this file; the RAG index is built from it
#: at ingest time without mutating it on disk.
GUIDELINES_DIR: Path = PROJECT_ROOT / "guidelines"

SRC_DIR: Path = PROJECT_ROOT / "src"

#: Trained model artifacts only (architecture rule 10).
MODELS_DIR: Path = PROJECT_ROOT / "models"

#: Local ChromaDB index only (architecture rule 11).
VECTORSTORE_DIR: Path = PROJECT_ROOT / "vectorstore"

#: Generated artifacts, kept out of the source tree (architecture rule 12).
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"

DASHBOARD_DIR: Path = PROJECT_ROOT / "dashboard"
TESTS_DIR: Path = PROJECT_ROOT / "tests"


# ---------------------------------------------------------------------------
# Dataset contract (FR-1.1, FR-1.2, FR-1.3)
# ---------------------------------------------------------------------------

RAW_PATIENT_CSV: Path = DATA_RAW_DIR / "patient_records.csv"
PROCESSED_PATIENT_CSV: Path = DATA_PROCESSED_DIR / "patient_records_clean.csv"

GUIDELINE_DOCX: Path = GUIDELINES_DIR / "Moderna_mRNA_Safety_Guidelines.docx"

#: The 7 columns the PRD requires the loader to validate, in dataset order.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "Patient_ID",
    "Age",
    "Biomarker_Level",
    "Pre_Existing_Conditions",
    "Vaccine_Dose_mcg",
    "Prior_Reaction_History",
    "Reaction_Score",
)

TARGET_COL: str = "Reaction_Score"
ID_COL: str = "Patient_ID"

#: Standardised to zero mean / unit variance per FR-1.3.
NUMERIC_COLS: tuple[str, ...] = ("Age", "Biomarker_Level", "Vaccine_Dose_mcg")

#: Labelled-encoded per PRD Section 5.4. Imputed with the column mode (FR-1.2).
CATEGORICAL_COL: str = "Prior_Reaction_History"
CATEGORICAL_VALUES: tuple[str, ...] = ("None", "Mild", "Moderate")

#: Integer count feature carried through untransformed.
COUNT_COL: str = "Pre_Existing_Conditions"

#: Deterministic model input order. Column order is part of the model contract:
#: the fitted scaler and encoder must always receive features in this sequence.
FEATURE_COLS: tuple[str, ...] = NUMERIC_COLS + (CATEGORICAL_COL, COUNT_COL)


# ---------------------------------------------------------------------------
# Module 1 - Data pipeline and risk classifier (PRD Section 5)
# ---------------------------------------------------------------------------

#: FR-1.4 requires a fixed seed so repeated splits are identical.
RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2

#: Strings that mean "field absent" in the supplied CSV. Deliberately narrow.
#:
#: Pandas' default NA list contains the *string* ``"None"``. In this dataset
#: ``"None"`` is not an absent value - it is a legitimate
#: ``Prior_Reaction_History`` category meaning "no prior reaction", present in
#: 291 of 500 rows. Only 17 rows (3.4%) have a genuinely empty field, which is
#: the figure the PRD documents as "~3.5% missing". Reading with pandas defaults
#: therefore collapses 291 real observations into NaN, and the FR-1.2 mode
#: imputation would then overwrite the whole column with the majority category.
#: See ``src/data/preprocessing.py`` for the full evidence.
CSV_KEEP_DEFAULT_NA: bool = False
CSV_NA_VALUES: tuple[str, ...] = ("",)

MODEL_NAMES: tuple[str, str] = ("logistic_regression", "random_forest")
LOGREG_PARAMS: dict[str, object] = {"max_iter": 1000}
RANDOM_FOREST_PARAMS: dict[str, object] = {
    "n_estimators": 200,
    "random_state": RANDOM_SEED,
}

#: Human-readable names used in figures, legends and the machine-readable results.
MODEL_DISPLAY_NAMES: dict[str, str] = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}

#: FR-1.7 selects the winner on test-set F1.
SELECTION_METRIC: str = "f1"

#: Safety net from the PRD Scope & Constraints: a suspiciously perfect score is a
#: bug or leakage, not a success. Reported during evaluation, never enforced.
EXPECTED_TEST_F1_RANGE: tuple[float, float] = (0.60, 0.95)


# ---------------------------------------------------------------------------
# Module 2 - Self-attention engine (FR-2.1 ... FR-2.5)
# ---------------------------------------------------------------------------
#
# The PRD (Section 6.3) prescribes the input representation for this module: a
# nucleotide sequence over {A, C, G, U} mapped to a small learnable embedding
# table of 4 or 8 dimensions. That is the primary input mode and is what FR-2.1
# and FR-2.2 govern.
#
# A second input mode is provided because the module is also required to accept
# the project's processed patient representation. In that mode each encoded
# feature becomes one token, so the attention matrix is over real project features
# rather than over synthetic bases. It introduces no new variables: the tokens are
# exactly the columns produced by ``src.data.preprocessing``.

NUCLEOTIDE_VOCAB: tuple[str, ...] = ("A", "C", "G", "U")

#: Embedding width. The PRD says "start with 4 or 8 dimensions"; 8 is used.
ATTENTION_EMBED_DIM: int = 8

#: Query/key dimensionality. The equation divides by sqrt(d_k), so this is the
#: value that actually controls the softmax temperature.
ATTENTION_D_K: int = ATTENTION_EMBED_DIM

#: Value dimensionality. May differ from d_k without changing the equation.
ATTENTION_D_V: int = ATTENTION_EMBED_DIM

#: Scale applied to the weight initialisation. Small, so attention starts close to
#: uniform and the softmax does not saturate on the first forward pass.
ATTENTION_INIT_SCALE: float = 0.1

#: FR-2.1 accepts nucleotide strings of length 5-50.
MIN_SEQUENCE_LENGTH: int = 5
MAX_SEQUENCE_LENGTH: int = 50

#: FR-2.1 requires encode_sequence('ACGU') to return shape (4, embed_dim), so
#: the length floor must not be enforced during encoding - only by the separate
#: :func:`validate_sequence_length` helper. Keeping the two concerns apart is what
#: lets the four-base acceptance case and the 5-50 documented range both hold.
ATTENTION_ENCODING_EXAMPLE: str = "ACGU"

#: Seed used for parameter initialisation, so the same configuration reproduces
#: the same attention matrix. Distinct from RANDOM_SEED so attention
#: initialisation can be varied independently of the train/test split.
ATTENTION_SEED: int = 2024

#: The PRD's own worked example sequence (Section 6.4).
ATTENTION_DEMO_SEQUENCE: str = "ACGUACGUACGGCUA"

#: The required heatmap artefact name for this stage.
ATTENTION_HEATMAP_FILENAME: str = "mrna_attention_heatmap.png"

#: Neutral alias the dashboard front end requests for Panel B, mirroring the
#: ROC figure convention. Both names resolve to the same file.
ATTENTION_HEATMAP_SERVED_NAME: str = "attention_heatmap.png"
ATTENTION_HEATMAP_FILENAMES: tuple[str, ...] = (
    ATTENTION_HEATMAP_FILENAME,
    ATTENTION_HEATMAP_SERVED_NAME,
)

#: Figure size in inches for the standalone FR-2.5 heatmap.
ATTENTION_HEATMAP_FIGSIZE: tuple[float, float] = (8.0, 6.0)

#: Tick label size for the heatmap axes. Nucleotide labels are single characters,
#: but feature-mode labels are column names, so the size is chosen for the longer
#: case and rotated on the x axis.
ATTENTION_TICK_FONT_SIZE: int = 7

#: Rotation applied to the x-axis tick labels, which are rotated because
#: feature-mode column names are too long to sit horizontally.
ATTENTION_TICK_ROTATION: int = 90

#: Tolerance for the FR-2.4 row-sum check. float32 attention weights carry about
#: seven significant digits, so 1e-6 is comfortably above rounding noise and far
#: below any real deviation from 1.0.
ATTENTION_SUM_TOLERANCE: float = 1e-6

#: Floating-point tolerance for the PyTorch/NumPy parity check.
ATTENTION_PARITY_TOLERANCE: float = 1e-5

#: Input modes accepted by the module.
ATTENTION_INPUT_MODES: tuple[str, ...] = ("nucleotide", "feature")


# ---------------------------------------------------------------------------
# Module 3 - RAG retrieval index (FR-3.1 ... FR-3.5)
# ---------------------------------------------------------------------------

GUIDELINE_COLLECTION_NAME: str = "mrna_safety_guidelines"
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

#: Chunk sizing.
#:
#: **Measured deviation from PRD FR-3.1, and why.** FR-3.1 asks for chunks of
#: 300-500 tokens. The required embedding model, ``all-MiniLM-L6-v2``, has
#: ``max_seq_length = 256``: anything longer is silently truncated to its first
#: 256 tokens. That was verified empirically - two chunks sharing a 256-token
#: prefix but differing afterwards produce *identical* vectors (cosine
#: 1.00000007). A 300-500 token chunk would therefore have its tail discarded and
#: could collide with a neighbour, quietly degrading retrieval while still
#: appearing to satisfy FR-3.1.
#:
#: The supplied guideline document is only ~906 tokens in total, so chunks sized
#: under the model's limit lose nothing: the whole document is represented. These
#: values keep every chunk inside the embedding window, and
#: ``tests/test_rag.py`` asserts both the size bound and that no two chunks embed
#: to identical vectors.
CHUNK_SIZE_TOKENS: int = 180
CHUNK_OVERLAP_TOKENS: int = 40

#: Hard ceiling imposed by the embedding model. No chunk may exceed it, or its
#: tail is dropped by the tokenizer.
EMBEDDING_MAX_TOKENS: int = 256

#: Chunk sizing expressed in characters, used because the splitter counts
#: characters. ~4.1 characters per token for this document's prose.
CHUNK_SIZE_CHARS: int = 740
CHUNK_OVERLAP_CHARS: int = 165
CHUNK_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ")

#: Token bounds the implementation holds itself to. Tighter than FR-3.1's
#: 300-500 for the truncation reason documented above; FR-3.1's intent - chunks
#: large enough to retain context, with visible overlap - is preserved.
CHUNK_MIN_TOKENS: int = 120
CHUNK_MAX_TOKENS: int = 220

#: FR-3.1's literal bounds, retained so the deviation stays visible and testable
#: rather than quietly forgotten.
FR31_MIN_TOKENS: int = 300
FR31_MAX_TOKENS: int = 500

#: FR-3.4 default and ceiling for top-k retrieval.
RETRIEVAL_TOP_K: int = 3
RETRIEVAL_MAX_K: int = 10

#: Minimum cosine similarity between two distinct chunks' embeddings. Used as a
#: truncation probe: identical vectors would indicate the model discarded a tail.
EMBEDDING_COLLISION_THRESHOLD: float = 0.9999

#: FR-3.5 acceptance window for a single local retrieval call, in seconds.
RETRIEVAL_LATENCY_MIN_S: float = 0.0
RETRIEVAL_LATENCY_MAX_S: float = 5.0

#: Physical source document name, used verbatim as the ``source`` metadata value
#: on every chunk so retrieved evidence is traceable to the exact file it came from.
GUIDELINE_DOCUMENT_NAME: str = "Moderna_mRNA_Safety_Guidelines.docx"

#: ChromaDB collection and document-ID scheme.
#:
#: IDs are deterministic - ``<prefix>-<sha256[:16]>`` over the chunk text - so
#: re-running ingest replaces the same records instead of appending duplicates.
#: That makes ingestion idempotent, which is what stops repeated runs from
#: inflating the collection and skewing retrieval.
CHUNK_ID_PREFIX: str = "mrna_guideline"
CHUNK_ID_HASH_LENGTH: int = 16

#: Collection metadata key recording which embedding model produced the stored
#: vectors. Checked on load, because a collection cannot safely hold vectors from
#: two different models - their similarity spaces are not comparable.
CHROMA_COLLECTION_METADATA_KEY: str = "embedding_model"

#: Minimum distinct chunks and minimum cosine distance between the two closest
#: chunk embeddings. A collision means the embedding model truncated a tail.
EMBEDDING_COLLISION_THRESHOLD: float = 0.9999

#: Minimum similarity for a retrieved chunk to count as on-topic. Results below
#: this are still returned - the evidence is reported as retrieved - but the
#: caller can see they are weak.
RETRIEVAL_RELEVANCE_FLOOR: float = 0.15


# ---------------------------------------------------------------------------
# Module 4 - Structured LLM report (FR-4.1 ... FR-4.6)
# ---------------------------------------------------------------------------

#: Non-secret runtime settings. The provider API key is read from the
#: environment by ``src.llm.report_generator`` and never stored in source.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE: float = 0.0

#: FR-4.3 requires exactly one retry after a schema validation failure.
LLM_MAX_RETRIES: int = 1

#: FR-4.5 requires this notice in every report, without exception. The report
#: validator overwrites the field rather than trusting the model's output, and
#: the interface displays the same sentence in its disclaimer block.
REQUIRED_DISCLAIMER: str = (
    "This is a student engineering project and is not intended to provide real "
    "clinical guidance."
)

#: Upper bound on a generated dose, in micrograms. The guideline document treats
#: doses above 100 mcg as high-reactogenicity, so an LLM output above this value
#: is rejected by FR-4.4 regardless of what the model claims it read.
GUIDELINE_HARD_DOSE_CEILING_MCG: float = 100.0


# ---------------------------------------------------------------------------
# Module 5 - Dashboard (FR-5.1, FR-5.2, FR-5.3)
# ---------------------------------------------------------------------------

#: FR-5.3 requires the saved figure to report at least 150 DPI.
DASHBOARD_DPI: int = 150
DASHBOARD_FILENAME: str = "dashboard.png"
DASHBOARD_PANEL_TITLES: tuple[str, ...] = (
    "Risk Classifier ROC Curve",
    "mRNA Self-Attention Weights",
    "Pipeline Stage Latency (seconds)",
)


# ---------------------------------------------------------------------------
# Artifact filenames
# ---------------------------------------------------------------------------

#: Canonical locations (this repository's architecture).
#:
#: Each artifact is a complete ``sklearn.pipeline.Pipeline`` containing the fitted
#: preprocessing and the fitted estimator, so reloading one is enough to reproduce
#: predictions without re-deriving any transformation.
RISK_MODEL_PATH: Path = MODELS_DIR / "risk_model.joblib"
LOGISTIC_REGRESSION_MODEL_PATH: Path = MODELS_DIR / "logistic_regression.joblib"
RANDOM_FOREST_MODEL_PATH: Path = MODELS_DIR / "random_forest.joblib"
SELECTED_MODEL_PATH: Path = MODELS_DIR / "selected_model.joblib"
PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.joblib"
CHROMA_PERSIST_DIR: Path = VECTORSTORE_DIR

#: PyTorch state dictionary for the self-attention module. Weights only - the
#: architecture is reconstructed from this module's code, which is PyTorch's
#: recommended checkpoint format.
ATTENTION_MODEL_PATH: Path = MODELS_DIR / "self_attention.pt"

#: Model artifact path per model name, for the loader in ``src/ml/predict.py``.
MODEL_PATHS: dict[str, Path] = {
    "logistic_regression": LOGISTIC_REGRESSION_MODEL_PATH,
    "random_forest": RANDOM_FOREST_MODEL_PATH,
}

#: Machine-readable artefacts the dashboard reads back (FR-5.2). Each panel is
#: rebuilt from one of these files rather than from values held in memory.
ML_RESULTS_JSON_PATH: Path = LOGS_DIR / "ml_results.json"
METRICS_JSON_PATH: Path = LOGS_DIR / "metrics.json"
CLASSIFICATION_REPORT_JSON_PATH: Path = LOGS_DIR / "classification_report.json"
CONFUSION_MATRIX_JSON_PATH: Path = LOGS_DIR / "confusion_matrix.json"
ROC_CURVE_JSON_PATH: Path = LOGS_DIR / "roc_curve.json"
ATTENTION_WEIGHTS_PATH: Path = LOGS_DIR / "attention_weights.npy"
ATTENTION_SEQUENCE_PATH: Path = LOGS_DIR / "attention_sequence.txt"

#: Token labels for the attention matrix, one per position. Written alongside the
#: weights so the heatmap and the interface can label axes without re-deriving
#: which input mode produced the matrix.
ATTENTION_TOKEN_LABELS_PATH: Path = LOGS_DIR / "attention_token_labels.json"

#: Machine-readable summary of an attention run: shapes, dimensions, the strongest
#: attended pairs and the heatmap path. Written by ``save_attention_results``.
ATTENTION_RESULTS_PATH: Path = LOGS_DIR / "attention_results.json"
RETRIEVAL_LOG_JSON_PATH: Path = LOGS_DIR / "retrieval_log.json"
LATENCY_LOG_JSON_PATH: Path = LOGS_DIR / "latency_log.json"

#: Per-run LLM generation latencies. This is the file dashboard Panel C reads for
#: its "LLM Report" bar (``load_latency_panel_data``), which is distinct from
#: LATENCY_LOG_JSON_PATH above - that one holds per-stage pipeline timings, not
#: report generation times. Declared here so the writer and the reader name the
#: same constant rather than each spelling the path.
GENERATION_LOG_JSON_PATH: Path = LOGS_DIR / "generation_log.json"

FIGURE_CONFUSION_MATRIX_FILENAME: str = "confusion_matrix.png"
FIGURE_CLASSIFICATION_REPORT_FILENAME: str = "classification_report.png"
FIGURE_ROC_FILENAME: str = "ml_roc_curve.png"

#: Names the dashboard's client script requests for each panel figure.
#:
#: Panel A is served under the neutral name the front end already uses, and the
#: prompt's name is written as an identical copy, so both the FR-5.2 artifact path
#: and the interface reference resolve without the UI hard-coding a decision that
#: belongs here. ``dashboard/components.py`` maps one to the other.
FIGURE_ROC_SERVED_NAME: str = "roc_curve.png"
FIGURE_ROC_FILENAMES: tuple[str, ...] = (FIGURE_ROC_FILENAME, FIGURE_ROC_SERVED_NAME)

#: Confusion-matrix figure per model, matching the names the prompt requires.
CONFUSION_MATRIX_FILENAMES: dict[str, str] = {
    "logistic_regression": "logistic_regression_confusion_matrix.png",
    "random_forest": "random_forest_confusion_matrix.png",
}

#: Metrics reported for every model, in the order they are printed and stored.
METRIC_KEYS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1", "roc_auc")

REPORT_JSON_SUFFIX: str = ".json"
REPORT_MARKDOWN_SUFFIX: str = ".md"


# ---------------------------------------------------------------------------
# PRD path compatibility
# ---------------------------------------------------------------------------
# The PRD writes some acceptance criteria against a flat ``outputs/`` layout
# (for example FR-1.7's ``outputs/risk_model.joblib`` and Week 3's
# ``outputs/chroma_store/``). This repository keeps generated state in the
# dedicated directories required by its own architecture rules. The pipeline
# exports the PRD-named copies from the canonical paths so both documents are
# satisfied literally. See README.md, "PRD Path Mapping".

PRD_COMPAT_MODEL_PATH: Path = OUTPUTS_DIR / "risk_model.joblib"
PRD_COMPAT_VECTORSTORE_DIR: Path = OUTPUTS_DIR / "chroma_store"


# ---------------------------------------------------------------------------
# Generated-directory management
# ---------------------------------------------------------------------------

#: Directories that must exist before any stage runs. Created on demand by
#: ``ensure_directories``; the empty ones are tracked with ``.gitkeep``.
GENERATED_DIRS: tuple[Path, ...] = (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    GUIDELINES_DIR,
    MODELS_DIR,
    VECTORSTORE_DIR,
    OUTPUTS_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    LOGS_DIR,
)


def ensure_directories() -> None:
    """Create every generated-artifact directory if it does not already exist.

    Safe to call repeatedly and from any entry point. Keeping this in one place
    means no module has to defend against a missing output directory.
    """
    for directory in GENERATED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def describe_paths() -> dict[str, str]:
    """Return the resolved path map, for logging and for the README's audit."""
    return {
        "project_root": str(PROJECT_ROOT),
        "raw_patient_csv": str(RAW_PATIENT_CSV),
        "processed_patient_csv": str(PROCESSED_PATIENT_CSV),
        "guideline_docx": str(GUIDELINE_DOCX),
        "risk_model": str(RISK_MODEL_PATH),
        "chroma_persist_dir": str(CHROMA_PERSIST_DIR),
        "figures_dir": str(FIGURES_DIR),
        "reports_dir": str(REPORTS_DIR),
        "logs_dir": str(LOGS_DIR),
    }