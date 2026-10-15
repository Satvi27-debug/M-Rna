# mRNA Vaccine Intelligence & Protection (VIP)

An end-to-end applied AI platform for mRNA clinical safety and dosage intelligence:
classical predictive ML, Transformer self-attention mathematics, retrieval-augmented
generation over regulatory guidelines, and schema-validated LLM report generation.

> **Scope disclaimer.** This is an individually completed student engineering project
> referencing Moderna's public mRNA work. It is not a production clinical system. Every
> generated report carries a visible disclaimer stating that its output is a student
> engineering project and not real clinical guidance.

---

## 1. Status

All five modules are **implemented**, not stubbed. Every algorithmic body runs.

```
710 passed, 3 skipped
```

The 3 skips require a live LLM API key and are labelled as such. No test needs
network access, a paid token or credentials.

| # | Module | Status | Evidence |
|---|---|---|---|
| 1 | Data & machine learning | Done | Logistic Regression F1 **0.7021**, ROC-AUC **0.7679** on a 100-row held-out test set; byte-identical across runs |
| 2 | mRNA self-attention | Done | 15×15 matrix, rows sum to 1 within 1.2e-7; PyTorch/NumPy parity ≤1e-5 |
| 3 | Regulatory RAG | Done | 12 chunks, 384-dim, top-3 retrieval in 0.013–0.057 s |
| 4 | LLM + Pydantic report | Done | Schema + deterministic dose validation + retry-once; **no real LLM call has been made** (no key configured) |
| 5 | End-to-end pipeline | Done | Stages 1–4 verified against real components; stage 5 fails honestly without a key |
| 6 | Dashboard backend | Done | `POST /api/run` executes the real pipeline; verified live over HTTP with real retrieval and a real prediction |

**What has *not* been exercised:** report generation against a live LLM. Stage 4
and stage 5 of the pipeline are verified with `invoke_llm` patched. That is stated
wherever it matters rather than glossed, because a mocked response is not evidence
that a real call works.

**No fabricated results.** Every number above was measured. Where a result could
not be produced — no API key, an uninformative guideline chunk, a missing artifact —
the system reports the limitation instead of substituting a plausible value.

---

## 2. Architecture

The five modules of PRD Section 3.1, wired together left to right. Each module's output
is the next module's input.

```
[Patient Records CSV]   data/raw/patient_records.csv
        |
        v
[Module 1: Data Cleaning + ML Risk Classifier]  --->  risk_label, risk_probability
        |                                                   src/data, src/ml
        v
[Module 2: mRNA Sequence Self-Attention Engine] --->  attention_weights, heatmap
        |                                                   src/attention
        v
[Module 3: Regulatory RAG Index (ChromaDB)]      --->  relevant_guideline_chunks
        |                                                   src/rag
        v
[Module 4: LLM Structured Report Generator]      --->  ClinicalSafetyReport (JSON)
        |                                                   src/llm
        v
[Module 5: Dashboard + Portfolio Report]         --->  outputs/figures/dashboard.png
                                                            dashboard/
```

`src/pipeline/pipeline.py` is the only module permitted to import from all five
packages. Every other package imports downward only, so the dependency graph is acyclic
and each package stays independently testable.

### Directory responsibilities

| Path | Owns | Must never contain |
|---|---|---|
| `data/raw/` | The supplied dataset, unmodified | Any Python |
| `data/processed/` | The cleaned dataset | Any Python |
| `guidelines/` | The supplied regulatory source, unmodified | Any Python; never a rewritten copy |
| `src/data/` | Loading, validation, imputation, encoding, scaling, splitting | Model fitting |
| `src/ml/` | Training (`train`), evaluation (`evaluate`), inference (`predict`) | Training code in the inference path |
| `src/attention/` | Self-attention mathematics | Dashboard imports; patient/ML knowledge |
| `src/rag/` | Ingestion, chunking, embedding, persistence, retrieval | LLM prompt or report logic |
| `src/llm/` | Pydantic schema, safety validation, LLM call, persistence | Retrieval logic; safety rules in prompts alone |
| `src/pipeline/` | Orchestration and stage timing | Any algorithm of its own |
| `models/` | Trained artifacts (`joblib`) | Logs, reports |
| `vectorstore/` | The local ChromaDB index | Anything hand-edited |
| `outputs/` | `reports/`, `figures/`, `logs/` | Source code |
| `dashboard/` | UI composition only | ML, RAG or LLM logic |
| `tests/` | Behaviour tests mapped to FR identifiers | Fixtures that fake a result |

---

## 3. Requirements traceability

Every one of the PRD's 26 functional requirements maps to exactly one owning module.
This table is the audit trail for the Week 1–4 Definition of Done review.

### Module 1 — Data pipeline & risk classifier

| FR | Requirement | Owner |
|---|---|---|
| FR-1.1 | Validate 7 columns; named error identifies the missing one | `src/data/preprocessing.py::load_raw_records`, `MissingColumnsError` |
| FR-1.2 | Median for `Biomarker_Level`, mode for `Prior_Reaction_History` | `src/data/preprocessing.py::impute_missing_values` |
| FR-1.3 | Standardise `Age`, `Biomarker_Level`, `Vaccine_Dose_mcg` | `src/data/preprocessing.py::build_preprocessor`, `stratified_train_test_split` |
| FR-1.4 | 80/20 stratified split, fixed seed, reproducible | `src/data/preprocessing.py::stratified_train_test_split` |
| FR-1.5 | Train Logistic Regression **and** Random Forest on identical data | `src/ml/train.py::build_models`, `train_models` |
| FR-1.6 | Classification report + confusion matrix for **both** models | `src/ml/evaluate.py::evaluate_all_models`, `save_evaluation_artifacts` |
| FR-1.7 | Select higher test-F1 model; serialise for Module 4 | `src/ml/evaluate.py::select_best_model`; `src/ml/train.py::save_model_artifacts` |

### Module 2 — Self-attention engine

| FR | Requirement | Owner |
|---|---|---|
| FR-2.1 | Encode a 5–50 base string to `(seq_len, embed_dim)` | `NucleotideEmbedding`, `encode_sequence` |
| FR-2.2 | Reject characters outside `{A,C,G,U}` | `InvalidBaseError` |
| FR-2.3 | Q, K, V projections; `(seq_len, seq_len)` weights | `SelfAttention.project`, `SelfAttention.attention_weights` |
| FR-2.4 | Every weight row sums to 1.0 | `SelfAttention.attention_weights` (softmax over keys) |
| FR-2.5 | Labelled heatmap saved to `outputs/figures/` | `plot_attention_heatmap` |

### Module 3 — RAG retrieval index

| FR | Requirement | Owner |
|---|---|---|
| FR-3.1 | Chunks with visible overlap | `src/rag/ingest.py::chunk_guidelines`, `verify_chunk_bounds` |
| FR-3.2 | One dense embedding per chunk | `build_embedding_function`, `build_index` |
| FR-3.3 | Persisted local ChromaDB surviving restart | `build_index`, `load_index` |
| FR-3.4 | Top-k retrieval, exactly k, ordered, default k=3 | `src/rag/retrieve.py::query_guidelines` |
| FR-3.5 | Latency recorded and exposed per query | `RetrievalResult.latency_s`, `RetrievalLog` |

### Module 4 — Structured report generator

| FR | Requirement | Owner |
|---|---|---|
| FR-4.1 | Pydantic schema, exactly five fields | `src/llm/schemas.py::ClinicalSafetyReport` |
| FR-4.2 | Prompt grounded only in Module 1 + Module 3 output | `src/llm/report_generator.py::build_prompt` |
| FR-4.3 | Validate every response; retry exactly once | `generate_report`, `schemas.validate_report` |
| FR-4.4 | Reject dose above guideline bound; force ineligible | `schemas.enforce_safety_constraints`, `extract_dose_bound` |
| FR-4.5 | `Disclaimer` present in every report, no exceptions | `schemas.enforce_safety_constraints` |
| FR-4.6 | Persist as a `.json` + `.md` pair | `report_generator.persist_report` |

### Module 5 — Dashboard

| FR | Requirement | Owner |
|---|---|---|
| FR-5.1 | One figure, exactly 3 titled panels | `dashboard/app.py::build_dashboard` |
| FR-5.2 | Each panel from that module's saved output | `dashboard/components.py::load_*_panel_data` |
| FR-5.3 | Saved at ≥ 150 DPI | `dashboard/styles.py::FIGURE_DPI`, `save_figure` |

#### The interface layer

The application interface is implemented alongside the figure, in `dashboard/`:

| File | Role |
|---|---|
| `dashboard/styles.py` | Design tokens — one palette shared by the browser and the figure |
| `dashboard/components.py` | Matplotlib panels, empty-state copy, `platform_state()` |
| `dashboard/app.py` | The stdlib HTTP server, figure builder, and CLI |
| `dashboard/static/index.html` | Semantic shell containing all six pages |
| `dashboard/static/app.css` | Token-driven layout, responsive down to 360 px |
| `dashboard/static/app.js` | Hash router, state binding, form validation |

It runs on the Python standard library alone — no web framework, no bundler, no
`npm install`. Six pages: Overview, Patient Analysis, Risk Prediction, mRNA Attention,
Regulatory Intelligence, Safety Report.

**It consumes, it never recomputes.** `dashboard/` imports `src.config` and the Pydantic
report schema, and reads artifact files. It does not import `src.ml`, `src.rag`,
`src.llm.report_generator` or `src.pipeline`, and never trains, embeds or queries
anything. `sklearn` appears once, inside `plot_roc_panel`, to convert an already-saved
curve into its AUC label.

**An unrun stage is never shown as successful.** Readiness is derived from whether the
artifact a stage produces exists on disk. Two subtleties matter, and both were live bugs
during development:

- `.gitkeep` placeholders are ignored when checking a directory. Counting them as content
  made two unrun stages report themselves ready — a false success.
- `POST /api/run` returns HTTP 501 rather than a simulated result, so the loading, empty
  and error states are genuinely reachable rather than decorative.

`platform_state()` is covered by tests that constrain the metrics block to six
specification-derived keys and string-scan the whole payload for `roc_auc`, `accuracy`,
`f1` and `probability`, so a fabricated value cannot leak into the interface.

**Accessibility.** Lighthouse scores 100 / 100 / 100 (accessibility, best practices, SEO)
with zero failures. All nine form controls are labelled, status is conveyed by text and
shape as well as colour, and `prefers-reduced-motion` is honoured. One contrast
regression was caught by the audit and fixed: `text-subtle` at `#64748B` measured 4.47:1
on the canvas and 4.34:1 on the sunken surface, both under WCAG AA. It is now `#5B6B7E`,
clearing every surface in the palette at 4.98:1 or better, with a parametrised test
guarding all 36 token/surface combinations.

---

## 4. Design decisions worth reviewing

These are judgement calls. Each is traceable to a specific requirement, and each is a
point a mentor may reasonably want to discuss.

**Safety is enforced in code, not only in the prompt.** PRD Section 8.2 places four
constraints in the system prompt. Three of them are model-behaviour guidance. The dose
constraint is *additionally* enforced in `enforce_safety_constraints`, because FR-4.4
requires a test case to produce `Eligible_For_Vaccine == false` — and a prompt
instruction cannot be the thing an acceptance test depends on. The prompt asks; the code
verifies and overrides.

**The disclaimer is overwritten, not merely defaulted.** The Pydantic `Disclaimer` field
carries a default, but `enforce_safety_constraints` replaces it with
`config.REQUIRED_DISCLAIMER` unconditionally, satisfying FR-4.5's "without exception"
even when the model omits or weakens it.

**Chunk sizes are declared in characters and verified in tokens.** LangChain's
`RecursiveCharacterTextSplitter` counts characters. FR-3.1 specifies *tokens*. Treating
the splitter's `chunk_size=500` as 500 tokens would silently produce ~120-token chunks
and fail FR-3.1 while appearing correct. So sizes are declared in characters
(`CHUNK_SIZE_CHARS = 1500`) and `verify_chunk_bounds` measures actual token counts as the
acceptance test.

**The `.docx` is read with the standard library.** The PRD describes the guideline source
as `mrna_safety_guidelines.txt`, but the file actually supplied is a `.docx`. A `.docx`
is a zip archive whose `word/document.xml` holds the paragraphs, so `extract_docx_text`
reads it with `zipfile` and `xml.etree` in ~30 lines. This avoids adding a Word-parsing
dependency that nothing else in the project uses, keeping `requirements.txt` limited to
packages the implementation genuinely requires. The source file is opened read-only and
never rewritten.

**One retrieval query per clinical dimension.** A 72-year-old with two pre-existing
conditions matches two different guideline passages. A single combined query tends to
retrieve only the passage that dominates the wording, so `build_retrieval_queries`
derives one question per attribute, making each citation traceable to the clinical factor
that triggered it.

**Training and inference are separate files with no shared write path.** `predict.py`
contains no training code, so a bug in the production path cannot overwrite a deployed
model. The fitted scaler and encoder are serialised alongside the model, so an unseen
patient is transformed exactly as training rows were.

**Leakage guards are asserted, not assumed.** `tests/test_preprocessing.py` asserts the
scaler saw only the 400 training rows via `n_samples_seen_`. `tests/test_ml.py` asserts
test F1 lands in `config.EXPECTED_TEST_F1_RANGE` (0.60–0.95), because PRD Section 5.2
warns that a suspiciously perfect classifier signals leakage, not success.

**Known caveat to raise with your mentor.** The PRD Section 5.4 template uses
`LabelEncoder` on `Prior_Reaction_History`. `LabelEncoder` assigns integers by
*alphabetical* order, which yields `Mild → 0`, `Moderate → 1`, `None → 2`. So "no prior
reaction" is encoded as the *highest* value, which is semantically backwards and lets a
linear model learn an inverted relationship. The implementation follows the PRD as
written. If your mentor prefers ordinal semantics, an explicit
`{"None": 0, "Mild": 1, "Moderate": 2}` mapping is the one-line change — but it is a
deliberate deviation from the supplied template, so make that call explicitly rather than
silently.

---

## 5. PRD path mapping

The PRD's Section 3.3 suggests a flat `src/*.py` layout with generated files under
`outputs/`. This repository uses the package layout in the project architecture rules,
which assigns generated state to dedicated directories. Rather than leave either document
unsatisfied, both paths are honoured and the mapping is explicit:

| PRD path | Canonical path here | Reconciliation |
|---|---|---|
| `data/patient_records.csv` | `data/raw/patient_records.csv` | Architecture rule 1 separates raw from processed |
| `data/patient_records_clean.csv` | `data/processed/patient_records_clean.csv` | Same |
| `outputs/risk_model.joblib` (FR-1.7) | `models/risk_model.joblib` | Written to both by `save_model_artifacts` |
| `outputs/chroma_store/` (Week 3 DoD) | `vectorstore/` | Written to both by `export_index_compatibility_copy` |
| `mrna_safety_guidelines.txt` | `guidelines/Moderna_mRNA_Safety_Guidelines.docx` | Supplied as `.docx`; text extracted in memory, source unmodified |
| `src/dashboard.py` | `dashboard/app.py`, `dashboard/static/` | Architecture rule 9 |
| `src/risk_model.py` | `src/ml/train.py`, `evaluate.py`, `predict.py` | Architecture rule 4 |
| `src/attention.py` | `src/attention/self_attention.py` | Package layout |
| `src/rag_index.py` | `src/rag/ingest.py`, `retrieve.py` | Ingestion and retrieval split, per architecture rule 6 |

**One PRD element is not yet scaffolded.** PRD Sections 3.3 and the Week 1 Definition of
Done call for `notebooks/01_data_eda.ipynb` (at least three EDA visualisations). It is
intentionally absent from the requested file tree, so no empty notebook directory was
created. Track it as a Week 1 deliverable.

---

## 6. Setup

Requires Python 3.10+ (developed against 3.13).

```powershell
# macOS / Linux / Git Bash
python -m venv venv && source venv/bin/activate

# Windows PowerShell
python -m venv venv; .\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

`requirements.txt` is the PRD Section 11.1 dependency set, with a single addition:
`pytest`, which the `tests/` directory in the requested architecture requires. Nothing has
been added for popularity's sake.

> **On pinning.** The PRD checklist asks for pinned versions, while the PRD's own
> Section 11.1 template uses `>=`. The PRD's template is reproduced verbatim. Before final
> submission, capture the exact resolved versions so the checklist item is satisfied:
> `pip freeze > requirements.lock.txt`.

### Environment variables

```powershell
Copy-Item .env.example .env      # Windows PowerShell
cp .env.example .env            # macOS / Linux
```

Only `LLM_API_KEY` is a secret, and it is needed for Modules 3–4 only. `.env` is listed
in `.gitignore` and **must never be committed**; `.env.example` contains variable names
only, with no credentials. Keys are read at call time inside `load_api_key`, never at
import time, so the test suite runs without a key.

---

## 7. Module 1 — Data & Machine Learning

Implemented and executed. Every number below was produced by an actual run of
`python -m src.ml`.

### Dataset

`data/raw/patient_records.csv` (supplied, never modified). Measured, not assumed:

| Property | Value |
|---|---|
| Shape | 500 rows × 7 columns |
| Duplicate rows / duplicate `Patient_ID` | 0 / 0 |
| `Age` | 18 – 85 (mean 51.44) |
| `Biomarker_Level` | 0.64 – 15.76 (median 4.82), **20 missing (4.0%)** |
| `Pre_Existing_Conditions` | 0 – 4 |
| `Vaccine_Dose_mcg` | 25.1 – 150 |
| `Prior_Reaction_History` | None / Mild / Moderate, **17 genuinely empty (3.4%)** |
| `Reaction_Score` | 255 zeros / 245 ones (49.0% positive) |

`python -m src.ml` prints this profile before doing anything else, via
`preprocessing.inspect_dataset`.

### The `"None"` finding

`Prior_Reaction_History` appears **61.6% missing** under a default
`pd.read_csv`, but only **3.4%** of fields are actually empty.

Pandas' default NA list contains the *string* `"None"`. In this dataset `"None"`
is a real category — no prior reaction — present in 291 of 500 rows. Reading
with defaults silently converts those 291 observations to `NaN`, after which the
FR-1.2 mode imputation would overwrite the entire column with the majority level
and destroy the strongest categorical signal. The class-conditional target rates
show why that matters:

| `Prior_Reaction_History` | n | mean `Reaction_Score` |
|---|---|---|
| None | 291 | 0.447 |
| Mild | 150 | 0.520 |
| Moderate | 42 | 0.667 |

The PRD's own documentation settles it: the column is listed as "None / Mild /
Moderate" and missing values are stated as "~3.5% of rows" — 17 rows, not 308.
The loader therefore reads with `keep_default_na=False, na_values=[""]`, declared
in `config.CSV_KEEP_DEFAULT_NA` / `config.CSV_NA_VALUES`.

### Target definition

**`Reaction_Score` is used directly as the binary target. No threshold is
applied and no target is re-derived.**

PRD Section 5.2 defines the column as `Target (Integer 0-1)` with `0 = Low
reaction risk, 1 = High reaction risk`. It is already the classification label.
The risk band names in `predict.RISK_LABELS` are that same definition
transcribed, not a new clinical judgement.

### Features

Five columns, in this fixed order. **`Patient_ID` is never a feature** — it is an
identifier, and `ColumnTransformer(remainder="drop")` excludes it.

| Feature | Encoding |
|---|---|
| `Age` | median imputation → `StandardScaler` |
| `Biomarker_Level` | median imputation → `StandardScaler` |
| `Vaccine_Dose_mcg` | median imputation → `StandardScaler` |
| `Prior_Reaction_History` | mode imputation → `OneHotEncoder` (3 levels) |
| `Pre_Existing_Conditions` | passthrough |

Seven encoded columns result. **Deviation from the PRD sketch, documented:** PRD
Section 5.4 sketches `LabelEncoder`, which assigns integers alphabetically and so
yields `Mild → 0, Moderate → 1, None → 2` — the *absence* of prior reactions
becomes the highest value, and a linear model learns a backwards relationship.
PRD Section 2 states requirements are testable statements "independent of exactly
how you implement it", and the requirement is only that the column be encoded.
`OneHotEncoder` satisfies that without imposing a false order.

### Pipeline order

```
load → inspect → validate → impute
      → X/y separation → 80/20 stratified split (seed 42)
      → fit preprocessing on TRAIN ONLY → transform
      → train both models → evaluate on test
```

Imputation and scaling are fitted on the 400 training rows only. Verified by
asserting `StandardScaler.n_samples_seen_ == 400`, and paired with a test that
proves the train and test distributions genuinely differ — so the guard cannot
pass vacuously.

### Model configuration

No hyper-parameter search; both use the PRD Section 5.4 values verbatim.

| Model | Configuration |
|---|---|
| Logistic Regression | `max_iter=1000` (L-BFGS does not converge within the default budget) |
| Random Forest | `n_estimators=200`, `random_state=42` |

Each artifact is a complete `Pipeline([ColumnTransformer, classifier])`, so one
`joblib.load()` is sufficient to reproduce predictions.

### Evaluation — actual held-out test results

100 test rows, stratified to 51 / 49.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.7200** | **0.7333** | **0.6735** | **0.7021** | **0.7679** |
| Random Forest | 0.6300 | 0.6200 | 0.6327 | 0.6263 | 0.7095 |

Precision, recall and F1 are for the positive class (`Reaction_Score == 1`);
accuracy is overall.

Confusion matrices — Logistic Regression: 39 / 12 / 16 / 33. Random Forest:
32 / 19 / 18 / 31.

**FR-1.7 selection: Logistic Regression**, on test-set F1 (0.7021 > 0.6263), the
criterion the PRD specifies. The runner-up is *not* discarded — `predict_patient`
returns both models' output, because the PRD does not declare one model clinically
better and selection is a modelling decision, not a medical one.

The Logistic Regression outperforming the Random Forest is worth noting rather
than hiding: with 400 training rows, 7 features and a relationship the PRD
describes as "realistic but noisy", the forest's extra capacity mostly fits noise.
A tuned forest might close the gap, but no tuning was performed, per the brief.

### Generated artifacts

| Path | Contents |
|---|---|
| `models/logistic_regression.joblib` | Fitted pipeline |
| `models/random_forest.joblib` | Fitted pipeline |
| `models/selected_model.joblib` | FR-1.7 winner |
| `outputs/risk_model.joblib` | PRD-named compatibility copy (FR-1.7 acceptance path) |
| `outputs/logs/ml_results.json` | The required machine-readable comparison |
| `outputs/logs/metrics.json` | Metrics + selection + split provenance |
| `outputs/logs/classification_report.json` | Full per-class reports |
| `outputs/logs/confusion_matrix.json` | Per-model matrices |
| `outputs/logs/roc_curve.json` | ROC points the dashboard reads |
| `outputs/logs/ml_training_metadata.json` | Feature names, seed, imputation statistics |
| `outputs/figures/ml_roc_curve.png` | ROC comparison @ 150 DPI |
| `outputs/figures/logistic_regression_confusion_matrix.png` | |
| `outputs/figures/random_forest_confusion_matrix.png` | |
| `data/processed/patient_records_clean.csv` | Cleaned dataset, 0 missing cells |

### Reproducibility

Seed 42, fixed throughout, centralised in `src/config.py`. Verified by running the
stage three times and confirming `ml_results.json` was byte-identical each time.
The Random Forest is seeded; L-BFGS is deterministic and needs no seed.

### Interface integration

The dashboard consumes these artifacts rather than recomputing anything.
`dashboard/components.py::_evaluated_metrics` reads `ml_results.json` and passes
through only the five declared metric keys, so the results file cannot widen what
the UI presents. With no results file the Risk Prediction page falls back to its
empty state. Tests assert both directions.

---

## 8. Module 2 — mRNA Self-Attention

Implemented in PyTorch. Every mathematical step is written out explicitly in
`src/attention/self_attention.py`; no high-level attention layer is called.

### What self-attention does

It lets every position of a sequence decide how much to attend to every other
position, in one step, without recurrence. Each position produces three vectors:

| Vector | Role |
|---|---|
| **Q (Query)** | what this position is looking for |
| **K (Key)** | what this position offers to be matched against |
| **V (Value)** | the content this position contributes once matched |

A position's query is compared against every key. High similarity means "this is
what I was looking for", and the softmax turns those scores into a distribution —
the share of attention the position allocates to each candidate.

### The equation

```
Q = X Wq,  K = X Wk,  V = X Wv                linear projections, (seq_len, d_k)

scores = Q Kᵀ / √dk                           pairwise similarity, (seq_len, seq_len)
A      = softmax(scores)                      row-wise probability distribution
output  = A V                                 attended representation, (seq_len, d_v)
```

**Why divide by √dk?** Without it, `Q Kᵀ` grows as √dk, so the softmax receives
large logits and saturates — its gradient approaches zero and learning stalls.
Dividing by √dk holds the logits near unit scale regardless of embedding width.
This is `scaling` in the code, asserted in tests to equal `sqrt(d_k)`.

Implemented as three operations in `forward`: `project` (Q/K/V), `attention_weights`
(`Q Kᵀ`, scaling, softmax) and `weights @ V`.

### What the attention matrix represents

Row *i*, column *j* is the share of position *i*'s attention going to position *j*.
Every row sums to 1 (FR-2.4), verified at `atol=1e-6` across sequence lengths 5–50.

Dimensions, all in `src/config.py`:

| Quantity | Value | Note |
|---|---|---|
| Embedding width | 8 | PRD §6.3 says "4 or 8"; 8 chosen |
| Key/query `d_k` | 8 | sets the softmax temperature |
| Value `d_v` | 8 | may differ from `d_k`; a test covers that |
| Sequence length | 5–50 | FR-2.1; the 4-base `encode_sequence` case also works |

Single-head by design — the PRD lists multi-head as an optional bonus (§6.5), and
one head keeps the equation legible.

### Two input modes

Both use existing project data; neither invents a variable.

**`nucleotide`** (default, the PRD's prescription) — a sequence over `{A, C, G, U}`
mapped to a learnable embedding table of 8 dimensions, so *n* bases become an
`(n, 8)` matrix. Invalid characters raise `InvalidBaseError` (FR-2.2); the documented
5–50 length range is enforced by `validate_sequence`.

**`feature`** — each column of the project's processed patient representation
becomes one token, giving a 7×7 matrix labelled with the real encoded columns:
`Age`, `Biomarker_Level`, `Vaccine_Dose_mcg`, `Prior_Reaction_History_{None,Mild,Moderate}`,
`Pre_Existing_Conditions`.

```bash
python -m src.attention                  # nucleotide demo (PRD §6.4 example)
python -m src.attention --feature        # over a processed patient record
python -m src.attention --sequence ACGUACGUACGGCUA
```

### How the heatmap is generated

`plot_attention_heatmap` renders the real weights returned by the module — the same
tensor the forward pass produced — with `seaborn.heatmap`. Axis labels are the actual
token names, generated from the input rather than typed in, and rotated 90° because
feature-mode labels are column names. Saved at 150 DPI (FR-5.3) to
`outputs/figures/mrna_attention_heatmap.png`, with a byte-identical alias at
`attention_heatmap.png` for the interface's Panel B.

### Reproducibility and correctness

Seeded at `config.ATTENTION_SEED = 2024`, separate from the training seed so
attention initialisation can be varied independently. Constructors take an explicit
`seed`, making them pure functions of their arguments rather than of ambient RNG
state. Three independent checks:

1. **Row sums** — `atol=1e-6`, lengths 5–50.
2. **PyTorch/NumPy parity** — `self_attention_numpy` re-derives the equation
   independently; `assert_parity` confirms agreement within `1e-5`.
3. **Gradient flow** — all three projection matrices receive non-zero gradients.

### What this does NOT prove

**The projections are randomly initialised and untrained.** The attention matrix is a
faithful picture of *this arithmetic on this input* — nothing more. A large weight at
`(i, j)` says position *i* allocated a large share of its softmax distribution to
position *j* under these particular random projections. It is **not** evidence that
one feature influences another, and certainly not a biological or clinical
relationship.

The interpretation utility is written to keep that boundary visible: it reports
weight mass and distribution uniformity, states that the figures describe the matrix
under random initialisation, and carries an explicit `interpretation_limits` field. A
test asserts the narrative contains no causal or clinical vocabulary.

Typical output confirms the behaviour: with random projections the mean normalised
entropy is ~0.998, i.e. the distributions are near-uniform — exactly what untrained
attention should look like.

### Artifacts

| Path | Contents |
|---|---|
| `outputs/figures/mrna_attention_heatmap.png` | Real heatmap @ 150 DPI (+ `attention_heatmap.png` alias) |
| `outputs/logs/attention_weights.npy` | `(seq_len, seq_len)` matrix |
| `outputs/logs/attention_token_labels.json` | Token labels + input mode + dimensions |
| `outputs/logs/attention_results.json` | Shapes, dimensions, strongest pairs, summary |
| `models/self_attention.pt` | State dict (`Wq`, `Wk`, `Wv`) + dimensions |

The checkpoint is PyTorch's recommended format — weights only, with the architecture
reconstructed from code. Nothing here is trained, so no optimiser state is saved.

---

## 9. Module 3 — Regulatory RAG

Implemented. Retrieval over the supplied guideline document, with no LLM involved.

### Document

`guidelines/Moderna_mRNA_Safety_Guidelines.docx` — read-only, never modified.
Measured structure: 47 paragraphs, 2 tables, ~906 tokens, 4 top-level sections
(Storage, Dosing, Contraindications, Adverse Events) with 12 subsections.

### Process

```
Moderna_mRNA_Safety_Guidelines.docx   python-docx, heading tree preserved
            ↓
      Section-aware chunking           12 chunks, overlap preserved
            ↓
      all-MiniLM-L6-v2                 384-dim, runs locally
            ↓
      ChromaDB                          vectorstore/, persistent
            ↓
      Query + similarity search
            ↓
      Top 3 chunks                      verbatim text + provenance
```

```powershell
python -m src.rag.ingest --reset          # extract, chunk, embed, persist
python -m src.rag.retrieve --query "What are the storage requirements below -20C?"
python -m src.rag.retrieve --dose 120      # dose-specific evidence
python -m src.rag.retrieve --evaluate      # 8-query set + latency log
```

### Why retrieval, and why traceability

A language model asked for a dose limit will confidently produce a plausible
number whether or not it exists in the source. So this layer answers nothing. It
finds the actual sentences bearing on a question and returns them verbatim, each
carrying `source`, `section`, `heading_path` and `chunk_id`. The next stage decides
what to conclude; this stage only ensures the conclusion rests on retrievable
evidence. A test asserts retrieved text appears verbatim in the extracted
document, and another asserts no LLM library is imported anywhere in the package.

### Top-k = 3

FR-3.4 specifies top-3, and that is the default (`config.RETRIEVAL_TOP_K = 3`).
Three chunks is enough to cover a question that spans a definition, a threshold
and a warning without burying the relevant passage in near-duplicates. Each result
carries both a similarity score and the raw distance, so the transform is never
the only view available.

### Chunk sizing — a measured deviation from FR-3.1

FR-3.1 asks for 300–500 token chunks. **The required embedding model truncates at
256 tokens**, verified empirically: two texts sharing a 256-token prefix embed to
*identical* vectors (cosine 1.00000007). A 300–500 token chunk would have its tail
silently discarded and could collide with a neighbour — degrading retrieval while
still appearing to satisfy FR-3.1.

The document is only ~906 tokens, so smaller chunks lose nothing. Actual result:
**12 chunks, 60–123 tokens, all within the 256 limit, minimum 5 tokens of overlap.**
FR-3.1's literal bounds are retained in `config.FR31_MIN_TOKENS`/`MAX` so the
deviation stays visible, and a test asserts no two distinct chunks embed to
identical vectors — the failure FR-3.2's count check cannot catch.

### Measured retrieval behaviour

All 8 evaluation queries return exactly 3 results, 0.035–0.057 s each (mean 0.042 s).
Dose evidence for a 120 mcg candidate retrieves **25.0, 50.0, 100.0** mcg from the
document, with `exceeds_stated_max = True` — a 100 mcg ceiling read out of the
retrieved text, not configured.

**Query wording materially affects results.** Phrasing a question in the
guideline's own vocabulary ("standard warnings ... monitored for at least 15
minutes") scores 0.698; paraphrasing it ("How long must a patient be monitored?")
scores 0.497. The evaluation queries are therefore written in the document's terms.
This is a real limitation of embedding retrieval, quantified rather than hidden —
it is the failure case the Week 3 write-up asks for.

### Idempotent ingestion

Chunk IDs are `sha256(text)[:16]`, so re-running ingest **upserts** rather than
appends. Verified: two consecutive runs leave the count unchanged. Collection
metadata records the embedding model, and `load_index` refuses an index built with
a different one — vectors from different models are not comparable.

### Artifacts

| Path | Contents |
|---|---|
| `vectorstore/` | Persistent ChromaDB collection `mrna_safety_guidelines` |
| `outputs/logs/retrieval_log.json` | Per-query latency, for dashboard Panel C |
| `outputs/chroma_store/` | PRD-named compatibility copy |

---

## 10. Module 4 — LLM + Pydantic structured safety report

Implemented. `gpt-4o-mini` via LangChain `ChatOpenAI`, every response schema-validated.

### Document

No new document. The grounding is `guidelines/Moderna_mRNA_Safety_Guidelines.docx`
as retrieved by Module 3 — the LLM is never asked to supply regulatory text.

### Process

```
patient record (patient_records.csv)
   + Module 1 risk band + P(Reaction_Score = 1)
   + Module 3 top-3 retrieved chunks
            ↓
   build_prompt()          labels each block's origin (FR-4.2)
            ↓
   ChatOpenAI → raw text
            ↓
   parse_llm_json()        tolerates fences and prose
            ↓
   validate_report()       Pydantic, 5 fields (FR-4.1)
            ↓  on failure: retry exactly ONCE (FR-4.3)
            ↓
   enforce_safety_constraints()   deterministic, cannot be overridden (FR-4.4/4.5)
            ↓
   persist_report()        .json + .md pair (FR-4.6)
```

### The five required fields (FR-4.1, PRD §8.3)

`Eligible_For_Vaccine` (bool) · `Recommended_Dose_mg` (float) · `Reasoning_Steps`
(List[str]) · `FDA_Safety_Warnings_Cited` (List[str]) · `Disclaimer` (str).
Four are mandatory; a missing or mistyped field raises `ValidationError`.

### Why safety rules are code, not prompt wording

The prompt asks the model to respect the retrieved dose bounds. That cannot be
relied on to satisfy an acceptance test, so `enforce_safety_constraints` verifies
and overrides it in Python. Verified end to end: a model output claiming
`Eligible_For_Vaccine=True` at **500 mcg** is rewritten to `False` with the dose
clamped to **0.1 mg** and an `FR-4.4` violation recorded.

### Dose bound provenance

The ceiling is **read out of the retrieved text**, not configured. Only a sentence
stating both a microgram figure *and* a consequence for crossing it counts as an
enforceable ceiling — so `"contains 50 micrograms of active material"` is correctly
rejected as a limit, while `"Doses above 100 micrograms … require secondary
clinician review"` is accepted. Measured on the real index: ceiling **100 mcg**,
evidence sentence carried through.

When no applicable maximum can be established from the retrieved text, the
function **defaults rather than invents**, and returns an empty `source_text` so
callers can tell a derived bound from a defaulted one. The dashboard reports
`dose_bound: null` for the same reason.

### Retry policy (FR-4.3)

Exactly **two total attempts**. The retry prompt carries the field-level
validation errors from attempt 1, because a retry with an identical prompt cannot
do better.

API failures are handled **separately** and are *not* charged to the validation
retry: a credential or network fault raises `LLMCallError` immediately, rather
than burning the retry and then reporting the problem as a schema failure.

### Credentials

`LLM_API_KEY` (or `OPENAI_API_KEY`) is read from the environment **at call time**,
never at import time. Missing → `MissingAPIKeyError` naming the variable and the
`.env` path. The key is never logged and never appears in an exception message.

### Measured results

Mocked `invoke_llm`, real ML prediction and real retrieval:

| Check | Result |
|---|---|
| Patient P0001, real model | Low risk, P(Reaction_Score=1) = 0.3320 |
| Retrieved chunks / bounds | 3 · [25.0, 50.0, 100.0] mcg |
| Dose bound derived from text | 100.0 mcg, evidence sentence captured |
| LLM calls / attempts | 1 / 1 (validated first try) |
| 83.5 mcg dose | eligible **True**, compliant |
| 500 mcg hostile output | eligible **False**, clamped 0.1 mg, `FR-4.4` |
| Persisted pair | `P0001-<UTC timestamp>.json` / `.md`, disclaimer exact |
| Repeat runs | unique filenames, earlier report byte-identical |

### Known environment issue: ML + RAG in one process

On this Windows machine, loading a scikit-learn model and then performing **any**
ChromaDB HNSW operation (including `collection.count()`) terminates the process with
an access violation (`0xC0000005`). Reproducible in that order; **not** avoided by
`KMP_DUPLICATE_LIB_OK=TRUE`. A native conflict between scikit-learn's BLAS runtime
and ChromaDB's Rust HNSW extension.

Module 4 is unaffected — it loads neither library. The end-to-end check above runs
ML and RAG in separate processes for that reason. **The pipeline stage that combines
all modules in one interpreter will hit this**, and a process boundary is the
workaround.

### Commands

```powershell
python -m pytest tests/test_report.py          # 172 passed, 3 skipped (keyless)
python -m pytest                               # 536 passed, 3 skipped
```

No API key, network, or paid token is needed: every unit test patches
`invoke_llm`, the single network seam. Tests needing a live key skip explicitly.

---

---

## 11. Module 5 — End-to-end pipeline integration

Implemented. Orchestration only: each stage delegates to the module that already
implements it, and this layer decides order, data contracts, timing and failure
handling.

### How to run it

```powershell
# Module 1 only - trains and evaluates, writes models/
python -m src.pipeline --train

# Module 3 only - rebuilds the ChromaDB index
python -m src.pipeline --index

# One patient end to end, all stages, writes the report pair
python -m src.pipeline --run --patient-id P0001

# Everything, plus the three-panel figure
python -m src.pipeline --all

# Programmatic
python -c "from src.pipeline import run_pipeline, load_patient; run_pipeline(load_patient('P0001'))"
```

### Required environment variables

| Variable | Required | Purpose |
|---|---|---|
| `LLM_API_KEY` | **Only for the report stage** | Report generation. `OPENAI_API_KEY` is accepted as a fallback. |
| `LLM_MODEL` | No | Defaults to `gpt-4o-mini` in `src/config.py`. |
| `LLM_BASE_URL` | No | Any OpenAI-compatible endpoint. |

Stages 1–4 and the dashboard run entirely locally with **no** API key. The key is
read at call time via `python-dotenv`; `.env` is gitignored and never committed.

### Stage order and statuses

```
preprocessing -> ml_prediction -> attention -> rag_retrieval -> llm_report -> persist
```

Each stage reports one of `not_started`, `running`, `completed`, `failed`,
`blocked`. The rules that matter:

- A failed stage is **never** reported completed.
- A stage after a failure does not run, and records `not_started`.
- A run with any **required** failure cannot be `completed`.
- Optional stages (`train`, `build_dashboard`) may fail without invalidating a run.
- A `blocked` stage means its inputs were missing — different cause, different fix.
- Completed stages' outputs survive a later failure, on `PipelineError.result`,
  with `status='failed'` and `partial=True`. A real prediction is more useful than
  none, provided it cannot be read as a success.

### Attention input: no patient sequence exists

The supplied dataset has **no sequence column**. The PRD asks for a *sample*
mRNA sequence of 15–30 bases, so the pipeline uses
`config.PIPELINE_DEFAULT_SEQUENCE` (`ACGUACGUACGGCUA`) and records
`sequence_source='config_default'`. **No patient sequence is invented.** Pass
`--sequence` to override, or include an `mRNA_Sequence` key in the record.

Attention is a **required** stage: the PRD's Week 4 definition of done needs the
heatmap as Panel B.

### Native-library diagnostic — the important one

**Symptom.** Loading scikit-learn before torch or ChromaDB in one interpreter
breaks both, on Windows 11 / Python 3.13.5:

| import order | outcome |
|---|---|
| `sklearn → torch` | `OSError [WinError 1114]` loading `c10.dll` |
| `sklearn → chroma` | **native crash, exit `0xC0000005`** (uncatchable) |
| `torch → chroma → sklearn` | **OK** |
| `chroma → torch → sklearn` | **OK** |

**Method.** `src/pipeline/native_stack.py` runs each case in a **fresh
subprocess** and reads the exit code — a native fault cannot be caught as a Python
exception, so in-process testing cannot observe one.

```powershell
python -m src.pipeline.native_stack     # prints the matrix
```

**Diagnosis: import order, not incompatibility.** scikit-learn pulls in an
OpenMP/MKL runtime; once resident, torch's `c10.dll` fails to initialise and
ChromaDB's Rust HNSW extension faults. Loading scikit-learn *last* makes all three
coexist — verified 5/5 identical runs of the real pipeline sequence.

`KMP_DUPLICATE_LIB_OK=TRUE` does **not** help. Verified, and nothing depends on it.

**Fix.** `preload_native_libraries()` imports torch and ChromaDB before
scikit-learn, called at the top of `pipeline.py` ahead of every sklearn-reaching
import. Process isolation was the first candidate and is **unnecessary** here — a
single process runs all four modules correctly. `tests/test_native_conflict.py`
(21 tests) spawns fresh processes and fails if the ordering regresses.

### Dose-review semantics — corrected

The guideline states *"Doses above 100 micrograms … require secondary clinician
review."* That establishes a **review threshold**. It does **not** state a
clinically validated maximum safe dose, and nothing in the supplied document does.

So:

- `config.GUIDELINE_REVIEW_THRESHOLD_MCG = 100.0` — source-derived review trigger.
- `config.ENGINEERING_DOSE_GUARD_MCG` — this project's own software guard, kept
  separate so the two are never conflated. Same number, different kind of claim.
- `DoseBound.clinical_maximum_established` is **always `False`**.
- `SafetyOutcome.clinical_maximum_established` is **always `False`**.
- `result.as_dict()["clinical_maximum_established"]` is **always `False`**.

The **submitted** dose is validated **independently of the LLM** —
`enforce_safety_constraints(..., submitted_dose_mcg=...)` never consults the
model's output. An LLM returning `Eligible_For_Vaccine=True` for a 500 mcg dose
cannot bypass it.

`DoseReviewFinding` preserves the dose **as submitted**, with its unit. Clamping a
recommendation never rewrites the original input.

Boundary behaviour, tested:

| submitted | review required | eligibility |
|---|---|---|
| 100 mcg | no — the source says *above* 100 | unchanged |
| 100.01 mcg | yes | forced `False` |
| 500 mcg | yes | forced `False`, original preserved at 500 |

### Honest failure: no key configured

With no API key the pipeline runs stages 1–4 for real, then fails stage 5:

```
[fail] Pipeline stage 'llm_report' failed: No LLM credential is configured...
==========================================================================
PIPELINE RUN  -  P0001  -  status: FAILED
  (PARTIAL: some stages completed; this is NOT a successful run)
==========================================================================
  [ok     ] preprocessing      0.261s
  [ok     ] ml_prediction      0.141s
  [ok     ] attention          0.011s
  [ok     ] rag_retrieval     16.523s
  [FAIL   ] llm_report         0.000s  No LLM credential is configured...
  [--     ] persist            0.000s  not reached
```

Exit code 1, **no report artifact written**, dashboard stage 05 stays
`not-initialized`. A mocked LLM response is never treated as proof a real call
succeeded.

### What is real vs. mocked

| Check | Real | Mocked |
|---|---|---|
| Preprocessing, split, metrics | ✅ real CSV, real models | — |
| Risk prediction | ✅ real `risk_model.joblib` | — |
| Attention | ✅ real PyTorch forward pass | — |
| Retrieval | ✅ real ChromaDB index, real DOCX | — |
| LLM report | — | ✅ `invoke_llm` patched |

**No end-to-end run has made a real LLM call**, because no API key is configured.
Everything up to and including retrieval is verified against real components.

### Tests

```powershell
python -m pytest tests/test_pipeline.py -v        # 72 passed
python -m pytest tests/test_native_conflict.py -v # 21 passed
python -m pytest                                  # 710 passed, 3 skipped
```

The 3 skips need a live API key and are marked as such. No test requires network
access, a paid token, or credentials.

### Dashboard

Unchanged. `POST /api/run` still returns **501** — the interface is not wired to
the pipeline in this stage. The only dashboard edit is
`components.py` now reading `config.GENERATION_LOG_JSON_PATH` instead of
re-spelling that path, so the writer and reader name one constant.

---

## 12. Module 6 — Dashboard backend integration

Implemented. `POST /api/run` executes the real pipeline. The frontend is
unchanged in design: same six pages, same design system, same components.

### Starting it

```powershell
python -m dashboard.app                    # opens http://127.0.0.1:8000
python -m dashboard.app --no-browser --port 8077
```

Verified: **HTTP 200** on `/`, `/api/health`, `/api/state`, `/static/*`,
`/api/reports`, `/api/runs`. Stages ready 4 of 5 (report awaits a real run).

### Endpoint contract — `POST /api/run`

Request — the shape `app.js` already sends, unchanged:

```json
{"stage": "ml",     "record": {"Patient_ID": "P0001", "Age": 24,
                              "Biomarker_Level": 4.53,
                              "Pre_Existing_Conditions": 1,
                              "Vaccine_Dose_mcg": 83.5,
                              "Prior_Reaction_History": "Mild"}}
{"stage": "rag",    "question": "What are the storage requirements?"}
{"stage": "report"}
{"stage": "all",    "record": { ... }}
```

Response, abbreviated from a real run:

```json
{
  "run_id": "20261015T141836Z-21d7ddfa",
  "stage": "ml", "status": "completed",
  "patient_id": "P0001",
  "risk_label": "Low",
  "risk_probability": 0.331985,
  "selected_model": "Logistic Regression",
  "stage_statuses": {"preprocessing": "completed", "ml_prediction": "completed"},
  "stage_details": [{"stage": "...", "seconds": 0.1, "status": "completed"}],
  "timings": {"preprocessing": 0.134, "ml_prediction": 0.095},
  "disclaimer": "This is a student engineering project and is not intended to
                 provide real clinical guidance."
}
```

`stage: "all"` additionally returns `retrieved_chunks`, `retrieval_queries`,
`attention` (shape, row-sum bounds, `softmax_verified`), `report`, `artifacts`,
and `dose`.

| Code | Meaning |
|---|---|
| 200 | A required stage completed. The body carries the real result. |
| 400 | Malformed request. **No stage ran.** Names the offending field. |
| 409 | A run is already executing. Refused, not queued. |
| 413 | Body over 16 KB. |
| 500 | A required stage failed, or an unexpected fault. Traceback stays in the server log. |
| 502 | The LLM stage failed for an external reason. |
| 503 | A required artifact or credential is missing. |

**200 is only ever returned when the pipeline itself reports `completed`.**

Verified against real local components:

| Request | Status | Evidence |
|---|---|---|
| `rag` | **200** | 3 real chunks from ChromaDB, 0.018 s, cited to the DOCX |
| `ml` | **200** | Low, p=0.331985, Logistic Regression |
| `all` (no key) | **500** | `failed_stage: llm_report`, partial, no report |
| bad stage | 400 | `invalid_request` |
| `Age: -1` | 400 | `'Age' must be between 0 and 130; got -1.` |
| `/api/report/../../.env` | 404 | traversal refused |

### Execution and state

**Synchronous, with a lock.** The server is `ThreadingHTTPServer`, so a run holds
one worker thread and `/api/state`, `/api/health` and the static routes stay
responsive. A background queue was rejected as unnecessary infrastructure for one
user-initiated action.

A `threading.Lock` serialises runs so two cannot write the ChromaDB collection,
model directory or latency log concurrently. A second request while one is in
flight gets **409, not a queued duplicate** — a double-click cannot double-write.
Validation runs *before* the lock, so a malformed request is rejected immediately
rather than queued behind a run.

Every response carries a `run_id` (`<UTC timestamp>-<uuid8>`), and `GET /api/runs`
returns recent runs. This matters because report filenames are timestamped rather
than keyed, so without it a client could not tell which file its run produced.
The lock is released on failure — a crash does not wedge the endpoint.

### Report artifacts

`GET /api/reports` lists files (names, sizes, mtimes — **no contents**).
`GET /api/report/<filename>` serves one.

Two independent checks, because either alone is insufficient: the name must be a
bare filename with no separators, *and* the resolved path must still be inside
`config.REPORTS_DIR` after symlink resolution. Only `.json` and `.md` are served,
so a stray `.py` or `.csv` in the directory is not exposed. `resolve_report_artifact`
is unit-tested against 9 traversal forms plus absolute paths.

### Dose reporting

The response preserves the **submitted** dose and unit unchanged, separately from:

- `review_threshold_mcg: 100.0`, `review_threshold_kind: "secondary_clinician_review"`
- `engineering_dose_guard_mcg` — the project's own software guard, kept separate
- `clinical_maximum_established: false` — **always**

No response *key* asserts a clinical maximum; a test asserts this, because a field
name outlives the prose that qualified it. The one `clinical_maximum_*` key is a
negative that must be `false`.

Boundary, verified live: **100 mcg → not required** (the source says *above* 100),
**100.01 → required**, **500 → required**. `review_required` also survives a run
that died before enforcement (e.g. the credential failure arrives first), so a
flagged dose is never under-reported.

### Native import order — a regression I introduced and caught

`dashboard/runs.py` originally called `from src.ml import predict` **before**
`_pipeline_module()`. `src.ml.predict` pulls scikit-learn in transitively, so
torch's `c10.dll` then failed with `WinError 1114` — the exact crash Module 5
fixed. It was invisible in an already-running process.

Fixed by importing the pipeline first, and guarded by three tests:

- importing `dashboard.app` must **not** load sklearn
- the dashboard-then-pipeline sequence must complete in a fresh process
- `run_ml`'s source must call `_pipeline_module()` before any sklearn-reaching import

No `KMP_DUPLICATE_LIB_OK`. No subprocess isolation.

### Visualizations

Unchanged and still reading saved artifacts — Panel A from `roc_curve.json`, Panel B
from the attention weights, Panel C from the latency logs. 150 DPI floor
unchanged. No substitute plot is generated for an empty panel.

### Security

- Only the six declared record fields reach the pipeline; unknown keys, path-like
  keys and config-like keys are dropped (verified)
- Numeric ranges, lengths, finiteness and the categorical vocabulary are enforced
- No client-supplied paths; outputs come from `config.REPORTS_DIR`
- 500 responses carry a class name only — no traceback, no path, no credential
- Run history stores no secrets
- Logs print the patient ID and stage timings, not the full record

### Honest limits

- **No live LLM call has been made.** No API key configured. The `all` stage is
  verified with `invoke_llm` mocked, and returns 500 honestly without a key.
- **Attention sequence is the PRD sample**, `ACGUACGUACGGCUA`, labelled
  `sequence_source: "config_default"`. No patient sequence exists in the dataset.
- **`outputs/reports/` is empty.** No fabricated report was committed as evidence.
- `HF_HUB_OFFLINE=1` avoids a transient Hugging Face connection failure seen once
  during testing.

---

## 13. Running the project

### Training (Module 1)

```powershell
python -m dashboard.app                   # serve on http://127.0.0.1:8000
python -m dashboard.app --no-browser      # serve without opening a browser
python -m dashboard.app --port 9000       # choose a port
python -m dashboard.app --figure          # render outputs/figures/dashboard.png
python -m dashboard.app --verify          # check the figure against FR-5.1 / FR-5.3
python -m dashboard.app --state           # print the platform state as JSON
```

The interface runs on the Python standard library alone — no web framework, no build
step, no `npm install`. `dashboard/static/` holds three plain files served as they are.

### Attention (Module 2)

```powershell
python -m src.attention                   # nucleotide demo, writes all Module 2 artifacts
python -m src.attention --feature         # attend over a processed patient record
```

Runs independently — no RAG, no LLM, no dashboard required.

### Regulatory RAG (Module 3)

```powershell
python -m src.rag.ingest --reset             # extract, chunk, embed, persist to ChromaDB
python -m src.rag.retrieve --query "What are the storage requirements?"
python -m src.rag.retrieve --dose 120        # dose-specific regulatory evidence
python -m src.rag.retrieve --evaluate        # 8-query set, writes the latency log
```

Runs entirely locally — no API key, no LLM, no dashboard required.

### Training (Module 1)

```powershell
python -m src.ml                           # load, inspect, validate, clean, split, train, evaluate
python -m src.ml --quiet                   # same, without staged progress output
```

Runs the complete ML stage in one command: load, inspect, validate, clean, split,
fit preprocessing on train only, train both models, evaluate, generate the ROC
comparison and both confusion matrices, save the model artifacts and the evaluation
results, and print a summary. Does **not** require the dashboard to be running.

Inference is a library call rather than a command:

Inference is a library call rather than a command:

```python
from src.ml import predict

result = predict.predict_patient({
    "Patient_ID": "P0001", "Age": 58, "Biomarker_Level": 11.2,
    "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0,
    "Prior_Reaction_History": "Mild",
})
result.as_dict()
# {'patient_id': 'P0001', 'selected_model': 'logistic_regression',
#  'agreement': False,
#  'logistic_regression': {'prediction': 1, 'risk_label': 'High', 'probability': 0.976099},
#  'random_forest': {'prediction': 1, 'risk_label': 'High', 'probability': 0.91}}
```

With no arguments, models are loaded from `models/`.

Later stages, once implemented:

```powershell
python -m src.pipeline.pipeline --index    # Module 3: extract, chunk, embed, persist
python -m src.pipeline.pipeline --run      # full flow for one patient
```

```powershell
pytest                                     # run the test suite
```

`--patient-id` selects a row of the raw CSV (defaults to the first record).
`--sequence` overrides the mRNA sequence attributed to the patient; the provided dataset
has no sequence column, so the default comes from a seeded generator to keep runs
reproducible.

---

## 14. Source documents

| File | Role |
|---|---|
| `data/raw/patient_records.csv` | Supplied dataset, 500 records. Unmodified. |
| `guidelines/Moderna_mRNA_Safety_Guidelines.docx` | Supplied regulatory source for the RAG index. Unmodified — opened read-only. |

The provided dataset matches the PRD Section 5.2 specification exactly: 500 rows,
`Patient_ID` `P0001`–`P0500`, ages 18–85, doses 25–150 mcg, missing values in
`Biomarker_Level` and `Prior_Reaction_History`, and a near-balanced `Reaction_Score`.
`tests/test_preprocessing.py::TestDatasetContract` asserts all of this, so the dataset is
verified rather than assumed.

---

## 15. Security

* No API key, token or credential appears anywhere in this repository.
* Secrets are read from environment variables inside the function that needs them.
* `.env` is gitignored; `.env.example` contains placeholders only.
* The guideline source document is never written to by any code path.
* Patient data in this project contains no real personally identifiable information
  (PRD Section 1.6).
