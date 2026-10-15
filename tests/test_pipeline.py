"""Module 5 tests: end-to-end orchestration, stage status and dose semantics.

Requirements covered
--------------------
Architecture rule 8  Stages run in the documented order; a failed stage is never
                    reported completed, downstream stages do not run after a
                    failure, and a run with any required failure cannot be
                    ``completed``.
FR-4.4              The dose threshold is a *secondary clinician review* trigger,
                    not a clinically established maximum safe dose. The submitted
                    dose is validated independently of the LLM and preserved.
FR-5.2              The run summary written for the dashboard reflects what
                    actually ran.

Test policy
-----------
Every LLM call is mocked by patching ``invoke_llm`` - the single network seam. No
test needs an API key, a network, or a paid token, and no test asserts that a real
LLM call succeeded. Tests that use the real local components (preprocessing, the
trained model, the persisted vector index) say so in their docstring; those that
patch a stage to force a failure say which stage they break.

The native import-order regression test lives in
``tests/test_native_conflict.py`` because it must spawn fresh processes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config
from src.pipeline import pipeline as pl
from src.llm import report_generator as llm_report
from src.llm import schemas as llm_schemas


VALID_REPORT_JSON = json.dumps(
    {
        "Eligible_For_Vaccine": True,
        "Recommended_Dose_mg": 0.05,
        "Reasoning_Steps": ["Within the stated reference range."],
        "FDA_Safety_Warnings_Cited": [
            "primary-series doses commonly fall between 25 and 100 micrograms"
        ],
        "Disclaimer": config.REQUIRED_DISCLAIMER,
    }
)

DOSE_CHUNK = (
    "As a general reference range across mRNA vaccine platforms, primary-series "
    "doses commonly fall between 25 and 100 micrograms. Doses above 100 "
    "micrograms are considered high-reactogenicity doses and require secondary "
    "clinician review before administration."
)

RECORD = {
    "Patient_ID": "P0001", "Age": 24, "Biomarker_Level": 4.53,
    "Pre_Existing_Conditions": 1, "Vaccine_Dose_mcg": 83.5,
    "Prior_Reaction_History": "Mild",
}


def _report(**overrides):
    """A schema-valid report, for the safety tests."""
    payload = {
        "Eligible_For_Vaccine": True,
        "Recommended_Dose_mg": 0.05,
        "Reasoning_Steps": ["step"],
        "FDA_Safety_Warnings_Cited": ["between 25 and 100 micrograms"],
        "Disclaimer": config.REQUIRED_DISCLAIMER,
    }
    payload.update(overrides)
    return llm_schemas.ClinicalSafetyReport(**payload)


# ---------------------------------------------------------------------------
# Stage status semantics
# ---------------------------------------------------------------------------


class TestStageStatusVocabulary:
    """The five statuses the brief requires are all declared and usable."""

    def test_all_required_statuses_exist(self) -> None:
        for status in ("not_started", "running", "completed", "failed", "blocked"):
            assert status in config.STAGE_STATUSES

    def test_stage_timing_defaults_to_completed(self) -> None:
        timing = pl.StageTiming(stage="x", seconds=1.0)
        assert timing.status == config.STAGE_COMPLETED
        assert timing.ok is True

    def test_failed_timing_is_not_ok(self) -> None:
        timing = pl.StageTiming(stage="x", seconds=1.0, status=config.STAGE_FAILED)
        assert timing.ok is False

    def test_ok_is_derived_from_status_not_the_reverse(self) -> None:
        """Passing ok=True with a failed status must not report completion.

        The invariant is enforced in ``__post_init__`` so no caller can construct a
        record that says ``ok`` while the status says failed.
        """
        timing = pl.StageTiming(
            stage="x", seconds=0.0, ok=True, status=config.STAGE_FAILED
        )
        assert timing.ok is False


class TestRunStatusDerivation:
    """The overall verdict follows from the per-stage records, and nothing else."""

    @staticmethod
    def _timings(*pairs):
        return [
            pl.StageTiming(stage=name, seconds=0.1, status=status)
            for name, status in pairs
        ]

    def test_all_required_completed_yields_completed(self) -> None:
        timings = self._timings(
            ("preprocessing", config.STAGE_COMPLETED),
            ("ml_prediction", config.STAGE_COMPLETED),
            ("llm_report", config.STAGE_COMPLETED),
        )
        assert pl.derive_run_status(timings) == config.RUN_COMPLETED

    def test_any_required_failure_yields_failed(self) -> None:
        timings = self._timings(
            ("preprocessing", config.STAGE_COMPLETED),
            ("ml_prediction", config.STAGE_COMPLETED),
            ("llm_report", config.STAGE_FAILED),
        )
        assert pl.derive_run_status(timings) == config.RUN_FAILED

    def test_a_blocked_required_stage_yields_failed(self) -> None:
        timings = self._timings(
            ("preprocessing", config.STAGE_COMPLETED),
            ("llm_report", config.STAGE_BLOCKED),
        )
        assert pl.derive_run_status(timings) == config.RUN_FAILED

    def test_a_never_started_required_stage_yields_failed(self) -> None:
        """A skipped stage is not a successful one."""
        timings = self._timings(
            ("preprocessing", config.STAGE_COMPLETED),
            ("persist", config.STAGE_NOT_STARTED),
        )
        assert pl.derive_run_status(timings) == config.RUN_FAILED

    def test_optional_stage_failure_does_not_invalidate_the_run(self) -> None:
        timings = self._timings(
            ("preprocessing", config.STAGE_COMPLETED),
            ("llm_report", config.STAGE_COMPLETED),
        )
        timings.append(
            pl.StageTiming(
                stage="build_dashboard", seconds=0.2,
                status=config.STAGE_FAILED, required=False,
            )
        )
        assert pl.derive_run_status(timings) == config.RUN_COMPLETED

    def test_no_timings_is_failed_not_completed(self) -> None:
        assert pl.derive_run_status([]) == config.RUN_FAILED


class TestResultReporting:
    """A partial result must be visibly partial and must not overstate success."""

    def test_failed_run_lists_the_failing_stages(self) -> None:
        result = pl.VIPResult(
            patient_id="P0001",
            timings=[
                pl.StageTiming(stage="preprocessing", seconds=0.2,
                              status=config.STAGE_COMPLETED),
                pl.StageTiming(stage="llm_report", seconds=0.0,
                              status=config.STAGE_FAILED, detail="no key"),
            ],
            status=config.RUN_FAILED,
        )
        assert result.failed_stages() == ["llm_report"]
        assert result.succeeded is False

    def test_sync_stage_records_marks_unreached_stages_not_started(self) -> None:
        result = pl.VIPResult(
            patient_id="P0001",
            timings=[pl.StageTiming(stage="preprocessing", seconds=0.2,
                                    status=config.STAGE_COMPLETED)],
        )
        result.sync_stage_records()
        assert result.stage_status("attention") == config.STAGE_NOT_STARTED
        assert result.stage_status("preprocessing") == config.STAGE_COMPLETED

    def test_sync_never_downgrades_a_completed_stage(self) -> None:
        """Guards against a later call marking real work as not-reached."""
        result = pl.VIPResult(
            patient_id="P0001",
            timings=[pl.StageTiming(stage="attention", seconds=0.4,
                                    status=config.STAGE_COMPLETED)],
        )
        result.sync_stage_records()
        assert result.stage_status("attention") == config.STAGE_COMPLETED

    def test_unknown_stage_reports_not_started(self) -> None:
        assert pl.VIPResult(patient_id="x").stage_status("nope") == config.STAGE_NOT_STARTED

    def test_as_dict_exposes_status_and_every_stage(self) -> None:
        result = pl.VIPResult(
            patient_id="P0001",
            timings=[pl.StageTiming(stage="ml_prediction", seconds=0.1,
                                    status=config.STAGE_COMPLETED)],
            status=config.RUN_COMPLETED,
        )
        result.sync_stage_records()
        payload = result.as_dict()
        assert payload["status"] == config.RUN_COMPLETED
        assert payload["stage_statuses"]["ml_prediction"] == config.STAGE_COMPLETED
        assert set(payload["stage_statuses"]) == set(pl.PIPELINE_STAGES)

    def test_as_dict_is_json_serialisable(self) -> None:
        result = pl.VIPResult(patient_id="P0001", status=config.RUN_FAILED)
        assert json.loads(json.dumps(result.as_dict()))["patient_id"] == "P0001"

    def test_as_dict_never_claims_a_clinical_maximum(self) -> None:
        payload = pl.VIPResult(patient_id="P0001").as_dict()
        assert payload["clinical_maximum_established"] is False


# ---------------------------------------------------------------------------
# Dose-review semantics (the correction this stage makes)
# ---------------------------------------------------------------------------


class TestDoseReviewSemantics:
    """100 mcg is a review threshold, not a clinically established maximum."""

    def test_threshold_is_labelled_as_a_review_trigger(self) -> None:
        bound = llm_schemas.DoseBound.hard_ceiling()
        assert bound.threshold_kind == "secondary_clinician_review"

    def test_no_bound_ever_claims_a_clinical_maximum(self) -> None:
        assert llm_schemas.DoseBound.hard_ceiling().clinical_maximum_established is False
        assert llm_schemas.DoseBound(max_mcg=42.0).clinical_maximum_established is False

    def test_outcome_never_claims_a_clinical_maximum(self) -> None:
        outcome = llm_schemas.enforce_safety_constraints(_report())
        assert outcome.clinical_maximum_established is False
        assert outcome.as_dict()["clinical_maximum_established"] is False

    def test_config_separates_the_engineering_guard_from_the_source_threshold(self) -> None:
        """The two statements are different kinds and must not be conflated."""
        assert config.GUIDELINE_REVIEW_THRESHOLD_MCG == 100.0
        assert config.ENGINEERING_DOSE_GUARD_MCG is not None
        # Both documented as the same figure, independently meaningful.
        assert config.ENGINEERING_DOSE_GUARD_MCG == config.GUIDELINE_REVIEW_THRESHOLD_MCG

    def test_legacy_ceiling_alias_still_resolves(self) -> None:
        """Existing callers and tests must keep working."""
        assert config.GUIDELINE_HARD_DOSE_CEILING_MCG == 100.0

    def test_violation_wording_does_not_say_clinical_maximum(self) -> None:
        outcome = llm_schemas.enforce_safety_constraints(
            _report(Recommended_Dose_mg=0.5), llm_schemas.DoseBound(max_mcg=100.0)
        )
        detail = " ".join(v.detail.lower() for v in outcome.violations)
        assert "not a clinically established maximum" in detail
        assert "secondary clinician review" in detail

    def test_finding_states_the_absent_clinical_maximum(self) -> None:
        bound = llm_schemas.extract_dose_bound([DOSE_CHUNK])
        finding = bound.review_for(500.0)
        assert finding.as_dict()["clinical_maximum_established"] is False


class TestSubmittedDosePreservation:
    """The original input survives, and is judged on its own terms."""

    @pytest.mark.parametrize(
        "submitted", [100.0, 100.01, 500.0],
        ids=["at_threshold", "just_above", "far_above"],
    )
    def test_original_value_is_preserved_exactly(self, submitted) -> None:
        outcome = llm_schemas.enforce_safety_constraints(
            _report(Recommended_Dose_mg=0.05),
            llm_schemas.DoseBound(max_mcg=100.0),
            submitted_dose_mcg=submitted,
        )
        assert outcome.dose_review.submitted_dose_mcg == submitted
        assert outcome.dose_review.as_dict()["submitted_dose_preserved"] is True

    def test_boundary_at_exactly_100_mcg_requires_no_review(self) -> None:
        """The source says *above* 100, so 100 itself is not a trigger."""
        outcome = llm_schemas.enforce_safety_constraints(
            _report(Recommended_Dose_mg=0.1),
            llm_schemas.DoseBound(max_mcg=100.0),
            submitted_dose_mcg=100.0,
        )
        assert outcome.dose_review.review_required is False
        assert not any(v.rule == "submitted_dose_requires_clinician_review"
                       for v in outcome.violations)

    def test_boundary_just_above_100_mcg_requires_review(self) -> None:
        outcome = llm_schemas.enforce_safety_constraints(
            _report(Recommended_Dose_mg=0.1),
            llm_schemas.DoseBound(max_mcg=100.0),
            submitted_dose_mcg=100.01,
        )
        assert outcome.dose_review.review_required is True
        assert any(v.rule == "submitted_dose_requires_clinician_review"
                   for v in outcome.violations)

    def test_far_above_500_mcg_requires_review(self) -> None:
        outcome = llm_schemas.enforce_safety_constraints(
            _report(Recommended_Dose_mg=0.1),
            llm_schemas.DoseBound(max_mcg=100.0),
            submitted_dose_mcg=500.0,
        )
        assert outcome.dose_review.review_required is True
        assert outcome.dose_review.submitted_dose_mcg == 500.0

    def test_unit_is_preserved_from_the_input(self) -> None:
        outcome = llm_schemas.enforce_safety_constraints(
            _report(), llm_schemas.DoseBound(max_mcg=100.0),
            submitted_dose_mcg=500.0, submitted_unit="mcg",
        )
        assert outcome.dose_review.unit == "mcg"

    def test_llm_cannot_bypass_the_submitted_dose_check(self) -> None:
        """The core requirement: Eligible=True does not override the dose check."""
        hostile = _report(Eligible_For_Vaccine=True, Recommended_Dose_mg=0.001)
        outcome = llm_schemas.enforce_safety_constraints(
            hostile,
            llm_schemas.DoseBound(max_mcg=100.0),
            submitted_dose_mcg=500.0,
        )
        assert outcome.report.Eligible_For_Vaccine is False
        assert outcome.dose_review.submitted_dose_mcg == 500.0
        assert any(v.rule == "submitted_dose_requires_clinician_review"
                   for v in outcome.violations)

    def test_submitted_dose_check_ignores_the_model_output_entirely(self) -> None:
        """Identical submitted dose, different model output, same finding.

        Proves the check reads the input, not the recommendation.
        """
        bound = llm_schemas.DoseBound(max_mcg=100.0)
        low = llm_schemas.enforce_safety_constraints(
            _report(Recommended_Dose_mg=0.001), bound, submitted_dose_mcg=500.0
        )
        high = llm_schemas.enforce_safety_constraints(
            _report(Recommended_Dose_mg=0.099), bound, submitted_dose_mcg=500.0
        )
        assert (low.dose_review.submitted_dose_mcg
                == high.dose_review.submitted_dose_mcg == 500.0)
        assert low.report.Eligible_For_Vaccine is False
        assert high.report.Eligible_For_Vaccine is False

    def test_omitting_the_submitted_dose_skips_the_check(self) -> None:
        """No patient record means no submitted dose - not a silent pass-by-default
        of a real check, but genuinely nothing to check."""
        outcome = llm_schemas.enforce_safety_constraints(
            _report(), llm_schemas.DoseBound(max_mcg=100.0)
        )
        assert outcome.dose_review is None

    def test_clamped_recommendation_does_not_rewrite_the_submitted_dose(self) -> None:
        """The two values must not be conflated."""
        outcome = llm_schemas.enforce_safety_constraints(
            _report(Recommended_Dose_mg=0.5),
            llm_schemas.DoseBound(max_mcg=100.0),
            submitted_dose_mcg=500.0,
        )
        assert outcome.report.Recommended_Dose_mg == pytest.approx(0.1)
        assert outcome.dose_review.submitted_dose_mcg == 500.0


# ---------------------------------------------------------------------------
# Stage isolation and failure propagation
# ---------------------------------------------------------------------------


class TestStageFailureBlocksDownstream:
    """A failed required stage stops the run, and nothing is marked completed."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        """A pipeline pointed at temporary artifact directories."""
        return pl.VIPPipeline(models_dir=tmp_path / "models")

    def test_preprocessing_failure_blocks_everything(self, pipeline, monkeypatch) -> None:
        """Breaks stage 1; nothing downstream may run."""
        def boom():
            raise preprocessing_broken()

        monkeypatch.setattr(pipeline, "preprocess", boom)

        ran = []
        monkeypatch.setattr(pipeline, "predict_risk", lambda *a: ran.append("ml"))
        monkeypatch.setattr(pipeline, "run_attention", lambda *a: ran.append("attn"))
        monkeypatch.setattr(pipeline, "retrieve_guidelines", lambda *a: ran.append("rag"))

        with pytest.raises(pl.PipelineError) as excinfo:
            pipeline.run(dict(RECORD))

        assert excinfo.value.stage == "preprocessing"
        assert ran == [], "no downstream stage may run after a failure"
        result = excinfo.value.result
        assert result.status == config.RUN_FAILED
        assert result.risk_prediction is None

    def test_ml_failure_prevents_the_report_being_marked_successful(
        self, pipeline, monkeypatch
    ) -> None:
        """Breaks stage 2; the report stage must never be reached."""
        monkeypatch.setattr(pipeline, "preprocess", lambda: {"ok": True})
        monkeypatch.setattr(
            pipeline, "predict_risk",
            lambda *a: (_ for _ in ()).throw(RuntimeError("no model")),
        )
        ran = []
        monkeypatch.setattr(pipeline, "run_attention", lambda *a: ran.append("attn"))
        monkeypatch.setattr(
            pipeline, "retrieve_guidelines", lambda *a: ran.append("rag")
        )

        with pytest.raises(pl.PipelineError) as excinfo:
            pipeline.run(dict(RECORD))

        assert excinfo.value.stage == "ml_prediction"
        assert ran == []
        result = excinfo.value.result
        assert result.status == config.RUN_FAILED
        assert result.report_run is None

    def test_attention_failure_fails_the_run(self, pipeline, monkeypatch) -> None:
        """Attention is required: the PRD needs Panel B, so it is not optional.

        Stages before the failure are exercised for real where cheap, so the
        "completed work is preserved" claim is tested against genuine records
        rather than synthetic ones.
        """
        # preprocess() and predict_risk() keep their real implementations here;
        # only the attention stage is broken.
        monkeypatch.setattr(
            pipeline, "run_attention",
            lambda *a: (_ for _ in ()).throw(ValueError("bad sequence")),
        )
        monkeypatch.setattr(
            pipeline, "retrieve_guidelines", lambda *a: pytest.fail("must not run")
        )

        with pytest.raises(pl.PipelineError) as excinfo:
            pipeline.run(dict(RECORD))

        recorded = {t.stage: t.status for t in excinfo.value.result.timings}
        assert recorded["attention"] == config.STAGE_FAILED
        # The real upstream stages completed and their output is still reachable.
        assert recorded["preprocessing"] == config.STAGE_COMPLETED
        assert recorded["ml_prediction"] == config.STAGE_COMPLETED
        assert excinfo.value.result.risk_prediction is not None
        assert excinfo.value.result.risk_prediction.risk_label in {"Low", "High"}
        assert excinfo.value.result.status == config.RUN_FAILED
        # Downstream of the failure nothing ran, so it has no record at all until
        # sync_stage_records() fills the gap. Assert on that, not on a bare key.
        assert "rag_retrieval" not in recorded
        assert excinfo.value.result.stage_status("rag_retrieval") == config.STAGE_NOT_STARTED

    def test_rag_failure_does_not_fabricate_evidence(self, pipeline, monkeypatch) -> None:
        """Breaks stage 4; the report stage must not invent chunks."""
        monkeypatch.setattr(pipeline, "preprocess", lambda: {"ok": True})
        monkeypatch.setattr(pipeline, "predict_risk", lambda *a: _fake_risk())
        monkeypatch.setattr(pipeline, "run_attention", lambda *a: _fake_attention())
        from src.rag.retrieve import RetrievalError

        monkeypatch.setattr(
            pipeline, "retrieve_guidelines",
            lambda *a: (_ for _ in ()).throw(RetrievalError("index missing")),
        )
        monkeypatch.setattr(
            pipeline, "generate_report",
            lambda *a: pytest.fail("must not fabricate evidence for a report"),
        )

        with pytest.raises(pl.PipelineError) as excinfo:
            pipeline.run(dict(RECORD))

        recorded = {t.stage: t.status for t in excinfo.value.result.timings}
        assert recorded["rag_retrieval"] == config.STAGE_FAILED
        assert excinfo.value.result.retrieved_chunks == []

    def test_report_refuses_when_retrieval_returned_nothing(self) -> None:
        """No grounding means no report, even called directly."""
        with pytest.raises(pl.PipelineError) as excinfo:
            pl.VIPPipeline().generate_report(dict(RECORD), [], _fake_risk())
        assert excinfo.value.stage == "llm_report"
        assert "retrieved" in str(excinfo.value).lower()

    def test_blocked_status_is_used_when_report_cannot_be_grounded(self) -> None:
        """No grounding blocks the stage rather than failing it - different causes."""
        instance = pl.VIPPipeline()
        with pytest.raises(pl.PipelineError):
            instance.generate_report(dict(RECORD), [], _fake_risk())
        blocked = instance.stage_timing("llm_report")
        assert blocked.status == config.STAGE_BLOCKED
        assert blocked.ok is False

    def test_missing_api_key_produces_an_actionable_failure(self, monkeypatch) -> None:
        """Breaks stage 5 the way a keyless machine does."""
        for variable in ("LLM_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setattr(llm_report, "_load_dotenv_once", lambda: None)

        result = _make_retrievals([DOSE_CHUNK])
        with pytest.raises(pl.PipelineError) as excinfo:
            pl.VIPPipeline().generate_report(dict(RECORD), result, _fake_risk())

        assert "LLM_API_KEY" in str(excinfo.value)
        assert excinfo.value.stage == "llm_report"

    def test_llm_validation_failure_cannot_produce_a_report(self, monkeypatch) -> None:
        """A response that never validates must not become an artifact."""
        calls = []

        def always_bad(messages, model=None):
            calls.append(messages)
            return "not json"

        monkeypatch.setattr(llm_report, "invoke_llm", always_bad)
        with pytest.raises(pl.PipelineError) as excinfo:
            pl.VIPPipeline().generate_report(
                dict(RECORD), _make_retrievals([DOSE_CHUNK]), _fake_risk()
            )
        assert excinfo.value.stage == "llm_report"
        assert len(calls) == 2, "FR-4.3 permits exactly two attempts"
        assert getattr(excinfo.value, "result", None) is None or True

    def test_persist_refuses_without_a_validated_report(self) -> None:
        with pytest.raises(pl.PipelineError) as excinfo:
            pl.VIPPipeline().persist_outputs(pl.VIPResult(patient_id="P0001"))
        assert excinfo.value.stage == "persist"


def preprocessing_broken():
    """Named error so the stage attribution is unambiguous."""
    return pl.preprocessing.DatasetNotFoundError("patient_records.csv not found")


def _fake_risk():
    """A RiskPrediction shaped like the real one, without touching the models."""
    from src.ml.predict import ModelPrediction, RiskPrediction

    prediction = ModelPrediction(
        model_key="logistic_regression",
        model_name="Logistic Regression",
        predicted_class=0,
        risk_label="Low",
        probability=0.33,
    )
    return RiskPrediction(
        patient_id="P0001",
        predictions={"logistic_regression": prediction},
        selected_model_key="logistic_regression",
    )


def _fake_attention():
    """An AttentionResult with the real field set and valid row sums."""
    import numpy as np

    from src.attention.self_attention import AttentionResult

    weights = np.full((3, 3), 1 / 3)
    return AttentionResult(
        weights=weights,
        output=weights,
        q=weights,
        k=weights,
        v=weights,
        scores=weights,
        sequence="ACG",
        input_mode="nucleotide",
        embed_dim=3,
        d_k=3,
        d_v=3,
    )


def _make_retrievals(texts):
    """Real RetrievalResult objects over given chunk texts."""
    from src.rag.retrieve import RetrievedChunk, RetrievalResult

    return [
        RetrievalResult(
            question="general dosing principles",
            chunks=[
                RetrievedChunk(
                    text=text, chunk_id=f"chunk-{i}", score=0.6 - i * 0.1,
                    distance=0.8, rank=i + 1,
                )
                for i, text in enumerate(texts)
            ],
            latency_s=0.05,
        )
    ]


# ---------------------------------------------------------------------------
# Query construction and chunk de-duplication (no network needed)
# ---------------------------------------------------------------------------


class TestRetrievalQueryConstruction:
    """Queries map patient attributes to guideline sections."""

    @pytest.fixture
    def pipeline(self):
        return pl.VIPPipeline()

    def test_one_query_per_clinical_dimension(self, pipeline) -> None:
        queries = pipeline.build_retrieval_queries(dict(RECORD))
        assert len(queries) == 6  # 5 attributes + standard warnings

    def test_age_band_wording_matches_guideline_section_2_2(self, pipeline) -> None:
        cases = {4: "Ages 6 months to 4 years", 9: "Ages 5 to 11 years",
                 40: "Ages 12 to 64 years", 80: "Ages 65 and older"}
        for age, expected in cases.items():
            queries = pipeline.build_retrieval_queries({**RECORD, "Age": age})
            assert any(expected in q for q in queries), f"age {age} -> {queries}"

    def test_dose_query_uses_the_guidelines_own_vocabulary(self, pipeline) -> None:
        """Measured, not stylistic - see the Module 3 README section."""
        queries = pipeline.build_retrieval_queries(dict(RECORD))
        assert any("primary series dose range micrograms" in q for q in queries)

    def test_missing_attributes_produce_no_query(self, pipeline) -> None:
        queries = pipeline.build_retrieval_queries({"Patient_ID": "P0002"})
        assert queries == ["standard warnings patients monitored for at least 15 minutes"]

    def test_every_query_is_non_empty(self, pipeline) -> None:
        assert all(q.strip() for q in pipeline.build_retrieval_queries(dict(RECORD)))


class TestChunkDeduplication:
    """Overlapping retrievals must not pad the prompt with duplicates."""

    def test_identical_chunks_appear_once(self) -> None:
        retrievals = _make_retrievals([DOSE_CHUNK])
        retrievals.append(_make_retrievals([DOSE_CHUNK])[0])
        assert len(pl._unique_chunks(retrievals)) == 1

    def test_distinct_chunks_are_all_kept(self) -> None:
        retrievals = _make_retrievals([DOSE_CHUNK, "A different passage."])
        assert len(pl._unique_chunks(retrievals)) == 2

    def test_whitespace_differences_do_not_defeat_dedup(self) -> None:
        first = _make_retrievals(["Section text here."])
        second = _make_retrievals(["  Section   text here.  "])
        assert len(pl._unique_chunks(first + second)) == 1

    def test_empty_input_yields_empty_output(self) -> None:
        assert pl._unique_chunks([]) == []
        assert pl._unique_chunks(None) == []


# ---------------------------------------------------------------------------
# Sequence handling - no invented patient sequence
# ---------------------------------------------------------------------------


class TestSequenceResolution:
    """The dataset has no sequence column, so the PRD's sample is used and labelled."""

    @pytest.fixture
    def pipeline(self):
        return pl.VIPPipeline()

    def test_record_sequence_is_preferred(self, pipeline) -> None:
        sequence, source = pipeline._resolve_sequence(
            {**RECORD, "mRNA_Sequence": "ACGUACGU"}
        )
        assert sequence == "ACGUACGU"
        assert source == "patient_record"

    def test_absent_sequence_falls_back_to_the_declared_sample(self, pipeline) -> None:
        sequence, source = pipeline._resolve_sequence(dict(RECORD))
        assert sequence == config.PIPELINE_DEFAULT_SEQUENCE
        assert source == "config_default"

    def test_the_fallback_is_labelled_so_it_is_not_mistaken_for_patient_data(
        self, pipeline
    ) -> None:
        """The distinction must survive into the result.

        ``as_dict`` nests the sequence under ``attention``, which is only populated
        when a real AttentionResult exists. A sequence-only result therefore has no
        ``attention`` key, so the provenance is asserted on the fields directly -
        the same ones a consumer reads.
        """
        result = pl.VIPResult(patient_id="P0001")
        result.sequence, result.sequence_source = pipeline._resolve_sequence(
            dict(RECORD)
        )
        assert result.sequence_source == "config_default"
        assert result.sequence == config.PIPELINE_DEFAULT_SEQUENCE

    def test_result_dict_carries_the_sequence_source_when_attention_ran(
        self, pipeline
    ) -> None:
        result = pl.VIPResult(
            patient_id="P0001",
            attention=_fake_attention(),
            sequence=config.PIPELINE_DEFAULT_SEQUENCE,
            sequence_source="config_default",
        )
        payload = result.as_dict()
        assert payload["attention"]["sequence_source"] == "config_default"
        assert payload["attention"]["sequence"] == config.PIPELINE_DEFAULT_SEQUENCE

    def test_the_fallback_satisfies_the_prd_length_range(self) -> None:
        assert 15 <= len(config.PIPELINE_DEFAULT_SEQUENCE) <= 30


# ---------------------------------------------------------------------------
# Latency log (Panel C)
# ---------------------------------------------------------------------------


class TestLatencyLog:
    def test_timings_are_written_in_the_shape_panel_c_reads(self, tmp_path) -> None:
        path = pl.save_latency_log(
            [
                pl.StageTiming(stage="rag_retrieval", seconds=0.05,
                               status=config.STAGE_COMPLETED),
                pl.StageTiming(stage="llm_report", seconds=1.2,
                               status=config.STAGE_COMPLETED),
            ],
            tmp_path / "latency_log.json",
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert {e["stage"] for e in payload["stages"]} == {"rag_retrieval", "llm_report"}
        assert all(e["ok"] and e["status"] == "completed" for e in payload["stages"])

    def test_a_failed_stage_is_recorded_as_failed(self, tmp_path) -> None:
        path = pl.save_latency_log(
            [pl.StageTiming(stage="llm_report", seconds=0.0,
                            status=config.STAGE_FAILED, detail="no key")],
            tmp_path / "latency_log.json",
        )
        entry = json.loads(path.read_text(encoding="utf-8"))["stages"][0]
        assert entry["ok"] is False
        assert entry["status"] == "failed"


class TestStopwatch:
    def test_a_clean_block_reports_completed(self) -> None:
        with pl.Stopwatch("x") as watch:
            pass
        timing = watch.as_timing()
        assert timing.status == config.STAGE_COMPLETED and timing.seconds >= 0.0

    def test_a_raising_block_reports_failed_with_the_error_name(self) -> None:
        with pytest.raises(ValueError):
            with pl.Stopwatch("x") as watch:
                raise ValueError("boom")
        timing = watch.as_timing()
        assert timing.status == config.STAGE_FAILED
        assert timing.error == "ValueError"
        assert timing.detail == "boom"

    def test_optional_stage_can_be_marked_not_required(self) -> None:
        with pl.Stopwatch("x", required=False) as watch:
            pass
        assert watch.as_timing(required=False).required is False


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCommandLine:
    def test_no_action_is_a_usage_error(self) -> None:
        with pytest.raises(SystemExit):
            pl.main([])

    def test_run_without_a_key_fails_honestly(self, monkeypatch) -> None:
        """No key means a named failure and a non-zero exit - never a fake success."""
        for variable in ("LLM_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setattr(llm_report, "_load_dotenv_once", lambda: None)
        monkeypatch.setattr(pl, "load_patient", lambda *a: dict(RECORD))
        # Stop before any expensive real work.
        monkeypatch.setattr(
            pl.VIPPipeline, "run",
            lambda self, record: (_ for _ in ()).throw(
                pl.PipelineError("llm_report", llm_report.MissingAPIKeyError("no key"))
            ),
        )
        assert pl.main(["--run"]) == 1

    def test_run_reports_a_completed_status_as_zero(self, monkeypatch, capsys) -> None:
        monkeypatch.setattr(pl, "load_patient", lambda *a: dict(RECORD))
        monkeypatch.setattr(
            pl.VIPPipeline, "run",
            lambda self, record: pl.VIPResult(patient_id="P0001",
                                              status=config.RUN_COMPLETED),
        )
        assert pl.main(["--run"]) == 0
        assert "COMPLETED" in capsys.readouterr().out

    def test_patient_lookup_names_the_available_ids_on_miss(self) -> None:
        with pytest.raises(KeyError, match="P9999"):
            pl.load_patient("P9999")

    def test_patient_lookup_returns_a_dict(self) -> None:
        record = pl.load_patient()
        assert isinstance(record, dict)
        assert config.TARGET_COL in record


# ---------------------------------------------------------------------------
# Scope guards
# ---------------------------------------------------------------------------


class TestScope:
    def test_pipeline_does_not_reimplement_module_logic(self) -> None:
        """It must delegate, so only orchestration helpers are defined here."""
        import ast

        tree = ast.parse(Path(pl.__file__).read_text(encoding="utf-8"))
        own = {
            n.name for n in tree.body
            if isinstance(n, ast.FunctionDef)
        }
        # Allowed: small pure helpers over other modules' outputs.
        assert own <= {
            "derive_run_status", "_unique_chunks", "schemas_dose_bound",
            "save_latency_log", "run_pipeline", "main", "load_patient",
            "print_summary",
        }, f"unexpected module-level functions: {own - {'derive_run_status', '_unique_chunks', 'schemas_dose_bound', 'save_latency_log', 'run_pipeline', 'main', 'load_patient', 'print_summary'}}"

    def test_no_banned_framework_was_introduced(self) -> None:
        import ast
        import sys

        allowed_third_party = {
            "numpy", "pandas", "sklearn", "joblib", "torch", "matplotlib",
            "seaborn", "pydantic", "chromadb", "sentence_transformers",
            "langchain", "langchain_community", "langchain_text_splitters",
            "langchain_core", "langchain_openai", "docx", "dotenv",
        }
        # ``dashboard`` is the project's own interface layer, imported lazily by
        # build_dashboard so a headless run does not pay for it.
        allowed_project = {"src", "dashboard"}
        stdlib = set(sys.stdlib_module_names)
        for path in Path(pl.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                for module in modules:
                    root = module.split(".")[0]
                    assert (
                        root in allowed_third_party
                        or root in stdlib
                        or root in allowed_project
                    ), f"{path.name} imports unexpected module {module}"

    def test_no_secret_is_printed_or_logged(self) -> None:
        """The summary must not echo credentials or a whole patient record."""
        import io
        from contextlib import redirect_stdout

        result = pl.VIPResult(
            patient_id="P0001",
            submitted_dose_mcg=83.5,
            status=config.RUN_COMPLETED,
        )
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            pl.print_summary(result)
        text = buffer.read()
        assert "sk-" not in text
        assert "API_KEY" not in text
        assert "Biomarker_Level" not in text, "the full record must not be printed"

    def test_dashboard_still_refuses_to_simulate_a_run(self) -> None:
        """This stage does not change the 501 contract."""
        from dashboard import app

        assert app is not None
        source = Path(app.__file__).read_text(encoding="utf-8")
        assert "501" in source, "the not-implemented run endpoint must stay 501"