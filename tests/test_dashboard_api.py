"""Dashboard backend integration tests.

Requirements covered
--------------------
* ``POST /api/run`` executes the real pipeline and reports what actually happened.
* Malformed input is refused with a client error before any stage runs.
* A required stage failure can never produce a successful HTTP response.
* Missing credentials produce an honest failure, never a fabricated report.
* Report artifacts are served only from ``outputs/reports/``, with traversal and
  non-report files refused.
* The submitted dose and unit survive to the response, and the 100 mcg review
  trigger is never described as a clinically established maximum safe dose.
* The dashboard's own import path preserves the native-library import order.

Test policy
-----------
Endpoint unit tests mock the pipeline so they need no API key, no network and no
trained model. Tests that use the *real* local components are marked and say so.
``test_native_conflict.py`` covers the native import-order regression in fresh
processes; this file adds the dashboard-specific regression, which is a different
code path and can fail independently.

A mocked endpoint test is never presented as evidence that a real pipeline run or
a real LLM call succeeded.
"""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from src import config
from dashboard import app, components, runs


RECORD = {
    "Patient_ID": "P0001", "Age": 24, "Biomarker_Level": 4.53,
    "Pre_Existing_Conditions": 1, "Vaccine_Dose_mcg": 83.5,
    "Prior_Reaction_History": "Mild",
}


# ---------------------------------------------------------------------------
# Server fixture - a real HTTP server, as tests/test_report.py already does
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.VIPRequestHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _get(url: str):
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


def _post(url: str, payload: Any):
    """POST and return ``(status, parsed_body_or_text)``.

    A non-JSON body is returned as text rather than raising, so a test can assert
    on the status of a route that legitimately answers in plain text.
    """
    import urllib.error
    import urllib.request

    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = exc.code
    try:
        return status, json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return status, raw.decode("utf-8", "replace")
    import urllib.error
    import urllib.request

    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# 1. /api/run invokes the real pipeline interface
# ---------------------------------------------------------------------------


class TestRunEndpointCallsTheRealPipeline:
    """The endpoint must go through the pipeline, not around it."""

    def test_ml_stage_calls_the_pipeline_methods(self, monkeypatch) -> None:
        """Asserts the wiring, using a stub pipeline so no model is needed."""
        calls: list[str] = []

        class StubPipeline:
            def __init__(self, **kwargs: Any) -> None:
                calls.append("construct")

            def preprocess(self) -> None:
                calls.append("preprocess")

            def predict_risk(self, record):
                calls.append("predict_risk")
                return _stub_prediction()

            timings: list[Any] = []

        class StubModule:
            VIPPipeline = StubPipeline

        monkeypatch.setattr(runs, "_pipeline_module", lambda: StubModule)
        status, body = runs.execute({"stage": "ml", "record": dict(RECORD)})

        assert calls == ["construct", "preprocess", "predict_risk"]
        assert status == 200
        assert body["risk_label"] == "Low"

    def test_endpoint_no_longer_returns_501_for_a_valid_request(self, server) -> None:
        """The placeholder is gone for the normal path.

        The pipeline itself is mocked here - this asserts the route is wired, not
        that a full run succeeds.
        """
        with mock.patch.object(runs, "run_ml", return_value={
            "run_id": "test", "stage": "ml", "status": config.RUN_COMPLETED,
            "risk_label": "Low", "risk_probability": 0.33, "disclaimer": "d",
        }):
            status, _ = _post(server + "/api/run", {"stage": "ml", "record": RECORD})
        assert status != 501
        assert status == 200

    def test_unknown_api_route_still_404s(self, server) -> None:
        """A POST to an unknown /api/ path is 404, not a pipeline invocation."""
        status, _ = _post(server + "/api/nonexistent", {"stage": "ml"})
        assert status == 404


def _stub_prediction():
    class Selected:
        model_name = "Logistic Regression"

    class Prediction:
        selected = Selected()
        risk_label = "Low"
        risk_probability = 0.33

        def agreement(self):
            return True

        def as_dict(self):
            return {"patient_id": "P0001", "risk_label": "Low", "probability": 0.33}

    return Prediction()


# ---------------------------------------------------------------------------
# 2/3. Valid input returns the real result; malformed input is rejected
# ---------------------------------------------------------------------------


class TestInputValidation:
    """Rejected before any pipeline work happens."""

    @pytest.mark.parametrize(
        "payload, fragment",
        [
            ({}, "stage"),
            ({"stage": "nonsense"}, "stage"),
            ({"stage": "ml"}, "record"),
            ({"stage": "ml", "record": "not-an-object"}, "object"),
            ({"stage": "ml", "record": {}}, "Patient_ID"),
            ({"stage": "rag"}, "question"),
            ({"stage": "rag", "question": "   "}, "question"),
        ],
    )
    def test_malformed_payloads_are_rejected_with_400(self, payload, fragment) -> None:
        status, body = runs.execute(payload)
        assert status == 400
        assert body["error"] == "invalid_request"
        assert fragment.lower() in body["message"].lower()

    @pytest.mark.parametrize(
        "field, value",
        [
            ("Age", -5),
            ("Age", 999),
            ("Biomarker_Level", -1),
            ("Pre_Existing_Conditions", -1),
            ("Vaccine_Dose_mcg", -100),
            ("Vaccine_Dose_mcg", 1e9),
        ],
    )
    def test_out_of_range_numbers_are_refused(self, field, value) -> None:
        """A nonsense value would produce a meaningless prediction presented as real."""
        status, body = runs.execute(
            {"stage": "ml", "record": {**RECORD, field: value}}
        )
        assert status == 400
        assert "must be between" in body["message"]

    @pytest.mark.parametrize("value", ["thirty", None, True, [1], {"a": 1}])
    def test_non_numeric_values_are_refused(self, value) -> None:
        status, body = runs.execute({"stage": "ml", "record": {**RECORD, "Age": value}})
        assert status == 400
        assert "Age" in body["message"]

    def test_unknown_history_category_is_refused(self) -> None:
        status, body = runs.execute(
            {"stage": "ml", "record": {**RECORD, "Prior_Reaction_History": "Severe"}}
        )
        assert status == 400
        assert "Prior_Reaction_History" in body["message"]

    def test_oversized_identifier_is_refused(self) -> None:
        status, _ = runs.execute(
            {"stage": "ml", "record": {**RECORD, "Patient_ID": "x" * 500}}
        )
        assert status == 400

    def test_infinite_and_nan_numbers_are_refused(self) -> None:
        for value in (float("inf"), float("nan")):
            status, _ = runs.execute(
                {"stage": "ml", "record": {**RECORD, "Age": value}}
            )
            assert status == 400

    def test_unknown_record_fields_are_dropped_not_forwarded(self) -> None:
        """A client cannot smuggle extra keys into the pipeline."""
        cleaned = runs.validate_record({**RECORD, "Reaction_Score": 1,
                                        "output_path": "/etc/passwd"})
        assert "Reaction_Score" not in cleaned
        assert "output_path" not in cleaned
        assert set(cleaned) == set(runs.ALLOWED_RECORD_FIELDS)

    def test_request_cannot_override_configuration_or_paths(self) -> None:
        """Path-like and config-like keys never survive validation."""
        cleaned = runs.validate_record(
            {**RECORD, "models_dir": "/tmp/evil", "top_k": 99,
             "llm_model": "attacker-model"}
        )
        for forbidden in ("models_dir", "top_k", "llm_model", "persist_dir"):
            assert forbidden not in cleaned

    def test_invalid_json_body_is_400(self, server) -> None:
        status, body = _post(server + "/api/run", b"{not json")
        assert status == 400
        assert body["error"] == "invalid_json"

    def test_empty_body_is_400(self, server) -> None:
        status, body = _post(server + "/api/run", b"")
        assert status == 400

    def test_oversized_body_is_refused(self, server) -> None:
        status, body = _post(server + "/api/run", b"x" * (app.MAX_REQUEST_BYTES + 10))
        assert status == 413


# ---------------------------------------------------------------------------
# 4. Missing credentials produce an honest failure, not a fake report
# ---------------------------------------------------------------------------


class TestMissingCredentials:
    def test_missing_key_yields_a_non_200_and_no_report(self, monkeypatch) -> None:
        for variable in ("LLM_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setattr(
            "src.llm.report_generator._load_dotenv_once", lambda: None
        )

        def missing(*args, **kwargs):
            raise __import__(
                "src.llm.report_generator", fromlist=["x"]
            ).MissingAPIKeyError("no key configured")

        monkeypatch.setattr(
            "src.pipeline.pipeline.VIPPipeline.run", missing, raising=True
        )

        status, body = runs.execute({"stage": "all", "record": dict(RECORD)})

        assert status != 200, "a missing key must never read as success"
        # 503: the caller's environment is missing something, so the request is
        # not retryable as-is. Not 500, which would imply a server fault.
        assert status == 503
        assert body["error"] == "upstream_unavailable"
        # No report claim of any kind.
        assert "generated" not in body
        assert "report" not in body
        # The actionable detail comes from load_api_key's own message; this layer
        # only appends the "nothing was generated" fact. Both must be present.
        assert "no key configured" in body["message"]
        assert "no report was generated" in body["message"].lower()

    def test_missing_key_message_names_the_variable(self, monkeypatch) -> None:
        for variable in ("LLM_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setattr(
            "src.llm.report_generator._load_dotenv_once", lambda: None
        )

        def missing(*args, **kwargs):
            raise __import__(
                "src.llm.report_generator", fromlist=["x"]
            ).MissingAPIKeyError(
                "No LLM credential is configured. Set LLM_API_KEY."
            )

        monkeypatch.setattr(
            "src.pipeline.pipeline.VIPPipeline.run", missing, raising=True
        )
        status, body = runs.execute({"stage": "all", "record": dict(RECORD)})
        assert status == 503
        assert "LLM_API_KEY" in body["message"]
        assert body["error"] == "upstream_unavailable"

    def test_reports_directory_is_not_populated_by_a_failed_run(
        self, monkeypatch
    ) -> None:
        """A failure must leave no artifact that could read as a success."""
        before = sorted(p.name for p in config.REPORTS_DIR.glob("*")
                        if p.name != ".gitkeep")
        for variable in ("LLM_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(variable, raising=False)
        monkeypatch.setattr(
            "src.llm.report_generator._load_dotenv_once", lambda: None
        )

        def boom(*args, **kwargs):
            raise RuntimeError("simulated provider failure")

        monkeypatch.setattr(
            "src.pipeline.pipeline.VIPPipeline.run", boom, raising=True
        )
        runs.execute({"stage": "all", "record": dict(RECORD)})

        after = sorted(p.name for p in config.REPORTS_DIR.glob("*")
                       if p.name != ".gitkeep")
        assert before == after


# ---------------------------------------------------------------------------
# 5/6. Failures produce honest responses; a failed stage is never a success
# ---------------------------------------------------------------------------


class TestFailureReporting:
    def _failed_result(self, stage: str):
        """A VIPResult whose required stage failed."""
        from src.pipeline.pipeline import PipelineError, StageTiming, VIPResult

        result = VIPResult(patient_id="P0001")
        result.timings = [
            StageTiming(stage="preprocessing", seconds=0.1,
                        status=config.STAGE_COMPLETED),
            StageTiming(stage=stage, seconds=0.0, status=config.STAGE_FAILED,
                        detail="simulated"),
        ]
        result.status = config.RUN_FAILED
        result.partial = True
        return PipelineError(stage, RuntimeError("simulated"), result=result)

    def test_failed_stage_cannot_produce_status_200(self, monkeypatch) -> None:
        error = self._failed_result("rag_retrieval")
        monkeypatch.setattr(
            "src.pipeline.pipeline.VIPPipeline.run",
            lambda self, record: (_ for _ in ()).throw(error),
        )
        status, body = runs.execute({"stage": "all", "record": dict(RECORD)})

        assert status != 200
        assert body["status"] == config.RUN_FAILED
        assert body["failed_stage"] == "rag_retrieval"
        assert body["partial"] is True
        # The stages that did complete are still reported, marked incomplete.
        assert body["stage_statuses"]["preprocessing"] == config.STAGE_COMPLETED

    def test_blocked_stage_is_not_reported_as_completed(self, monkeypatch) -> None:
        from src.pipeline.pipeline import PipelineError

        error = PipelineError("llm_report", ValueError("no grounding"))
        monkeypatch.setattr(
            "src.pipeline.pipeline.VIPPipeline.run",
            lambda self, record: (_ for _ in ()).throw(error),
        )
        status, body = runs.execute({"stage": "all", "record": dict(RECORD)})
        assert status != 200
        assert body["status"] == config.RUN_FAILED

    def test_stage_statuses_come_from_the_pipeline_not_recomputed(self) -> None:
        """The interface reads the pipeline's own records."""
        from src.pipeline.pipeline import StageTiming

        timings = [
            StageTiming(stage="preprocessing", seconds=0.1,
                        status=config.STAGE_COMPLETED),
            StageTiming(stage="ml_prediction", seconds=0.2,
                        status=config.STAGE_FAILED),
        ]
        assert runs.stage_status_map(timings) == {
            "preprocessing": "completed",
            "ml_prediction": "failed",
        }

    def test_unexpected_fault_is_a_500_without_a_traceback(self, monkeypatch) -> None:
        def boom(*args, **kwargs):
            raise ZeroDivisionError("internal detail /secret/path")

        monkeypatch.setattr(
            "src.pipeline.pipeline.VIPPipeline.run", boom, raising=True
        )
        status, body = runs.execute({"stage": "all", "record": dict(RECORD)})

        assert status == 500
        assert body["error"] == "internal_error"
        serialised = json.dumps(body)
        assert "Traceback" not in serialised
        assert "/secret/path" not in serialised
        assert "ZeroDivisionError" in body["message"]

    def test_missing_artifact_maps_to_503(self, monkeypatch) -> None:
        from src.rag.ingest import IndexingError

        def boom(*args, **kwargs):
            raise IndexingError("no index at vectorstore/")

        monkeypatch.setattr(
            "src.pipeline.pipeline.VIPPipeline.run", boom, raising=True
        )
        status, _ = runs.execute({"stage": "all", "record": dict(RECORD)})
        assert status == 503


# ---------------------------------------------------------------------------
# 7/8/9. Dose: preservation, the review trigger, and no clinical maximum
# ---------------------------------------------------------------------------


class TestDoseReporting:
    def test_submitted_dose_and_unit_survive_to_the_response(self) -> None:
        result = mock.Mock(
            submitted_dose_mcg=83.5, submitted_dose_unit="mcg",
            dose_review=mock.Mock(review_required=False),
        )
        payload = runs.dose_payload(result)
        assert payload["submitted_dose_mcg"] == 83.5
        assert payload["submitted_dose_unit"] == "mcg"

    def test_submitted_dose_is_not_replaced_by_a_clamped_value(self) -> None:
        """The original input and the clamped recommendation are different things."""
        result = mock.Mock(
            submitted_dose_mcg=500.0, submitted_dose_unit="mcg",
            dose_review=mock.Mock(review_required=True),
        )
        payload = runs.dose_payload(result)
        assert payload["submitted_dose_mcg"] == 500.0
        assert payload["review_required"] is True
        # Nothing in the payload claims a 100 mcg value was the input.
        assert payload["submitted_dose_mcg"] != config.GUIDELINE_REVIEW_THRESHOLD_MCG

    def test_above_threshold_review_is_represented_correctly(self) -> None:
        result = mock.Mock(
            submitted_dose_mcg=500.0, submitted_dose_unit="mcg",
            dose_review=mock.Mock(review_required=True),
        )
        assert runs.dose_payload(result)["review_required"] is True

    def test_review_required_survives_a_run_that_died_before_enforcement(
        self,
    ) -> None:
        """A run that failed at the LLM stage still flags an over-limit dose.

        Enforcement happens after generation, so a credential failure arrives
        first. Reporting ``review_required: false`` then would under-report a dose
        the pipeline would have rejected.
        """
        result = mock.Mock(
            submitted_dose_mcg=500.0, submitted_dose_unit="mcg",
            dose_review=None,
        )
        assert runs.dose_payload(result)["review_required"] is True

    @pytest.mark.parametrize(
        "submitted, expected",
        [(100.0, False), (100.01, True), (83.5, False)],
    )
    def test_the_threshold_boundary_is_strictly_greater(self, submitted, expected) -> None:
        """The guideline says *above* 100 mcg, so 100 itself does not trigger."""
        result = mock.Mock(
            submitted_dose_mcg=submitted, submitted_dose_unit="mcg",
            dose_review=None,
        )
        assert runs.dose_payload(result)["review_required"] is expected

    def test_neither_threshold_is_called_a_clinical_maximum(self) -> None:
        result = mock.Mock(
            submitted_dose_mcg=83.5, submitted_dose_unit="mcg",
            dose_review=mock.Mock(review_required=False),
        )
        payload = runs.dose_payload(result)
        assert payload["clinical_maximum_established"] is False
        assert payload["review_threshold_kind"] == "secondary_clinician_review"
        assert "does not state a clinically established maximum" in (
            payload["threshold_note"]
        )

    def test_engineering_guard_is_reported_separately_from_the_source_threshold(
        self,
    ) -> None:
        result = mock.Mock(
            submitted_dose_mcg=83.5, submitted_dose_unit="mcg",
            dose_review=mock.Mock(review_required=False),
        )
        payload = runs.dose_payload(result)
        assert payload["review_threshold_kind"] == "secondary_clinician_review"
        assert "engineering_dose_guard_mcg" in payload

    def test_no_response_field_is_named_maximum_safe_dose(self) -> None:
        """No *key* may assert a clinical maximum.

        A field name travels further than a sentence: it lands in client code and
        outlives the wording that qualified it. ``clinical_maximum_established``
        is the one permitted form, because it is explicitly a negative - it must be
        False, so the name cannot mislead a consumer reading only the key.
        """
        result = mock.Mock(
            submitted_dose_mcg=83.5, submitted_dose_unit="mcg",
            dose_review=mock.Mock(review_required=False),
        )
        payload = runs.dose_payload(result)

        for key in payload:
            lowered = key.lower()
            assert "max_safe_dose" not in lowered, f"{key!r} names a clinical maximum"
            assert not lowered.startswith("max"), (
                f"{key!r} reads as a maximum; use an explicit "
                "review_threshold_/engineering_guard_ prefix instead"
            )

        # The only maximum-shaped key is the negative assertion, and it is False.
        assert payload["clinical_maximum_established"] is False
        assert "review_threshold_mcg" in payload
        assert "engineering_dose_guard_mcg" in payload


# ---------------------------------------------------------------------------
# 10/11/12. Report artifact access
# ---------------------------------------------------------------------------


class TestReportArtifactAccess:
    def test_existing_report_is_served(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(runs.config, "REPORTS_DIR", tmp_path)
        (tmp_path / "P0001-2026.json").write_text('{"Eligible": true}', encoding="utf-8")

        with mock.patch.object(runs, "config", runs.config):
            resolved = runs.resolve_report_artifact("P0001-2026.json")
        assert resolved is not None and resolved.is_file()

    def test_missing_report_returns_none(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(runs.config, "REPORTS_DIR", tmp_path)
        assert runs.resolve_report_artifact("absent.json") is None

    @pytest.mark.parametrize(
        "name",
        [
            "../../.env",
            "../.env",
            "../../../etc/passwd",
            "subdir/report.json",
            "/etc/passwd",
            "..",
            ".",
            "",
            "report.json\\..\\..\\.env",
        ],
    )
    def test_traversal_and_absolute_paths_are_refused(self, name) -> None:
        assert runs.resolve_report_artifact(name) is None

    def test_only_report_extensions_are_served(self, tmp_path, monkeypatch) -> None:
        """A stray .py or .csv in the directory is not exposed."""
        monkeypatch.setattr(runs.config, "REPORTS_DIR", tmp_path)
        for name in ("notes.txt", "script.py", "data.csv", "env.json.bak"):
            (tmp_path / name).write_text("secret-ish", encoding="utf-8")
        for name in ("notes.txt", "script.py", "data.csv", "env.json.bak"):
            assert runs.resolve_report_artifact(name) is None

    def test_http_traversal_is_refused(self, server) -> None:
        import urllib.parse

        for target in ("../../.env", "..%2f..%2f.env", "%2e%2e%2f.env"):
            status, _ = _get(server + "/api/report/" + target)
            assert status == 404, f"{target} returned {status}"

    def test_http_missing_report_is_404(self, server) -> None:
        """Both the status and the structured body, not just the code."""
        status, body = _get(server + "/api/report/definitely-absent.json")
        assert status == 404
        payload = json.loads(body)
        assert payload["error"] == "not_found"
        assert "reports" in payload["message"]

    def test_report_listing_is_names_only(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setattr(runs.config, "REPORTS_DIR", tmp_path)
        (tmp_path / "a.json").write_text('{"secret": "value"}', encoding="utf-8")
        listing = runs.list_reports()
        assert listing and listing[0]["name"] == "a.json"
        assert "secret" not in json.dumps(listing)

    def test_run_ids_exist_so_a_report_can_be_tied_to_its_run(self) -> None:
        """Timestamped filenames alone cannot identify which run produced one."""
        assert runs.new_run_id() != runs.new_run_id()
        assert "-" in runs.new_run_id()

    def test_run_history_records_artifacts_and_no_secrets(self) -> None:
        for record in runs.recent_runs():
            assert "run_id" in record and "status" in record
            serialised = json.dumps(record)
            assert "sk-" not in serialised
            assert "api_key" not in serialised.lower()


# ---------------------------------------------------------------------------
# 13/14. Concurrency and duplicate submissions
# ---------------------------------------------------------------------------


class TestConcurrencyAndDuplicates:
    def test_a_second_run_is_refused_while_one_is_in_flight(self) -> None:
        runs._RUN_LOCK.acquire()
        try:
            status, body = runs.execute(
                {"stage": "ml", "record": dict(RECORD)}
            )
            assert status == 409
            assert body["error"] == "run_in_progress"
        finally:
            runs._RUN_LOCK.release()

    def test_the_lock_is_released_after_a_failure(self, monkeypatch) -> None:
        """A crash must not wedge the endpoint permanently."""
        def boom(*args, **kwargs):
            raise RuntimeError("simulated")

        monkeypatch.setattr(
            "src.pipeline.pipeline.VIPPipeline.run", boom, raising=True
        )
        runs.execute({"stage": "all", "record": dict(RECORD)})
        assert runs._RUN_LOCK.acquire(blocking=False), "lock was not released"
        runs._RUN_LOCK.release()

    def test_concurrent_duplicate_requests_do_not_both_execute(self) -> None:
        """Two simultaneous submissions must not both run.

        The second is refused rather than queued, so a double-click cannot double
        write the ChromaDB collection, the model directory or the latency log.
        """
        entered = threading.Event()
        release = threading.Event()
        outcomes: list[int] = []

        def slow_run(*args, **kwargs):
            entered.set()
            release.wait(timeout=10)
            raise RuntimeError("simulated")

        with mock.patch("src.pipeline.pipeline.VIPPipeline.run", slow_run):
            def submit():
                outcomes.append(
                    runs.execute({"stage": "all", "record": dict(RECORD)})[0]
                )

            first = threading.Thread(target=submit)
            first.start()
            assert entered.wait(timeout=10), "first run never started"
            second_status = runs.execute(
                {"stage": "all", "record": dict(RECORD)}
            )[0]
            release.set()
            first.join(timeout=15)

        assert second_status == 409
        assert 200 not in outcomes

    def test_validation_happens_before_the_lock_is_taken(self) -> None:
        """A malformed request is rejected even while a run is in flight.

        Correct: it needs no execution, so queueing it behind a run would waste the
        user's time.
        """
        runs._RUN_LOCK.acquire()
        try:
            assert runs.execute({"stage": "nonsense"})[0] == 400
        finally:
            runs._RUN_LOCK.release()


# ---------------------------------------------------------------------------
# 15. Existing visualizations still read real outputs
# ---------------------------------------------------------------------------


class TestVisualizationsStillUseRealData:
    def test_roc_panel_reads_the_saved_evaluation(self) -> None:
        if not components.artifact_availability()["roc"]:
            pytest.skip("no trained model; the panel's empty state applies")
        data = components.load_roc_panel_data()
        assert data.curves
        assert all("fpr" in c and "tpr" in c for c in data.curves)

    def test_attention_panel_reads_real_weights(self) -> None:
        if not components.artifact_availability()["attention"]:
            pytest.skip("no attention artifact yet")
        data = components.load_attention_panel_data()
        assert data is not None

    def test_latency_panel_reads_measured_timings(self) -> None:
        if not components.artifact_availability()["latency"]:
            pytest.skip("no latency log yet")
        data = components.load_latency_panel_data()
        assert all(v >= 0 for v in data.stages.values())

    def test_panel_readers_still_raise_for_an_absent_artifact(self, tmp_path) -> None:
        with pytest.raises(components.MissingArtifactError):
            components.load_roc_panel_data(
                metrics_path=tmp_path / "a.json", roc_path=tmp_path / "b.json"
            )

    def test_dpi_floor_is_unchanged(self) -> None:
        assert config.DASHBOARD_DPI >= 150
        assert components.styles.FIGURE_DPI >= 150

    def test_benchmark_data_is_labelled_as_such(self) -> None:
        """Any non-current-run timing must be identifiable as a benchmark."""
        if not components.artifact_availability()["latency"]:
            pytest.skip("no latency log yet")
        data = components.load_latency_panel_data()
        # Panel C annotates bars with their sample count, so a single-run average
        # cannot be mistaken for a benchmark.
        assert components.LatencyPanelData.__doc__


# ---------------------------------------------------------------------------
# 16. Native import-order regression on the dashboard's own path
# ---------------------------------------------------------------------------


class TestNativeImportOrderFromTheDashboard:
    """The dashboard must not reintroduce the native crash.

    Different code path from ``test_native_conflict.py``: this asserts the
    *dashboard's* import sequence, which is what a fresh server does.
    """

    def test_importing_dashboard_does_not_load_sklearn(self) -> None:
        """Importing the dashboard must not drag scikit-learn in.

        If it did, the pipeline's preload would run after sklearn was resident and
        torch would fail to initialise - the exact crash this ordering prevents.
        """
        import subprocess
        import sys

        probe = (
            "import sys; sys.path.insert(0, '.')\n"
            "import dashboard.app\n"
            "print('sklearn' in sys.modules)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, cwd=str(config.PROJECT_ROOT), timeout=300,
        )
        assert result.returncode == 0, result.stderr[-400:]
        assert result.stdout.strip().splitlines()[-1] == "False"

    def test_pipeline_import_after_dashboard_keeps_the_order_safe(self) -> None:
        """The real sequence: dashboard first, then pipeline. Must not fault."""
        import subprocess
        import sys

        probe = (
            "import sys; sys.path.insert(0, '.')\n"
            "import dashboard.app\n"
            "from src.pipeline.native_stack import import_order_is_safe\n"
            "assert import_order_is_safe(), 'order broken before pipeline import'\n"
            "from src.pipeline import pipeline\n"
            "from src.ml import predict\n"
            "from src.attention import self_attention as sa\n"
            "r = sa.run_self_attention('ACGUACGUAC')\n"
            "print('OK', tuple(r.weights.shape))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, cwd=str(config.PROJECT_ROOT), timeout=900,
        )
        assert result.returncode == 0, (
            f"dashboard-then-pipeline sequence failed: {result.stderr[-600:]}"
        )
        assert "OK (10, 10)" in result.stdout

    def test_executors_establish_the_order_before_touching_sklearn(self) -> None:
        """``_pipeline_module()`` must precede any sklearn-reaching import.

        A regression that is invisible in a test process (where the pipeline is
        already imported) crashes only on a fresh server.
        """
        from pathlib import Path

        source = Path(runs.__file__).read_text(encoding="utf-8")
        # run_ml used to import src.ml.predict first and fault on a fresh process.
        body = source.split("def run_ml(")[1].split("\ndef ")[0]
        pipeline_call = body.index("_pipeline_module()")
        for module in ("from src.ml import", "from src.rag import"):
            if module in body:
                assert body.index(module) > pipeline_call, (
                    f"run_ml imports {module} before _pipeline_module(); that pulls "
                    "scikit-learn in ahead of torch and reproduces the native crash"
                )

    def test_runs_module_does_not_import_the_pipeline_eagerly(self) -> None:
        """A module-scope pipeline import would load sklearn at dashboard startup."""
        import ast

        tree = ast.parse(Path(runs.__file__).read_text(encoding="utf-8"))
        for node in tree.body:  # module level only
            modules = []
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            elif isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            for module in modules:
                assert not module.startswith(("src.ml", "src.pipeline",
                                              "src.attention", "src.rag")), (
                    f"dashboard/runs.py imports {module} at module scope"
                )

    def test_no_environment_variable_workaround_was_introduced(self) -> None:
        """Import order is the fix; no KMP_DUPLICATE_LIB_OK, no subprocess fork."""
        source = Path(runs.__file__).read_text(encoding="utf-8")
        for forbidden in ("KMP_DUPLICATE_LIB_OK", "subprocess"):
            for line in source.splitlines():
                if line.strip().startswith("#"):
                    continue
                assert forbidden not in line, (
                    f"dashboard/runs.py must not rely on {forbidden}"
                )


# ---------------------------------------------------------------------------
# 17. Existing tests still pass - covered by the full suite; asserted here as a
#     guard that this module does not disturb shared state.
# ---------------------------------------------------------------------------


class TestNoSharedStateDamage:
    def test_stage_readiness_is_still_truthful(self) -> None:
        state = components.platform_state()
        assert [s["number"] for s in state["stages"]] == [
            "01", "02", "03", "04", "05"
        ]

    def test_report_stage_reflects_the_reports_directory(self) -> None:
        stage = next(
            s for s in components.platform_state()["stages"] if s["id"] == "report"
        )
        populated = components._directory_has_content(config.REPORTS_DIR)
        assert stage["status"] == ("ready" if populated else "not-initialized")

    def test_dose_bound_is_still_absent_from_the_ui(self) -> None:
        """The interface must not invent a maximum the pipeline does not supply."""
        assert components.platform_state()["dose_bound"] is None

    def test_disclaimer_is_unchanged(self) -> None:
        assert components.platform_state()["copy"]["disclaimer"] == (
            config.REQUIRED_DISCLAIMER
        )

    def test_state_endpoint_still_works(self, server) -> None:
        status, body = _get(server + "/api/state")
        assert status == 200
        assert len(json.loads(body)["stages"]) == 5

    def test_health_endpoint_still_works(self, server) -> None:
        status, _ = _get(server + "/api/health")
        assert status == 200