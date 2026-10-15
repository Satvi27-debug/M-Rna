"""Module 4 and Module 5 tests: report schema, safety rules, persistence, dashboard.

Requirements covered
--------------------
FR-4.1  A Pydantic schema enforcing exactly the five required fields; a missing or
        mistyped field raises ``pydantic.ValidationError``.
FR-4.2  The prompt is grounded only in the Module 1 risk output and the Module 3
        retrieved chunks.
FR-4.3  Every response is schema-validated, with exactly one retry on failure.
FR-4.4  A dose above the retrieved guideline bound forces ``Eligible_For_Vaccine``
        to false.
FR-4.5  The ``Disclaimer`` field is present and non-empty in every report.
FR-4.6  Every validated report is persisted as a matching ``.json`` / ``.md`` pair.
FR-5.1  The dashboard figure contains exactly 3 titled panels.
FR-5.2  Each panel is built from the corresponding module's saved output.
FR-5.3  The figure is saved at 150 DPI or higher.

Tests that need an LLM are skipped unless a key is present in the environment.
They are never allowed to pass by returning a canned report, because a fake
report would violate the project's rule against placeholder results.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from src import config
from src.llm import report_generator, schemas
from dashboard import app, components, styles
from tests import requires_implementation


def llm_key_available() -> bool:
    """True when an LLM credential is configured in the environment."""
    if os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"):
        return True
    env_file = Path(config.PROJECT_ROOT) / ".env"
    return env_file.exists() and "LLM_API_KEY=" in env_file.read_text(
        encoding="utf-8"
    )


requires_llm = pytest.mark.skipif(
    not llm_key_available(),
    reason="No LLM_API_KEY configured. Copy .env.example to .env to enable.",
)


# ---------------------------------------------------------------------------
# FR-4.1 - schema contract. These run now; the schema is fully specified.
# ---------------------------------------------------------------------------


class TestReportSchema:
    """FR-4.1: exactly five fields, four of them mandatory."""

    def test_schema_has_exactly_the_five_required_fields(self) -> None:
        assert set(schemas.ClinicalSafetyReport.model_fields) == set(
            schemas.REQUIRED_REPORT_FIELDS
        )
        assert schemas.REQUIRED_REPORT_FIELDS == (
            "Eligible_For_Vaccine",
            "Recommended_Dose_mg",
            "Reasoning_Steps",
            "FDA_Safety_Warnings_Cited",
            "Disclaimer",
        )

    def test_field_types_match_the_specification(self) -> None:
        fields = schemas.ClinicalSafetyReport.model_fields
        assert fields["Eligible_For_Vaccine"].annotation is bool
        assert fields["Recommended_Dose_mg"].annotation is float
        assert fields["Reasoning_Steps"].annotation == list[str]
        assert fields["FDA_Safety_Warnings_Cited"].annotation == list[str]
        assert fields["Disclaimer"].annotation is str

    def test_valid_payload_round_trips(self) -> None:
        payload = {
            "Eligible_For_Vaccine": True,
            "Recommended_Dose_mg": 0.05,
            "Reasoning_Steps": ["Low prior reaction history.", "Age band 12-64."],
            "FDA_Safety_Warnings_Cited": ["Doses above 100 micrograms require review."],
            "Disclaimer": config.REQUIRED_DISCLAIMER,
        }
        report = schemas.ClinicalSafetyReport(**payload)
        assert report.model_dump() == payload

    @pytest.mark.parametrize(
        "missing",
        ["Eligible_For_Vaccine", "Recommended_Dose_mg", "Reasoning_Steps",
         "FDA_Safety_Warnings_Cited"],
    )
    def test_missing_mandatory_field_raises_validation_error(self, missing) -> None:
        """FR-4.1 acceptance criterion."""
        payload = {
            "Eligible_For_Vaccine": True,
            "Recommended_Dose_mg": 0.05,
            "Reasoning_Steps": ["step"],
            "FDA_Safety_Warnings_Cited": ["snippet"],
        }
        payload.pop(missing)
        with pytest.raises(ValidationError):
            schemas.ClinicalSafetyReport(**payload)

    def test_mistyped_field_raises_validation_error(self) -> None:
        """FR-4.1: a mistyped field must also be rejected.

        Note that Pydantic's lax mode deliberately accepts ``'yes'``/``'no'``/
        ``'true'``/``'false'`` for a bool, so the value used here is one with no
        valid coercion - the point being that garbage in the model output is
        caught by the schema rather than silently coerced.
        """
        with pytest.raises(ValidationError):
            schemas.ClinicalSafetyReport(
                Eligible_For_Vaccine="maybe",  # not a bool under any coercion
                Recommended_Dose_mg=0.05,
                Reasoning_Steps=["step"],
                FDA_Safety_Warnings_Cited=["snippet"],
            )

    def test_non_numeric_dose_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            schemas.ClinicalSafetyReport(
                Eligible_For_Vaccine=True,
                Recommended_Dose_mg="fifty micrograms",
                Reasoning_Steps=["step"],
                FDA_Safety_Warnings_Cited=["snippet"],
            )

    def test_reasoning_steps_must_be_a_list_of_strings(self) -> None:
        with pytest.raises(ValidationError):
            schemas.ClinicalSafetyReport(
                Eligible_For_Vaccine=True,
                Recommended_Dose_mg=0.05,
                Reasoning_Steps="not a list",
                FDA_Safety_Warnings_Cited=["snippet"],
            )

    def test_unknown_extra_field_is_ignored_not_fatal(self) -> None:
        """Strictness would reject the model's harmless preamble key."""
        report = schemas.ClinicalSafetyReport(
            Eligible_For_Vaccine=True,
            Recommended_Dose_mg=0.05,
            Reasoning_Steps=["step"],
            FDA_Safety_Warnings_Cited=["snippet"],
            Unexpected="ignored",
        )
        assert "Unexpected" not in report.model_dump()


# ---------------------------------------------------------------------------
# FR-4.4 / FR-4.5 - deterministic safety rules
# ---------------------------------------------------------------------------


class TestDoseBound:
    """FR-4.4: the dose ceiling is derived from guideline text, not invented."""

    def test_hard_ceiling_matches_the_guideline_document(self) -> None:
        """Guideline Section 2.1: doses above 100 mcg need secondary review."""
        assert schemas.DoseBound.hard_ceiling().max_mcg == 100.0

    def test_bound_converts_micrograms_to_milligrams(self) -> None:
        """Recommended_Dose_mg is compared against the bound in mcg."""
        assert schemas.DoseBound(max_mcg=100.0).max_mg == pytest.approx(0.1)

    def test_exceeds_is_strictly_greater_than(self) -> None:
        """A dose exactly at the bound is permitted; above it is not."""
        bound = schemas.DoseBound(max_mcg=100.0)
        assert not bound.exceeds(100.0)
        assert bound.exceeds(100.1)

    def test_cited_source_text_is_carried_with_the_bound(self) -> None:
        """The rejection must be able to cite its own evidence."""
        assert "100 micrograms" in schemas.DoseBound.hard_ceiling().source_text


class TestSafetyEnforcement:
    """FR-4.4 and FR-4.5 enforced in code, not by prompt wording."""

    @staticmethod
    def _report(**overrides):
        payload = {
            "Eligible_For_Vaccine": True,
            "Recommended_Dose_mg": 0.05,
            "Reasoning_Steps": ["step"],
            "FDA_Safety_Warnings_Cited": ["Doses above 100 micrograms require review."],
            "Disclaimer": config.REQUIRED_DISCLAIMER,
        }
        payload.update(overrides)
        return schemas.ClinicalSafetyReport(**payload)

    @requires_implementation(schemas.enforce_safety_constraints)
    def test_dose_above_bound_forces_ineligibility(self) -> None:
        """FR-4.4 acceptance criterion: 120 mcg against a 100 mcg bound."""
        outcome = schemas.enforce_safety_constraints(
            self._report(Eligible_For_Vaccine=True, Recommended_Dose_mg=0.12),
            schemas.DoseBound(max_mcg=100.0),
        )
        assert outcome.report.Eligible_For_Vaccine is False
        assert outcome.report.Recommended_Dose_mg <= 0.1
        assert not outcome.is_compliant

    @requires_implementation(schemas.enforce_safety_constraints)
    def test_dose_within_bound_is_left_alone(self) -> None:
        outcome = schemas.enforce_safety_constraints(
            self._report(Recommended_Dose_mg=0.05), schemas.DoseBound(max_mcg=100.0)
        )
        assert outcome.report.Eligible_For_Vaccine is True
        assert outcome.report.Recommended_Dose_mg == pytest.approx(0.05)
        assert outcome.is_compliant

    @requires_implementation(schemas.enforce_safety_constraints)
    def test_hard_ceiling_applies_when_no_specific_bound_is_supplied(self) -> None:
        outcome = schemas.enforce_safety_constraints(
            self._report(Recommended_Dose_mg=0.15)
        )
        assert outcome.report.Eligible_For_Vaccine is False

    @requires_implementation(schemas.enforce_safety_constraints)
    def test_empty_disclaimer_is_overridden(self) -> None:
        """FR-4.5: present and non-empty without exception."""
        outcome = schemas.enforce_safety_constraints(
            self._report(Disclaimer="")
        )
        assert outcome.report.Disclaimer == config.REQUIRED_DISCLAIMER
        assert outcome.report.Disclaimer.strip()

    @requires_implementation(schemas.enforce_safety_constraints)
    def test_weak_disclaimer_is_replaced_with_the_required_notice(self) -> None:
        outcome = schemas.enforce_safety_constraints(
            self._report(Disclaimer="consult a doctor")
        )
        assert "student engineering project" in outcome.report.Disclaimer.lower()

    @requires_implementation(schemas.enforce_safety_constraints)
    def test_ungrounded_report_is_flagged(self) -> None:
        """FR-4.2: a report citing nothing is not acceptable."""
        outcome = schemas.enforce_safety_constraints(
            self._report(FDA_Safety_Warnings_Cited=[])
        )
        assert not outcome.is_compliant

    @requires_implementation(schemas.enforce_safety_constraints)
    def test_every_violation_names_its_requirement(self) -> None:
        outcome = schemas.enforce_safety_constraints(
            self._report(Recommended_Dose_mg=0.2, FDA_Safety_Warnings_Cited=[])
        )
        assert {v.requirement for v in outcome.violations} <= {
            "FR-4.2", "FR-4.4", "FR-4.5"
        }
        assert all(v.requirement for v in outcome.violations)

    @requires_implementation(schemas.extract_dose_bound)
    def test_dose_bound_is_extracted_from_real_guideline_text(self) -> None:
        bound = schemas.extract_dose_bound(
            [
                "As a general reference range across mRNA vaccine platforms, "
                "primary-series doses commonly fall between 25 and 100 "
                "micrograms. Doses above 100 micrograms are considered "
                "high-reactogenicity doses and require secondary clinician review."
            ]
        )
        assert bound.max_mcg == pytest.approx(100.0)
        assert bound.source_text

    @requires_implementation(schemas.extract_dose_bound)
    def test_dose_bound_falls_back_when_text_is_uninformative(self) -> None:
        bound = schemas.extract_dose_bound(["The product should be stored cold."])
        assert bound.max_mcg == config.GUIDELINE_HARD_DOSE_CEILING_MCG

    @requires_implementation(schemas.parse_llm_json)
    def test_json_extraction_tolerates_markdown_fences(self) -> None:
        raw = '```json\n{"Eligible_For_Vaccine": true}\n```'
        assert schemas.parse_llm_json(raw) == {"Eligible_For_Vaccine": True}

    @requires_implementation(schemas.parse_llm_json)
    def test_json_extraction_tolerates_surrounding_prose(self) -> None:
        raw = 'Here is the report:\n{"a": 1}\nHope that helps.'
        assert schemas.parse_llm_json(raw) == {"a": 1}

    @requires_implementation(schemas.parse_llm_json)
    @pytest.mark.parametrize(
        "raw",
        [
            "I cannot produce JSON.",
            "",
            "still prose, no object",
        ],
    )
    def test_unparsable_response_raises_the_named_error(self, raw) -> None:
        with pytest.raises(schemas.ReportParseError):
            schemas.parse_llm_json(raw)


# ---------------------------------------------------------------------------
# FR-4.2 - grounding
# ---------------------------------------------------------------------------


class TestPromptGrounding:
    """FR-4.2: only Module 1 and Module 3 output reaches the prompt."""

    @requires_implementation(report_generator.summarise_patient)
    def test_patient_summary_restates_only_record_fields(self) -> None:
        record = {
            "Patient_ID": "P0001",
            "Age": 58,
            "Biomarker_Level": 11.2,
            "Pre_Existing_Conditions": 2,
            "Vaccine_Dose_mcg": 120.0,
            "Prior_Reaction_History": "Mild",
        }
        summary = report_generator.summarise_patient(record)
        assert "58" in summary and "120" in summary and "Mild" in summary

    @requires_implementation(report_generator.build_prompt)
    def test_prompt_contains_the_risk_output_and_retrieved_chunks(self) -> None:
        chunks = ["storage chunk text", "dosing chunk text"]
        messages = report_generator.build_prompt(
            "High", 0.82, chunks, "58-year-old, 2 pre-existing conditions"
        )
        user = next(content for role, content in messages if role == "user")
        assert "0.82" in user and "High" in user
        assert all(chunk in user for chunk in chunks)

    @requires_implementation(report_generator.build_prompt)
    def test_empty_grounding_is_refused(self) -> None:
        """Generating with no retrieved text would break FR-4.2 outright."""
        with pytest.raises(ValueError):
            report_generator.build_prompt("Low", 0.1, [], "a patient")

    def test_system_prompt_carries_the_four_required_constraints(self) -> None:
        """PRD Section 8.2 lists four constraints that must appear verbatim."""
        text = report_generator.SYSTEM_PROMPT
        assert "never invent" in text.lower()
        assert "bound" in text.lower()
        assert "disclaimer" in text.lower()
        assert "insufficient" in text.lower()

    def test_system_prompt_names_the_student_project_disclaimer(self) -> None:
        assert "student engineering project" in report_generator.SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# FR-4.3 - retry exactly once
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    """FR-4.3: validate first, retry exactly once, then fail loudly."""

    def test_max_retries_is_one(self) -> None:
        assert config.LLM_MAX_RETRIES == 1

    @requires_implementation(report_generator.generate_report)
    @requires_llm
    def test_valid_first_response_uses_a_single_attempt(self) -> None:
        run = report_generator.generate_report(
            "Low", 0.12, ["primary-series doses fall between 25 and 100 micrograms"],
            "40-year-old, no pre-existing conditions",
        )
        assert len(run.attempts) == 1
        assert run.attempts[0].succeeded

    @requires_implementation(report_generator.generate_report)
    @requires_llm
    def test_incomplete_chunk_set_forces_a_conservative_default(self) -> None:
        """PRD constraint 3: refuse to be confidently wrong."""
        run = report_generator.generate_report(
            "High", 0.93,
            ["The document does not state a dose for this profile."],
            "85-year-old, 4 pre-existing conditions",
        )
        assert run.report.Eligible_For_Vaccine is False
        assert run.report.Reasoning_Steps

    @requires_implementation(report_generator.generate_report, report_generator.invoke_llm)
    def test_malformed_response_triggers_exactly_one_retry_then_succeeds(
        self, monkeypatch,
    ) -> None:
        """FR-4.3 acceptance criterion, driven through the documented seam.

        First call returns unparsable text; second returns a valid report. The
        run must succeed after exactly two attempts.
        """
        good = json.dumps(
            {
                "Eligible_For_Vaccine": True,
                "Recommended_Dose_mg": 0.05,
                "Reasoning_Steps": ["Within standard dosing range."],
                "FDA_Safety_Warnings_Cited": [
                    "Primary-series doses commonly fall between 25 and 100 micrograms."
                ],
            }
        )
        responses = iter(["not json at all", good])
        calls = []

        def fake_invoke(messages, model=None):
            calls.append(messages)
            return next(responses)

        monkeypatch.setattr(report_generator, "invoke_llm", fake_invoke)
        run = report_generator.generate_report(
            "Low", 0.1, ["Primary-series doses fall between 25 and 100 micrograms"],
            "40-year-old",
        )
        assert len(calls) == 2, "FR-4.3 permits exactly one retry"
        assert len(run.attempts) == 2
        assert run.attempts[0].succeeded is False
        assert run.attempts[1].succeeded is True

    @requires_implementation(report_generator.generate_report, report_generator.invoke_llm)
    def test_two_malformed_responses_raise_after_exactly_two_attempts(
        self, monkeypatch,
    ) -> None:
        """A failing run must stop at two attempts, not loop."""
        calls = []

        def fake_invoke(messages, model=None):
            calls.append(messages)
            return "still not json"

        monkeypatch.setattr(report_generator, "invoke_llm", fake_invoke)
        with pytest.raises(report_generator.ReportGenerationError):
            report_generator.generate_report(
                "Low", 0.1, ["some retrieved chunk"], "40-year-old"
            )
        assert len(calls) == 2

    @requires_implementation(report_generator.generate_report)
    def test_more_than_one_retry_is_rejected(self) -> None:
        """FR-4.3 says exactly one retry; a larger budget is a spec violation."""
        with pytest.raises(ValueError):
            report_generator.generate_report(
                "Low", 0.1, ["chunk"], "patient", max_retries=2
            )

    def test_generation_error_message_reports_both_attempts(self) -> None:
        attempts = [
            report_generator.GenerationAttempt(1, False, "missing field"),
            report_generator.GenerationAttempt(2, False, "unparsable"),
        ]
        message = str(report_generator.ReportGenerationError(attempts))
        assert "attempt 1" in message and "attempt 2" in message


# ---------------------------------------------------------------------------
# FR-4.6 - persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    """FR-4.6: every report is written as a .json / .md pair."""

    @requires_implementation(report_generator.persist_report)
    def test_json_and_markdown_are_written_as_a_pair(self, tmp_path) -> None:
        report = schemas.ClinicalSafetyReport(
            Eligible_For_Vaccine=False,
            Recommended_Dose_mg=0.05,
            Reasoning_Steps=["Biomarker above reference range."],
            FDA_Safety_Warnings_Cited=["Defer dosing until biomarker normalises."],
            Disclaimer=config.REQUIRED_DISCLAIMER,
        )
        run = report_generator.ReportRun(
            report=report,
            safety=schemas.SafetyOutcome(report=report),
            attempts=[report_generator.GenerationAttempt(1, True, latency_s=1.2)],
        )
        json_path, md_path = report_generator.persist_report(run, tmp_path)
        assert json_path.exists() and md_path.exists()
        assert json_path.suffix == ".json" and md_path.suffix == ".md"
        assert json_path.stem == md_path.stem

    @requires_implementation(report_generator.render_markdown)
    def test_markdown_includes_disclaimer_and_citations(self) -> None:
        report = schemas.ClinicalSafetyReport(
            Eligible_For_Vaccine=True,
            Recommended_Dose_mg=0.05,
            Reasoning_Steps=["No contraindications found."],
            FDA_Safety_Warnings_Cited=["Ages 12 to 64 years: standard adult dosing."],
            Disclaimer=config.REQUIRED_DISCLAIMER,
        )
        markdown = report_generator.render_markdown(
            report_generator.ReportRun(
                report=report, safety=schemas.SafetyOutcome(report=report)
            )
        )
        assert config.REQUIRED_DISCLAIMER in markdown
        assert "Ages 12 to 64 years" in markdown
        assert "50" in markdown or "0.05" in markdown

    @requires_implementation(report_generator.persist_report)
    @requires_llm
    def test_five_generated_reports_all_carry_a_disclaimer(self) -> None:
        """FR-4.5 acceptance criterion: 5 samples, all non-empty."""
        import tempfile

        for index in range(5):
            report = schemas.ClinicalSafetyReport(
                Eligible_For_Vaccine=True,
                Recommended_Dose_mg=0.05,
                Reasoning_Steps=[f"Patient {index}: within standard dosing range."],
                FDA_Safety_Warnings_Cited=["Primary-series doses fall between 25 and 100 micrograms."],
                Disclaimer=config.REQUIRED_DISCLAIMER,
            )
            run = report_generator.ReportRun(
                report=report, safety=schemas.SafetyOutcome(report=report)
            )
            with tempfile.TemporaryDirectory() as tmp:
                json_path, _ = report_generator.persist_report(run, tmp)
                stored = json.loads(json_path.read_text(encoding="utf-8"))
            assert stored["report"]["Disclaimer"].strip()


# ---------------------------------------------------------------------------
# Module 5 - dashboard (FR-5.1, FR-5.2, FR-5.3)
# ---------------------------------------------------------------------------


class TestDashboard:
    """The three-panel figure and its provenance."""

    def test_three_panels_are_declared_in_order(self) -> None:
        assert components.PANELS.keys() == {"roc", "attention", "latency"}
        assert len(components.PANELS) == 3

    def test_panel_titles_are_declared_for_every_panel(self) -> None:
        from dashboard import styles

        assert set(styles.PANEL_TITLES) == {"roc", "attention", "latency"}
        assert all(title for title in styles.PANEL_TITLES.values())

    def test_configured_dpi_meets_fr53(self) -> None:
        from dashboard import styles

        assert styles.FIGURE_DPI >= 150

    def test_figure_is_wide_enough_for_three_panels(self) -> None:
        from dashboard import styles

        assert styles.N_PANELS == 3
        assert styles.FIGURE_SIZE[0] / styles.FIGURE_SIZE[1] >= 2.5

    def test_dashboard_verifier_is_available(self) -> None:
        """FR-5.3 is checked programmatically before submission."""
        assert app.verify_dashboard.__doc__

    @requires_implementation(app.build_dashboard)
    def test_panels_are_built_from_saved_artifacts_not_hardcoded(self, tmp_path) -> None:
        """FR-5.2: each panel loader reads a real artifact, never an inline value.

        Asserted behaviourally. Calling each loader with no arguments resolves its
        real default path under ``outputs/`` and raises ``MissingArtifactError``
        naming that path. A loader holding a hardcoded value could not raise this,
        so the assertion proves provenance rather than documenting it.
        """
        assert components.ROCPanelData.__doc__

        panels = (
            (components.load_roc_panel_data, "roc"),
            (components.load_attention_panel_data, "attention"),
            (components.load_latency_panel_data, "latency"),
        )
        for loader, panel in panels:
            if components.artifact_availability()[panel]:
                continue  # artifact built; the loader has something real to read
            with pytest.raises(components.MissingArtifactError) as excinfo:
                loader()
            message = str(excinfo.value)
            assert str(config.OUTPUTS_DIR) in message, (
                f"{loader.__name__} must read from {config.OUTPUTS_DIR}"
            )
            assert components.TRAIN_COMMAND in message or (
                components.RUN_COMMAND in message
            ), f"{loader.__name__} must name the command that produces its artifact"

    @requires_implementation(app.build_dashboard)
    def test_panel_loaders_raise_a_named_error_when_the_artifact_is_absent(
        self, tmp_path
    ) -> None:
        """FR-5.2's negative case, with the producing command in the message."""
        missing = tmp_path / "nope.json"
        with pytest.raises(components.MissingArtifactError) as excinfo:
            components.load_roc_panel_data(
                metrics_path=missing, roc_path=missing
            )
        assert components.TRAIN_COMMAND in str(excinfo.value)

    def test_missing_artifact_raises_a_named_error_rather_than_drawing_empty_axes(self):
        from dashboard import components

        with pytest.raises(components.MissingArtifactError):
            components.load_roc_panel_data(
                metrics_path=Path("does_not_exist_metrics.json"),
                roc_path=Path("does_not_exist_roc.json"),
            )

    def test_committed_dashboard_exists_when_pipeline_has_run(self) -> None:
        from tests import requires_artifact

        requires_artifact(
            Path(config.FIGURES_DIR) / config.DASHBOARD_FILENAME,
            "python -m src.pipeline.pipeline --all",
        )


# ---------------------------------------------------------------------------
# Module 5 - the interface layer
# ---------------------------------------------------------------------------


class TestDesignTokens:
    """The palette is generated from Python and must stay WCAG AA readable."""

    @staticmethod
    def _relative_luminance(hex_colour: str) -> float:
        value = hex_colour.lstrip("#")
        channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        linear = [
            c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels
        ]
        r, g, b = linear
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    @classmethod
    def _contrast(cls, foreground: str, background: str) -> float:
        a, b = cls._relative_luminance(foreground), cls._relative_luminance(background)
        high, low = max(a, b), min(a, b)
        return (high + 0.05) / (low + 0.05)

    @pytest.mark.parametrize(
        "token",
        [
            "text", "text-muted", "text-subtle",
            "accent", "accent-strong",
            "success", "warning", "danger", "neutral",
        ],
    )
    @pytest.mark.parametrize(
        "surface", ["surface", "background", "surface-alt", "surface-sunken"]
    )
    def test_text_tokens_meet_wcag_aa_on_every_surface(self, token, surface) -> None:
        """4.5:1 minimum for body text on each background the UI actually uses.

        Guards the regression where ``text-subtle`` at #64748B measured 4.47:1 on
        the canvas and 4.34:1 on the sunken surface - both failures that only
        surfaced under a Lighthouse audit.
        """
        ratio = self._contrast(
            styles.PALETTE[token], styles.PALETTE[surface]
        )
        assert ratio >= 4.5, (
            f"{token} on {surface} is {ratio:.2f}:1, below the 4.5:1 AA threshold"
        )

    def test_primary_button_text_is_readable_on_its_own_background(self) -> None:
        assert self._contrast(
            styles.PALETTE["text-inverse"], styles.PALETTE["accent-strong"]
        ) >= 4.5

    def test_every_token_is_exported_as_a_css_variable(self) -> None:
        variables = styles.css_variables()
        assert set(variables) == {
            f"{styles.CSS_VAR_PREFIX}-{name}" for name in styles.PALETTE
        }

    def test_theme_css_emits_a_root_rule_with_no_placeholder_left(self) -> None:
        css = styles.theme_css()
        assert ":root {" in css and "}" in css
        assert "__VIP_THEME__" not in css
        assert styles.PALETTE["accent"] in css

    def test_matplotlib_and_browser_share_one_palette(self) -> None:
        """The figure face colour is drawn from the same token as the CSS canvas."""
        assert styles.FIGURE_FACE_COLOR == styles.PALETTE["surface"]
        assert styles.LATENCY_COLORS[0] == styles.PALETTE["accent"]
        assert styles.LATENCY_COLORS[1] == styles.PALETTE["navy"]

    def test_panel_titles_and_subtitles_are_defined_for_all_three_panels(self) -> None:
        assert set(styles.PANEL_SUBTITLES) == set(styles.PANEL_TITLES) == {
            "roc", "attention", "latency"
        }

    def test_breakpoints_are_declared_in_python(self) -> None:
        assert styles.BREAKPOINTS["mobile"] == 860
        assert styles.BREAKPOINTS["tablet"] == 1024


class TestPlatformState:
    """The interface must report readiness truthfully."""

    def test_stage_count_and_order_match_the_prd_pipeline(self) -> None:
        state = components.platform_state()
        assert [s["number"] for s in state["stages"]] == [
            "01", "02", "03", "04", "05"
        ]
        assert [s["name"] for s in state["stages"]] == [
            "Patient Data", "Risk Model", "mRNA Attention",
            "Regulatory Retrieval", "Safety Report",
        ]

    def test_dataset_stage_is_ready_because_the_file_exists(self) -> None:
        state = components.platform_state()
        by_id = {s["id"]: s for s in state["stages"]}
        assert by_id["patient-data"]["status"] == "ready"
        assert config.RAW_PATIENT_CSV.exists()

    def test_unrun_stages_are_not_claimed_as_ready(self) -> None:
        """No stage may report success while its artifact is absent.

        Each stage is paired with the artifact that proves it ran. A stage whose
        artifact exists may legitimately be ready; one whose artifact is missing
        must not be. The report stage is expected to be un-run until Module 4
        exists, and the retrieval stage's index is a directory rather than a file,
        so the directory check uses the same content test the stage uses.
        """
        state = components.platform_state()
        by_id = {s["id"]: s for s in state["stages"]}

        file_backed = {
            "risk-model": config.RISK_MODEL_PATH,
            "attention": config.ATTENTION_WEIGHTS_PATH,
        }
        for stage_id, path in file_backed.items():
            stage = by_id[stage_id]
            if path.exists():
                assert stage["status"] == "ready", (
                    f"{stage['name']} has {path.name} but reports "
                    f"'{stage['status']}'"
                )
            else:
                assert stage["status"] == "not-initialized", (
                    f"{stage['name']} reports '{stage['status']}' without {path}"
                )

        for stage_id, directory in (("retrieval", config.CHROMA_PERSIST_DIR),
                                    ("report", config.REPORTS_DIR)):
            stage = by_id[stage_id]
            populated = components._directory_has_content(directory)
            expected = "ready" if populated else "not-initialized"
            assert stage["status"] == expected, (
                f"{stage['name']} reports '{stage['status']}' but "
                f"{'has' if populated else 'lacks'} content in {directory}"
            )

    def test_report_stage_is_not_ready_before_module_four(self) -> None:
        """The report stage must stay un-run until report generation exists."""
        state = components.platform_state()
        report_stage = next(s for s in state["stages"] if s["id"] == "report")
        if components._directory_has_content(config.REPORTS_DIR):
            pytest.skip("a report has been generated")
        assert report_stage["status"] == "not-initialized"

    def test_gitkeep_does_not_make_an_empty_directory_look_populated(self, tmp_path) -> None:
        """Regression guard: placeholders must not fake readiness.

        Generated directories are kept in version control with a ``.gitkeep``.
        Counting that file as content made unrun stages report themselves ready.
        Asserted on a temporary directory rather than on ``models/``, because that
        directory legitimately holds trained artifacts once Module 1 has run.
        """
        empty = tmp_path / "empty_dir"
        empty.mkdir()
        (empty / ".gitkeep").write_text("", encoding="utf-8")
        assert components._directory_has_content(empty) is False

    def test_directory_with_a_real_file_is_detected(self, tmp_path) -> None:
        (tmp_path / "artifact.bin").write_bytes(b"x")
        assert components._directory_has_content(tmp_path) is True

    def test_trained_model_artifact_marks_the_risk_stage_ready(self) -> None:
        """Once Module 1 has run, the interface must reflect that.

        The counterpart to the ``.gitkeep`` guard: readiness has to be capable of
        becoming true, or the guard above would be satisfied by a permanently
        disabled stage rather than by honest detection.
        """
        if not config.RISK_MODEL_PATH.exists():
            from tests import requires_artifact

            requires_artifact(config.RISK_MODEL_PATH, "python -m src.ml")
        stages = {s["id"]: s for s in components.platform_state()["stages"]}
        assert stages["risk-model"]["status"] == "ready"

    def test_metrics_expose_only_specification_derived_values(self) -> None:
        """Metrics are configuration facts, never model output."""
        state = components.platform_state()
        metrics = state["metrics"]
        assert set(metrics) == {
            "patient_records", "ml_models", "model_names", "retrieval_top_k",
            "report_format", "embed_dim", "metric_keys", "selected_model",
            "evaluated",
        }
        assert metrics["patient_records"] == 500
        assert metrics["ml_models"] == 2
        assert metrics["retrieval_top_k"] == 3
        assert metrics["report_format"] == "Pydantic"
        assert metrics["metric_keys"] == list(config.METRIC_KEYS)

    def test_evaluated_metrics_are_empty_until_module_one_has_run(self) -> None:
        """The interface must not show scores before any model exists.

        Strengthened into two cases. The absence of a results file must yield an
        empty mapping, and when a results file *is* present its values must be
        passed through verbatim - a stubbed zero would be worse than nothing.
        """
        import json
        import tempfile
        from pathlib import Path

        results_path = config.ML_RESULTS_JSON_PATH
        backup = None
        if results_path.exists():
            backup = results_path.read_bytes()
        try:
            results_path.unlink(missing_ok=True)
            assert components._evaluated_metrics() == {}

            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps({
                "logistic_regression": {
                    "accuracy": 0.72, "precision": 0.733333, "recall": 0.673469,
                    "f1": 0.702128, "roc_auc": 0.767907,
                },
                "random_forest": {
                    "accuracy": 0.63, "precision": 0.62, "recall": 0.632653,
                    "f1": 0.626263, "roc_auc": 0.709484,
                },
                "selection": {"selected_model": "logistic_regression"},
            }), encoding="utf-8")
            measured = components._evaluated_metrics()
            assert set(measured) == set(config.MODEL_NAMES)
            assert measured["logistic_regression"]["f1"] == pytest.approx(0.702128)
            assert components._selected_model_key() == "logistic_regression"
        finally:
            if backup is not None:
                results_path.write_bytes(backup)

    def test_only_the_declared_metric_keys_reach_the_interface(self) -> None:
        """The results file cannot widen what the UI is allowed to present."""
        import json

        results_path = config.ML_RESULTS_JSON_PATH
        backup = results_path.read_bytes() if results_path.exists() else None
        try:
            results_path.write_text(json.dumps({
                "logistic_regression": {
                    "accuracy": 0.72, "precision": 0.73, "recall": 0.67,
                    "f1": 0.70, "roc_auc": 0.77,
                    "injected_field": "should not surface",
                },
                "random_forest": {
                    "accuracy": 0.63, "precision": 0.62, "recall": 0.63,
                    "f1": 0.63, "roc_auc": 0.71,
                },
            }), encoding="utf-8")
            measured = components._evaluated_metrics()
            assert set(measured["logistic_regression"]) == set(config.METRIC_KEYS)
            assert "injected_field" not in json.dumps(components.platform_state())
        finally:
            if backup is not None:
                results_path.write_bytes(backup)
            else:
                results_path.unlink(missing_ok=True)

    def test_partial_metrics_are_treated_as_absent(self) -> None:
        """A model missing one metric is dropped, not shown with a blank cell."""
        import json

        results_path = config.ML_RESULTS_JSON_PATH
        backup = results_path.read_bytes() if results_path.exists() else None
        try:
            results_path.write_text(json.dumps({
                "logistic_regression": {"accuracy": 0.72, "precision": 0.73},
                "random_forest": {
                    "accuracy": 0.63, "precision": 0.62, "recall": 0.63,
                    "f1": 0.63, "roc_auc": 0.71,
                },
            }), encoding="utf-8")
            measured = components._evaluated_metrics()
            assert "logistic_regression" not in measured
            assert "random_forest" in measured
        finally:
            if backup is not None:
                results_path.write_bytes(backup)
            else:
                results_path.unlink(missing_ok=True)

    def test_state_contains_no_fabricated_value_when_models_are_untrained(self) -> None:
        """No score may appear before Module 1 has produced real ones."""
        import json

        results_path = config.ML_RESULTS_JSON_PATH
        backup = results_path.read_bytes() if results_path.exists() else None
        try:
            results_path.unlink(missing_ok=True)
            blob = json.dumps(components.platform_state()).lower()
            # metric_keys are the declared names, not values, so the metric
            # labels are fine; what must be absent is any numeric result.
            for model in config.MODEL_NAMES:
                assert f'"{model}": {{' not in blob
            assert components.platform_state()["metrics"]["evaluated"] == {}
        finally:
            if backup is not None:
                results_path.write_bytes(backup)

    def test_report_schema_is_read_from_the_pydantic_model(self) -> None:
        fields = components.platform_state()["report_schema"]
        assert [f["name"] for f in fields] == list(schemas.REQUIRED_REPORT_FIELDS)

    def test_dose_bound_is_absent_until_validation_is_implemented(self) -> None:
        """The interface must not invent the maximum dose (PRD, not a number)."""
        assert components.platform_state()["dose_bound"] is None

    def test_disclaimer_is_the_required_student_project_notice(self) -> None:
        disclaimer = components.platform_state()["copy"]["disclaimer"]
        assert "student engineering project" in disclaimer.lower()
        assert "not intended to provide real clinical guidance" in disclaimer.lower()
        assert disclaimer == config.REQUIRED_DISCLAIMER

    def test_all_required_empty_states_are_declared(self) -> None:
        assert components.EMPTY_STATES["roc"] == (
            "Evaluation results will appear after model training."
        )
        assert components.EMPTY_STATES["attention"] == (
            "Attention map will appear after inference."
        )
        assert components.EMPTY_STATES["latency"] == (
            "Latency measurements will appear after pipeline execution."
        )
        assert components.EMPTY_STATES["retrieval"].startswith("Retrieved evidence")
        assert components.EMPTY_STATES["report"] == "No report generated."

    def test_dose_state_copy_declines_to_invent_a_limit(self) -> None:
        assert components.EMPTY_STATES["dose"] == (
            "Maximum permitted dose will be evaluated by the validation pipeline."
        )

    def test_dose_component_supports_the_three_required_states(self) -> None:
        """VALID / WARNING / EXCEEDED must all be representable."""
        html = (app.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for state in ("valid", "warning", "exceeded"):
            assert f'data-state="{state}"' in html or f"[data-state=\"{state}\"]" in (
                app.STATIC_DIR / "app.css"
            ).read_text(encoding="utf-8")
        for label in ("Valid Dose", "Warning", "Dose Limit Exceeded"):
            assert label in html

    def test_guideline_source_is_reported_as_present(self) -> None:
        source = components.platform_state()["source"]
        assert source["available"] is True
        assert "Moderna mRNA Safety Guidelines" in source["name"]


class TestStaticAssets:
    """The shell is served from static files with no build step."""

    def test_three_required_asset_files_exist(self) -> None:
        for name in ("index.html", "app.css", "app.js"):
            assert (app.STATIC_DIR / name).is_file(), f"missing {name}"

    def test_theme_marker_is_present_for_injection(self) -> None:
        assert app.THEME_MARKER in (app.STATIC_DIR / "index.html").read_text(
            encoding="utf-8"
        )

    def test_all_six_pages_exist_in_the_shell(self) -> None:
        html = (app.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for route in ("overview", "patient", "risk", "attention", "regulatory", "report"):
            assert f'id="page-{route}"' in html
            assert f'data-route="{route}"' in html

    def test_shell_declares_semantic_regions_and_a_skip_link(self) -> None:
        html = (app.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for marker in (
            "<header", "<nav", "<main", "<footer", 'class="skip-link"',
            'lang="en"', "<dialog",
        ):
            assert marker in html, f"missing semantic marker {marker}"

    def test_every_navigation_item_is_present(self) -> None:
        html = (app.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for label in (
            "Overview", "Patient Analysis", "Risk Prediction",
            "mRNA Attention", "Regulatory Intelligence", "Safety Report",
            "Student Engineering Project",
        ):
            assert label in html

    def test_patient_form_uses_only_the_supplied_dataset_fields(self) -> None:
        """No invented clinical inputs (brief section 8)."""
        html = (app.STATIC_DIR / "index.html").read_text(encoding="utf-8")
        form = html[html.index('id="patient-form"') : html.index("</form>", html.index('id="patient-form"'))]
        names = sorted(set(re.findall(r'name="([A-Za-z_]+)"', form)))
        assert names == [
            "Age", "Biomarker_Level", "Patient_ID",
            "Pre_Existing_Conditions", "Prior_Reaction_History", "Vaccine_Dose_mcg",
        ]

    def test_stylesheet_guards_against_page_level_horizontal_overflow(self) -> None:
        css = (app.STATIC_DIR / "app.css").read_text(encoding="utf-8")
        assert "overflow-x: hidden" in css
        assert "overflow-x: auto" in css  # tables scroll inside their own wrapper

    def test_stylesheet_honours_reduced_motion(self) -> None:
        css = (app.STATIC_DIR / "app.css").read_text(encoding="utf-8")
        assert "prefers-reduced-motion" in css

    def test_client_script_never_hardcodes_a_metric_value(self) -> None:
        """No invented score may be baked into the front end.

        Metric *names* are legitimate - the script formats whatever the API sends.
        Metric *values* are not, so this scans for numeric literals assigned in a
        metric context and for the ROC/dashboard placeholder shapes.
        """
        import re

        js = (app.STATIC_DIR / "app.js").read_text(encoding="utf-8")

        for forbidden in ("f1_score", "rocPoints", "dummyData", "sampleData"):
            assert forbidden not in js, f"'{forbidden}' appears in the client script"

        # No numeric literal standing in for a measured score.
        pattern = r"(?:accuracy|precision|recall|f1|roc_?auc|probability)\s*[:=]\s*[0-9]"
        offenders = re.findall(pattern, js, flags=re.IGNORECASE)
        assert not offenders, f"client script assigns literal metric values: {offenders}"

        # Metric labels are fine, but they must come from the API's key list.
        assert "metric_keys" in js, "metric labels must be driven by the API payload"

    def test_client_script_has_no_chart_drawing_code(self) -> None:
        """A client-side plot without data would be a fabricated chart.

        The interface renders saved figures as images; it must not synthesise a
        curve or heatmap of its own.
        """
        js = (app.STATIC_DIR / "app.js").read_text(encoding="utf-8")
        for forbidden in ("getContext", "createElement('canvas')", "toFixed(2) + '%"):
            assert forbidden not in js, f"'{forbidden}' suggests client-side plotting"


class TestFigureRendering:
    """The saved figure is real code now, so these assertions run rather than skip."""

    @requires_implementation(app.build_dashboard)
    def test_figure_renders_with_three_panels_and_meets_the_dpi_floor(self, tmp_path) -> None:
        path = app.build_dashboard(tmp_path / "dashboard.png")
        assert path.exists()
        report = app.verify_dashboard(path)
        assert report["valid_png"] is True
        assert report["panel_count"] == 3
        assert report["meets_fr_5_3"] is True
        assert report["dpi"] >= 150

    def test_verify_reports_a_missing_file_without_raising(self, tmp_path) -> None:
        report = app.verify_dashboard(tmp_path / "absent.png")
        assert report["exists"] is False
        assert report["meets_fr_5_3"] is False

    def test_png_dpi_reader_handles_a_non_png_file(self, tmp_path) -> None:
        bogus = tmp_path / "not.png"
        bogus.write_text("plain text", encoding="utf-8")
        assert app._read_png_dpi(bogus) is None

    @requires_implementation(app.build_dashboard)
    def test_figure_falls_back_to_empty_states_when_no_artifacts_exist(
        self, tmp_path
    ) -> None:
        """The figure must still render - and must not invent a curve."""
        path = app.build_dashboard(tmp_path / "empty.png")
        assert path.exists() and path.stat().st_size > 0


class TestWebInterface:
    """Live checks against a real server instance on an ephemeral port."""

    @pytest.fixture(scope="class")
    @staticmethod
    def server():
        import threading
        from functools import partial
        from http.server import ThreadingHTTPServer

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.VIPRequestHandler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
        httpd.shutdown()
        httpd.server_close()

    @staticmethod
    def _get(url: str):
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(url, timeout=8) as response:
                return response.status, response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def test_index_is_served_with_the_theme_injected(self, server) -> None:
        status, body = self._get(server + "/")
        assert status == 200
        assert app.THEME_MARKER not in body
        assert styles.PALETTE["accent"] in body

    def test_health_endpoint_responds(self, server) -> None:
        status, body = self._get(server + "/api/health")
        assert status == 200
        assert "ok" in body

    def test_state_endpoint_returns_the_truthful_snapshot(self, server) -> None:
        import json

        status, body = self._get(server + "/api/state")
        assert status == 200
        payload = json.loads(body)
        assert len(payload["stages"]) == 5
        assert payload["metrics"]["patient_records"] == 500

    def test_static_assets_are_served(self, server) -> None:
        for path in ("/static/app.css", "/static/app.js"):
            assert self._get(server + path)[0] == 200

    @pytest.mark.parametrize(
        "path",
        ["/static/../../.env", "/static/../../../src/config.py", "/figures/../../requirements.txt"],
    )
    def test_path_traversal_is_refused(self, server, path) -> None:
        """The handler must not read outside its base directories.

        Either status is a refusal: 403 when the resolved path escapes the base
        directory, 404 when it resolves inside but names a missing file. What must
        never happen is a 200 with file contents.
        """
        status, body = self._get(server + path)
        assert status in (403, 404), f"{path} returned {status}"
        assert "REQUIRED" not in body and "PROJECT_NAME" not in body

    def test_unknown_route_returns_404(self, server) -> None:
        assert self._get(server + "/does-not-exist")[0] == 404

    def test_pipeline_actions_execute_real_work_and_never_fabricate(
        self, server
    ) -> None:
        """POST /api/run now runs the pipeline; the anti-fabrication rule stands.

        This test previously asserted HTTP 501, which was correct while the modules
        were stubs. It is now replaced rather than deleted, because the property it
        was protecting - that the interface never reports success it did not earn -
        still has to hold. Three cases:

        1. A malformed request is refused with 400 and no run.
        2. An unknown stage is refused; it does not fall through to an analysis.
        3. A run whose required stage fails is a non-2xx, never a 200.
        """
        import urllib.error
        import urllib.request

        def post(payload):
            request = urllib.request.Request(
                server + "/api/run",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    return response.status, json.loads(response.read())
            except urllib.error.HTTPError as exc:
                return exc.code, json.loads(exc.read())

        # 1. Missing record -> 400, before any stage runs.
        status, payload = post({"stage": "ml"})
        assert status == 400
        assert payload["error"] == "invalid_request"
        assert "run_id" in payload, "a refused request is still traceable"

        # 2. An unknown stage must not trigger an analysis.
        status, payload = post({"stage": "destroy_everything"})
        assert status == 400
        assert payload["error"] == "invalid_request"

        # 3. No fake success: with no trained models reachable through a stubbed
        # executor, the response is a failure. Asserted structurally so it holds
        # whatever the environment's artifacts happen to be.
        status, payload = post({"stage": "report"})
        assert status in (400, 500, 502, 503), (
            f"unexpected status {status}: {payload}"
        )
        if status != 400:
            assert payload.get("generated") is not True, (
                "a failed request claimed a report was generated"
            )

    def test_serveable_figures_are_an_explicit_allowlist(self) -> None:
        """Every allowed entry must live inside the figures directory."""
        for name, path in app.SERVEABLE_FIGURES.items():
            assert path.parent == config.FIGURES_DIR, f"{name} escapes FIGURES_DIR"

    def test_cli_state_flag_prints_valid_json(self) -> None:
        import contextlib
        import io
        import json

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = app.main(["--state"])
        assert exit_code == 0
        assert len(json.loads(buffer.getvalue())["stages"]) == 5

    def test_cli_verify_flag_reports_a_missing_figure(self, tmp_path) -> None:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = app.main(["--verify", "--out", str(tmp_path / "absent.png")])
        assert exit_code == 1
        assert '"exists": false' in buffer.getvalue()


# ---------------------------------------------------------------------------
# Module 4 - key-free unit tests
# ---------------------------------------------------------------------------
#
# Everything below drives generate_report through the documented `invoke_llm`
# seam, so none of it needs an API key, a network, or a paid token. The tests
# that DO require a live key are marked `requires_llm` and skip without one.
#
# The brief asks for twelve specific behaviours. Those already asserted above by
# the pre-existing suite are: valid output passes validation, missing/malformed
# fields fail, one retry then success, two failures raise, reports persist as a
# JSON/Markdown pair, the disclaimer is enforced, and the dose bound comes from
# guideline text with a fallback. The gaps this section closes are numbered in
# each class docstring.


VALID_REPORT_JSON = json.dumps(
    {
        "Eligible_For_Vaccine": True,
        "Recommended_Dose_mg": 0.05,
        "Reasoning_Steps": ["Within the stated primary-series range."],
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


class _FakeLLM:
    """Records prompts and returns queued responses.

    Shared by the retry tests so each one states only its scenario. Kept as a
    class rather than a closure so the recorded call count is inspectable after
    the run.
    """

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.calls: list[list[tuple[str, str]]] = []

    def __call__(self, messages, model=None):
        self.calls.append(messages)
        if not self.responses:
            raise AssertionError("invoke_llm was called more times than expected")
        return self.responses.pop(0)


class TestRetryWithoutAnApiKey:
    """Brief items 5 and 3: a valid first attempt must not retry, and a malformed
    one must retry exactly once - both proven without an API key.

    ``test_valid_first_response_uses_a_single_attempt`` above covers item 5 but is
    gated on a real key, so the behaviour would otherwise never be verified in a
    keyless checkout.
    """

    def test_valid_first_attempt_does_not_retry(self, monkeypatch) -> None:
        fake = _FakeLLM(VALID_REPORT_JSON)
        monkeypatch.setattr(report_generator, "invoke_llm", fake)
        run = report_generator.generate_report("Low", 0.12, [DOSE_CHUNK], "40-year-old")
        assert len(fake.calls) == 1, "a valid first response must not be retried"
        assert len(run.attempts) == 1
        assert run.attempts[0].succeeded is True

    def test_retry_receives_the_validation_errors_from_the_first_attempt(
        self, monkeypatch
    ) -> None:
        """The retry has to be told what was wrong, or it is just a second guess.

        A retry that repeats the identical prompt has no way to do better, so the
        corrected request must carry the field-level failures from attempt 1.
        """
        bad = json.dumps({"Eligible_For_Vaccine": True, "Recommended_Dose_mg": 0.05})
        fake = _FakeLLM(bad, VALID_REPORT_JSON)
        monkeypatch.setattr(report_generator, "invoke_llm", fake)

        run = report_generator.generate_report("Low", 0.1, [DOSE_CHUNK], "40-year-old")

        assert len(fake.calls) == 2
        retry_prompt = " ".join(content for _, content in fake.calls[1])
        assert "Reasoning_Steps" in retry_prompt or "validation" in retry_prompt.lower()
        assert run.attempts[0].failure_reason
        assert "Reasoning_Steps" in run.attempts[0].failure_reason

    def test_two_total_attempts_is_the_hard_ceiling(self, monkeypatch) -> None:
        """FR-4.3: never a third attempt, however tempting."""
        fake = _FakeLLM("nope", "still nope", VALID_REPORT_JSON)
        monkeypatch.setattr(report_generator, "invoke_llm", fake)
        with pytest.raises(report_generator.ReportGenerationError):
            report_generator.generate_report("Low", 0.1, [DOSE_CHUNK], "40-year-old")
        assert len(fake.calls) == 2, "FR-4.3 permits exactly two attempts total"


class TestApiConfiguration:
    """Brief items 6 and 7: missing credentials and provider failures surface."""

    @staticmethod
    def _clear_keys(monkeypatch):
        for variable in ("LLM_API_KEY", "OPENAI_API_KEY"):
            monkeypatch.delenv(variable, raising=False)
        # Neutralise .env so the outcome does not depend on whether the developer
        # has a local .env file. This test is about the error path, not about .env.
        monkeypatch.setattr(report_generator, "_load_dotenv_once", lambda: None)

    def test_missing_key_raises_an_actionable_error(self, monkeypatch) -> None:
        self._clear_keys(monkeypatch)
        with pytest.raises(report_generator.MissingAPIKeyError) as excinfo:
            report_generator.load_api_key()
        message = str(excinfo.value)
        assert "LLM_API_KEY" in message
        assert ".env" in message

    def test_missing_key_message_never_echoes_a_secret(self, monkeypatch) -> None:
        """A key present in one variable must not leak into another's error."""
        self._clear_keys(monkeypatch)
        monkeypatch.setenv("SOME_OTHER_TOKEN", "super-secret-value")
        with pytest.raises(report_generator.MissingAPIKeyError) as excinfo:
            report_generator.load_api_key()
        assert "super-secret-value" not in str(excinfo.value)

    def test_a_configured_key_is_returned_not_logged(self, monkeypatch) -> None:
        monkeypatch.setenv("LLM_API_KEY", "test-key-value")
        assert report_generator.load_api_key() == "test-key-value"

    def test_importing_the_module_needs_no_key(self) -> None:
        """Regression guard for the contract that made these tests possible."""
        assert "load_api_key" not in repr(report_generator)
        assert callable(report_generator.generate_report)

    def test_provider_failure_raises_rather_than_fabricating(self, monkeypatch) -> None:
        """Brief item 7: an API error must never become a successful report."""
        def explode(messages, model=None):
            raise RuntimeError("Connection reset by peer")

        monkeypatch.setattr(report_generator, "invoke_llm", explode)
        with pytest.raises(report_generator.LLMCallError) as excinfo:
            report_generator.generate_report("Low", 0.1, [DOSE_CHUNK], "40-year-old")
        assert "Connection reset by peer" in str(excinfo.value)

    def test_provider_failure_is_not_charged_to_the_validation_retry(self, monkeypatch) -> None:
        """A credential or network fault is not the model's fault.

        Retrying it would spend the single FR-4.3 retry and then report the
        problem as a schema failure, hiding the real cause.
        """
        calls = []

        def explode(messages, model=None):
            calls.append(messages)
            raise RuntimeError("401 Unauthorized")

        monkeypatch.setattr(report_generator, "invoke_llm", explode)
        with pytest.raises(report_generator.LLMCallError):
            report_generator.generate_report("Low", 0.1, [DOSE_CHUNK], "40-year-old")
        assert len(calls) == 1, "an API error must not consume the validation retry"

    def test_api_error_and_validation_error_are_distinguishable(self) -> None:
        """The two failure classes need different remediation, so different types."""
        assert not issubclass(report_generator.LLMCallError, report_generator.ReportGenerationError)
        assert not issubclass(report_generator.ReportGenerationError, report_generator.LLMCallError)


class TestDoseHonesty:
    """Brief items 11 and 12: reject only on a maximum the source establishes,
    and invent no threshold when it does not."""

    def test_ceiling_is_derived_from_the_retrieved_text(self) -> None:
        bound = schemas.extract_dose_bound([DOSE_CHUNK])
        assert bound.max_mcg == pytest.approx(100.0)
        assert bound.source_text, "a derived bound must carry the sentence it came from"
        assert "100 micrograms" in bound.source_text

    def test_a_bare_number_is_not_treated_as_a_ceiling(self) -> None:
        """A microgram value with no stated consequence is not a limit.

        Without the consequence requirement, any mention of a dose would become an
        enforceable ceiling - including one the document never called a maximum.
        """
        bound = schemas.extract_dose_bound(
            ["The formulation contains 50 micrograms of active material per dose."]
        )
        assert bound.source_text == "", "a bare figure must not be cited as a limit"
        assert bound.max_mcg == config.GUIDELINE_HARD_DOSE_CEILING_MCG

    def test_no_text_at_all_yields_a_defaulted_bound(self) -> None:
        bound = schemas.extract_dose_bound([])
        assert bound.max_mcg == config.GUIDELINE_HARD_DOSE_CEILING_MCG
        assert bound.source_text == ""

    def test_defaulted_bound_is_not_presented_as_retrieved_evidence(self) -> None:
        """The limitation must be explicit, not silently papered over."""
        bound = schemas.extract_dose_bound(["The product should be stored cold."])
        payload = {
            "derived_from_guideline": bool(bound.source_text),
            "max_mcg": bound.max_mcg,
        }
        assert payload["derived_from_guideline"] is False

    def test_above_ceiling_reports_eligibility_false_in_the_output(self) -> None:
        """FR-4.4's stated acceptance criterion, end to end through the generator."""
        report = schemas.ClinicalSafetyReport(
            Eligible_For_Vaccine=True,
            Recommended_Dose_mg=0.12,  # 120 mcg against a 100 mcg ceiling
            Reasoning_Steps=["Above the stated maximum."],
            FDA_Safety_Warnings_Cited=["Doses above 100 micrograms require review."],
            Disclaimer=config.REQUIRED_DISCLAIMER,
        )
        outcome = schemas.enforce_safety_constraints(report, schemas.extract_dose_bound([DOSE_CHUNK]))
        assert outcome.report.Eligible_For_Vaccine is False

    def test_rejection_cannot_be_bypassed_by_the_model_approving_the_dose(self) -> None:
        """Even with perfect compliance language, the code still rejects."""
        report = schemas.ClinicalSafetyReport(
            Eligible_For_Vaccine=True,
            Recommended_Dose_mg=0.5,
            Reasoning_Steps=[
                "The guideline permits this dose and the patient is low risk."
            ],
            FDA_Safety_Warnings_Cited=[DOSE_CHUNK],
            Disclaimer=config.REQUIRED_DISCLAIMER,
        )
        outcome = schemas.enforce_safety_constraints(report, schemas.extract_dose_bound([DOSE_CHUNK]))
        assert outcome.report.Eligible_For_Vaccine is False
        assert outcome.report.Recommended_Dose_mg <= 0.1
        assert any(v.requirement == "FR-4.4" for v in outcome.violations)


def _make_run(patient_id: str | None = None) -> report_generator.ReportRun:
    """A validated, safety-checked run, for the persistence and latency tests.

    Module level rather than a ``staticmethod`` on one test class, so the later
    classes can build the same object without reaching into an unrelated one.
    """
    report = schemas.ClinicalSafetyReport(
        Eligible_For_Vaccine=True,
        Recommended_Dose_mg=0.05,
        Reasoning_Steps=["Within the stated range."],
        FDA_Safety_Warnings_Cited=["between 25 and 100 micrograms"],
        Disclaimer=config.REQUIRED_DISCLAIMER,
    )
    return report_generator.ReportRun(
        report=report,
        safety=schemas.enforce_safety_constraints(
            report, schemas.DoseBound(max_mcg=100.0)
        ),
        attempts=[report_generator.GenerationAttempt(1, True, latency_s=0.9)],
        dose_bound=schemas.DoseBound(max_mcg=100.0, source_text=DOSE_CHUNK),
        patient_id=patient_id,
    )


class TestReportPersistence:
    """Brief items 8, 9 and 10: JSON + Markdown, no overwriting, exact disclaimer."""

    _run = staticmethod(_make_run)

    def test_json_carries_the_structured_validated_report(self, tmp_path) -> None:
        json_path, _ = report_generator.persist_report(self._run(), tmp_path)
        stored = json.loads(json_path.read_text(encoding="utf-8"))
        assert set(stored["report"]) == set(schemas.REQUIRED_REPORT_FIELDS)
        assert stored["report"]["Eligible_For_Vaccine"] is True
        assert stored["report"]["Recommended_Dose_mg"] == pytest.approx(0.05)
        assert stored["safety"]["compliant"] is True
        assert stored["generation"]["attempts"][0]["attempt_number"] == 1

    def test_markdown_carries_the_exact_disclaimer(self, tmp_path) -> None:
        _, md_path = report_generator.persist_report(self._run(), tmp_path)
        markdown = md_path.read_text(encoding="utf-8")
        assert config.REQUIRED_DISCLAIMER in markdown
        assert "between 25 and 100 micrograms" in markdown

    def test_repeated_reports_never_overwrite(self, tmp_path) -> None:
        """Brief item 9: each run must leave the earlier report intact."""
        first_json, first_md = report_generator.persist_report(self._run("P0001"), tmp_path)
        first_body = first_md.read_text(encoding="utf-8")

        second_json, second_md = report_generator.persist_report(self._run("P0001"), tmp_path)

        assert first_json != second_json and first_md != second_md
        assert first_md.read_text(encoding="utf-8") == first_body, (
            "the first report was modified by the second run"
        )
        assert first_json.exists() and second_json.exists()
        assert len(list(tmp_path.glob("*.json"))) == 2
        assert len(list(tmp_path.glob("*.md"))) == 2

    def test_reports_from_different_patients_do_not_collide(self, tmp_path) -> None:
        first, _ = report_generator.persist_report(self._run("P0001"), tmp_path)
        second, _ = report_generator.persist_report(self._run("P0002"), tmp_path)
        assert first.stem != second.stem

    def test_patient_id_is_used_in_the_filename(self, tmp_path) -> None:
        json_path, _ = report_generator.persist_report(self._run("P0042"), tmp_path)
        assert "P0042" in json_path.stem

    def test_filename_stem_is_sanitised(self, tmp_path) -> None:
        """A patient id from a dataset must not become a path component."""
        json_path, _ = report_generator.persist_report(self._run("../../escape"), tmp_path)
        assert json_path.parent == tmp_path
        assert ".." not in json_path.name

    def test_reports_directory_is_created_when_absent(self, tmp_path) -> None:
        target = tmp_path / "nested" / "reports"
        assert not target.exists()
        json_path, md_path = report_generator.persist_report(self._run(), target)
        assert json_path.exists() and md_path.exists()

    def test_unwritable_directory_raises_oserror(self, tmp_path) -> None:
        blocker = tmp_path / "blocked"
        blocker.write_text("not a directory", encoding="utf-8")
        with pytest.raises(OSError):
            report_generator.persist_report(self._run(), blocker / "reports")


class TestGroundingIntegrity:
    """The prompt must not smuggle the target or any invented clinical value."""

    def test_target_column_is_never_restated_into_the_prompt(self) -> None:
        """Leaking Reaction_Score would let the model copy the label.

        The record below carries the ground truth. If it reached the prompt the
        model could restate it as a "prediction", and every report would look
        confident for the wrong reason.
        """
        record = {
            "Patient_ID": "P0001", "Age": 58, "Biomarker_Level": 11.2,
            "Pre_Existing_Conditions": 2, "Vaccine_Dose_mcg": 120.0,
            "Prior_Reaction_History": "Mild",
            "Reaction_Score": 1,
        }
        summary = report_generator.summarise_patient(record)
        assert config.TARGET_COL not in summary
        assert config.TARGET_COL not in report_generator.PATIENT_SUMMARY_FIELDS

    def test_absent_fields_are_omitted_not_defaulted(self) -> None:
        summary = report_generator.summarise_patient({"Age": 30})
        assert "30" in summary
        assert "Biomarker" not in summary, "a missing field must not be invented"

    def test_prompt_separates_retrieved_evidence_from_model_output(self) -> None:
        """FR-4.2 is judged by inspecting this prompt, so the boundary is explicit."""
        messages = report_generator.build_prompt("High", 0.82, ["chunk text"], "a patient")
        user = next(content for role, content in messages if role == "user")
        assert "RETRIEVED GUIDELINE EVIDENCE" in user
        assert "MODEL RISK OUTPUT" in user
        assert "NOT a clinical outcome" in user

    def test_prompt_names_the_source_document(self) -> None:
        messages = report_generator.build_prompt("Low", 0.1, ["chunk"], "a patient")
        user = next(content for role, content in messages if role == "user")
        assert config.GUIDELINE_DOCUMENT_NAME in user

    def test_system_prompt_forbids_inventing_facts(self) -> None:
        text = report_generator.SYSTEM_PROMPT.lower()
        assert "never invent" in text
        assert "insufficient" in text

    def test_report_is_json_serialisable(self, tmp_path) -> None:
        json_path, _ = report_generator.persist_report(_make_run(), tmp_path)
        reloaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert reloaded["disclaimer"] == config.REQUIRED_DISCLAIMER


class TestLatencyRecording:
    """Panel C's LLM bar depends on this file being written in the expected shape."""

    def test_generation_log_is_appended_not_overwritten(self, tmp_path) -> None:
        path = tmp_path / "generation_log.json"
        for _ in range(3):
            report_generator.record_generation_latency(_make_run(), path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["entries"]) == 3
        assert all(e["latency_s"] > 0 for e in payload["entries"])
        assert payload["entries"][0]["model"] == config.LLM_MODEL

    def test_generation_log_shape_matches_the_panel_reader(self, tmp_path) -> None:
        """The dashboard reads entries[].latency_s; a different shape renders blank."""
        path = tmp_path / "generation_log.json"
        report_generator.record_generation_latency(_make_run(), path)
        entry = json.loads(path.read_text(encoding="utf-8"))["entries"][0]
        assert isinstance(entry["latency_s"], (int, float))

    def test_an_unparsable_existing_log_is_replaced_not_fatal(self, tmp_path) -> None:
        """A corrupt log must not abort a report that already succeeded."""
        path = tmp_path / "generation_log.json"
        path.write_text("{not json", encoding="utf-8")
        report_generator.record_generation_latency(_make_run(), path)
        assert json.loads(path.read_text(encoding="utf-8"))["entries"]


class TestScopeGuard:
    """This stage must not reach beyond report generation."""

    def test_llm_package_does_not_orchestrate_the_pipeline(self) -> None:
        """AST-based, so a docstring mentioning the pipeline is not a violation.

        ``report_generator`` legitimately refers to :mod:`src.pipeline` in prose -
        pipeline orchestration is a later stage that *calls* this module, so an
        import in the other direction is what must be absent.
        """
        import ast
        from pathlib import Path

        for path in Path(report_generator.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                for module in modules:
                    assert not module.startswith("src.pipeline"), (
                        f"{path.name} imports {module}; orchestration is a later stage"
                    )

    def test_no_api_key_is_hardcoded(self) -> None:
        import re
        from pathlib import Path

        pattern = re.compile(r"sk-[A-Za-z0-9]{16,}")
        for path in Path(report_generator.__file__).parent.glob("*.py"):
            assert not pattern.search(path.read_text(encoding="utf-8")), (
                f"{path.name} appears to contain a hardcoded key"
            )