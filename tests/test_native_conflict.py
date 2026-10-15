"""Regression test for the native import-order conflict.

Kept in its own file because every case runs in a **fresh process**. A native
access violation cannot be caught as a Python exception - it terminates the
interpreter - so the only way to observe one is to run it and read the exit code.
An in-process test could not detect a regression in the failing direction; it
would simply take the test runner down with it.

The diagnosis this file guards against is documented in
:mod:`src.pipeline.native_stack`: loading scikit-learn before torch or ChromaDB
breaks both. The fix is import order, verified below to hold on every run.
"""

from __future__ import annotations

import pytest

from src import config
from src.pipeline import native_stack as ns


#: The orders that must work. If either regresses, the pipeline cannot run in one
#: process and process isolation would become necessary.
MUST_PASS = (
    ("torch", "chroma", "sklearn"),
    ("chroma", "torch", "sklearn"),
)

#: The orders known to break, kept so a platform where they start passing is
#: noticed rather than silently changing the diagnosis.
KNOWN_BROKEN = (
    ("sklearn", "torch"),
    ("sklearn", "chroma"),
)


class TestRequiredOrderingsHold:
    """The workaround is verified, not assumed."""

    @pytest.mark.parametrize("order", MUST_PASS, ids=lambda o: "_".join(o))
    def test_order_completes_in_a_fresh_process(self, order) -> None:
        result = ns.run_case(list(order), timeout=900.0)
        assert result["outcome"] == "ok", (
            f"import order {' -> '.join(order)} must work; got "
            f"{result['outcome']} ({result['detail']}). The pipeline needs all "
            "three libraries in one interpreter."
        )
        assert result["exit_code"] == 0

    def test_diagnose_reports_the_workaround_as_sufficient(self) -> None:
        """Runs the full matrix once. Slow but decisive."""
        report = ns.diagnose()
        assert report["required_failing"] == [], (
            "import order is no longer sufficient for: "
            f"{report['required_failing']}. Process isolation would be required."
        )
        assert report["import_order_is_sufficient"] is True


class TestKnownBrokenOrderingsStillFail:
    """Confirms the failure mode is still present, so the diagnosis is current."""

    @pytest.mark.parametrize("order", KNOWN_BROKEN, ids=lambda o: "_".join(o))
    def test_broken_order_does_not_silently_succeed(self, order) -> None:
        result = ns.run_case(list(order), timeout=900.0)
        assert result["outcome"] != "ok", (
            f"{' -> '.join(order)} previously broke. If it now works, the "
            "conflict is resolved upstream and preload_native_libraries() may be "
            "removable - worth re-checking rather than keeping by habit."
        )


class TestFailureDetection:
    """The diagnostic must classify failures rather than just reporting them."""

    def test_a_native_crash_is_reported_as_such(self) -> None:
        """The chromadb-after-sklearn order faults natively today."""
        result = ns.run_case(["sklearn", "chroma"], timeout=900.0)
        assert result["outcome"] in {"native_crash", "python_error", "ok"}
        if result["outcome"] == "native_crash":
            assert result["exit_code"] & 0xFFFFFFFF == 0xC0000005

    def test_a_python_exception_is_distinguished_from_a_native_crash(self) -> None:
        result = ns.run_case(["sklearn", "torch"], timeout=900.0)
        # torch's c10.dll failure is a catchable OSError, not a segfault.
        assert result["outcome"] == "python_error"
        assert result["detail"] == "OSError"

    def test_an_unknown_step_is_reported_as_a_python_error_not_a_pass(self) -> None:
        """A typo in a step name must not report a broken order as working.

        This was a real bug in the probe: an unrecognised name fell through the
        if/elif chain and was recorded as a success.
        """
        result = ns.run_case(["definitely_not_a_library"], timeout=300.0)
        assert result["outcome"] == "python_error"
        assert result["detail"] == "KeyError"


class TestTimeoutHandling:
    """A hung worker must be reported, never waited on forever."""

    def test_a_timeout_is_reported_with_a_null_exit_code(self, monkeypatch) -> None:
        import subprocess

        def explode(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

        monkeypatch.setattr(subprocess, "run", explode)
        result = ns.run_case(["torch"], timeout=0.01)
        assert result["outcome"] == "timeout"
        assert result["exit_code"] is None

    def test_a_timeout_does_not_satisfy_a_required_case(self, monkeypatch) -> None:
        """A hung process must not be read as 'working'."""
        import subprocess

        def explode(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="x", timeout=1)

        monkeypatch.setattr(subprocess, "run", explode)
        report = ns.diagnose(cases=(("torch", "chroma", "sklearn"),))
        assert report["import_order_is_sufficient"] is False
        assert report["required_failing"] == [["torch", "chroma", "sklearn"]]


class TestNonZeroExitHandling:
    """Exit codes are the signal, so they must not be flattened to 'ok'."""

    def test_a_native_fault_code_is_named(self, monkeypatch) -> None:
        import subprocess

        class Result:
            returncode = 0xC0000005 - (1 << 32)  # Windows sign-extends the status
            stdout = ""
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
        result = ns.run_case(["torch"])
        assert result["outcome"] == "native_crash"
        assert "ACCESS_VIOLATION" in result["detail"]

    def test_an_ordinary_nonzero_exit_is_not_called_a_native_crash(
        self, monkeypatch
    ) -> None:
        """A script bug must not be reported as a segfault."""
        import subprocess

        class Result:
            returncode = 2
            stdout = ""
            stderr = "bad argument"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
        result = ns.run_case(["torch"])
        assert result["outcome"] == "nonzero_exit"
        assert result["exit_code"] == 2

    def test_output_is_not_parsed_as_a_successful_prediction(self, monkeypatch) -> None:
        """Console text on stdout must never be mistaken for a result.

        A child that printed something and exited non-zero is still a failure.
        """
        import subprocess

        class Result:
            returncode = 1
            stdout = '{"probability": 0.99}'  # plausible-looking output
            stderr = "crashed"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
        result = ns.run_case(["torch"])
        assert result["outcome"] != "ok"
        # Nothing from stdout leaks into the result - only the exit code decides.
        assert "probability" not in str(result)

    def test_a_stdout_error_marker_is_not_read_as_success(self, monkeypatch) -> None:
        """Even exit code 3, which carries an error payload, is not a pass."""
        import subprocess

        class Result:
            returncode = 3
            stdout = '{"steps": [], "error": "OSError"}'
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
        result = ns.run_case(["sklearn", "torch"])
        assert result["outcome"] == "python_error"
        assert result["detail"] == "OSError"


class TestImportOrderGuard:
    """The pipeline's own import order must satisfy the constraint."""

    def test_pipeline_preloads_before_any_sklearn_import(self) -> None:
        """Reads the source order, since by import time it is already too late."""
        from pathlib import Path

        source = Path(
            __import__("src.pipeline.pipeline", fromlist=["x"]).__file__
        ).read_text(encoding="utf-8")
        preload_at = source.index("preload_native_libraries()")
        # Every scikit-learn-reaching import must come after the preload call.
        for module in ("src.data", "src.ml", "preprocessing"):
            position = source.index(f"import {module}") if f"import {module}" in source else -1
            if position != -1:
                assert position > preload_at, (
                    f"{module} is imported before preload_native_libraries(); "
                    "that ordering faults natively."
                )

    def test_import_order_is_satisfied_in_the_running_process(self) -> None:
        """This process already imported everything, so the guard must hold."""
        assert ns.import_order_is_safe() is True

    def test_preload_is_idempotent(self) -> None:
        """Calling it twice must not raise or reload."""
        first = ns.preload_native_libraries()
        second = ns.preload_native_libraries()
        assert set(first) == set(second)

    def test_preload_reports_unavailable_dependencies_without_raising(self) -> None:
        """A missing optional dependency is reported, not fatal at import time."""
        loaded = ns.preload_native_libraries()
        assert set(loaded) == {name for name, _ in ns.NATIVE_LOAD_ORDER}


class TestEnvironmentIsRecorded:
    """The diagnosis needs the environment it was measured in."""

    def test_versions_are_reported_for_the_three_conflicting_libraries(self) -> None:
        versions = ns.library_versions()
        for name in ("numpy", "scikit-learn", "torch", "chromadb"):
            assert name in versions

    def test_no_environment_variable_is_used_as_a_fix(self) -> None:
        """KMP_DUPLICATE_LIB_OK was tried and does not work; nothing depends on it.

        Checks for an actual *use* - reading, setting or exporting the variable -
        rather than a mention. The documentation is expected to name what was
        ruled out, and that is not a dependency on it.
        """
        import ast
        import sys
        from pathlib import Path

        stdlib = set(sys.stdlib_module_names)
        for path in Path(ns.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                for module in modules:
                    root = module.split(".")[0]
                    assert root in stdlib or root in {
                        "src", "dashboard", "chromadb", "sentence_transformers",
                        "langchain", "langchain_community", "numpy", "pandas",
                        "torch", "sklearn", "joblib", "matplotlib", "docx",
                        "pydantic", "langchain_core", "langchain_openai",
                        "langchain_text_splitters", "dotenv",
                    }, f"{path.name} imports unexpected module {module}"

                # Any string literal naming the variable inside executable code
                # would be a use; inside a docstring it is only documentation.
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if isinstance(getattr(node, "parent", None), ast.Expr):
                        continue  # a bare string expression - a docstring

    def test_the_variable_is_not_set_by_the_project(self) -> None:
        """Nothing in the project writes this variable into the environment."""
        from pathlib import Path

        for path in Path("src").rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                assert "environ" not in stripped or "KMP_DUPLICATE" not in stripped, (
                    f"{path} appears to set KMP_DUPLICATE_LIB_OK; the fix must be "
                    "import order, not an environment variable"
                )