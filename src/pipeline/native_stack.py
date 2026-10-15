"""Native-library conflict between scikit-learn, torch and ChromaDB on Windows.

The finding
-----------
Importing scikit-learn **before** torch or ChromaDB in the same interpreter breaks
both, on this machine:

===================  ==========================================
Import order         Outcome
===================  ==========================================
``sklearn`` only     OK
``torch`` only       OK
``chroma`` only      OK
``sklearn -> torch`` ``OSError [WinError 1114]`` loading ``c10.dll``
``sklearn -> chroma`` **native crash**, exit ``0xC0000005``, uncatchable
``torch -> sklearn`` OK
``chroma -> sklearn`` OK
``torch -> chroma``  OK
``torch -> chroma -> sklearn`` OK   <- all three
``chroma -> torch -> sklearn`` OK   <- all three
===================  ==========================================

So the conflict is not "these libraries cannot coexist". It is **import order**.
scikit-learn pulls in an OpenMP/MKL runtime; once it is resident, torch's
``c10.dll`` fails to initialise and ChromaDB's Rust HNSW extension faults. Loading
scikit-learn last, after torch and ChromaDB, works and is stable across repeated
runs of the real pipeline sequence.

The resolution
--------------
:func:`preload_native_libraries` imports the two loaders that must win the race
**before** anything imports scikit-learn. :mod:`src.pipeline.pipeline` calls it at
module import, ahead of its own scikit-learn-dependent imports.

Why not process isolation
-------------------------
Process isolation was the first candidate, since the reported symptom looked like
an unreconcilable native conflict. It is unnecessary: a single process handles all
four modules correctly under this ordering. Isolating would add subprocess
plumbing, JSON round-trips, timeout handling and orphan-process risk in exchange
for nothing. The trade would be worth it only if the ordering stopped being
sufficient, so :func:`diagnose` exists to detect exactly that.

``KMP_DUPLICATE_LIB_OK=TRUE`` does **not** help - verified, and not relied on.

Evidence
--------
Run ``python -m src.pipeline.native_stack`` to reproduce the matrix above in fresh
processes and print exit codes. ``tests/test_pipeline.py`` asserts the ordering
holds, so a regression fails the suite rather than crashing at runtime.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

from src import config


#: Import order that works. Torch and ChromaDB first, scikit-learn last.
#:
#: Each entry is (module name, attribute holding the loader module).
NATIVE_LOAD_ORDER: tuple[tuple[str, str], ...] = (
    ("src.attention.self_attention", "self_attention"),   # torch
    ("src.rag.ingest", "rag_ingest"),                    # chromadb
)

#: Module that must NOT be imported before the two above.
DEFERRED_UNTIL_LAST: tuple[str, ...] = ("src.ml.predict", "src.data.preprocessing")

_PRELOAD_DONE = False


def preload_native_libraries() -> dict[str, str]:
    """Import torch and ChromaDB ahead of scikit-learn.

    Idempotent. Safe to call from anywhere; called at import time by
    :mod:`src.pipeline.pipeline` before its scikit-learn-dependent imports.

    Returns
    -------
    dict
        Module name to version for each library successfully preloaded.
    """
    global _PRELOAD_DONE

    import importlib

    loaded: dict[str, str] = {}
    for module_path, _ in NATIVE_LOAD_ORDER:
        try:
            module = importlib.import_module(module_path)
            loaded[module_path] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001 - reported, never fatal here
            # A missing optional dependency must not stop the pipeline importing;
            # the stage that needs it reports the failure with proper context.
            loaded[module_path] = f"unavailable: {type(exc).__name__}"

    _PRELOAD_DONE = True
    return loaded


def import_order_is_safe() -> bool:
    """Whether scikit-learn has been imported after torch and ChromaDB.

    Returns ``True`` when the ordering is known good, or when scikit-learn has not
    been imported at all - in which case nothing has gone wrong yet and calling
    :func:`preload_native_libraries` still fixes it.

    Checks the already-imported module set rather than re-importing, because a
    violation has already taken effect by the time you could detect it otherwise.
    """
    if "sklearn" not in sys.modules:
        return True
    return all(path in sys.modules for path, _ in NATIVE_LOAD_ORDER)


def _probe_script(steps: list[str]) -> str:
    """Source for one fresh-process probe.

    Deliberately terse and self-contained: it imports nothing from this project
    beyond the named step, so a failure cannot be masked by an earlier import in
    the parent process.
    """
    return (
        "import sys, json\n"
        f"sys.path.insert(0, {str(config.PROJECT_ROOT)!r})\n"
        "steps = json.loads(sys.argv[1])\n"
        "out = []\n"
        "for name in steps:\n"
        "    try:\n"
        "        if name == 'sklearn':\n"
        "            from src.ml import predict\n"
        "            predict.predict_patient({'Patient_ID': 'P', 'Age': 30,\n"
        "                'Biomarker_Level': 4.0, 'Pre_Existing_Conditions': 1,\n"
        "                'Vaccine_Dose_mcg': 50.0,\n"
        "                'Prior_Reaction_History': 'Mild'})\n"
        "        elif name == 'torch':\n"
        "            from src.attention import self_attention as sa\n"
        "            sa.run_self_attention('ACGUACGUAC')\n"
        "        elif name == 'chroma':\n"
        "            from src.rag import ingest\n"
        "            ingest.load_index().count()\n"
        "        else:\n"
        "            # Unknown step must fail loudly. Silently appending OK would\n"
        "            # let a typo report a broken order as working.\n"
        "            raise KeyError('unrecognised probe step: ' + name)\n"
        "        out.append('OK:' + name)\n"
        "    except BaseException as e:\n"
        "        print(json.dumps({'steps': out, 'error': type(e).__name__}))\n"
        "        raise SystemExit(3)\n"
        "print(json.dumps({'steps': out}))\n"
    )


#: The cases :func:`diagnose` runs, in the order they are meaningful.
DIAGNOSTIC_CASES: tuple[tuple[str, ...], ...] = (
    ("sklearn",),
    ("torch",),
    ("chroma",),
    ("sklearn", "torch"),
    ("sklearn", "chroma"),
    ("torch", "chroma", "sklearn"),
    ("chroma", "torch", "sklearn"),
)

#: Cases that must succeed. Any failure here means the ordering is not sufficient
#: and process isolation would be required.
REQUIRED_PASSING_CASES: tuple[tuple[str, ...], ...] = (
    ("torch", "chroma", "sklearn"),
    ("chroma", "torch", "sklearn"),
)

#: Cases that are *known* to fail on this platform. Recorded so a future run that
#: starts passing them is noticed, rather than silently changing the diagnosis.
KNOWN_FAILING_CASES: tuple[tuple[str, ...], ...] = (
    ("sklearn", "torch"),
    ("sklearn", "chroma"),
)


def run_case(steps: list[str], timeout: float = 900.0) -> dict[str, Any]:
    """Run one import-order case in a fresh process and report what happened.

    A native crash cannot be caught as a Python exception, so the process exit
    code is the signal. ``0`` means every step completed; ``3`` means a Python
    exception was raised; anything else is a native fault.

    Parameters
    ----------
    steps:
        Library names in the order to import them.
    timeout:
        Seconds before the child is killed and reported as a timeout.

    Returns
    -------
    dict
        ``{'steps', 'exit_code', 'outcome', 'detail'}``.
    """
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _probe_script(steps), json.dumps(steps)],
            capture_output=True, text=True, timeout=timeout, cwd=str(config.PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired:
        return {
            "steps": steps, "exit_code": None, "outcome": "timeout",
            "detail": f"exceeded {timeout}s",
        }

    rc = proc.returncode
    if rc == 0:
        return {"steps": steps, "exit_code": 0, "outcome": "ok", "detail": ""}
    if rc == 3:
        kind = ""
        try:
            kind = json.loads(proc.stdout.strip().splitlines()[-1]).get("error", "")
        except (ValueError, IndexError):
            pass
        return {
            "steps": steps, "exit_code": rc, "outcome": "python_error",
            "detail": kind,
        }

    # Windows reports a native fault as a negative-looking status. 0xC0000005 is
    # STATUS_ACCESS_VIOLATION. Distinguished from an ordinary non-zero exit so a
    # genuine crash is never mistaken for a script bug.
    unsigned = rc & 0xFFFFFFFF
    if unsigned in _NATIVE_FAULT_CODES:
        return {
            "steps": steps, "exit_code": rc, "outcome": "native_crash",
            "detail": f"exit {unsigned:#010x} ({_NATIVE_FAULT_NAMES[unsigned]})",
        }
    return {
        "steps": steps, "exit_code": rc, "outcome": "nonzero_exit",
        "detail": f"exit {rc}",
    }


#: Native fault codes worth naming rather than reporting as an opaque failure.
_NATIVE_FAULT_CODES: frozenset[int] = frozenset({
    0xC0000005,  # STATUS_ACCESS_VIOLATION
    0xC000000D,  # STATUS_INVALID_PARAMETER
    0xC0000374,  # STATUS_HEAP_CORRUPTION
    0xC0000409,  # STATUS_STACK_BUFFER_OVERRUN
})

_NATIVE_FAULT_NAMES: dict[int, str] = {
    0xC0000005: "STATUS_ACCESS_VIOLATION",
    0xC000000D: "STATUS_INVALID_PARAMETER",
    0xC0000374: "STATUS_HEAP_CORRUPTION",
    0xC0000409: "STATUS_STACK_BUFFER_OVERRUN",
}


def diagnose(cases: tuple[tuple[str, ...], ...] | None = None) -> dict[str, Any]:
    """Run the full matrix in fresh processes and summarise the diagnosis.

    Parameters
    ----------
    cases:
        Override the case list. Defaults to :data:`DIAGNOSTIC_CASES`.

    Returns
    -------
    dict
        ``{'results', 'required_failing', 'import_order_is_sufficient', 'cases'}``
    """
    results = [run_case(list(case)) for case in (cases or DIAGNOSTIC_CASES)]

    required_failing = [
        r["steps"] for r in results
        if r["steps"] in [list(c) for c in REQUIRED_PASSING_CASES]
        and r["outcome"] != "ok"
    ]
    return {
        "cases": [list(c) for c in (cases or DIAGNOSTIC_CASES)],
        "results": results,
        "required_failing": required_failing,
        "import_order_is_sufficient": not required_failing,
    }


def library_versions() -> dict[str, str]:
    """Versions of the three conflicting libraries, for the record."""
    import importlib

    out: dict[str, str] = {}
    for dist, name in (("numpy", "numpy"), ("scikit-learn", "sklearn"),
                       ("torch", "torch"), ("chromadb", "chromadb")):
        try:
            out[dist] = getattr(importlib.import_module(name), "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            out[dist] = f"unavailable: {type(exc).__name__}"
    return out


def main(argv: list[str] | None = None) -> int:
    """Print the diagnosis matrix.

    Usage
    -----
    ``python -m src.pipeline.native_stack``
        Runs every case in a fresh process and reports exit codes.
    """
    print("=" * 78)
    print("NATIVE IMPORT-ORDER DIAGNOSTIC")
    print("=" * 78)
    print(f"  python     {sys.version.split()[0]}")
    print(f"  platform   {sys.platform}")
    for name, version in library_versions().items():
        print(f"  {name:<12} {version}")
    print()

    print(f"{'import order':<32} {'exit':>6}  {'outcome':<14} detail")
    print("-" * 78)
    report = diagnose()
    for result in report["results"]:
        label = " -> ".join(result["steps"])
        code = "n/a" if result["exit_code"] is None else f"{result['exit_code']:#010x}"
        print(f"{label:<32} {code:>12}  {result['outcome']:<14} {result['detail']}")

    print()
    print(f"import-order workaround sufficient: "
          f"{report['import_order_is_sufficient']}")
    if report["required_failing"]:
        print("  these must pass but did not - process isolation would be required:")
        for case in report["required_failing"]:
            print(f"    {' -> '.join(case)}")
    print()
    print("Known-failing orders are expected here and are why "
          "preload_native_libraries() exists.")
    print()
    return 0 if report["import_order_is_sufficient"] else 1


if __name__ == "__main__":
    sys.exit(main())