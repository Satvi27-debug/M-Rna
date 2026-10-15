"""Project-level guarantees that belong to no single functional module.

Two concerns live here:

* **Dependency honesty.** ``requirements.txt`` must cover every third-party
  import the code actually makes. Added after an audit found
  ``langchain-text-splitters`` undeclared: ``src/rag/ingest.py`` imports it
  directly, ``langchain`` 1.x is a meta-package that does *not* require it, and
  only ``langchain-classic`` (unused here) pulls it in. A clean
  ``pip install -r requirements.txt`` therefore failed at guideline ingestion.

* **Supplied source integrity.** The PRD-supplied CSV and guideline DOCX are
  inputs, never outputs. These tests assert the loader never writes to the raw
  directory and that no absolute path from the author's machine is baked into the
  source.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

#: Distribution name -> the module name(s) it provides. Without this mapping a
#: correct ``requirements.txt`` looks like it is missing ``sklearn``/``docx``.
_DISTRIBUTION_TO_MODULE = {
    "scikit-learn": "sklearn",
    "python-docx": "docx",
    "python-dotenv": "dotenv",
    "pillow": "pil",
    "pyyaml": "yaml",
    "beautifulsoup4": "bs4",
    "typing-extensions": "typing_extensions",
}

#: First-party packages, not distribution dependencies.
_LOCAL = {
    "src", "dashboard", "tests", "data", "outputs", "models", "guidelines",
    "vectorstore", "run", "cli", "mrnа", "mrna",
}


def _declared_distributions() -> dict[str, str]:
    """Distribution names in ``requirements.txt``, lowercased and normalised."""
    declared: dict[str, str] = {}
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[; ]", line)[0].strip()
        if name:
            declared[name.lower().replace("_", "-")] = line
    return declared


def _module_is_declared(module: str, declared: dict[str, str]) -> bool:
    """Whether ``import module`` is satisfiable from ``requirements.txt``.

    Three ways an import can legitimately be covered:

    1. The module's own distribution is declared (``chromadb`` -> ``chromadb``).
    2. The distribution has a different name (``scikit-learn`` -> ``sklearn``).
    3. The distribution arrives transitively (``matplotlib`` requires
       ``pillow``, so ``from PIL import Image`` works).
    """
    key = module.lower().replace("_", "-")
    if key in declared:
        return True
    for dist, provided in _DISTRIBUTION_TO_MODULE.items():
        if provided == key and dist in declared:
            return True

    import importlib.metadata as metadata

    # Ask the installed environment which distributions provide this module,
    # then check whether any of them is declared or a declared dependency.
    try:
        providers = set(
            metadata.packages_distributions().get(module.lower(), [])
        ) | set(metadata.packages_distributions().get(module, []))
    except Exception:
        providers = set()
    providers |= {
        dist for dist, provided in _DISTRIBUTION_TO_MODULE.items()
        if provided == key
    }
    if not providers:
        return False
    if providers & set(declared):
        return True

    # Transitive: is any provider a requirement of a declared distribution?
    for dist in declared:
        try:
            requires = metadata.distribution(dist).requires or []
        except Exception:
            continue
        for requirement in requires:
            name = re.split(r"[<>=!~\[; ]", requirement)[0].strip().lower()
            if name in providers or name.replace("_", "-") in providers:
                return True
    return False


def _third_party_imports() -> dict[str, set[str]]:
    """Top-level third-party module names imported anywhere in the source."""
    stdlib = set(sys.stdlib_module_names)
    found: dict[str, set[str]] = {}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if rel.parts[0] in {".git", "venv", ".venv", "build", "dist"}:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:          # relative import: first-party
                    continue
                modules = [(node.module or "").split(".")[0]]
            else:
                continue
            for module in modules:
                if module and module not in stdlib and module != "__future__":
                    found.setdefault(module, set()).add(str(rel).replace("\\", "/"))
    return found


class TestDependencyListIsHonest:
    """``requirements.txt`` must cover what the code imports."""

    def test_every_third_party_import_is_declared(self) -> None:
        declared = _declared_distributions()
        undeclared = {
            module: sorted(where)
            for module, where in sorted(_third_party_imports().items())
            if module.lower() not in _LOCAL
            and not _module_is_declared(module, declared)
        }
        assert not undeclared, (
            "requirements.txt does not cover these imports; a clean install "
            f"would fail: {undeclared}"
        )

    def test_langchain_text_splitters_is_declared_explicitly(self) -> None:
        """The exact gap found by the audit, pinned so it cannot regress.

        ``src/rag/ingest.py`` imports ``RecursiveCharacterTextSplitters`` at
        module level, and ``langchain`` 1.x does not pull it in.
        """
        ingest = (ROOT / "src" / "rag" / "ingest.py").read_text(encoding="utf-8")
        if "langchain_text_splitters" not in ingest:
            pytest.skip("ingest.py no longer uses langchain_text_splitters")
        assert "langchain-text-splitters" in _declared_distributions(), (
            "src/rag/ingest.py imports langchain_text_splitters directly, but "
            "requirements.txt does not declare it. langchain 1.x is a "
            "meta-package that does NOT require it, so a clean install fails "
            "at guideline ingestion."
        )

    def test_no_banned_framework_is_declared(self) -> None:
        """The brief excludes these; the dependency list must not reintroduce."""
        banned = ("react", "next", "fastapi", "streamlit", "docker",
                  "kubernetes", "postgres", "qdrant", "firebase")
        declared = _declared_distributions()
        offenders = [d for d in declared if d.split("-")[0] in banned]
        assert not offenders, f"banned package(s) declared: {offenders}"

    def test_pinned_lower_bounds_exist_for_every_requirement(self) -> None:
        """Every declared package carries a lower bound, so setup is reproducible."""
        text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        unpinned = [
            line.split("#")[0].strip()
            for line in text.splitlines()
            if line.strip() and not line.strip().startswith(("#", "-"))
            and ">=" not in line
        ]
        assert not unpinned, f"requirements without a version bound: {unpinned}"


class TestSuppliedInputsAreImmutable:
    """The PRD-supplied files are inputs. Nothing in the project may rewrite them."""

    def test_loader_never_writes_into_the_raw_data_directory(self) -> None:
        from src import config
        from src.data import preprocessing

        raw = preprocessing.load_raw_records(config.RAW_PATIENT_CSV)
        assert len(raw) == 500, "the supplied dataset must still hold 500 rows"

        # Exercising the cleaning path must not touch the source file.
        before = config.RAW_PATIENT_CSV.read_bytes()
        preprocessing.save_clean_records(
            preprocessing.impute_missing_values(raw),
            config.DATA_PROCESSED_DIR / "clean.csv",
        )
        assert config.RAW_PATIENT_CSV.read_bytes() == before

    def test_no_absolute_author_paths_in_tracked_source(self) -> None:
        """A machine-specific path breaks every other checkout."""
        offenders = []
        for path in sorted(ROOT.rglob("*.py")):
            rel = path.relative_to(ROOT)
            if rel.parts[0] in {".git", "venv", ".venv"}:
                continue
            for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if re.search(r"[A-Za-z]:\\\\?Users\\\\?", line) or re.search(
                    r"/Users/\w+/", line
                ):
                    offenders.append(f"{rel}:{number}")
        assert not offenders, (
            f"absolute author-machine path(s) committed: {offenders[:5]}"
        )

    def test_env_file_is_ignored_and_only_the_example_is_tracked(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert ".env" in gitignore, ".env must be gitignored"
        assert not (ROOT / ".env").exists(), (
            "a real .env exists in the tree; it must never be committed"
        )
        assert (ROOT / ".env.example").is_file(), ".env.example must ship"