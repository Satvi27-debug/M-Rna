"""CLI shim so ``python -m src.rag`` runs retrieval against the existing index.

``src/rag/__init__.py`` imports both submodules, so invoking
``python -m src.rag.retrieve`` directly triggers a ``RuntimeWarning`` about the
module already being present in ``sys.modules``. This shim keeps both spellings
clean.

Usage
-----
``python -m src.rag --query "What are the storage requirements?"``
``python -m src.rag --evaluate``
``python -m src.rag --dose 120``
``python -m src.rag --ingest``   build the index first
"""

from __future__ import annotations

import sys

from src.rag import ingest as rag_ingest
from src.rag import retrieve as rag_retrieve


def main(argv: list[str] | None = None) -> int:
    """Ingest when asked, then retrieve."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--ingest" in args:
        index = args.index("--ingest")
        del args[index]
        code = rag_ingest.main(args or ["--no-export"])
        if code:
            return code
        if not args:
            return 0
        print()
    return rag_retrieve.main(args)


if __name__ == "__main__":
    sys.exit(main())