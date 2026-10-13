"""CLI shim so ``python -m src.ml`` runs the training stage.

``src/ml/__init__.py`` imports every submodule, so ``python -m src.ml.cli``
triggers a ``RuntimeWarning`` about ``cli`` already being in ``sys.modules``. This
shim is a distinct module that imports nothing from the package at module scope
beyond the CLI runner, keeping the documented ``python -m src.ml.train`` and
``python -m src.ml`` invocations clean.

Usage
-----
``python -m src.ml``            run the full data + ML stage
``python -m src.ml --quiet``    suppress staged progress output
"""

from __future__ import annotations

import sys

from src.ml.cli import main

if __name__ == "__main__":
    sys.exit(main())