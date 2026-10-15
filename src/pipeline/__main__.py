"""CLI shim so ``python -m src.pipeline`` runs without a runpy warning.

``src/pipeline/__init__.py`` imports :mod:`src.pipeline.pipeline` to re-export
:class:`~src.pipeline.pipeline.VIPPipeline`, so invoking
``python -m src.pipeline.pipeline`` triggers a ``RuntimeWarning`` about the module
already being present in ``sys.modules``. This shim keeps both spellings clean and
matches the pattern used by ``src.ml`` and ``src.attention``.

Usage
-----
``python -m src.pipeline --run --patient-id P0001``
``python -m src.pipeline --all``
"""

from __future__ import annotations

import sys

from src.pipeline.pipeline import main


if __name__ == "__main__":
    sys.exit(main())