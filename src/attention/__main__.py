"""CLI shim so ``python -m src.attention`` runs the attention demo.

``src/attention/__init__.py`` imports the submodules, so invoking
``python -m src.attention.self_attention`` directly triggers a ``RuntimeWarning``
about the module already being present in ``sys.modules``. This shim is a separate
module that keeps both spellings clean.

Usage
-----
``python -m src.attention``                      nucleotide demo
``python -m src.attention --feature``            run over a processed patient record
``python -m src.attention --sequence ACGUACGU``  a specific sequence
"""

from __future__ import annotations

import sys

from src.attention.self_attention import main

if __name__ == "__main__":
    sys.exit(main())