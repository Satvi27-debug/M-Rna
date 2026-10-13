"""mRNA Vaccine Intelligence & Protection (VIP) - application source package.

Layering rules for everything under ``src/``:

* ``data``        - patient record loading, validation and feature preparation.
* ``ml``           - model training, inference and evaluation. Training and
                     inference are separate entry points.
* ``attention``    - from-scratch Transformer self-attention over mRNA
                     sequences. Independent of the dashboard.
* ``rag``          - document ingestion, chunking, embedding, persistence and
                     top-k retrieval. Contains no report-generation logic.
* ``llm``          - Pydantic report schema, safety validation and structured
                     LLM generation.
* ``pipeline``     - the single integration layer that wires the five modules
                     together in order.

Modules must import path and hyper-parameter constants from :mod:`src.config`
rather than hard-coding them, and must not read files from ``data/``,
``guidelines/``, ``models/``, ``vectorstore/`` or ``outputs/`` using literal
strings.
"""

from src.config import PROJECT_NAME, PROJECT_VERSION

__all__ = ["PROJECT_NAME", "PROJECT_VERSION"]