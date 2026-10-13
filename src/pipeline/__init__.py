"""Pipeline subpackage: the single integration layer for the VIP platform.

Architecture rule 8. This is the only module permitted to import from ``src.data``,
``src.ml``, ``src.attention``, ``src.rag`` and ``src.llm`` together. Every other
package imports downward only, so the dependency graph stays acyclic and each
module remains independently testable.

The end-to-end flow (PRD Section 3.1)::

    patient input
        -> preprocessing        src.data.preprocessing
        -> ML risk prediction   src.ml.predict
        -> attention            src.attention.self_attention
        -> RAG retrieval        src.rag.retrieve
        -> LLM report           src.llm.report_generator
        -> outputs              outputs/reports, outputs/figures, outputs/logs
"""

from src.pipeline.pipeline import (
    PipelineError,
    VIPPipeline,
    VIPResult,
    run_pipeline,
)

__all__ = ["VIPPipeline", "VIPResult", "PipelineError", "run_pipeline"]