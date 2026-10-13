"""Retrieval-augmented generation subpackage over the regulatory guidelines.

Boundary rules
--------------
* This package owns ingestion, chunking, embedding, persistence and retrieval.
  It contains **no** report-generation or LLM-prompting logic (architecture
  rule 6). Prompt assembly belongs to :mod:`src.llm`.
* ``guidelines/Moderna_mRNA_Safety_Guidelines.docx`` is a read-only source.
  Architecture rule 2 forbids modifying it, so its text is extracted in memory at
  ingest time and never written back.
* **The layer retrieves evidence; it does not interpret it.** Nothing here
  summarises, paraphrases, or decides whether a dose is acceptable. Every returned
  chunk is the verbatim source text with its provenance intact, so a statement in a
  generated report can always be traced back to a paragraph of the DOCX.
* No LLM is called anywhere in this package. Retrieval is embedding similarity
  over the source document, nothing more.

Requirements owned here
-----------------------
FR-3.1  Chunk the guideline document with visible overlap between chunks.
FR-3.2  Generate a dense vector embedding for every chunk.
FR-3.3  Persist to a local ChromaDB collection that survives process restart.
FR-3.4  Return exactly k chunks, ordered by similarity, default k=3.
FR-3.5  Record and expose retrieval latency in seconds for every query.

A note on FR-3.1
----------------
The PRD specifies 300-500 token chunks. The required embedding model,
``all-MiniLM-L6-v2``, truncates at 256 tokens - verified empirically, since two
texts sharing a 256-token prefix embed to *identical* vectors. A 300-500 token
chunk would have its tail silently discarded. The supplied document is only ~906
tokens in total, so smaller chunks lose no content. See the module docstring in
:mod:`src.rag.ingest` for the full reasoning.
"""

from src.rag import ingest, retrieve
from src.rag.ingest import (
    ChunkingError,
    GuidelineExtractionError,
    IndexingError,
    GuidelineIndex,
    build_embedding_function,
    build_index,
    chunk_guidelines,
    chunk_with_metadata,
    extract_docx_text,
    load_index,
    make_chunk_id,
    verify_chunk_bounds,
)
from src.rag.retrieve import (
    DoseEvidence,
    RetrievalError,
    RetrievalLog,
    RetrievalResult,
    RetrievedChunk,
    query_guidelines,
    retrieve_dose_evidence,
    retrieve_guidelines,
    run_evaluation_queries,
)

__all__ = [
    "ingest",
    "retrieve",
    "extract_docx_text",
    "chunk_guidelines",
    "chunk_with_metadata",
    "verify_chunk_bounds",
    "build_embedding_function",
    "build_index",
    "load_index",
    "make_chunk_id",
    "GuidelineIndex",
    "GuidelineExtractionError",
    "ChunkingError",
    "IndexingError",
    "query_guidelines",
    "retrieve_guidelines",
    "retrieve_dose_evidence",
    "run_evaluation_queries",
    "RetrievalResult",
    "RetrievedChunk",
    "RetrievalLog",
    "DoseEvidence",
    "RetrievalError",
]