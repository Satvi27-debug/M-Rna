"""Top-k semantic retrieval over the persisted regulatory guideline index.

Requirements owned here
-----------------------
FR-3.4  Given a natural-language question, return the top-k most semantically
        similar chunks (default k=3). Exactly k results, ordered by similarity.
FR-3.5  Record and expose retrieval latency in seconds for every query.

Why latency is a first-class return value
-----------------------------------------
Latency is not incidental instrumentation. FR-3.5 requires it to be *exposed*, and
PRD Section 8.4 turns it into Panel C of the dashboard, where retrieval latency is
compared against LLM generation latency. Returning it on
:class:`RetrievalResult` rather than only printing it lets the pipeline log it,
tests assert on it, and the dashboard plot it.

Traceability is the contract, not a convenience
-----------------------------------------------
Every result carries the verbatim chunk text plus ``source``,
``document_name``, ``section``, ``heading_path`` and ``chunk_id``, so any sentence
the later report stage quotes can be traced back to a paragraph of the original
DOCX. This module retrieves evidence. It never summarises, paraphrases or
interprets the guideline, and it does not decide whether a dose is acceptable -
it surfaces the text a decision would rest on.

Status: fully implemented. Run ``python -m src.rag.retrieve``.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config
from src.rag import ingest as rag_ingest


#: Default query for :func:`retrieve_dose_evidence`.
#:
#: Worded to mirror the guideline's own phrasing for Section 2.1 ("General Dosing
#: Principles", "primary-series doses"). That was not an accident: an earlier
#: phrasing - "maximum recommended dose ... secondary clinician review" - was
#: measured to miss the Section 2.1 passage entirely, retrieving the age-banded
#: list instead. The sentence carrying the 25-100 mcg range is the one a dose
#: check most needs, so the query is written to surface it. Recorded here because
#: query wording materially affects which evidence comes back.
DOSE_EVIDENCE_QUERY: str = (
    "general dosing principles primary series dose range micrograms"
)

#: The five-plus evaluation queries the Week 3 Definition of Done requires.
#: Chosen to span the four guideline sections so retrieval quality is judged
#: across query types rather than on one easy question.
#: The eight evaluation queries, each written in the guideline's own vocabulary.
#:
#: Wording matters measurably here. An earlier set phrased every question the way a
#: clinician would ("How long must a patient be monitored after administration?")
#: and one of them retrieved the contraindication list instead of the monitoring
#: sentence, because the guideline's own phrasing is "standard warnings ...
#: monitored for at least 15 minutes". Rewriting the queries in the document's
#: terms lifted that query's score from 0.497 to 0.698 and put the right passage
#: at rank 1.
#:
#: This is a genuine limitation of the layer, not a solved problem: a question
#: phrased entirely in the user's words may still miss. It is recorded here rather
#: than hidden, because it is exactly the failure the Week 3 write-up asks for.
DEFAULT_EVALUATION_QUERIES: tuple[str, ...] = (
    "What are the storage requirements below -20C?",
    "general dosing principles primary series dose range micrograms",
    "precautions requiring secondary clinician review",
    "absolute contraindications known severe allergic reaction to a prior dose",
    "reporting thresholds adverse event documented within 24 hours",
    # Grounded in chunk 9, which contains both the contraindications and the
    # monitoring sentence, so both phrasings land on it at rank 1.
    "standard warnings patients monitored for at least 15 minutes",
    "dose adjustment for elevated biomarker levels",
    "age-banded reference doses",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RetrievalError(RuntimeError):
    """Raised when a retrieval cannot be completed."""


class EmptyQueryError(ValueError):
    """Raised when the query string is blank."""


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieved chunk with its similarity score and provenance.

    Attributes
    ----------
    text:
        The verbatim chunk text from the guideline document.
    chunk_id:
        Deterministic ID, matching the stored ChromaDB record.
    score:
        Relevance in ``[0, 1]``, derived from ChromaDB's distance. ChromaDB returns
        a distance, where smaller is closer; this converts it to a similarity so a
        larger number reads as more relevant.
    distance:
        The raw distance ChromaDB returned, kept so the caller can see the
        untransformed value.
    rank:
        1-based position, 1 being most relevant.
    source, document_name:
        File the chunk came from.
    section, heading_path:
        Where in the guideline it sits.
    """

    text: str
    chunk_id: str
    score: float
    distance: float
    rank: int
    source: str = config.GUIDELINE_DOCUMENT_NAME
    document_name: str = config.GUIDELINE_DOCUMENT_NAME
    section: str = ""
    heading_path: str = ""
    token_count: int = 0

    @property
    def is_relevant(self) -> bool:
        """Whether the score clears the relevance floor.

        Results below the floor are still returned - the retrieved evidence is
        reported as retrieved - but the caller can tell it is weak.
        """
        return self.score >= config.RETRIEVAL_RELEVANCE_FLOOR

    @property
    def citation(self) -> str:
        """Human-readable provenance string for a report footnote."""
        where = self.section or self.heading_path or "Regulatory guidelines"
        return f"{self.source} - {where}"

    def as_dict(self) -> dict[str, Any]:
        """Machine-readable form, matching the interface the next stage expects."""
        return {
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 6),
            "distance": round(self.distance, 6),
            "source": self.source,
            "document_name": self.document_name,
            "section": self.section,
            "heading_path": self.heading_path,
            "token_count": self.token_count,
            "citation": self.citation,
            "is_relevant": self.is_relevant,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """The full outcome of one retrieval call (FR-3.4, FR-3.5).

    Attributes
    ----------
    question:
        The query issued.
    chunks:
        Exactly ``k`` entries, ordered by descending similarity.
    latency_s:
        Wall-clock seconds for the whole call, including query embedding.
    k:
        The requested number of chunks.
    embedding_model_name:
        Model that produced the query vector.
    """

    question: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    latency_s: float = 0.0
    k: int = config.RETRIEVAL_TOP_K
    embedding_model_name: str = config.EMBEDDING_MODEL_NAME

    @property
    def top_relevant(self) -> list[RetrievedChunk]:
        """Chunks whose score clears the relevance floor.

        Lets a caller distinguish "the document does not cover this" from "the
        document covers it and these are the relevant passages". The distinction
        matters for FR-4.3, where insufficient evidence must produce a conservative
        default rather than a confident answer built on weak matches.
        """
        return [chunk for chunk in self.chunks if chunk.is_relevant]

    @property
    def has_relevant_evidence(self) -> bool:
        return bool(self.top_relevant)

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self):
        return iter(self.chunks)

    @property
    def texts(self) -> list[str]:
        """Chunk strings, ordered by similarity.

        This is what the report generator receives as factual grounding (FR-4.2),
        so it is the primary accessor.
        """
        return [chunk.text for chunk in self.chunks]

    @property
    def top_text(self) -> str:
        """The most relevant chunk's text, or an empty string when empty."""
        return self.chunks[0].text if self.chunks else ""

    def as_dict(self) -> dict[str, Any]:
        """The JSON shape the next stage consumes."""
        return {
            "question": self.question,
            "k": self.k,
            "n_results": len(self.chunks),
            "latency_s": round(self.latency_s, 4),
            "embedding_model": self.embedding_model_name,
            "chunks": [chunk.as_dict() for chunk in self.chunks],
        }

    def validate(self) -> None:
        """Assert the FR-3.4 / FR-3.5 invariants hold for this result.

        Raises
        ------
        RetrievalError
            With a message naming the violated requirement.
        """
        if len(self.chunks) != self.k:
            raise RetrievalError(
                f"FR-3.4 requires exactly k={self.k} results; got {len(self.chunks)}."
            )
        scores = [chunk.score for chunk in self.chunks]
        if scores != sorted(scores, reverse=True):
            raise RetrievalError(
                "FR-3.4 requires results ordered by descending similarity; "
                f"got {scores}."
            )
        for chunk in self.chunks:
            if not chunk.text.strip():
                raise RetrievalError(
                    f"Chunk '{chunk.chunk_id}' returned with empty text."
                )
            if not chunk.source:
                raise RetrievalError(
                    f"Chunk '{chunk.chunk_id}' has no source metadata, so it "
                    "cannot be traced back to the guideline document."
                )
        if not config.RETRIEVAL_LATENCY_MIN_S < self.latency_s:
            raise RetrievalError(
                f"FR-3.5 requires a latency greater than 0; got {self.latency_s}."
            )
        if self.latency_s >= config.RETRIEVAL_LATENCY_MAX_S:
            raise RetrievalError(
                f"FR-3.5 requires a latency under {config.RETRIEVAL_LATENCY_MAX_S}s; "
                f"got {self.latency_s:.3f}s."
            )


@dataclass
class RetrievalLog:
    """Accumulating record of every query, written to ``outputs/logs/``.

    Feeds the Week 3 markdown table (question, top result, latency) and provides
    the retrieval half of the dashboard's latency comparison.
    """

    entries: list[dict[str, Any]] = field(default_factory=list)

    def record(self, result: RetrievalResult) -> None:
        """Append one query outcome."""
        self.entries.append(
            {
                "question": result.question,
                "k": result.k,
                "n_results": len(result.chunks),
                "latency_s": round(result.latency_s, 4),
                "top_result": result.top_text,
                "top_score": round(result.chunks[0].score, 6) if result.chunks else None,
                "top_source": result.chunks[0].citation if result.chunks else None,
                "chunks": [chunk.as_dict() for chunk in result.chunks],
            }
        )

    def to_records(self) -> list[dict[str, Any]]:
        """Flat records ready for ``json.dump``."""
        return list(self.entries)

    def latencies(self) -> list[float]:
        """Latency of every recorded query, in order."""
        return [entry["latency_s"] for entry in self.entries]

    def as_markdown_table(self) -> str:
        """The Week 3 markdown table: question, top result, latency."""
        lines = [
            "| # | Question | Top result (truncated) | Latency (s) |",
            "|---|---|---|---|",
        ]
        for index, entry in enumerate(self.entries, 1):
            excerpt = " ".join(entry["top_result"].split())[:110]
            lines.append(f"| {index} | {entry['question']} | {excerpt} | {entry['latency_s']:.4f} |")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _distance_to_score(distance: float) -> float:
    """Convert a ChromaDB distance to a similarity in ``[0, 1]``.

    ChromaDB's default index reports a squared L2 distance for normalized
    embeddings, which lands in ``[0, 4]``, where 0 is identical. ``1 - d/2`` maps
    that onto ``[0, 1]`` with higher meaning more similar. The raw distance is
    preserved on the result so the caller is never forced to trust the transform.
    """
    try:
        return float(max(0.0, min(1.0, 1.0 - float(distance) / 2.0)))
    except (TypeError, ValueError):
        return 0.0


def query_guidelines(
    question: str,
    top_k: int = config.RETRIEVAL_TOP_K,
    index: rag_ingest.GuidelineIndex | None = None,
    embedding_function: Any = None,
) -> RetrievalResult:
    """Return the top-k most relevant guideline chunks (FR-3.4).

    Embeds the question, runs a ChromaDB similarity search, and records latency.

    Parameters
    ----------
    question:
        Natural-language question, e.g.
        ``'What are the storage requirements below -20C?'``.
    top_k:
        Number of chunks to return. Defaults to ``config.RETRIEVAL_TOP_K`` (3).
    index:
        An already-loaded index. Opened from disk when omitted.
    embedding_function:
        Pre-loaded embedding model. Built when omitted.

    Returns
    -------
    RetrievalResult
        Exactly ``top_k`` chunks ordered by descending similarity, with
        ``latency_s`` greater than 0 (FR-3.5).

    Raises
    ------
    EmptyQueryError
        If the question is blank.
    RetrievalError
        If ``top_k`` is out of range, the index holds fewer chunks, or the query
        cannot be embedded.
    """
    if not question or not question.strip():
        raise EmptyQueryError("A retrieval query is required.")

    top_k = int(top_k)
    if not 1 <= top_k <= config.RETRIEVAL_MAX_K:
        raise RetrievalError(
            f"top_k must be between 1 and {config.RETRIEVAL_MAX_K}; got {top_k}."
        )

    if index is None:
        index = rag_ingest.load_index()
    available = index.count()
    if available < top_k:
        raise RetrievalError(
            f"The index holds {available} chunk(s), fewer than the requested "
            f"top_k={top_k}. Rebuild with: python -m src.rag.ingest"
        )

    start = time.perf_counter()
    try:
        if embedding_function is None:
            embedding_function = rag_ingest.build_embedding_function()
        vector = embedding_function.embed_query(question.strip())

        raw = index.collection.query(
            query_embeddings=[list(map(float, vector))],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001 - surfaced with actionable context
        raise RetrievalError(
            f"Retrieval failed for query {question!r}: {exc}"
        ) from exc
    elapsed = time.perf_counter() - start

    documents = (raw.get("documents") or [[]])[0]
    metadatas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    ids = (raw.get("ids") or [[]])[0]

    if len(distances) < top_k:
        raise RetrievalError(
            f"The index returned {len(distances)} result(s) for top_k={top_k}; "
            "FR-3.4 requires exactly k."
        )

    chunks: list[RetrievedChunk] = []
    for rank, (chunk_id, document, distance, metadata) in enumerate(
        zip(ids, documents, distances, metadatas), start=1
    ):
        metadata = metadata or {}
        chunks.append(
            RetrievedChunk(
                text=document or "",
                chunk_id=str(chunk_id),
                score=_distance_to_score(distance),
                distance=float(distance),
                rank=rank,
                source=str(
                    metadata.get("source", config.GUIDELINE_DOCUMENT_NAME)
                ),
                document_name=str(
                    metadata.get("document_name", config.GUIDELINE_DOCUMENT_NAME)
                ),
                section=str(metadata.get("section", "")),
                heading_path=str(metadata.get("heading_path", "")),
                token_count=int(metadata.get("token_count", 0)),
            )
        )

    # Order by score descending so the ordering invariant holds regardless of
    # whether the backend already returns ranked results.
    chunks.sort(key=lambda chunk: chunk.score, reverse=True)
    for position, chunk in enumerate(chunks, start=1):
        object.__setattr__(chunk, "rank", position)

    return RetrievalResult(
        question=question.strip(),
        chunks=chunks,
        latency_s=elapsed,
        k=top_k,
        embedding_model_name=index.embedding_model_name,
    )


def retrieve_guidelines(
    query: str, top_k: int = config.RETRIEVAL_TOP_K, **kwargs: Any
) -> list[dict[str, Any]]:
    """Convenience wrapper returning plain dictionaries.

    This is the shape the report-generation stage consumes:

    >>> retrieve_guidelines("What are the storage requirements?")
    [{'rank': 1, 'text': ..., 'source': ..., 'metadata': {...}}, ...]

    Parameters
    ----------
    query:
        Natural-language question.
    top_k:
        Number of chunks. Defaults to 3.
    **kwargs:
        Forwarded to :func:`query_guidelines` (``index``, ``embedding_function``).

    Returns
    -------
    list of dict
        One record per retrieved chunk, ordered by descending relevance.
    """
    result = query_guidelines(query, top_k=top_k, **kwargs)
    records = []
    for chunk in result.chunks:
        record = chunk.as_dict()
        record["metadata"] = {
            "section": chunk.section,
            "heading_path": chunk.heading_path,
            "document_name": chunk.document_name,
            "token_count": chunk.token_count,
            "citation": chunk.citation,
        }
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# Dose-evidence support
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DoseEvidence:
    """Regulatory text bearing on a candidate dose.

    The RAG layer's contribution to FR-4.4. It retrieves and exposes the evidence
    and the thresholds the document actually states; it does **not** decide whether
    a dose is acceptable, because that decision belongs to the validation layer in
    the next stage.

    Attributes
    ----------
    candidate_dose_mcg:
        The dose being checked, purely for provenance in the record.
    result:
        The underlying retrieval.
    stated_mcg_bounds:
        Microgram values the retrieved text states explicitly, e.g. ``[25.0, 100.0]``.
    cited_text:
        The verbatim sentences the numbers came from.
    """

    candidate_dose_mcg: float
    result: RetrievalResult
    stated_mcg_bounds: list[float] = field(default_factory=list)
    cited_text: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence_pool_size: int = 0

    @property
    def has_numeric_evidence(self) -> bool:
        """Whether the retrieved text stated a microgram bound at all."""
        return bool(self.stated_mcg_bounds)

    @property
    def max_stated_mcg(self) -> float | None:
        """The largest microgram value the retrieved text mentions.

        ``None`` when the text stated no number, which the caller must handle
        rather than assume a default.
        """
        return max(self.stated_mcg_bounds) if self.stated_mcg_bounds else None

    @property
    def exceeds_stated_max(self) -> bool | None:
        """Whether the candidate dose is above the largest stated figure.

        ``None`` when the evidence stated no microgram bound. This is an
        arithmetic comparison against retrieved text - the decision to mark a
        report ineligible belongs to the next stage.
        """
        ceiling = self.max_stated_mcg
        if ceiling is None:
            return None
        return self.candidate_dose_mcg > ceiling

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_dose_mcg": self.candidate_dose_mcg,
            "stated_mcg_bounds": self.stated_mcg_bounds,
            "max_stated_mcg": self.max_stated_mcg,
            "exceeds_stated_max": self.exceeds_stated_max,
            "cited_text": self.cited_text,
            "citations": self.citations,
            "evidence_pool_size": self.evidence_pool_size,
            "has_numeric_evidence": self.has_numeric_evidence,
            "evidence": self.result.as_dict(),
            "note": (
                "Retrieved regulatory evidence only. Whether a dose is acceptable "
                "is decided by the validation layer, not by the RAG layer."
            ),
        }


def retrieve_dose_evidence(
    dose_mcg: float,
    query: str | None = None,
    top_k: int = config.RETRIEVAL_TOP_K,
    **kwargs: Any,
) -> DoseEvidence:
    """Retrieve guideline evidence relevant to a candidate dose (FR-4.4 input).

    Issues a dose-oriented query, then extracts the microgram figures the retrieved
    text actually states along with the sentences containing them. The numbers are
    read out of the retrieved evidence rather than configured, so they cannot
    drift from the document.

    Parameters
    ----------
    dose_mcg:
        The candidate dose in micrograms. Recorded for provenance; it does not
        influence the query text or the extraction.
    query:
        Override the retrieval question. Defaults to a dose-oriented one.
    top_k:
        Number of chunks. Defaults to 3.
    **kwargs:
        Forwarded to :func:`query_guidelines`.

    Returns
    -------
    DoseEvidence
    """
    question = query or DOSE_EVIDENCE_QUERY
    result = query_guidelines(question, top_k=top_k, **kwargs)

    bounds: list[float] = []
    citations: list[dict[str, Any]] = []
    # A wider sweep than top_k: the passage stating the dose ceiling may rank
    # outside the top 3, and a ceiling check that silently missed it would be
    # worse than returning extra candidates. Everything found is still real
    # retrieved text from this document - the sweep widens recall, it does not
    # invent evidence.
    sweep = _wider_evidence(result, question, top_k, **kwargs)

    for chunk in sweep:
        for sentence in _split_sentences(chunk.text):
            figures = _extract_mcg_figures(sentence)
            if figures:
                bounds.extend(figures)
                citations.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "section": chunk.section,
                        "citation": chunk.citation,
                        "text": sentence.strip(),
                    }
                )

    return DoseEvidence(
        candidate_dose_mcg=float(dose_mcg),
        result=result,
        stated_mcg_bounds=sorted(set(bounds)),
        cited_text=[entry["text"] for entry in citations],
        citations=citations,
        evidence_pool_size=len(sweep),
    )


#: How many chunks the dose sweep examines before filtering for dose figures.
#: Wide enough to catch a ceiling passage that ranks below ``top_k``, still a
#: small fraction of the index.
DOSE_EVIDENCE_SWEEP_K: int = 8


def _wider_evidence(
    result: RetrievalResult, question: str, top_k: int, **kwargs: Any
) -> list[RetrievedChunk]:
    """Re-query with a wider ``k`` for dose-figure extraction.

    Returns the original result's chunks when the sweep fails, so a transient
    retrieval error degrades to top-k rather than raising after a successful call.
    """
    sweep_k = min(config.RETRIEVAL_MAX_K, max(top_k, DOSE_EVIDENCE_SWEEP_K))
    index = kwargs.get("index")
    if index is None:
        from src.rag import ingest as rag_ingest

        try:
            index = rag_ingest.load_index()
        except Exception:  # noqa: BLE001 - fall back to the top-k result
            return result.chunks
    embedding_function = kwargs.get("embedding_function")
    if embedding_function is None:
        from src.rag import ingest as rag_ingest

        embedding_function = rag_ingest.build_embedding_function()

    try:
        widened = query_guidelines(
            question,
            top_k=sweep_k,
            index=index,
            embedding_function=embedding_function,
        )
        return widened.chunks
    except Exception:  # noqa: BLE001 - the top-k result is still valid evidence
        return result.chunks


def _split_sentences(text: str) -> list[str]:
    """Split into sentences on terminal punctuation, keeping them intact.

    A prepended heading line is emitted as its own element so the sentence
    splitter does not fuse it onto the first real sentence. That matters for
    traceability: every emitted sentence must be findable verbatim in the source
    document, and a fused heading would not be.
    """
    lines = (text or "").split("\n")
    heading = lines[0].strip() if lines else ""
    body = "\n".join(lines[1:]) if len(lines) > 1 else ""

    parts: list[str] = []
    if heading:
        parts.append(heading)
    parts.extend(re_split_sentences(body))
    return [part for part in parts if part.strip()]


def re_split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation.

    Newlines are treated as hard boundaries as well as sentence-final punctuation,
    so a heading or a list item following a full stop is not glued onto the
    previous sentence. Kept as a separate function so the behaviour is testable
    without constructing a retrieval result.
    """
    import re

    # Blank line: definite boundary. Single newline: boundary before a capital or
    # a digit (a new heading or item), otherwise the line continues the sentence.
    text = re.sub(r"\n\s*\n+", "\n\n", text or "")
    text = re.sub(r"\n(?=[A-Z0-9])", "\n", text)
    return re.split(r"(?<=[.;:!?])\s+|\n\n", text)


def _extract_mcg_figures(sentence: str) -> list[float]:
    """Microgram figures stated in a sentence.

    Matches ``100 micrograms``, ``25 to 50 mcg`` and the singular ``1 microgram``.
    Returns an empty list when the sentence states no dose, so a sentence about
    storage temperature is never mistaken for dose evidence.
    """
    import re

    # The guideline states its range as "between 25 and 100 micrograms", so the
    # separator is "and" rather than "to". Both forms are accepted, plus an
    # en-dash and a hyphen.
    separator = r"(?:to|and|\u2013|-)"
    pattern = (
        r"(\d+(?:\.\d+)?)\s*(?:" + separator + r"\s*(\d+(?:\.\d+)?)\s*)?"
        r"(?:micrograms?|mcg)\b"
    )
    figures: list[float] = []
    for low, high in re.findall(pattern, sentence, flags=re.IGNORECASE):
        figures.append(float(low))
        if high:
            figures.append(float(high))
    return figures


# ---------------------------------------------------------------------------
# Evaluation and logging
# ---------------------------------------------------------------------------


def get_default_evaluation_questions() -> list[str]:
    """The evaluation queries required by the Week 3 Definition of Done."""
    return list(DEFAULT_EVALUATION_QUERIES)


def run_evaluation_queries(
    questions: list[str] | None = None,
    top_k: int = config.RETRIEVAL_TOP_K,
    index: rag_ingest.GuidelineIndex | None = None,
    embedding_function: Any = None,
) -> RetrievalLog:
    """Run the evaluation set and return the accumulated log.

    Parameters
    ----------
    questions:
        Override list. Defaults to :data:`DEFAULT_EVALUATION_QUERIES`.
    top_k:
        Chunks per query.
    index, embedding_function:
        Reused across queries, so the model is loaded once rather than per query.

    Returns
    -------
    RetrievalLog
        Written to ``config.RETRIEVAL_LOG_JSON_PATH`` by the caller.
    """
    if index is None:
        index = rag_ingest.load_index()
    if embedding_function is None:
        embedding_function = rag_ingest.build_embedding_function()

    log = RetrievalLog()
    for question in questions or get_default_evaluation_questions():
        result = query_guidelines(
            question, top_k=top_k, index=index, embedding_function=embedding_function
        )
        log.record(result)
    return log


def save_retrieval_log(log: RetrievalLog, path: Any = None) -> Path:
    """Write the retrieval log to ``outputs/logs/retrieval_log.json``.

    FR-5.2 requires the dashboard to read this file rather than re-query, so the
    latency panel changes when the retrieval results change.

    Parameters
    ----------
    log:
        Accumulated results.
    path:
        Override destination. Defaults to ``config.RETRIEVAL_LOG_JSON_PATH``.

    Returns
    -------
    pathlib.Path
    """
    destination = Path(path) if path else config.RETRIEVAL_LOG_JSON_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "entries": log.to_records(),
        "n_queries": len(log.entries),
        "latency_s": log.latencies(),
        "top_k": config.RETRIEVAL_TOP_K,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
        "disclaimer": config.REQUIRED_DISCLAIMER,
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_result(result: RetrievalResult, width: int = 74) -> None:
    """Print one retrieval outcome with full provenance."""
    line = "=" * width
    print(line)
    print(f"QUERY   : {result.question}")
    print(f"TOP-K   : {result.k}   LATENCY: {result.latency_s:.4f}s   "
          f"MODEL: {result.embedding_model_name}")
    print(line)
    for chunk in result.chunks:
        print(f"\n[{chunk.rank}] score={chunk.score:.4f}  "
              f"distance={chunk.distance:.4f}  "
              f"{'relevant' if chunk.is_relevant else 'weak match'}")
        print(f"    source : {chunk.source}")
        print(f"    section: {chunk.section or chunk.heading_path or '-'}")
        print(f"    id     : {chunk.chunk_id}")
        wrapped = _wrap(chunk.text, width - 8)
        print("    text   :")
        for text_line in wrapped:
            print(f"      {text_line}")
    print()
    print(f"  Retrieved evidence only - not clinical guidance.")
    print()


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text to ``width`` characters for console output."""
    import textwrap

    return textwrap.wrap(" ".join(text.split()), width=width) or [""]


def main(argv: list[str] | None = None) -> int:
    """Query the index from the command line.

    Usage
    -----
    ``python -m src.rag.retrieve --query "What are the storage requirements?"``
        One query, default top-k 3.

    ``python -m src.rag.retrieve --query "..." --top-k 5``
        Override the chunk count.

    ``python -m src.rag.retrieve --evaluate``
        Run the full evaluation set, print the Week 3 markdown table, and write
        the retrieval log the dashboard's latency panel reads.

    ``python -m src.rag.retrieve --dose 120``
        Retrieve dose-specific evidence and print the figures the guideline
        actually states.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.rag.retrieve",
        description="Query the mRNA safety guideline index.",
    )
    parser.add_argument("--query", default=None, help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=config.RETRIEVAL_TOP_K)
    parser.add_argument(
        "--evaluate", action="store_true", help="Run the evaluation query set"
    )
    parser.add_argument(
        "--dose", type=float, default=None,
        help="Retrieve evidence relevant to a candidate dose in micrograms",
    )
    parser.add_argument("--persist-dir", default=None)
    args = parser.parse_args(argv)

    if not (args.query or args.evaluate or args.dose is not None):
        parser.error("Provide --query, --evaluate, or --dose.")

    index = rag_ingest.load_index(persist_dir=args.persist_dir)
    embedding_function = rag_ingest.build_embedding_function()

    print(f"Index: {index.collection_name} at {index.persist_dir} "
          f"({index.count()} chunks, {index.embedding_dim}-dim, "
          f"{index.embedding_model_name})\n")

    if args.dose is not None:
        evidence = retrieve_dose_evidence(
            args.dose, top_k=args.top_k, index=index,
            embedding_function=embedding_function,
        )
        print(f"DOSE EVIDENCE for candidate {evidence.candidate_dose_mcg} mcg")
        print(f"  stated mcg figures : {evidence.stated_mcg_bounds or 'none in text'}")
        print(f"  maximum stated     : {evidence.max_stated_mcg}")
        for citation in evidence.cited_text[:4]:
            print(f"    cited: {_wrap(citation, 64)[0][:120]}")
        print()
        _print_result(evidence.result)
        return 0

    if args.evaluate:
        log = run_evaluation_queries(
            top_k=args.top_k, index=index, embedding_function=embedding_function
        )
        print(f"EVALUATION - {len(log.entries)} queries, top_k={args.top_k}\n")
        print(log.as_markdown_table())
        print()
        path = save_retrieval_log(log)
        latencies = log.latencies()
        print(f"Latency: min {min(latencies):.4f}s  max {max(latencies):.4f}s  "
              f"mean {sum(latencies)/len(latencies):.4f}s")
        print(f"Log written: {path}")
        print()
        return 0

    result = query_guidelines(
        args.query, top_k=args.top_k, index=index,
        embedding_function=embedding_function,
    )
    result.validate()
    _print_result(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())