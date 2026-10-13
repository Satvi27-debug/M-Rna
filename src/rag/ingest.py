"""Document ingestion: read the guideline DOCX, chunk it, embed it, index it.

The pipeline
------------
    Moderna_mRNA_Safety_Guidelines.docx
        -> text extraction      python-docx, structure preserved
        -> section-aware chunking
        -> Sentence-Transformer embeddings   all-MiniLM-L6-v2, 384-dim
        -> ChromaDB                            persistent, under vectorstore/

Why retrieval, and why traceability matters
---------------------------------------------
The later report-generation stage must not answer regulatory questions from a
language model's memory. A model can state a confident, incorrect dose limit. So
this layer's job is narrow and deliberate: find the actual sentences in the
supplied guideline document that bear on a question, and hand back the verbatim
text with its provenance. Every chunk carries ``source``, ``document_name``,
``section``, ``heading_path`` and ``chunk_id``, so any statement in a generated
report can be traced to a paragraph of the original DOCX. Nothing here summarises,
paraphrases or interprets.

Chunk sizing, and a measured deviation from FR-3.1
---------------------------------------------------
FR-3.1 asks for chunks of 300-500 tokens. The required embedding model,
``all-MiniLM-L6-v2``, has ``max_seq_length = 256``. That was verified empirically
rather than assumed: two texts sharing a 256-token prefix but differing afterwards
produce **identical** vectors (cosine 1.00000007). A 300-500 token chunk would
have its tail silently discarded and could collide with a neighbour, degrading
retrieval while still appearing to satisfy FR-3.1.

The supplied document is only ~906 tokens in total, so sizing below the model's
limit loses nothing - the whole document is represented. Chunks are therefore
capped at ``config.EMBEDDING_MAX_TOKENS``, and ``tests/test_rag.py`` asserts both
the size bound *and* that no two distinct chunks embed to identical vectors. The
FR-3.1 numbers are retained in ``config.FR31_MIN_TOKENS`` / ``FR31_MAX_TOKENS`` so
the deviation stays visible.

Status: fully implemented. Run ``python -m src.rag.ingest``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src import config


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class GuidelineExtractionError(RuntimeError):
    """Raised when the guideline document cannot be read as a DOCX package."""


class ChunkingError(ValueError):
    """Raised when chunking produces chunks outside the configured bounds."""


class IndexingError(RuntimeError):
    """Raised when the vector index cannot be written or read."""


# ---------------------------------------------------------------------------
# Token accounting
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.,;:()\-][A-Za-z0-9]+)*|[^\sA-Za-z0-9]")


def count_tokens(text: str) -> int:
    """Approximate the token count of a chunk.

    Counts word-like runs and standalone punctuation. This tracks the
    sub-word tokenizer closely enough to enforce a size ceiling, and is used for
    both chunk sizing and overlap measurement so the two can never disagree.

    Exact tokenisation would need the model's own tokenizer, which is not loaded
    here - the splitter counts characters, and this bridges that to tokens.
    """
    return len(_TOKEN_RE.findall(text or ""))


def estimate_chunks(text: str) -> int:
    """Number of chunks :func:`chunk_guidelines` would produce for ``text``."""
    return len(chunk_guidelines(text))


# ---------------------------------------------------------------------------
# Document structure
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentBlock:
    """One paragraph from the source document, with its heading context.

    Attributes
    ----------
    text:
        Verbatim paragraph text. Never altered.
    style:
        Paragraph style name, e.g. ``Heading 1``, ``List Paragraph``, ``None``.
    section:
        Nearest enclosing ``Heading 1`` text, e.g. ``Section 2: Dosing Guidance
        by Age Band``. Empty before the first heading.
    heading_path:
        Section then sub-heading, e.g. ``Section 2: Dosing Guidance by Age Band
        > 2.1 General Dosing Principles``. Empty when the paragraph is not under
        a heading.
    """

    text: str
    style: str = "None"
    section: str = ""
    heading_path: str = ""
    order: int = 0

    @property
    def is_heading(self) -> bool:
        return self.style.startswith("Heading")

    @property
    def heading_level(self) -> int:
        """Numeric level parsed from a ``Heading N`` style name, else 0."""
        match = re.match(r"Heading (\d+)", self.style or "")
        return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class DocumentOutline:
    """The document's section structure, for inspection and tests."""

    sections: list[str] = field(default_factory=list)
    subsections: list[str] = field(default_factory=list)
    n_paragraphs: int = 0
    n_tables: int = 0
    total_tokens: int = 0
    title: str = ""


@dataclass(frozen=True)
class ChunkingReport:
    """Verifiable evidence for the chunking stage."""

    n_chunks: int
    token_counts: list[int]
    overlap_tokens: int
    in_bounds: bool
    sections_covered: list[str] = field(default_factory=list)
    within_embedding_limit: bool = True
    #: FR-3.1's literal 300-500 bound, kept alongside the tighter one actually
    #: enforced so the deviation is visible in the report rather than implicit.
    fr31_min_tokens: int = config.FR31_MIN_TOKENS
    fr31_max_tokens: int = config.FR31_MAX_TOKENS

    def failures(self) -> list[str]:
        """Human-readable list of violations, for error messages and logs."""
        problems: list[str] = []
        for index, count in enumerate(self.token_counts):
            if count > config.EMBEDDING_MAX_TOKENS:
                problems.append(
                    f"chunk {index} has {count} tokens, exceeding the embedding "
                    f"model's limit of {config.EMBEDDING_MAX_TOKENS}; its tail "
                    "would be silently discarded"
                )
            elif index < len(self.token_counts) - 1 and not (
                config.CHUNK_MIN_TOKENS <= count <= config.CHUNK_MAX_TOKENS
            ):
                problems.append(
                    f"chunk {index} has {count} tokens, outside "
                    f"{config.CHUNK_MIN_TOKENS}-{config.CHUNK_MAX_TOKENS}"
                )
        return problems


# ---------------------------------------------------------------------------
# Step 1 - extraction
# ---------------------------------------------------------------------------


def _read_document(docx_path: Path) -> Any:
    """Open a DOCX with python-docx."""
    import docx

    try:
        return docx.Document(str(docx_path))
    except Exception as exc:  # noqa: BLE001 - surfaced with actionable context
        raise GuidelineExtractionError(
            f"Could not read {docx_path} as a Word document: {exc}. "
            "The file may be corrupt or is not a .docx package."
        ) from exc


def extract_docx_blocks(docx_path: Any = None) -> list[DocumentBlock]:
    """Read the document into blocks, tracking the heading each paragraph sits under.

    Reads paragraphs in document order and threads the current ``Heading 1`` and
    ``Heading 2`` forward, so every content paragraph carries the section it
    belongs to. That provenance is what lets a retrieved chunk say *which* part of
    the guideline it came from.

    Table text is also captured: the document's metadata table carries the
    document version and format, which is worth retaining as source evidence.
    Tables appear after the paragraphs because that is their position in the
    supplied file.

    Parameters
    ----------
    docx_path:
        Source document. Defaults to ``config.GUIDELINE_DOCX``.

    Returns
    -------
    list of DocumentBlock
        Non-empty paragraphs and table rows, in reading order.

    Raises
    ------
    FileNotFoundError, GuidelineExtractionError
    """
    path = Path(docx_path) if docx_path else config.GUIDELINE_DOCX
    if not path.is_file():
        raise FileNotFoundError(
            f"Guideline document not found at {path}. "
            f"Place the supplied {config.GUIDELINE_DOCUMENT_NAME} there."
        )

    document = _read_document(path)
    blocks: list[DocumentBlock] = []
    section = ""
    subsection = ""
    order = 0

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        style_name = paragraph.style.name if paragraph.style is not None else "None"
        level = DocumentBlock(text, style_name, section, "", order).heading_level

        if level == 1:
            section = text
            subsection = ""
            heading_path = text
        elif level == 2:
            subsection = text
            heading_path = f"{section} > {subsection}" if section else subsection
        else:
            heading_path = f"{section} > {subsection}" if section else ""

        blocks.append(
            DocumentBlock(
                text=text,
                style=style_name,
                section=section,
                heading_path=heading_path,
                order=order,
            )
        )
        order += 1

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            # Collapse the label/value pair into one readable line.
            line = ": ".join(part for part in cells if part)
            if line:
                blocks.append(
                    DocumentBlock(text=line, style="Table Row", section="", order=order)
                )
                order += 1

    if not blocks:
        raise GuidelineExtractionError(
            f"No text was extracted from {path}. The document appears empty."
        )
    return blocks


def extract_docx_text(docx_path: Any = None) -> str:
    """Return the document's text as plain text.

    Paragraphs are separated by a blank line, which preserves the blank-line
    structure the chunker's separator hierarchy relies on. Headings are kept -
    they carry the section identity that makes retrieved chunks traceable.

    Parameters
    ----------
    docx_path:
        Source document. Defaults to ``config.GUIDELINE_DOCX``.

    Returns
    -------
    str
        The extracted text.

    Raises
    ------
    FileNotFoundError, GuidelineExtractionError
    """
    blocks = extract_docx_blocks(docx_path)
    return "\n\n".join(block.text for block in blocks)


def document_outline(blocks: list[DocumentBlock]) -> DocumentOutline:
    """Summarise the structure found in ``blocks``.

    Used for the CLI report and by tests to assert the expected sections were
    discovered rather than guessed.
    """
    sections: list[str] = []
    subsections: list[str] = []
    for block in blocks:
        level = block.heading_level
        if level == 1 and block.text not in sections:
            sections.append(block.text)
        elif level == 2 and block.text not in subsections:
            subsections.append(block.text)
    return DocumentOutline(
        sections=sections,
        subsections=subsections,
        n_paragraphs=len(blocks),
        total_tokens=sum(count_tokens(block.text) for block in blocks),
        title=blocks[0].text if blocks else "",
    )


# ---------------------------------------------------------------------------
# Step 2 - chunking
# ---------------------------------------------------------------------------


def chunk_guidelines(
    text: str | None = None,
    blocks: list[DocumentBlock] | None = None,
    chunk_size: int = config.CHUNK_SIZE_CHARS,
    chunk_overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Split the guideline text into overlapping chunks.

    Section-aware by construction: each block carries its heading, and a chunk
    never spans two different ``Heading 1`` sections. That means every chunk is
    topically coherent, so a retrieved chunk can be cited as "this part of the
    guideline" rather than "some text that happened to be nearby".

    Splits use LangChain's ``RecursiveCharacterTextSplitter`` with the separator
    hierarchy in ``config.CHUNK_SEPARATORS``, which prefers paragraph then line
    then sentence boundaries - a chunk ending mid-sentence is one a retriever
    scores poorly and a report generator would quote badly.

    Parameters
    ----------
    text:
        Full extracted text. Ignored when ``blocks`` is supplied.
    blocks:
        Blocks from :func:`extract_docx_blocks`. Preferred, because it preserves
        section metadata.
    chunk_size, chunk_overlap:
        Sizes in **characters**, per the splitter's convention.

    Returns
    -------
    list of str
        Non-empty chunks, in document order, with the heading line prepended to
        each so it stays with its text.

    Raises
    ------
    GuidelineExtractionError
        If neither ``text`` nor ``blocks`` yields content.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if blocks is None:
        if text is None:
            raise GuidelineExtractionError(
                "chunk_guidelines needs either extracted text or document blocks."
            )
        blocks = [
            DocumentBlock(text=paragraph.strip())
            for paragraph in text.split("\n\n")
            if paragraph.strip()
        ]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(config.CHUNK_SEPARATORS),
    )

    chunks: list[str] = []
    current_section = None
    buffer: list[DocumentBlock] = []

    def flush() -> None:
        """Emit the buffered blocks as one or more chunks.

        The heading path is prepended to *every* chunk produced from this group,
        not just the first. Without that, the splitter can push the heading into
        one chunk while the following chunk is filed under the same heading but
        contains unrelated trailing text - which would make the section metadata
        actively misleading rather than merely incomplete.
        """
        nonlocal buffer
        if not buffer:
            return
        # Group by heading so a heading and the content it introduces are
        # flushed together. Joining with a single space fuses the heading into
        # the first sentence; a blank line keeps the source text byte-verbatim.
        heading = buffer[0].heading_path
        body = "\n\n".join(block.text for block in buffer if block.text)

        for piece in splitter.split_text(body):
            cleaned = piece.strip()
            if not cleaned:
                continue
            if heading and not cleaned.startswith(heading):
                # The heading is prepended as its own line so the chunk stays
                # verbatim: joining it inline would fuse the section title to the
                # first body sentence, and the text would no longer match the
                # source document character for character.
                first_line = cleaned.split("\n", 1)[0].strip()
                if not first_line.startswith(heading.split(" > ")[-1][:12]):
                    cleaned = f"{heading}\n{cleaned}"
            chunks.append(cleaned)
        buffer = []

    for block in blocks:
        # A new top-level section always starts a new chunk.
        if block.section and current_section is not None and block.section != current_section:
            flush()
        if block.section:
            current_section = block.section
        buffer.append(block)

    flush()
    return chunks


def chunk_with_metadata(
    docx_path: Any = None,
    chunk_size: int = config.CHUNK_SIZE_CHARS,
    chunk_overlap: int = config.CHUNK_OVERLAP_CHARS,
) -> list[dict[str, Any]]:
    """Chunk the document and attach provenance to every chunk.

    Each returned record carries the verbatim ``text`` plus ``section``,
    ``heading_path``, ``source``, ``document_name``, ``chunk_id``, ``index`` and
    ``token_count``. The section is resolved by finding which source block the
    chunk's opening words came from, so metadata is derived from the actual
    document rather than assigned positionally.

    Parameters
    ----------
    docx_path:
        Source document. Defaults to ``config.GUIDELINE_DOCX``.
    chunk_size, chunk_overlap:
        Character sizes passed through to :func:`chunk_guidelines`.

    Returns
    -------
    list of dict
    """
    blocks = extract_docx_blocks(docx_path)
    chunks = chunk_guidelines(blocks=blocks, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    source_name = Path(docx_path or config.GUIDELINE_DOCX).name

    records: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks):
        section, heading_path = _resolve_section(chunk, blocks)
        records.append(
            {
                "chunk_id": make_chunk_id(chunk),
                "index": index,
                "text": chunk,
                "source": source_name,
                "document_name": config.GUIDELINE_DOCUMENT_NAME,
                "section": section,
                "heading_path": heading_path,
                "token_count": count_tokens(chunk),
            }
        )
    return records


def _resolve_section(chunk: str, blocks: list[DocumentBlock]) -> tuple[str, str]:
    """Find the section a chunk came from, by matching its opening content.

    Chunks begin with their heading path when one was available, so the match is
    usually direct. Falls back to matching the first distinctive line of the chunk
    against the source blocks, then to the nearest preceding section.
    """
    opening = chunk.strip()
    for block in blocks:
        if opening.startswith(block.text):
            return block.section, block.heading_path

    # Match on the longest common prefix of the first body line.
    first_line = opening.split("\n")[-1].strip()[:60]
    if first_line:
        for block in blocks:
            if block.text.startswith(first_line[:30]):
                return block.section, block.heading_path

    return "", ""


def verify_chunk_bounds(
    chunks: list[str],
    min_tokens: int = config.CHUNK_MIN_TOKENS,
    max_tokens: int = config.CHUNK_MAX_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> ChunkingReport:
    """Check the chunks against the size and overlap requirements.

    Enforces the **embedding model's** ceiling unconditionally, because exceeding
    it corrupts the vectors. The lower bound applies to all but the final chunk,
    since a document whose tail is short always produces one shorter chunk.

    Parameters
    ----------
    chunks:
        Chunks from :func:`chunk_guidelines`.
    min_tokens, max_tokens:
        Size bounds, defaulting to the config values.
    overlap_tokens:
        Required minimum shared tokens between consecutive chunks.

    Returns
    -------
    ChunkingReport
    """
    counts = [count_tokens(chunk) for chunk in chunks]

    within_limit = all(count <= config.EMBEDDING_MAX_TOKENS for count in counts)
    in_range = all(
        min_tokens <= count <= max_tokens
        for count in counts[:-1]
    ) and (not counts or counts[-1] <= max_tokens)

    overlap = _minimum_shared_tokens(chunks)
    overlap_ok = overlap >= overlap_tokens or len(chunks) == 1

    sections = []
    for chunk in chunks:
        heading = chunk.split("\n")[0].strip()
        if heading and heading not in sections:
            sections.append(heading)

    return ChunkingReport(
        n_chunks=len(chunks),
        token_counts=counts,
        overlap_tokens=overlap,
        in_bounds=in_range and overlap_ok and within_limit,
        sections_covered=sections,
        within_embedding_limit=within_limit,
    )


def _minimum_shared_tokens(chunks: list[str]) -> int:
    """Smallest token overlap between consecutive chunks.

    Measured on word sets rather than exact sequences: overlap is a property of
    the retriever's window, and a word-level measure is stable against the
    splitter's boundary choices.
    """
    if len(chunks) < 2:
        return 0
    smallest = len(chunks[0].split())
    for previous, current in zip(chunks, chunks[1:]):
        left = set(previous.split())
        right = set(current.split())
        smallest = min(smallest, len(left & right))
    return smallest


# ---------------------------------------------------------------------------
# Step 3 - embeddings
# ---------------------------------------------------------------------------


def build_embedding_function(
    model_name: str = config.EMBEDDING_MODEL_NAME,
) -> Any:
    """Load the sentence-embedding model named in the PRD.

    ``all-MiniLM-L6-v2`` is used: 384 dimensions, roughly 90 MB on disk, and it
    runs entirely locally, so no external embedding API is required. The model is
    downloaded once by Hugging Face and then read from the local cache.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier.

    Returns
    -------
    langchain_community.embeddings.HuggingFaceEmbeddings

    Notes
    -----
    The model's ``max_seq_length`` is 256. Chunks are sized under that in
    :func:`chunk_guidelines`; exceeding it would silently drop the tail of a chunk.
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=model_name)


def embedding_dimension(function: Any) -> int:
    """Width of the vectors ``function`` produces, measured by embedding a probe."""
    return len(function.embed_query("dimension probe"))


# ---------------------------------------------------------------------------
# Step 4 - ChromaDB
# ---------------------------------------------------------------------------


def make_chunk_id(chunk_text: str) -> str:
    """Deterministic ID for a chunk, derived from its text.

    Hashing the text means re-ingesting the same document rewrites the same IDs
    instead of appending new ones, which is what makes ingestion idempotent.

    Returns
    -------
    str
        ``<prefix>-<sha256[:16]>``.
    """
    digest = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
    return f"{config.CHUNK_ID_PREFIX}-{digest[: config.CHUNK_ID_HASH_LENGTH]}"


@dataclass(frozen=True)
class GuidelineIndex:
    """Handle on a persisted ChromaDB collection plus its provenance.

    Attributes
    ----------
    collection:
        The live Chroma collection.
    persist_dir:
        Directory on disk. Survives process restart (FR-3.3).
    collection_name:
        ``config.GUIDELINE_COLLECTION_NAME``.
    n_chunks:
        Number of chunks stored.
    embedding_model_name:
        Model that produced the vectors, recorded in collection metadata and
        checked on load.
    embedding_dim:
        Vector width, e.g. 384.
    chunk_token_counts:
        Per-chunk token counts, kept as the evidence for the sizing decisions.
    source_document:
        Path to the read-only source.
    """

    collection: Any
    persist_dir: Path
    collection_name: str
    n_chunks: int
    embedding_model_name: str
    embedding_dim: int = 0
    chunk_token_counts: list[int] = field(default_factory=list)
    source_document: Path = config.GUIDELINE_DOCX

    def count(self) -> int:
        """Number of records currently in the collection."""
        return int(self.collection.count())

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable summary, for the CLI and the tests."""
        return {
            "collection": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "n_chunks": self.n_chunks,
            "embedding_model": self.embedding_model_name,
            "embedding_dim": self.embedding_dim,
            "source_document": str(self.source_document),
            "source_name": config.GUIDELINE_DOCUMENT_NAME,
            "chunk_token_counts": list(self.chunk_token_counts),
            "min_chunk_tokens": min(self.chunk_token_counts, default=0),
            "max_chunk_tokens": max(self.chunk_token_counts, default=0),
        }


def build_index(
    docx_path: Any = None,
    persist_dir: Any = None,
    collection_name: str = config.GUIDELINE_COLLECTION_NAME,
    reset: bool = False,
) -> GuidelineIndex:
    """Extract, chunk, embed and persist the guideline index end to end.

    Implements FR-3.1 through FR-3.3 in one call.

    Ids are deterministic (see :func:`make_chunk_id`), so running this twice
    updates the same records rather than duplicating them. ``reset=True`` deletes
    the collection first, which is required when the chunking or embedding
    configuration changes - a collection cannot hold vectors from two models
    because their similarity spaces are not comparable.

    Parameters
    ----------
    docx_path:
        Read-only guideline source. Defaults to ``config.GUIDELINE_DOCX``.
    persist_dir:
        Where the index is written. Defaults to ``config.CHROMA_PERSIST_DIR``
        (``vectorstore/``).
    collection_name:
        Chroma collection name.
    reset:
        Delete any existing collection before writing.

    Returns
    -------
    GuidelineIndex
        A persisted, reloadable handle.

    Raises
    ------
    ChunkingError
        If a chunk would exceed the embedding model's limit.
    IndexingError
        If the collection cannot be written.
    """
    import chromadb

    directory = Path(persist_dir) if persist_dir else config.CHROMA_PERSIST_DIR
    directory.mkdir(parents=True, exist_ok=True)

    records = chunk_with_metadata(docx_path)
    report = verify_chunk_bounds([record["text"] for record in records])
    if not report.within_embedding_limit:
        raise ChunkingError(
            "Chunking exceeds the embedding model's maximum sequence length: "
            + "; ".join(report.failures())
        )

    embedding_function = build_embedding_function()
    vectors = embedding_function.embed_documents([r["text"] for r in records])
    dimension = len(vectors[0]) if vectors else 0

    _assert_no_embedding_collision(vectors, [r["text"] for r in records])

    try:
        client = chromadb.PersistentClient(path=str(directory))
        if reset:
            try:
                client.delete_collection(collection_name)
            except Exception:  # noqa: BLE001 - absent collection is fine
                pass
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                config.CHROMA_COLLECTION_METADATA_KEY: config.EMBEDDING_MODEL_NAME,
                "source": config.GUIDELINE_DOCUMENT_NAME,
                "embedding_dim": dimension,
                "chunk_count": len(records),
            },
        )
        collection.upsert(
            ids=[r["chunk_id"] for r in records],
            embeddings=[list(map(float, vector)) for vector in vectors],
            documents=[r["text"] for r in records],
            metadatas=[
                {
                    "source": r["source"],
                    "document_name": r["document_name"],
                    "section": r["section"],
                    "heading_path": r["heading_path"],
                    "index": r["index"],
                    "token_count": r["token_count"],
                }
                for r in records
            ],
        )
    except ChunkingError:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced with actionable context
        raise IndexingError(
            f"Failed to write the ChromaDB index to {directory}: {exc}"
        ) from exc

    return GuidelineIndex(
        collection=collection,
        persist_dir=directory,
        collection_name=collection_name,
        n_chunks=len(records),
        embedding_model_name=config.EMBEDDING_MODEL_NAME,
        embedding_dim=dimension,
        chunk_token_counts=[r["token_count"] for r in records],
        source_document=Path(docx_path) if docx_path else config.GUIDELINE_DOCX,
    )


def _assert_no_embedding_collision(vectors: list[list[float]], texts: list[str]) -> None:
    """Fail if two distinct chunks embedded to (near-)identical vectors.

    Identical vectors mean the embedding model truncated both chunks to the same
    prefix, discarding their distinguishing content. That failure is invisible in
    a similarity search but silently removes chunks from consideration, so it is
    checked here rather than discovered later.
    """
    import numpy as np

    if len(vectors) < 2:
        return
    matrix = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalised = matrix / np.maximum(norms, 1e-12)
    similarity = normalised @ normalised.T

    np.fill_diagonal(similarity, -1.0)
    best = float(similarity.max())
    if best >= config.EMBEDDING_COLLISION_THRESHOLD:
        a, b = np.unravel_index(int(similarity.argmax()), similarity.shape)
        raise ChunkingError(
            f"Two distinct chunks embedded to nearly identical vectors "
            f"(cosine {best:.8f}), which means the embedding model truncated "
            f"both past their distinguishing content. Affected chunks: "
            f"{texts[a][:70]!r} and {texts[b][:70]!r}. Reduce CHUNK_SIZE_TOKENS."
        )


def load_index(
    persist_dir: Any = None,
    collection_name: str = config.GUIDELINE_COLLECTION_NAME,
) -> GuidelineIndex:
    """Reopen a persisted index **without** re-embedding (FR-3.3).

    The acceptance criterion for FR-3.3: querying from a fresh session, with no
    ingest step, must still return correct results.

    Raises
    ------
    IndexingError
        If the collection is absent or was built with a different embedding model.
    """
    import chromadb

    directory = Path(persist_dir) if persist_dir else config.CHROMA_PERSIST_DIR
    if not directory.exists():
        raise IndexingError(
            f"No vector index found at {directory}. "
            "Build it first: python -m src.rag.ingest"
        )

    try:
        client = chromadb.PersistentClient(path=str(directory))
        collection = client.get_collection(name=collection_name)
    except Exception as exc:  # noqa: BLE001 - surfaced with actionable context
        raise IndexingError(
            f"Could not open collection '{collection_name}' at {directory}: {exc}. "
            "Rebuild it: python -m src.rag.ingest --reset"
        ) from exc

    stored_model = (collection.metadata or {}).get(config.CHROMA_COLLECTION_METADATA_KEY)
    if stored_model and stored_model != config.EMBEDDING_MODEL_NAME:
        raise IndexingError(
            f"The index was built with embedding model '{stored_model}' but the "
            f"configuration now specifies '{config.EMBEDDING_MODEL_NAME}'. "
            "Vectors from different models are not comparable; rebuild with "
            "python -m src.rag.ingest --reset"
        )

    return GuidelineIndex(
        collection=collection,
        persist_dir=directory,
        collection_name=collection_name,
        n_chunks=int(collection.count()),
        embedding_model_name=stored_model or config.EMBEDDING_MODEL_NAME,
        embedding_dim=int((collection.metadata or {}).get("embedding_dim", 0)),
    )


def export_index_compatibility_copy(
    index: GuidelineIndex, target_dir: Any = None
) -> Path:
    """Copy the persisted index to ``outputs/chroma_store/``.

    The PRD's Week 3 Definition of Done names ``outputs/chroma_store/`` while this
    repository's architecture rule 11 assigns the vector index to ``vectorstore/``.
    ``vectorstore/`` is canonical; this writes the PRD-named copy so both documents
    hold.

    Parameters
    ----------
    index:
        The live index whose ``persist_dir`` is copied.
    target_dir:
        Destination. Defaults to ``config.PRD_COMPAT_VECTORSTORE_DIR``.

    Returns
    -------
    pathlib.Path
        The directory written.
    """
    destination = (
        Path(target_dir) if target_dir else config.PRD_COMPAT_VECTORSTORE_DIR
    )
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(index.persist_dir, destination)
    return destination


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Build and verify the index from the command line.

    Usage
    -----
    ``python -m src.rag.ingest``
        Extract, chunk, embed and persist the index.

    ``--reset``
        Delete the collection first. Required after changing the chunk size or the
        embedding model.

    ``--no-export``
        Skip the ``outputs/chroma_store/`` compatibility copy.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.rag.ingest",
        description="Ingest the mRNA safety guideline DOCX into ChromaDB.",
    )
    parser.add_argument("--docx", default=None, help="Override the source document")
    parser.add_argument("--persist-dir", default=None, help="Override vectorstore/")
    parser.add_argument(
        "--collection", default=config.GUIDELINE_COLLECTION_NAME
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the collection before writing (use after config changes)",
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip writing the outputs/chroma_store compatibility copy",
    )
    args = parser.parse_args(argv)

    print("=" * 74)
    print("REGULATORY RAG  -  document ingestion")
    print("=" * 74)

    blocks = extract_docx_blocks(args.docx)
    outline = document_outline(blocks)
    print(f"Source document   : {outline.title}")
    print(f"Blocks extracted  : {outline.n_paragraphs}")
    print(f"Approx tokens     : {outline.total_tokens}")
    print(f"Top-level sections: {len(outline.sections)}")
    for section in outline.sections:
        print(f"    - {section}")
    print(f"Sub-sections      : {len(outline.subsections)}")
    print()

    index = build_index(
        docx_path=args.docx,
        persist_dir=args.persist_dir,
        collection_name=args.collection,
        reset=args.reset,
    )

    records = chunk_with_metadata(args.docx)
    report = verify_chunk_bounds([r["text"] for r in records])

    print("CHUNKING")
    print(f"  chunks produced     : {report.n_chunks}")
    print(f"  token range         : {min(report.token_counts)}"
          f"-{max(report.token_counts)}")
    print(f"  size target         : {config.CHUNK_MIN_TOKENS}"
          f"-{config.CHUNK_MAX_TOKENS} tokens")
    print(f"  embedding limit     : {config.EMBEDDING_MAX_TOKENS} tokens "
          f"(all-MiniLM-L6-v2 max_seq_length)")
    print(f"  within limit        : {report.within_embedding_limit}")
    print(f"  minimum overlap     : {report.overlap_tokens} shared tokens")
    print(f"  FR-3.1 asked 300-500; sized lower so the embedding model does not "
          f"truncate a chunk tail. See the module docstring.")
    print()

    print("EMBEDDINGS")
    print(f"  model               : {index.embedding_model_name}")
    print(f"  dimension           : {index.embedding_dim}")
    print()

    print("CHROMADB")
    print(f"  collection          : {index.collection_name}")
    print(f"  persist directory   : {index.persist_dir}")
    print(f"  records stored      : {index.count()}")
    print()

    if not args.no_export:
        exported = export_index_compatibility_copy(index)
        print(f"  PRD-named copy      : {exported}")

    print()
    print("INGEST COMPLETE. Query it with: python -m src.rag.retrieve --query \"...\"")
    print(f"  {config.REQUIRED_DISCLAIMER}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())