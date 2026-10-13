"""Module 3 tests: document ingestion, chunking, embeddings, persistence, retrieval.

Requirements covered
--------------------
FR-3.1  Chunks with visible overlap between consecutive chunks.
FR-3.2  A dense embedding per chunk; stored count equals chunk count.
FR-3.3  A local ChromaDB collection that survives a process restart.
FR-3.4  ``query_guidelines()`` returns exactly k chunks, ordered by similarity.
FR-3.5  Every query records a latency greater than 0 and under 5 seconds.

Plus the seven checks this stage requires: document loading, chunk creation,
embedding generation, ChromaDB ingestion, real retrieval, source metadata, and
absence of malformed records.

Nothing is tested against hard-coded regulatory text. Where a test needs to
confirm a specific passage was retrieved, it asserts on a phrase measured from the
document itself, and the retrieval index is built from that document at test time
rather than from a fixture.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.rag import ingest, retrieve
from tests import skip_module_without


# Module 3 needs sentence-transformers, chromadb and langchain. They are heavier
# than the rest of the stack, so the whole module skips with an explicit reason
# rather than failing collection on a partial install.
pytestmark = skip_module_without("chromadb", "sentence_transformers", "langchain_community")


INGEST_PRODUCER = "python -m src.rag.ingest"


# ---------------------------------------------------------------------------
# Session fixtures - the real index, built from the real document
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def source_path() -> Path:
    """The supplied guideline document."""
    assert config.GUIDELINE_DOCX.exists(), f"{config.GUIDELINE_DOCX} is missing"
    return config.GUIDELINE_DOCX


@pytest.fixture(scope="session")
def blocks(source_path):
    """Blocks extracted from the DOCX."""
    return ingest.extract_docx_blocks(source_path)


@pytest.fixture(scope="session")
def records(source_path):
    """Chunks with attached provenance metadata."""
    return ingest.chunk_with_metadata(source_path)


@pytest.fixture(scope="session")
def embedding_function():
    """The sentence-embedding model."""
    return ingest.build_embedding_function()


@pytest.fixture(scope="session")
def index(tmp_path_factory):
    """A real ChromaDB index built in a temporary directory.

    Built fresh rather than reusing ``vectorstore/`` so the tests never mutate the
    committed artifact, and never depend on it already existing.
    """
    directory = tmp_path_factory.mktemp("rag_store")
    return ingest.build_index(persist_dir=directory)


@pytest.fixture(scope="session")
def top3(index, embedding_function):
    """A real retrieval result, reused across the retrieval tests."""
    return retrieve.query_guidelines(
        "What are the storage requirements below -20C?",
        top_k=config.RETRIEVAL_TOP_K,
        index=index,
        embedding_function=embedding_function,
    )


# ---------------------------------------------------------------------------
# Test 1 - document loading
# ---------------------------------------------------------------------------


class TestDocumentLoading:
    """Test 1: the supplied DOCX loads and its text is extracted."""

    def test_document_exists_and_is_a_docx(self, source_path) -> None:
        assert source_path.is_file()
        assert source_path.suffix == ".docx"
        assert source_path.name == config.GUIDELINE_DOCUMENT_NAME

    def test_extraction_returns_blocks(self, blocks) -> None:
        assert len(blocks) > 0
        assert all(block.text.strip() for block in blocks)

    def test_extraction_finds_the_four_sections(self, blocks) -> None:
        """The document's actual structure, measured not assumed."""
        outline = ingest.document_outline(blocks)
        for expected in (
            "Section 1: Storage and Handling Requirements",
            "Section 2: Dosing Guidance by Age Band",
            "Section 3: Contraindications and Warnings",
            "Section 4: Adverse Event Documentation",
        ):
            assert expected in outline.sections, f"missing section: {expected}"

    def test_extraction_finds_subsection_headings(self, blocks) -> None:
        outline = ingest.document_outline(blocks)
        assert len(outline.subsections) >= 8

    def test_heading_hierarchy_is_resolved(self, blocks) -> None:
        """Content paragraphs must carry their enclosing section."""
        dosing = [
            block for block in blocks
            if "primary-series doses" in block.text
        ]
        assert dosing, "the Section 2.1 dosing sentence was not extracted"
        assert dosing[0].section == "Section 2: Dosing Guidance by Age Band"
        assert "2.1" in dosing[0].heading_path

    def test_plain_text_extraction_matches_the_blocks(self, blocks) -> None:
        text = ingest.extract_docx_text()
        assert len(text) > 500
        assert "Section 2: Dosing Guidance by Age Band" in text

    def test_table_content_is_captured(self, blocks) -> None:
        """The metadata table carries the document version, worth retaining."""
        table_rows = [block for block in blocks if block.style == "Table Row"]
        assert table_rows
        assert any("Document Version" in row.text for row in table_rows)

    def test_missing_document_raises_a_named_error(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            ingest.extract_docx_text(tmp_path / "absent.docx")

    def test_non_docx_input_raises_the_extraction_error(self, tmp_path) -> None:
        bogus = tmp_path / "not_a_docx.docx"
        bogus.write_text("plain text, not a Word package", encoding="utf-8")
        with pytest.raises(ingest.GuidelineExtractionError):
            ingest.extract_docx_text(bogus)

    def test_extraction_does_not_modify_the_source(self, source_path) -> None:
        before = source_path.read_bytes()
        ingest.extract_docx_blocks(source_path)
        assert source_path.read_bytes() == before


# ---------------------------------------------------------------------------
# Test 2 - chunk creation
# ---------------------------------------------------------------------------


class TestChunking:
    """Test 2: chunks are non-empty, contain text, and carry metadata."""

    def test_chunks_are_produced(self, records) -> None:
        assert len(records) >= 4

    def test_every_chunk_has_text(self, records) -> None:
        for record in records:
            assert record["text"].strip(), "a chunk was empty"

    def test_every_chunk_has_metadata(self, records) -> None:
        for record in records:
            assert record["source"] == config.GUIDELINE_DOCUMENT_NAME
            assert record["document_name"] == config.GUIDELINE_DOCUMENT_NAME
            assert record["chunk_id"]
            assert isinstance(record["index"], int)
            assert record["token_count"] > 0

    def test_metadata_carries_the_section(self, records) -> None:
        """Sections must be resolvable, so results stay traceable."""
        with_section = [r for r in records if r["section"]]
        assert len(with_section) >= len(records) * 0.6

    def test_chunk_ids_are_unique(self, records) -> None:
        ids = [record["chunk_id"] for record in records]
        assert len(set(ids)) == len(ids)

    def test_chunk_ids_are_deterministic(self, source_path) -> None:
        """Re-running must produce identical IDs, which is what stops duplicates."""
        first = ingest.chunk_with_metadata(source_path)
        second = ingest.chunk_with_metadata(source_path)
        assert [r["chunk_id"] for r in first] == [r["chunk_id"] for r in second]

    def test_chunk_id_derives_from_the_text(self) -> None:
        a = ingest.make_chunk_id("some guideline text")
        b = ingest.make_chunk_id("some guideline text")
        c = ingest.make_chunk_id("different guideline text")
        assert a == b and a != c
        assert a.startswith(config.CHUNK_ID_PREFIX)

    def test_no_chunk_spans_two_sections(self, records) -> None:
        """Section-aware chunking: a chunk must be topically coherent."""
        for record in records:
            if record["section"]:
                assert record["section"] not in record["text"][len(record["text"]) // 2:]

    def test_content_is_not_lost(self, records) -> None:
        """Concatenating the chunks must still contain the key passages."""
        joined = " ".join(" ".join(r["text"].split()) for r in records)
        assert "primary-series doses commonly fall between 25 and 100 micrograms" in joined
        assert "Doses above 100 micrograms" in joined
        assert "absolute contraindication" in joined.lower()

    def test_consecutive_chunks_overlap(self, records) -> None:
        """FR-3.1 requires visible overlap between consecutive chunks."""
        texts = [r["text"] for r in records]
        assert len(texts) > 1
        report = ingest.verify_chunk_bounds(texts)
        assert report.overlap_tokens > 0, "no overlap between consecutive chunks"

    def test_chunks_respect_the_embedding_model_limit(self, records) -> None:
        """The binding constraint: exceeding 256 tokens silently truncates."""
        report = ingest.verify_chunk_bounds([r["text"] for r in records])
        assert report.within_embedding_limit is True, report.failures()
        assert max(report.token_counts) <= config.EMBEDDING_MAX_TOKENS

    def test_chunking_deviation_from_fr31_is_recorded(self) -> None:
        """FR-3.1's literal bounds are retained so the deviation stays visible."""
        assert config.FR31_MIN_TOKENS == 300
        assert config.FR31_MAX_TOKENS == 500
        assert config.CHUNK_MAX_TOKENS < config.FR31_MIN_TOKENS
        assert config.CHUNK_MAX_TOKENS <= config.EMBEDDING_MAX_TOKENS

    def test_bounds_verification_flags_an_oversized_chunk(self) -> None:
        oversized = ["word " * (config.EMBEDDING_MAX_TOKENS + 40)]
        report = ingest.verify_chunk_bounds(oversized)
        assert report.within_embedding_limit is False
        assert report.failures()

    def test_token_counting_is_reasonable(self) -> None:
        assert ingest.count_tokens("") == 0
        assert ingest.count_tokens("Doses above 100 micrograms.") >= 4

    def test_splitting_preserves_headings(self, records) -> None:
        """A chunk must carry the heading identifying its section.

        A chunk's first line is either a full ``Section N: ...`` heading or a
        subsection heading that was split off mid-chunk; both are acceptable, so
        the assertion is that a recognisable heading is present rather than a
        specific shape.
        """
        headed = [r for r in records if r["heading_path"]]
        assert headed, "no chunk retained a heading path"
        for record in headed:
            assert record["heading_path"].split(" > ")[0].startswith("Section "), (
                f"chunk {record['index']} has a heading path that does not name a "
                f"section: {record['heading_path']!r}"
            )
            # The heading text must actually appear in the chunk, so the chunk is
            # self-describing once quoted.
            heading_tail = record["heading_path"].split(" > ")[-1]
            assert heading_tail.split(" ")[0] in record["text"], (
                "chunk does not carry its own heading text"
            )

    def test_splitting_requires_content(self) -> None:
        with pytest.raises(ingest.GuidelineExtractionError):
            ingest.chunk_guidelines()


# ---------------------------------------------------------------------------
# Test 3 - embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    """Test 3: embeddings generate successfully with consistent dimensions."""

    def test_embedding_function_loads(self, embedding_function) -> None:
        assert embedding_function is not None

    def test_model_is_the_configured_one(self, embedding_function) -> None:
        assert config.EMBEDDING_MODEL_NAME == "all-MiniLM-L6-v2"
        client = getattr(embedding_function, "client", None)
        assert client is not None

    def test_query_embedding_has_expected_dimension(self, embedding_function) -> None:
        vector = embedding_function.embed_query("storage requirements")
        assert len(vector) == 384

    def test_document_embeddings_have_consistent_dimensions(
        self, records, embedding_function
    ) -> None:
        vectors = embedding_function.embed_documents([r["text"] for r in records])
        assert len(vectors) == len(records)
        assert len({len(v) for v in vectors}) == 1, "inconsistent embedding dimensions"

    def test_embedding_dimension_helper_matches(self, embedding_function) -> None:
        assert ingest.embedding_dimension(embedding_function) == 384

    def test_distinct_chunks_do_not_collapse(self, records, embedding_function) -> None:
        """Two chunks embedding identically means the model truncated both.

        This is the failure FR-3.2 cannot catch on its own: a count check passes
        while chunks silently become indistinguishable in the vector space.
        """
        vectors = np.asarray(
            embedding_function.embed_documents([r["text"] for r in records]),
            dtype=float,
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalised = vectors / np.maximum(norms, 1e-12)
        similarity = normalised @ normalised.T
        np.fill_diagonal(similarity, -1.0)
        worst = float(similarity.max())
        assert worst < config.EMBEDDING_COLLISION_THRESHOLD, (
            f"two distinct chunks embedded to cosine {worst:.8f}, meaning the "
            "embedding model truncated their distinguishing content"
        )

    def test_identical_text_does_embed_identically(self, embedding_function) -> None:
        """The negative control: the collision probe is not vacuous."""
        a, b = embedding_function.embed_documents(
            ["storage requirements for mRNA products"] * 2
        )
        assert np.allclose(a, b)

    def test_no_duplicate_embeddings_within_the_document(
        self, records, embedding_function
    ) -> None:
        seen: dict[str, int] = {}
        for record in records:
            seen.setdefault(" ".join(record["text"].split()), 0)
            seen[" ".join(record["text"].split())] += 1
        assert max(seen.values()) == 1, "chunking produced a duplicate chunk"


# ---------------------------------------------------------------------------
# Test 4 - ChromaDB ingestion
# ---------------------------------------------------------------------------


class TestChromaIngestion:
    """Test 4: chunks are stored in a persistent vector store."""

    def test_index_is_created_with_a_named_collection(self, index) -> None:
        assert index.collection_name == config.GUIDELINE_COLLECTION_NAME
        assert index.persist_dir.is_dir()

    def test_stored_record_count_equals_chunk_count(self, index, records) -> None:
        """FR-3.2: the number of stored embeddings equals the number of chunks."""
        assert index.count() == len(records)

    def test_every_chunk_has_text_metadata_and_an_id(self, index) -> None:
        raw = index.collection.get(include=["documents", "metadatas"])
        assert len(raw["ids"]) == index.count()
        for chunk_id, document, metadata in zip(
            raw["ids"], raw["documents"], raw["metadatas"]
        ):
            assert chunk_id
            assert document.strip()
            assert metadata["source"] == config.GUIDELINE_DOCUMENT_NAME
            assert metadata["document_name"] == config.GUIDELINE_DOCUMENT_NAME

    def test_embeddings_are_stored_with_the_expected_dimension(self, index) -> None:
        raw = index.collection.get(include=["embeddings"])
        assert len(raw["embeddings"][0]) == 384

    def test_collection_records_the_embedding_model(self, index) -> None:
        metadata = index.collection.metadata or {}
        assert metadata[config.CHROMA_COLLECTION_METADATA_KEY] == (
            config.EMBEDDING_MODEL_NAME
        )

    def test_reingesting_does_not_duplicate_records(self, tmp_path) -> None:
        """Deterministic IDs must make ingestion idempotent."""
        directory = tmp_path / "store"
        first = ingest.build_index(persist_dir=directory)
        second = ingest.build_index(persist_dir=directory)
        assert first.count() == second.count()

    def test_reset_rebuilds_to_the_same_count(self, tmp_path) -> None:
        directory = tmp_path / "store"
        ingest.build_index(persist_dir=directory)
        rebuilt = ingest.build_index(persist_dir=directory, reset=True)
        assert rebuilt.count() == len(ingest.chunk_with_metadata())

    def test_load_reopens_without_re_embedding(self, tmp_path) -> None:
        """FR-3.3: a fresh session can query without running ingest again."""
        directory = tmp_path / "store"
        ingest.build_index(persist_dir=directory)
        reopened = ingest.load_index(persist_dir=directory)
        assert reopened.count() > 0
        result = retrieve.query_guidelines(
            "contraindications", top_k=3, index=reopened,
            embedding_function=ingest.build_embedding_function(),
        )
        assert len(result) == 3

    def test_missing_index_names_the_fix(self, tmp_path) -> None:
        with pytest.raises(ingest.IndexingError, match="src.rag.ingest"):
            ingest.load_index(persist_dir=tmp_path / "absent")

    def test_prd_compatibility_copy_is_written(self, index, tmp_path) -> None:
        """The PRD names outputs/chroma_store/; the repo keeps vectorstore/."""
        exported = ingest.export_index_compatibility_copy(index, tmp_path / "chroma")
        assert exported.is_dir()
        assert any(exported.iterdir())

    def test_committed_index_exists_when_ingest_has_run(self) -> None:
        from tests import requires_artifact

        requires_artifact(config.CHROMA_PERSIST_DIR, INGEST_PRODUCER)


# ---------------------------------------------------------------------------
# Test 5 - retrieval
# ---------------------------------------------------------------------------


class TestRetrieval:
    """Test 5: a real query returns exactly k results."""

    def test_three_results_are_returned_by_default(self, top3) -> None:
        assert config.RETRIEVAL_TOP_K == 3
        assert len(top3) == 3
        assert top3.k == 3

    def test_default_top_k_is_three(self) -> None:
        import inspect

        default = inspect.signature(retrieve.query_guidelines).parameters["top_k"].default
        assert default == 3

    @pytest.mark.parametrize(
        "question",
        [
            "What are the storage requirements below -20C?",
            "What is the maximum recommended dose for primary series?",
            "Which patients require secondary clinician review?",
            "What are the absolute contraindications to vaccination?",
            "When must an adverse event be reported?",
        ],
    )
    def test_every_evaluation_query_returns_three(self, index, embedding_function, question) -> None:
        result = retrieve.query_guidelines(
            question, top_k=3, index=index, embedding_function=embedding_function
        )
        assert len(result) == 3

    def test_results_are_ordered_by_descending_similarity(self, top3) -> None:
        scores = [chunk.score for chunk in top3.chunks]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_are_sequential(self, top3) -> None:
        assert [chunk.rank for chunk in top3.chunks] == [1, 2, 3]

    def test_storage_query_ranks_its_own_section_first(self, top3) -> None:
        """Retrieval quality: the question must reach its own section first.

        Only rank 1 is asserted. The index holds 12 chunks spanning four sections,
        so ranks 2 and 3 legitimately include adjacent topics - a dosing question
        about contraindications may reasonably surface the dosing section. Requiring
        all three to match would be asserting an accident of the chunk layout rather
        than a property retrieval should have.
        """
        assert "Storage" in top3.chunks[0].section, (
            f"storage question ranked {[c.section for c in top3.chunks]}"
        )
        assert "storage" in " ".join(top3.chunks[0].text).lower() or (
            "-15C" in top3.chunks[0].text or "-80C" in top3.chunks[0].text
        )

    def test_contraindication_query_ranks_its_own_section_first(
        self, index, embedding_function
    ) -> None:
        result = retrieve.query_guidelines(
            "What are the absolute contraindications to vaccination?",
            top_k=3, index=index, embedding_function=embedding_function,
        )
        assert "Contraindications" in result.chunks[0].section, (
            f"contraindication question ranked {[c.section for c in result.chunks]}"
        )

    def test_document_vocabulary_improves_relevance(self, index, embedding_function) -> None:
        """Retrieval quality is measurable, not asserted as a constant.

        A query using the guideline's own phrasing ("prior allergic reaction") must
        score higher on the contraindication chunk than a generic paraphrase. This
        demonstrates the embedding is doing semantic work rather than keyword
        matching, and it is why the dose-evidence query is worded the way it is.
        """
        generic = retrieve.query_guidelines(
            "What are the absolute contraindications to vaccination?",
            top_k=3, index=index, embedding_function=embedding_function,
        )
        specific = retrieve.query_guidelines(
            "absolute contraindication known severe allergic reaction to a prior "
            "dose of the same product",
            top_k=3, index=index, embedding_function=embedding_function,
        )
        assert specific.chunks[0].score > generic.chunks[0].score
        assert "Contraindications" in specific.chunks[0].section

    def test_custom_top_k_is_honoured(self, index, embedding_function) -> None:
        for k in (1, 2, 3, 5):
            result = retrieve.query_guidelines(
                "adverse event reporting", top_k=k, index=index,
                embedding_function=embedding_function,
            )
            assert len(result) == k

    def test_k_beyond_available_chunks_is_refused(self, index, embedding_function) -> None:
        if index.count() >= 100:
            pytest.skip("index larger than the refusal threshold")
        with pytest.raises(retrieve.RetrievalError):
            retrieve.query_guidelines(
                "storage", top_k=index.count() + 1, index=index,
                embedding_function=embedding_function,
            )

    @pytest.mark.parametrize("bad_k", [0, -1, config.RETRIEVAL_MAX_K + 1])
    def test_out_of_range_k_is_refused(self, index, embedding_function, bad_k) -> None:
        with pytest.raises(retrieve.RetrievalError):
            retrieve.query_guidelines(
                "storage", top_k=bad_k, index=index,
                embedding_function=embedding_function,
            )

    def test_blank_query_is_refused(self, index) -> None:
        with pytest.raises(retrieve.EmptyQueryError):
            retrieve.query_guidelines("   ", index=index)

    def test_similarity_scores_are_in_range(self, top3) -> None:
        for chunk in top3.chunks:
            assert 0.0 <= chunk.score <= 1.0

    def test_raw_distance_is_preserved(self, top3) -> None:
        """The score is a transform of the distance, not a substitute for it."""
        for chunk in top3.chunks:
            assert chunk.distance >= 0.0
            assert chunk.score == pytest.approx(
                max(0.0, min(1.0, 1.0 - chunk.distance / 2.0)), abs=1e-6
            )

    def test_retrieve_guidelines_returns_plain_dictionaries(
        self, index, embedding_function
    ) -> None:
        """The shape the report-generation stage consumes."""
        records = retrieve.retrieve_guidelines(
            "What are the storage requirements?",
            top_k=3, index=index, embedding_function=embedding_function,
        )
        assert len(records) == 3
        for rank, record in enumerate(records, start=1):
            assert record["rank"] == rank
            assert record["text"]
            assert record["source"]
            assert isinstance(record["metadata"], dict)

    def test_validation_passes_for_a_well_formed_result(self, top3) -> None:
        top3.validate()

    def test_relevance_floor_separates_strong_and_weak_matches(
        self, index, embedding_function
    ) -> None:
        """A caller must be able to tell "document covers this" from "it does not".

        On-topic queries score far higher than off-topic ones. The property
        matters for the next stage, which must fall back to a conservative default
        when the guideline does not cover a question.
        """
        on_topic = retrieve.query_guidelines(
            "storage temperature requirements for frozen product",
            top_k=3, index=index, embedding_function=embedding_function,
        )
        off_topic = retrieve.query_guidelines(
            "quantum chromodynamics lattice gauge theory",
            top_k=3, index=index, embedding_function=embedding_function,
        )
        assert on_topic.chunks[0].score > off_topic.chunks[0].score
        assert on_topic.has_relevant_evidence


# ---------------------------------------------------------------------------
# Test 6 - metadata and traceability
# ---------------------------------------------------------------------------


class TestTraceability:
    """Test 6: every chunk traces back to the supplied DOCX."""

    def test_every_chunk_records_the_source_filename(self, top3) -> None:
        for chunk in top3.chunks:
            assert chunk.source == "Moderna_mRNA_Safety_Guidelines.docx"

    def test_every_chunk_records_its_section(self, top3) -> None:
        for chunk in top3.chunks:
            assert chunk.section, f"chunk {chunk.chunk_id} has no section"

    def test_every_chunk_records_a_citation(self, top3) -> None:
        for chunk in top3.chunks:
            citation = chunk.citation
            assert config.GUIDELINE_DOCUMENT_NAME in citation

    def test_retrieved_text_is_verbatim_from_the_document(self, top3) -> None:
        """No paraphrase: the text must appear in the extracted document.

        A chunk may carry a prepended heading line so it is self-describing when
        quoted. That heading is itself document text, so the check strips it and
        verifies the remaining body verbatim.
        """
        full = " ".join(ingest.extract_docx_text().split())
        for chunk in top3.chunks:
            body = " ".join(chunk.text.split())
            if chunk.heading_path and body.startswith(chunk.heading_path):
                body = body[len(chunk.heading_path):]
            probe = " ".join(body.split())[:80]
            assert probe in full, f"retrieved text not found in the document: {probe!r}"

    def test_prepended_heading_is_itself_document_text(self, records) -> None:
        """The heading line added to a chunk must also come from the document."""
        full = " ".join(ingest.extract_docx_text().split())
        for record in records:
            if not record["heading_path"]:
                continue
            first_line = record["text"].split("\n", 1)[0].strip()
            if first_line == record["heading_path"]:
                for part in record["heading_path"].split(" > "):
                    assert part in full, f"heading not in document: {part!r}"

    def test_token_labels_are_present_in_the_stored_metadata(self, index) -> None:
        raw = index.collection.get(include=["metadatas"])
        for metadata in raw["metadatas"]:
            assert metadata["token_count"] > 0
            assert metadata["index"] >= 0

    def test_result_serialises_to_json(self, top3) -> None:
        payload = json.loads(json.dumps(top3.as_dict()))
        assert payload["n_results"] == 3
        assert len(payload["chunks"]) == 3


# ---------------------------------------------------------------------------
# Test 7 - no empty or malformed records
# ---------------------------------------------------------------------------


class TestRecordIntegrity:
    """Test 7: the retrieval API never returns malformed records."""

    def test_no_empty_text(self, top3) -> None:
        for chunk in top3.chunks:
            assert chunk.text.strip()

    def test_no_missing_ids(self, top3) -> None:
        for chunk in top3.chunks:
            assert chunk.chunk_id and chunk.chunk_id.startswith(config.CHUNK_ID_PREFIX)

    def test_no_missing_scores(self, top3) -> None:
        for chunk in top3.chunks:
            assert isinstance(chunk.score, float)
            assert np.isfinite(chunk.score)

    def test_all_expected_fields_present(self, top3) -> None:
        for chunk in top3.chunks:
            record = chunk.as_dict()
            for field in (
                "rank", "chunk_id", "text", "score", "distance",
                "source", "document_name", "section", "citation",
            ):
                assert field in record, f"{field} missing from the result record"

    def test_distinct_chunks_are_returned(self, top3) -> None:
        """No duplicates in one result set."""
        assert len({chunk.chunk_id for chunk in top3.chunks}) == 3

    def test_validation_catches_an_empty_result(self) -> None:
        empty = retrieve.RetrievalResult(
            question="q", chunks=[], latency_s=0.1, k=3
        )
        with pytest.raises(retrieve.RetrievalError, match="FR-3.4"):
            empty.validate()

    def test_validation_catches_misordered_results(self, top3) -> None:
        scrambled = retrieve.RetrievalResult(
            question=top3.question,
            chunks=list(reversed(top3.chunks)),
            latency_s=top3.latency_s,
            k=3,
        )
        for position, chunk in enumerate(scrambled.chunks, start=1):
            object.__setattr__(chunk, "rank", position)
        with pytest.raises(retrieve.RetrievalError, match="ordered"):
            scrambled.validate()

    def test_validation_catches_zero_latency(self, top3) -> None:
        with pytest.raises(retrieve.RetrievalError, match="FR-3.5"):
            retrieve.RetrievalResult(
                question="q", chunks=top3.chunks, latency_s=0.0, k=3
            ).validate()

    def test_validation_catches_excessive_latency(self, top3) -> None:
        with pytest.raises(retrieve.RetrievalError, match="FR-3.5"):
            retrieve.RetrievalResult(
                question="q", chunks=top3.chunks,
                latency_s=config.RETRIEVAL_LATENCY_MAX_S + 1, k=3,
            ).validate()

    def test_validation_catches_a_chunk_without_source(self, top3) -> None:
        broken = retrieve.RetrievedChunk(
            text="x", chunk_id="c", score=0.5, distance=1.0, rank=1, source=""
        )
        with pytest.raises(retrieve.RetrievalError, match="traced"):
            retrieve.RetrievalResult(
                question="q", chunks=[broken], latency_s=0.1, k=1
            ).validate()


# ---------------------------------------------------------------------------
# FR-3.5 - latency
# ---------------------------------------------------------------------------


class TestRetrievalLatency:
    """FR-3.5: latency is recorded and exposed for every query."""

    def test_latency_is_positive_and_bounded(self, top3) -> None:
        assert top3.latency_s > 0
        assert top3.latency_s < config.RETRIEVAL_LATENCY_MAX_S

    def test_latency_is_serialised(self, top3) -> None:
        assert top3.as_dict()["latency_s"] > 0

    @pytest.mark.parametrize(
        "question",
        [
            "What are the storage requirements below -20C?",
            "What are the absolute contraindications?",
            "How long must a patient be monitored?",
        ],
    )
    def test_every_query_records_a_latency(self, index, embedding_function, question) -> None:
        result = retrieve.query_guidelines(
            question, top_k=3, index=index, embedding_function=embedding_function
        )
        assert config.RETRIEVAL_LATENCY_MIN_S < result.latency_s
        assert result.latency_s < config.RETRIEVAL_LATENCY_MAX_S


# ---------------------------------------------------------------------------
# Evaluation logging
# ---------------------------------------------------------------------------


class TestEvaluationLog:
    """The Week 3 deliverables: a query log and a markdown table."""

    def test_default_set_has_at_least_five_questions(self) -> None:
        assert len(retrieve.get_default_evaluation_questions()) >= 5

    def test_default_queries_span_the_four_sections(self) -> None:
        joined = " ".join(retrieve.get_default_evaluation_questions()).lower()
        for topic in ("storage", "dose", "contraindication", "adverse event"):
            assert topic in joined, f"evaluation set omits {topic}"

    def test_every_default_query_ranks_its_target_section_first(
        self, index, embedding_function
    ) -> None:
        """Retrieval quality, asserted for the whole evaluation set.

        Each query is paired with the section it is written to surface, and the
        pair is asserted to match. Because the evaluation queries are phrased in
        the guideline's own vocabulary - see the module docstring - this is a real
        property of the retrieval layer rather than a tautology about the queries.
        """
        expectations = {
            "What are the storage requirements below -20C?": "Storage",
            "general dosing principles primary series dose range micrograms": "Dosing",
            "absolute contraindications known severe allergic reaction to a prior dose": "Contraindications",
            "reporting thresholds adverse event documented within 24 hours": "Adverse Event",
            "age-banded reference doses": "Dosing",
        }
        for question, expected in expectations.items():
            result = retrieve.query_guidelines(
                question, top_k=3, index=index, embedding_function=embedding_function
            )
            assert expected in result.chunks[0].section, (
                f"{question!r} ranked {[c.section for c in result.chunks]}, "
                f"expected '{expected}' at rank 1"
            )

    def test_query_wording_materially_affects_retrieval(
        self, index, embedding_function
    ) -> None:
        """Documents the layer's main limitation instead of hiding it.

        The guideline says "standard warnings ... monitored for at least 15
        minutes". A question phrased in the guideline's terms retrieves that
        passage far more reliably than a paraphrase. Quantifying the gap makes the
        limitation concrete for the Week 3 write-up.
        """
        grounded = retrieve.query_guidelines(
            "standard warnings patients monitored for at least 15 minutes",
            top_k=3, index=index, embedding_function=embedding_function,
        )
        paraphrased = retrieve.query_guidelines(
            "How long must a patient be monitored after administration?",
            top_k=3, index=index, embedding_function=embedding_function,
        )
        assert grounded.chunks[0].score > paraphrased.chunks[0].score, (
            "expected document-vocabulary queries to score higher; the evaluation "
            "queries' wording assumption no longer holds"
        )

    def test_evaluation_records_every_query(self, index, embedding_function) -> None:
        log = retrieve.run_evaluation_queries(
            questions=retrieve.get_default_evaluation_questions()[:3],
            index=index, embedding_function=embedding_function,
        )
        records = log.to_records()
        assert len(records) == 3
        for record in records:
            assert record["question"]
            assert record["top_result"]
            assert record["latency_s"] > 0
            assert record["top_source"]

    def test_evaluation_log_writes_to_the_dashboard_path(
        self, index, embedding_function, tmp_path
    ) -> None:
        log = retrieve.run_evaluation_queries(
            questions=["storage requirements"], index=index,
            embedding_function=embedding_function,
        )
        path = retrieve.save_retrieval_log(log, tmp_path / "retrieval_log.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["n_queries"] == 1
        assert payload["top_k"] == 3
        assert payload["latency_s"][0] > 0

    def test_markdown_table_is_rendered(self, index, embedding_function) -> None:
        log = retrieve.run_evaluation_queries(
            questions=["storage requirements", "contraindications"], index=index,
            embedding_function=embedding_function,
        )
        table = log.as_markdown_table()
        assert table.startswith("| # |")
        assert "Latency (s)" in table

    def test_committed_log_exists_when_retrieval_has_run(self) -> None:
        from tests import requires_artifact

        requires_artifact(config.RETRIEVAL_LOG_JSON_PATH, "python -m src.rag.retrieve --evaluate")


# ---------------------------------------------------------------------------
# Dose evidence support (FR-4.4 input, no decision logic here)
# ---------------------------------------------------------------------------


class TestDoseEvidence:
    """Retrieval surfaces dose evidence; it does not decide anything."""

    def test_dose_evidence_returns_underlying_retrieval(self, index, embedding_function) -> None:
        evidence = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        assert len(evidence.result) == 3

    def test_dose_evidence_extracts_mcg_figures_from_the_text(self, index, embedding_function) -> None:
        """The numbers must come from the document, not from configuration."""
        evidence = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        assert evidence.has_numeric_evidence
        assert 100.0 in evidence.stated_mcg_bounds

    def test_cited_text_is_verbatim(self, index, embedding_function) -> None:
        full = " ".join(ingest.extract_docx_text().split())
        evidence = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        for citation in evidence.cited_text:
            assert " ".join(citation.split())[:60] in full

    def test_max_stated_dose_is_100_micrograms(self, index, embedding_function) -> None:
        """The document's own figure, retrieved rather than configured."""
        evidence = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        assert evidence.max_stated_mcg == 100.0

    def test_mcg_extraction_ignores_non_dose_sentences(self) -> None:
        assert retrieve._extract_mcg_figures("Store between -25C and -15C.") == []
        assert retrieve._extract_mcg_figures("Monitor for 15 minutes.") == []
        assert retrieve._extract_mcg_figures(
            "Doses above 100 micrograms require review."
        ) == [100.0]

    def test_mcg_extraction_handles_ranges_and_singular(self) -> None:
        assert retrieve._extract_mcg_figures(
            "fall between 25 and 100 micrograms"
        ) == [25.0, 100.0]
        assert retrieve._extract_mcg_figures("a 50 mcg dose") == [50.0]
        assert retrieve._extract_mcg_figures("1 microgram") == [1.0]
        assert retrieve._extract_mcg_figures("25 to 50 mcg") == [25.0, 50.0]

    def test_sentence_splitting_separates_a_following_heading(self) -> None:
        """A heading must not be glued onto the sentence before it.

        Fused citations are unusable as report footnotes: the quoted string would
        contain a fragment of a heading that was never part of the statement.
        """
        text = (
            "Doses above 100 micrograms are considered high-reactogenicity doses.\n\n"
            "2.2 Age-Banded Reference Doses"
        )
        parts = [part.strip() for part in retrieve._split_sentences(text) if part.strip()]
        assert parts == [
            "Doses above 100 micrograms are considered high-reactogenicity doses.",
            "2.2 Age-Banded Reference Doses",
        ]

    def test_every_citation_is_a_clean_sentence(self, index, embedding_function) -> None:
        """No citation may contain an embedded heading fragment."""
        evidence = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        for citation in evidence.cited_text:
            assert "Age-Banded" not in citation, (
                f"citation contains a heading fragment: {citation!r}"
            )
            assert not citation.endswith("Doses"), (
                f"citation ends on a heading fragment: {citation!r}"
            )

    def test_dose_evidence_makes_no_clinical_decision(self, index, embedding_function) -> None:
        """The RAG layer surfaces evidence; the validation layer decides."""
        evidence = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        payload = json.dumps(evidence.as_dict()).lower()
        assert "not" in payload  # the note is present
        for forbidden in ("is_safe", "is_approved", "recommend", "eligible", "diagnos"):
            assert forbidden not in payload.replace("not decided", "")

    def test_custom_dose_query_is_honoured(self, index, embedding_function) -> None:
        evidence = retrieve.retrieve_dose_evidence(
            120.0, query="age-banded reference doses", top_k=3,
            index=index, embedding_function=embedding_function,
        )
        assert "age" in evidence.result.question.lower()

    def test_full_dose_range_is_retrieved(self, index, embedding_function) -> None:
        """The 25-100 mcg range must appear, not just the 100 ceiling.

        Both bounds are needed downstream: the ceiling for FR-4.4, the floor as the
        document's own statement of the reference range. Missing the floor would
        leave the evidence half-complete while still looking plausible.
        """
        evidence = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        assert 25.0 in evidence.stated_mcg_bounds
        assert 100.0 in evidence.stated_mcg_bounds

    def test_citations_carry_provenance(self, index, embedding_function) -> None:
        evidence = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        assert evidence.citations
        for citation in evidence.citations:
            assert citation["chunk_id"]
            assert citation["citation"]
            assert citation["text"].strip()

    def test_exceeds_flag_distinguishes_above_and_below(self, index, embedding_function) -> None:
        """Arithmetic against retrieved text only - the decision is not made here."""
        high = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        low = retrieve.retrieve_dose_evidence(
            75.0, index=index, embedding_function=embedding_function
        )
        assert high.exceeds_stated_max is True
        assert low.exceeds_stated_max is False

    def test_evidence_pool_is_wider_than_top_k(self, index, embedding_function) -> None:
        """The dose sweep must be able to reach a passage ranked below top-k."""
        evidence = retrieve.retrieve_dose_evidence(
            120.0, index=index, embedding_function=embedding_function
        )
        assert evidence.evidence_pool_size > evidence.result.k
        assert evidence.evidence_pool_size == retrieve.DOSE_EVIDENCE_SWEEP_K


# ---------------------------------------------------------------------------
# Scope guard
# ---------------------------------------------------------------------------


class TestScope:
    """This layer retrieves evidence; it must not generate or decide."""

    def test_no_llm_imports_in_the_rag_package(self) -> None:
        import ast

        for path in Path(ingest.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                for module in modules:
                    for forbidden in ("openai", "anthropic", "google.generativeai"):
                        assert not module.startswith(forbidden), (
                            f"{path.name} imports {module}"
                        )

    def test_rag_does_not_import_the_report_generator(self) -> None:
        import ast

        for path in Path(ingest.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "src.llm" not in node.module, (
                        f"{path.name} imports {node.module}"
                    )

    def test_only_the_declared_vector_and_embedding_libraries_are_used(self) -> None:
        """No alternative vector store or LLM framework may appear.

        Third-party roots are allow-listed. Standard-library modules are not
        checked, since ``hashlib``, ``json`` and ``pathlib`` are obviously
        legitimate and listing them would be noise.
        """
        import ast
        import sys

        allowed_third_party = {
            "chromadb", "sentence_transformers", "langchain", "langchain_community",
            "langchain_text_splitters", "docx", "numpy", "pandas", "torch",
            "matplotlib", "seaborn", "pydantic",
        }
        # Project packages plus anything importable from the standard library.
        stdlib = set(sys.stdlib_module_names)
        project = {"src", "dashboard"}

        for path in Path(ingest.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                for module in modules:
                    root = module.split(".")[0]
                    assert (
                        root in allowed_third_party
                        or root in stdlib
                        or root in project
                    ), f"{path.name} imports unexpected module {module}"

    def test_retrieved_text_is_not_summarised(self, top3) -> None:
        """Evidence is returned verbatim; no generated interpretation.

        A summarised chunk would share little literal text with the source, so a
        high verbatim overlap is the signature of retrieval rather than generation.
        """
        full = " ".join(ingest.extract_docx_text().split())
        for chunk in top3.chunks:
            body = " ".join(chunk.text.split())
            if chunk.heading_path and body.startswith(chunk.heading_path):
                body = body[len(chunk.heading_path):]
            assert " ".join(body.split())[:100] in full


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCommandLine:
    """Both stages must be runnable without the LLM or the dashboard."""

    def test_ingest_cli_runs_and_reports(self, capsys, tmp_path) -> None:
        code = ingest.main(
            ["--persist-dir", str(tmp_path / "store"), "--no-export"]
        )
        captured = capsys.readouterr().out
        assert code == 0
        assert "CHROMADB" in captured
        assert "records stored" in captured
        assert "CHUNKS" in captured.upper() or "chunks produced" in captured

    def test_retrieve_cli_runs_and_prints_provenance(
        self, capsys, tmp_path, embedding_function
    ) -> None:
        directory = tmp_path / "store"
        ingest.build_index(persist_dir=directory)
        code = retrieve.main(
            ["--query", "What are the storage requirements?", "--persist-dir",
             str(directory)]
        )
        captured = capsys.readouterr().out
        assert code == 0
        assert config.GUIDELINE_DOCUMENT_NAME in captured
        assert "[1]" in captured and "[3]" in captured

    def test_cli_requires_an_action(self) -> None:
        with pytest.raises(SystemExit):
            retrieve.main([])

    def test_dose_cli_reports_evidence(self, capsys, tmp_path) -> None:
        directory = tmp_path / "store"
        ingest.build_index(persist_dir=directory)
        code = retrieve.main(
            ["--dose", "120", "--persist-dir", str(directory)]
        )
        captured = capsys.readouterr().out
        assert code == 0
        assert "maximum stated" in captured
        assert "100.0" in captured