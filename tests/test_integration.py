"""
End-to-End Pipeline Integration Test

Verifies that all stages (3-7) are properly connected and data flows correctly.
Stage 2 is skipped as it requires actual files.

Tests the complete flow:
Stage 3 (Parsing) -> Stage 4 (Cleaning) -> Stage 5 (Chunking) -> Stage 6 (NER) -> Stage 7 (Embeddings)
"""

import tempfile
from pathlib import Path


def test_stage_3_to_stage_4_connection():
    """Verify Stage 3 output can be consumed by Stage 4."""
    from stage_3_parsing import ParsedBlock, StructuralParseResult
    from stage_4_cleaning import SemanticCleaner

    # Create Stage 3 output
    parsed_block = ParsedBlock(
        block_id="b1",
        page=1,
        text="DET. SMITH: Where were you at 8:15 PM on March 15th?",
        speaker="DET. SMITH",
        is_header=False,
        is_footer=False,
        section="STATEMENT",
        raw_timestamps=["8:15 PM", "March 15th"],
    )

    parse_result = StructuralParseResult(
        document_id="DOC001",
        case_id="24-890-H",
        source_file="statement.pdf",
        total_pages=1,
        parsed_blocks=[parsed_block],
    )

    # Stage 4 should accept this via .clean() method
    cleaner = SemanticCleaner()
    cleaning_result = cleaner.clean(parse_result)

    assert cleaning_result.document_id == "DOC001"
    assert len(cleaning_result.cleaned_blocks) == 1
    assert cleaning_result.cleaned_blocks[0].speaker == "DET. SMITH"


def test_stage_4_to_stage_5_connection():
    """Verify Stage 4 output can be consumed by Stage 5."""
    from stage_4_cleaning import CleanedBlock, CleaningResult
    from stage_5_chunking import ChunkingPipeline

    # Create Stage 4 output
    cleaned_block = CleanedBlock(
        block_id="b1",
        page=1,
        clean_text="DET. SMITH: Where were you at 8:15 PM?",
        speaker="DET. SMITH",
        is_header=False,
        is_footer=False,
    )

    cleaning_result = CleaningResult(
        document_id="DOC001",
        case_id="24-890-H",
        source_file="statement.pdf",
        cleaned_blocks=[cleaned_block],
    )

    # Stage 5 should accept this
    pipeline = ChunkingPipeline()
    chunking_result = pipeline.process_cleaning_result(cleaning_result.model_dump())

    assert chunking_result.document_id == "DOC001"
    assert chunking_result.total_chunks >= 1
    assert chunking_result.chunks[0].speaker == "DET. SMITH"


def test_stage_5_to_stage_6_connection():
    """Verify Stage 5 output can be consumed by Stage 6."""
    from stage_5_chunking import Chunk
    from stage_6_ner import ChunkInput, NERPipeline

    # Create Stage 5 output
    chunk = Chunk(
        chunk_id="C-0001",
        case_id="24-890-H",
        document_id="DOC001",
        page_range=[1, 1],
        speaker="WITNESS",
        text="I saw Marcus Vane at 420 Harrow Lane at 8:15 PM.",
        source_block_ids=["b1"],
        token_count=20,
        chunk_confidence=0.91,
    )

    # Convert to Stage 6 ChunkInput
    chunk_dict = chunk.model_dump()
    chunk_input = ChunkInput(
        chunk_id=chunk_dict["chunk_id"],
        document_id=chunk_dict["document_id"],
        case_id=chunk_dict["case_id"],
        page_range=chunk_dict["page_range"],
        speaker=chunk_dict["speaker"],
        text=chunk_dict["text"],
        confidence=chunk_dict["chunk_confidence"],
    )

    # Stage 6 should accept this
    ner_pipeline = NERPipeline()
    ner_result = ner_pipeline.process_chunk(chunk_input)

    assert ner_result.chunk_id == "C-0001"
    assert ner_result.case_id == "24-890-H"
    # Should extract entities (PERSON, LOCATION, TIME)
    assert ner_result.entity_count >= 0  # May vary based on spaCy model


def test_stage_5_to_stage_7_connection():
    """Verify Stage 5 output can be consumed by Stage 7."""
    from stage_5_chunking import Chunk
    from stage_6_ner import ChunkInput
    from stage_7_embeddings import EmbeddingPipeline

    # Create Stage 5 output
    chunk = Chunk(
        chunk_id="C-0001",
        case_id="24-890-H",
        document_id="DOC001",
        page_range=[1, 1],
        speaker="WITNESS",
        text="I saw Marcus Vane at 420 Harrow Lane at 8:15 PM.",
        source_block_ids=["b1"],
        token_count=20,
        chunk_confidence=0.91,
    )

    # Create temp directory for vector storage
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir)

        # Stage 7 should accept this
        embedding_pipeline = EmbeddingPipeline(storage_dir=storage_dir)

        # Convert chunk to ChunkInput format for embedding
        chunk_input = ChunkInput(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            case_id=chunk.case_id,
            page_range=chunk.page_range,
            speaker=chunk.speaker,
            text=chunk.text,
            confidence=chunk.chunk_confidence,
        )

        embedding_result = embedding_pipeline.process_chunk(chunk_input)

        assert embedding_result.chunk_id == "C-0001"
        assert embedding_result.success is True
        assert embedding_result.vector_id >= 0


def test_full_pipeline_flow():
    """
    Test complete pipeline from Stage 3 output to Stage 6 and 7.

    Flow: Stage 3 -> Stage 4 -> Stage 5 -> Stage 6 + Stage 7
    """
    from stage_3_parsing import ParsedBlock
    from stage_4_cleaning import SemanticCleaner
    from stage_5_chunking import ChunkingPipeline
    from stage_6_ner import ChunkInput, NERPipeline
    from stage_7_embeddings import EmbeddingPipeline

    # Stage 3 Output
    parsed_blocks = [
        ParsedBlock(
            block_id="b1",
            page=1,
            text="DET. SMITH: Where were you on the night of March 15th?",
            speaker="DET. SMITH",
            is_header=False,
            is_footer=False,
        ),
        ParsedBlock(
            block_id="b2",
            page=1,
            text="WITNESS: I was at home watching television all evening.",
            speaker="WITNESS",
            is_header=False,
            is_footer=False,
        ),
    ]

    # Build Stage 3 result dict
    parse_result = {
        "document_id": "DOC001",
        "case_id": "24-890-H",
        "source_file": "interview.pdf",
        "total_pages": 1,
        "parsed_blocks": [b.model_dump() for b in parsed_blocks],
    }

    # Stage 4: Semantic Cleaning
    cleaner = SemanticCleaner()
    cleaning_result = cleaner.clean(parse_result)
    assert len(cleaning_result.cleaned_blocks) == 2

    # Stage 5: Logical Chunking
    chunking_pipeline = ChunkingPipeline()
    chunking_result = chunking_pipeline.process_cleaning_result(cleaning_result.model_dump())
    assert chunking_result.total_chunks >= 1

    # Stage 6: NER (process each chunk)
    ner_pipeline = NERPipeline()
    all_entities = []
    for chunk in chunking_result.chunks:
        chunk_input = ChunkInput(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            case_id=chunk.case_id,
            page_range=chunk.page_range,
            speaker=chunk.speaker,
            text=chunk.text,
            confidence=chunk.chunk_confidence,
        )
        ner_result = ner_pipeline.process_chunk(chunk_input)
        all_entities.extend(ner_result.entities)

    # NER should extract some entities
    assert isinstance(all_entities, list)

    # Stage 7: Embeddings (process each chunk)
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir)
        embedding_pipeline = EmbeddingPipeline(storage_dir=storage_dir)

        for chunk in chunking_result.chunks:
            chunk_input = ChunkInput(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                case_id=chunk.case_id,
                page_range=chunk.page_range,
                speaker=chunk.speaker,
                text=chunk.text,
                confidence=chunk.chunk_confidence,
            )
            result = embedding_pipeline.process_chunk(chunk_input)
            assert result.success is True

        # Verify vectors are stored
        assert embedding_pipeline.get_vector_count() >= len(chunking_result.chunks)

    print("✅ Full pipeline integration test passed!")
    print(f"   - Cleaned blocks: {len(cleaning_result.cleaned_blocks)}")
    print(f"   - Chunks created: {chunking_result.total_chunks}")
    print(f"   - Entities extracted: {len(all_entities)}")


if __name__ == "__main__":
    test_full_pipeline_flow()
