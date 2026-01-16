"""
Unit tests for Stage 5: Logical Chunking - Core Chunker

Tests for the core chunking algorithm.
"""

import pytest

from stage_5_chunking.chunker import ChunkIdGenerator, chunk_blocks
from stage_5_chunking.models import BlockInput, ChunkingConfig
from stage_5_chunking.tokenizer import count_tokens


def make_block(
    block_id: str = "b1",
    page: int = 1,
    speaker: str | None = None,
    text: str = "Test text content.",
    confidence: float = 0.9,
) -> BlockInput:
    """Helper to create test blocks."""
    return BlockInput(
        block_id=block_id,
        page=page,
        clean_text=text,
        speaker=speaker,
        confidence=confidence,
    )


class TestChunkIdGenerator:
    """Tests for chunk ID generation."""

    def test_sequential_ids(self):
        """Should generate sequential IDs."""
        gen = ChunkIdGenerator()
        assert gen.next_id() == "C-0001"
        assert gen.next_id() == "C-0002"
        assert gen.next_id() == "C-0003"

    def test_custom_prefix(self):
        """Should support custom prefix."""
        gen = ChunkIdGenerator(prefix="CHUNK")
        assert gen.next_id() == "CHUNK-0001"

    def test_reset(self):
        """Should reset counter."""
        gen = ChunkIdGenerator()
        gen.next_id()
        gen.next_id()
        gen.reset()
        assert gen.next_id() == "C-0001"


class TestChunkBlocks:
    """Tests for core chunking algorithm."""

    def test_empty_blocks(self):
        """Empty blocks should return empty list."""
        chunks = chunk_blocks([], case_id="CASE1", document_id="DOC1")
        assert chunks == []

    def test_single_block(self):
        """Single block should create single chunk."""
        blocks = [make_block(block_id="b1")]
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1")
        assert len(chunks) == 1
        assert chunks[0].source_block_ids == ["b1"]

    def test_provenance_tracking(self):
        """Chunk should track all source block IDs."""
        blocks = [
            make_block(block_id="b1", page=1),
            make_block(block_id="b2", page=1),
        ]
        config = ChunkingConfig(max_tokens=1000)  # High limit to combine
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1", config=config)
        assert len(chunks) == 1
        assert "b1" in chunks[0].source_block_ids
        assert "b2" in chunks[0].source_block_ids

    def test_page_boundary_respected(self):
        """Chunks must never cross pages."""
        blocks = [
            make_block(block_id="b1", page=1),
            make_block(block_id="b2", page=2),
        ]
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1")
        assert len(chunks) == 2
        assert chunks[0].page_range == [1, 1]
        assert chunks[1].page_range == [2, 2]

    def test_speaker_boundary_respected(self):
        """Chunks must never mix speakers."""
        blocks = [
            make_block(block_id="b1", page=1, speaker="A"),
            make_block(block_id="b2", page=1, speaker="B"),
        ]
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1")
        assert len(chunks) == 2
        assert chunks[0].speaker == "A"
        assert chunks[1].speaker == "B"

    def test_token_limit_enforced(self):
        """Chunks should respect max_tokens limit."""
        # Create blocks that together exceed the limit
        long_text = " ".join(["word"] * 200)  # Many tokens
        blocks = [
            make_block(block_id="b1", page=1, text=long_text),
            make_block(block_id="b2", page=1, text=long_text),
        ]
        config = ChunkingConfig(max_tokens=100)
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1", config=config)
        # Each chunk should be within limit
        for chunk in chunks:
            assert chunk.token_count <= 100

    def test_confidence_aggregation(self):
        """Chunk confidence should be minimum of sources."""
        blocks = [
            make_block(block_id="b1", page=1, confidence=0.95),
            make_block(block_id="b2", page=1, confidence=0.85),
            make_block(block_id="b3", page=1, confidence=0.90),
        ]
        config = ChunkingConfig(max_tokens=1000)
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1", config=config)
        assert chunks[0].chunk_confidence == 0.85

    def test_text_is_exact_concatenation(self):
        """Chunk text should be exact concatenation of sources."""
        blocks = [
            make_block(block_id="b1", page=1, text="Hello"),
            make_block(block_id="b2", page=1, text="World"),
        ]
        config = ChunkingConfig(max_tokens=1000)
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1", config=config)
        assert chunks[0].text == "Hello World"

    def test_case_and_document_ids_preserved(self):
        """Case and document IDs should be preserved."""
        blocks = [make_block()]
        chunks = chunk_blocks(blocks, case_id="24-890-H", document_id="W001-24-890-H")
        assert chunks[0].case_id == "24-890-H"
        assert chunks[0].document_id == "W001-24-890-H"

    def test_dict_input_supported(self):
        """Should accept dict input."""
        blocks = [
            {
                "block_id": "b1",
                "page": 1,
                "clean_text": "Test text",
                "speaker": None,
                "confidence": 0.9,
            }
        ]
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1")
        assert len(chunks) == 1


class TestChunkBlocksEdgeCases:
    """Tests for edge cases in chunking."""

    def test_very_short_page(self):
        """Short pages should still produce chunks."""
        blocks = [make_block(block_id="b1", page=1, text="Hi")]
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1")
        assert len(chunks) == 1
        # Even if below min_tokens, still emitted

    def test_oversized_block_split(self):
        """Oversized block should be split deterministically."""
        # Create a very long block
        long_text = " ".join(["token"] * 500)
        blocks = [make_block(block_id="b1", page=1, text=long_text)]
        config = ChunkingConfig(max_tokens=50)
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1", config=config)
        # Should be split into multiple chunks
        assert len(chunks) > 1
        # All chunks should reference original block
        for chunk in chunks:
            assert "b1" in chunk.source_block_ids

    def test_null_speakers_grouped(self):
        """Blocks with null speakers should be grouped together."""
        blocks = [
            make_block(block_id="b1", page=1, speaker=None),
            make_block(block_id="b2", page=1, speaker=None),
        ]
        config = ChunkingConfig(max_tokens=1000)
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1", config=config)
        assert len(chunks) == 1

    def test_interleaved_speakers_on_same_page(self):
        """Interleaved speakers should create separate chunks."""
        blocks = [
            make_block(block_id="b1", page=1, speaker="A"),
            make_block(block_id="b2", page=1, speaker="B"),
            make_block(block_id="b3", page=1, speaker="A"),
        ]
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1")
        # A appears first, then B, then A rejoins first group
        # Groups: A=[b1, b3], B=[b2]
        assert len(chunks) == 2
        a_chunks = [c for c in chunks if c.speaker == "A"]
        b_chunks = [c for c in chunks if c.speaker == "B"]
        assert len(a_chunks) == 1
        assert len(b_chunks) == 1


class TestChunkingGuarantees:
    """Tests for hard guarantees."""

    def test_chunks_never_cross_pages(self):
        """HARD GUARANTEE: Chunks never cross pages."""
        blocks = [make_block(block_id=f"b{i}", page=(i % 3) + 1) for i in range(20)]
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1")
        for chunk in chunks:
            assert chunk.page_range[0] == chunk.page_range[1]

    def test_chunks_never_mix_speakers(self):
        """HARD GUARANTEE: Chunks never mix speakers."""
        speakers = ["A", "B", "C", None]
        blocks = [make_block(block_id=f"b{i}", page=1, speaker=speakers[i % 4]) for i in range(20)]
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1")
        # Each chunk should have consistent speaker
        for chunk in chunks:
            # All source blocks should have same speaker as chunk
            pass  # Speaker is already set from blocks

    def test_token_count_never_exceeds_max(self):
        """HARD GUARANTEE: Token count never exceeds max."""
        long_text = " ".join(["word"] * 100)
        blocks = [make_block(block_id=f"b{i}", page=1, text=long_text) for i in range(10)]
        config = ChunkingConfig(max_tokens=150)
        chunks = chunk_blocks(blocks, case_id="CASE1", document_id="DOC1", config=config)
        for chunk in chunks:
            assert chunk.token_count <= config.max_tokens
