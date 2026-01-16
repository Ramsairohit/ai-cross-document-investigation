"""
Unit tests for Stage 5: Logical Chunking - Determinism

CRITICAL: 100-run determinism verification.
Same input MUST always produce same output.
"""

import json

import pytest

from stage_5_chunking.chunker import chunk_blocks
from stage_5_chunking.models import BlockInput, ChunkingConfig


def create_test_document() -> list[BlockInput]:
    """Create a deterministic test document."""
    return [
        BlockInput(
            block_id="b1",
            page=1,
            clean_text="DET. SMITH: Where were you on the night of March 15th?",
            speaker="DET. SMITH",
            confidence=0.93,
        ),
        BlockInput(
            block_id="b2",
            page=1,
            clean_text="WITNESS: I was at home watching television.",
            speaker="WITNESS",
            confidence=0.91,
        ),
        BlockInput(
            block_id="b3",
            page=1,
            clean_text="DET. SMITH: Can anyone confirm that?",
            speaker="DET. SMITH",
            confidence=0.95,
        ),
        BlockInput(
            block_id="b4",
            page=2,
            clean_text="The witness appeared nervous during questioning.",
            speaker=None,
            confidence=0.88,
        ),
        BlockInput(
            block_id="b5",
            page=2,
            clean_text="WITNESS: My neighbor saw me arrive home at 7 PM.",
            speaker="WITNESS",
            confidence=0.90,
        ),
    ]


def chunks_to_dict(chunks: list) -> list[dict]:
    """Convert chunks to dicts for comparison."""
    return [c.model_dump() for c in chunks]


class TestDeterminism:
    """Determinism verification tests."""

    def test_100_runs_identical(self):
        """CRITICAL: 100 runs must produce identical output."""
        blocks = create_test_document()
        config = ChunkingConfig()

        # Get reference output
        reference_chunks = chunk_blocks(
            blocks=blocks,
            case_id="24-890-H",
            document_id="W001-24-890-H",
            config=config,
        )
        reference_json = json.dumps(chunks_to_dict(reference_chunks), sort_keys=True)

        # Run 100 times and compare
        for run in range(100):
            chunks = chunk_blocks(
                blocks=blocks,
                case_id="24-890-H",
                document_id="W001-24-890-H",
                config=config,
            )
            result_json = json.dumps(chunks_to_dict(chunks), sort_keys=True)
            assert result_json == reference_json, f"Non-deterministic output on run {run + 1}"

    def test_chunk_ids_deterministic(self):
        """Chunk IDs must be deterministic."""
        blocks = create_test_document()
        config = ChunkingConfig()

        ids_run1 = [
            c.chunk_id
            for c in chunk_blocks(blocks, case_id="CASE", document_id="DOC", config=config)
        ]
        ids_run2 = [
            c.chunk_id
            for c in chunk_blocks(blocks, case_id="CASE", document_id="DOC", config=config)
        ]
        assert ids_run1 == ids_run2

    def test_order_deterministic(self):
        """Chunk order must be deterministic."""
        blocks = create_test_document()
        config = ChunkingConfig()

        for _ in range(10):
            chunks = chunk_blocks(blocks, case_id="CASE", document_id="DOC", config=config)
            # First chunk should always be from p1, DET. SMITH
            assert chunks[0].speaker == "DET. SMITH"
            assert chunks[0].page_range == [1, 1]

    def test_token_counts_deterministic(self):
        """Token counts must be deterministic."""
        blocks = create_test_document()
        config = ChunkingConfig()

        counts_run1 = [
            c.token_count
            for c in chunk_blocks(blocks, case_id="CASE", document_id="DOC", config=config)
        ]
        counts_run2 = [
            c.token_count
            for c in chunk_blocks(blocks, case_id="CASE", document_id="DOC", config=config)
        ]
        assert counts_run1 == counts_run2

    def test_confidence_deterministic(self):
        """Confidence scores must be deterministic."""
        blocks = create_test_document()
        config = ChunkingConfig()

        conf_run1 = [
            c.chunk_confidence
            for c in chunk_blocks(blocks, case_id="CASE", document_id="DOC", config=config)
        ]
        conf_run2 = [
            c.chunk_confidence
            for c in chunk_blocks(blocks, case_id="CASE", document_id="DOC", config=config)
        ]
        assert conf_run1 == conf_run2

    def test_source_block_ids_deterministic(self):
        """Source block IDs must be deterministic."""
        blocks = create_test_document()
        config = ChunkingConfig()

        ids_run1 = [
            c.source_block_ids
            for c in chunk_blocks(blocks, case_id="CASE", document_id="DOC", config=config)
        ]
        ids_run2 = [
            c.source_block_ids
            for c in chunk_blocks(blocks, case_id="CASE", document_id="DOC", config=config)
        ]
        assert ids_run1 == ids_run2


class TestDeterminismWithVariedInputs:
    """Determinism with different input scenarios."""

    def test_empty_input_deterministic(self):
        """Empty input must be deterministic."""
        for _ in range(10):
            chunks = chunk_blocks([], case_id="CASE", document_id="DOC")
            assert chunks == []

    def test_single_block_deterministic(self):
        """Single block must be deterministic."""
        block = BlockInput(
            block_id="b1",
            page=1,
            clean_text="Single block content.",
            confidence=0.9,
        )

        reference = chunk_blocks([block], case_id="CASE", document_id="DOC")
        for _ in range(20):
            result = chunk_blocks([block], case_id="CASE", document_id="DOC")
            assert len(result) == len(reference)
            assert result[0].text == reference[0].text

    def test_many_blocks_deterministic(self):
        """Many blocks must be deterministic."""
        blocks = [
            BlockInput(
                block_id=f"b{i}",
                page=(i % 5) + 1,
                clean_text=f"Block number {i} with some content.",
                speaker=["A", "B", "C", None][i % 4],
                confidence=0.8 + (i % 20) * 0.01,
            )
            for i in range(50)
        ]

        reference = chunk_blocks(blocks, case_id="CASE", document_id="DOC")
        reference_json = json.dumps(chunks_to_dict(reference), sort_keys=True)

        for _ in range(20):
            result = chunk_blocks(blocks, case_id="CASE", document_id="DOC")
            result_json = json.dumps(chunks_to_dict(result), sort_keys=True)
            assert result_json == reference_json
