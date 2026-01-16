"""
Unit tests for Stage 5: Logical Chunking - Boundary Rules

Tests for page and speaker boundary detection.
"""

import pytest

from stage_5_chunking.chunk_rules import (
    blocks_share_boundary,
    check_page_boundary,
    check_speaker_boundary,
    convert_to_block_input,
    get_boundary_key,
    group_blocks_by_boundary,
    validate_chunk_blocks,
)
from stage_5_chunking.models import BlockInput


def make_block(
    block_id: str = "b1",
    page: int = 1,
    speaker: str | None = None,
    text: str = "Test text",
) -> BlockInput:
    """Helper to create test blocks."""
    return BlockInput(
        block_id=block_id,
        page=page,
        clean_text=text,
        speaker=speaker,
    )


class TestCheckPageBoundary:
    """Tests for page boundary checking."""

    def test_empty_blocks(self):
        """Empty blocks should pass."""
        assert check_page_boundary([]) is True

    def test_single_block(self):
        """Single block should pass."""
        blocks = [make_block(page=1)]
        assert check_page_boundary(blocks) is True

    def test_same_page(self):
        """Blocks on same page should pass."""
        blocks = [
            make_block(block_id="b1", page=1),
            make_block(block_id="b2", page=1),
            make_block(block_id="b3", page=1),
        ]
        assert check_page_boundary(blocks) is True

    def test_different_pages(self):
        """Blocks on different pages should fail."""
        blocks = [
            make_block(block_id="b1", page=1),
            make_block(block_id="b2", page=2),
        ]
        assert check_page_boundary(blocks) is False


class TestCheckSpeakerBoundary:
    """Tests for speaker boundary checking."""

    def test_empty_blocks(self):
        """Empty blocks should pass."""
        assert check_speaker_boundary([]) is True

    def test_single_block(self):
        """Single block should pass."""
        blocks = [make_block(speaker="WITNESS")]
        assert check_speaker_boundary(blocks) is True

    def test_same_speaker(self):
        """Blocks with same speaker should pass."""
        blocks = [
            make_block(block_id="b1", speaker="DET. SMITH"),
            make_block(block_id="b2", speaker="DET. SMITH"),
        ]
        assert check_speaker_boundary(blocks) is True

    def test_different_speakers(self):
        """Blocks with different speakers should fail."""
        blocks = [
            make_block(block_id="b1", speaker="DET. SMITH"),
            make_block(block_id="b2", speaker="WITNESS"),
        ]
        assert check_speaker_boundary(blocks) is False

    def test_null_speakers(self):
        """All null speakers should pass."""
        blocks = [
            make_block(block_id="b1", speaker=None),
            make_block(block_id="b2", speaker=None),
        ]
        assert check_speaker_boundary(blocks) is True


class TestValidateChunkBlocks:
    """Tests for chunk block validation."""

    def test_empty_blocks(self):
        """Empty blocks should fail."""
        valid, error = validate_chunk_blocks([])
        assert valid is False
        assert "empty" in error.lower()

    def test_valid_blocks(self):
        """Valid blocks should pass."""
        blocks = [
            make_block(block_id="b1", page=1, speaker="DET"),
            make_block(block_id="b2", page=1, speaker="DET"),
        ]
        valid, error = validate_chunk_blocks(blocks)
        assert valid is True
        assert error == ""

    def test_page_violation(self):
        """Page crossing should fail."""
        blocks = [
            make_block(block_id="b1", page=1),
            make_block(block_id="b2", page=2),
        ]
        valid, error = validate_chunk_blocks(blocks)
        assert valid is False
        assert "page" in error.lower()

    def test_speaker_violation(self):
        """Speaker mixing should fail."""
        blocks = [
            make_block(block_id="b1", page=1, speaker="A"),
            make_block(block_id="b2", page=1, speaker="B"),
        ]
        valid, error = validate_chunk_blocks(blocks)
        assert valid is False
        assert "speaker" in error.lower()


class TestGroupBlocksByBoundary:
    """Tests for block grouping."""

    def test_empty_blocks(self):
        """Empty blocks should return empty list."""
        assert group_blocks_by_boundary([]) == []

    def test_single_group(self):
        """All blocks with same boundary should be one group."""
        blocks = [
            make_block(block_id="b1", page=1, speaker="A"),
            make_block(block_id="b2", page=1, speaker="A"),
        ]
        groups = group_blocks_by_boundary(blocks)
        assert len(groups) == 1
        assert groups[0][0] == (1, "A")
        assert len(groups[0][1]) == 2

    def test_multiple_groups_by_page(self):
        """Different pages should create different groups."""
        blocks = [
            make_block(block_id="b1", page=1),
            make_block(block_id="b2", page=2),
        ]
        groups = group_blocks_by_boundary(blocks)
        assert len(groups) == 2

    def test_multiple_groups_by_speaker(self):
        """Different speakers on same page should create different groups."""
        blocks = [
            make_block(block_id="b1", page=1, speaker="A"),
            make_block(block_id="b2", page=1, speaker="B"),
        ]
        groups = group_blocks_by_boundary(blocks)
        assert len(groups) == 2

    def test_order_preserved(self):
        """Groups should be in order of first appearance."""
        blocks = [
            make_block(block_id="b1", page=1, speaker="A"),
            make_block(block_id="b2", page=1, speaker="B"),
            make_block(block_id="b3", page=1, speaker="A"),  # Back to A
        ]
        groups = group_blocks_by_boundary(blocks)
        # A appears first, B second. b3 joins A's group.
        assert len(groups) == 2
        assert groups[0][0] == (1, "A")
        assert len(groups[0][1]) == 2  # b1 and b3
        assert groups[1][0] == (1, "B")


class TestGetBoundaryKey:
    """Tests for boundary key extraction."""

    def test_with_speaker(self):
        """Should return (page, speaker) tuple."""
        block = make_block(page=3, speaker="WITNESS")
        key = get_boundary_key(block)
        assert key == (3, "WITNESS")

    def test_without_speaker(self):
        """Should handle null speaker."""
        block = make_block(page=1, speaker=None)
        key = get_boundary_key(block)
        assert key == (1, None)


class TestBlocksShareBoundary:
    """Tests for boundary comparison."""

    def test_same_boundary(self):
        """Blocks with same page and speaker share boundary."""
        b1 = make_block(page=1, speaker="A")
        b2 = make_block(page=1, speaker="A")
        assert blocks_share_boundary(b1, b2) is True

    def test_different_page(self):
        """Different page means different boundary."""
        b1 = make_block(page=1, speaker="A")
        b2 = make_block(page=2, speaker="A")
        assert blocks_share_boundary(b1, b2) is False

    def test_different_speaker(self):
        """Different speaker means different boundary."""
        b1 = make_block(page=1, speaker="A")
        b2 = make_block(page=1, speaker="B")
        assert blocks_share_boundary(b1, b2) is False


class TestConvertToBlockInput:
    """Tests for block conversion."""

    def test_from_dict(self):
        """Should convert dict to BlockInput."""
        data = {
            "block_id": "b1",
            "page": 1,
            "clean_text": "Test",
            "speaker": "WITNESS",
            "confidence": 0.9,
        }
        block = convert_to_block_input(data)
        assert isinstance(block, BlockInput)
        assert block.block_id == "b1"
        assert block.speaker == "WITNESS"

    def test_from_block_input(self):
        """Should pass through BlockInput unchanged."""
        original = make_block()
        result = convert_to_block_input(original)
        assert result is original
