"""Tests for Stage 3: Structural Parsing - Header/Footer Detection."""

import pytest

from stage_3_parsing.header_footer import (
    HeaderFooterResult,
    detect_headers_footers,
)


class TestPageNumberDetection:
    """Tests for page number pattern detection."""

    def test_page_x_of_y_format(self) -> None:
        """Test 'Page 1 of 5' format detected as footer."""
        blocks = [
            {"block_id": "b1", "text": "Page 1 of 5", "page": 1},
        ]
        results = detect_headers_footers(blocks)
        assert results["b1"].is_footer is True
        assert results["b1"].is_header is False

    def test_dash_number_dash_format(self) -> None:
        """Test '— 3 —' format detected as footer."""
        blocks = [
            {"block_id": "b1", "text": "— 3 —", "page": 3},
        ]
        results = detect_headers_footers(blocks)
        assert results["b1"].is_footer is True

    def test_simple_number_format(self) -> None:
        """Test simple page number detected as footer."""
        blocks = [
            {"block_id": "b1", "text": "  5  ", "page": 5},
        ]
        results = detect_headers_footers(blocks)
        assert results["b1"].is_footer is True


class TestRepeatedTextDetection:
    """Tests for repeated text across pages."""

    def test_repeated_header(self) -> None:
        """Test text repeated on first block of multiple pages as header."""
        blocks = [
            {"block_id": "b1", "text": "POLICE DEPARTMENT REPORT", "page": 1},
            {"block_id": "b2", "text": "Content on page 1", "page": 1},
            {"block_id": "b3", "text": "POLICE DEPARTMENT REPORT", "page": 2},
            {"block_id": "b4", "text": "Content on page 2", "page": 2},
        ]
        results = detect_headers_footers(blocks, min_page_repetition=2)

        # First occurrence on page is header
        assert results["b1"].is_header is True
        assert results["b3"].is_header is True

        # Content blocks are neither
        assert results["b2"].is_header is False
        assert results["b2"].is_footer is False

    def test_repeated_footer(self) -> None:
        """Test text repeated on last block of multiple pages as footer."""
        blocks = [
            {"block_id": "b1", "text": "Content on page 1", "page": 1},
            {"block_id": "b2", "text": "CONFIDENTIAL", "page": 1},
            {"block_id": "b3", "text": "Content on page 2", "page": 2},
            {"block_id": "b4", "text": "CONFIDENTIAL", "page": 2},
        ]
        results = detect_headers_footers(blocks, min_page_repetition=2)

        # Last occurrence on page is footer
        assert results["b2"].is_footer is True
        assert results["b4"].is_footer is True

    def test_single_occurrence_not_flagged(self) -> None:
        """Test text appearing only once is not flagged."""
        blocks = [
            {"block_id": "b1", "text": "Unique Title", "page": 1},
            {"block_id": "b2", "text": "Some content", "page": 1},
        ]
        results = detect_headers_footers(blocks, min_page_repetition=2)

        assert results["b1"].is_header is False
        assert results["b1"].is_footer is False


class TestEmptyInput:
    """Tests for edge cases with empty input."""

    def test_empty_blocks_list(self) -> None:
        """Test empty blocks list returns empty dict."""
        results = detect_headers_footers([])
        assert results == {}

    def test_empty_text_block(self) -> None:
        """Test block with empty text."""
        blocks = [
            {"block_id": "b1", "text": "", "page": 1},
        ]
        results = detect_headers_footers(blocks)
        assert results["b1"].is_header is False
        assert results["b1"].is_footer is False


class TestLongTextNotFlagged:
    """Tests that long text is not flagged as header/footer."""

    def test_long_repeated_text_not_flagged(self) -> None:
        """Long blocks should not be considered header/footer."""
        long_text = "This is a very long paragraph " * 10
        blocks = [
            {"block_id": "b1", "text": long_text, "page": 1},
            {"block_id": "b2", "text": long_text, "page": 2},
        ]
        results = detect_headers_footers(blocks)

        # Long text shouldn't be flagged even if repeated
        assert results["b1"].is_header is False
        assert results["b1"].is_footer is False


class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_same_input_same_output(self) -> None:
        """Verify same input produces identical output."""
        blocks = [
            {"block_id": "b1", "text": "HEADER TEXT", "page": 1},
            {"block_id": "b2", "text": "Content", "page": 1},
            {"block_id": "b3", "text": "Page 1", "page": 1},
            {"block_id": "b4", "text": "HEADER TEXT", "page": 2},
        ]

        result1 = detect_headers_footers(blocks)
        result2 = detect_headers_footers(blocks)

        for block_id in result1:
            assert result1[block_id].is_header == result2[block_id].is_header
            assert result1[block_id].is_footer == result2[block_id].is_footer
