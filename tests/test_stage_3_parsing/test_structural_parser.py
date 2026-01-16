"""Tests for Stage 3: Structural Parsing - Main Orchestrator."""

import pytest

from stage_3_parsing import (
    ParsedBlock,
    ParsingConfig,
    StructuralParseResult,
    StructuralParser,
    parse_document_sync,
)


class TestStructuralParser:
    """Tests for the main StructuralParser class."""

    def test_parse_simple_document(self) -> None:
        """Test parsing a simple document."""
        input_doc = {
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "source_file": "witness_statement.pdf",
            "content_blocks": [
                {
                    "block_id": "b1",
                    "type": "paragraph",
                    "text": "DET. SMITH: Where were you on the night of March 15?",
                    "page": 1,
                    "confidence": 0.94,
                },
            ],
        }

        parser = StructuralParser()
        result = parser.parse(input_doc)

        assert result.document_id == "DOC123"
        assert result.case_id == "24-890-H"
        assert result.source_file == "witness_statement.pdf"
        assert len(result.parsed_blocks) == 1

        block = result.parsed_blocks[0]
        assert block.block_id == "b1"
        assert block.speaker == "DET. SMITH"
        assert block.text == "Where were you on the night of March 15?"
        assert "March 15" in block.raw_timestamps

    def test_parse_with_sections(self) -> None:
        """Test parsing document with sections."""
        input_doc = {
            "document_id": "DOC456",
            "case_id": "24-890-H",
            "source_file": "report.pdf",
            "content_blocks": [
                {
                    "block_id": "b1",
                    "type": "heading",
                    "text": "INTERVIEW",
                    "page": 1,
                    "confidence": 1.0,
                },
                {
                    "block_id": "b2",
                    "type": "paragraph",
                    "text": "Q: State your name.",
                    "page": 1,
                    "confidence": 0.95,
                },
                {
                    "block_id": "b3",
                    "type": "paragraph",
                    "text": "A: John Doe.",
                    "page": 1,
                    "confidence": 0.95,
                },
            ],
        }

        parser = StructuralParser()
        result = parser.parse(input_doc)

        assert result.parsed_blocks[0].section == "INTERVIEW"
        assert result.parsed_blocks[1].section == "INTERVIEW"
        assert result.parsed_blocks[2].section == "INTERVIEW"

        assert result.parsed_blocks[1].speaker == "Q"
        assert result.parsed_blocks[2].speaker == "A"

    def test_parse_empty_document(self) -> None:
        """Test parsing document with no content blocks."""
        input_doc = {
            "document_id": "DOC789",
            "case_id": "24-890-H",
            "source_file": "empty.pdf",
            "content_blocks": [],
        }

        parser = StructuralParser()
        result = parser.parse(input_doc)

        assert result.document_id == "DOC789"
        assert result.parsed_blocks == []


class TestParseDocumentSync:
    """Tests for the synchronous parse function."""

    def test_parse_document_sync(self) -> None:
        """Test synchronous parsing function."""
        input_doc = {
            "document_id": "DOC001",
            "case_id": "TEST-001",
            "source_file": "test.pdf",
            "content_blocks": [
                {
                    "block_id": "b1",
                    "type": "paragraph",
                    "text": "Test content at 8:15 PM.",
                    "page": 1,
                    "confidence": 0.9,
                },
            ],
        }

        result = parse_document_sync(input_doc)

        assert isinstance(result, StructuralParseResult)
        assert len(result.parsed_blocks) == 1
        assert "8:15 PM" in result.parsed_blocks[0].raw_timestamps


class TestParsingConfig:
    """Tests for custom parsing configuration."""

    def test_custom_config(self) -> None:
        """Test parser with custom configuration."""
        config = ParsingConfig(
            min_page_repetition=3,
            max_section_header_length=30,
        )

        parser = StructuralParser(config=config)
        assert parser.config.min_page_repetition == 3
        assert parser.config.max_section_header_length == 30


class TestHeaderFooterIntegration:
    """Tests for header/footer detection in full parsing."""

    def test_page_numbers_marked_as_footer(self) -> None:
        """Test page numbers are marked as footers."""
        input_doc = {
            "document_id": "DOC001",
            "case_id": "TEST-001",
            "source_file": "test.pdf",
            "content_blocks": [
                {
                    "block_id": "b1",
                    "type": "paragraph",
                    "text": "Content on page 1.",
                    "page": 1,
                    "confidence": 0.9,
                },
                {
                    "block_id": "b2",
                    "type": "paragraph",
                    "text": "Page 1 of 3",
                    "page": 1,
                    "confidence": 1.0,
                },
            ],
        }

        result = parse_document_sync(input_doc)

        assert result.parsed_blocks[0].is_footer is False
        assert result.parsed_blocks[1].is_footer is True


class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_same_input_same_output(self) -> None:
        """Verify same input produces identical output."""
        input_doc = {
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "source_file": "test.pdf",
            "content_blocks": [
                {
                    "block_id": "b1",
                    "type": "paragraph",
                    "text": "DET. SMITH: Interview at 8:15 PM.",
                    "page": 1,
                    "confidence": 0.9,
                },
                {
                    "block_id": "b2",
                    "type": "paragraph",
                    "text": "STATEMENT",
                    "page": 2,
                    "confidence": 1.0,
                },
                {
                    "block_id": "b3",
                    "type": "paragraph",
                    "text": "Content in statement.",
                    "page": 2,
                    "confidence": 0.9,
                },
            ],
        }

        parser = StructuralParser()

        result1 = parser.parse(input_doc)
        result2 = parser.parse(input_doc)

        # Compare all fields
        assert result1.document_id == result2.document_id
        assert result1.case_id == result2.case_id
        assert len(result1.parsed_blocks) == len(result2.parsed_blocks)

        for b1, b2 in zip(result1.parsed_blocks, result2.parsed_blocks):
            assert b1.block_id == b2.block_id
            assert b1.text == b2.text
            assert b1.speaker == b2.speaker
            assert b1.section == b2.section
            assert b1.is_header == b2.is_header
            assert b1.is_footer == b2.is_footer
            assert b1.raw_timestamps == b2.raw_timestamps

    def test_multiple_runs_identical(self) -> None:
        """Run parsing 100 times and verify identical results."""
        input_doc = {
            "document_id": "DOC999",
            "case_id": "TEST",
            "source_file": "test.pdf",
            "content_blocks": [
                {
                    "block_id": "b1",
                    "type": "paragraph",
                    "text": "WITNESS: I saw him at 10:30 PM on March 15.",
                    "page": 1,
                    "confidence": 0.95,
                },
            ],
        }

        parser = StructuralParser()
        baseline = parser.parse(input_doc)

        for i in range(100):
            result = parser.parse(input_doc)
            assert result.parsed_blocks[0].speaker == baseline.parsed_blocks[0].speaker
            assert result.parsed_blocks[0].text == baseline.parsed_blocks[0].text
            assert (
                result.parsed_blocks[0].raw_timestamps == baseline.parsed_blocks[0].raw_timestamps
            )
