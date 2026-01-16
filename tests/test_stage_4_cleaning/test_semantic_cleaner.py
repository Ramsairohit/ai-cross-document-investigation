"""
Unit tests for Stage 4: Semantic Cleaning - Main Orchestrator

Tests for the SemanticCleaner and full pipeline.
"""

from datetime import datetime

import pytest

from stage_4_cleaning import (
    CleaningConfig,
    CleaningResult,
    SemanticCleaner,
    clean_document,
    clean_document_sync,
)


class TestSemanticCleaner:
    """Tests for SemanticCleaner class."""

    def test_init_default_config(self):
        """Should initialize with default config."""
        cleaner = SemanticCleaner()
        assert cleaner.config is not None
        assert cleaner.config.collapse_whitespace is True
        assert cleaner.config.normalize_newlines is True

    def test_init_custom_config(self):
        """Should accept custom config."""
        config = CleaningConfig(
            collapse_whitespace=False,
            reference_date=datetime(2024, 3, 15),
        )
        cleaner = SemanticCleaner(config=config)
        assert cleaner.config.collapse_whitespace is False
        assert cleaner.config.reference_date.year == 2024

    def test_clean_empty_document(self):
        """Should handle empty document."""
        cleaner = SemanticCleaner()
        result = cleaner.clean(
            {
                "document_id": "DOC1",
                "case_id": "CASE1",
                "source_file": "test.pdf",
                "parsed_blocks": [],
            }
        )

        assert result.document_id == "DOC1"
        assert result.case_id == "CASE1"
        assert result.cleaned_blocks == []

    def test_clean_single_block(self):
        """Should clean a single block correctly."""
        cleaner = SemanticCleaner()
        result = cleaner.clean(
            {
                "document_id": "DOC1",
                "case_id": "CASE1",
                "source_file": "test.pdf",
                "parsed_blocks": [
                    {
                        "block_id": "b1",
                        "page": 1,
                        "text": "  I   heard   a loud   crash at 8:15 PM  ",
                        "speaker": "WITNESS",
                        "section": "STATEMENT",
                        "is_header": False,
                        "is_footer": False,
                        "raw_timestamps": ["8:15 PM"],
                    }
                ],
            }
        )

        assert len(result.cleaned_blocks) == 1
        block = result.cleaned_blocks[0]

        # Check cleaned text
        assert block.clean_text == "I heard a loud crash at 8:15 PM"

        # Check metadata preserved
        assert block.block_id == "b1"
        assert block.page == 1
        assert block.speaker == "WITNESS"
        assert block.section == "STATEMENT"
        assert block.raw_timestamps == ["8:15 PM"]

        # Check timestamp normalized
        assert len(block.normalized_timestamps) == 1
        assert block.normalized_timestamps[0].original == "8:15 PM"

    def test_clean_multiple_blocks(self):
        """Should clean multiple blocks correctly."""
        cleaner = SemanticCleaner()
        result = cleaner.clean(
            {
                "document_id": "DOC1",
                "case_id": "CASE1",
                "source_file": "test.pdf",
                "parsed_blocks": [
                    {
                        "block_id": "b1",
                        "page": 1,
                        "text": "  First   block  ",
                        "speaker": None,
                        "section": "INTRO",
                        "is_header": False,
                        "is_footer": False,
                        "raw_timestamps": [],
                    },
                    {
                        "block_id": "b2",
                        "page": 1,
                        "text": "  Second   block  ",
                        "speaker": "DET",
                        "section": "INTERVIEW",
                        "is_header": False,
                        "is_footer": False,
                        "raw_timestamps": [],
                    },
                ],
            }
        )

        assert len(result.cleaned_blocks) == 2
        assert result.cleaned_blocks[0].clean_text == "First block"
        assert result.cleaned_blocks[1].clean_text == "Second block"

    def test_clean_preserves_header_footer_flags(self):
        """Should preserve header/footer flags."""
        cleaner = SemanticCleaner()
        result = cleaner.clean(
            {
                "document_id": "DOC1",
                "case_id": "CASE1",
                "source_file": "test.pdf",
                "parsed_blocks": [
                    {
                        "block_id": "h1",
                        "page": 1,
                        "text": "HEADER TEXT",
                        "speaker": None,
                        "section": None,
                        "is_header": True,
                        "is_footer": False,
                        "raw_timestamps": [],
                    },
                    {
                        "block_id": "f1",
                        "page": 1,
                        "text": "Page 1",
                        "speaker": None,
                        "section": None,
                        "is_header": False,
                        "is_footer": True,
                        "raw_timestamps": [],
                    },
                ],
            }
        )

        assert result.cleaned_blocks[0].is_header is True
        assert result.cleaned_blocks[0].is_footer is False
        assert result.cleaned_blocks[1].is_header is False
        assert result.cleaned_blocks[1].is_footer is True

    def test_clean_with_reference_date(self):
        """Should use reference date for timestamp parsing."""
        config = CleaningConfig(reference_date=datetime(2024, 3, 15))
        cleaner = SemanticCleaner(config=config)

        result = cleaner.clean(
            {
                "document_id": "DOC1",
                "case_id": "CASE1",
                "source_file": "test.pdf",
                "parsed_blocks": [
                    {
                        "block_id": "b1",
                        "page": 1,
                        "text": "Event at 8:15 PM",
                        "speaker": None,
                        "section": None,
                        "is_header": False,
                        "is_footer": False,
                        "raw_timestamps": ["8:15 PM"],
                    }
                ],
            }
        )

        ts = result.cleaned_blocks[0].normalized_timestamps[0]
        assert ts.iso is not None
        assert "2024-03-15" in ts.iso
        assert "20:15" in ts.iso

    def test_clean_text_only(self):
        """Should clean text without block structure."""
        cleaner = SemanticCleaner()
        result = cleaner.clean_text_only("  Hello   World  ")
        assert result == "Hello World"


class TestCleanDocumentSync:
    """Tests for clean_document_sync function."""

    def test_basic_cleaning(self):
        """Should perform basic cleaning synchronously."""
        result = clean_document_sync(
            {
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "source_file": "witness_statement.pdf",
                "parsed_blocks": [
                    {
                        "block_id": "b1",
                        "page": 1,
                        "text": "  I   heard   a loud   crash at 8:15 PM  ",
                        "speaker": "WITNESS",
                        "section": "STATEMENT",
                        "is_header": False,
                        "is_footer": False,
                        "raw_timestamps": ["8:15 PM"],
                    }
                ],
            }
        )

        assert isinstance(result, CleaningResult)
        assert result.document_id == "DOC123"
        assert len(result.cleaned_blocks) == 1


class TestCleanDocumentAsync:
    """Tests for clean_document async function."""

    @pytest.mark.asyncio
    async def test_basic_cleaning_async(self):
        """Should perform basic cleaning asynchronously."""
        result = await clean_document(
            {
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "source_file": "witness_statement.pdf",
                "parsed_blocks": [
                    {
                        "block_id": "b1",
                        "page": 1,
                        "text": "  I   heard   a loud   crash at 8:15 PM  ",
                        "speaker": "WITNESS",
                        "section": "STATEMENT",
                        "is_header": False,
                        "is_footer": False,
                        "raw_timestamps": ["8:15 PM"],
                    }
                ],
            }
        )

        assert isinstance(result, CleaningResult)
        assert result.document_id == "DOC123"


class TestDeterminism:
    """Tests to verify deterministic behavior."""

    def test_same_input_same_output(self):
        """Same input should always produce exactly same output."""
        input_doc = {
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "source_file": "witness_statement.pdf",
            "parsed_blocks": [
                {
                    "block_id": "b1",
                    "page": 1,
                    "text": "  I   heard   a loud | crash at 8:15 PM  ",
                    "speaker": "WITNESS",
                    "section": "STATEMENT",
                    "is_header": False,
                    "is_footer": False,
                    "raw_timestamps": ["8:15 PM"],
                }
            ],
        }

        # Run 100 times
        results = [clean_document_sync(input_doc) for _ in range(100)]

        # All should be identical
        first_result = results[0].model_dump_json()
        for result in results[1:]:
            assert result.model_dump_json() == first_result


class TestMeaningPreservation:
    """Tests to ensure meaning is preserved (forensic-grade requirement)."""

    def test_no_word_removal(self):
        """Should never remove actual words."""
        cleaner = SemanticCleaner()
        text = "The suspect allegedly stole approximately $500 from the victim"
        result = cleaner.clean_text_only(text)

        # All words must be present
        for word in ["suspect", "allegedly", "stole", "approximately", "$500", "victim"]:
            assert word in result

    def test_no_punctuation_change(self):
        """Should preserve punctuation."""
        cleaner = SemanticCleaner()
        text = "Hello, World! How are you? I'm fine, thanks."
        result = cleaner.clean_text_only(text)

        # All punctuation preserved
        assert "," in result
        assert "!" in result
        assert "?" in result
        assert "'" in result

    def test_no_interpretation(self):
        """Should not interpret or correct text."""
        cleaner = SemanticCleaner()

        # Misspelled word should NOT be corrected
        text = "The wintess said she saw the suspekt"
        result = cleaner.clean_text_only(text)
        assert "wintess" in result  # NOT corrected to "witness"
        assert "suspekt" in result  # NOT corrected to "suspect"

    def test_ambiguity_preserved(self):
        """Should preserve ambiguous statements."""
        cleaner = SemanticCleaner()

        # Ambiguous statement should not be clarified
        text = "He said he saw her there"
        result = cleaner.clean_text_only(text)
        assert result == text  # Exactly preserved

    def test_quote_preservation(self):
        """Should preserve quoted text exactly."""
        cleaner = SemanticCleaner()
        text = 'The witness stated: "I saw him at 8 PM"'
        result = cleaner.clean_text_only(text)

        assert '"I saw him at 8 PM"' in result


class TestOutputSchema:
    """Tests to verify mandatory output schema compliance."""

    def test_cleaning_result_schema(self):
        """Should produce correct CleaningResult schema."""
        result = clean_document_sync(
            {
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "source_file": "witness_statement.pdf",
                "parsed_blocks": [
                    {
                        "block_id": "b1",
                        "page": 1,
                        "text": "Test text",
                        "speaker": "WITNESS",
                        "section": "STATEMENT",
                        "is_header": False,
                        "is_footer": False,
                        "raw_timestamps": ["8:15 PM"],
                    }
                ],
            }
        )

        # Check required fields exist
        assert hasattr(result, "document_id")
        assert hasattr(result, "case_id")
        assert hasattr(result, "source_file")
        assert hasattr(result, "cleaned_blocks")

        # Check block schema
        block = result.cleaned_blocks[0]
        assert hasattr(block, "block_id")
        assert hasattr(block, "page")
        assert hasattr(block, "clean_text")
        assert hasattr(block, "speaker")
        assert hasattr(block, "section")
        assert hasattr(block, "raw_timestamps")
        assert hasattr(block, "normalized_timestamps")

        # Check timestamp schema
        ts = block.normalized_timestamps[0]
        assert hasattr(ts, "original")
        assert hasattr(ts, "iso")
        assert hasattr(ts, "confidence")

    def test_json_serializable(self):
        """Result should be JSON serializable."""
        result = clean_document_sync(
            {
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "source_file": "test.pdf",
                "parsed_blocks": [
                    {
                        "block_id": "b1",
                        "page": 1,
                        "text": "Test",
                        "speaker": None,
                        "section": None,
                        "is_header": False,
                        "is_footer": False,
                        "raw_timestamps": [],
                    }
                ],
            }
        )

        # Should not raise
        json_str = result.model_dump_json()
        assert isinstance(json_str, str)
        assert "DOC123" in json_str
