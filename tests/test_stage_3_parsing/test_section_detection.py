"""Tests for Stage 3: Structural Parsing - Section Detection."""

import pytest

from stage_3_parsing.section_detection import (
    SectionTracker,
    assign_sections,
    detect_section,
)


class TestSectionKeywords:
    """Tests for known section keyword detection."""

    def test_statement_section(self) -> None:
        """Test STATEMENT detected as section header."""
        result = detect_section("STATEMENT")
        assert result.is_section_header is True
        assert result.section_name == "STATEMENT"

    def test_interview_section(self) -> None:
        """Test INTERVIEW detected as section header."""
        result = detect_section("INTERVIEW")
        assert result.is_section_header is True
        assert result.section_name == "INTERVIEW"

    def test_observations_section(self) -> None:
        """Test OBSERVATIONS detected as section header."""
        result = detect_section("OBSERVATIONS")
        assert result.is_section_header is True
        assert result.section_name == "OBSERVATIONS"

    def test_summary_section(self) -> None:
        """Test SUMMARY detected as section header."""
        result = detect_section("SUMMARY")
        assert result.is_section_header is True
        assert result.section_name == "SUMMARY"

    def test_witness_statement_section(self) -> None:
        """Test WITNESS STATEMENT detected as section header."""
        result = detect_section("WITNESS STATEMENT")
        assert result.is_section_header is True
        assert result.section_name == "WITNESS STATEMENT"


class TestSectionPatterns:
    """Tests for section header patterns."""

    def test_numbered_section(self) -> None:
        """Test '1. INTRODUCTION' pattern."""
        result = detect_section("1. INTRODUCTION")
        assert result.is_section_header is True
        assert "INTRODUCTION" in result.section_name

    def test_roman_numeral_section(self) -> None:
        """Test 'I. BACKGROUND' pattern."""
        result = detect_section("I. BACKGROUND")
        assert result.is_section_header is True

    def test_all_caps_short_line(self) -> None:
        """Test short all-caps line detected as section."""
        result = detect_section("EVIDENCE SUMMARY")
        assert result.is_section_header is True


class TestNonSections:
    """Tests for text that should not be detected as sections."""

    def test_regular_sentence(self) -> None:
        """Test regular sentence is not detected as section."""
        result = detect_section("The witness stated that he saw the suspect.")
        assert result.is_section_header is False
        assert result.section_name is None

    def test_long_text(self) -> None:
        """Test long text is not detected as section."""
        long_text = "THIS IS A VERY LONG SENTENCE THAT GOES ON AND ON AND ON"
        result = detect_section(long_text, max_length=50)
        assert result.is_section_header is False

    def test_empty_text(self) -> None:
        """Test empty text."""
        result = detect_section("")
        assert result.is_section_header is False


class TestSectionTracker:
    """Tests for SectionTracker class."""

    def test_initial_state(self) -> None:
        """Test tracker starts with no section."""
        tracker = SectionTracker()
        assert tracker.current_section is None

    def test_update_section(self) -> None:
        """Test updating section."""
        tracker = SectionTracker()
        tracker.update_section("INTERVIEW")
        assert tracker.current_section == "INTERVIEW"

    def test_reset(self) -> None:
        """Test resetting tracker."""
        tracker = SectionTracker()
        tracker.update_section("INTERVIEW")
        tracker.reset()
        assert tracker.current_section is None


class TestAssignSections:
    """Tests for section assignment across blocks."""

    def test_assigns_section_to_following_blocks(self) -> None:
        """Test section header assigns to subsequent blocks."""
        blocks = [
            {"block_id": "b1", "text": "INTERVIEW"},
            {"block_id": "b2", "text": "Q: What happened?"},
            {"block_id": "b3", "text": "A: I was at home."},
        ]
        results = assign_sections(blocks)

        assert results["b1"] == "INTERVIEW"
        assert results["b2"] == "INTERVIEW"
        assert results["b3"] == "INTERVIEW"

    def test_section_changes(self) -> None:
        """Test section changes when new header appears."""
        blocks = [
            {"block_id": "b1", "text": "INTERVIEW"},
            {"block_id": "b2", "text": "Interview content"},
            {"block_id": "b3", "text": "SUMMARY"},
            {"block_id": "b4", "text": "Summary content"},
        ]
        results = assign_sections(blocks)

        assert results["b1"] == "INTERVIEW"
        assert results["b2"] == "INTERVIEW"
        assert results["b3"] == "SUMMARY"
        assert results["b4"] == "SUMMARY"

    def test_no_initial_section(self) -> None:
        """Test blocks before first section header have None."""
        blocks = [
            {"block_id": "b1", "text": "Preamble text"},
            {"block_id": "b2", "text": "STATEMENT"},
            {"block_id": "b3", "text": "Statement content"},
        ]
        results = assign_sections(blocks)

        assert results["b1"] is None
        assert results["b2"] == "STATEMENT"
        assert results["b3"] == "STATEMENT"


class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_same_input_same_output(self) -> None:
        """Verify same input produces identical output."""
        blocks = [
            {"block_id": "b1", "text": "INTERVIEW"},
            {"block_id": "b2", "text": "Content here"},
        ]

        result1 = assign_sections(blocks)
        result2 = assign_sections(blocks)

        assert result1 == result2
