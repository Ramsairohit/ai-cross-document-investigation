"""
Unit tests for Stage 4: Semantic Cleaning - Whitespace Normalizer

Tests for whitespace and newline normalization.
"""

from stage_4_cleaning.whitespace_normalizer import (
    clean_space_around_newlines,
    collapse_multiple_newlines,
    collapse_multiple_spaces,
    normalize_newlines,
    normalize_whitespace,
    trim_whitespace,
)


class TestNormalizeNewlines:
    """Tests for normalize_newlines function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_newlines("") == ""

    def test_unix_newlines_preserved(self):
        """Unix newlines should be unchanged."""
        text = "Line1\nLine2\nLine3"
        assert normalize_newlines(text) == text

    def test_windows_newlines_converted(self):
        """Windows CRLF should be converted to LF."""
        text = "Line1\r\nLine2\r\nLine3"
        expected = "Line1\nLine2\nLine3"
        assert normalize_newlines(text) == expected

    def test_old_mac_newlines_converted(self):
        """Old Mac CR should be converted to LF."""
        text = "Line1\rLine2\rLine3"
        expected = "Line1\nLine2\nLine3"
        assert normalize_newlines(text) == expected

    def test_mixed_newlines(self):
        """Mixed newline formats should all become LF."""
        text = "Line1\r\nLine2\rLine3\nLine4"
        expected = "Line1\nLine2\nLine3\nLine4"
        assert normalize_newlines(text) == expected


class TestCollapseMultipleSpaces:
    """Tests for collapse_multiple_spaces function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert collapse_multiple_spaces("") == ""

    def test_single_spaces_preserved(self):
        """Single spaces between words should be preserved."""
        text = "Hello World Test"
        assert collapse_multiple_spaces(text) == text

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces should become single space."""
        text = "Hello    World   Test"
        expected = "Hello World Test"
        assert collapse_multiple_spaces(text) == expected

    def test_tabs_collapsed_to_space(self):
        """Tabs should be collapsed with spaces."""
        text = "Hello\t\tWorld"
        expected = "Hello World"
        assert collapse_multiple_spaces(text) == expected

    def test_newlines_not_affected(self):
        """Newlines should not be affected by space collapse."""
        text = "Hello   World\n\nTest"
        expected = "Hello World\n\nTest"
        assert collapse_multiple_spaces(text) == expected


class TestCollapseMultipleNewlines:
    """Tests for collapse_multiple_newlines function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert collapse_multiple_newlines("") == ""

    def test_single_newline_preserved(self):
        """Single newline should be preserved."""
        text = "Line1\nLine2"
        assert collapse_multiple_newlines(text) == text

    def test_double_newline_preserved(self):
        """Double newline (paragraph break) should be preserved."""
        text = "Para1\n\nPara2"
        assert collapse_multiple_newlines(text) == text

    def test_triple_plus_newlines_collapsed(self):
        """3+ newlines should be collapsed to 2."""
        text = "Para1\n\n\nPara2"
        expected = "Para1\n\nPara2"
        assert collapse_multiple_newlines(text) == expected

    def test_many_newlines_collapsed(self):
        """Many newlines should be collapsed to 2."""
        text = "Para1\n\n\n\n\n\nPara2"
        expected = "Para1\n\nPara2"
        assert collapse_multiple_newlines(text) == expected


class TestCleanSpaceAroundNewlines:
    """Tests for clean_space_around_newlines function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert clean_space_around_newlines("") == ""

    def test_trailing_spaces_before_newline(self):
        """Trailing spaces before newline should be removed."""
        text = "Hello   \nWorld"
        expected = "Hello\nWorld"
        assert clean_space_around_newlines(text) == expected

    def test_leading_spaces_after_newline(self):
        """Leading spaces after newline should be removed."""
        text = "Hello\n   World"
        expected = "Hello\nWorld"
        assert clean_space_around_newlines(text) == expected

    def test_spaces_both_sides(self):
        """Spaces on both sides of newline should be removed."""
        text = "Hello   \n   World"
        expected = "Hello\nWorld"
        assert clean_space_around_newlines(text) == expected


class TestTrimWhitespace:
    """Tests for trim_whitespace function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert trim_whitespace("") == ""

    def test_no_trim_needed(self):
        """Text without leading/trailing whitespace unchanged."""
        text = "Hello World"
        assert trim_whitespace(text) == text

    def test_leading_whitespace_removed(self):
        """Leading whitespace should be removed."""
        text = "   Hello World"
        expected = "Hello World"
        assert trim_whitespace(text) == expected

    def test_trailing_whitespace_removed(self):
        """Trailing whitespace should be removed."""
        text = "Hello World   "
        expected = "Hello World"
        assert trim_whitespace(text) == expected

    def test_both_ends_trimmed(self):
        """Whitespace on both ends should be removed."""
        text = "   Hello World   "
        expected = "Hello World"
        assert trim_whitespace(text) == expected

    def test_internal_whitespace_preserved(self):
        """Internal whitespace should be preserved."""
        text = "   Hello   World   "
        expected = "Hello   World"
        assert trim_whitespace(text) == expected


class TestNormalizeWhitespace:
    """Tests for the main normalize_whitespace function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_whitespace("") == ""

    def test_full_normalization(self):
        """Should apply all whitespace normalizations."""
        text = "  I   heard   a loud   crash at 8:15 PM  "
        expected = "I heard a loud crash at 8:15 PM"
        assert normalize_whitespace(text) == expected

    def test_preserves_sentence_order(self):
        """Should preserve sentence order."""
        text = "First sentence.   Second sentence.   Third sentence."
        result = normalize_whitespace(text)
        assert result.index("First") < result.index("Second") < result.index("Third")

    def test_preserves_punctuation(self):
        """Should preserve punctuation exactly."""
        text = "  Hello, World! How are you?  "
        result = normalize_whitespace(text)
        assert "," in result
        assert "!" in result
        assert "?" in result

    def test_determinism(self):
        """Same input should always produce same output."""
        text = "  Multiple   spaces   and\n\n\nnewlines  \r\n  here  "
        results = [normalize_whitespace(text) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_complex_document(self):
        """Should handle complex document formatting."""
        text = """  WITNESS STATEMENT  

        Name:   John Smith
        Date:    March 15, 2024

        I   heard   a loud   crash   at approximately 8:15 PM.
        The   vehicle   was   traveling   westbound.  """

        result = normalize_whitespace(text)

        # Check normalization occurred
        assert "  " not in result  # No double spaces
        assert result.startswith("WITNESS")  # Trimmed
        assert result.endswith("westbound.")  # Trimmed

        # Check content preserved
        assert "John Smith" in result
        assert "March 15, 2024" in result
        assert "8:15 PM" in result

    def test_configurable_options(self):
        """Should respect configuration options."""
        text = "  Hello   World  "

        # With all options enabled (default)
        assert normalize_whitespace(text) == "Hello World"

        # Without trimming
        result = normalize_whitespace(text, trim=False)
        assert result.strip() != result  # Has leading/trailing

        # Without space collapse
        result = normalize_whitespace(text, collapse_spaces=False)
        assert "   " in result  # Multiple spaces preserved
