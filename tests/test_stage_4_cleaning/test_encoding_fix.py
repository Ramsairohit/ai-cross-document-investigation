"""
Unit tests for Stage 4: Semantic Cleaning - Encoding Module

Tests for encoding normalization and invalid character removal.
"""

from stage_4_cleaning.encoding_fix import (
    fix_encoding,
    normalize_encoding,
    remove_replacement_chars,
)


class TestNormalizeEncoding:
    """Tests for normalize_encoding function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_encoding("") == ""

    def test_normal_ascii(self):
        """Normal ASCII text should be preserved."""
        text = "Hello, World! This is a test."
        assert normalize_encoding(text) == text

    def test_unicode_preserved(self):
        """Valid Unicode characters should be preserved."""
        text = "Café résumé naïve"
        result = normalize_encoding(text)
        assert "Café" in result
        assert "résumé" in result
        assert "naïve" in result

    def test_unicode_normalization_nfc(self):
        """Unicode should be normalized to NFC form."""
        # é can be represented as single char or e + combining accent
        # NFC should combine them
        decomposed = "cafe\u0301"  # e + combining acute accent
        composed = "café"  # single é character
        result = normalize_encoding(decomposed)
        assert result == composed

    def test_control_chars_removed(self):
        """Control characters except tabs/newlines should be removed."""
        text = "Hello\x00World\x01Test\x02End"
        result = normalize_encoding(text)
        assert result == "HelloWorldTestEnd"
        assert "\x00" not in result
        assert "\x01" not in result

    def test_tabs_preserved(self):
        """Tab characters should be preserved."""
        text = "Column1\tColumn2\tColumn3"
        assert normalize_encoding(text) == text

    def test_newlines_preserved(self):
        """Newline characters should be preserved."""
        text = "Line1\nLine2\nLine3"
        assert normalize_encoding(text) == text

    def test_carriage_returns_preserved(self):
        """Carriage returns should be preserved."""
        text = "Line1\r\nLine2\r\nLine3"
        assert normalize_encoding(text) == text

    def test_null_bytes_removed(self):
        """Null bytes should be removed."""
        text = "Hello\x00World"
        assert normalize_encoding(text) == "HelloWorld"

    def test_private_use_chars_removed(self):
        """Private use characters should be removed."""
        text = "Hello\ue000World"  # Private use character
        assert normalize_encoding(text) == "HelloWorld"

    def test_punctuation_preserved(self):
        """Punctuation should be preserved."""
        text = "Hello! How are you? I'm fine, thanks."
        assert normalize_encoding(text) == text

    def test_numbers_preserved(self):
        """Numbers should be preserved."""
        text = "Case #24-890: Report dated 03/15/2024"
        assert normalize_encoding(text) == text


class TestRemoveReplacementChars:
    """Tests for remove_replacement_chars function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert remove_replacement_chars("") == ""

    def test_no_replacement_chars(self):
        """Text without replacement chars should be unchanged."""
        text = "Normal text without issues"
        assert remove_replacement_chars(text) == text

    def test_single_replacement_char(self):
        """Single replacement character should be removed."""
        text = "Hello\ufffdWorld"
        assert remove_replacement_chars(text) == "HelloWorld"

    def test_multiple_replacement_chars(self):
        """Multiple replacement characters should be removed."""
        text = "Hello\ufffd\ufffdWorld\ufffd"
        assert remove_replacement_chars(text) == "HelloWorld"


class TestFixEncoding:
    """Tests for the main fix_encoding function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert fix_encoding("") == ""

    def test_combines_all_fixes(self):
        """Should apply all encoding fixes."""
        text = "Hello\x00\ufffdWorld"
        result = fix_encoding(text)
        assert result == "HelloWorld"
        assert "\x00" not in result
        assert "\ufffd" not in result

    def test_determinism(self):
        """Same input should always produce same output."""
        text = "Café\x00résumé\ufffdnaïve"
        results = [fix_encoding(text) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_preserves_meaning(self):
        """Should preserve the meaning of text (words, order, punctuation)."""
        text = "The witness stated: 'I saw the suspect at 8:15 PM.'"
        result = fix_encoding(text)
        assert "witness" in result
        assert "suspect" in result
        assert "8:15 PM" in result
        assert "'" in result
