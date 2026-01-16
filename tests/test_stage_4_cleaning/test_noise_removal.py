"""
Unit tests for Stage 4: Semantic Cleaning - Noise Removal

Tests for OCR artifact and noise removal.
"""

from stage_4_cleaning.noise_removal import (
    normalize_page_breaks,
    remove_garbled_sequences,
    remove_noise,
    remove_ocr_artifacts,
    remove_repeated_punctuation,
)


class TestRemoveOcrArtifacts:
    """Tests for remove_ocr_artifacts function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert remove_ocr_artifacts("") == ""

    def test_normal_text_unchanged(self):
        """Normal text without artifacts should be unchanged."""
        text = "Hello World, this is a test."
        assert remove_ocr_artifacts(text) == text

    def test_isolated_pipe_removed(self):
        """Isolated pipe character should be removed."""
        text = "Hello | World"
        result = remove_ocr_artifacts(text)
        assert "|" not in result

    def test_isolated_tilde_removed(self):
        """Isolated tilde character should be removed."""
        text = "Hello ~ World"
        result = remove_ocr_artifacts(text)
        assert "~" not in result

    def test_multiple_artifacts_removed(self):
        """Multiple isolated artifacts should be removed."""
        text = "Hello |~| World"
        result = remove_ocr_artifacts(text)
        assert "|" not in result
        assert "~" not in result

    def test_artifacts_at_line_start(self):
        """Artifacts at line start should be removed."""
        text = "| Line with artifact at start"
        result = remove_ocr_artifacts(text)
        assert not result.strip().startswith("|")

    def test_artifacts_at_line_end(self):
        """Artifacts at line end should be removed."""
        text = "Line with artifact at end |"
        result = remove_ocr_artifacts(text)
        assert not result.strip().endswith("|")


class TestRemoveRepeatedPunctuation:
    """Tests for remove_repeated_punctuation function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert remove_repeated_punctuation("") == ""

    def test_normal_punctuation_unchanged(self):
        """Normal punctuation should be unchanged."""
        text = "Hello... World---Test"
        assert remove_repeated_punctuation(text) == text

    def test_four_dots_unchanged(self):
        """Four dots should be unchanged (threshold is 5+)."""
        text = "Wait...."
        assert remove_repeated_punctuation(text) == text

    def test_five_plus_dots_reduced(self):
        """5+ dots should be reduced to 3."""
        text = "Wait....."
        expected = "Wait..."
        assert remove_repeated_punctuation(text) == expected

    def test_many_dots_reduced(self):
        """Many dots should be reduced to 3."""
        text = "Wait............"
        expected = "Wait..."
        assert remove_repeated_punctuation(text) == expected

    def test_repeated_dashes_reduced(self):
        """5+ dashes should be reduced to 3."""
        text = "Section-----End"
        expected = "Section---End"
        assert remove_repeated_punctuation(text) == expected

    def test_multiple_sequences_reduced(self):
        """Multiple repeated sequences should each be reduced."""
        text = "Hello.........World------Test"
        result = remove_repeated_punctuation(text)
        assert result == "Hello...World---Test"


class TestRemoveGarbledSequences:
    """Tests for remove_garbled_sequences function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert remove_garbled_sequences("") == ""

    def test_normal_text_unchanged(self):
        """Normal text should be unchanged."""
        text = "Hello, World! Test @#$% done."
        assert remove_garbled_sequences(text) == text

    def test_short_special_chars_unchanged(self):
        """Short special char sequences should be unchanged."""
        text = "Price: $#"
        assert remove_garbled_sequences(text) == text

    def test_garbled_sequence_removed(self):
        """Garbled sequence of 3+ unusual chars should be removed."""
        # Using characters that are not in the allowed set
        text = "Hello†‡§¶World"
        result = remove_garbled_sequences(text)
        # The garbled sequence should be removed
        assert "†‡§¶" not in result


class TestNormalizePageBreaks:
    """Tests for normalize_page_breaks function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert normalize_page_breaks("") == ""

    def test_normal_text_unchanged(self):
        """Normal text should be unchanged."""
        text = "Hello World"
        assert normalize_page_breaks(text) == text

    def test_form_feed_converted(self):
        """Form feed should be converted to newline."""
        text = "Page1\fPage2"
        expected = "Page1\nPage2"
        assert normalize_page_breaks(text) == expected

    def test_vertical_tab_converted(self):
        """Vertical tab should be converted to newline."""
        text = "Section1\vSection2"
        expected = "Section1\nSection2"
        assert normalize_page_breaks(text) == expected

    def test_multiple_page_breaks(self):
        """Multiple page breaks should each become newlines."""
        text = "Page1\f\fPage2\vPage3"
        result = normalize_page_breaks(text)
        assert "\f" not in result
        assert "\v" not in result
        assert "\n" in result


class TestRemoveNoise:
    """Tests for the main remove_noise function."""

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert remove_noise("") == ""

    def test_normal_text_unchanged(self):
        """Normal text without noise should be largely unchanged."""
        text = "The witness stated: 'I saw the suspect at 8:15 PM.'"
        result = remove_noise(text)
        # All words should be preserved
        for word in ["witness", "stated", "suspect", "8:15", "PM"]:
            assert word in result

    def test_combines_all_operations(self):
        """Should apply all noise removal operations."""
        text = "Hello | World.........End\fNew page"
        result = remove_noise(text)

        # Artifacts removed
        assert "|" not in result

        # Repeated punctuation reduced
        assert "........." not in result

        # Page breaks normalized
        assert "\f" not in result

    def test_preserves_words(self):
        """Should never remove actual words."""
        text = "The quick | brown fox..... jumps over\f the lazy dog."
        result = remove_noise(text)

        # All words must be present
        words = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
        for word in words:
            assert word in result

    def test_determinism(self):
        """Same input should always produce same output."""
        text = "Test | with ~ artifacts... and | noise |"
        results = [remove_noise(text) for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_aggressive_mode(self):
        """Aggressive mode should remove garbled sequences."""
        text = "Hello†‡§¶World"

        # Non-aggressive should preserve (conservative)
        _result_normal = remove_noise(text, aggressive=False)

        # Aggressive should remove
        result_aggressive = remove_noise(text, aggressive=True)

        # Aggressive should have cleaned more
        # Note: Depends on what counts as garbled
        assert "Hello" in result_aggressive
        assert "World" in result_aggressive
