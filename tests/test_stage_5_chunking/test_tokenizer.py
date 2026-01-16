"""
Unit tests for Stage 5: Logical Chunking - Tokenizer

Tests for deterministic token counting.
"""

import pytest

from stage_5_chunking.tokenizer import (
    count_tokens,
    count_tokens_batch,
    get_encoding,
    split_text_by_tokens,
)


class TestGetEncoding:
    """Tests for encoding retrieval."""

    def test_default_encoding(self):
        """Should get cl100k_base encoding by default."""
        encoding = get_encoding()
        assert encoding is not None
        assert encoding.name == "cl100k_base"

    def test_encoding_cached(self):
        """Same encoding should be cached."""
        enc1 = get_encoding("cl100k_base")
        enc2 = get_encoding("cl100k_base")
        assert enc1 is enc2


class TestCountTokens:
    """Tests for token counting."""

    def test_empty_string(self):
        """Empty string should have 0 tokens."""
        assert count_tokens("") == 0

    def test_simple_text(self):
        """Should count tokens in simple text."""
        count = count_tokens("Hello, world!")
        assert count > 0
        assert isinstance(count, int)

    def test_deterministic(self):
        """Same text should always produce same count."""
        text = "This is a test sentence for token counting."
        count1 = count_tokens(text)
        count2 = count_tokens(text)
        assert count1 == count2

    def test_longer_text_more_tokens(self):
        """Longer text should have more tokens."""
        short = "Hello"
        long = "Hello there, this is a much longer piece of text."
        assert count_tokens(long) > count_tokens(short)

    def test_special_characters(self):
        """Should handle special characters."""
        text = "Test with émojis 🚀 and spëcial characters!"
        count = count_tokens(text)
        assert count > 0


class TestCountTokensBatch:
    """Tests for batch token counting."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert count_tokens_batch([]) == []

    def test_batch_matches_individual(self):
        """Batch counts should match individual counts."""
        texts = ["Hello", "World", "Test text"]
        batch_counts = count_tokens_batch(texts)
        individual_counts = [count_tokens(t) for t in texts]
        assert batch_counts == individual_counts

    def test_handles_empty_strings(self):
        """Should handle empty strings in batch."""
        texts = ["Hello", "", "World"]
        counts = count_tokens_batch(texts)
        assert counts[1] == 0


class TestSplitTextByTokens:
    """Tests for text splitting."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert split_text_by_tokens("", 100) == []

    def test_short_text_no_split(self):
        """Short text should not be split."""
        text = "This is a short text."
        chunks = split_text_by_tokens(text, 100)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_long_text(self):
        """Long text should be split into chunks."""
        # Create text that will definitely exceed 10 tokens
        text = " ".join(["word"] * 100)
        chunks = split_text_by_tokens(text, 10)
        assert len(chunks) > 1

    def test_each_chunk_within_limit(self):
        """Each chunk should be within token limit."""
        text = " ".join(["hello"] * 50)
        max_tokens = 5
        chunks = split_text_by_tokens(text, max_tokens)
        for chunk in chunks:
            assert count_tokens(chunk) <= max_tokens

    def test_deterministic_splitting(self):
        """Splitting should be deterministic."""
        text = " ".join(["test"] * 100)
        chunks1 = split_text_by_tokens(text, 20)
        chunks2 = split_text_by_tokens(text, 20)
        assert chunks1 == chunks2


class TestDeterminismGuarantee:
    """Comprehensive determinism tests."""

    def test_100_runs_identical(self):
        """100 runs should produce identical results."""
        text = "The detective interrogated the suspect about the incident at 8:15 PM."
        expected_count = count_tokens(text)

        for _ in range(100):
            assert count_tokens(text) == expected_count

    def test_varied_inputs_deterministic(self):
        """Various inputs should all be deterministic."""
        test_cases = [
            "Simple text",
            "Text with numbers 123 456",
            "DET. SMITH: Where were you?",
            "Multi\nline\ntext",
            "Tab\tseparated\ttext",
        ]

        for text in test_cases:
            count1 = count_tokens(text)
            count2 = count_tokens(text)
            assert count1 == count2, f"Non-deterministic for: {text}"
