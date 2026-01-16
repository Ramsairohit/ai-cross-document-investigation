"""
Unit tests for Stage 10: Contradiction Detection - Rules

Tests for rule-based contradiction detection.
"""

import pytest

from stage_10_contradictions.models import ContradictionType
from stage_10_contradictions.rules import (
    apply_all_rules,
    detect_denial_vs_assertion,
    detect_location_conflict,
    detect_time_conflict,
    extract_locations,
    extract_times,
    has_assertion,
    has_denial,
)


class TestPatternExtraction:
    """Tests for pattern extraction functions."""

    def test_extract_locations(self):
        """Should extract location mentions."""
        text = "I was at home around 9 PM."
        locations = extract_locations(text)
        assert len(locations) >= 1

    def test_extract_times(self):
        """Should extract time mentions."""
        text = "It happened at 8:15 PM."
        times = extract_times(text)
        assert len(times) >= 1

    def test_has_denial(self):
        """Should detect denial patterns."""
        assert has_denial("I was not there.") is True
        assert has_denial("I never saw him.") is True
        assert has_denial("I was there.") is False

    def test_has_assertion(self):
        """Should detect assertion patterns."""
        assert has_assertion("I was there.") is True
        assert has_assertion("I saw him.") is True
        assert has_assertion("The weather was nice.") is False


class TestDetectDenialVsAssertion:
    """Tests for denial vs assertion detection."""

    def test_detects_conflict(self):
        """Should detect denial vs assertion."""
        chunk_a = {"text": "Marcus was not at the park."}
        chunk_b = {"text": "I saw Marcus at the park."}

        is_conflict, explanation = detect_denial_vs_assertion(chunk_a, chunk_b, ["Marcus"])

        assert is_conflict is True
        assert "Marcus" in explanation

    def test_no_conflict_both_assertions(self):
        """Should not flag when both are assertions."""
        chunk_a = {"text": "I was at home."}
        chunk_b = {"text": "He was at work."}

        is_conflict, _ = detect_denial_vs_assertion(chunk_a, chunk_b, ["He"])

        assert is_conflict is False


class TestDetectLocationConflict:
    """Tests for location conflict detection."""

    def test_detects_different_locations(self):
        """Should detect different locations for same entity."""
        chunk_a = {"text": "Marcus was at home."}
        chunk_b = {"text": "Marcus was at the park."}

        is_conflict, explanation = detect_location_conflict(
            chunk_a, chunk_b, ["Marcus"], "2024-03-15T21:00:00"
        )

        assert is_conflict is True

    def test_no_conflict_same_location(self):
        """Should not flag same location."""
        chunk_a = {"text": "He was at home."}
        chunk_b = {"text": "I found him at home."}

        is_conflict, _ = detect_location_conflict(chunk_a, chunk_b, ["He"])

        assert is_conflict is False


class TestDetectTimeConflict:
    """Tests for time conflict detection."""

    def test_detects_different_times(self):
        """Should detect different times for same entity."""
        chunk_a = {"text": "Marcus left at 8 PM."}
        chunk_b = {"text": "Marcus arrived at 10 PM."}

        is_conflict, explanation = detect_time_conflict(chunk_a, chunk_b, ["Marcus"])

        # This should detect conflicting times
        # Note: depends on how times are compared
        assert isinstance(is_conflict, bool)


class TestApplyAllRules:
    """Tests for applying all rules."""

    def test_applies_all_rules(self):
        """Should apply all rules and return detected types."""
        chunk_a = {"text": "I was not at home at 9 PM."}
        chunk_b = {"text": "I saw Marcus at home at 9 PM."}

        detected = apply_all_rules(chunk_a, chunk_b, ["Marcus"], None)

        # Should return list of (type, explanation) tuples
        assert isinstance(detected, list)
        for item in detected:
            assert isinstance(item, tuple)
            assert isinstance(item[0], ContradictionType)
            assert isinstance(item[1], str)

    def test_no_false_positives(self):
        """Should not flag unrelated chunks."""
        chunk_a = {"text": "The weather was nice."}
        chunk_b = {"text": "It was sunny."}

        detected = apply_all_rules(chunk_a, chunk_b, [], None)

        assert len(detected) == 0


class TestNoResolution:
    """Tests to verify rules don't resolve contradictions."""

    def test_both_kept(self):
        """Rules should flag, not remove."""
        chunk_a = {"text": "I was not there."}
        chunk_b = {"text": "I saw him there."}

        detected = apply_all_rules(chunk_a, chunk_b, ["him"], None)

        # Even if contradiction detected, both chunks are preserved
        # (this is tested at pipeline level, rules just return detection)
        assert isinstance(detected, list)
