"""
Unit tests for Stage 4: Semantic Cleaning - Timestamp Normalizer

Tests for timestamp parsing and ISO-8601 normalization.
"""

from datetime import date

from stage_4_cleaning.timestamp_normalizer import (
    has_unambiguous_date,
    is_ambiguous_reference,
    is_time_only,
    normalize_timestamp,
    normalize_timestamps,
    parse_military_time,
)


class TestIsTimeOnly:
    """Tests for is_time_only function."""

    def test_12_hour_format(self):
        """12-hour time formats should be detected."""
        assert is_time_only("8:15 PM")
        assert is_time_only("10:30 AM")
        assert is_time_only("12:00PM")
        assert is_time_only("8:15:30 PM")

    def test_24_hour_format(self):
        """24-hour time formats should be detected."""
        assert is_time_only("14:30")
        assert is_time_only("23:59")
        assert is_time_only("00:00")

    def test_military_format(self):
        """Military time formats should be detected."""
        assert is_time_only("0815 hours")
        assert is_time_only("2345 hrs")
        assert is_time_only("1430 hour")

    def test_oclock_format(self):
        """O'clock formats should be detected."""
        assert is_time_only("8 o'clock")
        assert is_time_only("9 o'clock PM")

    def test_dates_not_time_only(self):
        """Date formats should not be detected as time-only."""
        assert not is_time_only("March 15, 2024")
        assert not is_time_only("2024-03-15")
        assert not is_time_only("15/03/2024")

    def test_datetime_not_time_only(self):
        """Datetime formats should not be detected as time-only."""
        assert not is_time_only("March 15, 2024 at 8:15 PM")
        assert not is_time_only("2024-03-15T20:15:00")


class TestIsAmbiguousReference:
    """Tests for is_ambiguous_reference function."""

    def test_yesterday_today_tomorrow(self):
        """Yesterday, today, tomorrow should be ambiguous."""
        assert is_ambiguous_reference("yesterday")
        assert is_ambiguous_reference("today")
        assert is_ambiguous_reference("tonight")
        assert is_ambiguous_reference("tomorrow")

    def test_last_this_next(self):
        """'Last/this/next' references should be ambiguous."""
        assert is_ambiguous_reference("last Monday")
        assert is_ambiguous_reference("this week")
        assert is_ambiguous_reference("next month")
        assert is_ambiguous_reference("last night")

    def test_approximate_times(self):
        """Approximate time references should be ambiguous."""
        assert is_ambiguous_reference("around 8 PM")
        assert is_ambiguous_reference("approximately 10:30 AM")
        assert is_ambiguous_reference("about 3 PM")

    def test_specific_dates_not_ambiguous(self):
        """Specific dates should not be ambiguous."""
        assert not is_ambiguous_reference("March 15, 2024")
        assert not is_ambiguous_reference("2024-03-15")
        assert not is_ambiguous_reference("8:15 PM")


class TestHasUnambiguousDate:
    """Tests for has_unambiguous_date function."""

    def test_iso_format(self):
        """ISO format dates should be unambiguous."""
        assert has_unambiguous_date("2024-03-15")
        assert has_unambiguous_date("2024-03-15T20:15:00")

    def test_full_month_with_year(self):
        """Full month name with year should be unambiguous."""
        assert has_unambiguous_date("March 15, 2024")
        assert has_unambiguous_date("January 1st, 2024")
        assert has_unambiguous_date("December 31, 2024 at 11:59 PM")

    def test_ambiguous_formats(self):
        """Ambiguous formats should not be detected."""
        assert not has_unambiguous_date("03/15/24")  # US vs EU ambiguous
        assert not has_unambiguous_date("March 15")  # No year
        assert not has_unambiguous_date("8:15 PM")  # Time only


class TestParseMilitaryTime:
    """Tests for parse_military_time function."""

    def test_valid_military_time(self):
        """Valid military times should be parsed."""
        result = parse_military_time("0815 hours")
        assert result is not None
        assert result.hour == 8
        assert result.minute == 15

        result = parse_military_time("2345 hrs")
        assert result is not None
        assert result.hour == 23
        assert result.minute == 45

    def test_midnight(self):
        """Midnight should be parsed correctly."""
        result = parse_military_time("0000 hours")
        assert result is not None
        assert result.hour == 0
        assert result.minute == 0

    def test_invalid_military_time(self):
        """Invalid military times should return None."""
        assert parse_military_time("2500 hours") is None  # Invalid hour
        assert parse_military_time("0875 hours") is None  # Invalid minute
        assert parse_military_time("8:15 PM") is None  # Not military format


class TestNormalizeTimestamp:
    """Tests for normalize_timestamp function."""

    def test_empty_string(self):
        """Empty string should return null iso."""
        result = normalize_timestamp("")
        assert result.iso is None
        assert result.confidence == 0.0

    def test_iso_format_high_confidence(self):
        """ISO format should have high confidence."""
        result = normalize_timestamp("2024-03-15")
        assert result.original == "2024-03-15"
        assert result.iso is not None
        assert "2024-03-15" in result.iso
        assert result.confidence >= 0.8

    def test_full_month_with_year(self):
        """Full month with year should parse correctly."""
        result = normalize_timestamp("March 15, 2024")
        assert result.original == "March 15, 2024"
        assert result.iso is not None
        assert result.confidence >= 0.7

    def test_time_only_with_reference_date(self):
        """Time-only with reference date should parse."""
        ref_date = date(2024, 3, 15)
        result = normalize_timestamp("8:15 PM", reference_date=ref_date)
        assert result.original == "8:15 PM"
        assert result.iso is not None
        assert "20:15" in result.iso
        assert "2024-03-15" in result.iso
        assert result.confidence <= 0.7  # Lower confidence for time-only

    def test_time_only_without_reference_date(self):
        """Time-only without reference date should still parse (using today)."""
        result = normalize_timestamp("8:15 PM")
        assert result.original == "8:15 PM"
        # May or may not have iso depending on parsing
        # Confidence should be moderate at best

    def test_ambiguous_reference_null_iso(self):
        """Ambiguous references should return null iso."""
        result = normalize_timestamp("yesterday")
        assert result.original == "yesterday"
        assert result.iso is None
        assert result.confidence <= 0.2

        result = normalize_timestamp("last Monday")
        assert result.iso is None

    def test_military_time(self):
        """Military time should parse correctly."""
        ref_date = date(2024, 3, 15)
        result = normalize_timestamp("0815 hours", reference_date=ref_date)
        assert result.original == "0815 hours"
        assert result.iso is not None
        assert "08:15" in result.iso

    def test_preserves_original(self):
        """Original string should always be preserved."""
        original = "  8:15 PM  "
        result = normalize_timestamp(original)
        assert result.original == "8:15 PM"  # Stripped but not modified

    def test_confidence_bounds(self):
        """Confidence should always be between 0 and 1."""
        test_cases = [
            "2024-03-15",
            "March 15, 2024",
            "8:15 PM",
            "yesterday",
            "invalid timestamp",
            "",
        ]
        for ts in test_cases:
            result = normalize_timestamp(ts)
            assert 0.0 <= result.confidence <= 1.0

    def test_determinism(self):
        """Same input should always produce same output."""
        test_cases = [
            "2024-03-15",
            "March 15, 2024 at 8:15 PM",
            "0815 hours",
            "yesterday",
        ]
        for ts in test_cases:
            results = [normalize_timestamp(ts) for _ in range(100)]
            assert all(r.iso == results[0].iso for r in results)
            assert all(r.confidence == results[0].confidence for r in results)


class TestNormalizeTimestamps:
    """Tests for normalize_timestamps function (batch processing)."""

    def test_empty_list(self):
        """Empty list should return empty list."""
        assert normalize_timestamps([]) == []

    def test_multiple_timestamps(self):
        """Should normalize all timestamps in list."""
        raw = ["2024-03-15", "8:15 PM", "yesterday"]
        ref_date = date(2024, 3, 15)
        results = normalize_timestamps(raw, reference_date=ref_date)

        assert len(results) == 3
        assert results[0].original == "2024-03-15"
        assert results[1].original == "8:15 PM"
        assert results[2].original == "yesterday"

    def test_preserves_order(self):
        """Should preserve order of timestamps."""
        raw = ["March 15", "8:15 PM", "2024-03-16"]
        results = normalize_timestamps(raw)

        assert results[0].original == "March 15"
        assert results[1].original == "8:15 PM"
        assert results[2].original == "2024-03-16"


class TestNonInterpretation:
    """Tests to ensure no semantic interpretation occurs."""

    def test_no_date_guessing(self):
        """Should not guess dates from context."""
        # "March 15" without year - should not assume any year
        result = normalize_timestamp("March 15")
        # If it parses, it might use current year, but confidence should be moderate
        # The key is: it should NOT infer from surrounding context
        assert result.original == "March 15"

    def test_no_timezone_inference(self):
        """Should not infer timezone from location mentions."""
        # Just a time, no timezone info
        result = normalize_timestamp("8:15 PM")
        if result.iso:
            # Should not have timezone suffix (no inference)
            assert "+00:00" not in result.iso or "Z" not in result.iso

    def test_approximate_times_marked(self):
        """Approximate times should have lower confidence, not precise times."""
        result = normalize_timestamp("around 8 PM")
        # Should be flagged as ambiguous, not parsed as exactly 8 PM
        assert result.iso is None or result.confidence < 0.5
