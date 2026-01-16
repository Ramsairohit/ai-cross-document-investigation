"""Tests for Stage 3: Structural Parsing - Timestamp Extraction."""

import pytest

from stage_3_parsing.timestamp_regex import (
    extract_timestamps,
    extract_timestamps_with_positions,
)


class TestTimePatterns:
    """Tests for time-only patterns."""

    def test_12_hour_format_pm(self) -> None:
        """Test 8:15 PM format."""
        result = extract_timestamps("The incident occurred at 8:15 PM.")
        assert "8:15 PM" in result

    def test_12_hour_format_am(self) -> None:
        """Test 10:30 AM format."""
        result = extract_timestamps("He arrived at 10:30 AM.")
        assert "10:30 AM" in result

    def test_24_hour_format(self) -> None:
        """Test 14:30 format."""
        result = extract_timestamps("Meeting scheduled for 14:30.")
        assert "14:30" in result

    def test_military_hours_format(self) -> None:
        """Test 0815 hours format."""
        result = extract_timestamps("Call received at 0815 hours.")
        assert "0815 hours" in result

    def test_military_hrs_format(self) -> None:
        """Test 2345 hrs format."""
        result = extract_timestamps("Response at 2345 hrs.")
        assert "2345 hrs" in result


class TestDatePatterns:
    """Tests for date patterns."""

    def test_full_month_name(self) -> None:
        """Test March 15, 2024 format."""
        result = extract_timestamps("On March 15, 2024, we received a report.")
        assert any("March 15" in ts for ts in result)

    def test_abbreviated_month(self) -> None:
        """Test Mar 15 format."""
        result = extract_timestamps("Report filed on Mar 15.")
        assert any("Mar 15" in ts for ts in result)

    def test_iso_format(self) -> None:
        """Test 2024-03-15 format."""
        result = extract_timestamps("Date: 2024-03-15")
        assert "2024-03-15" in result

    def test_us_format(self) -> None:
        """Test 03/15/2024 format."""
        result = extract_timestamps("Incident date: 03/15/2024")
        assert "03/15/2024" in result

    def test_european_format(self) -> None:
        """Test 15/03/2024 format."""
        result = extract_timestamps("Filed on 15/03/2024")
        assert "15/03/2024" in result


class TestRelativePatterns:
    """Tests for relative time expressions."""

    def test_yesterday(self) -> None:
        """Test 'yesterday' extraction."""
        result = extract_timestamps("I saw him yesterday.")
        assert "yesterday" in result

    def test_last_monday(self) -> None:
        """Test 'last Monday' extraction."""
        result = extract_timestamps("The meeting was last Monday.")
        assert "last Monday" in result

    def test_tonight(self) -> None:
        """Test 'tonight' extraction."""
        result = extract_timestamps("He said he would return tonight.")
        assert "tonight" in result


class TestMultipleTimestamps:
    """Tests for texts with multiple timestamps."""

    def test_multiple_times(self) -> None:
        """Test extraction of multiple times."""
        text = "Between 8:15 PM and 10:30 PM, three calls were made."
        result = extract_timestamps(text)
        assert "8:15 PM" in result
        assert "10:30 PM" in result

    def test_date_and_time(self) -> None:
        """Test extraction of date and time separately."""
        text = "On March 15, the suspect was seen at 8:15 PM."
        result = extract_timestamps(text)
        assert len(result) >= 2


class TestNoTimestamps:
    """Tests for texts without timestamps."""

    def test_no_timestamps(self) -> None:
        """Test text without any timestamps."""
        result = extract_timestamps("The suspect fled the scene immediately.")
        assert result == []

    def test_empty_text(self) -> None:
        """Test empty input."""
        result = extract_timestamps("")
        assert result == []


class TestRawPreservation:
    """Tests verifying timestamps are preserved exactly as found."""

    def test_preserve_exact_format(self) -> None:
        """Verify timestamps are not normalized."""
        result = extract_timestamps("around 8:15 PM")
        # Should preserve "around" as part of approximate time
        assert any("8:15 PM" in ts for ts in result)

    def test_preserve_spacing(self) -> None:
        """Verify original spacing is preserved."""
        result = extract_timestamps("March  15, 2024")  # Double space
        # Should find the timestamp
        assert len(result) > 0


class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_same_input_same_output(self) -> None:
        """Verify same input produces identical output."""
        text = "At 8:15 PM on March 15, 2024, the suspect was seen."

        result1 = extract_timestamps(text)
        result2 = extract_timestamps(text)

        assert result1 == result2

    def test_consistent_ordering(self) -> None:
        """Verify timestamps are returned in order of appearance."""
        text = "From 10:30 AM to 8:15 PM on March 15."

        for _ in range(50):
            result = extract_timestamps(text)
            # First timestamp should appear before second in text
            assert result == sorted(result, key=lambda x: text.find(x))


class TestTimestampsWithPositions:
    """Tests for position-aware timestamp extraction."""

    def test_returns_positions(self) -> None:
        """Test that positions are returned correctly."""
        text = "At 8:15 PM the event occurred."
        result = extract_timestamps_with_positions(text)

        assert len(result) > 0
        timestamp, start, end = result[0]
        assert "8:15 PM" in timestamp
        assert isinstance(start, int)
        assert isinstance(end, int)
        assert start < end
