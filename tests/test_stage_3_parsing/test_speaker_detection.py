"""Tests for Stage 3: Structural Parsing - Speaker Detection."""

import pytest

from stage_3_parsing.speaker_detection import (
    SpeakerDetectionResult,
    detect_speaker,
    normalize_speaker_name,
)


class TestDetectSpeaker:
    """Tests for speaker label detection."""

    def test_detective_format(self) -> None:
        """Test DET. NAME: format."""
        result = detect_speaker("DET. SMITH: Where were you on the night of March 15?")
        assert result.speaker == "DET. SMITH"
        assert result.cleaned_text == "Where were you on the night of March 15?"

    def test_officer_format(self) -> None:
        """Test OFFICER NAME: format."""
        result = detect_speaker("OFFICER JOHN DOE: Please state your name.")
        assert result.speaker == "OFFICER JOHN DOE"
        assert result.cleaned_text == "Please state your name."

    def test_witness_format(self) -> None:
        """Test WITNESS: format."""
        result = detect_speaker("WITNESS: I was at home that evening.")
        assert result.speaker == "WITNESS"
        assert result.cleaned_text == "I was at home that evening."

    def test_qa_format_question(self) -> None:
        """Test Q: format."""
        result = detect_speaker("Q: What time did you arrive?")
        assert result.speaker == "Q"
        assert result.cleaned_text == "What time did you arrive?"

    def test_qa_format_answer(self) -> None:
        """Test A: format."""
        result = detect_speaker("A: Around 8:15 PM.")
        assert result.speaker == "A"
        assert result.cleaned_text == "Around 8:15 PM."

    def test_the_court_format(self) -> None:
        """Test THE COURT: format."""
        result = detect_speaker("THE COURT: Objection sustained.")
        assert result.speaker == "THE COURT"
        assert result.cleaned_text == "Objection sustained."

    def test_honorific_format(self) -> None:
        """Test MR./MRS./DR. NAME: format."""
        result = detect_speaker("MR. JOHNSON: I object to this line of questioning.")
        assert result.speaker == "MR. JOHNSON"
        assert result.cleaned_text == "I object to this line of questioning."

    def test_no_speaker_label(self) -> None:
        """Test text without speaker label."""
        text = "I was walking down the street when I heard a noise."
        result = detect_speaker(text)
        assert result.speaker is None
        assert result.cleaned_text == text

    def test_empty_text(self) -> None:
        """Test empty input."""
        result = detect_speaker("")
        assert result.speaker is None
        assert result.cleaned_text == ""

    def test_whitespace_only(self) -> None:
        """Test whitespace-only input."""
        result = detect_speaker("   ")
        assert result.speaker is None
        assert result.cleaned_text == "   "

    def test_colon_in_middle_not_speaker(self) -> None:
        """Test that colons in middle of text don't trigger detection."""
        text = "The time was 8:15 PM when I arrived."
        result = detect_speaker(text)
        assert result.speaker is None
        assert result.cleaned_text == text

    def test_sergeant_format(self) -> None:
        """Test SGT. NAME: format."""
        result = detect_speaker("SGT. MILLER: Report to my office.")
        assert result.speaker == "SGT. MILLER"
        assert result.cleaned_text == "Report to my office."

    def test_detective_full_word(self) -> None:
        """Test DETECTIVE NAME: format."""
        result = detect_speaker("DETECTIVE JONES: Tell me what happened.")
        assert result.speaker == "DETECTIVE JONES"
        assert result.cleaned_text == "Tell me what happened."


class TestNormalizeSpeakerName:
    """Tests for speaker name normalization."""

    def test_already_normalized(self) -> None:
        """Test already normalized name."""
        assert normalize_speaker_name("DET. SMITH") == "DET. SMITH"

    def test_abbreviation_expansion(self) -> None:
        """Test abbreviation is preserved with proper formatting."""
        # The function normalizes spacing but preserves abbreviations
        result = normalize_speaker_name("DET SMITH")
        assert result == "DETECTIVE SMITH"

    def test_empty_input(self) -> None:
        """Test empty input returns empty."""
        assert normalize_speaker_name("") == ""


class TestDeterminism:
    """Tests verifying deterministic behavior."""

    def test_same_input_same_output(self) -> None:
        """Verify same input produces identical output."""
        text = "DET. SMITH: Where were you on March 15?"

        result1 = detect_speaker(text)
        result2 = detect_speaker(text)

        assert result1.speaker == result2.speaker
        assert result1.cleaned_text == result2.cleaned_text

    def test_multiple_runs_consistent(self) -> None:
        """Run detection multiple times and verify consistency."""
        text = "OFFICER JOHNSON: Please describe the suspect."

        results = [detect_speaker(text) for _ in range(100)]

        assert all(r.speaker == "OFFICER JOHNSON" for r in results)
        assert all(r.cleaned_text == "Please describe the suspect." for r in results)
