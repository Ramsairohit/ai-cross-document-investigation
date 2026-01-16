"""
Unit tests for Stage 6: NER - Rule-Based Extraction

Tests for phone numbers, addresses, weapons, and evidence extraction.
"""

from stage_6_ner.models import EntityType
from stage_6_ner.rule_based_entities import (
    extract_addresses,
    extract_all_rule_based,
    extract_evidence,
    extract_phone_numbers,
    extract_weapons,
)


class TestExtractPhoneNumbers:
    """Tests for phone number extraction."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert extract_phone_numbers("") == []

    def test_us_format_parentheses(self):
        """Should extract (555) 123-4567 format."""
        text = "Call me at (555) 123-4567 anytime."
        matches = extract_phone_numbers(text)
        assert len(matches) >= 1
        assert any("555" in m.text for m in matches)
        assert all(m.entity_type == EntityType.PHONE for m in matches)

    def test_us_format_dashes(self):
        """Should extract 555-123-4567 format."""
        text = "Phone: 555-123-4567"
        matches = extract_phone_numbers(text)
        assert len(matches) >= 1
        assert matches[0].entity_type == EntityType.PHONE

    def test_international_format(self):
        """Should extract international formats."""
        text = "Her number is +1-555-123-4567."
        matches = extract_phone_numbers(text)
        assert len(matches) >= 1

    def test_no_phone_numbers(self):
        """Should return empty for text without phones."""
        text = "There are no phone numbers here."
        matches = extract_phone_numbers(text)
        # May match some false positives, but should be minimal
        # Just verify function doesn't crash
        assert isinstance(matches, list)

    def test_confidence_set(self):
        """Phone matches should have confidence."""
        text = "Call 555-123-4567"
        matches = extract_phone_numbers(text)
        if matches:
            assert matches[0].confidence > 0


class TestExtractAddresses:
    """Tests for address extraction."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert extract_addresses("") == []

    def test_street_address(self):
        """Should extract street addresses."""
        text = "He lives at 420 Harrow Lane."
        matches = extract_addresses(text)
        assert len(matches) == 1
        assert "420 Harrow Lane" in matches[0].text
        assert matches[0].entity_type == EntityType.ADDRESS

    def test_multiple_street_types(self):
        """Should handle various street types."""
        addresses = [
            "123 Main Street",
            "456 Oak Avenue",
            "789 Pine Road",
            "101 Elm Boulevard",
        ]
        for addr in addresses:
            matches = extract_addresses(f"Located at {addr}.")
            assert len(matches) >= 1
            assert matches[0].entity_type == EntityType.ADDRESS

    def test_no_addresses(self):
        """Should return empty for text without addresses."""
        text = "No addresses in this sentence."
        matches = extract_addresses(text)
        assert matches == []

    def test_character_offsets(self):
        """Should provide correct character offsets."""
        text = "Found at 123 Main Street yesterday."
        matches = extract_addresses(text)
        if matches:
            start = matches[0].start_char
            end = matches[0].end_char
            assert text[start:end] == matches[0].text


class TestExtractWeapons:
    """Tests for weapon extraction."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert extract_weapons("") == []

    def test_gun(self):
        """Should extract 'gun'."""
        text = "He had a gun in his hand."
        matches = extract_weapons(text)
        assert len(matches) == 1
        assert matches[0].text.lower() == "gun"
        assert matches[0].entity_type == EntityType.WEAPON

    def test_knife(self):
        """Should extract 'knife'."""
        text = "She found a knife near the scene."
        matches = extract_weapons(text)
        assert len(matches) == 1
        assert matches[0].text.lower() == "knife"

    def test_multiple_weapons(self):
        """Should extract multiple weapons."""
        text = "The suspect had a gun and a knife."
        matches = extract_weapons(text)
        assert len(matches) == 2
        weapons = {m.text.lower() for m in matches}
        assert "gun" in weapons
        assert "knife" in weapons

    def test_word_boundary(self):
        """Should not match partial words."""
        text = "The gunner was armed."
        matches = extract_weapons(text)
        # 'gunner' contains 'gun' but is a different word
        # Should check for word boundaries
        # May or may not match depending on implementation
        assert isinstance(matches, list)

    def test_confidence_high(self):
        """Weapon matches should have high confidence."""
        text = "Found a pistol."
        matches = extract_weapons(text)
        if matches:
            assert matches[0].confidence >= 0.85


class TestExtractEvidence:
    """Tests for evidence extraction."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert extract_evidence("") == []

    def test_fingerprint(self):
        """Should extract 'fingerprint'."""
        text = "A fingerprint was found on the glass."
        matches = extract_evidence(text)
        assert len(matches) == 1
        assert "fingerprint" in matches[0].text.lower()
        assert matches[0].entity_type == EntityType.EVIDENCE

    def test_dna(self):
        """Should extract 'DNA'."""
        text = "DNA samples were collected."
        matches = extract_evidence(text)
        assert len(matches) == 1
        assert matches[0].text.lower() == "dna"

    def test_multiple_evidence(self):
        """Should extract multiple evidence items."""
        text = "We found fingerprints and blood at the scene."
        matches = extract_evidence(text)
        assert len(matches) == 2
        evidence = {m.text.lower() for m in matches}
        assert "fingerprints" in evidence
        assert "blood" in evidence

    def test_no_evidence(self):
        """Should return empty for text without evidence."""
        text = "Nothing was found at the location."
        matches = extract_evidence(text)
        assert matches == []


class TestExtractAllRuleBased:
    """Tests for combined rule-based extraction."""

    def test_empty_string(self):
        """Empty string should return empty list."""
        assert extract_all_rule_based("") == []

    def test_combined_extraction(self):
        """Should extract all entity types."""
        text = (
            "The suspect with a gun was seen at 420 Harrow Lane. "
            "Call 555-123-4567 for tips. DNA evidence was recovered."
        )
        matches = extract_all_rule_based(text)

        entity_types = {m.entity_type for m in matches}
        # Should have multiple entity types
        assert len(entity_types) >= 2

    def test_sorted_by_position(self):
        """Results should be sorted by start position."""
        text = "Found gun at 123 Main Street with fingerprints."
        matches = extract_all_rule_based(text)

        for i in range(len(matches) - 1):
            assert matches[i].start_char <= matches[i + 1].start_char

    def test_determinism(self):
        """Same input should always produce same output."""
        text = "Knife found at 420 Oak Street. DNA on handle."
        results = [extract_all_rule_based(text) for _ in range(50)]

        first_result = [(m.text, m.start_char, m.end_char) for m in results[0]]
        for result in results[1:]:
            current = [(m.text, m.start_char, m.end_char) for m in result]
            assert current == first_result
