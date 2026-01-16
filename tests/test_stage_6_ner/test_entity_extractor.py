"""
Unit tests for Stage 6: NER - Entity Extractor

Tests for the main entity extraction logic.
"""

from stage_6_ner.entity_extractor import (
    extract_entities,
    get_role_from_speaker,
    map_spacy_label,
)
from stage_6_ner.models import EntityType, ExtractionSource


class TestMapSpacyLabel:
    """Tests for spaCy label mapping."""

    def test_person_mapped(self):
        """PERSON should map to EntityType.PERSON."""
        assert map_spacy_label("PERSON") == EntityType.PERSON

    def test_gpe_mapped_to_location(self):
        """GPE should map to LOCATION."""
        assert map_spacy_label("GPE") == EntityType.LOCATION

    def test_unknown_label_returns_none(self):
        """Unknown labels should return None."""
        assert map_spacy_label("UNKNOWN_LABEL") is None
        assert map_spacy_label("MONEY") is None
        assert map_spacy_label("QUANTITY") is None


class TestGetRoleFromSpeaker:
    """Tests for role extraction from speaker metadata."""

    def test_none_speaker(self):
        """None speaker should return None role."""
        assert get_role_from_speaker(None) is None

    def test_witness_role(self):
        """WITNESS speaker should return WITNESS role."""
        assert get_role_from_speaker("WITNESS") == "WITNESS"
        assert get_role_from_speaker("witness") == "WITNESS"
        assert get_role_from_speaker("Witness_1") == "WITNESS"

    def test_suspect_role(self):
        """SUSPECT speaker should return SUSPECT role."""
        assert get_role_from_speaker("SUSPECT") == "SUSPECT"
        assert get_role_from_speaker("suspect_a") == "SUSPECT"

    def test_officer_role(self):
        """Officer-related speakers should return OFFICER role."""
        assert get_role_from_speaker("OFFICER") == "OFFICER"
        assert get_role_from_speaker("DETECTIVE") == "OFFICER"
        assert get_role_from_speaker("DET SMITH") == "OFFICER"

    def test_unknown_speaker(self):
        """Unknown speaker should return None."""
        assert get_role_from_speaker("NARRATOR") is None
        assert get_role_from_speaker("UNKNOWN") is None


class TestExtractEntities:
    """Tests for main entity extraction."""

    def test_dict_input(self):
        """Should accept dict input."""
        chunk = {
            "chunk_id": "CHUNK_001",
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "page_range": [1, 1],
            "text": "Test text.",
            "speaker": None,
            "confidence": 0.9,
        }
        result = extract_entities(chunk)
        assert result.chunk_id == "CHUNK_001"
        assert result.document_id == "DOC123"
        assert result.case_id == "24-890-H"

    def test_extracts_phone_number(self):
        """Should extract phone numbers via rules."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "Call 555-123-4567 for information.",
            "speaker": None,
            "confidence": 1.0,
        }
        result = extract_entities(chunk)

        phone_entities = [e for e in result.entities if e.entity_type == EntityType.PHONE]
        assert len(phone_entities) >= 1

    def test_extracts_weapon(self):
        """Should extract weapons via rules."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "The suspect had a gun.",
            "speaker": None,
            "confidence": 1.0,
        }
        result = extract_entities(chunk)

        weapon_entities = [e for e in result.entities if e.entity_type == EntityType.WEAPON]
        assert len(weapon_entities) == 1
        assert weapon_entities[0].text.lower() == "gun"

    def test_extracts_address(self):
        """Should extract addresses via rules."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "Located at 420 Harrow Lane.",
            "speaker": None,
            "confidence": 1.0,
        }
        result = extract_entities(chunk)

        address_entities = [e for e in result.entities if e.entity_type == EntityType.ADDRESS]
        assert len(address_entities) == 1
        assert "420 Harrow Lane" in address_entities[0].text

    def test_provenance_complete(self):
        """All entities should have complete provenance."""
        chunk = {
            "chunk_id": "CHUNK_001",
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "page_range": [1, 2],
            "text": "Found a gun at the scene.",
            "speaker": None,
            "confidence": 0.95,
        }
        result = extract_entities(chunk)

        for entity in result.entities:
            # All provenance fields must be set
            assert entity.entity_id is not None
            assert entity.chunk_id == "CHUNK_001"
            assert entity.document_id == "DOC123"
            assert entity.case_id == "24-890-H"
            assert entity.page_range == [1, 2]
            assert entity.start_char >= 0
            assert entity.end_char > entity.start_char
            assert 0 <= entity.confidence <= 1
            assert entity.source is not None

    def test_role_from_metadata_only(self):
        """Role should only come from speaker metadata."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "I am a witness to the crime.",  # Contains 'witness' in text
            "speaker": None,  # No speaker metadata
            "confidence": 1.0,
        }
        result = extract_entities(chunk)

        # PERSON entities should NOT have role set from text
        person_entities = [e for e in result.entities if e.entity_type == EntityType.PERSON]
        for entity in person_entities:
            assert entity.role is None

    def test_role_assigned_from_speaker(self):
        """Role should be assigned from speaker metadata."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "I saw John Smith at the park.",
            "speaker": "WITNESS",
            "confidence": 1.0,
        }
        result = extract_entities(chunk)

        # PERSON entities should have WITNESS role
        person_entities = [e for e in result.entities if e.entity_type == EntityType.PERSON]
        for entity in person_entities:
            assert entity.role == "WITNESS"

    def test_entity_count_matches(self):
        """Entity count should match entities list length."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "Gun found at 123 Main Street.",
            "speaker": None,
            "confidence": 1.0,
        }
        result = extract_entities(chunk)
        assert result.entity_count == len(result.entities)

    def test_unique_entity_ids(self):
        """All entities should have unique IDs."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "Gun and knife found. Blood on both. Call 555-123-4567.",
            "speaker": None,
            "confidence": 1.0,
        }
        result = extract_entities(chunk)

        entity_ids = [e.entity_id for e in result.entities]
        assert len(entity_ids) == len(set(entity_ids))

    def test_extraction_source_set(self):
        """Each entity should have extraction source set."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "CASE1",
            "page_range": [1, 1],
            "text": "Knife found at 420 Oak Street.",
            "speaker": None,
            "confidence": 1.0,
        }
        result = extract_entities(chunk)

        for entity in result.entities:
            assert entity.source in [ExtractionSource.SPACY, ExtractionSource.RULE_BASED]


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_input_same_output(self):
        """Same input should produce same output."""
        chunk = {
            "chunk_id": "CHUNK_001",
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "page_range": [1, 1],
            "text": "Marcus Vane had a gun at 420 Harrow Lane at 8:15 PM.",
            "speaker": "WITNESS",
            "confidence": 0.93,
        }

        results = [extract_entities(chunk) for _ in range(20)]

        # Compare entity counts
        first_count = results[0].entity_count
        for result in results[1:]:
            assert result.entity_count == first_count

        # Compare entity texts and types
        first_entities = [(e.text, e.entity_type.value) for e in results[0].entities]
        for result in results[1:]:
            current = [(e.text, e.entity_type.value) for e in result.entities]
            assert current == first_entities
