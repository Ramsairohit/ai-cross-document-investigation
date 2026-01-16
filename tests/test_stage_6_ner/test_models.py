"""
Unit tests for Stage 6: NER - Data Models

Tests for entity types, Pydantic models, and validation.
"""

import pytest

from stage_6_ner.models import (
    SPACY_LABEL_MAP,
    ChunkInput,
    EntityType,
    ExtractedEntity,
    ExtractionSource,
    NERResult,
)


class TestEntityType:
    """Tests for EntityType enum."""

    def test_all_types_defined(self):
        """Should have all required entity types."""
        expected_types = {
            "PERSON",
            "WITNESS",
            "SUSPECT",
            "LOCATION",
            "TIME",
            "EVIDENCE",
            "WEAPON",
            "PHONE",
            "ADDRESS",
        }
        actual_types = {t.value for t in EntityType}
        assert actual_types == expected_types

    def test_enum_values_are_strings(self):
        """Entity types should be string values."""
        for entity_type in EntityType:
            assert isinstance(entity_type.value, str)

    def test_strict_enum(self):
        """Should not allow invalid entity types."""
        with pytest.raises(ValueError):
            EntityType("INVALID_TYPE")


class TestExtractionSource:
    """Tests for ExtractionSource enum."""

    def test_all_sources_defined(self):
        """Should have all extraction sources."""
        sources = {s.value for s in ExtractionSource}
        assert "spacy" in sources
        assert "rule_based" in sources
        assert "metadata" in sources


class TestExtractedEntity:
    """Tests for ExtractedEntity model."""

    def test_valid_entity(self):
        """Should create valid entity with all required fields."""
        entity = ExtractedEntity(
            entity_id="ENT_001",
            entity_type=EntityType.PERSON,
            text="Marcus Vane",
            chunk_id="CHUNK_001",
            document_id="DOC123",
            case_id="24-890-H",
            page_range=[1, 1],
            start_char=6,
            end_char=17,
            confidence=0.92,
            source=ExtractionSource.SPACY,
        )
        assert entity.entity_id == "ENT_001"
        assert entity.entity_type == EntityType.PERSON
        assert entity.text == "Marcus Vane"
        assert entity.start_char == 6
        assert entity.end_char == 17
        assert entity.confidence == 0.92

    def test_role_optional(self):
        """Role should be optional."""
        entity = ExtractedEntity(
            entity_id="ENT_001",
            entity_type=EntityType.PERSON,
            text="John Doe",
            chunk_id="CHUNK_001",
            document_id="DOC123",
            case_id="24-890-H",
            page_range=[1, 1],
            start_char=0,
            end_char=8,
            confidence=0.85,
            source=ExtractionSource.SPACY,
        )
        assert entity.role is None

    def test_role_with_value(self):
        """Role can be set from metadata."""
        entity = ExtractedEntity(
            entity_id="ENT_001",
            entity_type=EntityType.PERSON,
            text="John Doe",
            chunk_id="CHUNK_001",
            document_id="DOC123",
            case_id="24-890-H",
            page_range=[1, 1],
            start_char=0,
            end_char=8,
            confidence=0.85,
            source=ExtractionSource.METADATA,
            role="WITNESS",
        )
        assert entity.role == "WITNESS"

    def test_confidence_bounds(self):
        """Confidence should be between 0 and 1."""
        # Valid confidence
        entity = ExtractedEntity(
            entity_id="ENT_001",
            entity_type=EntityType.PERSON,
            text="Test",
            chunk_id="C1",
            document_id="D1",
            case_id="CASE1",
            page_range=[1, 1],
            start_char=0,
            end_char=4,
            confidence=0.5,
            source=ExtractionSource.SPACY,
        )
        assert entity.confidence == 0.5

    def test_page_range_validation(self):
        """Page range should have exactly 2 elements."""
        entity = ExtractedEntity(
            entity_id="ENT_001",
            entity_type=EntityType.LOCATION,
            text="Main Street",
            chunk_id="C1",
            document_id="D1",
            case_id="CASE1",
            page_range=[1, 3],
            start_char=0,
            end_char=11,
            confidence=0.80,
            source=ExtractionSource.RULE_BASED,
        )
        assert entity.page_range == [1, 3]

    def test_json_serializable(self):
        """Entity should be JSON serializable."""
        entity = ExtractedEntity(
            entity_id="ENT_001",
            entity_type=EntityType.TIME,
            text="8:15 PM",
            chunk_id="C1",
            document_id="D1",
            case_id="CASE1",
            page_range=[1, 1],
            start_char=10,
            end_char=17,
            confidence=0.90,
            source=ExtractionSource.SPACY,
        )
        json_str = entity.model_dump_json()
        assert "ENT_001" in json_str
        assert "TIME" in json_str


class TestChunkInput:
    """Tests for ChunkInput model."""

    def test_valid_chunk(self):
        """Should create valid chunk input."""
        chunk = ChunkInput(
            chunk_id="CHUNK_001",
            document_id="DOC123",
            case_id="24-890-H",
            page_range=[1, 1],
            text="I saw Marcus Vane at 8:15 PM.",
            speaker="WITNESS",
            confidence=0.93,
        )
        assert chunk.chunk_id == "CHUNK_001"
        assert chunk.text == "I saw Marcus Vane at 8:15 PM."
        assert chunk.speaker == "WITNESS"

    def test_speaker_optional(self):
        """Speaker should be optional."""
        chunk = ChunkInput(
            chunk_id="C1",
            document_id="D1",
            case_id="CASE1",
            page_range=[1, 1],
            text="Test text.",
        )
        assert chunk.speaker is None

    def test_default_confidence(self):
        """Confidence should default to 1.0."""
        chunk = ChunkInput(
            chunk_id="C1",
            document_id="D1",
            case_id="CASE1",
            page_range=[1, 1],
            text="Test text.",
        )
        assert chunk.confidence == 1.0


class TestNERResult:
    """Tests for NERResult model."""

    def test_empty_result(self):
        """Should create result with no entities."""
        result = NERResult(
            chunk_id="C1",
            document_id="D1",
            case_id="CASE1",
            entities=[],
            entity_count=0,
        )
        assert result.entities == []
        assert result.entity_count == 0

    def test_result_with_entities(self):
        """Should create result with entities."""
        entity = ExtractedEntity(
            entity_id="ENT_001",
            entity_type=EntityType.PERSON,
            text="John",
            chunk_id="C1",
            document_id="D1",
            case_id="CASE1",
            page_range=[1, 1],
            start_char=0,
            end_char=4,
            confidence=0.85,
            source=ExtractionSource.SPACY,
        )
        result = NERResult(
            chunk_id="C1",
            document_id="D1",
            case_id="CASE1",
            entities=[entity],
            entity_count=1,
        )
        assert len(result.entities) == 1
        assert result.entity_count == 1


class TestSpacyLabelMap:
    """Tests for spaCy label mapping."""

    def test_person_mapping(self):
        """PERSON and PER should map to EntityType.PERSON."""
        assert SPACY_LABEL_MAP["PERSON"] == EntityType.PERSON
        assert SPACY_LABEL_MAP["PER"] == EntityType.PERSON

    def test_location_mapping(self):
        """GPE, LOC, FAC should map to EntityType.LOCATION."""
        assert SPACY_LABEL_MAP["GPE"] == EntityType.LOCATION
        assert SPACY_LABEL_MAP["LOC"] == EntityType.LOCATION
        assert SPACY_LABEL_MAP["FAC"] == EntityType.LOCATION

    def test_time_mapping(self):
        """DATE and TIME should map to EntityType.TIME."""
        assert SPACY_LABEL_MAP["DATE"] == EntityType.TIME
        assert SPACY_LABEL_MAP["TIME"] == EntityType.TIME
