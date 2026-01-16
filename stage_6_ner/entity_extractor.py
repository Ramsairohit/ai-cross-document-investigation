"""
Stage 6: NER - Entity Extractor

Main extraction logic combining spaCy NER with rule-based extensions.

IMPORTANT:
- One chunk processed independently
- Entities always linked to chunk_id
- Same input → same output
- No inference or cross-chunk analysis
"""

import uuid
from typing import Any, Optional, Union

from .confidence_scoring import calculate_rule_confidence, calculate_spacy_confidence
from .models import (
    SPACY_LABEL_MAP,
    ChunkInput,
    EntityType,
    ExtractedEntity,
    ExtractionSource,
    NERResult,
)
from .rule_based_entities import extract_all_rule_based
from .spacy_loader import get_spacy_model


def generate_entity_id() -> str:
    """
    Generate a unique entity ID.

    Returns:
        Unique entity identifier.
    """
    return f"ENT_{uuid.uuid4().hex[:8].upper()}"


def map_spacy_label(label: str) -> Optional[EntityType]:
    """
    Map a spaCy label to our EntityType enum.

    Args:
        label: spaCy entity label.

    Returns:
        EntityType if mapped, None otherwise.
    """
    return SPACY_LABEL_MAP.get(label)


def get_role_from_speaker(speaker: Optional[str]) -> Optional[str]:
    """
    Determine role from speaker metadata.

    Roles are ONLY derived from metadata, never inferred from text.

    Args:
        speaker: Speaker label from chunk metadata.

    Returns:
        Role string if applicable, None otherwise.
    """
    if not speaker:
        return None

    speaker_upper = speaker.upper()

    if "WITNESS" in speaker_upper:
        return "WITNESS"
    elif "SUSPECT" in speaker_upper:
        return "SUSPECT"
    elif "VICTIM" in speaker_upper:
        return "VICTIM"
    elif "OFFICER" in speaker_upper or "DETECTIVE" in speaker_upper or "DET" in speaker_upper:
        return "OFFICER"

    return None


def extract_spacy_entities(
    text: str,
    chunk_id: str,
    document_id: str,
    case_id: str,
    page_range: list[int],
    chunk_confidence: float,
    speaker: Optional[str] = None,
) -> list[ExtractedEntity]:
    """
    Extract entities using spaCy NER.

    Args:
        text: Chunk text to process.
        chunk_id: Source chunk ID.
        document_id: Source document ID.
        case_id: Case ID.
        page_range: [start_page, end_page].
        chunk_confidence: Confidence of the source chunk.
        speaker: Optional speaker label for role assignment.

    Returns:
        List of extracted entities.
    """
    entities: list[ExtractedEntity] = []

    # Get spaCy model
    nlp = get_spacy_model()
    doc = nlp(text)

    # Get role from speaker metadata
    role = get_role_from_speaker(speaker)

    for ent in doc.ents:
        # Map spaCy label to our EntityType
        entity_type = map_spacy_label(ent.label_)

        if entity_type is None:
            # Skip unmapped entity types
            continue

        # Calculate confidence
        # Note: spaCy doesn't expose per-entity confidence easily
        confidence = calculate_spacy_confidence(None, chunk_confidence)

        entity = ExtractedEntity(
            entity_id=generate_entity_id(),
            entity_type=entity_type,
            text=ent.text,
            chunk_id=chunk_id,
            document_id=document_id,
            case_id=case_id,
            page_range=page_range,
            start_char=ent.start_char,
            end_char=ent.end_char,
            confidence=round(confidence, 2),
            source=ExtractionSource.SPACY,
            role=role if entity_type == EntityType.PERSON else None,
        )
        entities.append(entity)

    return entities


def extract_rule_based_entities(
    text: str,
    chunk_id: str,
    document_id: str,
    case_id: str,
    page_range: list[int],
    chunk_confidence: float,
) -> list[ExtractedEntity]:
    """
    Extract entities using rule-based patterns.

    Args:
        text: Chunk text to process.
        chunk_id: Source chunk ID.
        document_id: Source document ID.
        case_id: Case ID.
        page_range: [start_page, end_page].
        chunk_confidence: Confidence of the source chunk.

    Returns:
        List of extracted entities.
    """
    entities: list[ExtractedEntity] = []

    # Get all rule-based matches
    rule_matches = extract_all_rule_based(text)

    for match in rule_matches:
        confidence = calculate_rule_confidence(match.confidence, chunk_confidence)

        entity = ExtractedEntity(
            entity_id=generate_entity_id(),
            entity_type=match.entity_type,
            text=match.text,
            chunk_id=chunk_id,
            document_id=document_id,
            case_id=case_id,
            page_range=page_range,
            start_char=match.start_char,
            end_char=match.end_char,
            confidence=round(confidence, 2),
            source=ExtractionSource.RULE_BASED,
            role=None,
        )
        entities.append(entity)

    return entities


def merge_entities(
    spacy_entities: list[ExtractedEntity],
    rule_entities: list[ExtractedEntity],
) -> list[ExtractedEntity]:
    """
    Merge spaCy and rule-based entities, avoiding duplicates.

    Prefers rule-based entities for overlapping spans
    (more specific patterns).

    Args:
        spacy_entities: Entities from spaCy.
        rule_entities: Entities from rule-based extraction.

    Returns:
        Merged list of entities.
    """
    # Track spans covered by rule-based entities
    rule_spans: set[tuple[int, int]] = {(e.start_char, e.end_char) for e in rule_entities}

    # Filter spaCy entities that don't overlap with rule-based
    filtered_spacy: list[ExtractedEntity] = []
    for entity in spacy_entities:
        spacy_span = (entity.start_char, entity.end_char)

        # Check for overlap with any rule-based span
        has_overlap = False
        for rule_start, rule_end in rule_spans:
            # Overlap if not completely before or after
            if not (spacy_span[1] <= rule_start or spacy_span[0] >= rule_end):
                has_overlap = True
                break

        if not has_overlap:
            filtered_spacy.append(entity)

    # Combine and sort by start position
    all_entities = filtered_spacy + rule_entities
    all_entities.sort(key=lambda e: e.start_char)

    return all_entities


def extract_entities(
    chunk: Union[ChunkInput, dict[str, Any]],
) -> NERResult:
    """
    Extract all entities from a chunk.

    Combines spaCy NER with rule-based extensions.

    Args:
        chunk: ChunkInput or dict with chunk data.

    Returns:
        NERResult with all extracted entities.
    """
    # Handle dict input
    if isinstance(chunk, dict):
        chunk_id = chunk.get("chunk_id", "")
        document_id = chunk.get("document_id", "")
        case_id = chunk.get("case_id", "")
        page_range = chunk.get("page_range", [1, 1])
        text = chunk.get("text", "")
        speaker = chunk.get("speaker")
        chunk_confidence = chunk.get("confidence", 1.0)
    else:
        chunk_id = chunk.chunk_id
        document_id = chunk.document_id
        case_id = chunk.case_id
        page_range = chunk.page_range
        text = chunk.text
        speaker = chunk.speaker
        chunk_confidence = chunk.confidence

    # Extract with spaCy
    spacy_entities = extract_spacy_entities(
        text=text,
        chunk_id=chunk_id,
        document_id=document_id,
        case_id=case_id,
        page_range=page_range,
        chunk_confidence=chunk_confidence,
        speaker=speaker,
    )

    # Extract with rule-based patterns
    rule_entities = extract_rule_based_entities(
        text=text,
        chunk_id=chunk_id,
        document_id=document_id,
        case_id=case_id,
        page_range=page_range,
        chunk_confidence=chunk_confidence,
    )

    # Merge entities
    all_entities = merge_entities(spacy_entities, rule_entities)

    return NERResult(
        chunk_id=chunk_id,
        document_id=document_id,
        case_id=case_id,
        entities=all_entities,
        entity_count=len(all_entities),
    )
