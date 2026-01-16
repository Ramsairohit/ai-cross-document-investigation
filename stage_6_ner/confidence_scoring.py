"""
Stage 6: NER - Confidence Scoring

Calculate confidence scores for extracted entities.

Confidence is based on:
- Extraction source (spaCy vs rule-based)
- Chunk confidence
- Pattern specificity

IMPORTANT: Confidence scores are deterministic.
"""


def calculate_spacy_confidence(
    spacy_confidence: float | None,
    chunk_confidence: float,
) -> float:
    """
    Calculate confidence for a spaCy-extracted entity.

    Args:
        spacy_confidence: Raw confidence from spaCy (may be None).
        chunk_confidence: Confidence of the source chunk.

    Returns:
        Combined confidence score (0.0-1.0).
    """
    # spaCy doesn't always provide confidence, use default
    base_confidence = spacy_confidence if spacy_confidence is not None else 0.85

    # Combine with chunk confidence
    combined = base_confidence * chunk_confidence

    # Ensure bounds
    return max(0.0, min(1.0, combined))


def calculate_rule_confidence(
    rule_confidence: float,
    chunk_confidence: float,
) -> float:
    """
    Calculate confidence for a rule-based entity.

    Args:
        rule_confidence: Base confidence from the rule.
        chunk_confidence: Confidence of the source chunk.

    Returns:
        Combined confidence score (0.0-1.0).
    """
    # Combine with chunk confidence
    combined = rule_confidence * chunk_confidence

    # Ensure bounds
    return max(0.0, min(1.0, combined))


def get_default_confidence_by_type(entity_type: str) -> float:
    """
    Get default confidence for an entity type.

    Used when no specific confidence is available.

    Args:
        entity_type: The entity type string.

    Returns:
        Default confidence score.
    """
    # Higher confidence for well-defined types
    high_confidence_types = {"PHONE", "ADDRESS", "TIME"}
    medium_confidence_types = {"PERSON", "LOCATION", "WEAPON", "EVIDENCE"}

    if entity_type in high_confidence_types:
        return 0.90
    elif entity_type in medium_confidence_types:
        return 0.85
    else:
        return 0.80
