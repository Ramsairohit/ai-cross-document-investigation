"""
Stage 10: Contradiction Detection - Rule-Based Detection

Deterministic, explainable rule-based contradiction checks.

IMPORTANT:
- Rules must be deterministic
- Rules must be explainable
- Rules must be testable
- NO inference beyond text
"""

import re
from typing import Any, Union

from .models import ChunkReference, ContradictionType


# Denial patterns
DENIAL_PATTERNS = [
    r"\b(?:was\s+not|wasn't|were\s+not|weren't)\b",
    r"\b(?:did\s+not|didn't)\b",
    r"\b(?:never|not)\b",
    r"\b(?:deny|denied|denies)\b",
    r"\b(?:no\s+way|impossible)\b",
]

# Assertion patterns (positive statements)
ASSERTION_PATTERNS = [
    r"\b(?:i\s+was|he\s+was|she\s+was|they\s+were)\b",
    r"\b(?:i\s+did|he\s+did|she\s+did|they\s+did)\b",
    r"\b(?:i\s+saw|he\s+saw|she\s+saw|they\s+saw)\b",
    r"\b(?:i\s+have|he\s+has|she\s+has|they\s+have)\b",
]

# Location indicators
LOCATION_PATTERNS = [
    r"\b(?:at\s+(?:the\s+)?(?:home|house|office|work|scene|park|store))\b",
    r"\b(?:in\s+(?:the\s+)?(?:house|room|car|building|street))\b",
    r"\b(?:near|at|inside|outside)\s+\w+",
]

# Time patterns
TIME_PATTERNS = [
    r"\b(?:at\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)\b",
    r"\b(?:around|approximately|about)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)?)\b",
]


def extract_locations(text: str) -> list[str]:
    """
    Extract location mentions from text.

    Args:
        text: Text to analyze.

    Returns:
        List of location strings found.
    """
    locations = []
    text_lower = text.lower()

    for pattern in LOCATION_PATTERNS:
        matches = re.findall(pattern, text_lower)
        locations.extend(matches)

    return locations


def extract_times(text: str) -> list[str]:
    """
    Extract time mentions from text.

    Args:
        text: Text to analyze.

    Returns:
        List of time strings found.
    """
    times = []

    for pattern in TIME_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                times.extend([m for m in match if m])
            else:
                times.append(match)

    return times


def has_denial(text: str) -> bool:
    """Check if text contains denial patterns."""
    text_lower = text.lower()
    for pattern in DENIAL_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def has_assertion(text: str) -> bool:
    """Check if text contains positive assertion patterns."""
    text_lower = text.lower()
    for pattern in ASSERTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False


def detect_denial_vs_assertion(
    chunk_a: Union[dict[str, Any], Any],
    chunk_b: Union[dict[str, Any], Any],
    shared_entities: list[str],
) -> tuple[bool, str]:
    """
    Detect if one chunk denies what another asserts.

    Args:
        chunk_a: First chunk.
        chunk_b: Second chunk.
        shared_entities: Entities shared between chunks.

    Returns:
        Tuple of (is_contradiction, explanation).
    """
    if isinstance(chunk_a, dict):
        text_a = chunk_a.get("text", "")
    else:
        text_a = chunk_a.text

    if isinstance(chunk_b, dict):
        text_b = chunk_b.get("text", "")
    else:
        text_b = chunk_b.text

    # Check for denial in one and assertion in another
    a_has_denial = has_denial(text_a)
    b_has_denial = has_denial(text_b)
    a_has_assertion = has_assertion(text_a)
    b_has_assertion = has_assertion(text_b)

    if a_has_denial and b_has_assertion:
        # Check if they're about the same entity
        for entity in shared_entities:
            entity_lower = entity.lower().split()[0]
            if entity_lower in text_a.lower() and entity_lower in text_b.lower():
                return True, f"Chunk A denies while Chunk B asserts about {entity}."

    if b_has_denial and a_has_assertion:
        for entity in shared_entities:
            entity_lower = entity.lower().split()[0]
            if entity_lower in text_a.lower() and entity_lower in text_b.lower():
                return True, f"Chunk B denies while Chunk A asserts about {entity}."

    return False, ""


def detect_location_conflict(
    chunk_a: Union[dict[str, Any], Any],
    chunk_b: Union[dict[str, Any], Any],
    shared_entities: list[str],
    timestamp: str | None = None,
) -> tuple[bool, str]:
    """
    Detect if same entity is claimed to be at different locations.

    Args:
        chunk_a: First chunk.
        chunk_b: Second chunk.
        shared_entities: Entities shared between chunks.
        timestamp: Shared timestamp if applicable.

    Returns:
        Tuple of (is_contradiction, explanation).
    """
    if isinstance(chunk_a, dict):
        text_a = chunk_a.get("text", "")
    else:
        text_a = chunk_a.text

    if isinstance(chunk_b, dict):
        text_b = chunk_b.get("text", "")
    else:
        text_b = chunk_b.text

    locations_a = extract_locations(text_a)
    locations_b = extract_locations(text_b)

    if not locations_a or not locations_b:
        return False, ""

    # Check if different locations are mentioned
    locations_a_set = set(loc.strip() for loc in locations_a)
    locations_b_set = set(loc.strip() for loc in locations_b)

    if locations_a_set and locations_b_set and not (locations_a_set & locations_b_set):
        # Different locations - check if same entity
        for entity in shared_entities:
            entity_lower = entity.lower().split()[0]
            if entity_lower in text_a.lower() and entity_lower in text_b.lower():
                loc_a = list(locations_a_set)[0]
                loc_b = list(locations_b_set)[0]
                time_note = f" at {timestamp}" if timestamp else ""
                return True, f"{entity} claimed {loc_a} vs {loc_b}{time_note}."

    return False, ""


def detect_time_conflict(
    chunk_a: Union[dict[str, Any], Any],
    chunk_b: Union[dict[str, Any], Any],
    shared_entities: list[str],
) -> tuple[bool, str]:
    """
    Detect if same entity has incompatible time claims.

    Args:
        chunk_a: First chunk.
        chunk_b: Second chunk.
        shared_entities: Entities shared between chunks.

    Returns:
        Tuple of (is_contradiction, explanation).
    """
    if isinstance(chunk_a, dict):
        text_a = chunk_a.get("text", "")
    else:
        text_a = chunk_a.text

    if isinstance(chunk_b, dict):
        text_b = chunk_b.get("text", "")
    else:
        text_b = chunk_b.text

    times_a = extract_times(text_a)
    times_b = extract_times(text_b)

    if not times_a or not times_b:
        return False, ""

    # Normalize times for comparison
    def normalize_time(t: str) -> str:
        t = t.lower().strip()
        t = re.sub(r"\s+", "", t)
        return t

    times_a_norm = set(normalize_time(t) for t in times_a)
    times_b_norm = set(normalize_time(t) for t in times_b)

    # If same times mentioned with conflicting info (handled by location)
    # This checks for explicit time discrepancies
    if times_a_norm and times_b_norm and not (times_a_norm & times_b_norm):
        for entity in shared_entities:
            entity_lower = entity.lower().split()[0]
            if entity_lower in text_a.lower() and entity_lower in text_b.lower():
                time_a = list(times_a)[0]
                time_b = list(times_b)[0]
                return True, f"{entity} has conflicting times: {time_a} vs {time_b}."

    return False, ""


def detect_statement_vs_evidence(
    chunk_a: Union[dict[str, Any], Any],
    chunk_b: Union[dict[str, Any], Any],
    shared_entities: list[str],
) -> tuple[bool, str]:
    """
    Detect if a statement contradicts physical evidence.

    This requires one chunk to be evidence-type and another to be statement.

    Args:
        chunk_a: First chunk.
        chunk_b: Second chunk.
        shared_entities: Entities shared between chunks.

    Returns:
        Tuple of (is_contradiction, explanation).
    """
    # Evidence indicators
    evidence_keywords = [
        "forensic",
        "dna",
        "fingerprint",
        "blood",
        "evidence",
        "camera",
        "cctv",
        "footage",
        "record",
        "log",
    ]

    if isinstance(chunk_a, dict):
        text_a = chunk_a.get("text", "")
        doc_a = chunk_a.get("document_id", "")
    else:
        text_a = chunk_a.text
        doc_a = chunk_a.document_id

    if isinstance(chunk_b, dict):
        text_b = chunk_b.get("text", "")
        doc_b = chunk_b.get("document_id", "")
    else:
        text_b = chunk_b.text
        doc_b = chunk_b.document_id

    text_a_lower = text_a.lower()
    text_b_lower = text_b.lower()

    # Check if one is evidence-related
    a_is_evidence = any(kw in text_a_lower for kw in evidence_keywords)
    b_is_evidence = any(kw in text_b_lower for kw in evidence_keywords)

    if not (a_is_evidence or b_is_evidence):
        return False, ""

    if a_is_evidence == b_is_evidence:
        return False, ""  # Both or neither are evidence

    # One is evidence, one is statement - check for conflict
    evidence_text = text_a_lower if a_is_evidence else text_b_lower
    statement_text = text_b_lower if a_is_evidence else text_a_lower

    # Check for denial patterns in statement that conflict with evidence
    if has_denial(statement_text):
        for entity in shared_entities:
            entity_lower = entity.lower().split()[0]
            if entity_lower in evidence_text and entity_lower in statement_text:
                return True, f"Statement denies evidence regarding {entity}."

    return False, ""


def apply_all_rules(
    chunk_a: Union[dict[str, Any], Any],
    chunk_b: Union[dict[str, Any], Any],
    shared_entities: list[str],
    timestamp: str | None = None,
) -> list[tuple[ContradictionType, str]]:
    """
    Apply all contradiction rules to a pair.

    Args:
        chunk_a: First chunk.
        chunk_b: Second chunk.
        shared_entities: Entities shared between chunks.
        timestamp: Shared timestamp if applicable.

    Returns:
        List of (ContradictionType, explanation) for detected contradictions.
    """
    detected: list[tuple[ContradictionType, str]] = []

    # Check denial vs assertion
    is_denial, explanation = detect_denial_vs_assertion(chunk_a, chunk_b, shared_entities)
    if is_denial:
        detected.append((ContradictionType.DENIAL_VS_ASSERTION, explanation))

    # Check location conflict
    is_location, explanation = detect_location_conflict(
        chunk_a, chunk_b, shared_entities, timestamp
    )
    if is_location:
        detected.append((ContradictionType.LOCATION_CONFLICT, explanation))

    # Check time conflict
    is_time, explanation = detect_time_conflict(chunk_a, chunk_b, shared_entities)
    if is_time:
        detected.append((ContradictionType.TIME_CONFLICT, explanation))

    # Check statement vs evidence
    is_evidence, explanation = detect_statement_vs_evidence(chunk_a, chunk_b, shared_entities)
    if is_evidence:
        detected.append((ContradictionType.STATEMENT_VS_EVIDENCE, explanation))

    return detected
