"""
Stage 10: Contradiction Detection - Severity

Severity classification for contradictions.

IMPORTANT:
- LOW: Weak/indirect contradiction
- MEDIUM: Clear but indirect
- HIGH: Direct inconsistency
- CRITICAL: Mutually exclusive facts
"""

from .models import ContradictionSeverity, ContradictionType


def classify_severity(
    contradiction_type: ContradictionType,
    confidence: float,
    shared_entities: list[str],
    has_timestamp_overlap: bool = False,
) -> ContradictionSeverity:
    """
    Classify the severity of a contradiction.

    Args:
        contradiction_type: Type of contradiction.
        confidence: Confidence score.
        shared_entities: Entities shared between chunks.
        has_timestamp_overlap: If chunks share a timestamp.

    Returns:
        Severity level.
    """
    # Start with base severity by type
    base_severity = get_base_severity(contradiction_type)

    # Adjust based on factors
    severity_score = {
        ContradictionSeverity.LOW: 1,
        ContradictionSeverity.MEDIUM: 2,
        ContradictionSeverity.HIGH: 3,
        ContradictionSeverity.CRITICAL: 4,
    }[base_severity]

    # Higher confidence = potentially higher severity
    if confidence >= 0.9:
        severity_score += 1
    elif confidence < 0.6:
        severity_score -= 1

    # More shared entities = stronger connection
    if len(shared_entities) >= 2:
        severity_score += 1

    # Timestamp overlap makes location conflicts critical
    if has_timestamp_overlap and contradiction_type == ContradictionType.LOCATION_CONFLICT:
        severity_score += 1

    # Clamp to valid range
    severity_score = max(1, min(4, severity_score))

    # Convert back to severity level
    return {
        1: ContradictionSeverity.LOW,
        2: ContradictionSeverity.MEDIUM,
        3: ContradictionSeverity.HIGH,
        4: ContradictionSeverity.CRITICAL,
    }[severity_score]


def get_base_severity(contradiction_type: ContradictionType) -> ContradictionSeverity:
    """
    Get base severity for a contradiction type.

    Args:
        contradiction_type: Type of contradiction.

    Returns:
        Base severity level.
    """
    base_map = {
        ContradictionType.TIME_CONFLICT: ContradictionSeverity.MEDIUM,
        ContradictionType.LOCATION_CONFLICT: ContradictionSeverity.HIGH,
        ContradictionType.STATEMENT_VS_EVIDENCE: ContradictionSeverity.CRITICAL,
        ContradictionType.DENIAL_VS_ASSERTION: ContradictionSeverity.HIGH,
    }
    return base_map.get(contradiction_type, ContradictionSeverity.MEDIUM)


def is_critical(severity: ContradictionSeverity) -> bool:
    """Check if severity is CRITICAL."""
    return severity == ContradictionSeverity.CRITICAL


def is_high_or_critical(severity: ContradictionSeverity) -> bool:
    """Check if severity is HIGH or CRITICAL."""
    return severity in (ContradictionSeverity.HIGH, ContradictionSeverity.CRITICAL)


def severity_to_int(severity: ContradictionSeverity) -> int:
    """Convert severity to integer for comparison."""
    return {
        ContradictionSeverity.LOW: 1,
        ContradictionSeverity.MEDIUM: 2,
        ContradictionSeverity.HIGH: 3,
        ContradictionSeverity.CRITICAL: 4,
    }[severity]


def compare_severity(
    a: ContradictionSeverity,
    b: ContradictionSeverity,
) -> int:
    """
    Compare two severity levels.

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b.
    """
    val_a = severity_to_int(a)
    val_b = severity_to_int(b)

    if val_a < val_b:
        return -1
    elif val_a > val_b:
        return 1
    else:
        return 0
