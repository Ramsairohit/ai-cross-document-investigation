"""
Stage 11: RAG - Contradiction Checker

Checks for contradictions related to the query.

IMPORTANT:
- Adds contradictions to limitations
- NEVER resolves contradictions
- NEVER suppresses contradictions
"""

from typing import Any

from .models import RetrievedChunk


def find_related_contradictions(
    chunk_ids: list[str],
    contradictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Find contradictions involving retrieved chunks.

    Args:
        chunk_ids: Retrieved chunk IDs.
        contradictions: All contradiction objects from Stage 10.

    Returns:
        Contradictions involving any of the chunks.
    """
    related = []

    chunk_id_set = set(chunk_ids)

    for contradiction in contradictions:
        # Check if either chunk in contradiction is retrieved
        chunk_a_id = contradiction.get("chunk_a", {}).get("chunk_id", "")
        chunk_b_id = contradiction.get("chunk_b", {}).get("chunk_id", "")

        if chunk_a_id in chunk_id_set or chunk_b_id in chunk_id_set:
            related.append(contradiction)

    return related


def contradictions_to_limitations(
    contradictions: list[dict[str, Any]],
) -> list[str]:
    """
    Convert contradictions to limitation strings.

    Contradictions are REPORTED, not resolved.

    Args:
        contradictions: Related contradictions.

    Returns:
        List of limitation strings.
    """
    limitations = []

    for contradiction in contradictions:
        cont_type = contradiction.get("type", "CONFLICT")
        explanation = contradiction.get("explanation", "")
        severity = contradiction.get("severity", "")

        if explanation:
            limitations.append(f"Evidence contradiction ({severity}): {explanation}")
        else:
            limitations.append(f"Evidence contradiction detected: {cont_type}")

    return limitations


def check_contradictions(
    chunks: list[RetrievedChunk],
    contradictions: list[dict[str, Any]],
) -> list[str]:
    """
    Check for contradictions and return as limitations.

    Args:
        chunks: Retrieved chunks.
        contradictions: All contradictions from Stage 10.

    Returns:
        List of limitation strings for contradictions.
    """
    chunk_ids = [c.chunk_id for c in chunks]
    related = find_related_contradictions(chunk_ids, contradictions)
    return contradictions_to_limitations(related)


def has_critical_contradictions(
    chunk_ids: list[str],
    contradictions: list[dict[str, Any]],
) -> bool:
    """
    Check if any CRITICAL contradictions exist.

    Args:
        chunk_ids: Retrieved chunk IDs.
        contradictions: All contradictions.

    Returns:
        True if CRITICAL contradictions found.
    """
    related = find_related_contradictions(chunk_ids, contradictions)

    for cont in related:
        if cont.get("severity", "") == "CRITICAL":
            return True

    return False
