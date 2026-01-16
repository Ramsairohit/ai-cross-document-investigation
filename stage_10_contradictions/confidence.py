"""
Stage 10: Contradiction Detection - Confidence

Confidence score calculation for contradictions.

IMPORTANT:
- Contradiction confidence = min(chunk_a, chunk_b, nli_if_used)
- Conservative approach - lowest confidence limits the result
"""

from typing import Any, Union


def get_chunk_confidence(chunk: Union[dict[str, Any], Any]) -> float:
    """
    Extract confidence from a chunk.

    Args:
        chunk: Chunk from Stage 5.

    Returns:
        Confidence score (0.0-1.0).
    """
    if isinstance(chunk, dict):
        return chunk.get("chunk_confidence", chunk.get("confidence", 1.0))
    return getattr(chunk, "chunk_confidence", getattr(chunk, "confidence", 1.0))


def calculate_contradiction_confidence(
    chunk_a: Union[dict[str, Any], Any],
    chunk_b: Union[dict[str, Any], Any],
    nli_confidence: float | None = None,
) -> float:
    """
    Calculate contradiction confidence.

    Confidence is the MINIMUM of all contributing scores.
    This is the conservative approach for forensic use.

    Args:
        chunk_a: First chunk.
        chunk_b: Second chunk.
        nli_confidence: Optional NLI confidence score.

    Returns:
        Minimum confidence score.
    """
    conf_a = get_chunk_confidence(chunk_a)
    conf_b = get_chunk_confidence(chunk_b)

    scores = [conf_a, conf_b]

    if nli_confidence is not None:
        scores.append(nli_confidence)

    return min(scores)


def get_confidence_level(confidence: float) -> str:
    """
    Get human-readable confidence level.

    Args:
        confidence: Confidence score (0.0-1.0).

    Returns:
        Level string: "high", "medium", or "low".
    """
    if confidence >= 0.9:
        return "high"
    elif confidence >= 0.7:
        return "medium"
    else:
        return "low"


def meets_threshold(confidence: float, min_confidence: float = 0.5) -> bool:
    """
    Check if confidence meets minimum threshold.

    Args:
        confidence: Calculated confidence.
        min_confidence: Minimum required confidence.

    Returns:
        True if confidence >= min_confidence.
    """
    return confidence >= min_confidence
