"""
Stage 5: Logical Chunking - Confidence Aggregation

Conservative confidence scoring for legal safety.

RULE: Weakest link determines chunk confidence.
This is court-defensible: we never overstate confidence.
"""

from typing import Sequence


def aggregate_confidence(confidences: Sequence[float]) -> float:
    """
    Aggregate multiple confidence scores conservatively.

    Uses minimum for legal safety: the weakest source
    determines the chunk's overall confidence.

    Args:
        confidences: Sequence of confidence scores (0.0-1.0).

    Returns:
        Minimum confidence score, or 1.0 if empty.
    """
    if not confidences:
        return 1.0
    return min(confidences)


def validate_confidence(confidence: float) -> float:
    """
    Validate and clamp confidence to valid range.

    Args:
        confidence: Input confidence score.

    Returns:
        Confidence clamped to [0.0, 1.0].
    """
    return max(0.0, min(1.0, confidence))


def compute_chunk_confidence(block_confidences: Sequence[float]) -> float:
    """
    Compute final chunk confidence from source blocks.

    Args:
        block_confidences: Confidence scores of source blocks.

    Returns:
        Conservative (minimum) confidence for the chunk.
    """
    validated = [validate_confidence(c) for c in block_confidences]
    return aggregate_confidence(validated)
