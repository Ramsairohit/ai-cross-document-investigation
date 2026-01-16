"""
Stage 10: Contradiction Detection - NLI Engine

Restricted Natural Language Inference wrapper.

IMPORTANT:
- NLI is SECONDARY, not primary
- Only used on pre-filtered pairs (after rule-based detection)
- Labels allowed: contradiction, neutral, entailment
- NO zero-shot discovery
"""

from typing import Any


class NLIResult:
    """Result from NLI classification."""

    def __init__(
        self,
        label: str,
        confidence: float,
        is_contradiction: bool,
    ):
        self.label = label
        self.confidence = confidence
        self.is_contradiction = is_contradiction


def classify_pair(text_a: str, text_b: str) -> NLIResult:
    """
    Classify a text pair using NLI.

    This is a STUB implementation. In production, this would use
    a pre-trained NLI model (e.g., roberta-large-mnli).

    For forensic use, NLI is SECONDARY and only confirms
    pairs already flagged by rule-based detection.

    Args:
        text_a: First text (premise).
        text_b: Second text (hypothesis).

    Returns:
        NLIResult with label, confidence, and contradiction flag.
    """
    # STUB: Simple heuristic-based classification
    # In production, use a proper NLI model

    text_a_lower = text_a.lower()
    text_b_lower = text_b.lower()

    # Check for explicit contradiction indicators
    contradiction_score = 0.0

    # Denial in one vs assertion in other
    denial_words = ["not", "never", "didn't", "wasn't", "weren't", "deny"]
    has_denial_a = any(w in text_a_lower for w in denial_words)
    has_denial_b = any(w in text_b_lower for w in denial_words)

    if has_denial_a != has_denial_b:
        contradiction_score += 0.3

    # Check for opposite location claims
    if ("home" in text_a_lower and "scene" in text_b_lower) or (
        "scene" in text_a_lower and "home" in text_b_lower
    ):
        contradiction_score += 0.4

    # Check for same entity mentioned with different claims
    if contradiction_score > 0.5:
        return NLIResult(
            label="contradiction",
            confidence=min(contradiction_score, 0.95),
            is_contradiction=True,
        )
    elif contradiction_score > 0.2:
        return NLIResult(
            label="neutral",
            confidence=0.6,
            is_contradiction=False,
        )
    else:
        return NLIResult(
            label="entailment",
            confidence=0.5,
            is_contradiction=False,
        )


def confirm_contradiction(
    text_a: str,
    text_b: str,
    min_confidence: float = 0.7,
) -> tuple[bool, float]:
    """
    Use NLI to confirm a potential contradiction.

    This should ONLY be called on pairs already flagged by rules.

    Args:
        text_a: First text.
        text_b: Second text.
        min_confidence: Minimum confidence to confirm.

    Returns:
        Tuple of (is_confirmed, nli_confidence).
    """
    result = classify_pair(text_a, text_b)

    if result.is_contradiction and result.confidence >= min_confidence:
        return True, result.confidence
    else:
        return False, result.confidence


def get_nli_label(text_a: str, text_b: str) -> tuple[str, float]:
    """
    Get NLI label for a text pair.

    Labels: contradiction, neutral, entailment

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Tuple of (label, confidence).
    """
    result = classify_pair(text_a, text_b)
    return result.label, result.confidence
