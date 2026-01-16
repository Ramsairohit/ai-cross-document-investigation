"""
Stage 2: Document Extraction - Confidence Scoring

Confidence calculation and status determination for extracted documents.
Scores are used to determine if documents need human review.

IMPORTANT: Confidence thresholds are legally significant.
Documents below threshold require manual verification.

Thresholds:
    >= 0.85 → ACCEPTED
    0.70-0.85 → FLAGGED_FOR_REVIEW
    < 0.70 → REJECTED
"""

from __future__ import annotations

from .models import ContentBlock, ExtractionStatus

# Confidence thresholds (non-negotiable per requirements)
THRESHOLD_ACCEPTED = 0.85
THRESHOLD_FLAGGED = 0.70

# OCR penalty factor (OCR text is generally less reliable than native text)
OCR_CONFIDENCE_PENALTY = 0.05

# Minimum confidence for OCR text
MIN_OCR_CONFIDENCE = 0.30


def calculate_block_confidence(raw_confidence: float | None, ocr_used: bool = False) -> float:
    """
    Calculate confidence score for a single content block.

    Args:
        raw_confidence: Raw confidence from Docling/Tesseract (0.0-1.0).
                       If None, defaults to 0.90 for native text, 0.70 for OCR.
        ocr_used: Whether OCR was used for this block.

    Returns:
        Adjusted confidence score (0.0-1.0).
    """
    if raw_confidence is None:
        # Default confidence based on extraction method
        confidence = 0.70 if ocr_used else 0.90
    else:
        confidence = raw_confidence

    # Apply OCR penalty if applicable
    if ocr_used:
        confidence = max(MIN_OCR_CONFIDENCE, confidence - OCR_CONFIDENCE_PENALTY)

    # Clamp to valid range
    return max(0.0, min(1.0, confidence))


def calculate_page_confidence(blocks: list[ContentBlock]) -> float:
    """
    Calculate average confidence score for a page.

    Args:
        blocks: List of content blocks from a single page.

    Returns:
        Average confidence (0.0-1.0), or 0.0 if no blocks.
    """
    if not blocks:
        return 0.0

    total_confidence = sum(block.confidence for block in blocks)
    return total_confidence / len(blocks)


def calculate_document_confidence(
    blocks: list[ContentBlock], weight_by_length: bool = True
) -> float:
    """
    Calculate overall confidence score for a document.

    Uses weighted average where longer blocks have more influence
    (more text = more opportunity for errors to be detected).

    Args:
        blocks: List of all content blocks in the document.
        weight_by_length: If True, weight by text length.

    Returns:
        Overall confidence (0.0-1.0), or 0.0 if no blocks.
    """
    if not blocks:
        return 0.0

    if weight_by_length:
        # Weight by text length (minimum weight of 1)
        total_weight = 0.0
        weighted_sum = 0.0

        for block in blocks:
            weight = max(1, len(block.text))
            weighted_sum += block.confidence * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0
    else:
        # Simple average
        return sum(b.confidence for b in blocks) / len(blocks)


def determine_status(confidence: float) -> ExtractionStatus:
    """
    Determine extraction status based on confidence score.

    Thresholds (per requirements):
        >= 0.85 → ACCEPTED
        0.70-0.85 → FLAGGED_FOR_REVIEW
        < 0.70 → REJECTED

    Args:
        confidence: Overall document confidence (0.0-1.0).

    Returns:
        ExtractionStatus enum value.
    """
    if confidence >= THRESHOLD_ACCEPTED:
        return ExtractionStatus.ACCEPTED
    elif confidence >= THRESHOLD_FLAGGED:
        return ExtractionStatus.FLAGGED_FOR_REVIEW
    else:
        return ExtractionStatus.REJECTED


def get_page_confidences(blocks: list[ContentBlock], page_count: int) -> dict[int, float]:
    """
    Calculate confidence scores for each page.

    Args:
        blocks: List of all content blocks.
        page_count: Total number of pages.

    Returns:
        Dictionary mapping page number to confidence score.
    """
    page_blocks: dict[int, list[ContentBlock]] = {}

    # Group blocks by page
    for block in blocks:
        if block.page not in page_blocks:
            page_blocks[block.page] = []
        page_blocks[block.page].append(block)

    # Calculate confidence for each page
    page_confidences: dict[int, float] = {}
    for page_num in range(1, page_count + 1):
        if page_num in page_blocks:
            page_confidences[page_num] = calculate_page_confidence(page_blocks[page_num])
        else:
            # Empty page defaults to 1.0 (no content to fail)
            page_confidences[page_num] = 1.0

    return page_confidences


def identify_low_confidence_pages(
    blocks: list[ContentBlock], page_count: int, threshold: float = THRESHOLD_FLAGGED
) -> set[int]:
    """
    Identify pages with confidence below threshold.

    Used to determine which pages may need OCR fallback or review.

    Args:
        blocks: List of all content blocks.
        page_count: Total number of pages.
        threshold: Confidence threshold for flagging.

    Returns:
        Set of page numbers with low confidence.
    """
    page_confidences = get_page_confidences(blocks, page_count)

    return {page_num for page_num, confidence in page_confidences.items() if confidence < threshold}
