"""
Stage 4: Semantic Cleaning - Noise Removal

Remove non-semantic noise from text (OCR artifacts, page artifacts).

IMPORTANT:
- NO word removal
- NO phrase rewriting
- Only removes clear non-semantic artifacts
- Deterministic: same input → same output
"""

import re

# OCR artifact patterns - single characters that are typically OCR errors
# when appearing in isolation (not part of words)
OCR_ARTIFACT_CHARS = {
    "|",  # Vertical bar (common OCR artifact)
    "~",  # Tilde (when isolated)
    "¬",  # Not sign (OCR artifact)
    "¦",  # Broken bar
    "§",  # Section sign (when isolated, not in legal context)
    "¶",  # Pilcrow/paragraph sign (when isolated)
    "°",  # Degree sign (when isolated, not with numbers)
}

# Pattern for isolated OCR artifacts (not part of words)
# Matches artifact characters surrounded by whitespace or at start/end
ISOLATED_ARTIFACT_PATTERN = re.compile(
    r"(?:^|(?<=\s))[|~¬¦]+(?=\s|$)",
    re.MULTILINE,
)

# Pattern for repeated punctuation that's likely OCR noise
# e.g., "....." or "-----" (more than 4 consecutive)
REPEATED_PUNCT_PATTERN = re.compile(r"([.…\-_=])\1{4,}")

# Pattern for garbled character sequences (random bytes that became text)
# Matches sequences of 3+ uncommon punctuation in a row
GARBLED_CHARS_PATTERN = re.compile(r"[^\w\s,.!?;:\'\"()\[\]{}<>@#$%&*+=/\\-]{3,}")

# Pattern for form feed and other page break characters
PAGE_BREAK_PATTERN = re.compile(r"[\f\v]+")


def remove_ocr_artifacts(text: str) -> str:
    """
    Remove common OCR artifact characters when they appear in isolation.

    Only removes characters that are clearly not part of words.
    Preserves these characters when they might be meaningful
    (e.g., "|" in a table context is preserved if within word boundaries).

    Args:
        text: Input text

    Returns:
        Text with isolated OCR artifacts removed
    """
    if not text:
        return ""

    # Remove isolated artifact characters
    text = ISOLATED_ARTIFACT_PATTERN.sub("", text)

    return text


def remove_repeated_punctuation(text: str) -> str:
    """
    Replace excessive repeated punctuation with a reasonable amount.

    Reduces sequences like "....." to "..." and "-----" to "---".
    This handles OCR errors that duplicate characters.

    Args:
        text: Input text

    Returns:
        Text with reduced repeated punctuation
    """
    if not text:
        return ""

    # Replace 5+ repeated punctuation with 3
    def replace_repeated(match: re.Match[str]) -> str:
        char = match.group(1)
        return char * 3

    return REPEATED_PUNCT_PATTERN.sub(replace_repeated, text)


def remove_garbled_sequences(text: str) -> str:
    """
    Remove sequences of garbled characters (likely encoding errors).

    Only removes sequences of 3+ unusual punctuation characters
    that are clearly not meaningful text.

    Args:
        text: Input text

    Returns:
        Text with garbled sequences removed
    """
    if not text:
        return ""

    return GARBLED_CHARS_PATTERN.sub("", text)


def normalize_page_breaks(text: str) -> str:
    """
    Normalize page break characters to newlines.

    Converts form feeds (\\f) and vertical tabs (\\v) to newlines.

    Args:
        text: Input text

    Returns:
        Text with normalized page breaks
    """
    if not text:
        return ""

    return PAGE_BREAK_PATTERN.sub("\n", text)


def remove_noise(text: str, aggressive: bool = False) -> str:
    """
    Apply all noise removal operations in the correct order.

    This is the main entry point for noise removal.

    Args:
        text: Input text to clean
        aggressive: If True, also remove garbled sequences.
                   Default is False to be conservative.

    Returns:
        Noise-cleaned text

    Note:
        This function is intentionally conservative. It only removes
        characters that are very clearly non-semantic noise.
        When in doubt, it preserves the original.
    """
    if not text:
        return ""

    # Order of operations:
    # 1. Normalize page breaks first
    text = normalize_page_breaks(text)

    # 2. Remove clearly isolated OCR artifacts
    text = remove_ocr_artifacts(text)

    # 3. Reduce excessive repeated punctuation
    text = remove_repeated_punctuation(text)

    # 4. Optionally remove garbled sequences (more aggressive)
    if aggressive:
        text = remove_garbled_sequences(text)

    return text
