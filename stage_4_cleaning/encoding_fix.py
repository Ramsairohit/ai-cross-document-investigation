"""
Stage 4: Semantic Cleaning - Encoding Normalization

Normalize text encoding to UTF-8 and remove invalid characters.

IMPORTANT:
- Preserves all valid characters exactly
- NO autocorrect
- NO wording changes
- Deterministic: same input → same output
"""

import unicodedata

# Characters to always remove (control characters, null bytes, etc.)
# Preserving: tabs, newlines, carriage returns (will be normalized elsewhere)
ALLOWED_CONTROL_CHARS = {
    "\t",  # Tab
    "\n",  # Newline
    "\r",  # Carriage return
}


def normalize_encoding(text: str) -> str:
    """
    Normalize text encoding to UTF-8 and remove invalid characters.

    This function:
    1. Normalizes Unicode to NFC form (canonical composition)
    2. Removes non-printable control characters (except tabs/newlines)
    3. Removes null bytes and other invalid characters
    4. Preserves all valid Unicode characters exactly

    Args:
        text: Input text to normalize

    Returns:
        Encoding-normalized text

    Note:
        This function does NOT alter words or correct spelling.
        It only handles encoding-level issues.
    """
    if not text:
        return ""

    # Step 1: Normalize Unicode to NFC (Canonical Decomposition, then Canonical Composition)
    # This ensures consistent representation of characters like accents
    normalized = unicodedata.normalize("NFC", text)

    # Step 2: Remove invalid characters while preserving valid ones
    result_chars: list[str] = []

    for char in normalized:
        # Get the Unicode category
        category = unicodedata.category(char)

        # Remove control characters (category "Cc") except allowed ones
        if category == "Cc":
            if char in ALLOWED_CONTROL_CHARS:
                result_chars.append(char)
            # else: skip the control character
            continue

        # Remove unassigned characters (category "Cn")
        if category == "Cn":
            continue

        # Remove private use characters (category "Co")
        if category == "Co":
            continue

        # Remove surrogate characters (category "Cs")
        if category == "Cs":
            continue

        # Keep all other characters (letters, numbers, punctuation, symbols, etc.)
        result_chars.append(char)

    return "".join(result_chars)


def remove_replacement_chars(text: str) -> str:
    """
    Remove Unicode replacement characters (U+FFFD).

    These typically indicate encoding errors in the source document.
    We remove them rather than trying to guess what they should be.

    Args:
        text: Input text

    Returns:
        Text with replacement characters removed
    """
    if not text:
        return ""

    return text.replace("\ufffd", "")


def fix_encoding(text: str) -> str:
    """
    Apply all encoding fixes in the correct order.

    This is the main entry point for encoding normalization.

    Args:
        text: Input text to fix

    Returns:
        Encoding-fixed text
    """
    if not text:
        return ""

    # Order matters: normalize first, then remove replacement chars
    text = normalize_encoding(text)
    text = remove_replacement_chars(text)

    return text
