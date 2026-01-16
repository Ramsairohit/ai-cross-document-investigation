"""
Stage 4: Semantic Cleaning - Whitespace Normalization

Normalize whitespace while preserving sentence structure.

IMPORTANT:
- Preserves sentence order exactly
- Preserves punctuation exactly
- Deterministic: same input → same output
"""

import re

# Regex patterns for whitespace normalization
# These are compiled once for performance

# Multiple spaces (but not newlines)
MULTIPLE_SPACES_PATTERN = re.compile(r"[ \t]+")

# Multiple newlines (3+ becomes 2, preserving paragraph breaks)
MULTIPLE_NEWLINES_PATTERN = re.compile(r"\n{3,}")

# Spaces around newlines
SPACE_AROUND_NEWLINE_PATTERN = re.compile(r"[ \t]*\n[ \t]*")


def normalize_newlines(text: str) -> str:
    """
    Normalize different newline formats to Unix-style (LF).

    Converts:
    - Windows (CRLF: \\r\\n) → LF (\\n)
    - Old Mac (CR: \\r) → LF (\\n)

    Args:
        text: Input text

    Returns:
        Text with normalized newlines
    """
    if not text:
        return ""

    # Order matters: CRLF first, then standalone CR
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    return text


def collapse_multiple_spaces(text: str) -> str:
    """
    Collapse multiple consecutive spaces/tabs into a single space.

    Does NOT affect newlines - only horizontal whitespace.

    Args:
        text: Input text

    Returns:
        Text with collapsed spaces
    """
    if not text:
        return ""

    return MULTIPLE_SPACES_PATTERN.sub(" ", text)


def collapse_multiple_newlines(text: str) -> str:
    """
    Collapse 3+ consecutive newlines into exactly 2.

    This preserves paragraph structure while removing excessive blank lines.
    A single blank line (2 newlines) is preserved as a paragraph break.

    Args:
        text: Input text

    Returns:
        Text with collapsed newlines
    """
    if not text:
        return ""

    return MULTIPLE_NEWLINES_PATTERN.sub("\n\n", text)


def clean_space_around_newlines(text: str) -> str:
    """
    Remove trailing spaces before newlines and leading spaces after.

    Preserves the newline itself.

    Args:
        text: Input text

    Returns:
        Text with cleaned spaces around newlines
    """
    if not text:
        return ""

    return SPACE_AROUND_NEWLINE_PATTERN.sub("\n", text)


def trim_whitespace(text: str) -> str:
    """
    Remove leading and trailing whitespace from text.

    Args:
        text: Input text

    Returns:
        Trimmed text
    """
    if not text:
        return ""

    return text.strip()


def normalize_whitespace(
    text: str,
    collapse_spaces: bool = True,
    normalize_newline_format: bool = True,
    collapse_newlines: bool = True,
    trim: bool = True,
) -> str:
    """
    Apply all whitespace normalizations in the correct order.

    This is the main entry point for whitespace normalization.

    Order of operations:
    1. Normalize newline formats (CRLF → LF)
    2. Clean spaces around newlines
    3. Collapse multiple spaces
    4. Collapse multiple newlines
    5. Trim leading/trailing whitespace

    Args:
        text: Input text to normalize
        collapse_spaces: Whether to collapse multiple spaces
        normalize_newline_format: Whether to normalize newline formats
        collapse_newlines: Whether to collapse multiple newlines
        trim: Whether to trim leading/trailing whitespace

    Returns:
        Whitespace-normalized text
    """
    if not text:
        return ""

    # Apply normalizations in order
    if normalize_newline_format:
        text = normalize_newlines(text)

    # Clean spaces around newlines before other operations
    text = clean_space_around_newlines(text)

    if collapse_spaces:
        text = collapse_multiple_spaces(text)

    if collapse_newlines:
        text = collapse_multiple_newlines(text)

    if trim:
        text = trim_whitespace(text)

    return text
