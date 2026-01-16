"""
Stage 5: Logical Chunking - Tokenizer

Deterministic token counting using tiktoken.

IMPORTANT:
- Uses cl100k_base encoding (GPT-4 tokenizer)
- Thread-safe and stateless
- Same input ALWAYS produces same token count
"""

import tiktoken

# Global encoding instance for performance
# tiktoken encodings are thread-safe and stateless
_ENCODING_CACHE: dict[str, tiktoken.Encoding] = {}


def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """
    Get or create a tiktoken encoding.

    Uses a module-level cache for efficiency.
    Thread-safe: tiktoken encodings are stateless.

    Args:
        encoding_name: Name of the tiktoken encoding.

    Returns:
        Tiktoken encoding instance.
    """
    if encoding_name not in _ENCODING_CACHE:
        _ENCODING_CACHE[encoding_name] = tiktoken.get_encoding(encoding_name)
    return _ENCODING_CACHE[encoding_name]


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.

    DETERMINISTIC: Same text always produces same count.

    Args:
        text: Text to count tokens for.
        encoding_name: Tiktoken encoding name.

    Returns:
        Exact token count.
    """
    if not text:
        return 0
    encoding = get_encoding(encoding_name)
    return len(encoding.encode(text))


def count_tokens_batch(texts: list[str], encoding_name: str = "cl100k_base") -> list[int]:
    """
    Count tokens for multiple texts efficiently.

    Args:
        texts: List of texts to count tokens for.
        encoding_name: Tiktoken encoding name.

    Returns:
        List of token counts, one per input text.
    """
    encoding = get_encoding(encoding_name)
    return [len(encoding.encode(text)) if text else 0 for text in texts]


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """
    Split text into chunks of at most max_tokens.

    DETERMINISTIC: Same input always produces same splits.
    Splits on token boundaries to preserve meaning.

    Args:
        text: Text to split.
        max_tokens: Maximum tokens per chunk.
        encoding_name: Tiktoken encoding name.

    Returns:
        List of text chunks, each with at most max_tokens.
    """
    if not text:
        return []

    encoding = get_encoding(encoding_name)
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return [text]

    chunks: list[str] = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks
