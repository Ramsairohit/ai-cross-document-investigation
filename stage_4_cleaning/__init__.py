"""
Stage 4: Semantic Cleaning

Transform structurally parsed text from Stage 3 into a safe, consistent,
normalized representation suitable for chunking and NLP later.

This stage cleans text while:
- Preserving original meaning
- Preserving ambiguity
- Avoiding any form of interpretation

HARD RULES (NON-NEGOTIABLE):
❌ No summarization
❌ No inference
❌ No NLP
❌ No entity extraction
❌ No contradiction detection
❌ No meaning correction

✔ Deterministic transformations only
✔ Same input → same output
"""

from .encoding_fix import fix_encoding, normalize_encoding, remove_replacement_chars
from .models import CleanedBlock, CleaningConfig, CleaningResult, NormalizedTimestamp
from .noise_removal import (
    remove_garbled_sequences,
    remove_noise,
    remove_ocr_artifacts,
    remove_repeated_punctuation,
)
from .semantic_cleaner import (
    SemanticCleaner,
    clean_document,
    clean_document_sync,
)
from .timestamp_normalizer import (
    normalize_timestamp,
    normalize_timestamps,
)
from .whitespace_normalizer import (
    collapse_multiple_newlines,
    collapse_multiple_spaces,
    normalize_newlines,
    normalize_whitespace,
    trim_whitespace,
)

__all__ = [
    # Main API
    "SemanticCleaner",
    "clean_document",
    "clean_document_sync",
    # Models
    "CleaningResult",
    "CleanedBlock",
    "NormalizedTimestamp",
    "CleaningConfig",
    # Encoding
    "fix_encoding",
    "normalize_encoding",
    "remove_replacement_chars",
    # Whitespace
    "normalize_whitespace",
    "normalize_newlines",
    "collapse_multiple_spaces",
    "collapse_multiple_newlines",
    "trim_whitespace",
    # Noise removal
    "remove_noise",
    "remove_ocr_artifacts",
    "remove_repeated_punctuation",
    "remove_garbled_sequences",
    # Timestamps
    "normalize_timestamp",
    "normalize_timestamps",
]
