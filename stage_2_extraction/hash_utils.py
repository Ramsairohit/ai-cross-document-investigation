"""
Stage 2: Document Extraction - Hash Utilities

SHA256 hashing utilities for chain-of-custody integrity verification.
These hashes ensure forensic traceability of evidence files.

IMPORTANT: Original files must remain immutable.
Hashes provide cryptographic proof of file integrity.
"""

import hashlib
from pathlib import Path


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file for chain-of-custody tracking.

    Args:
        file_path: Path to the file to hash.

    Returns:
        Hash string in format "sha256:xxxxxxxxxxxx..."

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
    """
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(65536), b""):
            sha256_hash.update(chunk)

    return f"sha256:{sha256_hash.hexdigest()}"


def compute_content_hash(content: str) -> str:
    """
    Compute SHA256 hash of extracted text content.

    Used to create a hash of the extraction output for audit logging.

    Args:
        content: The extracted text content to hash.

    Returns:
        Hash string in format "sha256:xxxxxxxxxxxx..."
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(content.encode("utf-8"))
    return f"sha256:{sha256_hash.hexdigest()}"


def compute_blocks_hash(blocks: list[dict]) -> str:
    """
    Compute SHA256 hash of content blocks for audit logging.

    Creates a deterministic hash of the extracted content blocks
    by serializing them to a canonical JSON format.

    Args:
        blocks: List of content block dictionaries.

    Returns:
        Hash string in format "sha256:xxxxxxxxxxxx..."
    """
    import json

    # Sort keys for deterministic output
    canonical_json = json.dumps(blocks, sort_keys=True, ensure_ascii=False)
    return compute_content_hash(canonical_json)


def verify_file_hash(file_path: Path, expected_hash: str) -> bool:
    """
    Verify that a file matches an expected hash.

    Used to verify chain-of-custody integrity.

    Args:
        file_path: Path to the file to verify.
        expected_hash: Expected hash in format "sha256:xxxx..."

    Returns:
        True if hashes match, False otherwise.
    """
    actual_hash = compute_file_hash(file_path)
    return actual_hash == expected_hash
