"""
Stage 5: Logical Chunking - Core Chunking Algorithm

Deterministic chunking with page/speaker boundaries.

ALGORITHM:
1. Group blocks by (page, speaker)
2. For each group, accumulate blocks until max_tokens
3. Emit chunk when threshold reached
4. Handle edge cases: large blocks, short pages

GUARANTEES:
- Deterministic: same input → same output
- Chunks never cross pages
- Chunks never mix speakers
- Text is exact concatenation of source blocks
"""

from typing import Any, Sequence

from .chunk_rules import convert_to_block_input, group_blocks_by_boundary
from .confidence import compute_chunk_confidence
from .models import BlockInput, Chunk, ChunkingConfig
from .tokenizer import count_tokens, split_text_by_tokens


class ChunkIdGenerator:
    """
    Deterministic chunk ID generator.

    Produces IDs in format: C-{counter:04d}
    Thread-safe when used within a single document processing.
    """

    def __init__(self, prefix: str = "C") -> None:
        """Initialize the generator."""
        self._prefix = prefix
        self._counter = 0

    def next_id(self) -> str:
        """Generate the next chunk ID."""
        self._counter += 1
        return f"{self._prefix}-{self._counter:04d}"

    def reset(self) -> None:
        """Reset the counter to 0."""
        self._counter = 0


def chunk_blocks(
    blocks: Sequence[BlockInput | dict[str, Any]],
    case_id: str,
    document_id: str,
    config: ChunkingConfig | None = None,
) -> list[Chunk]:
    """
    Core chunking algorithm.

    DETERMINISTIC: Same blocks + config → same chunks.

    Args:
        blocks: Sequence of blocks to chunk.
        case_id: Case ID for provenance.
        document_id: Document ID for provenance.
        config: Chunking configuration.

    Returns:
        List of chunks with full provenance.
    """
    if config is None:
        config = ChunkingConfig()

    # Convert all blocks to BlockInput
    typed_blocks = [convert_to_block_input(b) for b in blocks]

    if not typed_blocks:
        return []

    # Group by (page, speaker) boundary
    groups = group_blocks_by_boundary(typed_blocks)

    # Process each group
    id_gen = ChunkIdGenerator()
    all_chunks: list[Chunk] = []

    for (page, speaker), group_blocks in groups:
        chunks = _chunk_group(
            blocks=group_blocks,
            page=page,
            speaker=speaker,
            case_id=case_id,
            document_id=document_id,
            config=config,
            id_gen=id_gen,
        )
        all_chunks.extend(chunks)

    return all_chunks


def _chunk_group(
    blocks: list[BlockInput],
    page: int,
    speaker: str | None,
    case_id: str,
    document_id: str,
    config: ChunkingConfig,
    id_gen: ChunkIdGenerator,
) -> list[Chunk]:
    """
    Chunk a single boundary group.

    All blocks in group have same page and speaker.

    Args:
        blocks: Blocks in this group (same page/speaker).
        page: Page number for all blocks.
        speaker: Speaker for all blocks.
        case_id: Case ID for provenance.
        document_id: Document ID for provenance.
        config: Chunking configuration.
        id_gen: Chunk ID generator.

    Returns:
        List of chunks from this group.
    """
    chunks: list[Chunk] = []

    current_texts: list[str] = []
    current_block_ids: list[str] = []
    current_confidences: list[float] = []
    current_token_count = 0

    for block in blocks:
        block_tokens = count_tokens(block.clean_text, config.encoding_name)

        # Handle oversized single block
        if block_tokens > config.max_tokens:
            # Emit any accumulated content first
            if current_texts:
                chunk = _create_chunk(
                    chunk_id=id_gen.next_id(),
                    case_id=case_id,
                    document_id=document_id,
                    page=page,
                    speaker=speaker,
                    texts=current_texts,
                    block_ids=current_block_ids,
                    confidences=current_confidences,
                    encoding_name=config.encoding_name,
                )
                chunks.append(chunk)
                current_texts = []
                current_block_ids = []
                current_confidences = []
                current_token_count = 0

            # Split the oversized block
            split_chunks = _split_oversized_block(
                block=block,
                page=page,
                speaker=speaker,
                case_id=case_id,
                document_id=document_id,
                config=config,
                id_gen=id_gen,
            )
            chunks.extend(split_chunks)
            continue

        # Check if adding this block exceeds max_tokens
        if current_token_count + block_tokens > config.max_tokens:
            # Emit current chunk
            if current_texts:
                chunk = _create_chunk(
                    chunk_id=id_gen.next_id(),
                    case_id=case_id,
                    document_id=document_id,
                    page=page,
                    speaker=speaker,
                    texts=current_texts,
                    block_ids=current_block_ids,
                    confidences=current_confidences,
                    encoding_name=config.encoding_name,
                )
                chunks.append(chunk)
                current_texts = []
                current_block_ids = []
                current_confidences = []
                current_token_count = 0

        # Accumulate block
        current_texts.append(block.clean_text)
        current_block_ids.append(block.block_id)
        current_confidences.append(block.confidence)
        current_token_count += block_tokens

    # Emit remaining content (even if < min_tokens - short pages)
    if current_texts:
        chunk = _create_chunk(
            chunk_id=id_gen.next_id(),
            case_id=case_id,
            document_id=document_id,
            page=page,
            speaker=speaker,
            texts=current_texts,
            block_ids=current_block_ids,
            confidences=current_confidences,
            encoding_name=config.encoding_name,
        )
        chunks.append(chunk)

    return chunks


def _create_chunk(
    chunk_id: str,
    case_id: str,
    document_id: str,
    page: int,
    speaker: str | None,
    texts: list[str],
    block_ids: list[str],
    confidences: list[float],
    encoding_name: str,
) -> Chunk:
    """
    Create a chunk from accumulated content.

    Text is EXACT concatenation with single space separator.
    Token count is computed fresh for accuracy.

    Args:
        chunk_id: Unique chunk identifier.
        case_id: Case ID.
        document_id: Document ID.
        page: Page number.
        speaker: Speaker label.
        texts: List of block texts.
        block_ids: List of source block IDs.
        confidences: List of source confidences.
        encoding_name: Tiktoken encoding name.

    Returns:
        Chunk with full provenance.
    """
    # Exact concatenation with space separator
    combined_text = " ".join(texts)
    token_count = count_tokens(combined_text, encoding_name)
    chunk_confidence = compute_chunk_confidence(confidences)

    return Chunk(
        chunk_id=chunk_id,
        case_id=case_id,
        document_id=document_id,
        page_range=[page, page],  # Never cross pages
        speaker=speaker,
        text=combined_text,
        source_block_ids=block_ids,
        token_count=token_count,
        chunk_confidence=chunk_confidence,
    )


def _split_oversized_block(
    block: BlockInput,
    page: int,
    speaker: str | None,
    case_id: str,
    document_id: str,
    config: ChunkingConfig,
    id_gen: ChunkIdGenerator,
) -> list[Chunk]:
    """
    Split an oversized block into multiple chunks.

    Uses tiktoken token boundaries for deterministic splitting.

    Args:
        block: Block that exceeds max_tokens.
        page: Page number.
        speaker: Speaker label.
        case_id: Case ID.
        document_id: Document ID.
        config: Chunking configuration.
        id_gen: Chunk ID generator.

    Returns:
        List of chunks from the split block.
    """
    text_chunks = split_text_by_tokens(
        block.clean_text,
        config.max_tokens,
        config.encoding_name,
    )

    chunks: list[Chunk] = []
    for text_chunk in text_chunks:
        token_count = count_tokens(text_chunk, config.encoding_name)
        chunk = Chunk(
            chunk_id=id_gen.next_id(),
            case_id=case_id,
            document_id=document_id,
            page_range=[page, page],
            speaker=speaker,
            text=text_chunk,
            source_block_ids=[block.block_id],  # Same source for all splits
            token_count=token_count,
            chunk_confidence=block.confidence,  # Inherits block confidence
        )
        chunks.append(chunk)

    return chunks
