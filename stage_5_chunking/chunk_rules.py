"""
Stage 5: Logical Chunking - Boundary Rules

Page and speaker boundary detection.

HARD CONSTRAINTS (NON-NEGOTIABLE):
1. Chunks NEVER cross pages
2. Chunks NEVER mix speakers
"""

from typing import Any, Sequence

from .models import BlockInput


def check_page_boundary(blocks: Sequence[BlockInput]) -> bool:
    """
    Check if all blocks are on the same page.

    Args:
        blocks: Sequence of blocks to check.

    Returns:
        True if all blocks are on the same page, False otherwise.
    """
    if not blocks:
        return True
    first_page = blocks[0].page
    return all(block.page == first_page for block in blocks)


def check_speaker_boundary(blocks: Sequence[BlockInput]) -> bool:
    """
    Check if all blocks have the same speaker.

    Args:
        blocks: Sequence of blocks to check.

    Returns:
        True if all blocks have the same speaker, False otherwise.
    """
    if not blocks:
        return True
    first_speaker = blocks[0].speaker
    return all(block.speaker == first_speaker for block in blocks)


def validate_chunk_blocks(blocks: Sequence[BlockInput]) -> tuple[bool, str]:
    """
    Validate that blocks can form a legal chunk.

    Args:
        blocks: Sequence of blocks to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not blocks:
        return False, "Cannot create chunk from empty blocks"

    if not check_page_boundary(blocks):
        pages = [b.page for b in blocks]
        return False, f"Blocks span multiple pages: {set(pages)}"

    if not check_speaker_boundary(blocks):
        speakers = [b.speaker for b in blocks]
        return False, f"Blocks have different speakers: {set(speakers)}"

    return True, ""


def group_blocks_by_boundary(
    blocks: Sequence[BlockInput],
) -> list[tuple[tuple[int, str | None], list[BlockInput]]]:
    """
    Group blocks by (page, speaker) boundary.

    Maintains input order within groups.
    Each group is guaranteed to have same page and speaker.

    Args:
        blocks: Sequence of blocks to group.

    Returns:
        List of ((page, speaker), blocks) tuples, ordered by first appearance.
    """
    if not blocks:
        return []

    groups: dict[tuple[int, str | None], list[BlockInput]] = {}
    order: list[tuple[int, str | None]] = []

    for block in blocks:
        key = (block.page, block.speaker)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(block)

    return [(key, groups[key]) for key in order]


def get_boundary_key(block: BlockInput) -> tuple[int, str | None]:
    """
    Get the boundary key for a block.

    Args:
        block: Block to get key for.

    Returns:
        Tuple of (page, speaker).
    """
    return (block.page, block.speaker)


def blocks_share_boundary(block1: BlockInput, block2: BlockInput) -> bool:
    """
    Check if two blocks share the same boundary.

    Args:
        block1: First block.
        block2: Second block.

    Returns:
        True if blocks can be in the same chunk.
    """
    return get_boundary_key(block1) == get_boundary_key(block2)


def convert_to_block_input(data: dict[str, Any] | BlockInput) -> BlockInput:
    """
    Convert dict or BlockInput to BlockInput.

    Args:
        data: Dict or BlockInput.

    Returns:
        BlockInput instance.
    """
    if isinstance(data, BlockInput):
        return data
    return BlockInput(**data)
