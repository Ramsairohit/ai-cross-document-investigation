"""
Stage 10: Contradiction Detection - Pairing Logic

Entity-based chunk pairing for contradiction detection.

IMPORTANT:
- Only compare chunks that share entities
- Same case only
- Deterministic pair ordering
- NO blind all-to-all comparison
"""

from typing import Any, Union

from .models import ChunkReference


def extract_chunk_reference(chunk: Union[dict[str, Any], Any]) -> ChunkReference:
    """
    Extract a ChunkReference from a chunk object or dict.

    Args:
        chunk: Chunk from Stage 5 (dict or model).

    Returns:
        ChunkReference with provenance.
    """
    if isinstance(chunk, dict):
        return ChunkReference(
            chunk_id=chunk.get("chunk_id", ""),
            document_id=chunk.get("document_id", ""),
            page_range=chunk.get("page_range", [1, 1]),
            speaker=chunk.get("speaker"),
            text=chunk.get("text", ""),
        )
    else:
        return ChunkReference(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            page_range=chunk.page_range,
            speaker=getattr(chunk, "speaker", None),
            text=chunk.text,
        )


def get_chunk_id(chunk: Union[dict[str, Any], Any]) -> str:
    """Extract chunk_id from chunk."""
    if isinstance(chunk, dict):
        return chunk.get("chunk_id", "")
    return chunk.chunk_id


def get_chunk_case_id(chunk: Union[dict[str, Any], Any]) -> str:
    """Extract case_id from chunk."""
    if isinstance(chunk, dict):
        return chunk.get("case_id", "")
    return getattr(chunk, "case_id", "")


def get_chunk_speaker(chunk: Union[dict[str, Any], Any]) -> str | None:
    """Extract speaker from chunk."""
    if isinstance(chunk, dict):
        return chunk.get("speaker")
    return getattr(chunk, "speaker", None)


def get_chunk_text(chunk: Union[dict[str, Any], Any]) -> str:
    """Extract text from chunk."""
    if isinstance(chunk, dict):
        return chunk.get("text", "")
    return chunk.text


def chunks_share_entity(
    chunk_a: Union[dict[str, Any], Any],
    chunk_b: Union[dict[str, Any], Any],
    entities_map: dict[str, list[str]] | None = None,
) -> tuple[bool, list[str]]:
    """
    Check if two chunks share any entities.

    Args:
        chunk_a: First chunk.
        chunk_b: Second chunk.
        entities_map: Optional map of chunk_id -> list of entity names.

    Returns:
        Tuple of (shares_entity, list of shared entities).
    """
    chunk_a_id = get_chunk_id(chunk_a)
    chunk_b_id = get_chunk_id(chunk_b)

    # If no entity map, check for speaker overlap
    if entities_map is None:
        speaker_a = get_chunk_speaker(chunk_a)
        speaker_b = get_chunk_speaker(chunk_b)

        # Check if one chunk mentions the other's speaker
        text_a = get_chunk_text(chunk_a).lower()
        text_b = get_chunk_text(chunk_b).lower()

        shared = []
        if speaker_a and speaker_a.lower() in text_b:
            shared.append(speaker_a)
        if speaker_b and speaker_b.lower() in text_a:
            shared.append(speaker_b)

        # Also check for name overlap in text
        if speaker_a and speaker_b:
            # Extract first name as simple heuristic
            name_a = speaker_a.split()[0].lower() if speaker_a else ""
            name_b = speaker_b.split()[0].lower() if speaker_b else ""

            if name_a and name_a in text_b and name_a not in [s.lower() for s in shared]:
                shared.append(speaker_a)
            if name_b and name_b in text_a and name_b not in [s.lower() for s in shared]:
                shared.append(speaker_b)

        return len(shared) > 0, list(set(shared))

    # Use entity map
    entities_a = set(entities_map.get(chunk_a_id, []))
    entities_b = set(entities_map.get(chunk_b_id, []))

    shared = list(entities_a & entities_b)
    return len(shared) > 0, shared


def generate_candidate_pairs(
    chunks: list[Union[dict[str, Any], Any]],
    entities_map: dict[str, list[str]] | None = None,
    require_entity_overlap: bool = True,
) -> list[tuple[Any, Any, list[str]]]:
    """
    Generate candidate pairs for contradiction detection.

    Only pairs chunks that:
    - Belong to same case
    - Share at least one entity (if required)

    Args:
        chunks: List of chunks from Stage 5.
        entities_map: Optional map of chunk_id -> entity names.
        require_entity_overlap: If True, only pair chunks with shared entities.

    Returns:
        List of (chunk_a, chunk_b, shared_entities) tuples.
    """
    pairs: list[tuple[Any, Any, list[str]]] = []

    # Group chunks by case
    chunks_by_case: dict[str, list[Any]] = {}
    for chunk in chunks:
        case_id = get_chunk_case_id(chunk)
        if case_id not in chunks_by_case:
            chunks_by_case[case_id] = []
        chunks_by_case[case_id].append(chunk)

    # Generate pairs within each case
    for case_id, case_chunks in chunks_by_case.items():
        n = len(case_chunks)
        for i in range(n):
            for j in range(i + 1, n):
                chunk_a = case_chunks[i]
                chunk_b = case_chunks[j]

                # Ensure deterministic ordering by chunk_id
                id_a = get_chunk_id(chunk_a)
                id_b = get_chunk_id(chunk_b)
                if id_a > id_b:
                    chunk_a, chunk_b = chunk_b, chunk_a

                if require_entity_overlap:
                    shares, shared_entities = chunks_share_entity(chunk_a, chunk_b, entities_map)
                    if shares:
                        pairs.append((chunk_a, chunk_b, shared_entities))
                else:
                    # No entity overlap required - compare all
                    _, shared_entities = chunks_share_entity(chunk_a, chunk_b, entities_map)
                    pairs.append((chunk_a, chunk_b, shared_entities))

    return pairs


def filter_pairs_by_timestamp(
    pairs: list[tuple[Any, Any, list[str]]],
    timeline_events: list[dict[str, Any]],
) -> list[tuple[Any, Any, list[str], str | None]]:
    """
    Filter pairs to those with overlapping timestamps.

    Args:
        pairs: Candidate pairs from generate_candidate_pairs.
        timeline_events: Events from Stage 9.

    Returns:
        Pairs with timestamp information added.
    """
    # Build chunk_id -> timestamp map
    chunk_timestamps: dict[str, str] = {}
    for event in timeline_events:
        if isinstance(event, dict):
            chunk_id = event.get("chunk_id", "")
            timestamp = event.get("timestamp", "")
        else:
            chunk_id = event.chunk_id
            timestamp = event.timestamp
        if chunk_id and timestamp:
            chunk_timestamps[chunk_id] = timestamp

    # Filter pairs with matching timestamps
    result: list[tuple[Any, Any, list[str], str | None]] = []
    for chunk_a, chunk_b, shared in pairs:
        id_a = get_chunk_id(chunk_a)
        id_b = get_chunk_id(chunk_b)

        ts_a = chunk_timestamps.get(id_a)
        ts_b = chunk_timestamps.get(id_b)

        # Include if both have same timestamp (potential conflict)
        if ts_a and ts_b and ts_a == ts_b:
            result.append((chunk_a, chunk_b, shared, ts_a))
        else:
            # Still include without timestamp
            result.append((chunk_a, chunk_b, shared, None))

    return result
