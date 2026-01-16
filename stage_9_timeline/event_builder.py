"""
Stage 9: Timeline Reconstruction - Event Builder

Chunk to event conversion with full provenance.

IMPORTANT:
- One event per timestamp per chunk
- Event description = EXACT chunk text
- No summarization or rewriting
- Deterministic event ID generation
"""

from typing import Any, Union

from .models import TimelineEvent


def generate_event_id(case_id: str, index: int) -> str:
    """
    Generate a deterministic event ID.

    Format: EVT_{case_id}_{index:04d}

    Args:
        case_id: Case identifier.
        index: Sequential event index.

    Returns:
        Deterministic event ID string.
    """
    # Sanitize case_id for ID use
    safe_case_id = case_id.replace("-", "_").replace(" ", "_")
    return f"EVT_{safe_case_id}_{index:04d}"


def calculate_event_confidence(chunk_confidence: float, timestamp_confidence: float) -> float:
    """
    Calculate event confidence as minimum of chunk and timestamp confidence.

    Args:
        chunk_confidence: Confidence from the source chunk.
        timestamp_confidence: Confidence from timestamp normalization.

    Returns:
        Minimum of both confidence values.
    """
    return min(chunk_confidence, timestamp_confidence)


def chunk_to_events(
    chunk: Union[dict[str, Any], Any],
    timestamps: list[dict[str, Any]],
    event_index_start: int,
    case_id: str,
) -> list[TimelineEvent]:
    """
    Convert a chunk and its timestamps into timeline events.

    Creates ONE event per timestamp in the chunk.

    Args:
        chunk: Chunk from Stage 5 (dict or Chunk model).
        timestamps: List of normalized timestamps for this chunk.
        event_index_start: Starting index for event ID generation.
        case_id: Case identifier.

    Returns:
        List of TimelineEvent objects.
    """
    events: list[TimelineEvent] = []

    # Extract chunk properties
    if isinstance(chunk, dict):
        chunk_id = chunk.get("chunk_id", "")
        document_id = chunk.get("document_id", "")
        page_range = chunk.get("page_range", [1, 1])
        text = chunk.get("text", "")
        speaker = chunk.get("speaker")
        chunk_confidence = chunk.get("chunk_confidence", chunk.get("confidence", 1.0))
    else:
        chunk_id = chunk.chunk_id
        document_id = chunk.document_id
        page_range = chunk.page_range
        text = chunk.text
        speaker = chunk.speaker
        chunk_confidence = getattr(chunk, "chunk_confidence", getattr(chunk, "confidence", 1.0))

    # Create one event per timestamp
    for i, ts in enumerate(timestamps):
        # Extract timestamp properties
        iso_timestamp = ts.get("iso")
        if not iso_timestamp:
            continue  # Skip timestamps without valid ISO representation

        timestamp_confidence = ts.get("confidence", 1.0)
        raw_timestamp = ts.get("original", ts.get("raw", ""))

        # Calculate combined confidence
        event_confidence = calculate_event_confidence(chunk_confidence, timestamp_confidence)

        # Generate event ID
        event_id = generate_event_id(case_id, event_index_start + i)

        event = TimelineEvent(
            event_id=event_id,
            timestamp=iso_timestamp,
            chunk_id=chunk_id,
            document_id=document_id,
            page_range=page_range,
            description=text,  # EXACT chunk text, no summarization
            speaker=speaker,
            confidence=event_confidence,
            raw_timestamp=raw_timestamp,
        )
        events.append(event)

    return events


def build_events(
    chunks: list[Union[dict[str, Any], Any]],
    timestamps_map: dict[str, list[dict[str, Any]]],
    case_id: str,
) -> list[TimelineEvent]:
    """
    Build all timeline events from chunks and timestamps.

    Args:
        chunks: List of chunks from Stage 5.
        timestamps_map: Map of chunk_id -> list of normalized timestamps.
        case_id: Case identifier.

    Returns:
        List of TimelineEvent objects (unsorted).
    """
    all_events: list[TimelineEvent] = []
    event_index = 0

    for chunk in chunks:
        # Get chunk_id
        if isinstance(chunk, dict):
            chunk_id = chunk.get("chunk_id", "")
        else:
            chunk_id = chunk.chunk_id

        # Get timestamps for this chunk
        timestamps = timestamps_map.get(chunk_id, [])
        if not timestamps:
            continue

        # Convert to events
        events = chunk_to_events(chunk, timestamps, event_index, case_id)
        all_events.extend(events)
        event_index += len(events)

    return all_events


def build_events_from_blocks(
    cleaned_blocks: list[Union[dict[str, Any], Any]],
    case_id: str,
) -> list[TimelineEvent]:
    """
    Build events directly from Stage 4 CleanedBlocks.

    This alternative entry point extracts timestamps from blocks
    that already contain normalized_timestamps.

    Args:
        cleaned_blocks: List of cleaned blocks from Stage 4.
        case_id: Case identifier.

    Returns:
        List of TimelineEvent objects (unsorted).
    """
    all_events: list[TimelineEvent] = []
    event_index = 0

    for block in cleaned_blocks:
        # Extract block properties
        if isinstance(block, dict):
            block_id = block.get("block_id", "")
            page = block.get("page", 1)
            text = block.get("clean_text", "")
            speaker = block.get("speaker")
            normalized_timestamps = block.get("normalized_timestamps", [])
            # Treat each block as if it had a document_id matching block naming
            document_id = block.get("document_id", f"DOC_{block_id}")
        else:
            block_id = block.block_id
            page = block.page
            text = block.clean_text
            speaker = block.speaker
            normalized_timestamps = getattr(block, "normalized_timestamps", [])
            document_id = getattr(block, "document_id", f"DOC_{block_id}")

        if not normalized_timestamps:
            continue

        # Convert timestamps to dict format if needed
        timestamps = []
        for ts in normalized_timestamps:
            if isinstance(ts, dict):
                timestamps.append(ts)
            else:
                timestamps.append(
                    {
                        "original": ts.original,
                        "iso": ts.iso,
                        "confidence": ts.confidence,
                    }
                )

        # Create chunk-like structure for event creation
        chunk_dict = {
            "chunk_id": block_id,
            "document_id": document_id,
            "page_range": [page, page],
            "text": text,
            "speaker": speaker,
            "chunk_confidence": 1.0,  # Blocks don't have chunk confidence
        }

        events = chunk_to_events(chunk_dict, timestamps, event_index, case_id)
        all_events.extend(events)
        event_index += len(events)

    return all_events
