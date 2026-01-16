"""
Stage 9: Timeline Reconstruction - Pipeline

Main pipeline orchestrator for timeline reconstruction.

PIPELINE FLOW:
1. Chunks + Timestamps → Event Creation
2. Chronological Sort
3. Gap Detection
4. Conflict Detection
5. TimelineResult

IMPORTANT:
- Deterministic: same input → same output
- No inference, no timeline correction
- Gaps and conflicts are reported, NOT resolved
"""

from typing import Any, Union

from .conflict_detector import detect_conflicts
from .event_builder import build_events, build_events_from_blocks
from .gap_detector import detect_gaps
from .models import TimelineConfig, TimelineEvent, TimelineResult
from .sorter import sort_events


class TimelinePipeline:
    """
    Main pipeline for timeline reconstruction.

    Provides a high-level API for processing case data
    into a chronological timeline with gaps and conflicts.
    """

    def __init__(self, config: TimelineConfig | None = None) -> None:
        """
        Initialize the timeline pipeline.

        Args:
            config: Configuration for gap/conflict detection.
        """
        self._config = config or TimelineConfig()

    def build_timeline(
        self,
        case_id: str,
        chunks: list[Union[dict[str, Any], Any]],
        timestamps_map: dict[str, list[dict[str, Any]]],
    ) -> TimelineResult:
        """
        Build a complete timeline from chunks and timestamps.

        Args:
            case_id: Case identifier.
            chunks: List of chunks from Stage 5.
            timestamps_map: Map of chunk_id -> list of normalized timestamps.

        Returns:
            TimelineResult with events, gaps, and conflicts.
        """
        # Count timestamps processed
        total_timestamps = sum(len(ts) for ts in timestamps_map.values())

        # Step 1: Build events from chunks
        events = build_events(chunks, timestamps_map, case_id)

        # Step 2: Sort chronologically
        sorted_events = sort_events(events)

        # Step 3: Detect gaps
        gaps = detect_gaps(sorted_events, self._config)

        # Step 4: Detect conflicts
        conflicts = detect_conflicts(sorted_events, self._config)

        return TimelineResult(
            case_id=case_id,
            events=sorted_events,
            gaps=gaps,
            conflicts=conflicts,
            total_events=len(sorted_events),
            total_gaps=len(gaps),
            total_conflicts=len(conflicts),
            chunks_processed=len(chunks),
            timestamps_processed=total_timestamps,
        )

    def build_timeline_from_blocks(
        self,
        case_id: str,
        cleaned_blocks: list[Union[dict[str, Any], Any]],
    ) -> TimelineResult:
        """
        Build a timeline directly from Stage 4 cleaned blocks.

        Alternative entry point when chunks aren't available.

        Args:
            case_id: Case identifier.
            cleaned_blocks: List of cleaned blocks from Stage 4.

        Returns:
            TimelineResult with events, gaps, and conflicts.
        """
        # Count timestamps
        total_timestamps = 0
        for block in cleaned_blocks:
            if isinstance(block, dict):
                total_timestamps += len(block.get("normalized_timestamps", []))
            else:
                total_timestamps += len(getattr(block, "normalized_timestamps", []))

        # Step 1: Build events from blocks
        events = build_events_from_blocks(cleaned_blocks, case_id)

        # Step 2: Sort chronologically
        sorted_events = sort_events(events)

        # Step 3: Detect gaps
        gaps = detect_gaps(sorted_events, self._config)

        # Step 4: Detect conflicts
        conflicts = detect_conflicts(sorted_events, self._config)

        return TimelineResult(
            case_id=case_id,
            events=sorted_events,
            gaps=gaps,
            conflicts=conflicts,
            total_events=len(sorted_events),
            total_gaps=len(gaps),
            total_conflicts=len(conflicts),
            chunks_processed=len(cleaned_blocks),
            timestamps_processed=total_timestamps,
        )

    def verify_determinism(
        self,
        case_id: str,
        chunks: list[Union[dict[str, Any], Any]],
        timestamps_map: dict[str, list[dict[str, Any]]],
        runs: int = 100,
    ) -> bool:
        """
        Verify that timeline building is deterministic.

        Builds the timeline multiple times and verifies identical results.

        Args:
            case_id: Case identifier.
            chunks: List of chunks.
            timestamps_map: Timestamps map.
            runs: Number of times to rebuild (default 100).

        Returns:
            True if all runs produce identical results.
        """
        results: list[tuple[int, int, int, list[str], list[str]]] = []

        for _ in range(runs):
            result = self.build_timeline(case_id, chunks, timestamps_map)

            event_ids = [e.event_id for e in result.events]
            event_timestamps = [e.timestamp for e in result.events]

            results.append(
                (
                    result.total_events,
                    result.total_gaps,
                    result.total_conflicts,
                    event_ids,
                    event_timestamps,
                )
            )

        # Compare all results to first
        first = results[0]
        for i, result in enumerate(results[1:], 2):
            if result != first:
                print(f"Determinism failed at run {i}")
                return False

        return True


def build_timeline_sync(
    case_id: str,
    chunks: list[Union[dict[str, Any], Any]],
    timestamps_map: dict[str, list[dict[str, Any]]],
    config: TimelineConfig | None = None,
) -> TimelineResult:
    """
    Synchronous timeline building.

    Convenience function for building a timeline.

    Args:
        case_id: Case identifier.
        chunks: List of chunks from Stage 5.
        timestamps_map: Map of chunk_id -> normalized timestamps.
        config: Optional configuration.

    Returns:
        TimelineResult with events, gaps, and conflicts.
    """
    pipeline = TimelinePipeline(config)
    return pipeline.build_timeline(case_id, chunks, timestamps_map)


async def build_timeline_async(
    case_id: str,
    chunks: list[Union[dict[str, Any], Any]],
    timestamps_map: dict[str, list[dict[str, Any]]],
    config: TimelineConfig | None = None,
) -> TimelineResult:
    """
    Async-safe timeline building.

    The actual processing is synchronous, but this wrapper
    allows integration with async pipelines.

    Args:
        case_id: Case identifier.
        chunks: List of chunks from Stage 5.
        timestamps_map: Map of chunk_id -> normalized timestamps.
        config: Optional configuration.

    Returns:
        TimelineResult with events, gaps, and conflicts.
    """
    pipeline = TimelinePipeline(config)
    return pipeline.build_timeline(case_id, chunks, timestamps_map)
