"""
Stage 10: Contradiction Detection - Pipeline

Main pipeline orchestrator for contradiction detection.

PIPELINE FLOW:
1. Chunks + Timeline + Graph → Candidate Pairing
2. Rule-Based Detection
3. Optional NLI Confirmation
4. Contradiction Objects

IMPORTANT:
- Contradictions are FLAGGED but NEVER resolved
- Full provenance preserved
- Deterministic output
"""

from typing import Any, Union

from .confidence import calculate_contradiction_confidence, meets_threshold
from .models import (
    ChunkReference,
    Contradiction,
    ContradictionConfig,
    ContradictionResult,
    ContradictionStatus,
    ContradictionType,
)
from .nli_engine import confirm_contradiction
from .pairing import (
    extract_chunk_reference,
    filter_pairs_by_timestamp,
    generate_candidate_pairs,
    get_chunk_id,
    get_chunk_text,
)
from .rules import apply_all_rules
from .severity import classify_severity


def generate_contradiction_id(case_id: str, index: int) -> str:
    """
    Generate a deterministic contradiction ID.

    Format: CONT_{case_id}_{index:04d}

    Args:
        case_id: Case identifier.
        index: Sequential contradiction index.

    Returns:
        Deterministic contradiction ID.
    """
    safe_case_id = case_id.replace("-", "_").replace(" ", "_")
    return f"CONT_{safe_case_id}_{index:04d}"


class ContradictionPipeline:
    """
    Main pipeline for contradiction detection.

    Provides a high-level API for detecting contradictions
    between chunks without resolving them.
    """

    def __init__(self, config: ContradictionConfig | None = None) -> None:
        """
        Initialize the contradiction pipeline.

        Args:
            config: Configuration for detection.
        """
        self._config = config or ContradictionConfig()

    def detect_contradictions(
        self,
        case_id: str,
        chunks: list[Union[dict[str, Any], Any]],
        entities_map: dict[str, list[str]] | None = None,
        timeline_events: list[dict[str, Any]] | None = None,
    ) -> ContradictionResult:
        """
        Detect contradictions between chunks.

        Args:
            case_id: Case identifier.
            chunks: List of chunks from Stage 5.
            entities_map: Optional map of chunk_id -> entity names.
            timeline_events: Optional timeline events from Stage 9.

        Returns:
            ContradictionResult with all detected contradictions.
        """
        # Step 1: Generate candidate pairs
        pairs = generate_candidate_pairs(
            chunks,
            entities_map,
            require_entity_overlap=self._config.require_entity_overlap,
        )

        # Step 2: Add timestamp information if available
        if timeline_events:
            pairs_with_ts = filter_pairs_by_timestamp(pairs, timeline_events)
        else:
            pairs_with_ts = [(a, b, e, None) for a, b, e in pairs]

        # Step 3: Apply rules to each pair
        contradictions: list[Contradiction] = []
        contradiction_index = 0

        for chunk_a, chunk_b, shared_entities, timestamp in pairs_with_ts:
            detected = apply_all_rules(chunk_a, chunk_b, shared_entities, timestamp)

            for contradiction_type, explanation in detected:
                # Calculate confidence
                confidence = calculate_contradiction_confidence(chunk_a, chunk_b)

                # Check minimum threshold
                if not meets_threshold(confidence, self._config.min_confidence):
                    continue

                # Optional NLI confirmation
                if self._config.use_nli:
                    text_a = get_chunk_text(chunk_a)
                    text_b = get_chunk_text(chunk_b)
                    confirmed, nli_conf = confirm_contradiction(text_a, text_b)
                    if not confirmed:
                        continue
                    confidence = min(confidence, nli_conf)

                # Classify severity
                severity = classify_severity(
                    contradiction_type,
                    confidence,
                    shared_entities,
                    has_timestamp_overlap=timestamp is not None,
                )

                # Create contradiction
                contradiction = Contradiction(
                    contradiction_id=generate_contradiction_id(case_id, contradiction_index),
                    case_id=case_id,
                    type=contradiction_type,
                    chunk_a=extract_chunk_reference(chunk_a),
                    chunk_b=extract_chunk_reference(chunk_b),
                    confidence=confidence,
                    severity=severity,
                    explanation=explanation,
                    status=ContradictionStatus.FLAGGED,
                    shared_entities=shared_entities,
                    timestamp=timestamp,
                )
                contradictions.append(contradiction)
                contradiction_index += 1

        # Build summary
        by_type = {}
        by_severity = {}
        for c in contradictions:
            type_key = c.type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            sev_key = c.severity.value
            by_severity[sev_key] = by_severity.get(sev_key, 0) + 1

        return ContradictionResult(
            case_id=case_id,
            contradictions=contradictions,
            total_contradictions=len(contradictions),
            chunks_analyzed=len(chunks),
            pairs_compared=len(pairs_with_ts),
            by_type=by_type,
            by_severity=by_severity,
        )

    def verify_determinism(
        self,
        case_id: str,
        chunks: list[Union[dict[str, Any], Any]],
        entities_map: dict[str, list[str]] | None = None,
        runs: int = 100,
    ) -> bool:
        """
        Verify that detection is deterministic.

        Args:
            case_id: Case identifier.
            chunks: List of chunks.
            entities_map: Optional entity map.
            runs: Number of runs to verify.

        Returns:
            True if all runs produce identical results.
        """
        results = []
        for _ in range(runs):
            result = self.detect_contradictions(case_id, chunks, entities_map)
            ids = [c.contradiction_id for c in result.contradictions]
            results.append((result.total_contradictions, tuple(ids)))

        first = results[0]
        for result in results[1:]:
            if result != first:
                return False
        return True


def detect_contradictions_sync(
    case_id: str,
    chunks: list[Union[dict[str, Any], Any]],
    entities_map: dict[str, list[str]] | None = None,
    timeline_events: list[dict[str, Any]] | None = None,
    config: ContradictionConfig | None = None,
) -> ContradictionResult:
    """
    Synchronous contradiction detection.

    Convenience function for detecting contradictions.

    Args:
        case_id: Case identifier.
        chunks: List of chunks from Stage 5.
        entities_map: Optional entity map.
        timeline_events: Optional timeline events.
        config: Optional configuration.

    Returns:
        ContradictionResult with detected contradictions.
    """
    pipeline = ContradictionPipeline(config)
    return pipeline.detect_contradictions(case_id, chunks, entities_map, timeline_events)


async def detect_contradictions_async(
    case_id: str,
    chunks: list[Union[dict[str, Any], Any]],
    entities_map: dict[str, list[str]] | None = None,
    timeline_events: list[dict[str, Any]] | None = None,
    config: ContradictionConfig | None = None,
) -> ContradictionResult:
    """
    Async-safe contradiction detection.

    Args:
        case_id: Case identifier.
        chunks: List of chunks from Stage 5.
        entities_map: Optional entity map.
        timeline_events: Optional timeline events.
        config: Optional configuration.

    Returns:
        ContradictionResult with detected contradictions.
    """
    pipeline = ContradictionPipeline(config)
    return pipeline.detect_contradictions(case_id, chunks, entities_map, timeline_events)
