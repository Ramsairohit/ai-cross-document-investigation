"""
Stage 10: Contradiction Detection

Identify explicit contradictions between evidence statements
without resolving them.

HARD RULES (NON-NEGOTIABLE):
❌ No contradiction resolution
❌ No prioritizing witnesses
❌ No "likely true" statements
❌ No inference beyond text
❌ No modification of chunks

✅ Flag only
✅ Preserve provenance
✅ Deterministic output
✅ Entity-based pairing

This stage FLAGS inconsistencies — it NEVER resolves them.
"""

from .confidence import (
    calculate_contradiction_confidence,
    get_chunk_confidence,
    get_confidence_level,
    meets_threshold,
)
from .contradiction_pipeline import (
    ContradictionPipeline,
    detect_contradictions_async,
    detect_contradictions_sync,
    generate_contradiction_id,
)
from .models import (
    ChunkReference,
    Contradiction,
    ContradictionConfig,
    ContradictionResult,
    ContradictionSeverity,
    ContradictionStatus,
    ContradictionType,
)
from .nli_engine import (
    NLIResult,
    classify_pair,
    confirm_contradiction,
    get_nli_label,
)
from .pairing import (
    chunks_share_entity,
    extract_chunk_reference,
    filter_pairs_by_timestamp,
    generate_candidate_pairs,
    get_chunk_case_id,
    get_chunk_id,
    get_chunk_speaker,
    get_chunk_text,
)
from .rules import (
    apply_all_rules,
    detect_denial_vs_assertion,
    detect_location_conflict,
    detect_statement_vs_evidence,
    detect_time_conflict,
    extract_locations,
    extract_times,
    has_assertion,
    has_denial,
)
from .severity import (
    classify_severity,
    compare_severity,
    get_base_severity,
    is_critical,
    is_high_or_critical,
    severity_to_int,
)

__all__ = [
    # Main Pipeline API
    "ContradictionPipeline",
    "detect_contradictions_sync",
    "detect_contradictions_async",
    "generate_contradiction_id",
    # Models
    "ContradictionType",
    "ContradictionSeverity",
    "ContradictionStatus",
    "ChunkReference",
    "Contradiction",
    "ContradictionResult",
    "ContradictionConfig",
    # Pairing
    "generate_candidate_pairs",
    "filter_pairs_by_timestamp",
    "chunks_share_entity",
    "extract_chunk_reference",
    "get_chunk_id",
    "get_chunk_text",
    "get_chunk_speaker",
    "get_chunk_case_id",
    # Rules
    "apply_all_rules",
    "detect_time_conflict",
    "detect_location_conflict",
    "detect_denial_vs_assertion",
    "detect_statement_vs_evidence",
    "extract_locations",
    "extract_times",
    "has_denial",
    "has_assertion",
    # NLI Engine
    "NLIResult",
    "classify_pair",
    "confirm_contradiction",
    "get_nli_label",
    # Confidence
    "calculate_contradiction_confidence",
    "get_chunk_confidence",
    "get_confidence_level",
    "meets_threshold",
    # Severity
    "classify_severity",
    "get_base_severity",
    "is_critical",
    "is_high_or_critical",
    "severity_to_int",
    "compare_severity",
]
