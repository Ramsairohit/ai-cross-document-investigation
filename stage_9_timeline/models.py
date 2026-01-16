"""
Stage 9: Timeline Reconstruction - Data Models

Pydantic models for forensic-grade timeline reconstruction.
These models define the exact schema for events, gaps, and conflicts.

IMPORTANT: This stage ORDERS evidence — it does NOT interpret it.
No inference, no timeline correction, no causal reasoning.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class GapSeverity(str, Enum):
    """
    Severity level for timeline gaps.

    MODERATE: 60-120 minutes gap
    SIGNIFICANT: > 120 minutes gap
    """

    MODERATE = "MODERATE"
    SIGNIFICANT = "SIGNIFICANT"


class TimelineEvent(BaseModel):
    """
    A single event in the timeline with full provenance.

    Every field is MANDATORY for legal traceability.
    Event description is the EXACT chunk text - no summarization.
    """

    event_id: str = Field(..., description="Unique event identifier (e.g., 'EVT_001')")
    timestamp: str = Field(..., description="ISO-8601 formatted timestamp (YYYY-MM-DDTHH:MM:SS)")
    chunk_id: str = Field(..., description="Source chunk ID for provenance")
    document_id: str = Field(..., description="Source document ID")
    page_range: list[int] = Field(
        ..., min_length=2, max_length=2, description="[start_page, end_page]"
    )
    description: str = Field(..., description="Exact chunk text - NO summarization or rewriting")
    speaker: Optional[str] = Field(default=None, description="Speaker label if known")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Minimum of chunk and timestamp confidence"
    )
    raw_timestamp: Optional[str] = Field(
        default=None, description="Original timestamp string as found in text"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "event_id": "EVT_001",
                "timestamp": "2024-03-15T20:15:00",
                "chunk_id": "CHUNK_001",
                "document_id": "DOC123",
                "page_range": [2, 2],
                "description": "I heard a loud crash around 8:15 PM.",
                "speaker": "Clara Higgins",
                "confidence": 0.92,
                "raw_timestamp": "8:15 PM",
            }
        }


class TimelineGap(BaseModel):
    """
    A detected gap in the timeline.

    Gaps are reported but NOT explained.
    This is descriptive, not interpretive.
    """

    start: str = Field(..., description="ISO-8601 timestamp of gap start")
    end: str = Field(..., description="ISO-8601 timestamp of gap end")
    duration_minutes: int = Field(..., ge=0, description="Gap duration in minutes")
    severity: GapSeverity = Field(..., description="Gap severity level")
    before_event_id: Optional[str] = Field(default=None, description="Event ID before the gap")
    after_event_id: Optional[str] = Field(default=None, description="Event ID after the gap")

    class Config:
        json_schema_extra = {
            "example": {
                "start": "2024-03-15T20:15:00",
                "end": "2024-03-15T22:00:00",
                "duration_minutes": 105,
                "severity": "SIGNIFICANT",
                "before_event_id": "EVT_001",
                "after_event_id": "EVT_002",
            }
        }


class TimelineConflict(BaseModel):
    """
    A detected conflict in the timeline.

    Conflicts are FLAGGED but NOT resolved.
    This stage does NOT decide truth.
    """

    timestamp: str = Field(..., description="Conflicting timestamp")
    conflicting_chunks: list[str] = Field(
        ..., min_length=2, description="Chunk IDs with conflicting information"
    )
    conflicting_event_ids: list[str] = Field(
        default_factory=list, description="Event IDs with conflicting information"
    )
    reason: str = Field(..., description="Reason for conflict detection")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum confidence of conflicting events",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2024-03-15T21:00:00",
                "conflicting_chunks": ["CHUNK_004", "CHUNK_007"],
                "conflicting_event_ids": ["EVT_004", "EVT_007"],
                "reason": "Multiple events at same time with different speakers",
                "confidence": 0.85,
            }
        }


class TimelineResult(BaseModel):
    """
    Complete timeline reconstruction result.

    Contains ordered events, detected gaps, and flagged conflicts.
    """

    case_id: str = Field(..., description="Case identifier")
    events: list[TimelineEvent] = Field(
        default_factory=list, description="Chronologically ordered events"
    )
    gaps: list[TimelineGap] = Field(default_factory=list, description="Detected timeline gaps")
    conflicts: list[TimelineConflict] = Field(
        default_factory=list, description="Flagged timeline conflicts"
    )
    total_events: int = Field(default=0, description="Total number of events")
    total_gaps: int = Field(default=0, description="Total number of gaps")
    total_conflicts: int = Field(default=0, description="Total number of conflicts")
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    timestamps_processed: int = Field(default=0, description="Number of timestamps processed")

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "24-890-H",
                "events": [],
                "gaps": [],
                "conflicts": [],
                "total_events": 0,
                "total_gaps": 0,
                "total_conflicts": 0,
                "chunks_processed": 0,
                "timestamps_processed": 0,
            }
        }


class TimelineConfig(BaseModel):
    """
    Configuration for timeline reconstruction behavior.

    Gap thresholds are configurable but defaults follow spec:
    - < 60 min: ignored
    - 60-120 min: MODERATE
    - > 120 min: SIGNIFICANT
    """

    gap_threshold_minutes: int = Field(
        default=60, ge=1, description="Minimum gap duration to report (minutes)"
    )
    significant_gap_minutes: int = Field(
        default=120, ge=1, description="Threshold for SIGNIFICANT severity (minutes)"
    )
    detect_speaker_conflicts: bool = Field(
        default=True, description="Flag conflicts when same time has different speakers"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "gap_threshold_minutes": 60,
                "significant_gap_minutes": 120,
                "detect_speaker_conflicts": True,
            }
        }


# Input type alias for timestamps map
# Maps chunk_id -> list of normalized timestamps
TimestampsMap = dict[str, list[dict[str, any]]]
