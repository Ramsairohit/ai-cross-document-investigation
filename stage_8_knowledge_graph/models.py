"""
Stage 8: Knowledge Graph - Data Models

Pydantic models and enums for forensic-grade knowledge graph construction.
These models define the exact schema for graph nodes and edges.

IMPORTANT: The graph is DESCRIPTIVE, not interpretive.
No inference, no conclusions, no causality.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class NodeType(str, Enum):
    """
    Strict enumeration of allowed node types.

    DO NOT add new types without legal review.
    """

    PERSON = "Person"
    EVIDENCE = "Evidence"
    LOCATION = "Location"
    EVENT = "Event"
    DOCUMENT = "Document"


class EdgeType(str, Enum):
    """
    Strict enumeration of allowed edge types.

    Edges represent ONLY factual connections explicitly stated in text.
    They NEVER imply guilt or causality.
    """

    WITNESSED = "WITNESSED"
    FOUND_IN = "FOUND_IN"
    OWNS = "OWNS"
    ACCOMPANIED_BY = "ACCOMPANIED_BY"
    ARGUED_WITH = "ARGUED_WITH"


class Provenance(BaseModel):
    """
    Complete provenance for legal traceability.

    Every node and edge MUST have provenance.
    Missing provenance = invalid data.
    """

    source_chunk_id: str = Field(..., description="Chunk ID where this was extracted")
    document_id: str = Field(..., description="Source document ID")
    page_range: list[int] = Field(
        ..., min_length=2, max_length=2, description="[start_page, end_page]"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from extraction")

    class Config:
        json_schema_extra = {
            "example": {
                "source_chunk_id": "CHUNK_001",
                "document_id": "DOC123",
                "page_range": [2, 2],
                "confidence": 0.92,
            }
        }


class GraphNode(BaseModel):
    """
    A single node in the knowledge graph with full provenance.

    Node ID format: {NodeType}:{name}:{case_id}
    This ensures deterministic, unique identifiers.
    """

    node_type: NodeType = Field(..., description="Type from strict enum")
    node_id: str = Field(
        ...,
        description="Unique deterministic ID: {NodeType}:{name}:{case_id}",
    )
    case_id: str = Field(..., description="Case ID for chain-of-custody")
    properties: dict[str, Any] = Field(default_factory=dict, description="Node-specific properties")
    provenance: Provenance = Field(..., description="Source provenance (MANDATORY)")

    class Config:
        json_schema_extra = {
            "example": {
                "node_type": "Person",
                "node_id": "Person:Marcus Vane:24-890-H",
                "case_id": "24-890-H",
                "properties": {"name": "Marcus Vane", "role": "WITNESS"},
                "provenance": {
                    "source_chunk_id": "CHUNK_001",
                    "document_id": "DOC123",
                    "page_range": [2, 2],
                    "confidence": 0.92,
                },
            }
        }


class GraphEdge(BaseModel):
    """
    A single edge in the knowledge graph with full provenance.

    Edges represent ONLY explicit relationships stated in source text.
    """

    edge_type: EdgeType = Field(..., description="Type from strict enum")
    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    case_id: str = Field(..., description="Case ID for chain-of-custody")
    provenance: Provenance = Field(..., description="Source provenance (MANDATORY)")

    class Config:
        json_schema_extra = {
            "example": {
                "edge_type": "ARGUED_WITH",
                "from_node": "Person:Marcus Vane:24-890-H",
                "to_node": "Person:Julian Thorne:24-890-H",
                "case_id": "24-890-H",
                "provenance": {
                    "source_chunk_id": "CHUNK_001",
                    "document_id": "DOC123",
                    "page_range": [2, 2],
                    "confidence": 0.92,
                },
            }
        }


class GraphBuildResult(BaseModel):
    """
    Complete result from graph construction.

    Contains all nodes and edges with full provenance.
    """

    case_id: str = Field(..., description="Case ID")
    nodes: list[GraphNode] = Field(default_factory=list, description="All nodes created")
    edges: list[GraphEdge] = Field(default_factory=list, description="All edges created")
    total_nodes: int = Field(default=0, description="Total node count")
    total_edges: int = Field(default=0, description="Total edge count")
    documents_processed: int = Field(default=0, description="Number of source documents")
    chunks_processed: int = Field(default=0, description="Number of chunks processed")
    entities_processed: int = Field(default=0, description="Number of entities processed")

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "24-890-H",
                "nodes": [],
                "edges": [],
                "total_nodes": 0,
                "total_edges": 0,
                "documents_processed": 0,
                "chunks_processed": 0,
                "entities_processed": 0,
            }
        }


# Mapping from Stage 6 EntityType to NodeType
ENTITY_TO_NODE_TYPE: dict[str, NodeType] = {
    "PERSON": NodeType.PERSON,
    "WITNESS": NodeType.PERSON,
    "SUSPECT": NodeType.PERSON,
    "LOCATION": NodeType.LOCATION,
    "ADDRESS": NodeType.LOCATION,
    "TIME": NodeType.EVENT,
    "EVIDENCE": NodeType.EVIDENCE,
    "WEAPON": NodeType.EVIDENCE,
    "PHONE": NodeType.EVIDENCE,
}


# Edge detection patterns (regex-based, no NLP inference)
# Each pattern maps to an EdgeType
EDGE_PATTERNS: dict[EdgeType, list[str]] = {
    EdgeType.WITNESSED: [
        r"\b(?:saw|witnessed|observed|noticed|watched)\b",
    ],
    EdgeType.FOUND_IN: [
        r"\b(?:found\s+(?:at|in|near)|located\s+(?:at|in)|discovered\s+(?:at|in))\b",
    ],
    EdgeType.OWNS: [
        r"\b(?:his|her|their|my)\s+\w+",
        r"\b(?:owned\s+by|belongs?\s+to|possession\s+of)\b",
    ],
    EdgeType.ACCOMPANIED_BY: [
        r"\b(?:with|accompanied\s+by|together\s+with|alongside)\b",
    ],
    EdgeType.ARGUED_WITH: [
        r"\b(?:argued\s+with|fought\s+with|dispute\s+with|quarreled\s+with)\b",
        r"\b(?:altercation\s+with|confrontation\s+with|argument\s+with)\b",
    ],
}
