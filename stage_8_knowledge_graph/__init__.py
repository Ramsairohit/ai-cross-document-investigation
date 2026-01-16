"""
Stage 8: Knowledge Graph Construction

Convert Stage 5 chunks and Stage 6 entities into a Neo4j knowledge graph
that represents factual connections with full provenance.

HARD RULES (NON-NEGOTIABLE):
❌ No guilt assignment
❌ No reasoning
❌ No prediction
❌ No missing provenance
❌ No cross-case edges
❌ No APOC procedures
❌ No graph algorithms
❌ No embeddings inside Neo4j

✔ Parameterized Cypher queries only
✔ Neo4j Community Edition compatible
✔ Deterministic graph construction
✔ Full provenance on all nodes and edges
✔ Court-auditable and explainable

The graph is DESCRIPTIVE, not interpretive.
It shows connections but does NOT tell stories or decide truth.
"""

from .edge_builder import (
    build_edges,
    create_edge_cypher,
    detect_edge_type,
    extract_edges_from_chunk,
)
from .graph_builder import (
    GraphBuilder,
    build_graph_async,
    build_graph_sync,
)
from .graph_pipeline import (
    GraphPipeline,
    process_case_async,
    process_case_sync,
)
from .models import (
    EDGE_PATTERNS,
    ENTITY_TO_NODE_TYPE,
    EdgeType,
    GraphBuildResult,
    GraphEdge,
    GraphNode,
    NodeType,
    Provenance,
)
from .neo4j_connection import (
    Neo4jConnection,
    close_neo4j,
    connect_neo4j,
    get_connection,
)
from .node_builder import (
    build_document_node,
    build_nodes,
    create_node_cypher,
    entity_to_node,
    generate_node_id,
    get_node_type,
    normalize_name,
)

__all__ = [
    # Main Pipeline API
    "GraphPipeline",
    "process_case_sync",
    "process_case_async",
    # Graph Builder
    "GraphBuilder",
    "build_graph_sync",
    "build_graph_async",
    # Models
    "NodeType",
    "EdgeType",
    "Provenance",
    "GraphNode",
    "GraphEdge",
    "GraphBuildResult",
    "ENTITY_TO_NODE_TYPE",
    "EDGE_PATTERNS",
    # Neo4j Connection
    "Neo4jConnection",
    "get_connection",
    "connect_neo4j",
    "close_neo4j",
    # Node Builder
    "normalize_name",
    "generate_node_id",
    "get_node_type",
    "entity_to_node",
    "build_nodes",
    "build_document_node",
    "create_node_cypher",
    # Edge Builder
    "detect_edge_type",
    "extract_edges_from_chunk",
    "build_edges",
    "create_edge_cypher",
]
