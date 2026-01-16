"""
Stage 8: Knowledge Graph - Graph Builder

Main orchestrator for knowledge graph construction.

PIPELINE FLOW:
1. Entities + Chunks → Node normalization
2. Node creation (MERGE)
3. Edge creation (MERGE)
4. Provenance attachment

IMPORTANT:
- Deterministic: same input → identical graph
- Full provenance on all nodes and edges
- No inference, no causality, no conclusions
"""

from typing import Any, Optional, Union

from stage_5_chunking.models import Chunk
from stage_6_ner.models import ExtractedEntity

from .edge_builder import build_edges, create_edge_cypher
from .models import GraphBuildResult, GraphEdge, GraphNode
from .neo4j_connection import Neo4jConnection, get_connection
from .node_builder import build_document_node, build_nodes, create_node_cypher


class GraphBuilder:
    """
    Main orchestrator for knowledge graph construction.

    Builds a Neo4j graph from Stage 5 chunks and Stage 6 entities.
    All operations are deterministic and preserve full provenance.
    """

    def __init__(self, connection: Optional[Neo4jConnection] = None) -> None:
        """
        Initialize the graph builder.

        Args:
            connection: Optional Neo4j connection. Uses singleton if not provided.
        """
        self._connection = connection

    def _get_connection(self) -> Neo4jConnection:
        """Get the Neo4j connection."""
        if self._connection is not None:
            return self._connection
        return get_connection()

    def build_graph(
        self,
        case_id: str,
        chunks: list[Union[Chunk, dict[str, Any]]],
        entities: list[Union[ExtractedEntity, dict[str, Any]]],
        persist_to_neo4j: bool = True,
    ) -> GraphBuildResult:
        """
        Build the knowledge graph from chunks and entities.

        Args:
            case_id: Case identifier.
            chunks: List of chunks from Stage 5.
            entities: List of entities from Stage 6.
            persist_to_neo4j: Whether to persist to Neo4j (default True).

        Returns:
            GraphBuildResult with all nodes and edges.
        """
        # Build nodes from entities
        nodes = build_nodes(entities)

        # Build document nodes
        doc_ids_seen: set[str] = set()
        doc_nodes: list[GraphNode] = []

        for chunk in chunks:
            if isinstance(chunk, dict):
                doc_id = chunk.get("document_id", "")
                chunk_id = chunk.get("chunk_id", "")
                page_range = chunk.get("page_range", [1, 1])
                c_case_id = chunk.get("case_id", case_id)
            else:
                doc_id = chunk.document_id
                chunk_id = chunk.chunk_id
                page_range = chunk.page_range
                c_case_id = chunk.case_id

            if doc_id and doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)
                doc_node = build_document_node(
                    document_id=doc_id,
                    case_id=c_case_id,
                    chunk_id=chunk_id,
                    page_range=page_range,
                )
                doc_nodes.append(doc_node)

        all_nodes = nodes + doc_nodes

        # Build edges from chunks
        edges = build_edges(chunks, entities)

        # Persist to Neo4j if requested
        if persist_to_neo4j:
            self._persist_nodes(all_nodes)
            self._persist_edges(edges)

        # Build result
        return GraphBuildResult(
            case_id=case_id,
            nodes=all_nodes,
            edges=edges,
            total_nodes=len(all_nodes),
            total_edges=len(edges),
            documents_processed=len(doc_ids_seen),
            chunks_processed=len(chunks),
            entities_processed=len(entities),
        )

    def _persist_nodes(self, nodes: list[GraphNode]) -> None:
        """
        Persist nodes to Neo4j.

        Args:
            nodes: List of nodes to persist.
        """
        conn = self._get_connection()

        for node in nodes:
            query, params = create_node_cypher(node)
            try:
                conn.execute_write(query, params)
            except Exception as e:
                # Log error but continue
                print(f"Error persisting node {node.node_id}: {e}")

    def _persist_edges(self, edges: list[GraphEdge]) -> None:
        """
        Persist edges to Neo4j.

        Args:
            edges: List of edges to persist.
        """
        conn = self._get_connection()

        for edge in edges:
            query, params = create_edge_cypher(edge)
            try:
                conn.execute_write(query, params)
            except Exception as e:
                # Log error but continue
                print(f"Error persisting edge {edge.edge_type}: {e}")

    def clear_case_graph(self, case_id: str) -> int:
        """
        Remove all nodes and edges for a case.

        Args:
            case_id: Case identifier.

        Returns:
            Number of nodes deleted.
        """
        conn = self._get_connection()

        query = """
        MATCH (n {case_id: $case_id})
        DETACH DELETE n
        RETURN count(n) as deleted_count
        """

        result = conn.execute_write(query, {"case_id": case_id})
        if result:
            return result[0].get("deleted_count", 0)
        return 0

    def get_graph_stats(self, case_id: str) -> dict[str, int]:
        """
        Get node and edge counts for a case.

        Args:
            case_id: Case identifier.

        Returns:
            Dictionary with node_count and edge_count.
        """
        conn = self._get_connection()

        node_query = """
        MATCH (n {case_id: $case_id})
        RETURN count(n) as node_count
        """

        edge_query = """
        MATCH (n {case_id: $case_id})-[r]->(m {case_id: $case_id})
        RETURN count(r) as edge_count
        """

        node_result = conn.execute_query(node_query, {"case_id": case_id})
        edge_result = conn.execute_query(edge_query, {"case_id": case_id})

        node_count = node_result[0].get("node_count", 0) if node_result else 0
        edge_count = edge_result[0].get("edge_count", 0) if edge_result else 0

        return {
            "node_count": node_count,
            "edge_count": edge_count,
        }

    def verify_determinism(
        self,
        case_id: str,
        chunks: list[Union[Chunk, dict[str, Any]]],
        entities: list[Union[ExtractedEntity, dict[str, Any]]],
        runs: int = 10,
    ) -> bool:
        """
        Verify that graph building is deterministic.

        Builds the graph multiple times and verifies identical results.

        Args:
            case_id: Case identifier.
            chunks: List of chunks.
            entities: List of entities.
            runs: Number of times to rebuild (default 10).

        Returns:
            True if all runs produce identical results.
        """
        results: list[tuple[int, int, set[str], set[str]]] = []

        for _ in range(runs):
            result = self.build_graph(
                case_id=case_id,
                chunks=chunks,
                entities=entities,
                persist_to_neo4j=False,  # Don't persist during verification
            )

            node_ids = {n.node_id for n in result.nodes}
            edge_keys = {f"{e.from_node}->{e.edge_type.value}->{e.to_node}" for e in result.edges}

            results.append(
                (
                    result.total_nodes,
                    result.total_edges,
                    node_ids,
                    edge_keys,
                )
            )

        # Compare all results to first
        first = results[0]
        for i, result in enumerate(results[1:], 2):
            if result != first:
                print(f"Determinism failed at run {i}")
                return False

        return True


def build_graph_sync(
    case_id: str,
    chunks: list[Union[Chunk, dict[str, Any]]],
    entities: list[Union[ExtractedEntity, dict[str, Any]]],
    persist_to_neo4j: bool = True,
) -> GraphBuildResult:
    """
    Synchronous graph building.

    Convenience function for building a graph.

    Args:
        case_id: Case identifier.
        chunks: List of chunks from Stage 5.
        entities: List of entities from Stage 6.
        persist_to_neo4j: Whether to persist to Neo4j.

    Returns:
        GraphBuildResult with all nodes and edges.
    """
    builder = GraphBuilder()
    return builder.build_graph(
        case_id=case_id,
        chunks=chunks,
        entities=entities,
        persist_to_neo4j=persist_to_neo4j,
    )


async def build_graph_async(
    case_id: str,
    chunks: list[Union[Chunk, dict[str, Any]]],
    entities: list[Union[ExtractedEntity, dict[str, Any]]],
    persist_to_neo4j: bool = True,
) -> GraphBuildResult:
    """
    Async-safe graph building.

    The actual graph building is synchronous, but this wrapper
    allows integration with async pipelines.

    Args:
        case_id: Case identifier.
        chunks: List of chunks from Stage 5.
        entities: List of entities from Stage 6.
        persist_to_neo4j: Whether to persist to Neo4j.

    Returns:
        GraphBuildResult with all nodes and edges.
    """
    builder = GraphBuilder()
    return builder.build_graph(
        case_id=case_id,
        chunks=chunks,
        entities=entities,
        persist_to_neo4j=persist_to_neo4j,
    )
