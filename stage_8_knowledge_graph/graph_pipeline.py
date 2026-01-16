"""
Stage 8: Knowledge Graph - Pipeline

Public API for knowledge graph construction.

This module provides the main entry points for:
- Processing case data into a knowledge graph
- Querying graph statistics
- Clearing case graphs
"""

from typing import Any, Optional, Union

from stage_5_chunking.models import Chunk
from stage_6_ner.models import ExtractedEntity

from .graph_builder import GraphBuilder
from .models import GraphBuildResult
from .neo4j_connection import Neo4jConnection, connect_neo4j, get_connection


class GraphPipeline:
    """
    Main pipeline for knowledge graph construction.

    Provides a high-level API for processing case data
    into a Neo4j knowledge graph.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        """
        Initialize the graph pipeline.

        Args:
            uri: Neo4j connection URI.
            user: Neo4j username.
            password: Neo4j password.
            database: Neo4j database name.
        """
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._builder: Optional[GraphBuilder] = None

    def _ensure_connected(self) -> Neo4jConnection:
        """Ensure Neo4j connection is established."""
        conn = get_connection()
        if not conn.is_connected():
            connect_neo4j(
                uri=self._uri,
                user=self._user,
                password=self._password,
                database=self._database,
            )
        return conn

    def _get_builder(self) -> GraphBuilder:
        """Get or create the graph builder."""
        if self._builder is None:
            self._builder = GraphBuilder()
        return self._builder

    def process_case(
        self,
        case_id: str,
        chunks: list[Union[Chunk, dict[str, Any]]],
        entities: list[Union[ExtractedEntity, dict[str, Any]]],
        clear_existing: bool = False,
    ) -> GraphBuildResult:
        """
        Process case data into a knowledge graph.

        This is the main entry point for graph construction.

        Args:
            case_id: Case identifier.
            chunks: List of chunks from Stage 5.
            entities: List of entities from Stage 6.
            clear_existing: Whether to clear existing graph data first.

        Returns:
            GraphBuildResult with all nodes and edges.
        """
        self._ensure_connected()
        builder = self._get_builder()

        if clear_existing:
            builder.clear_case_graph(case_id)

        return builder.build_graph(
            case_id=case_id,
            chunks=chunks,
            entities=entities,
            persist_to_neo4j=True,
        )

    def process_case_offline(
        self,
        case_id: str,
        chunks: list[Union[Chunk, dict[str, Any]]],
        entities: list[Union[ExtractedEntity, dict[str, Any]]],
    ) -> GraphBuildResult:
        """
        Process case data without persisting to Neo4j.

        Useful for testing and validation.

        Args:
            case_id: Case identifier.
            chunks: List of chunks from Stage 5.
            entities: List of entities from Stage 6.

        Returns:
            GraphBuildResult with all nodes and edges.
        """
        builder = GraphBuilder()
        return builder.build_graph(
            case_id=case_id,
            chunks=chunks,
            entities=entities,
            persist_to_neo4j=False,
        )

    def clear_case_graph(self, case_id: str) -> int:
        """
        Remove all nodes and edges for a case.

        Args:
            case_id: Case identifier.

        Returns:
            Number of nodes deleted.
        """
        self._ensure_connected()
        builder = self._get_builder()
        return builder.clear_case_graph(case_id)

    def get_graph_stats(self, case_id: str) -> dict[str, int]:
        """
        Get node and edge counts for a case.

        Args:
            case_id: Case identifier.

        Returns:
            Dictionary with node_count and edge_count.
        """
        self._ensure_connected()
        builder = self._get_builder()
        return builder.get_graph_stats(case_id)

    def verify_determinism(
        self,
        case_id: str,
        chunks: list[Union[Chunk, dict[str, Any]]],
        entities: list[Union[ExtractedEntity, dict[str, Any]]],
        runs: int = 100,
    ) -> bool:
        """
        Verify that graph building is deterministic.

        Builds the graph multiple times and verifies identical results.

        Args:
            case_id: Case identifier.
            chunks: List of chunks.
            entities: List of entities.
            runs: Number of times to rebuild (default 100).

        Returns:
            True if all runs produce identical results.
        """
        builder = GraphBuilder()
        return builder.verify_determinism(
            case_id=case_id,
            chunks=chunks,
            entities=entities,
            runs=runs,
        )


def process_case_sync(
    case_id: str,
    chunks: list[Union[Chunk, dict[str, Any]]],
    entities: list[Union[ExtractedEntity, dict[str, Any]]],
    clear_existing: bool = False,
) -> GraphBuildResult:
    """
    Synchronous case processing.

    Convenience function for processing a case.

    Args:
        case_id: Case identifier.
        chunks: List of chunks from Stage 5.
        entities: List of entities from Stage 6.
        clear_existing: Whether to clear existing graph data first.

    Returns:
        GraphBuildResult with all nodes and edges.
    """
    pipeline = GraphPipeline()
    return pipeline.process_case(
        case_id=case_id,
        chunks=chunks,
        entities=entities,
        clear_existing=clear_existing,
    )


async def process_case_async(
    case_id: str,
    chunks: list[Union[Chunk, dict[str, Any]]],
    entities: list[Union[ExtractedEntity, dict[str, Any]]],
    clear_existing: bool = False,
) -> GraphBuildResult:
    """
    Async-safe case processing.

    The actual processing is synchronous, but this wrapper
    allows integration with async pipelines.

    Args:
        case_id: Case identifier.
        chunks: List of chunks from Stage 5.
        entities: List of entities from Stage 6.
        clear_existing: Whether to clear existing graph data first.

    Returns:
        GraphBuildResult with all nodes and edges.
    """
    pipeline = GraphPipeline()
    return pipeline.process_case(
        case_id=case_id,
        chunks=chunks,
        entities=entities,
        clear_existing=clear_existing,
    )
