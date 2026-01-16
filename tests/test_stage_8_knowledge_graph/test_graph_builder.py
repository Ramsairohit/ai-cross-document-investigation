"""
Unit tests for Stage 8: Knowledge Graph - Graph Builder

Tests for the main orchestrator that builds the full graph.
"""

import pytest

from stage_8_knowledge_graph.graph_builder import GraphBuilder
from stage_8_knowledge_graph.models import NodeType


class TestGraphBuilder:
    """Tests for GraphBuilder class."""

    def test_build_graph_offline(self):
        """Should build graph without Neo4j connection."""
        builder = GraphBuilder()

        chunks = [
            {
                "chunk_id": "C1",
                "text": "I saw Marcus Vane at the scene.",
                "case_id": "24-890-H",
                "document_id": "DOC123",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]
        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Marcus Vane",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "24-890-H",
                "page_range": [1, 1],
                "confidence": 0.92,
            },
        ]

        result = builder.build_graph(
            case_id="24-890-H",
            chunks=chunks,
            entities=entities,
            persist_to_neo4j=False,
        )

        assert result.case_id == "24-890-H"
        assert result.chunks_processed == 1
        assert result.entities_processed == 1
        assert result.total_nodes >= 1  # At least the person node

    def test_builds_nodes_from_entities(self):
        """Should create nodes from all entities."""
        builder = GraphBuilder()

        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Marcus Vane",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.92,
            },
            {
                "entity_id": "E2",
                "entity_type": "LOCATION",
                "text": "420 Harrow Lane",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.88,
            },
            {
                "entity_id": "E3",
                "entity_type": "WEAPON",
                "text": "knife",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.95,
            },
        ]
        chunks = [
            {
                "chunk_id": "C1",
                "text": "Found a knife at 420 Harrow Lane.",
                "case_id": "001",
                "document_id": "DOC123",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]

        result = builder.build_graph(
            case_id="001",
            chunks=chunks,
            entities=entities,
            persist_to_neo4j=False,
        )

        # Should have person, location, evidence, and document nodes
        node_types = {n.node_type for n in result.nodes}
        assert NodeType.PERSON in node_types
        assert NodeType.LOCATION in node_types
        assert NodeType.EVIDENCE in node_types

    def test_builds_document_nodes(self):
        """Should create document nodes from chunks."""
        builder = GraphBuilder()

        chunks = [
            {
                "chunk_id": "C1",
                "text": "First chunk.",
                "case_id": "001",
                "document_id": "DOC_A",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "text": "Second chunk.",
                "case_id": "001",
                "document_id": "DOC_B",
                "page_range": [2, 2],
                "chunk_confidence": 0.9,
            },
        ]

        result = builder.build_graph(
            case_id="001",
            chunks=chunks,
            entities=[],
            persist_to_neo4j=False,
        )

        doc_nodes = [n for n in result.nodes if n.node_type == NodeType.DOCUMENT]
        assert len(doc_nodes) == 2
        assert result.documents_processed == 2

    def test_deduplicates_document_nodes(self):
        """Should create one node per unique document."""
        builder = GraphBuilder()

        chunks = [
            {
                "chunk_id": "C1",
                "text": "First chunk.",
                "case_id": "001",
                "document_id": "DOC_A",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
            {
                "chunk_id": "C2",
                "text": "Second chunk from same doc.",
                "case_id": "001",
                "document_id": "DOC_A",  # Same document
                "page_range": [2, 2],
                "chunk_confidence": 0.9,
            },
        ]

        result = builder.build_graph(
            case_id="001",
            chunks=chunks,
            entities=[],
            persist_to_neo4j=False,
        )

        doc_nodes = [n for n in result.nodes if n.node_type == NodeType.DOCUMENT]
        assert len(doc_nodes) == 1

    def test_all_nodes_have_provenance(self):
        """Every node should have provenance."""
        builder = GraphBuilder()

        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Test Person",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
        ]
        chunks = [
            {
                "chunk_id": "C1",
                "text": "Test chunk.",
                "case_id": "001",
                "document_id": "DOC123",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]

        result = builder.build_graph(
            case_id="001",
            chunks=chunks,
            entities=entities,
            persist_to_neo4j=False,
        )

        for node in result.nodes:
            assert node.provenance is not None
            assert node.provenance.source_chunk_id is not None

    def test_all_edges_have_provenance(self):
        """Every edge should have provenance."""
        builder = GraphBuilder()

        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Marcus",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
            {
                "entity_id": "E2",
                "entity_type": "PERSON",
                "text": "Julian",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.85,
            },
        ]
        chunks = [
            {
                "chunk_id": "C1",
                "text": "Marcus argued with Julian.",
                "case_id": "001",
                "document_id": "DOC123",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]

        result = builder.build_graph(
            case_id="001",
            chunks=chunks,
            entities=entities,
            persist_to_neo4j=False,
        )

        for edge in result.edges:
            assert edge.provenance is not None
            assert edge.provenance.source_chunk_id is not None

    def test_empty_input(self):
        """Should handle empty input gracefully."""
        builder = GraphBuilder()

        result = builder.build_graph(
            case_id="001",
            chunks=[],
            entities=[],
            persist_to_neo4j=False,
        )

        assert result.case_id == "001"
        assert result.total_nodes == 0
        assert result.total_edges == 0


class TestDeterminism:
    """Tests for deterministic graph building."""

    def test_same_input_same_output(self):
        """Same input should always produce same output."""
        builder = GraphBuilder()

        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Marcus Vane",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.92,
            },
            {
                "entity_id": "E2",
                "entity_type": "PERSON",
                "text": "Julian Thorne",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.88,
            },
        ]
        chunks = [
            {
                "chunk_id": "C1",
                "text": "Marcus Vane argued with Julian Thorne.",
                "case_id": "001",
                "document_id": "DOC123",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]

        results = [
            builder.build_graph("001", chunks, entities, persist_to_neo4j=False) for _ in range(10)
        ]

        # All results should have same counts
        first = results[0]
        for result in results[1:]:
            assert result.total_nodes == first.total_nodes
            assert result.total_edges == first.total_edges

    def test_node_ids_stable(self):
        """Node IDs should be identical across runs."""
        builder = GraphBuilder()

        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Test Person",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
        ]
        chunks = [
            {
                "chunk_id": "C1",
                "text": "Test chunk.",
                "case_id": "001",
                "document_id": "DOC123",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]

        results = [
            builder.build_graph("001", chunks, entities, persist_to_neo4j=False) for _ in range(10)
        ]

        first_node_ids = {n.node_id for n in results[0].nodes}
        for result in results[1:]:
            result_node_ids = {n.node_id for n in result.nodes}
            assert result_node_ids == first_node_ids

    def test_verify_determinism_method(self):
        """verify_determinism should return True for deterministic input."""
        builder = GraphBuilder()

        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Marcus",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
        ]
        chunks = [
            {
                "chunk_id": "C1",
                "text": "Test chunk.",
                "case_id": "001",
                "document_id": "DOC123",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]

        is_deterministic = builder.verify_determinism(
            case_id="001",
            chunks=chunks,
            entities=entities,
            runs=10,
        )

        assert is_deterministic is True
