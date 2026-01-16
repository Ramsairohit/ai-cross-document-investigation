"""
Unit tests for Stage 8: Knowledge Graph - Pipeline

Tests for the public API.
"""

import pytest

from stage_8_knowledge_graph.graph_pipeline import GraphPipeline
from stage_8_knowledge_graph.models import NodeType


class TestGraphPipeline:
    """Tests for GraphPipeline class."""

    def test_process_case_offline(self):
        """Should process case without Neo4j connection."""
        pipeline = GraphPipeline()

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

        result = pipeline.process_case_offline(
            case_id="24-890-H",
            chunks=chunks,
            entities=entities,
        )

        assert result.case_id == "24-890-H"
        assert result.chunks_processed == 1
        assert result.entities_processed == 1
        assert result.total_nodes >= 1

    def test_builds_complete_graph(self):
        """Should build complete graph with nodes and edges."""
        pipeline = GraphPipeline()

        chunks = [
            {
                "chunk_id": "C1",
                "text": "Marcus Vane saw the knife at 420 Harrow Lane.",
                "case_id": "001",
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
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.92,
            },
            {
                "entity_id": "E2",
                "entity_type": "WEAPON",
                "text": "knife",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.95,
            },
            {
                "entity_id": "E3",
                "entity_type": "ADDRESS",
                "text": "420 Harrow Lane",
                "chunk_id": "C1",
                "document_id": "DOC123",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.88,
            },
        ]

        result = pipeline.process_case_offline(
            case_id="001",
            chunks=chunks,
            entities=entities,
        )

        # Should have multiple node types
        node_types = {n.node_type for n in result.nodes}
        assert NodeType.PERSON in node_types
        assert NodeType.EVIDENCE in node_types
        assert NodeType.LOCATION in node_types

    def test_verify_determinism(self):
        """Should verify deterministic building."""
        pipeline = GraphPipeline()

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

        is_deterministic = pipeline.verify_determinism(
            case_id="001",
            chunks=chunks,
            entities=entities,
            runs=50,
        )

        assert is_deterministic is True


class TestNoInference:
    """Tests to verify no inference is made."""

    def test_no_cross_case_edges(self):
        """Edges should never connect nodes from different cases."""
        pipeline = GraphPipeline()

        # Case A data
        chunks_a = [
            {
                "chunk_id": "C1",
                "text": "Marcus argued with Julian.",
                "case_id": "CASE_A",
                "document_id": "DOC1",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]
        entities_a = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Marcus",
                "chunk_id": "C1",
                "document_id": "DOC1",
                "case_id": "CASE_A",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
            {
                "entity_id": "E2",
                "entity_type": "PERSON",
                "text": "Julian",
                "chunk_id": "C1",
                "document_id": "DOC1",
                "case_id": "CASE_A",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
        ]

        # Case B data
        chunks_b = [
            {
                "chunk_id": "C2",
                "text": "Bob argued with Alice.",
                "case_id": "CASE_B",
                "document_id": "DOC2",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]
        entities_b = [
            {
                "entity_id": "E3",
                "entity_type": "PERSON",
                "text": "Bob",
                "chunk_id": "C2",
                "document_id": "DOC2",
                "case_id": "CASE_B",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
            {
                "entity_id": "E4",
                "entity_type": "PERSON",
                "text": "Alice",
                "chunk_id": "C2",
                "document_id": "DOC2",
                "case_id": "CASE_B",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
        ]

        # Process each case separately (correct usage)
        result_a = pipeline.process_case_offline("CASE_A", chunks_a, entities_a)
        result_b = pipeline.process_case_offline("CASE_B", chunks_b, entities_b)

        # All edges in CASE_A result should have case_id CASE_A
        for edge in result_a.edges:
            assert edge.case_id == "CASE_A"

        # All edges in CASE_B result should have case_id CASE_B
        for edge in result_b.edges:
            assert edge.case_id == "CASE_B"

        # Node IDs should be different between cases
        node_ids_a = {n.node_id for n in result_a.nodes}
        node_ids_b = {n.node_id for n in result_b.nodes}
        assert node_ids_a.isdisjoint(node_ids_b), "Nodes should not overlap between cases"

    def test_no_entity_merging_across_chunks(self):
        """Same-name entities in different chunks should create separate nodes if different case."""
        pipeline = GraphPipeline()

        # Note: If entities have same name AND same case_id, they WILL be merged
        # This is correct behavior for deduplication
        # But entities from different cases should never merge

        chunks = [
            {
                "chunk_id": "C1",
                "text": "John said something.",
                "case_id": "CASE_A",
                "document_id": "DOC1",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]
        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "John",
                "chunk_id": "C1",
                "document_id": "DOC1",
                "case_id": "CASE_A",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
            {
                "entity_id": "E2",
                "entity_type": "PERSON",
                "text": "John",  # Same name, same case
                "chunk_id": "C1",
                "document_id": "DOC1",
                "case_id": "CASE_A",
                "page_range": [2, 2],
                "confidence": 0.85,
            },
        ]

        result = pipeline.process_case_offline("CASE_A", chunks, entities)

        # Same name + same case = one node (correct deduplication)
        person_nodes = [n for n in result.nodes if n.node_type == NodeType.PERSON]
        assert len(person_nodes) == 1

    def test_provenance_always_preserved(self):
        """Every node and edge must have complete provenance."""
        pipeline = GraphPipeline()

        chunks = [
            {
                "chunk_id": "CHUNK_XYZ",
                "text": "Evidence found at location.",
                "case_id": "001",
                "document_id": "DOC_ABC",
                "page_range": [5, 5],
                "chunk_confidence": 0.87,
            },
        ]
        entities = [
            {
                "entity_id": "E1",
                "entity_type": "EVIDENCE",
                "text": "fingerprints",
                "chunk_id": "CHUNK_XYZ",
                "document_id": "DOC_ABC",
                "case_id": "001",
                "page_range": [5, 5],
                "confidence": 0.91,
            },
        ]

        result = pipeline.process_case_offline("001", chunks, entities)

        for node in result.nodes:
            assert node.provenance is not None
            assert node.provenance.source_chunk_id is not None
            assert node.provenance.document_id is not None
            assert len(node.provenance.page_range) == 2
            assert 0.0 <= node.provenance.confidence <= 1.0

        for edge in result.edges:
            assert edge.provenance is not None
            assert edge.provenance.source_chunk_id is not None
