"""
Unit tests for Stage 8: Knowledge Graph - Data Models

Tests for node types, edge types, provenance, and graph schemas.
"""

import pytest
from pydantic import ValidationError

from stage_8_knowledge_graph.models import (
    EDGE_PATTERNS,
    ENTITY_TO_NODE_TYPE,
    EdgeType,
    GraphBuildResult,
    GraphEdge,
    GraphNode,
    NodeType,
    Provenance,
)


class TestNodeType:
    """Tests for NodeType enum."""

    def test_all_node_types_exist(self):
        """All required node types should exist."""
        assert NodeType.PERSON.value == "Person"
        assert NodeType.EVIDENCE.value == "Evidence"
        assert NodeType.LOCATION.value == "Location"
        assert NodeType.EVENT.value == "Event"
        assert NodeType.DOCUMENT.value == "Document"

    def test_node_type_count(self):
        """Should have exactly 5 node types."""
        assert len(NodeType) == 5


class TestEdgeType:
    """Tests for EdgeType enum."""

    def test_all_edge_types_exist(self):
        """All required edge types should exist."""
        assert EdgeType.WITNESSED.value == "WITNESSED"
        assert EdgeType.FOUND_IN.value == "FOUND_IN"
        assert EdgeType.OWNS.value == "OWNS"
        assert EdgeType.ACCOMPANIED_BY.value == "ACCOMPANIED_BY"
        assert EdgeType.ARGUED_WITH.value == "ARGUED_WITH"

    def test_edge_type_count(self):
        """Should have exactly 5 edge types."""
        assert len(EdgeType) == 5


class TestProvenance:
    """Tests for Provenance model."""

    def test_valid_provenance(self):
        """Should create valid provenance."""
        prov = Provenance(
            source_chunk_id="CHUNK_001",
            document_id="DOC123",
            page_range=[2, 2],
            confidence=0.92,
        )
        assert prov.source_chunk_id == "CHUNK_001"
        assert prov.document_id == "DOC123"
        assert prov.page_range == [2, 2]
        assert prov.confidence == 0.92

    def test_missing_source_chunk_id(self):
        """Should reject missing source_chunk_id."""
        with pytest.raises(ValidationError):
            Provenance(
                document_id="DOC123",
                page_range=[2, 2],
                confidence=0.92,
            )

    def test_invalid_page_range_length(self):
        """Should reject page_range with wrong length."""
        with pytest.raises(ValidationError):
            Provenance(
                source_chunk_id="CHUNK_001",
                document_id="DOC123",
                page_range=[2],
                confidence=0.92,
            )

    def test_confidence_range(self):
        """Should validate confidence range."""
        # Valid at bounds
        prov_low = Provenance(
            source_chunk_id="C1",
            document_id="D1",
            page_range=[1, 1],
            confidence=0.0,
        )
        assert prov_low.confidence == 0.0

        prov_high = Provenance(
            source_chunk_id="C1",
            document_id="D1",
            page_range=[1, 1],
            confidence=1.0,
        )
        assert prov_high.confidence == 1.0

        # Invalid below 0
        with pytest.raises(ValidationError):
            Provenance(
                source_chunk_id="C1",
                document_id="D1",
                page_range=[1, 1],
                confidence=-0.1,
            )

        # Invalid above 1
        with pytest.raises(ValidationError):
            Provenance(
                source_chunk_id="C1",
                document_id="D1",
                page_range=[1, 1],
                confidence=1.1,
            )


class TestGraphNode:
    """Tests for GraphNode model."""

    def test_valid_graph_node(self):
        """Should create valid graph node."""
        prov = Provenance(
            source_chunk_id="CHUNK_001",
            document_id="DOC123",
            page_range=[2, 2],
            confidence=0.92,
        )
        node = GraphNode(
            node_type=NodeType.PERSON,
            node_id="Person:marcus vane:24-890-H",
            case_id="24-890-H",
            properties={"name": "Marcus Vane", "role": "WITNESS"},
            provenance=prov,
        )
        assert node.node_type == NodeType.PERSON
        assert node.node_id == "Person:marcus vane:24-890-H"
        assert node.case_id == "24-890-H"
        assert node.properties["name"] == "Marcus Vane"

    def test_missing_provenance(self):
        """Should reject node without provenance."""
        with pytest.raises(ValidationError):
            GraphNode(
                node_type=NodeType.PERSON,
                node_id="Person:test:001",
                case_id="001",
                properties={},
            )

    def test_default_properties(self):
        """Properties should default to empty dict."""
        prov = Provenance(
            source_chunk_id="C1",
            document_id="D1",
            page_range=[1, 1],
            confidence=1.0,
        )
        node = GraphNode(
            node_type=NodeType.LOCATION,
            node_id="Location:test:001",
            case_id="001",
            provenance=prov,
        )
        assert node.properties == {}


class TestGraphEdge:
    """Tests for GraphEdge model."""

    def test_valid_graph_edge(self):
        """Should create valid graph edge."""
        prov = Provenance(
            source_chunk_id="CHUNK_001",
            document_id="DOC123",
            page_range=[2, 2],
            confidence=0.92,
        )
        edge = GraphEdge(
            edge_type=EdgeType.ARGUED_WITH,
            from_node="Person:marcus vane:24-890-H",
            to_node="Person:julian thorne:24-890-H",
            case_id="24-890-H",
            provenance=prov,
        )
        assert edge.edge_type == EdgeType.ARGUED_WITH
        assert edge.from_node == "Person:marcus vane:24-890-H"
        assert edge.to_node == "Person:julian thorne:24-890-H"

    def test_missing_provenance(self):
        """Should reject edge without provenance."""
        with pytest.raises(ValidationError):
            GraphEdge(
                edge_type=EdgeType.WITNESSED,
                from_node="Person:a:001",
                to_node="Event:b:001",
                case_id="001",
            )


class TestGraphBuildResult:
    """Tests for GraphBuildResult model."""

    def test_valid_result(self):
        """Should create valid build result."""
        result = GraphBuildResult(
            case_id="24-890-H",
            nodes=[],
            edges=[],
            total_nodes=0,
            total_edges=0,
            documents_processed=0,
            chunks_processed=0,
            entities_processed=0,
        )
        assert result.case_id == "24-890-H"
        assert result.total_nodes == 0

    def test_default_lists(self):
        """Lists should default to empty."""
        result = GraphBuildResult(case_id="001")
        assert result.nodes == []
        assert result.edges == []


class TestEntityToNodeTypeMapping:
    """Tests for entity type to node type mapping."""

    def test_person_types(self):
        """PERSON, WITNESS, SUSPECT should map to Person."""
        assert ENTITY_TO_NODE_TYPE["PERSON"] == NodeType.PERSON
        assert ENTITY_TO_NODE_TYPE["WITNESS"] == NodeType.PERSON
        assert ENTITY_TO_NODE_TYPE["SUSPECT"] == NodeType.PERSON

    def test_location_types(self):
        """LOCATION, ADDRESS should map to Location."""
        assert ENTITY_TO_NODE_TYPE["LOCATION"] == NodeType.LOCATION
        assert ENTITY_TO_NODE_TYPE["ADDRESS"] == NodeType.LOCATION

    def test_evidence_types(self):
        """EVIDENCE, WEAPON, PHONE should map to Evidence."""
        assert ENTITY_TO_NODE_TYPE["EVIDENCE"] == NodeType.EVIDENCE
        assert ENTITY_TO_NODE_TYPE["WEAPON"] == NodeType.EVIDENCE
        assert ENTITY_TO_NODE_TYPE["PHONE"] == NodeType.EVIDENCE

    def test_event_types(self):
        """TIME should map to Event."""
        assert ENTITY_TO_NODE_TYPE["TIME"] == NodeType.EVENT


class TestEdgePatterns:
    """Tests for edge detection patterns."""

    def test_all_edge_types_have_patterns(self):
        """Every edge type should have patterns."""
        for edge_type in EdgeType:
            assert edge_type in EDGE_PATTERNS
            assert len(EDGE_PATTERNS[edge_type]) > 0

    def test_patterns_are_valid_regex(self):
        """All patterns should be valid regex."""
        import re

        for edge_type, patterns in EDGE_PATTERNS.items():
            for pattern in patterns:
                try:
                    re.compile(pattern)
                except re.error as e:
                    pytest.fail(f"Invalid regex for {edge_type}: {pattern} - {e}")
