"""
Unit tests for Stage 8: Knowledge Graph - Node Builder

Tests for node ID generation, entity conversion, and Cypher query generation.
"""

import pytest

from stage_8_knowledge_graph.models import NodeType, Provenance
from stage_8_knowledge_graph.node_builder import (
    build_document_node,
    build_nodes,
    create_node_cypher,
    entity_to_node,
    generate_node_hash,
    generate_node_id,
    get_node_type,
    normalize_name,
)


class TestNormalizeName:
    """Tests for name normalization."""

    def test_lowercase(self):
        """Should convert to lowercase."""
        assert normalize_name("Marcus Vane") == "marcus vane"

    def test_trim_whitespace(self):
        """Should trim leading/trailing whitespace."""
        assert normalize_name("  Marcus Vane  ") == "marcus vane"

    def test_collapse_spaces(self):
        """Should collapse multiple spaces."""
        assert normalize_name("Marcus   Vane") == "marcus vane"

    def test_empty_string(self):
        """Should handle empty string."""
        assert normalize_name("") == ""

    def test_already_normalized(self):
        """Should return same if already normalized."""
        assert normalize_name("marcus vane") == "marcus vane"


class TestGenerateNodeId:
    """Tests for node ID generation."""

    def test_format(self):
        """Should follow {NodeType}:{name}:{case_id} format."""
        node_id = generate_node_id(NodeType.PERSON, "Marcus Vane", "24-890-H")
        assert node_id == "Person:marcus vane:24-890-H"

    def test_deterministic(self):
        """Same input should always produce same output."""
        ids = [generate_node_id(NodeType.PERSON, "Marcus Vane", "24-890-H") for _ in range(100)]
        assert all(id == ids[0] for id in ids)

    def test_case_insensitive_name(self):
        """Name should be normalized regardless of case."""
        id1 = generate_node_id(NodeType.PERSON, "Marcus Vane", "001")
        id2 = generate_node_id(NodeType.PERSON, "MARCUS VANE", "001")
        id3 = generate_node_id(NodeType.PERSON, "marcus vane", "001")
        assert id1 == id2 == id3

    def test_different_types(self):
        """Different node types should produce different IDs."""
        person_id = generate_node_id(NodeType.PERSON, "test", "001")
        location_id = generate_node_id(NodeType.LOCATION, "test", "001")
        assert person_id != location_id

    def test_different_cases(self):
        """Different case_ids should produce different IDs."""
        id1 = generate_node_id(NodeType.PERSON, "test", "001")
        id2 = generate_node_id(NodeType.PERSON, "test", "002")
        assert id1 != id2


class TestGenerateNodeHash:
    """Tests for hash-based node ID generation."""

    def test_returns_hash(self):
        """Should return hash-based ID."""
        hash_id = generate_node_hash(NodeType.PERSON, "Marcus Vane", "24-890-H")
        assert hash_id.startswith("Person:")
        assert "24-890-H" in hash_id

    def test_deterministic(self):
        """Same input should always produce same hash."""
        hashes = [
            generate_node_hash(NodeType.PERSON, "Marcus Vane", "24-890-H") for _ in range(100)
        ]
        assert all(h == hashes[0] for h in hashes)


class TestGetNodeType:
    """Tests for entity type to node type mapping."""

    def test_person_mapping(self):
        """PERSON should map to Person."""
        assert get_node_type("PERSON") == NodeType.PERSON

    def test_witness_mapping(self):
        """WITNESS should map to Person."""
        assert get_node_type("WITNESS") == NodeType.PERSON

    def test_suspect_mapping(self):
        """SUSPECT should map to Person."""
        assert get_node_type("SUSPECT") == NodeType.PERSON

    def test_location_mapping(self):
        """LOCATION should map to Location."""
        assert get_node_type("LOCATION") == NodeType.LOCATION

    def test_address_mapping(self):
        """ADDRESS should map to Location."""
        assert get_node_type("ADDRESS") == NodeType.LOCATION

    def test_evidence_mapping(self):
        """EVIDENCE should map to Evidence."""
        assert get_node_type("EVIDENCE") == NodeType.EVIDENCE

    def test_weapon_mapping(self):
        """WEAPON should map to Evidence."""
        assert get_node_type("WEAPON") == NodeType.EVIDENCE

    def test_unknown_type(self):
        """Unknown type should raise ValueError."""
        with pytest.raises(ValueError):
            get_node_type("UNKNOWN_TYPE")

    def test_case_insensitive(self):
        """Should handle different cases."""
        assert get_node_type("person") == NodeType.PERSON
        assert get_node_type("Person") == NodeType.PERSON


class TestEntityToNode:
    """Tests for entity to node conversion."""

    def test_convert_person_entity(self):
        """Should convert person entity to node."""
        entity = {
            "entity_id": "ENT_001",
            "entity_type": "PERSON",
            "text": "Marcus Vane",
            "chunk_id": "CHUNK_001",
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "page_range": [2, 2],
            "confidence": 0.92,
            "role": "WITNESS",
        }
        node = entity_to_node(entity)

        assert node.node_type == NodeType.PERSON
        assert node.node_id == "Person:marcus vane:24-890-H"
        assert node.case_id == "24-890-H"
        assert node.properties["name"] == "Marcus Vane"
        assert node.properties["role"] == "WITNESS"
        assert node.provenance.source_chunk_id == "CHUNK_001"
        assert node.provenance.confidence == 0.92

    def test_convert_evidence_entity(self):
        """Should convert evidence entity to node."""
        entity = {
            "entity_id": "ENT_002",
            "entity_type": "WEAPON",
            "text": "knife",
            "chunk_id": "CHUNK_002",
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "page_range": [3, 3],
            "confidence": 0.95,
        }
        node = entity_to_node(entity)

        assert node.node_type == NodeType.EVIDENCE
        assert node.properties["label"] == "knife"
        assert node.properties["evidence_type"] == "WEAPON"

    def test_convert_location_entity(self):
        """Should convert location entity to node."""
        entity = {
            "entity_id": "ENT_003",
            "entity_type": "ADDRESS",
            "text": "420 Harrow Lane",
            "chunk_id": "CHUNK_003",
            "document_id": "DOC123",
            "case_id": "24-890-H",
            "page_range": [1, 1],
            "confidence": 0.88,
        }
        node = entity_to_node(entity)

        assert node.node_type == NodeType.LOCATION
        assert node.properties["name"] == "420 Harrow Lane"

    def test_provenance_attached(self):
        """Provenance should always be attached."""
        entity = {
            "entity_id": "ENT_001",
            "entity_type": "PERSON",
            "text": "Test Person",
            "chunk_id": "C1",
            "document_id": "D1",
            "case_id": "001",
            "page_range": [1, 1],
            "confidence": 0.9,
        }
        node = entity_to_node(entity)

        assert node.provenance is not None
        assert node.provenance.source_chunk_id == "C1"
        assert node.provenance.document_id == "D1"
        assert node.provenance.page_range == [1, 1]
        assert node.provenance.confidence == 0.9


class TestBuildNodes:
    """Tests for batch node building."""

    def test_builds_nodes_from_entities(self):
        """Should build nodes from entity list."""
        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Person A",
                "chunk_id": "C1",
                "document_id": "D1",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
            {
                "entity_id": "E2",
                "entity_type": "LOCATION",
                "text": "123 Main St",
                "chunk_id": "C1",
                "document_id": "D1",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.85,
            },
        ]
        nodes = build_nodes(entities)

        assert len(nodes) == 2
        node_types = {n.node_type for n in nodes}
        assert NodeType.PERSON in node_types
        assert NodeType.LOCATION in node_types

    def test_deduplicates_nodes(self):
        """Should deduplicate by node_id."""
        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Marcus Vane",
                "chunk_id": "C1",
                "document_id": "D1",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
            {
                "entity_id": "E2",
                "entity_type": "PERSON",
                "text": "Marcus Vane",  # Same person
                "chunk_id": "C2",
                "document_id": "D1",
                "case_id": "001",
                "page_range": [2, 2],
                "confidence": 0.85,
            },
        ]
        nodes = build_nodes(entities)

        assert len(nodes) == 1

    def test_empty_list(self):
        """Should handle empty entity list."""
        nodes = build_nodes([])
        assert nodes == []

    def test_skips_unknown_types(self):
        """Should skip entities with unknown types."""
        entities = [
            {
                "entity_id": "E1",
                "entity_type": "UNKNOWN",
                "text": "Test",
                "chunk_id": "C1",
                "document_id": "D1",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
        ]
        nodes = build_nodes(entities)
        assert len(nodes) == 0


class TestBuildDocumentNode:
    """Tests for document node building."""

    def test_builds_document_node(self):
        """Should build document node."""
        node = build_document_node(
            document_id="DOC123",
            case_id="24-890-H",
            chunk_id="CHUNK_001",
            page_range=[1, 10],
        )

        assert node.node_type == NodeType.DOCUMENT
        assert node.node_id == "Document:DOC123:24-890-H"
        assert node.properties["document_id"] == "DOC123"
        assert node.provenance.confidence == 1.0


class TestCreateNodeCypher:
    """Tests for Cypher query generation."""

    def test_generates_merge_query(self):
        """Should generate MERGE query."""
        prov = Provenance(
            source_chunk_id="C1",
            document_id="D1",
            page_range=[1, 1],
            confidence=0.9,
        )
        from stage_8_knowledge_graph.models import GraphNode

        node = GraphNode(
            node_type=NodeType.PERSON,
            node_id="Person:test:001",
            case_id="001",
            properties={"name": "Test"},
            provenance=prov,
        )

        query, params = create_node_cypher(node)

        assert "MERGE" in query
        assert "Person" in query
        assert params["node_id"] == "Person:test:001"
        assert params["case_id"] == "001"

    def test_parameterized_query(self):
        """Query should use parameters, not inline values."""
        prov = Provenance(
            source_chunk_id="C1",
            document_id="D1",
            page_range=[1, 1],
            confidence=0.9,
        )
        from stage_8_knowledge_graph.models import GraphNode

        node = GraphNode(
            node_type=NodeType.EVIDENCE,
            node_id="Evidence:knife:001",
            case_id="001",
            properties={"label": "knife"},
            provenance=prov,
        )

        query, params = create_node_cypher(node)

        # Should use $param syntax, not literal values
        assert "$node_id" in query
        assert "$case_id" in query
