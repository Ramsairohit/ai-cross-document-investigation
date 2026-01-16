"""
Unit tests for Stage 8: Knowledge Graph - Determinism

Comprehensive tests to verify that graph building is deterministic.
The same input must always produce the identical graph.
"""

import pytest

from stage_8_knowledge_graph.graph_builder import GraphBuilder


class TestDeterminism100Runs:
    """Tests for 100-run determinism verification."""

    @pytest.fixture
    def sample_data(self):
        """Sample test data for determinism testing."""
        chunks = [
            {
                "chunk_id": "C-001",
                "text": "I, Marcus Vane, argued with Julian Thorne earlier that evening.",
                "case_id": "24-890-H",
                "document_id": "S001-24-890-H",
                "page_range": [2, 2],
                "chunk_confidence": 0.93,
            },
            {
                "chunk_id": "C-002",
                "text": "The knife was found at 420 Harrow Lane near the victim.",
                "case_id": "24-890-H",
                "document_id": "S001-24-890-H",
                "page_range": [3, 3],
                "chunk_confidence": 0.91,
            },
            {
                "chunk_id": "C-003",
                "text": "DNA evidence was discovered at the crime scene.",
                "case_id": "24-890-H",
                "document_id": "S002-24-890-H",
                "page_range": [1, 1],
                "chunk_confidence": 0.95,
            },
        ]
        entities = [
            {
                "entity_id": "E-001",
                "entity_type": "PERSON",
                "text": "Marcus Vane",
                "chunk_id": "C-001",
                "document_id": "S001-24-890-H",
                "case_id": "24-890-H",
                "page_range": [2, 2],
                "confidence": 0.92,
                "role": "WITNESS",
            },
            {
                "entity_id": "E-002",
                "entity_type": "PERSON",
                "text": "Julian Thorne",
                "chunk_id": "C-001",
                "document_id": "S001-24-890-H",
                "case_id": "24-890-H",
                "page_range": [2, 2],
                "confidence": 0.88,
            },
            {
                "entity_id": "E-003",
                "entity_type": "WEAPON",
                "text": "knife",
                "chunk_id": "C-002",
                "document_id": "S001-24-890-H",
                "case_id": "24-890-H",
                "page_range": [3, 3],
                "confidence": 0.95,
            },
            {
                "entity_id": "E-004",
                "entity_type": "ADDRESS",
                "text": "420 Harrow Lane",
                "chunk_id": "C-002",
                "document_id": "S001-24-890-H",
                "case_id": "24-890-H",
                "page_range": [3, 3],
                "confidence": 0.87,
            },
            {
                "entity_id": "E-005",
                "entity_type": "EVIDENCE",
                "text": "DNA",
                "chunk_id": "C-003",
                "document_id": "S002-24-890-H",
                "case_id": "24-890-H",
                "page_range": [1, 1],
                "confidence": 0.96,
            },
        ]
        return {"chunks": chunks, "entities": entities, "case_id": "24-890-H"}

    def test_100_run_determinism(self, sample_data):
        """
        Rebuild graph 100 times and verify identical results.

        This is a CRITICAL test for forensic-grade requirements.
        """
        builder = GraphBuilder()
        runs = 100

        results = []
        for _ in range(runs):
            result = builder.build_graph(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
                entities=sample_data["entities"],
                persist_to_neo4j=False,
            )
            results.append(result)

        # Compare all results to first
        first = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result.total_nodes == first.total_nodes, f"Node count differs at run {i}"
            assert result.total_edges == first.total_edges, f"Edge count differs at run {i}"

    def test_node_ids_identical_across_runs(self, sample_data):
        """Node IDs must be identical across all runs."""
        builder = GraphBuilder()
        runs = 100

        node_id_sets = []
        for _ in range(runs):
            result = builder.build_graph(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
                entities=sample_data["entities"],
                persist_to_neo4j=False,
            )
            node_ids = frozenset(n.node_id for n in result.nodes)
            node_id_sets.append(node_ids)

        first_set = node_id_sets[0]
        for i, node_ids in enumerate(node_id_sets[1:], 2):
            assert node_ids == first_set, f"Node IDs differ at run {i}"

    def test_edge_keys_identical_across_runs(self, sample_data):
        """Edge keys must be identical across all runs."""
        builder = GraphBuilder()
        runs = 100

        edge_key_sets = []
        for _ in range(runs):
            result = builder.build_graph(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
                entities=sample_data["entities"],
                persist_to_neo4j=False,
            )
            edge_keys = frozenset((e.from_node, e.to_node, e.edge_type.value) for e in result.edges)
            edge_key_sets.append(edge_keys)

        first_set = edge_key_sets[0]
        for i, edge_keys in enumerate(edge_key_sets[1:], 2):
            assert edge_keys == first_set, f"Edge keys differ at run {i}"

    def test_node_properties_identical_across_runs(self, sample_data):
        """Node properties must be identical across all runs."""
        builder = GraphBuilder()
        runs = 50  # Fewer runs for property comparison (expensive)

        property_snapshots = []
        for _ in range(runs):
            result = builder.build_graph(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
                entities=sample_data["entities"],
                persist_to_neo4j=False,
            )
            # Create sorted snapshot of properties
            snapshot = sorted(
                [(n.node_id, tuple(sorted(n.properties.items()))) for n in result.nodes]
            )
            property_snapshots.append(snapshot)

        first_snapshot = property_snapshots[0]
        for i, snapshot in enumerate(property_snapshots[1:], 2):
            assert snapshot == first_snapshot, f"Node properties differ at run {i}"

    def test_provenance_identical_across_runs(self, sample_data):
        """Provenance must be identical across all runs."""
        builder = GraphBuilder()
        runs = 50

        provenance_snapshots = []
        for _ in range(runs):
            result = builder.build_graph(
                case_id=sample_data["case_id"],
                chunks=sample_data["chunks"],
                entities=sample_data["entities"],
                persist_to_neo4j=False,
            )
            snapshot = sorted(
                [
                    (
                        n.node_id,
                        n.provenance.source_chunk_id,
                        n.provenance.document_id,
                        tuple(n.provenance.page_range),
                        n.provenance.confidence,
                    )
                    for n in result.nodes
                ]
            )
            provenance_snapshots.append(snapshot)

        first_snapshot = provenance_snapshots[0]
        for i, snapshot in enumerate(provenance_snapshots[1:], 2):
            assert snapshot == first_snapshot, f"Provenance differs at run {i}"


class TestDeterminismEdgeCases:
    """Edge case tests for determinism."""

    def test_empty_input_determinism(self):
        """Empty input should produce identical empty results."""
        builder = GraphBuilder()

        results = [builder.build_graph("001", [], [], persist_to_neo4j=False) for _ in range(100)]

        for result in results:
            assert result.total_nodes == 0
            assert result.total_edges == 0

    def test_single_entity_determinism(self):
        """Single entity should produce identical results."""
        builder = GraphBuilder()

        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "Test Person",
                "chunk_id": "C1",
                "document_id": "D1",
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
                "document_id": "D1",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]

        results = [
            builder.build_graph("001", chunks, entities, persist_to_neo4j=False) for _ in range(100)
        ]

        first = results[0]
        for result in results[1:]:
            assert result.total_nodes == first.total_nodes
            node_ids = {n.node_id for n in result.nodes}
            first_node_ids = {n.node_id for n in first.nodes}
            assert node_ids == first_node_ids

    def test_unicode_names_determinism(self):
        """Unicode entity names should not affect determinism."""
        builder = GraphBuilder()

        entities = [
            {
                "entity_id": "E1",
                "entity_type": "PERSON",
                "text": "José García",
                "chunk_id": "C1",
                "document_id": "D1",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.9,
            },
            {
                "entity_id": "E2",
                "entity_type": "LOCATION",
                "text": "北京市",
                "chunk_id": "C1",
                "document_id": "D1",
                "case_id": "001",
                "page_range": [1, 1],
                "confidence": 0.85,
            },
        ]
        chunks = [
            {
                "chunk_id": "C1",
                "text": "Test chunk.",
                "case_id": "001",
                "document_id": "D1",
                "page_range": [1, 1],
                "chunk_confidence": 0.9,
            },
        ]

        results = [
            builder.build_graph("001", chunks, entities, persist_to_neo4j=False) for _ in range(50)
        ]

        first_node_ids = {n.node_id for n in results[0].nodes}
        for result in results[1:]:
            node_ids = {n.node_id for n in result.nodes}
            assert node_ids == first_node_ids

    def test_case_sensitivity_determinism(self):
        """Different case variations should normalize consistently."""
        builder = GraphBuilder()

        # These should all produce the same node ID
        variants = ["Marcus Vane", "MARCUS VANE", "marcus vane", "  Marcus   Vane  "]

        node_ids = set()
        for name in variants:
            entities = [
                {
                    "entity_id": "E1",
                    "entity_type": "PERSON",
                    "text": name,
                    "chunk_id": "C1",
                    "document_id": "D1",
                    "case_id": "001",
                    "page_range": [1, 1],
                    "confidence": 0.9,
                },
            ]
            chunks = [
                {
                    "chunk_id": "C1",
                    "text": "Test.",
                    "case_id": "001",
                    "document_id": "D1",
                    "page_range": [1, 1],
                    "chunk_confidence": 0.9,
                },
            ]
            result = builder.build_graph("001", chunks, entities, persist_to_neo4j=False)
            person_nodes = [n for n in result.nodes if "Person" in n.node_id]
            if person_nodes:
                node_ids.add(person_nodes[0].node_id)

        # All variants should produce the same node ID
        assert len(node_ids) == 1
