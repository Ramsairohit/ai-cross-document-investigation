"""
Stage 8: Knowledge Graph - Node Builder

Deterministic node creation from Stage 6 entities.

IMPORTANT:
- Node IDs are stable across runs
- One node per unique (type, name, case_id) combination
- All nodes have full provenance
"""

import hashlib
from typing import Any, Union

from stage_6_ner.models import ExtractedEntity

from .models import ENTITY_TO_NODE_TYPE, GraphNode, NodeType, Provenance


def normalize_name(name: str) -> str:
    """
    Normalize entity name for consistent node IDs.

    Args:
        name: Raw entity name.

    Returns:
        Normalized name (lowercase, trimmed, single spaces).
    """
    return " ".join(name.lower().strip().split())


def generate_node_id(node_type: NodeType, name: str, case_id: str) -> str:
    """
    Generate a deterministic node ID.

    Format: {NodeType}:{normalized_name}:{case_id}

    Args:
        node_type: Type of the node.
        name: Entity name.
        case_id: Case identifier.

    Returns:
        Deterministic node ID string.
    """
    normalized = normalize_name(name)
    return f"{node_type.value}:{normalized}:{case_id}"


def generate_node_hash(node_type: NodeType, name: str, case_id: str) -> str:
    """
    Generate a hash-based node ID for very long names.

    Used as fallback when name exceeds reasonable length.

    Args:
        node_type: Type of the node.
        name: Entity name.
        case_id: Case identifier.

    Returns:
        Hash-based node ID.
    """
    normalized = normalize_name(name)
    content = f"{node_type.value}:{normalized}:{case_id}"
    hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{node_type.value}:{hash_val}:{case_id}"


def get_node_type(entity_type: str) -> NodeType:
    """
    Map Stage 6 EntityType to NodeType.

    Args:
        entity_type: Entity type string from Stage 6.

    Returns:
        Corresponding NodeType.

    Raises:
        ValueError: If entity type cannot be mapped.
    """
    entity_type_upper = entity_type.upper()
    if entity_type_upper in ENTITY_TO_NODE_TYPE:
        return ENTITY_TO_NODE_TYPE[entity_type_upper]
    raise ValueError(f"Unknown entity type: {entity_type}")


def entity_to_node(entity: Union[ExtractedEntity, dict[str, Any]]) -> GraphNode:
    """
    Convert a Stage 6 ExtractedEntity to a GraphNode.

    Args:
        entity: ExtractedEntity or dict with entity data.

    Returns:
        GraphNode with full provenance.
    """
    # Handle dict input
    if isinstance(entity, dict):
        entity_id = entity.get("entity_id", "")
        entity_type = entity.get("entity_type", "")
        if hasattr(entity_type, "value"):
            entity_type = entity_type.value
        text = entity.get("text", "")
        chunk_id = entity.get("chunk_id", "")
        document_id = entity.get("document_id", "")
        case_id = entity.get("case_id", "")
        page_range = entity.get("page_range", [1, 1])
        confidence = entity.get("confidence", 0.0)
        role = entity.get("role")
    else:
        entity_id = entity.entity_id
        entity_type = entity.entity_type.value
        text = entity.text
        chunk_id = entity.chunk_id
        document_id = entity.document_id
        case_id = entity.case_id
        page_range = entity.page_range
        confidence = entity.confidence
        role = entity.role

    # Get node type
    node_type = get_node_type(entity_type)

    # Generate deterministic node ID
    node_id = generate_node_id(node_type, text, case_id)

    # Build properties based on node type
    properties: dict[str, Any] = {"name": text}

    if node_type == NodeType.PERSON:
        if role:
            properties["role"] = role
    elif node_type == NodeType.EVIDENCE:
        properties["label"] = text
        properties["evidence_type"] = entity_type
    elif node_type == NodeType.LOCATION:
        properties["name"] = text
    elif node_type == NodeType.EVENT:
        properties["description"] = text
        properties["timestamp"] = None  # Will be populated if time entity

    # Create provenance
    provenance = Provenance(
        source_chunk_id=chunk_id,
        document_id=document_id,
        page_range=page_range,
        confidence=confidence,
    )

    return GraphNode(
        node_type=node_type,
        node_id=node_id,
        case_id=case_id,
        properties=properties,
        provenance=provenance,
    )


def build_nodes(
    entities: list[Union[ExtractedEntity, dict[str, Any]]],
) -> list[GraphNode]:
    """
    Build graph nodes from a list of entities.

    Deduplicates by node_id, keeping first occurrence.
    This ensures deterministic output.

    Args:
        entities: List of entities from Stage 6.

    Returns:
        List of unique GraphNode objects.
    """
    seen_ids: set[str] = set()
    nodes: list[GraphNode] = []

    for entity in entities:
        try:
            node = entity_to_node(entity)
            if node.node_id not in seen_ids:
                seen_ids.add(node.node_id)
                nodes.append(node)
        except ValueError:
            # Skip entities that cannot be mapped
            continue

    return nodes


def create_node_cypher(node: GraphNode) -> tuple[str, dict[str, Any]]:
    """
    Generate MERGE Cypher query for a node.

    Uses MERGE to ensure idempotent node creation.

    Args:
        node: GraphNode to create.

    Returns:
        Tuple of (query_string, parameters).
    """
    label = node.node_type.value

    # Build property string
    props = {
        "node_id": node.node_id,
        "case_id": node.case_id,
        "source_chunk_id": node.provenance.source_chunk_id,
        "document_id": node.provenance.document_id,
        "page_range": node.provenance.page_range,
        "confidence": node.provenance.confidence,
        **node.properties,
    }

    query = f"""
    MERGE (n:{label} {{node_id: $node_id, case_id: $case_id}})
    ON CREATE SET n += $props
    ON MATCH SET n.confidence = CASE
        WHEN n.confidence < $confidence THEN $confidence
        ELSE n.confidence
    END
    RETURN n.node_id as node_id
    """

    parameters = {
        "node_id": node.node_id,
        "case_id": node.case_id,
        "confidence": node.provenance.confidence,
        "props": props,
    }

    return query, parameters


def build_document_node(
    document_id: str,
    case_id: str,
    chunk_id: str,
    page_range: list[int],
) -> GraphNode:
    """
    Build a Document node.

    Args:
        document_id: Source document ID.
        case_id: Case identifier.
        chunk_id: Source chunk ID for provenance.
        page_range: Page range.

    Returns:
        GraphNode for the document.
    """
    node_id = f"{NodeType.DOCUMENT.value}:{document_id}:{case_id}"

    provenance = Provenance(
        source_chunk_id=chunk_id,
        document_id=document_id,
        page_range=page_range,
        confidence=1.0,  # Documents have full confidence
    )

    return GraphNode(
        node_type=NodeType.DOCUMENT,
        node_id=node_id,
        case_id=case_id,
        properties={"document_id": document_id},
        provenance=provenance,
    )
