"""
Stage 8: Knowledge Graph - Edge Builder

Relationship extraction from chunk text using pattern matching.

IMPORTANT:
- Regex-based only, no NLP inference
- Only explicit relationships from text
- All edges have full provenance
- No causal or guilt-implying edges
"""

import re
from typing import Any, Union

from stage_5_chunking.models import Chunk
from stage_6_ner.models import ExtractedEntity

from .models import EDGE_PATTERNS, EdgeType, GraphEdge, Provenance
from .node_builder import generate_node_id, get_node_type


def find_entities_in_chunk(
    chunk_text: str,
    entities: list[Union[ExtractedEntity, dict[str, Any]]],
    chunk_id: str,
) -> list[dict[str, Any]]:
    """
    Find entities that belong to a specific chunk.

    Args:
        chunk_text: Text content of the chunk.
        entities: All entities to filter.
        chunk_id: Target chunk ID.

    Returns:
        List of entities belonging to this chunk.
    """
    result: list[dict[str, Any]] = []

    for entity in entities:
        if isinstance(entity, dict):
            e_chunk_id = entity.get("chunk_id", "")
            e_text = entity.get("text", "")
            e_type = entity.get("entity_type", "")
            e_case_id = entity.get("case_id", "")
            e_document_id = entity.get("document_id", "")
            e_page_range = entity.get("page_range", [1, 1])
            e_confidence = entity.get("confidence", 0.0)
        else:
            e_chunk_id = entity.chunk_id
            e_text = entity.text
            e_type = (
                entity.entity_type.value
                if hasattr(entity.entity_type, "value")
                else str(entity.entity_type)
            )
            e_case_id = entity.case_id
            e_document_id = entity.document_id
            e_page_range = entity.page_range
            e_confidence = entity.confidence

        if e_chunk_id == chunk_id:
            result.append(
                {
                    "text": e_text,
                    "entity_type": e_type,
                    "case_id": e_case_id,
                    "document_id": e_document_id,
                    "page_range": e_page_range,
                    "confidence": e_confidence,
                }
            )

    return result


def detect_edge_type(text: str) -> list[EdgeType]:
    """
    Detect potential edge types from text using patterns.

    Args:
        text: Text to analyze.

    Returns:
        List of detected EdgeType values.
    """
    detected: list[EdgeType] = []
    text_lower = text.lower()

    for edge_type, patterns in EDGE_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                if edge_type not in detected:
                    detected.append(edge_type)
                break

    return detected


def extract_argued_with_edges(
    chunk_text: str,
    person_entities: list[dict[str, Any]],
    chunk_id: str,
    document_id: str,
    case_id: str,
    page_range: list[int],
    confidence: float,
) -> list[GraphEdge]:
    """
    Extract ARGUED_WITH edges from chunk text.

    Pattern: "[Person] argued with [Person]"

    Args:
        chunk_text: Text to analyze.
        person_entities: Person entities in the chunk.
        chunk_id: Source chunk ID.
        document_id: Source document ID.
        case_id: Case identifier.
        page_range: Page range.
        confidence: Confidence score.

    Returns:
        List of ARGUED_WITH edges.
    """
    edges: list[GraphEdge] = []

    if len(person_entities) < 2:
        return edges

    # Pattern: "argued with", "fought with", etc.
    argued_patterns = [
        r"(\w+(?:\s+\w+)?)\s+(?:argued|fought|quarreled|had\s+an?\s+argument)\s+with\s+(\w+(?:\s+\w+)?)",
    ]

    text_lower = chunk_text.lower()

    for pattern in argued_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            person1_text = match.group(1).strip()
            person2_text = match.group(2).strip()

            # Try to match with actual entities
            from_entity = None
            to_entity = None

            for entity in person_entities:
                entity_text_lower = entity["text"].lower()
                if person1_text in entity_text_lower or entity_text_lower in person1_text:
                    from_entity = entity
                if person2_text in entity_text_lower or entity_text_lower in person2_text:
                    to_entity = entity

            if from_entity and to_entity and from_entity != to_entity:
                from_node_type = get_node_type(from_entity["entity_type"])
                to_node_type = get_node_type(to_entity["entity_type"])

                from_node_id = generate_node_id(from_node_type, from_entity["text"], case_id)
                to_node_id = generate_node_id(to_node_type, to_entity["text"], case_id)

                provenance = Provenance(
                    source_chunk_id=chunk_id,
                    document_id=document_id,
                    page_range=page_range,
                    confidence=confidence,
                )

                edge = GraphEdge(
                    edge_type=EdgeType.ARGUED_WITH,
                    from_node=from_node_id,
                    to_node=to_node_id,
                    case_id=case_id,
                    provenance=provenance,
                )
                edges.append(edge)

    return edges


def extract_witnessed_edges(
    chunk_text: str,
    person_entities: list[dict[str, Any]],
    event_entities: list[dict[str, Any]],
    chunk_id: str,
    document_id: str,
    case_id: str,
    page_range: list[int],
    confidence: float,
) -> list[GraphEdge]:
    """
    Extract WITNESSED edges from chunk text.

    Pattern: "[Person] saw/witnessed [Event/Thing]"

    Args:
        chunk_text: Text to analyze.
        person_entities: Person entities in the chunk.
        event_entities: Event entities in the chunk.
        chunk_id: Source chunk ID.
        document_id: Source document ID.
        case_id: Case identifier.
        page_range: Page range.
        confidence: Confidence score.

    Returns:
        List of WITNESSED edges.
    """
    edges: list[GraphEdge] = []

    if not person_entities:
        return edges

    # Check if chunk contains witnessed patterns
    witnessed_patterns = [
        r"\b(?:saw|witnessed|observed|noticed|watched)\b",
    ]

    text_lower = chunk_text.lower()
    has_witness_verb = any(re.search(p, text_lower) for p in witnessed_patterns)

    if not has_witness_verb:
        return edges

    # If speaker is a person, they are the witness
    # Connect them to events/evidence in the chunk
    for person in person_entities:
        for event in event_entities:
            from_node_type = get_node_type(person["entity_type"])
            to_node_type = get_node_type(event["entity_type"])

            from_node_id = generate_node_id(from_node_type, person["text"], case_id)
            to_node_id = generate_node_id(to_node_type, event["text"], case_id)

            provenance = Provenance(
                source_chunk_id=chunk_id,
                document_id=document_id,
                page_range=page_range,
                confidence=confidence,
            )

            edge = GraphEdge(
                edge_type=EdgeType.WITNESSED,
                from_node=from_node_id,
                to_node=to_node_id,
                case_id=case_id,
                provenance=provenance,
            )
            edges.append(edge)

    return edges


def extract_found_in_edges(
    chunk_text: str,
    evidence_entities: list[dict[str, Any]],
    location_entities: list[dict[str, Any]],
    chunk_id: str,
    document_id: str,
    case_id: str,
    page_range: list[int],
    confidence: float,
) -> list[GraphEdge]:
    """
    Extract FOUND_IN edges from chunk text.

    Pattern: "[Evidence] found at/in [Location]"

    Args:
        chunk_text: Text to analyze.
        evidence_entities: Evidence entities in the chunk.
        location_entities: Location entities in the chunk.
        chunk_id: Source chunk ID.
        document_id: Source document ID.
        case_id: Case identifier.
        page_range: Page range.
        confidence: Confidence score.

    Returns:
        List of FOUND_IN edges.
    """
    edges: list[GraphEdge] = []

    if not evidence_entities or not location_entities:
        return edges

    # Check for found patterns
    found_patterns = [
        r"\b(?:found|located|discovered)\s+(?:at|in|near)\b",
    ]

    text_lower = chunk_text.lower()
    has_found_pattern = any(re.search(p, text_lower) for p in found_patterns)

    if not has_found_pattern:
        return edges

    # Connect evidence to locations
    for evidence in evidence_entities:
        for location in location_entities:
            from_node_type = get_node_type(evidence["entity_type"])
            to_node_type = get_node_type(location["entity_type"])

            from_node_id = generate_node_id(from_node_type, evidence["text"], case_id)
            to_node_id = generate_node_id(to_node_type, location["text"], case_id)

            provenance = Provenance(
                source_chunk_id=chunk_id,
                document_id=document_id,
                page_range=page_range,
                confidence=confidence,
            )

            edge = GraphEdge(
                edge_type=EdgeType.FOUND_IN,
                from_node=from_node_id,
                to_node=to_node_id,
                case_id=case_id,
                provenance=provenance,
            )
            edges.append(edge)

    return edges


def extract_accompanied_by_edges(
    chunk_text: str,
    person_entities: list[dict[str, Any]],
    chunk_id: str,
    document_id: str,
    case_id: str,
    page_range: list[int],
    confidence: float,
) -> list[GraphEdge]:
    """
    Extract ACCOMPANIED_BY edges from chunk text.

    Pattern: "[Person] with [Person]"

    Args:
        chunk_text: Text to analyze.
        person_entities: Person entities in the chunk.
        chunk_id: Source chunk ID.
        document_id: Source document ID.
        case_id: Case identifier.
        page_range: Page range.
        confidence: Confidence score.

    Returns:
        List of ACCOMPANIED_BY edges.
    """
    edges: list[GraphEdge] = []

    if len(person_entities) < 2:
        return edges

    # Check for accompanied patterns
    accompanied_patterns = [
        r"(\w+(?:\s+\w+)?)\s+(?:with|accompanied\s+by|together\s+with)\s+(\w+(?:\s+\w+)?)",
    ]

    text_lower = chunk_text.lower()

    for pattern in accompanied_patterns:
        matches = re.finditer(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            person1_text = match.group(1).strip()
            person2_text = match.group(2).strip()

            # Try to match with actual entities
            from_entity = None
            to_entity = None

            for entity in person_entities:
                entity_text_lower = entity["text"].lower()
                if person1_text in entity_text_lower or entity_text_lower in person1_text:
                    from_entity = entity
                if person2_text in entity_text_lower or entity_text_lower in person2_text:
                    to_entity = entity

            if from_entity and to_entity and from_entity != to_entity:
                from_node_type = get_node_type(from_entity["entity_type"])
                to_node_type = get_node_type(to_entity["entity_type"])

                from_node_id = generate_node_id(from_node_type, from_entity["text"], case_id)
                to_node_id = generate_node_id(to_node_type, to_entity["text"], case_id)

                provenance = Provenance(
                    source_chunk_id=chunk_id,
                    document_id=document_id,
                    page_range=page_range,
                    confidence=confidence,
                )

                edge = GraphEdge(
                    edge_type=EdgeType.ACCOMPANIED_BY,
                    from_node=from_node_id,
                    to_node=to_node_id,
                    case_id=case_id,
                    provenance=provenance,
                )
                edges.append(edge)

    return edges


def extract_owns_edges(
    chunk_text: str,
    person_entities: list[dict[str, Any]],
    evidence_entities: list[dict[str, Any]],
    chunk_id: str,
    document_id: str,
    case_id: str,
    page_range: list[int],
    confidence: float,
) -> list[GraphEdge]:
    """
    Extract OWNS edges from chunk text.

    Pattern: "[Person]'s [Evidence]" or "his/her [Evidence]"

    Args:
        chunk_text: Text to analyze.
        person_entities: Person entities in the chunk.
        evidence_entities: Evidence entities in the chunk.
        chunk_id: Source chunk ID.
        document_id: Source document ID.
        case_id: Case identifier.
        page_range: Page range.
        confidence: Confidence score.

    Returns:
        List of OWNS edges.
    """
    edges: list[GraphEdge] = []

    if not person_entities or not evidence_entities:
        return edges

    # Check for possession patterns
    possession_patterns = [
        r"\b(?:his|her|their)\s+(\w+)",
        r"(\w+(?:\s+\w+)?)'s\s+(\w+)",
        r"\b(?:owned\s+by|belongs?\s+to)\s+(\w+(?:\s+\w+)?)",
    ]

    text_lower = chunk_text.lower()
    has_possession = any(re.search(p, text_lower) for p in possession_patterns)

    if not has_possession:
        return edges

    # Simple heuristic: if possession pattern exists,
    # connect first person to evidence items
    if person_entities:
        person = person_entities[0]
        for evidence in evidence_entities:
            from_node_type = get_node_type(person["entity_type"])
            to_node_type = get_node_type(evidence["entity_type"])

            from_node_id = generate_node_id(from_node_type, person["text"], case_id)
            to_node_id = generate_node_id(to_node_type, evidence["text"], case_id)

            provenance = Provenance(
                source_chunk_id=chunk_id,
                document_id=document_id,
                page_range=page_range,
                confidence=confidence,
            )

            edge = GraphEdge(
                edge_type=EdgeType.OWNS,
                from_node=from_node_id,
                to_node=to_node_id,
                case_id=case_id,
                provenance=provenance,
            )
            edges.append(edge)

    return edges


def extract_edges_from_chunk(
    chunk: Union[Chunk, dict[str, Any]],
    entities: list[Union[ExtractedEntity, dict[str, Any]]],
) -> list[GraphEdge]:
    """
    Extract all edges from a chunk based on its text and entities.

    Args:
        chunk: Chunk from Stage 5.
        entities: All entities (will be filtered to this chunk).

    Returns:
        List of GraphEdge objects.
    """
    # Extract chunk properties
    if isinstance(chunk, dict):
        chunk_id = chunk.get("chunk_id", "")
        chunk_text = chunk.get("text", "")
        case_id = chunk.get("case_id", "")
        document_id = chunk.get("document_id", "")
        page_range = chunk.get("page_range", [1, 1])
        confidence = chunk.get("chunk_confidence", chunk.get("confidence", 0.9))
    else:
        chunk_id = chunk.chunk_id
        chunk_text = chunk.text
        case_id = chunk.case_id
        document_id = chunk.document_id
        page_range = chunk.page_range
        confidence = chunk.chunk_confidence

    # Get entities for this chunk
    chunk_entities = find_entities_in_chunk(chunk_text, entities, chunk_id)

    # Categorize entities
    person_entities = [
        e for e in chunk_entities if e["entity_type"].upper() in ("PERSON", "WITNESS", "SUSPECT")
    ]
    evidence_entities = [
        e for e in chunk_entities if e["entity_type"].upper() in ("EVIDENCE", "WEAPON", "PHONE")
    ]
    location_entities = [
        e for e in chunk_entities if e["entity_type"].upper() in ("LOCATION", "ADDRESS")
    ]
    event_entities = [e for e in chunk_entities if e["entity_type"].upper() == "TIME"]

    # Extract all edge types
    edges: list[GraphEdge] = []

    edges.extend(
        extract_argued_with_edges(
            chunk_text, person_entities, chunk_id, document_id, case_id, page_range, confidence
        )
    )

    edges.extend(
        extract_witnessed_edges(
            chunk_text,
            person_entities,
            event_entities,
            chunk_id,
            document_id,
            case_id,
            page_range,
            confidence,
        )
    )

    edges.extend(
        extract_found_in_edges(
            chunk_text,
            evidence_entities,
            location_entities,
            chunk_id,
            document_id,
            case_id,
            page_range,
            confidence,
        )
    )

    edges.extend(
        extract_accompanied_by_edges(
            chunk_text, person_entities, chunk_id, document_id, case_id, page_range, confidence
        )
    )

    edges.extend(
        extract_owns_edges(
            chunk_text,
            person_entities,
            evidence_entities,
            chunk_id,
            document_id,
            case_id,
            page_range,
            confidence,
        )
    )

    return edges


def build_edges(
    chunks: list[Union[Chunk, dict[str, Any]]],
    entities: list[Union[ExtractedEntity, dict[str, Any]]],
) -> list[GraphEdge]:
    """
    Build all edges from chunks and entities.

    Args:
        chunks: List of chunks from Stage 5.
        entities: List of entities from Stage 6.

    Returns:
        List of all GraphEdge objects.
    """
    all_edges: list[GraphEdge] = []

    for chunk in chunks:
        chunk_edges = extract_edges_from_chunk(chunk, entities)
        all_edges.extend(chunk_edges)

    # Deduplicate edges
    seen: set[tuple[str, str, str]] = set()
    unique_edges: list[GraphEdge] = []

    for edge in all_edges:
        key = (edge.from_node, edge.to_node, edge.edge_type.value)
        if key not in seen:
            seen.add(key)
            unique_edges.append(edge)

    return unique_edges


def create_edge_cypher(edge: GraphEdge) -> tuple[str, dict[str, Any]]:
    """
    Generate MERGE Cypher query for an edge.

    Uses MERGE to ensure idempotent edge creation.

    Args:
        edge: GraphEdge to create.

    Returns:
        Tuple of (query_string, parameters).
    """
    rel_type = edge.edge_type.value

    query = f"""
    MATCH (from {{node_id: $from_node, case_id: $case_id}})
    MATCH (to {{node_id: $to_node, case_id: $case_id}})
    MERGE (from)-[r:{rel_type}]->(to)
    ON CREATE SET
        r.source_chunk_id = $source_chunk_id,
        r.document_id = $document_id,
        r.page_range = $page_range,
        r.confidence = $confidence,
        r.case_id = $case_id
    RETURN type(r) as rel_type
    """

    parameters = {
        "from_node": edge.from_node,
        "to_node": edge.to_node,
        "case_id": edge.case_id,
        "source_chunk_id": edge.provenance.source_chunk_id,
        "document_id": edge.provenance.document_id,
        "page_range": edge.provenance.page_range,
        "confidence": edge.provenance.confidence,
    }

    return query, parameters
