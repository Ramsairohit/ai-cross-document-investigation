"""
Stage 11: RAG - Graph Lookup

Queries knowledge graph for related entities and facts.

IMPORTANT:
- READ-ONLY access to graph
- No graph algorithms
- No inference
"""

from typing import Any

from .models import GraphFact


def extract_entities_from_question(question: str) -> list[str]:
    """
    Extract potential entity names from investigator question.

    This is a simple extraction - no NLP inference.

    Args:
        question: Investigator question.

    Returns:
        List of potential entity names.
    """
    # Simple heuristic: find capitalized words that might be names
    words = question.split()
    entities = []

    for i, word in enumerate(words):
        # Clean word
        clean = word.strip("?.,!\"'")
        if not clean:
            continue

        # Skip common question words
        skip_words = {
            "Who",
            "What",
            "When",
            "Where",
            "Why",
            "How",
            "Did",
            "Does",
            "Was",
            "Were",
            "Is",
            "Are",
            "The",
            "A",
            "An",
            "To",
            "From",
            "With",
        }
        if clean in skip_words:
            continue

        # Keep capitalized words (potential names)
        if clean and clean[0].isupper():
            entities.append(clean)

    return entities


def lookup_person(
    name: str,
    graph_nodes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Find person nodes matching name.

    Args:
        name: Person name to search.
        graph_nodes: List of graph nodes.

    Returns:
        Matching person nodes.
    """
    name_lower = name.lower()
    matches = []

    for node in graph_nodes:
        node_type = node.get("type", "")
        if node_type not in ("PERSON", "WITNESS", "SUSPECT"):
            continue

        node_name = node.get("name", "").lower()
        if name_lower in node_name or node_name in name_lower:
            matches.append(node)

    return matches


def lookup_related_edges(
    entity_id: str,
    graph_edges: list[dict[str, Any]],
) -> list[GraphFact]:
    """
    Find edges involving an entity.

    Args:
        entity_id: Entity node ID.
        graph_edges: List of graph edges.

    Returns:
        List of facts as GraphFact objects.
    """
    facts = []

    for edge in graph_edges:
        source = edge.get("source_id", "")
        target = edge.get("target_id", "")

        if source == entity_id or target == entity_id:
            fact = GraphFact(
                subject=edge.get("source_name", source),
                predicate=edge.get("type", "RELATED_TO"),
                object=edge.get("target_name", target),
                source_chunk_id=edge.get("chunk_id"),
            )
            facts.append(fact)

    return facts


def lookup_graph_context(
    question: str,
    chunk_ids: list[str],
    graph_nodes: list[dict[str, Any]],
    graph_edges: list[dict[str, Any]],
) -> list[GraphFact]:
    """
    Lookup graph facts related to question and retrieved chunks.

    Args:
        question: Investigator question.
        chunk_ids: IDs of retrieved chunks.
        graph_nodes: All graph nodes.
        graph_edges: All graph edges.

    Returns:
        Relevant facts from graph.
    """
    facts: list[GraphFact] = []

    # Extract entities from question
    question_entities = extract_entities_from_question(question)

    # Find matching nodes
    matched_node_ids = set()
    for entity in question_entities:
        matches = lookup_person(entity, graph_nodes)
        for match in matches:
            matched_node_ids.add(match.get("node_id", ""))

    # Add nodes related to retrieved chunks
    for node in graph_nodes:
        if node.get("chunk_id") in chunk_ids:
            matched_node_ids.add(node.get("node_id", ""))

    # Get related edges
    for node_id in matched_node_ids:
        node_facts = lookup_related_edges(node_id, graph_edges)
        facts.extend(node_facts)

    # Deduplicate
    seen = set()
    unique_facts = []
    for fact in facts:
        key = (fact.subject, fact.predicate, fact.object)
        if key not in seen:
            seen.add(key)
            unique_facts.append(fact)

    return unique_facts


def facts_to_context(facts: list[GraphFact]) -> str:
    """
    Convert graph facts to context string.

    Args:
        facts: List of graph facts.

    Returns:
        Formatted context string.
    """
    if not facts:
        return ""

    lines = ["Related facts from evidence:"]
    for fact in facts:
        lines.append(f"- {fact.subject} {fact.predicate} {fact.object}")

    return "\n".join(lines)
