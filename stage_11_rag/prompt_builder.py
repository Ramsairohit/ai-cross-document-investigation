"""
Stage 11: RAG - Prompt Builder

Constructs strict prompts for evidence-bound answers.

IMPORTANT:
- Evidence-only context
- Citation requirements
- No hallucination instructions
"""

from .models import GraphFact, RetrievedChunk, TimelineEvent


def build_evidence_context(
    chunks: list[RetrievedChunk],
    facts: list[GraphFact] | None = None,
    events: list[TimelineEvent] | None = None,
) -> str:
    """
    Build complete evidence context from all sources.

    Args:
        chunks: Retrieved chunks.
        facts: Graph facts.
        events: Timeline events.

    Returns:
        Formatted evidence context.
    """
    sections = []

    # Chunk evidence
    if chunks:
        chunk_section = ["RETRIEVED EVIDENCE:"]
        for i, chunk in enumerate(chunks, 1):
            speaker_info = f" (Speaker: {chunk.speaker})" if chunk.speaker else ""
            chunk_section.append(f'\n[Source {i}: {chunk.chunk_id}]{speaker_info}\n"{chunk.text}"')
        sections.append("\n".join(chunk_section))

    # Graph facts
    if facts:
        fact_section = ["\nKNOWN RELATIONSHIPS:"]
        for fact in facts:
            fact_section.append(f"- {fact.subject} {fact.predicate} {fact.object}")
        sections.append("\n".join(fact_section))

    # Timeline events
    if events:
        event_section = ["\nTIMELINE:"]
        for event in events:
            event_section.append(f"- [{event.timestamp}] {event.description}")
        sections.append("\n".join(event_section))

    if not sections:
        return "No relevant evidence found."

    return "\n".join(sections)


def build_source_mapping(chunks: list[RetrievedChunk]) -> dict[int, str]:
    """
    Build mapping from source number to chunk ID.

    Args:
        chunks: Retrieved chunks.

    Returns:
        Dict mapping source number to chunk_id.
    """
    return {i: chunk.chunk_id for i, chunk in enumerate(chunks, 1)}


def format_limitations(
    gap_limitations: list[str],
    contradiction_limitations: list[str],
    other_limitations: list[str] | None = None,
) -> list[str]:
    """
    Format all limitations for response.

    Args:
        gap_limitations: Timeline gap limitations.
        contradiction_limitations: Contradiction limitations.
        other_limitations: Other limitations.

    Returns:
        Combined unique limitations.
    """
    all_limitations = []
    all_limitations.extend(gap_limitations)
    all_limitations.extend(contradiction_limitations)

    if other_limitations:
        all_limitations.extend(other_limitations)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for lim in all_limitations:
        if lim not in seen:
            seen.add(lim)
            unique.append(lim)

    return unique


def truncate_context(
    context: str,
    max_tokens: int = 4000,
    chars_per_token: int = 4,
) -> str:
    """
    Truncate context to fit token limit.

    Args:
        context: Full context string.
        max_tokens: Maximum tokens.
        chars_per_token: Estimated characters per token.

    Returns:
        Truncated context.
    """
    max_chars = max_tokens * chars_per_token

    if len(context) <= max_chars:
        return context

    truncated = context[:max_chars]
    # Find last complete source
    last_source_end = truncated.rfind("\n[Source")
    if last_source_end > 0:
        truncated = truncated[:last_source_end]

    return truncated + "\n[Context truncated due to length]"
