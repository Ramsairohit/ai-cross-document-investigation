"""
Stage 11: RAG - Main Pipeline

Orchestrates the RAG pipeline following MANDATORY ORDER:
1. Vector Search (FAISS)
2. Graph Lookup (Neo4j)
3. Timeline Check
4. Contradiction Awareness
5. LLM with STRICT CONTEXT

IMPORTANT:
- Evidence-bound answers only
- Full source traceability
- Deterministic output
"""

from typing import Any, Optional

import numpy as np

from .contradiction_checker import check_contradictions, has_critical_contradictions
from .graph_lookup import facts_to_context, lookup_graph_context
from .llm_client import calculate_answer_confidence, generate_answer
from .models import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    GraphFact,
    RAGAnswer,
    RAGConfig,
    RAGQuery,
    RetrievedChunk,
    SourceReference,
    TimelineEvent,
)
from .prompt_builder import (
    build_evidence_context,
    format_limitations,
    truncate_context,
)
from .retriever import filter_by_case, retrieve_chunks
from .timeline_checker import (
    detect_timeline_gaps,
    events_to_context,
    find_conflicting_timestamps,
    find_relevant_events,
)


class RAGPipeline:
    """
    Main RAG pipeline orchestrator.

    Follows MANDATORY ORDER for evidence retrieval and synthesis.
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            config: Pipeline configuration.
        """
        self._config = config or RAGConfig()

    def answer_query(
        self,
        query: RAGQuery,
        index: Any,
        chunk_metadata: list[dict[str, Any]],
        embedder_fn: callable,
        graph_nodes: list[dict[str, Any]] | None = None,
        graph_edges: list[dict[str, Any]] | None = None,
        timeline_events: list[dict[str, Any]] | None = None,
        timeline_gaps: list[dict[str, Any]] | None = None,
        contradictions: list[dict[str, Any]] | None = None,
        llm_fn: Optional[callable] = None,
    ) -> RAGAnswer:
        """
        Answer investigator query using evidence.

        MANDATORY ORDER:
        1. Vector Search
        2. Graph Lookup
        3. Timeline Check
        4. Contradiction Awareness
        5. LLM Synthesis

        Args:
            query: Investigator query.
            index: FAISS index.
            chunk_metadata: Chunk metadata list.
            embedder_fn: Function to embed text.
            graph_nodes: Optional graph nodes.
            graph_edges: Optional graph edges.
            timeline_events: Optional timeline events.
            timeline_gaps: Optional timeline gaps.
            contradictions: Optional contradictions.
            llm_fn: Optional LLM function.

        Returns:
            Evidence-based answer with citations.
        """
        # STEP 1: Vector Search (FAISS)
        query_embedding = embedder_fn(query.question)
        chunks = retrieve_chunks(query_embedding, index, chunk_metadata, self._config)

        # Filter to query case only (NO cross-case access)
        chunks = filter_by_case(chunks, query.case_id)

        if not chunks:
            return INSUFFICIENT_EVIDENCE_ANSWER

        chunk_ids = [c.chunk_id for c in chunks]

        # STEP 2: Graph Lookup (if available)
        facts: list[GraphFact] = []
        if self._config.include_graph and graph_nodes and graph_edges:
            facts = lookup_graph_context(query.question, chunk_ids, graph_nodes, graph_edges)

        # STEP 3: Timeline Check (if available)
        events: list[TimelineEvent] = []
        gap_limitations: list[str] = []
        conflict_limitations: list[str] = []

        if self._config.include_timeline and timeline_events:
            events = find_relevant_events(query.question, timeline_events, chunk_ids)

            if timeline_gaps:
                gap_limitations = detect_timeline_gaps(events, timeline_gaps)

            # Also check for timeline conflicts
            if timeline_events:
                conflict_limitations = find_conflicting_timestamps(events, timeline_events)

        # STEP 4: Contradiction Awareness (if available)
        contradiction_limitations: list[str] = []
        has_critical = False

        if self._config.include_contradictions and contradictions:
            contradiction_limitations = check_contradictions(chunks, contradictions)
            has_critical = has_critical_contradictions(chunk_ids, contradictions)

        # Combine limitations
        all_limitations = format_limitations(
            gap_limitations, contradiction_limitations, conflict_limitations
        )

        # STEP 5: LLM with STRICT CONTEXT
        context = build_evidence_context(chunks, facts, events)
        context = truncate_context(context, self._config.max_context_tokens)

        answer_text = generate_answer(query.question, context, all_limitations, llm_fn)

        # Build source references
        sources = [
            SourceReference(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page_range=chunk.page_range,
                excerpt=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                speaker=chunk.speaker,
            )
            for chunk in chunks
        ]

        # Calculate confidence
        confidence = calculate_answer_confidence(
            sources_count=len(sources),
            has_contradictions=len(contradiction_limitations) > 0,
            has_gaps=len(gap_limitations) > 0,
        )

        # Reduce confidence further for critical contradictions
        if has_critical:
            confidence *= 0.5

        return RAGAnswer(
            answer=answer_text,
            confidence=confidence,
            sources=sources,
            limitations=all_limitations,
            query=query.question,
        )

    def verify_determinism(
        self,
        query: RAGQuery,
        index: Any,
        chunk_metadata: list[dict[str, Any]],
        embedder_fn: callable,
        runs: int = 100,
    ) -> bool:
        """
        Verify that pipeline is deterministic.

        Args:
            query: Test query.
            index: FAISS index.
            chunk_metadata: Chunk metadata.
            embedder_fn: Embedding function.
            runs: Number of runs.

        Returns:
            True if all runs produce identical results.
        """
        results = []
        for _ in range(runs):
            result = self.answer_query(query, index, chunk_metadata, embedder_fn)
            snapshot = (
                result.answer,
                result.confidence,
                len(result.sources),
                tuple(result.limitations),
            )
            results.append(snapshot)

        first = results[0]
        for result in results[1:]:
            if result != first:
                return False
        return True


def answer_query_sync(
    case_id: str,
    question: str,
    index: Any,
    chunk_metadata: list[dict[str, Any]],
    embedder_fn: callable,
    config: RAGConfig | None = None,
    **kwargs: Any,
) -> RAGAnswer:
    """
    Synchronous query answering.

    Convenience function for answering queries.

    Args:
        case_id: Case identifier.
        question: Investigator question.
        index: FAISS index.
        chunk_metadata: Chunk metadata.
        embedder_fn: Embedding function.
        config: Optional configuration.
        **kwargs: Additional arguments for pipeline.

    Returns:
        Evidence-based answer.
    """
    pipeline = RAGPipeline(config)
    query = RAGQuery(case_id=case_id, question=question)
    return pipeline.answer_query(query, index, chunk_metadata, embedder_fn, **kwargs)


async def answer_query_async(
    case_id: str,
    question: str,
    index: Any,
    chunk_metadata: list[dict[str, Any]],
    embedder_fn: callable,
    config: RAGConfig | None = None,
    **kwargs: Any,
) -> RAGAnswer:
    """
    Async-safe query answering.

    Args:
        case_id: Case identifier.
        question: Investigator question.
        index: FAISS index.
        chunk_metadata: Chunk metadata.
        embedder_fn: Embedding function.
        config: Optional configuration.
        **kwargs: Additional arguments.

    Returns:
        Evidence-based answer.
    """
    pipeline = RAGPipeline(config)
    query = RAGQuery(case_id=case_id, question=question)
    return pipeline.answer_query(query, index, chunk_metadata, embedder_fn, **kwargs)
