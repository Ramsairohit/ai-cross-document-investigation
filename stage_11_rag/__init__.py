"""
Stage 11: Retrieval-Augmented Generation (RAG)

Answer investigator questions using only existing evidence
with full citations and source traceability.

HARD RULES (NON-NEGOTIABLE):
❌ No hallucination
❌ No summarization without citation
❌ No resolving contradictions
❌ No guilt judgments
❌ No probabilistic language
❌ No cross-case access

✅ Evidence-bound answers only
✅ Full source traceability
✅ Deterministic output
✅ Contradictions in limitations

This stage speaks for the evidence - it does NOT think, decide, or judge.
"""

from .contradiction_checker import (
    check_contradictions,
    contradictions_to_limitations,
    find_related_contradictions,
    has_critical_contradictions,
)
from .graph_lookup import (
    extract_entities_from_question,
    facts_to_context,
    lookup_graph_context,
    lookup_person,
    lookup_related_edges,
)
from .llm_client import (
    SYSTEM_PROMPT,
    build_user_prompt,
    calculate_answer_confidence,
    generate_answer,
    generate_stub_answer,
)
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
    build_source_mapping,
    format_limitations,
    truncate_context,
)
from .rag_pipeline import (
    RAGPipeline,
    answer_query_async,
    answer_query_sync,
)
from .retriever import (
    chunks_to_context,
    embed_query,
    filter_by_case,
    retrieve_chunks,
    search_index,
)
from .timeline_checker import (
    detect_timeline_gaps,
    events_to_context,
    find_conflicting_timestamps,
    find_relevant_events,
    get_event_timestamps,
)

# Optional Gemini integration (requires google-generativeai)
try:
    from .gemini_client import (
        GOOGLE_API_KEY_ENV,
        call_gemini_once,
        create_gemini_llm,
    )

    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False
    create_gemini_llm = None
    call_gemini_once = None
    GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

__all__ = [
    # Main Pipeline API
    "RAGPipeline",
    "answer_query_sync",
    "answer_query_async",
    # Models
    "RAGQuery",
    "RAGAnswer",
    "RAGConfig",
    "SourceReference",
    "RetrievedChunk",
    "GraphFact",
    "TimelineEvent",
    "INSUFFICIENT_EVIDENCE_ANSWER",
    # Retriever
    "search_index",
    "retrieve_chunks",
    "embed_query",
    "filter_by_case",
    "chunks_to_context",
    # Graph Lookup
    "extract_entities_from_question",
    "lookup_person",
    "lookup_related_edges",
    "lookup_graph_context",
    "facts_to_context",
    # Timeline
    "find_relevant_events",
    "detect_timeline_gaps",
    "events_to_context",
    "get_event_timestamps",
    "find_conflicting_timestamps",
    # Contradictions
    "find_related_contradictions",
    "contradictions_to_limitations",
    "check_contradictions",
    "has_critical_contradictions",
    # LLM
    "SYSTEM_PROMPT",
    "build_user_prompt",
    "generate_answer",
    "generate_stub_answer",
    "calculate_answer_confidence",
    # Prompt Builder
    "build_evidence_context",
    "build_source_mapping",
    "format_limitations",
    "truncate_context",
    # Gemini (optional)
    "create_gemini_llm",
    "call_gemini_once",
    "GOOGLE_API_KEY_ENV",
]
