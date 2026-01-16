"""
Stage 11: RAG - LLM Client

Restricted LLM wrapper with strict prompting.

IMPORTANT:
- Evidence-only context
- Strict system instructions
- No hallucination allowed
"""

from typing import Any, Optional


# Strict system prompt for forensic-grade RAG
SYSTEM_PROMPT = """You are a forensic evidence reporting system for law enforcement investigations.

CRITICAL RULES:
1. You MUST answer ONLY using the provided evidence.
2. You MUST NOT add facts that are not in the evidence.
3. You MUST NOT guess, assume, or infer beyond what is explicitly stated.
4. You MUST cite sources for every factual claim using [Source N] format.
5. If evidence is insufficient to answer, say so clearly.
6. You MUST NOT use probabilistic language like "likely", "probably", "might have".
7. You MUST NOT make guilt determinations or judgments.
8. You MUST NOT resolve contradictions - report them as found.

Your answers will be used in legal proceedings. Accuracy and citation are mandatory."""


def build_user_prompt(
    question: str,
    context: str,
    limitations: list[str] | None = None,
) -> str:
    """
    Build user prompt with evidence context.

    Args:
        question: Investigator question.
        context: Evidence context from retrieval.
        limitations: Known limitations to include.

    Returns:
        Formatted user prompt.
    """
    prompt_parts = [
        "EVIDENCE CONTEXT:",
        context,
        "",
        "QUESTION:",
        question,
        "",
        "Provide a factual answer citing the sources provided. "
        "Each claim must reference a [Source N].",
    ]

    if limitations:
        prompt_parts.append("")
        prompt_parts.append("KNOWN LIMITATIONS:")
        for lim in limitations:
            prompt_parts.append(f"- {lim}")

    return "\n".join(prompt_parts)


def generate_answer(
    question: str,
    context: str,
    limitations: list[str] | None = None,
    llm_fn: Optional[callable] = None,
) -> str:
    """
    Generate answer using LLM with strict context.

    If no llm_fn provided, attempts to use Gemini if available.

    Args:
        question: Investigator question.
        context: Evidence context.
        limitations: Known limitations.
        llm_fn: LLM function to call (for testing/mocking).

    Returns:
        Generated answer text.
    """
    user_prompt = build_user_prompt(question, context, limitations)

    if llm_fn is not None:
        return llm_fn(SYSTEM_PROMPT, user_prompt)

    # Try Gemini if available
    try:
        from .gemini_client import create_gemini_llm
        import os

        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            gemini_fn = create_gemini_llm(api_key)
            return gemini_fn(SYSTEM_PROMPT, user_prompt)
    except (ImportError, Exception):
        pass

    # Default stub response when no LLM available
    return generate_stub_answer(question, context, limitations)


def generate_stub_answer(
    question: str,
    context: str,
    limitations: list[str] | None = None,
) -> str:
    """
    Generate stub answer when no LLM available.

    This creates a template answer based on context.

    Args:
        question: Question asked.
        context: Evidence context.
        limitations: Known limitations.

    Returns:
        Stub answer with citations.
    """
    if not context or context == "No relevant evidence found.":
        return "The available evidence does not contain sufficient information to answer this question."

    # Extract source count
    source_count = context.count("[Source ")

    if source_count == 0:
        return "The available evidence does not contain sufficient information to answer this question."

    answer_parts = [
        f"Based on the available evidence ({source_count} source(s) reviewed):",
        "",
        "The evidence shows the following relevant information:",
    ]

    # Reference available sources
    for i in range(1, min(source_count + 1, 4)):
        answer_parts.append(f"- See [Source {i}] for details.")

    if limitations:
        answer_parts.append("")
        answer_parts.append("Note: " + "; ".join(limitations[:2]))

    return "\n".join(answer_parts)


def calculate_answer_confidence(
    sources_count: int,
    has_contradictions: bool,
    has_gaps: bool,
) -> float:
    """
    Calculate confidence score for answer.

    Args:
        sources_count: Number of sources cited.
        has_contradictions: Whether contradictions exist.
        has_gaps: Whether timeline gaps exist.

    Returns:
        Confidence score 0.0-1.0.
    """
    if sources_count == 0:
        return 0.0

    # Base confidence from source count
    confidence = min(0.9, 0.5 + sources_count * 0.1)

    # Reduce for contradictions
    if has_contradictions:
        confidence *= 0.7

    # Reduce for gaps
    if has_gaps:
        confidence *= 0.9

    return round(confidence, 2)
