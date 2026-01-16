"""
Stage 11: RAG - Gemini LLM Client

Optional Google Gemini integration for LLM-based answer generation.

IMPORTANT:
- Requires google-generativeai package
- Uses strict forensic prompting
- API key should be set via environment variable

Usage:
    from stage_11_rag.gemini_client import create_gemini_llm

    llm_fn = create_gemini_llm(api_key="your-api-key")
    result = pipeline.answer_query(..., llm_fn=llm_fn)
"""

import os
from typing import Optional

from .llm_client import SYSTEM_PROMPT


def create_gemini_llm(
    api_key: Optional[str] = None,
    model_name: str = "gemini-pro",
) -> callable:
    """
    Create a Gemini LLM function for use with RAG pipeline.

    Args:
        api_key: Google API key. If not provided, reads from
                 GOOGLE_API_KEY environment variable.
        model_name: Gemini model name (default: gemini-pro).

    Returns:
        LLM function compatible with RAG pipeline.

    Raises:
        ImportError: If google-generativeai is not installed.
        ValueError: If no API key is provided.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package required. Install with: pip install google-generativeai"
        )

    # Get API key
    key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "API key required. Provide api_key parameter or "
            "set GOOGLE_API_KEY environment variable."
        )

    # Configure Gemini
    genai.configure(api_key=key)

    def gemini_llm(system_prompt: str, user_prompt: str) -> str:
        """
        Call Gemini with strict forensic prompting.

        Args:
            system_prompt: System instructions (strict forensic rules).
            user_prompt: User prompt with evidence context.

        Returns:
            Generated answer text.
        """
        model = genai.GenerativeModel(model_name)

        # Combine prompts for Gemini
        full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_prompt}

{user_prompt}

Remember: Cite sources as [Source N] and do not add information not in the evidence."""

        try:
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            # On error, return a safe response
            return (
                f"Unable to generate response due to an error. "
                f"Please review the source evidence directly."
            )

    return gemini_llm


def call_gemini_once(
    question: str,
    context: str,
    api_key: str,
    model_name: str = "gemini-pro",
) -> str:
    """
    One-shot Gemini call for simple use cases.

    Args:
        question: User question.
        context: Evidence context.
        api_key: Google API key.
        model_name: Model to use.

    Returns:
        Generated answer.
    """
    llm_fn = create_gemini_llm(api_key, model_name)

    from .llm_client import build_user_prompt

    user_prompt = build_user_prompt(question, context)

    return llm_fn(SYSTEM_PROMPT, user_prompt)


# Environment variable name for API key
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
