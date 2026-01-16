"""
Stage 11 RAG - Example Usage with Google Gemini API

This script demonstrates how to use the RAG system with your API key.

IMPORTANT: Set your API key before running:
    - Set GOOGLE_API_KEY environment variable, OR
    - Pass api_key directly to create_gemini_llm()

Usage:
    python examples/rag_example.py
"""

import os

# Set your API key (or set GOOGLE_API_KEY environment variable)
os.environ["GOOGLE_API_KEY"] = "AIzaSyCU8aQMFWrnmM0rfUl1bLLWb5cOUeQzSkU"

import numpy as np

# Import RAG components
from stage_11_rag import (
    RAGPipeline,
    RAGQuery,
    RAGConfig,
    create_gemini_llm,
)


def create_mock_index_and_data():
    """Create mock FAISS index and data for demonstration."""
    import faiss

    # Create simple index
    dimension = 384
    index = faiss.IndexFlatL2(dimension)

    # Add some sample vectors
    np.random.seed(42)
    vectors = np.random.rand(5, dimension).astype(np.float32)
    index.add(vectors)

    # Sample evidence chunks
    chunks = [
        {
            "chunk_id": "CHUNK_001",
            "case_id": "24-890-H",
            "document_id": "W001-24-890-H",
            "page_range": [2, 2],
            "text": "I spoke with Julian Thorne at approximately 9:10 PM on March 15th.",
            "speaker": "Clara Higgins",
        },
        {
            "chunk_id": "CHUNK_002",
            "case_id": "24-890-H",
            "document_id": "W001-24-890-H",
            "page_range": [3, 3],
            "text": "Julian mentioned he was going to the park to meet someone.",
            "speaker": "Clara Higgins",
        },
        {
            "chunk_id": "CHUNK_003",
            "case_id": "24-890-H",
            "document_id": "W002-24-890-H",
            "page_range": [1, 1],
            "text": "At 9:30 PM, I saw Marcus Vane near the crime scene.",
            "speaker": "Officer Bennett",
        },
        {
            "chunk_id": "CHUNK_004",
            "case_id": "24-890-H",
            "document_id": "W002-24-890-H",
            "page_range": [2, 2],
            "text": "Marcus stated he was at home all evening.",
            "speaker": "Officer Bennett",
        },
        {
            "chunk_id": "CHUNK_005",
            "case_id": "24-890-H",
            "document_id": "E001-24-890-H",
            "page_range": [1, 1],
            "text": "Forensic evidence shows victim was last seen at 9:15 PM.",
            "speaker": None,
        },
    ]

    return index, chunks


def mock_embedder(text: str) -> np.ndarray:
    """Mock embedder function for demonstration."""
    np.random.seed(hash(text) % 2**32)
    return np.random.rand(384).astype(np.float32)


def main():
    print("=" * 60)
    print("Stage 11 RAG - Forensic Evidence Q&A with Gemini")
    print("=" * 60)

    # Create mock data
    index, chunks = create_mock_index_and_data()

    # Initialize pipeline
    config = RAGConfig(top_k=3)
    pipeline = RAGPipeline(config)

    # Create Gemini LLM function
    try:
        llm_fn = create_gemini_llm()
        print("\n✅ Gemini API connected successfully!")
    except Exception as e:
        print(f"\n⚠️ Gemini not available: {e}")
        print("Using stub responses instead.")
        llm_fn = None

    # Example query
    query = RAGQuery(case_id="24-890-H", question="Who last spoke to Julian Thorne and when?")

    print(f"\n📋 Case: {query.case_id}")
    print(f"❓ Question: {query.question}")
    print("-" * 60)

    # Get answer
    result = pipeline.answer_query(
        query=query,
        index=index,
        chunk_metadata=chunks,
        embedder_fn=mock_embedder,
        llm_fn=llm_fn,
    )

    # Display results
    print(f"\n📝 ANSWER:")
    print(result.answer)
    print(f"\n📊 Confidence: {result.confidence:.0%}")

    print(f"\n📚 SOURCES ({len(result.sources)}):")
    for i, source in enumerate(result.sources, 1):
        print(f"   [{i}] {source.chunk_id} - {source.document_id}")
        print(f'       "{source.excerpt[:80]}..."')

    if result.limitations:
        print(f"\n⚠️ LIMITATIONS:")
        for lim in result.limitations:
            print(f"   - {lim}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
