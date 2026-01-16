"""
Unit tests for Stage 10: Contradiction Detection - Pairing

Tests for entity-based chunk pairing.
"""

import pytest

from stage_10_contradictions.pairing import (
    chunks_share_entity,
    extract_chunk_reference,
    generate_candidate_pairs,
    get_chunk_id,
)


class TestExtractChunkReference:
    """Tests for chunk reference extraction."""

    def test_from_dict(self):
        """Should extract from dict."""
        chunk = {
            "chunk_id": "C1",
            "document_id": "D1",
            "page_range": [2, 2],
            "speaker": "Alice",
            "text": "Some text.",
        }
        ref = extract_chunk_reference(chunk)
        assert ref.chunk_id == "C1"
        assert ref.speaker == "Alice"


class TestChunksShareEntity:
    """Tests for entity overlap detection."""

    def test_speaker_in_text(self):
        """Should detect speaker mentioned in other text."""
        chunk_a = {
            "chunk_id": "C1",
            "speaker": "Marcus",
            "text": "I was at home.",
        }
        chunk_b = {
            "chunk_id": "C2",
            "speaker": "Julian",
            "text": "I saw Marcus at the scene.",
        }

        shares, entities = chunks_share_entity(chunk_a, chunk_b)

        assert shares is True
        assert "Marcus" in entities

    def test_no_overlap(self):
        """Should detect no overlap."""
        chunk_a = {
            "chunk_id": "C1",
            "speaker": "Alice",
            "text": "I was home.",
        }
        chunk_b = {
            "chunk_id": "C2",
            "speaker": "Bob",
            "text": "I was at work.",
        }

        shares, entities = chunks_share_entity(chunk_a, chunk_b)

        assert shares is False


class TestGenerateCandidatePairs:
    """Tests for candidate pair generation."""

    def test_same_case_only(self):
        """Should only pair chunks from same case."""
        chunks = [
            {
                "chunk_id": "C1",
                "case_id": "CASE_A",
                "speaker": "Marcus",
                "text": "Marcus was here.",
            },
            {"chunk_id": "C2", "case_id": "CASE_A", "speaker": "Julian", "text": "I saw Marcus."},
            {"chunk_id": "C3", "case_id": "CASE_B", "speaker": "Bob", "text": "I was there."},
        ]

        pairs = generate_candidate_pairs(chunks, require_entity_overlap=True)

        # Only C1 and C2 share case and entity
        for a, b, _ in pairs:
            case_a = a.get("case_id") if isinstance(a, dict) else a.case_id
            case_b = b.get("case_id") if isinstance(b, dict) else b.case_id
            assert case_a == case_b

    def test_entity_overlap_required(self):
        """With entity overlap required, no unrelated pairs."""
        chunks = [
            {"chunk_id": "C1", "case_id": "001", "speaker": "Alice", "text": "Alice was here."},
            {"chunk_id": "C2", "case_id": "001", "speaker": "Bob", "text": "Bob was there."},
        ]

        pairs = generate_candidate_pairs(chunks, require_entity_overlap=True)

        # No overlap between Alice and Bob without mentions
        assert len(pairs) == 0

    def test_deterministic_ordering(self):
        """Pairs should have deterministic ordering."""
        chunks = [
            {"chunk_id": "C2", "case_id": "001", "speaker": "Marcus", "text": "Marcus said."},
            {"chunk_id": "C1", "case_id": "001", "speaker": "Julian", "text": "I saw Marcus."},
        ]

        pairs1 = generate_candidate_pairs(chunks, require_entity_overlap=True)
        pairs2 = generate_candidate_pairs(chunks, require_entity_overlap=True)

        if pairs1:
            id1_a = get_chunk_id(pairs1[0][0])
            id1_b = get_chunk_id(pairs1[0][1])
            id2_a = get_chunk_id(pairs2[0][0])
            id2_b = get_chunk_id(pairs2[0][1])
            assert id1_a == id2_a
            assert id1_b == id2_b

    def test_no_entity_overlap_required(self):
        """Without entity requirement, all pairs generated."""
        chunks = [
            {"chunk_id": "C1", "case_id": "001", "speaker": "A", "text": "Text A"},
            {"chunk_id": "C2", "case_id": "001", "speaker": "B", "text": "Text B"},
            {"chunk_id": "C3", "case_id": "001", "speaker": "C", "text": "Text C"},
        ]

        pairs = generate_candidate_pairs(chunks, require_entity_overlap=False)

        # 3 chunks = 3 pairs (C1-C2, C1-C3, C2-C3)
        assert len(pairs) == 3

    def test_empty_chunks(self):
        """Should handle empty input."""
        pairs = generate_candidate_pairs([])
        assert pairs == []
