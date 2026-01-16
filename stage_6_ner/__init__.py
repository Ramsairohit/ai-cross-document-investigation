"""
Stage 6: NER - Named Entity Recognition

Extract named entities from Stage 5 chunks using spaCy with rule-based extensions
while preserving provenance and avoiding inference.

HARD RULES (NON-NEGOTIABLE):
❌ No summarization
❌ No inference
❌ No contradiction detection
❌ No timeline reasoning
❌ No cross-chunk analysis
❌ No entity merging across chunks

✔ One chunk processed independently
✔ Entities always linked to chunk_id
✔ Same input → same output
"""

from .entity_extractor import extract_entities
from .models import (
    ChunkInput,
    EntityType,
    ExtractedEntity,
    ExtractionSource,
    NERResult,
)
from .ner_pipeline import (
    NERPipeline,
    process_chunk_async,
    process_chunk_sync,
    process_chunks_async,
    process_chunks_sync,
)
from .rule_based_entities import (
    extract_addresses,
    extract_all_rule_based,
    extract_evidence,
    extract_phone_numbers,
    extract_weapons,
)
from .spacy_loader import get_spacy_model, is_model_loaded

__all__ = [
    # Main API
    "NERPipeline",
    "process_chunk_async",
    "process_chunk_sync",
    "process_chunks_async",
    "process_chunks_sync",
    "extract_entities",
    # Models
    "ChunkInput",
    "EntityType",
    "ExtractedEntity",
    "ExtractionSource",
    "NERResult",
    # spaCy
    "get_spacy_model",
    "is_model_loaded",
    # Rule-based
    "extract_phone_numbers",
    "extract_addresses",
    "extract_weapons",
    "extract_evidence",
    "extract_all_rule_based",
]
