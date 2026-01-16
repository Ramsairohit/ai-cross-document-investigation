"""
Stage 2: Document Extraction - Main Orchestrator

Primary extraction module that coordinates Docling parsing,
OCR fallback, confidence scoring, and audit logging.

IMPORTANT: This module prepares evidence for downstream processing.
It does NOT analyze, reason, or make decisions about content.

Build it like it will be examined in court.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Literal, Optional, Union

from .audit_logger import AuditLogger
from .confidence_scoring import (
    calculate_document_confidence,
    determine_status,
)
from .docling_loader import DoclingLoader, DoclingResult
from .hash_utils import compute_blocks_hash, compute_file_hash
from .models import (
    ContentBlock,
    ExtractionRequest,
    ExtractionResult,
    ExtractionStatus,
)
from .ocr_fallback import OCRFallback, check_tesseract_installed


class DocumentExtractor:
    """
    Main document extraction orchestrator.

    Coordinates the extraction pipeline:
    1. Compute input file hash (chain of custody)
    2. Load document with Docling
    3. Identify pages needing OCR fallback
    4. Run OCR on flagged pages only
    5. Merge content blocks
    6. Calculate confidence scores
    7. Determine status (ACCEPTED/FLAGGED/REJECTED)
    8. Log audit event
    9. Return structured result

    Attributes:
        audit_logger: Chain-of-custody audit logger.
        docling_loader: Primary document loader.
        ocr_fallback: OCR handler for image-only pages.
    """

    def __init__(
        self, audit_log_dir: Path, ocr_enabled: bool = True, ocr_language: str = "eng"
    ) -> None:
        """
        Initialize the document extractor.

        Args:
            audit_log_dir: Directory for audit log files.
            ocr_enabled: Enable OCR fallback for scanned pages.
            ocr_language: OCR language code (default: "eng").
        """
        self.audit_log_dir = Path(audit_log_dir)
        self.audit_logger: Optional[AuditLogger] = None

        self.docling_loader = DoclingLoader(ocr_enabled=ocr_enabled, language=ocr_language)

        self.ocr_fallback = OCRFallback(language=ocr_language)
        self.ocr_enabled = ocr_enabled and check_tesseract_installed()

    async def extract_batch(self, request: ExtractionRequest) -> list[ExtractionResult]:
        """
        Process a batch of documents for extraction.

        Each file is processed independently. Failures in one file
        do not affect processing of other files.

        Args:
            request: Extraction request with case_id and file list.

        Returns:
            List of ExtractionResult, one per input file.
        """
        # Initialize audit logger for this case
        self.audit_logger = AuditLogger(log_dir=self.audit_log_dir, case_id=request.case_id)

        results: list[ExtractionResult] = []

        # Process each file independently
        for file_path in request.uploaded_files:
            try:
                result = await self.extract_single(
                    case_id=request.case_id, file_path=Path(file_path)
                )
                results.append(result)
            except Exception as e:
                # Log error but continue processing other files
                error_result = self._create_error_result(
                    case_id=request.case_id, file_path=Path(file_path), error=str(e)
                )
                results.append(error_result)

        return results

    async def extract_single(
        self, case_id: str, file_path: Path, operator: str = "SYSTEM"
    ) -> ExtractionResult:
        """
        Extract content from a single document.

        Args:
            case_id: Case identifier for chain-of-custody.
            file_path: Path to the document file.
            operator: Operator name for audit log.

        Returns:
            ExtractionResult with structured content.
        """
        file_path = Path(file_path)

        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Step 1: Compute input file hash (chain of custody)
        input_hash = compute_file_hash(file_path)

        # Step 2: Generate unique document ID
        document_id = self._generate_document_id()

        # Step 3: Load document with Docling
        # Run in executor to avoid blocking async loop
        loop = asyncio.get_event_loop()
        docling_result: DoclingResult = await loop.run_in_executor(
            None, self.docling_loader.load_document, file_path
        )

        # Step 4: Run OCR fallback on flagged pages if needed
        content_blocks = docling_result.content_blocks
        extraction_method: Literal["docling", "docling+ocr"] = "docling"

        if docling_result.pages_needing_ocr and self.ocr_enabled:
            # Check if file is PDF (OCR fallback for PDF pages)
            if file_path.suffix.lower() == ".pdf":
                ocr_blocks = await loop.run_in_executor(
                    None,
                    self.ocr_fallback.ocr_pdf_pages,
                    file_path,
                    docling_result.pages_needing_ocr,
                )

                # Merge OCR blocks with existing blocks
                content_blocks = self._merge_ocr_blocks(
                    docling_blocks=content_blocks, ocr_blocks=ocr_blocks
                )
                extraction_method = "docling+ocr"

        # Handle image-only files
        if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".tif"}:
            if not content_blocks and self.ocr_enabled:
                image_blocks = await loop.run_in_executor(
                    None, self.ocr_fallback.ocr_image_file, file_path
                )
                content_blocks = image_blocks
                extraction_method = "docling+ocr"

        # Step 5: Renumber blocks sequentially
        content_blocks = self._renumber_blocks(content_blocks)

        # Step 6: Calculate confidence scores
        overall_confidence = calculate_document_confidence(content_blocks)

        # Step 7: Determine status based on confidence
        status = determine_status(overall_confidence)

        # Step 8: Compute output hash
        blocks_data = [block.model_dump() for block in content_blocks]
        output_hash = compute_blocks_hash(blocks_data)

        # Step 9: Log audit event
        if self.audit_logger:
            audit_event = self.audit_logger.create_extraction_event(
                case_id=case_id,
                document_id=document_id,
                input_hash=input_hash,
                output_hash=output_hash,
                tool=extraction_method,
                operator=operator,
            )
            self.audit_logger.log_extraction(audit_event)

        # Step 10: Create and return result
        return ExtractionResult(
            document_id=document_id,
            case_id=case_id,
            source_file=file_path.name,
            pages=docling_result.page_count,
            extraction_method=extraction_method,
            content_blocks=content_blocks,
            overall_confidence=round(overall_confidence, 2),
            status=status,
        )

    def _generate_document_id(self) -> str:
        """
        Generate a unique document ID.

        Returns:
            UUID-based document identifier.
        """
        return f"doc-{uuid.uuid4()}"

    def _merge_ocr_blocks(
        self, docling_blocks: list[ContentBlock], ocr_blocks: dict[int, list[ContentBlock]]
    ) -> list[ContentBlock]:
        """
        Merge OCR blocks into Docling blocks.

        Replaces low-confidence Docling blocks with OCR results
        for pages that needed OCR fallback.

        Args:
            docling_blocks: Original blocks from Docling.
            ocr_blocks: OCR results keyed by page number.

        Returns:
            Merged list of content blocks.
        """
        # Pages that have OCR results
        ocr_pages = set(ocr_blocks.keys())

        # Keep Docling blocks from pages that didn't need OCR
        result_blocks: list[ContentBlock] = [
            block for block in docling_blocks if block.page not in ocr_pages
        ]

        # Add OCR blocks for pages that needed fallback
        for page_num in sorted(ocr_pages):
            if page_num in ocr_blocks:
                result_blocks.extend(ocr_blocks[page_num])

        # Sort by page number
        result_blocks.sort(key=lambda b: (b.page, b.block_id))

        return result_blocks

    def _renumber_blocks(self, blocks: list[ContentBlock]) -> list[ContentBlock]:
        """
        Renumber blocks sequentially after merging.

        Args:
            blocks: List of content blocks.

        Returns:
            Blocks with sequential block_ids (b1, b2, b3, ...).
        """
        renumbered: list[ContentBlock] = []

        for idx, block in enumerate(blocks, start=1):
            renumbered_block = ContentBlock(
                block_id=f"b{idx}",
                type=block.type,
                text=block.text,
                page=block.page,
                confidence=block.confidence,
            )
            renumbered.append(renumbered_block)

        return renumbered

    def _create_error_result(self, case_id: str, file_path: Path, error: str) -> ExtractionResult:
        """
        Create an error result for failed extraction.

        Args:
            case_id: Case identifier.
            file_path: Path to the failed file.
            error: Error message.

        Returns:
            ExtractionResult with REJECTED status.
        """
        return ExtractionResult(
            document_id=self._generate_document_id(),
            case_id=case_id,
            source_file=file_path.name,
            pages=0,
            extraction_method="docling",
            content_blocks=[
                ContentBlock(
                    block_id="b1",
                    type="error",
                    text=f"Extraction failed: {error}",
                    page=1,
                    confidence=0.0,
                )
            ],
            overall_confidence=0.0,
            status=ExtractionStatus.REJECTED,
        )


# Convenience function for simple usage
async def extract_documents(
    case_id: str,
    file_paths: list[Union[str, Path]],
    audit_log_dir: Path = Path("./audit_logs"),
    ocr_enabled: bool = True,
) -> list[ExtractionResult]:
    """
    Convenience function to extract documents.

    Args:
        case_id: Case identifier.
        file_paths: List of file paths to process.
        audit_log_dir: Directory for audit logs.
        ocr_enabled: Enable OCR fallback.

    Returns:
        List of ExtractionResult objects.
    """
    extractor = DocumentExtractor(audit_log_dir=audit_log_dir, ocr_enabled=ocr_enabled)

    request = ExtractionRequest(case_id=case_id, uploaded_files=[str(p) for p in file_paths])

    return await extractor.extract_batch(request)
