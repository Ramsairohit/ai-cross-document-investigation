"""
Stage 2: Document Extraction - Docling Loader

Primary document parser using Docling for PDF, DOCX, and image extraction.
Docling provides AI-powered layout analysis and text extraction.

IMPORTANT: Text is extracted VERBATIM with no modifications.
Page numbers and document structure are preserved.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from .models import ContentBlock, ContentBlockType


@dataclass
class DoclingResult:
    """
    Result of Docling document extraction.

    Contains extracted content blocks, page metadata,
    and information about which pages may need OCR fallback.
    """

    content_blocks: list[ContentBlock] = field(default_factory=list)
    page_count: int = 0
    pages_needing_ocr: set[int] = field(default_factory=set)
    raw_confidence: dict[str, float] = field(default_factory=dict)
    source_file: str = ""
    extraction_method: str = "docling"


class DoclingLoader:
    """
    Primary document loader using Docling.

    Extracts text, tables, and structural elements from documents
    while preserving page numbers and layout information.

    Attributes:
        ocr_enabled: Whether to enable OCR for scanned content.
        language: OCR language setting (default: "eng").
    """

    def __init__(self, ocr_enabled: bool = True, language: str = "eng") -> None:
        """
        Initialize the Docling loader.

        Args:
            ocr_enabled: Enable OCR for scanned/image-only pages.
            language: OCR language (ISO 639-3 code, e.g., "eng", "spa").
        """
        self.ocr_enabled = ocr_enabled
        self.language = language
        self._converter: Optional[DocumentConverter] = None

    def _create_converter(self) -> DocumentConverter:
        """
        Create and configure the Docling DocumentConverter.

        Returns:
            Configured DocumentConverter instance.
        """
        # Configure PDF pipeline options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = self.ocr_enabled
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)

        # Use Tesseract for OCR (per requirements)
        if self.ocr_enabled:
            pipeline_options.ocr_options = TesseractOcrOptions(lang=[self.language])

        # Create converter with PDF options
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        return converter

    def load_document(self, file_path: Path) -> DoclingResult:
        """
        Load and extract content from a document.

        Args:
            file_path: Path to the document file.

        Returns:
            DoclingResult containing extracted content and metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        # Create converter if not already created
        if self._converter is None:
            self._converter = self._create_converter()

        # Convert document
        conversion_result = self._converter.convert(file_path)

        # Extract content blocks from the converted document
        content_blocks, pages_needing_ocr = self._extract_content_blocks(conversion_result)

        # Get page count
        page_count = self._get_page_count(conversion_result)

        # Get raw confidence scores if available
        raw_confidence = self._extract_confidence_scores(conversion_result)

        # Determine extraction method
        extraction_method = "docling+ocr" if pages_needing_ocr else "docling"

        return DoclingResult(
            content_blocks=content_blocks,
            page_count=page_count,
            pages_needing_ocr=pages_needing_ocr,
            raw_confidence=raw_confidence,
            source_file=file_path.name,
            extraction_method=extraction_method,
        )

    def _extract_content_blocks(
        self, conversion_result: Any
    ) -> tuple[list[ContentBlock], set[int]]:
        """
        Extract content blocks from Docling conversion result.

        Args:
            conversion_result: Docling ConversionResult object.

        Returns:
            Tuple of (content_blocks, pages_needing_ocr).
        """
        content_blocks: list[ContentBlock] = []
        pages_needing_ocr: set[int] = set()
        block_counter = 0

        doc = conversion_result.document

        # Iterate through document items
        for item, _ in doc.iterate_items():
            block_counter += 1
            block_id = f"b{block_counter}"

            # Determine block type
            block_type = self._get_block_type(item)

            # Get text content
            text = self._get_item_text(item)
            if not text or not text.strip():
                continue

            # Get page number (1-indexed)
            page_num = self._get_page_number(item)

            # Get confidence (if available)
            confidence = self._get_item_confidence(item)

            # Check if this looks like it needs OCR
            if confidence is not None and confidence < 0.5:
                pages_needing_ocr.add(page_num)

            content_block = ContentBlock(
                block_id=block_id,
                type=block_type,
                text=text,  # Verbatim - no modifications
                page=page_num,
                confidence=confidence if confidence is not None else 0.90,
            )
            content_blocks.append(content_block)

        return content_blocks, pages_needing_ocr

    def _get_block_type(self, item: Any) -> str:
        """
        Determine the content block type from Docling item.

        Args:
            item: Docling document item.

        Returns:
            Block type string.
        """
        # Get the item's label/type
        item_type = getattr(item, "label", None)
        if item_type is None:
            item_type = type(item).__name__

        item_type_str = str(item_type).lower()

        # Map Docling types to our types
        type_mapping = {
            "paragraph": ContentBlockType.paragraph.value,
            "text": ContentBlockType.paragraph.value,
            "heading": ContentBlockType.heading.value,
            "title": ContentBlockType.heading.value,
            "section_header": ContentBlockType.heading.value,
            "section-header": ContentBlockType.heading.value,
            "table": ContentBlockType.table.value,
            "list_item": ContentBlockType.list_item.value,
            "list-item": ContentBlockType.list_item.value,
            "listitem": ContentBlockType.list_item.value,
            "caption": ContentBlockType.caption.value,
            "footnote": ContentBlockType.footnote.value,
            "code": ContentBlockType.code.value,
            "formula": ContentBlockType.formula.value,
            "equation": ContentBlockType.formula.value,
        }

        for key, value in type_mapping.items():
            if key in item_type_str:
                return value

        return ContentBlockType.paragraph.value

    def _get_item_text(self, item: Any) -> str:
        """
        Extract text content from a Docling item.

        Args:
            item: Docling document item.

        Returns:
            Text content (verbatim, unmodified).
        """
        # Try different attribute names used by Docling
        if hasattr(item, "text"):
            return str(item.text)
        elif hasattr(item, "content"):
            return str(item.content)
        elif hasattr(item, "export_to_markdown"):
            return item.export_to_markdown()
        else:
            return str(item)

    def _get_page_number(self, item: Any) -> int:
        """
        Get the page number for a Docling item.

        Args:
            item: Docling document item.

        Returns:
            1-indexed page number.
        """
        # Try to get page from provenance
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                if hasattr(prov, "page_no"):
                    return prov.page_no
                elif hasattr(prov, "page"):
                    return prov.page

        # Try direct page attribute
        if hasattr(item, "page_no"):
            return item.page_no
        elif hasattr(item, "page"):
            return item.page

        # Default to page 1
        return 1

    def _get_item_confidence(self, item: Any) -> Optional[float]:
        """
        Get confidence score for a Docling item.

        Args:
            item: Docling document item.

        Returns:
            Confidence score (0.0-1.0) or None if not available.
        """
        # Try to get confidence from provenance
        if hasattr(item, "prov") and item.prov:
            for prov in item.prov:
                if hasattr(prov, "confidence"):
                    return float(prov.confidence)

        # Try direct confidence attribute
        if hasattr(item, "confidence"):
            return float(item.confidence)

        return None

    def _get_page_count(self, conversion_result: Any) -> int:
        """
        Get total page count from conversion result.

        Args:
            conversion_result: Docling ConversionResult object.

        Returns:
            Total number of pages.
        """
        doc = conversion_result.document

        # Try to get page count from document
        if hasattr(doc, "pages") and doc.pages:
            return len(doc.pages)

        # Try to get from metadata
        if hasattr(doc, "num_pages"):
            return doc.num_pages

        # Infer from content blocks
        max_page = 1
        for item, _ in doc.iterate_items():
            page_num = self._get_page_number(item)
            max_page = max(max_page, page_num)

        return max_page

    def _extract_confidence_scores(self, conversion_result: Any) -> dict[str, float]:
        """
        Extract raw confidence scores from Docling result.

        Args:
            conversion_result: Docling ConversionResult object.

        Returns:
            Dictionary of confidence scores by category.
        """
        confidence_scores: dict[str, float] = {}

        # Try to get confidence report from Docling
        if hasattr(conversion_result, "confidence"):
            conf = conversion_result.confidence
            if hasattr(conf, "layout_score"):
                confidence_scores["layout_score"] = float(conf.layout_score)
            if hasattr(conf, "ocr_score"):
                confidence_scores["ocr_score"] = float(conf.ocr_score)
            if hasattr(conf, "parse_score"):
                confidence_scores["parse_score"] = float(conf.parse_score)
            if hasattr(conf, "table_score"):
                confidence_scores["table_score"] = float(conf.table_score)

        return confidence_scores
