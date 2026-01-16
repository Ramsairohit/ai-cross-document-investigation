"""
Stage 2: Document Extraction - OCR Fallback

Tesseract OCR fallback for image-only pages and scanned documents.
Only processes pages explicitly flagged as needing OCR.

IMPORTANT: OCR is ONLY used when Docling reports:
- No text layer
- Image-only pages
- Low confidence text extraction

Text is extracted VERBATIM with no corrections.
"""

from pathlib import Path
from typing import Optional

import pdfplumber
import pytesseract
from PIL import Image

from .models import ContentBlock, ContentBlockType


class OCRFallback:
    """
    Tesseract OCR fallback for pages without embedded text.

    Used when Docling cannot extract text from specific pages
    (e.g., scanned documents, image-only PDFs).

    Attributes:
        language: Tesseract language code (default: "eng").
        config: Tesseract configuration options.
    """

    def __init__(self, language: str = "eng", config: Optional[str] = None) -> None:
        """
        Initialize the OCR fallback handler.

        Args:
            language: Tesseract language code (e.g., "eng", "spa", "deu").
            config: Additional Tesseract configuration options.
        """
        self.language = language
        self.config = config or "--oem 3 --psm 6"

    def ocr_page(
        self, image: Image.Image, page_num: int, start_block_id: int = 1
    ) -> list[ContentBlock]:
        """
        Perform OCR on a single page image.

        Args:
            image: PIL Image of the page.
            page_num: 1-indexed page number.
            start_block_id: Starting block ID counter.

        Returns:
            List of ContentBlock objects extracted via OCR.
        """
        content_blocks: list[ContentBlock] = []

        # Get OCR data with confidence scores
        ocr_data = pytesseract.image_to_data(
            image, lang=self.language, config=self.config, output_type=pytesseract.Output.DICT
        )

        # Extract text blocks with confidence
        current_paragraph: list[str] = []
        current_confidences: list[float] = []
        block_counter = start_block_id

        num_boxes = len(ocr_data["text"])

        for i in range(num_boxes):
            text = ocr_data["text"][i].strip()
            conf = ocr_data["conf"][i]

            # Skip empty text
            if not text:
                # End of paragraph - save if we have content
                if current_paragraph:
                    avg_conf = sum(current_confidences) / len(current_confidences)
                    content_block = ContentBlock(
                        block_id=f"b{block_counter}",
                        type=ContentBlockType.paragraph.value,
                        text=" ".join(current_paragraph),
                        page=page_num,
                        confidence=self._normalize_confidence(avg_conf),
                    )
                    content_blocks.append(content_block)
                    block_counter += 1
                    current_paragraph = []
                    current_confidences = []
                continue

            # Add text to current paragraph
            if conf >= 0:  # Valid confidence
                current_paragraph.append(text)
                current_confidences.append(conf)

        # Save any remaining paragraph
        if current_paragraph:
            avg_conf = sum(current_confidences) / len(current_confidences)
            content_block = ContentBlock(
                block_id=f"b{block_counter}",
                type=ContentBlockType.paragraph.value,
                text=" ".join(current_paragraph),
                page=page_num,
                confidence=self._normalize_confidence(avg_conf),
            )
            content_blocks.append(content_block)

        return content_blocks

    def ocr_pdf_pages(
        self, file_path: Path, pages: set[int], dpi: int = 300
    ) -> dict[int, list[ContentBlock]]:
        """
        Perform OCR on specific pages of a PDF.

        CRITICAL: Only processes pages in the 'pages' set.
        Does NOT OCR pages that already have valid text.

        Args:
            file_path: Path to the PDF file.
            pages: Set of 1-indexed page numbers to OCR.
            dpi: Resolution for rendering PDF pages (default: 300).

        Returns:
            Dictionary mapping page numbers to lists of ContentBlock.
        """
        results: dict[int, list[ContentBlock]] = {}
        block_counter = 1

        # Use pdfplumber to extract page images
        with pdfplumber.open(file_path) as pdf:
            for page_num in sorted(pages):
                # Convert to 0-indexed for pdfplumber
                page_idx = page_num - 1

                if page_idx >= len(pdf.pages):
                    continue

                page = pdf.pages[page_idx]

                # Convert page to image
                page_image = page.to_image(resolution=dpi)
                pil_image = page_image.original

                # Perform OCR on this page
                page_blocks = self.ocr_page(pil_image, page_num, start_block_id=block_counter)

                results[page_num] = page_blocks
                block_counter += len(page_blocks)

        return results

    def ocr_image_file(self, file_path: Path, page_num: int = 1) -> list[ContentBlock]:
        """
        Perform OCR on an image file (JPG, PNG, TIFF).

        Args:
            file_path: Path to the image file.
            page_num: Page number to assign (default: 1).

        Returns:
            List of ContentBlock objects extracted via OCR.
        """
        image = Image.open(file_path)
        return self.ocr_page(image, page_num)

    def _normalize_confidence(self, tesseract_conf: float) -> float:
        """
        Normalize Tesseract confidence (0-100) to 0.0-1.0 scale.

        Args:
            tesseract_conf: Tesseract confidence value (0-100).

        Returns:
            Normalized confidence (0.0-1.0).
        """
        if tesseract_conf < 0:
            return 0.0
        return min(1.0, tesseract_conf / 100.0)

    def get_full_page_text(self, image: Image.Image) -> tuple[str, float]:
        """
        Get full page text and average confidence from OCR.

        Args:
            image: PIL Image of the page.

        Returns:
            Tuple of (full_text, average_confidence).
        """
        text = pytesseract.image_to_string(image, lang=self.language, config=self.config)

        # Get confidence
        ocr_data = pytesseract.image_to_data(
            image, lang=self.language, config=self.config, output_type=pytesseract.Output.DICT
        )

        confidences = [c for c in ocr_data["conf"] if c >= 0]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

        return text.strip(), self._normalize_confidence(avg_conf)


def check_tesseract_installed() -> bool:
    """
    Check if Tesseract OCR is installed and accessible.

    Returns:
        True if Tesseract is available, False otherwise.
    """
    try:
        pytesseract.get_tesseract_version()
        return True
    except pytesseract.TesseractNotFoundError:
        return False
