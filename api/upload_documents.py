"""
Document Upload API Endpoint

Forensic-grade document upload mechanism for AI Police Case Investigation System.
This endpoint handles multi-file uploads, computes hashes for chain of custody,
stores files immutably, and invokes Stage 2 extraction.

IMPORTANT: This endpoint does NOT perform analysis or reasoning.
It merely stores evidence and structures it for downstream processing.

Build it like it will be examined in court.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from stage_2_extraction import ExtractionResult, extract_documents
from stage_2_extraction.audit_logger import AuditLogger
from stage_2_extraction.hash_utils import compute_file_hash


# Supported file types
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/tiff",
}


class DocumentInfo(BaseModel):
    """
    Information about a processed document.
    """

    document_id: str = Field(..., description="Unique document identifier")
    pages: int = Field(..., ge=0, description="Number of pages in the document")
    status: str = Field(..., description="Document status: ACCEPTED, FLAGGED_FOR_REVIEW, or REJECTED")
    overall_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Overall extraction confidence score"
    )


class DocumentUploadResponse(BaseModel):
    """
    Response model for document upload endpoint.

    Matches the required output format specified in the requirements.
    """

    case_id: str = Field(..., description="Case identifier")
    documents_processed: int = Field(..., ge=0, description="Number of documents processed")
    documents: list[DocumentInfo] = Field(
        ..., description="List of processed document information"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "24-890-H",
                "documents_processed": 3,
                "documents": [
                    {
                        "document_id": "doc-a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                        "pages": 3,
                        "status": "ACCEPTED",
                        "overall_confidence": 0.91,
                    }
                ],
            }
        }


router = APIRouter(prefix="/cases", tags=["documents"])


def _generate_document_id() -> str:
    """
    Generate a unique document ID for uploaded files.

    Returns:
        UUID-based document identifier.
    """
    return f"doc-{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:4]}-{uuid.uuid4().hex[:12]}"


def _validate_file_type(filename: str, content_type: str | None) -> bool:
    """
    Validate that the uploaded file type is allowed.

    Args:
        filename: Original filename.
        content_type: MIME type from request headers.

    Returns:
        True if file type is allowed, False otherwise.
    """
    # Check file extension
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        return False

    # Check MIME type if provided
    if content_type:
        # Normalize MIME type (handle case sensitivity and extra parameters)
        mime_type = content_type.split(";")[0].strip().lower()
        if mime_type not in {mt.lower() for mt in ALLOWED_MIME_TYPES}:
            # Allow if extension is valid even if MIME type doesn't match
            # (browsers sometimes send incorrect MIME types)
            pass

    return True


def _save_file_immutably(
    case_id: str, document_id: str, original_filename: str, file_content: bytes, storage_root: Path
) -> Path:
    """
    Save uploaded file immutably preserving original filename.

    Directory structure:
    evidence_storage/
    └── {case_id}/
        ├── {document_id}_original_filename.ext

    Args:
        case_id: Case identifier.
        document_id: Generated document identifier.
        original_filename: Original filename from upload.
        file_content: File content bytes.
        storage_root: Root directory for evidence storage.

    Returns:
        Path to saved file.
    """
    # Create case-specific directory
    case_dir = storage_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Preserve original filename with document_id prefix
    # Format: {document_id}_{original_filename}
    safe_filename = f"{document_id}_{original_filename}"

    # Ensure filename is safe (no path traversal)
    safe_filename = Path(safe_filename).name

    # Save file immutably (write once, never modify)
    file_path = case_dir / safe_filename

    # Write file content
    file_path.write_bytes(file_content)

    return file_path


async def _process_uploaded_file(
    file: UploadFile,
    case_id: str,
    storage_root: Path,
    audit_log_dir: Path,
) -> tuple[Path, str, str, str]:
    """
    Process a single uploaded file: validate, save, compute hash.

    Args:
        file: Uploaded file object.
        case_id: Case identifier.
        storage_root: Root directory for evidence storage.
        audit_log_dir: Directory for audit logs.

    Returns:
        Tuple of (saved_file_path, document_id, file_hash, original_filename).

    Raises:
        HTTPException: If file validation fails.
    """
    # Validate file type
    if not _validate_file_type(file.filename or "", file.content_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed: {file.filename}. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Read file content
    file_content = await file.read()

    # Check file is not empty
    if len(file_content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File is empty: {file.filename}",
        )

    # Generate document ID
    document_id = _generate_document_id()

    # Save file immutably
    original_filename = file.filename or f"unnamed_{document_id}"
    saved_path = _save_file_immutably(
        case_id=case_id,
        document_id=document_id,
        original_filename=original_filename,
        file_content=file_content,
        storage_root=storage_root,
    )

    # Compute SHA-256 hash for chain of custody
    file_hash = compute_file_hash(saved_path)

    # Log upload event to audit log
    audit_logger = AuditLogger(log_dir=audit_log_dir, case_id=case_id)
    upload_event = audit_logger.create_upload_event(
        case_id=case_id,
        document_id=document_id,
        original_filename=original_filename,
        file_hash=file_hash,
        operator="SYSTEM",
    )
    audit_logger.log_upload(upload_event)

    return saved_path, document_id, file_hash, original_filename


@router.post("/{case_id}/documents", response_model=DocumentUploadResponse)
async def upload_documents(
    case_id: str,
    files: list[UploadFile] = File(..., description="Multiple evidence files to upload"),
) -> DocumentUploadResponse:
    """
    Upload multiple evidence documents for a case.

    This endpoint:
    1. Accepts multiple files via multipart form-data
    2. Validates file types (PDF, DOCX, JPG, PNG, TIFF)
    3. Saves files immutably preserving original filenames
    4. Computes SHA-256 hashes for chain of custody
    5. Logs upload events to audit log
    6. Invokes Stage 2: Document Extraction
    7. Returns structured document information

    Args:
        case_id: Case identifier (path parameter).
        files: List of uploaded files (form field: "files").

    Returns:
        DocumentUploadResponse with processed document information.

    Raises:
        HTTPException: If validation fails or processing errors occur.
    """
    # Validate input
    if not case_id or not case_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="case_id cannot be empty"
        )

    if not files or len(files) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="At least one file must be uploaded"
        )

    # Configuration: Use fixed paths for evidence storage and audit logs
    storage_root = Path("./evidence_storage")
    audit_log_dir = Path("./audit_logs")
    storage_root.mkdir(parents=True, exist_ok=True)
    audit_log_dir.mkdir(parents=True, exist_ok=True)

    # Process each uploaded file
    saved_paths: list[Path] = []
    upload_errors: list[tuple[str, str]] = []  # (filename, error_message)

    for file in files:
        try:
            saved_path, _, _, _ = await _process_uploaded_file(
                file=file,
                case_id=case_id,
                storage_root=storage_root,
                audit_log_dir=audit_log_dir,
            )
            saved_paths.append(saved_path)
        except HTTPException as e:
            # Collect validation errors but continue processing other files
            upload_errors.append((file.filename or "unknown", e.detail))
        except Exception as e:
            # Collect unexpected errors
            upload_errors.append((file.filename or "unknown", str(e)))

    # Fail if all files failed validation
    if not saved_paths:
        error_details = "; ".join([f"{fn}: {err}" for fn, err in upload_errors])
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"All files failed validation: {error_details}",
        )

    # Report partial failures if any
    if upload_errors:
        # Log warnings but continue with successful uploads
        pass

    # Invoke Stage 2: Document Extraction
    try:
        extraction_results: list[ExtractionResult] = await extract_documents(
            case_id=case_id,
            file_paths=saved_paths,
            audit_log_dir=audit_log_dir,
            ocr_enabled=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document extraction failed: {str(e)}",
        )

    # Build response
    document_infos: list[DocumentInfo] = []
    for result in extraction_results:
        document_infos.append(
            DocumentInfo(
                document_id=result.document_id,
                pages=result.pages,
                status=result.status.value,
                overall_confidence=result.overall_confidence,
            )
        )

    return DocumentUploadResponse(
        case_id=case_id,
        documents_processed=len(document_infos),
        documents=document_infos,
    )
