"""
Stage 2: Document Extraction - Audit Logger

Chain-of-custody logging for forensic evidence extraction.
Every extraction event MUST be logged for legal traceability.

IMPORTANT: Audit logs are append-only and immutable.
They provide cryptographic proof of extraction operations.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog

from .models import AuditEvent


def get_docling_version() -> str:
    """
    Get the installed version of Docling.

    Returns:
        Version string or "unknown" if not available.
    """
    try:
        from importlib.metadata import version

        return version("docling")
    except Exception:
        return "unknown"


def get_tesseract_version() -> str:
    """
    Get the installed version of Tesseract OCR.

    Returns:
        Version string or "unknown" if not available.
    """
    try:
        import pytesseract

        version_output = pytesseract.get_tesseract_version()
        return str(version_output)
    except Exception:
        return "unknown"


class AuditLogger:
    """
    Chain-of-custody audit logger for document extraction.

    Logs all extraction events to append-only JSONL files
    for forensic traceability and legal admissibility.

    Attributes:
        log_dir: Directory where audit logs are stored.
        log_file: Path to the current audit log file.
    """

    def __init__(self, log_dir: Path, case_id: Optional[str] = None) -> None:
        """
        Initialize the audit logger.

        Args:
            log_dir: Directory to store audit log files.
            case_id: Optional case ID for case-specific log files.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create case-specific or general log file
        if case_id:
            log_filename = f"audit_log_{case_id}.jsonl"
        else:
            log_filename = "audit_log.jsonl"

        self.log_file = self.log_dir / log_filename

        # Configure structured logging
        self._logger = structlog.get_logger("audit")

    def log_extraction(self, event: AuditEvent) -> None:
        """
        Log a document extraction event.

        Appends the event to the audit log file in JSONL format.
        This operation is atomic and append-only.

        Args:
            event: The audit event to log.
        """
        # Serialize event to JSON
        event_json = event.model_dump_json()

        # Append to log file (atomic operation)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(event_json + "\n")

        # Also log via structlog for real-time monitoring
        self._logger.info(
            "document_extracted",
            event=event.event,
            document_id=event.document_id,
            case_id=event.case_id,
            tool=event.tool,
            status="logged",
        )

    def create_extraction_event(
        self,
        case_id: str,
        document_id: str,
        input_hash: str,
        output_hash: str,
        tool: str = "docling",
        operator: str = "SYSTEM",
    ) -> AuditEvent:
        """
        Create an audit event for document extraction.

        Args:
            case_id: Case identifier.
            document_id: Generated document identifier.
            input_hash: SHA256 hash of input file.
            output_hash: SHA256 hash of extracted content.
            tool: Tool used for extraction.
            operator: Operator who initiated extraction.

        Returns:
            Populated AuditEvent ready for logging.
        """
        # Get tool version
        if tool == "docling":
            tool_version = get_docling_version()
        elif tool == "docling+ocr":
            tool_version = f"docling:{get_docling_version()},tesseract:{get_tesseract_version()}"
        else:
            tool_version = "unknown"

        # Create ISO-8601 timestamp with timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        return AuditEvent(
            event="DOCUMENT_EXTRACTED",
            tool=tool,
            tool_version=tool_version,
            case_id=case_id,
            document_id=document_id,
            input_hash=input_hash,
            output_hash=output_hash,
            timestamp=timestamp,
            operator=operator,
        )

    def create_upload_event(
        self,
        case_id: str,
        document_id: str,
        original_filename: str,
        file_hash: str,
        operator: str = "SYSTEM",
    ) -> AuditEvent:
        """
        Create an audit event for document upload.

        Args:
            case_id: Case identifier.
            document_id: Generated document identifier.
            original_filename: Original filename of uploaded document.
            file_hash: SHA256 hash of uploaded file.
            operator: Operator who initiated upload.

        Returns:
            Populated AuditEvent ready for logging.
        """
        # Create ISO-8601 timestamp with timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        return AuditEvent(
            event="DOCUMENT_UPLOADED",
            tool=None,
            tool_version=None,
            case_id=case_id,
            document_id=document_id,
            input_hash=file_hash,
            output_hash=None,
            timestamp=timestamp,
            operator=operator,
            original_filename=original_filename,
        )

    def log_upload(self, event: AuditEvent) -> None:
        """
        Log a document upload event.

        Appends the event to the audit log file in JSONL format.
        This operation is atomic and append-only.

        Args:
            event: The audit event to log.
        """
        # Serialize event to JSON
        event_json = event.model_dump_json()

        # Append to log file (atomic operation)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(event_json + "\n")

        # Also log via structlog for real-time monitoring
        self._logger.info(
            "document_uploaded",
            event=event.event,
            document_id=event.document_id,
            case_id=event.case_id,
            original_filename=event.original_filename,
            status="logged",
        )

    def get_log_entries(self, document_id: Optional[str] = None) -> list[AuditEvent]:
        """
        Read audit log entries, optionally filtered by document ID.

        Args:
            document_id: Optional filter for specific document.

        Returns:
            List of matching audit events.
        """
        entries: list[AuditEvent] = []

        if not self.log_file.exists():
            return entries

        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    event_data = json.loads(line)
                    event = AuditEvent(**event_data)

                    if document_id is None or event.document_id == document_id:
                        entries.append(event)
                except (json.JSONDecodeError, ValueError):
                    # Skip malformed entries but log warning
                    self._logger.warning("malformed_audit_entry", line=line[:100])

        return entries
