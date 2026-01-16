"""
Stage 12: API Layer - Audit Logging

Comprehensive audit logging for every request.

Every request logs:
- User ID
- Role
- Action
- Case ID (if applicable)
- Timestamp
- IP Address
- Status
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import AuditAction, AuditLogEntry, UserRole

# Audit log directory
AUDIT_LOG_DIR = Path(os.environ.get("AUDIT_LOG_DIR", "audit_logs"))
AUDIT_LOG_DIR.mkdir(exist_ok=True)


def get_audit_log_file() -> Path:
    """Get current audit log file (daily rotation)."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    return AUDIT_LOG_DIR / f"audit_{date_str}.jsonl"


def log_audit_entry(entry: AuditLogEntry) -> None:
    """
    Write audit log entry to file.

    Args:
        entry: Audit log entry.
    """
    log_file = get_audit_log_file()

    log_data = {
        "timestamp": entry.timestamp.isoformat(),
        "user_id": entry.user_id,
        "role": entry.role.value,
        "action": entry.action.value,
        "case_id": entry.case_id,
        "ip_address": entry.ip_address,
        "status": entry.status,
        "details": entry.details,
    }

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_data) + "\n")


def create_audit_entry(
    user_id: str,
    role: UserRole,
    action: AuditAction,
    ip_address: str,
    case_id: Optional[str] = None,
    status: str = "SUCCESS",
    details: Optional[str] = None,
) -> AuditLogEntry:
    """
    Create and log an audit entry.

    Args:
        user_id: User identifier.
        role: User role.
        action: Action performed.
        ip_address: Client IP.
        case_id: Optional case ID.
        status: Success/failure status.
        details: Optional details.

    Returns:
        Created audit entry.
    """
    entry = AuditLogEntry(
        timestamp=datetime.utcnow(),
        user_id=user_id,
        role=role,
        action=action,
        case_id=case_id,
        ip_address=ip_address,
        status=status,
        details=details,
    )

    log_audit_entry(entry)

    return entry


def get_audit_logs(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: Optional[str] = None,
    case_id: Optional[str] = None,
    action: Optional[AuditAction] = None,
) -> list[dict]:
    """
    Query audit logs with filters.

    Args:
        start_date: Filter by start date.
        end_date: Filter by end date.
        user_id: Filter by user.
        case_id: Filter by case.
        action: Filter by action type.

    Returns:
        List of matching audit entries.
    """
    results = []

    # Find log files in date range
    for log_file in AUDIT_LOG_DIR.glob("audit_*.jsonl"):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())

                    # Apply filters
                    entry_time = datetime.fromisoformat(entry["timestamp"])

                    if start_date and entry_time < start_date:
                        continue
                    if end_date and entry_time > end_date:
                        continue
                    if user_id and entry["user_id"] != user_id:
                        continue
                    if case_id and entry["case_id"] != case_id:
                        continue
                    if action and entry["action"] != action.value:
                        continue

                    results.append(entry)
                except (json.JSONDecodeError, KeyError):
                    continue

    return results


class AuditLogger:
    """
    Audit logger for dependency injection.
    """

    def __init__(self, action: AuditAction):
        self.action = action

    def log(
        self,
        user_id: str,
        role: UserRole,
        ip_address: str,
        case_id: Optional[str] = None,
        status: str = "SUCCESS",
        details: Optional[str] = None,
    ) -> AuditLogEntry:
        return create_audit_entry(
            user_id=user_id,
            role=role,
            action=self.action,
            ip_address=ip_address,
            case_id=case_id,
            status=status,
            details=details,
        )
