"""
Stage 12: API Layer

Forensic-grade REST API with:
- JWT Authentication
- Role-Based Access Control (RBAC)
- Comprehensive Audit Logging

Usage:
    uvicorn api.main:app --reload
"""

from .auth import (
    authenticate_user,
    create_access_token,
    decode_token,
    hash_password,
    verify_password,
)
from .audit import (
    AuditLogger,
    create_audit_entry,
    get_audit_logs,
    log_audit_entry,
)
from .dependencies import get_client_ip, get_current_user
from .main import app
from .models import (
    AuditAction,
    AuditLogEntry,
    CaseInfo,
    ErrorResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    TokenPayload,
    TokenRequest,
    TokenResponse,
    User,
    UserRole,
)
from .rbac import (
    RBACChecker,
    check_case_access,
    has_permission,
    require_admin,
    require_create_case,
    require_query,
    require_upload,
)

__all__ = [
    # App
    "app",
    # Models
    "UserRole",
    "TokenRequest",
    "TokenResponse",
    "TokenPayload",
    "User",
    "AuditAction",
    "AuditLogEntry",
    "RAGQueryRequest",
    "RAGQueryResponse",
    "CaseInfo",
    "ErrorResponse",
    # Auth
    "authenticate_user",
    "create_access_token",
    "decode_token",
    "hash_password",
    "verify_password",
    # RBAC
    "has_permission",
    "check_case_access",
    "RBACChecker",
    "require_query",
    "require_upload",
    "require_create_case",
    "require_admin",
    # Audit
    "create_audit_entry",
    "log_audit_entry",
    "get_audit_logs",
    "AuditLogger",
    # Dependencies
    "get_current_user",
    "get_client_ip",
]
