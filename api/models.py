"""
Stage 12: API Layer - Data Models

Pydantic models for API requests, responses, and authentication.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """User roles for RBAC."""

    ADMIN = "ADMIN"
    INVESTIGATOR = "INVESTIGATOR"
    ANALYST = "ANALYST"
    VIEWER = "VIEWER"


class TokenRequest(BaseModel):
    """Login request schema."""

    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    """Login response with JWT token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Seconds until expiration")
    role: UserRole


class TokenPayload(BaseModel):
    """JWT token payload."""

    sub: str = Field(..., description="User ID")
    role: UserRole
    exp: datetime
    iat: datetime


class User(BaseModel):
    """User model."""

    user_id: str
    username: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AuditAction(str, Enum):
    """Audit log action types."""

    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    RAG_QUERY = "RAG_QUERY"
    CASE_VIEW = "CASE_VIEW"
    CASE_CREATE = "CASE_CREATE"
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    ADMIN_ACTION = "ADMIN_ACTION"


class AuditLogEntry(BaseModel):
    """Audit log entry schema."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    role: UserRole
    action: AuditAction
    case_id: Optional[str] = None
    ip_address: str
    status: str = "SUCCESS"
    details: Optional[str] = None


class RAGQueryRequest(BaseModel):
    """RAG query API request."""

    case_id: str = Field(..., description="Case identifier")
    question: str = Field(..., description="Investigator question")


class RAGQueryResponse(BaseModel):
    """RAG query API response."""

    answer: str
    confidence: float
    sources: list[dict]
    limitations: list[str]
    query_id: str


class CaseInfo(BaseModel):
    """Case information."""

    case_id: str
    title: str
    created_at: datetime
    status: str = "ACTIVE"
    owner_id: str


class ErrorResponse(BaseModel):
    """API error response."""

    error: str
    detail: Optional[str] = None
    status_code: int
