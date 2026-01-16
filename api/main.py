"""
Stage 12: API Layer - Main FastAPI Application

Forensic-grade API with JWT authentication, RBAC, and audit logging.

Usage:
    uvicorn api.main:app --reload

Default credentials:
    admin/admin123 (ADMIN)
    investigator/invest123 (INVESTIGATOR)
    analyst/analyst123 (ANALYST)
    viewer/viewer123 (VIEWER)
"""

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .dependencies import get_client_ip, get_current_user
from .middleware import RequestLoggingMiddleware, SecurityHeadersMiddleware
from .models import User
from .routes.auth_routes import router as auth_router

# Create FastAPI app
app = FastAPI(
    title="Forensic Case Investigation API",
    description="Forensic-grade API for police case investigations with RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)

# Include routers
app.include_router(auth_router, prefix="/api/v1")


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "forensic-api"}


# Protected endpoint example
@app.get("/api/v1/me")
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """Get current user information."""
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "role": current_user.role.value,
        "is_active": current_user.is_active,
    }


# RAG query endpoint (simplified)
@app.post("/api/v1/rag/query")
async def rag_query(
    request: Request,
    case_id: str,
    question: str,
    current_user: User = Depends(get_current_user),
):
    """
    Query the RAG system.

    Requires authentication.
    """
    from .audit import create_audit_entry
    from .models import AuditAction
    from .rbac import has_permission

    ip_address = await get_client_ip(request)

    # Check permission
    if not has_permission(current_user.role, "query"):
        return {"error": "Permission denied"}

    # Log query
    create_audit_entry(
        user_id=current_user.user_id,
        role=current_user.role,
        action=AuditAction.RAG_QUERY,
        ip_address=ip_address,
        case_id=case_id,
        status="SUCCESS",
        details=f"Question: {question[:100]}",
    )

    # Placeholder - connect to actual RAG pipeline
    return {
        "answer": "Connect RAG pipeline for actual responses.",
        "confidence": 0.0,
        "sources": [],
        "limitations": ["RAG pipeline not connected"],
        "case_id": case_id,
        "question": question,
    }


# Admin endpoint
@app.get("/api/v1/admin/audit-logs")
async def get_audit_logs(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    """
    Get audit logs (Admin only).
    """
    from .audit import get_audit_logs
    from .rbac import has_permission

    if not has_permission(current_user.role, "admin"):
        return {"error": "Admin permission required"}

    logs = get_audit_logs()
    return {"logs": logs[-100:]}  # Last 100 entries


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
