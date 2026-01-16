"""
AI Cross-Document Investigation System - FastAPI Application

Forensic-grade AI Police Case Investigation System with document upload
and extraction capabilities.

This application provides:
- Multi-file document upload endpoint
- Automatic Stage 2: Document Extraction
- Chain-of-custody audit logging
- Structured document output

Run with: uvicorn main:app --reload
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.upload_documents import router as upload_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application.

    Handles startup and shutdown tasks.
    """
    # Startup: Create required directories
    Path("./evidence_storage").mkdir(parents=True, exist_ok=True)
    Path("./audit_logs").mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown: Cleanup if needed
    pass


# Create FastAPI application
app = FastAPI(
    title="AI Cross-Document Investigation System",
    description="Forensic-grade AI Police Case Investigation System with document upload and extraction",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload_router)


@app.get("/")
async def root() -> dict[str, Any]:
    """
    Root endpoint providing API information.

    Returns:
        Welcome message and API documentation link.
    """
    return {
        "message": "AI Cross-Document Investigation System API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "upload_documents": "POST /cases/{case_id}/documents",
        },
    }


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Health check endpoint.

    Returns:
        Health status of the API.
    """
    return {"status": "healthy", "service": "ai-cross-document-investigation"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
