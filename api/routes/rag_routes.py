"""
Stage 12: API Layer - RAG Routes

RAG query endpoint with authentication and audit logging.
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from ..audit import create_audit_entry
from ..models import AuditAction, RAGQueryRequest, RAGQueryResponse, User
from ..rbac import has_permission

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query", response_model=RAGQueryResponse)
async def query_rag(
    request: Request,
    query: RAGQueryRequest,
    current_user: User = Depends(),
):
    """
    Query the RAG system for evidence-based answers.

    Requires 'query' permission.
    """
    ip_address = request.client.host if request.client else "unknown"

    # Check permission
    if not has_permission(current_user.role, "query"):
        create_audit_entry(
            user_id=current_user.user_id,
            role=current_user.role,
            action=AuditAction.RAG_QUERY,
            ip_address=ip_address,
            case_id=query.case_id,
            status="DENIED",
            details="Permission denied",
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Query permission required",
        )

    # Log query
    create_audit_entry(
        user_id=current_user.user_id,
        role=current_user.role,
        action=AuditAction.RAG_QUERY,
        ip_address=ip_address,
        case_id=query.case_id,
        status="SUCCESS",
        details=f"Question: {query.question[:100]}",
    )

    # Generate query ID
    query_id = f"Q_{uuid.uuid4().hex[:12]}"

    # In production, this would call the actual RAG pipeline
    # For now, return a stub response
    return RAGQueryResponse(
        answer="This is a placeholder answer. Connect RAG pipeline for actual responses.",
        confidence=0.0,
        sources=[],
        limitations=["RAG pipeline not connected"],
        query_id=query_id,
    )


def create_rag_router(
    rag_pipeline: Any = None,
    embedder_fn: Any = None,
    index: Any = None,
    chunks: list = None,
) -> APIRouter:
    """
    Create RAG router with injected dependencies.

    Args:
        rag_pipeline: RAG pipeline instance.
        embedder_fn: Embedding function.
        index: FAISS index.
        chunks: Chunk metadata.

    Returns:
        Configured router.
    """
    rag_router = APIRouter(prefix="/rag", tags=["RAG"])

    @rag_router.post("/query", response_model=RAGQueryResponse)
    async def connected_query(
        request: Request,
        query: RAGQueryRequest,
        current_user: User = Depends(),
    ):
        ip_address = request.client.host if request.client else "unknown"

        # Permission check
        if not has_permission(current_user.role, "query"):
            raise HTTPException(status_code=403, detail="Permission denied")

        # Log query
        create_audit_entry(
            user_id=current_user.user_id,
            role=current_user.role,
            action=AuditAction.RAG_QUERY,
            ip_address=ip_address,
            case_id=query.case_id,
            status="SUCCESS",
        )

        # Generate query ID
        query_id = f"Q_{uuid.uuid4().hex[:12]}"

        if rag_pipeline and index and chunks and embedder_fn:
            from stage_11_rag import RAGQuery

            rag_query = RAGQuery(case_id=query.case_id, question=query.question)
            result = rag_pipeline.answer_query(
                query=rag_query,
                index=index,
                chunk_metadata=chunks,
                embedder_fn=embedder_fn,
            )

            return RAGQueryResponse(
                answer=result.answer,
                confidence=result.confidence,
                sources=[s.model_dump() for s in result.sources],
                limitations=result.limitations,
                query_id=query_id,
            )

        return RAGQueryResponse(
            answer="RAG pipeline not configured",
            confidence=0.0,
            sources=[],
            limitations=["Pipeline not connected"],
            query_id=query_id,
        )

    return rag_router
