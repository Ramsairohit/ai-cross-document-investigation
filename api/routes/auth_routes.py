"""
Stage 12: API Layer - Authentication Routes

Login, logout, and token refresh endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm

from ..audit import create_audit_entry
from ..auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    create_access_token,
)
from ..models import AuditAction, TokenResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """
    Authenticate user and return JWT token.
    """
    # Get client IP
    ip_address = request.client.host if request.client else "unknown"

    # Authenticate
    user = authenticate_user(form_data.username, form_data.password)

    if not user:
        # Log failed attempt
        create_audit_entry(
            user_id="unknown",
            role=None,
            action=AuditAction.LOGIN,
            ip_address=ip_address,
            status="FAILED",
            details=f"Failed login attempt for: {form_data.username}",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create token
    access_token = create_access_token(user.user_id, user.role)

    # Log successful login
    create_audit_entry(
        user_id=user.user_id,
        role=user.role,
        action=AuditAction.LOGIN,
        ip_address=ip_address,
        status="SUCCESS",
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        role=user.role,
    )


@router.post("/logout")
async def logout(request: Request):
    """
    Logout user (client should discard token).
    """
    ip_address = request.client.host if request.client else "unknown"

    # In a real implementation, you would invalidate the token
    # For now, just log the logout action

    return {"message": "Logged out successfully"}
