"""
Stage 12: API Layer - Role-Based Access Control (RBAC)

Permission checks based on user roles.

Roles:
- ADMIN: Full access
- INVESTIGATOR: Query + Upload
- ANALYST: Query only
- VIEWER: Query only (read-only)
"""

from functools import wraps
from typing import Callable

from fastapi import HTTPException, status

from .models import UserRole


# Permission definitions
PERMISSIONS = {
    "query": [UserRole.ADMIN, UserRole.INVESTIGATOR, UserRole.ANALYST, UserRole.VIEWER],
    "upload": [UserRole.ADMIN, UserRole.INVESTIGATOR],
    "create_case": [UserRole.ADMIN, UserRole.INVESTIGATOR],
    "delete_case": [UserRole.ADMIN],
    "admin": [UserRole.ADMIN],
    "manage_users": [UserRole.ADMIN],
}


def has_permission(role: UserRole, permission: str) -> bool:
    """
    Check if role has permission.

    Args:
        role: User role.
        permission: Permission name.

    Returns:
        True if role has permission.
    """
    allowed_roles = PERMISSIONS.get(permission, [])
    return role in allowed_roles


def require_permission(permission: str) -> Callable:
    """
    Decorator to require permission for route.

    Args:
        permission: Required permission.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get current user from kwargs
            current_user = kwargs.get("current_user")
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )

            if not has_permission(current_user.role, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission} required",
                )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def check_case_access(user_role: UserRole, case_id: str, owner_id: str, user_id: str) -> bool:
    """
    Check if user can access a case.

    Args:
        user_role: User's role.
        case_id: Case identifier.
        owner_id: Case owner's ID.
        user_id: Current user's ID.

    Returns:
        True if access allowed.
    """
    # Admin can access all cases
    if user_role == UserRole.ADMIN:
        return True

    # Owner can access their cases
    if owner_id == user_id:
        return True

    # Investigators and Analysts can access all cases
    if user_role in [UserRole.INVESTIGATOR, UserRole.ANALYST]:
        return True

    # Viewers have limited access (could be restricted further)
    if user_role == UserRole.VIEWER:
        return True

    return False


class RBACChecker:
    """
    RBAC permission checker for dependency injection.
    """

    def __init__(self, required_permission: str):
        self.required_permission = required_permission

    def __call__(self, role: UserRole) -> bool:
        if not has_permission(role, self.required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {self.required_permission}",
            )
        return True


# Pre-configured permission checkers
require_query = RBACChecker("query")
require_upload = RBACChecker("upload")
require_create_case = RBACChecker("create_case")
require_admin = RBACChecker("admin")
