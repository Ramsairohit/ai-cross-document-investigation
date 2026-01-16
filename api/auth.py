"""
Stage 12: API Layer - JWT Authentication

JWT token creation, validation, and password hashing.

IMPORTANT:
- Tokens expire after configured time
- Passwords are hashed with bcrypt
- Secret key should be from environment
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from .models import TokenPayload, User, UserRole

# Configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "forensic-grade-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password against hash.

    Args:
        plain_password: Plain text password.
        hashed_password: Bcrypt hashed password.

    Returns:
        True if password matches.
    """
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str) -> str:
    """
    Hash password with bcrypt.

    Args:
        password: Plain text password.

    Returns:
        Bcrypt hashed password.
    """
    return pwd_context.hash(password)


def create_access_token(
    user_id: str,
    role: UserRole,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create JWT access token.

    Args:
        user_id: User identifier.
        role: User role for RBAC.
        expires_delta: Optional custom expiration.

    Returns:
        Encoded JWT token.
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": user_id,
        "role": role.value,
        "exp": expire,
        "iat": datetime.utcnow(),
    }

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[TokenPayload]:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token string.

    Returns:
        TokenPayload if valid, None if invalid.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        return TokenPayload(
            sub=payload["sub"],
            role=UserRole(payload["role"]),
            exp=datetime.fromtimestamp(payload["exp"]),
            iat=datetime.fromtimestamp(payload["iat"]),
        )
    except JWTError:
        return None


def is_token_expired(token_payload: TokenPayload) -> bool:
    """
    Check if token is expired.

    Args:
        token_payload: Decoded token payload.

    Returns:
        True if expired.
    """
    return datetime.utcnow() > token_payload.exp


# Mock user database (replace with real database in production)
# NOTE: In production, passwords should be pre-hashed in database
MOCK_USERS = {
    "admin": {
        "user_id": "USR_001",
        "username": "admin",
        "password": "admin123",  # Plain for dev mode
        "role": UserRole.ADMIN,
        "is_active": True,
    },
    "investigator": {
        "user_id": "USR_002",
        "username": "investigator",
        "password": "invest123",
        "role": UserRole.INVESTIGATOR,
        "is_active": True,
    },
    "analyst": {
        "user_id": "USR_003",
        "username": "analyst",
        "password": "analyst123",
        "role": UserRole.ANALYST,
        "is_active": True,
    },
    "viewer": {
        "user_id": "USR_004",
        "username": "viewer",
        "password": "viewer123",
        "role": UserRole.VIEWER,
        "is_active": True,
    },
}


def authenticate_user(username: str, password: str) -> Optional[User]:
    """
    Authenticate user with username and password.

    Args:
        username: Username.
        password: Plain text password.

    Returns:
        User if authenticated, None otherwise.
    """
    user_data = MOCK_USERS.get(username)
    if not user_data:
        return None

    # Simple comparison for dev mode (use verify_password with hashed in production)
    if password != user_data["password"]:
        return None

    if not user_data["is_active"]:
        return None

    return User(
        user_id=user_data["user_id"],
        username=user_data["username"],
        role=user_data["role"],
        is_active=user_data["is_active"],
    )


def get_user_by_id(user_id: str) -> Optional[User]:
    """
    Get user by ID.

    Args:
        user_id: User identifier.

    Returns:
        User if found.
    """
    for user_data in MOCK_USERS.values():
        if user_data["user_id"] == user_id:
            return User(
                user_id=user_data["user_id"],
                username=user_data["username"],
                role=user_data["role"],
                is_active=user_data["is_active"],
            )
    return None
