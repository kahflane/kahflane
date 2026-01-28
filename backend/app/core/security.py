"""
Security utilities for authentication and JWT handling.

JWT tokens include tenant context to ensure users can only access
data within their authorized tenants.
"""
from datetime import datetime, timedelta, timezone
from typing import Optional, Any
import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import settings


# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenPayload(BaseModel):
    """JWT token payload structure."""
    sub: str                    # user_id
    tenant_id: str              # tenant UUID
    tenant_slug: str            # tenant subdomain
    role: str                   # user's role in tenant
    exp: datetime               # expiration
    iat: datetime               # issued at
    type: str                   # "access" or "refresh"


class TokenPair(BaseModel):
    """Access and refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: The plaintext password to verify
        hashed_password: The bcrypt hash to verify against

    Returns:
        True if password matches, False otherwise.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: The plaintext password to hash

    Returns:
        The bcrypt hash of the password.
    """
    return pwd_context.hash(password)


def create_access_token(
    user_id: str,
    tenant_id: str,
    tenant_slug: str,
    role: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token with tenant context.

    Args:
        user_id: The user's UUID
        tenant_id: The tenant's UUID
        tenant_slug: The tenant's subdomain
        role: User's role in this tenant (owner, admin, member)
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT access token.
    """
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))

    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "tenant_slug": tenant_slug,
        "role": role,
        "exp": expire,
        "iat": now,
        "type": "access",
    }

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(
    user_id: str,
    tenant_id: str,
    tenant_slug: str,
    role: str,
) -> str:
    """
    Create a JWT refresh token with tenant context.

    Refresh tokens have longer expiration and are used to obtain new access tokens.

    Args:
        user_id: The user's UUID
        tenant_id: The tenant's UUID
        tenant_slug: The tenant's subdomain
        role: User's role in this tenant

    Returns:
        Encoded JWT refresh token.
    """
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "tenant_slug": tenant_slug,
        "role": role,
        "exp": expire,
        "iat": now,
        "type": "refresh",
    }

    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_token_pair(
    user_id: str,
    tenant_id: str,
    tenant_slug: str,
    role: str,
) -> TokenPair:
    """
    Create both access and refresh tokens.

    Args:
        user_id: The user's UUID
        tenant_id: The tenant's UUID
        tenant_slug: The tenant's subdomain
        role: User's role in this tenant

    Returns:
        TokenPair with access_token and refresh_token.
    """
    access_token = create_access_token(user_id, tenant_id, tenant_slug, role)
    refresh_token = create_refresh_token(user_id, tenant_id, tenant_slug, role)

    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
    )


def decode_token(token: str) -> TokenPayload:
    """
    Decode and validate a JWT token.

    Args:
        token: The JWT token to decode

    Returns:
        TokenPayload with decoded claims.

    Raises:
        jwt.ExpiredSignatureError: If token has expired
        jwt.InvalidTokenError: If token is invalid
    """
    payload = jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.ALGORITHM],
    )

    return TokenPayload(
        sub=payload["sub"],
        tenant_id=payload["tenant_id"],
        tenant_slug=payload["tenant_slug"],
        role=payload["role"],
        exp=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
        iat=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
        type=payload["type"],
    )


def decode_token_unsafe(token: str) -> dict[str, Any]:
    """
    Decode a JWT token without verification.

    WARNING: Only use this for debugging or logging purposes.
    Never trust the payload for authorization decisions.

    Args:
        token: The JWT token to decode

    Returns:
        Raw payload dict.
    """
    return jwt.decode(
        token,
        options={"verify_signature": False},
    )
