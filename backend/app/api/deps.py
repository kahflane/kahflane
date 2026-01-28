"""
FastAPI dependencies for authentication and authorization.
"""
from typing import Optional
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import jwt

from app.core.security import decode_token, TokenPayload
from app.core.tenant import get_current_tenant, TenantContext
from app.db.session import get_public_db, get_db

# HTTP Bearer token authentication
security = HTTPBearer(auto_error=False)


async def get_token_payload(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[TokenPayload]:
    """
    Extract and validate JWT token from Authorization header.

    Returns None if no token provided, raises HTTPException if token is invalid.
    """
    if not credentials:
        return None

    try:
        return decode_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    request: Request,
    token: Optional[TokenPayload] = Depends(get_token_payload),
    db: AsyncSession = Depends(get_public_db),
) -> dict:
    """
    Get current authenticated user with tenant validation.

    Validates that:
    1. Token is present and valid
    2. Token tenant matches request tenant (from subdomain)
    3. User exists and is active

    Returns user context dict with user_id, tenant_id, tenant_slug, role.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify token type
    if token.type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Use access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get current tenant from request (set by middleware)
    current_tenant = get_current_tenant()

    # If we have a tenant context, verify it matches the token
    if current_tenant and token.tenant_id != current_tenant.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Token is not valid for this tenant. Please re-authenticate.",
        )

    # Verify user exists and is active
    from app.models.tenant import User
    from sqlalchemy import select

    result = await db.execute(
        select(User).where(User.id == token.sub)
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "user_id": token.sub,
        "email": user.email,
        "full_name": user.full_name,
        "tenant_id": token.tenant_id,
        "tenant_slug": token.tenant_slug,
        "role": token.role,
    }


async def get_optional_user(
    request: Request,
    token: Optional[TokenPayload] = Depends(get_token_payload),
    db: AsyncSession = Depends(get_public_db),
) -> Optional[dict]:
    """
    Get current user if authenticated, None otherwise.

    Similar to get_current_user but doesn't raise if not authenticated.
    """
    if not token:
        return None

    try:
        return await get_current_user(request, token, db)
    except HTTPException:
        return None


async def require_admin(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Require user to have admin or owner role.

    Use this dependency for admin-only endpoints.
    """
    if current_user["role"] not in ("admin", "owner"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user


async def require_owner(
    current_user: dict = Depends(get_current_user),
) -> dict:
    """
    Require user to have owner role.

    Use this dependency for owner-only endpoints.
    """
    if current_user["role"] != "owner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Owner access required",
        )
    return current_user


def get_tenant_context() -> TenantContext:
    """
    Get current tenant context.

    Use this as a dependency when you need tenant info in a route.
    """
    tenant = get_current_tenant()
    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No tenant context available",
        )
    return tenant
