"""
Tenant resolution middleware for FastAPI.

This middleware extracts the tenant from the subdomain and sets the
tenant context for the duration of each request.
"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from sqlalchemy import select
from typing import Callable, Set
import logging

from app.core.config import settings
from app.core.tenant import (
    TenantContext,
    set_current_tenant,
    extract_subdomain_from_host,
)
from app.db.session import get_public_session

logger = logging.getLogger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """
    Middleware to extract and validate tenant from subdomain.

    Sets tenant context for the duration of the request using contextvars,
    ensuring thread-safe async operation.
    """

    # Paths that don't require tenant context
    EXCLUDED_PATHS: Set[str] = {
        "/health",
        "/healthz",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        f"{settings.API_V1_PREFIX}/auth/register",
        f"{settings.API_V1_PREFIX}/auth/verify-email",
        f"{settings.API_V1_PREFIX}/auth/resend-otp",
        f"{settings.API_V1_PREFIX}/auth/forgot-password",
        f"{settings.API_V1_PREFIX}/auth/reset-password",
        f"{settings.API_V1_PREFIX}/tenants",
        f"{settings.API_V1_PREFIX}/tenants/create",
    }

    # Path prefixes that don't require tenant context
    EXCLUDED_PREFIXES: Set[str] = {
        "/static",
        "/_next",
        f"{settings.API_V1_PREFIX}/tenants/invitations/",
    }

    def __init__(self, app, base_domain: str = None):
        """
        Initialize tenant middleware.

        Args:
            app: The FastAPI application
            base_domain: Base domain for subdomain extraction (default from settings)
        """
        super().__init__(app)
        self.base_domain = base_domain or settings.BASE_DOMAIN

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process each request to extract and validate tenant.

        Args:
            request: The incoming HTTP request
            call_next: The next middleware/handler in the chain

        Returns:
            Response from the handler
        """
        # Check if path is excluded from tenant requirement
        if self._is_excluded_path(request.url.path):
            return await call_next(request)

        # Extract subdomain from host header
        host = request.headers.get("host", "")
        subdomain = extract_subdomain_from_host(host, self.base_domain)

        # Also check X-Tenant-Slug header (for API clients and development)
        if not subdomain:
            subdomain = request.headers.get("x-tenant-slug", "").lower().strip()

        # If no subdomain found, check if this is a tenant-required endpoint
        if not subdomain:
            # For API endpoints that need tenant, return error
            if request.url.path.startswith(settings.API_V1_PREFIX):
                raise HTTPException(
                    status_code=400,
                    detail="Tenant identification required. Use subdomain or X-Tenant-Slug header."
                )
            # For other paths (like root), allow without tenant
            return await call_next(request)

        # Resolve tenant from database
        try:
            tenant_context = await self._resolve_tenant(subdomain)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error resolving tenant '{subdomain}': {e}")
            raise HTTPException(
                status_code=500,
                detail="Error resolving tenant"
            )

        # Set tenant context for this request
        set_current_tenant(tenant_context)

        # Store in request state for easy access in route handlers
        request.state.tenant = tenant_context

        try:
            response = await call_next(request)

            # Add tenant header to response for debugging/transparency
            response.headers["X-Tenant-ID"] = tenant_context.tenant_id
            response.headers["X-Tenant-Slug"] = tenant_context.tenant_slug

            return response
        finally:
            # Always clear tenant context after request
            set_current_tenant(None)

    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from tenant requirement."""
        # Check exact matches
        if path in self.EXCLUDED_PATHS:
            return True

        # Check prefixes
        for prefix in self.EXCLUDED_PREFIXES:
            if path.startswith(prefix):
                return True

        return False

    async def _resolve_tenant(self, subdomain: str) -> TenantContext:
        """
        Resolve tenant from database by subdomain.

        Args:
            subdomain: The tenant's subdomain slug

        Returns:
            TenantContext for the resolved tenant

        Raises:
            HTTPException: If tenant not found or inactive
        """
        # Import here to avoid circular imports
        from app.models.tenant import Tenant

        async with get_public_session() as session:
            result = await session.execute(
                select(Tenant).where(
                    Tenant.slug == subdomain,
                    Tenant.is_active == True
                )
            )
            tenant = result.scalar_one_or_none()

        if not tenant:
            raise HTTPException(
                status_code=404,
                detail=f"Tenant '{subdomain}' not found or inactive"
            )

        return TenantContext(
            tenant_id=str(tenant.id),
            tenant_slug=tenant.slug,
            schema_name=tenant.schema_name,
        )


def get_tenant_from_request(request: Request) -> TenantContext:
    """
    Get tenant context from request state.

    This is a convenience function for route handlers that need
    access to tenant information.

    Args:
        request: The FastAPI request object

    Returns:
        TenantContext from request state

    Raises:
        HTTPException: If no tenant context is available
    """
    tenant = getattr(request.state, "tenant", None)
    if not tenant:
        raise HTTPException(
            status_code=400,
            detail="No tenant context available"
        )
    return tenant
