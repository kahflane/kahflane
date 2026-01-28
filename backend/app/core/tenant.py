"""
Tenant context management using contextvars for thread-safe async operations.

This module provides the foundation for multi-tenant isolation by maintaining
tenant context throughout the request lifecycle.
"""
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional
import re


@dataclass(frozen=True)
class TenantContext:
    """Immutable tenant context for the current request."""
    tenant_id: str
    tenant_slug: str
    schema_name: str

    @property
    def is_valid(self) -> bool:
        """Check if tenant context has all required fields."""
        return bool(self.tenant_id and self.tenant_slug and self.schema_name)


# Context variable for tenant - thread-safe for async operations
_current_tenant: ContextVar[Optional[TenantContext]] = ContextVar(
    "current_tenant", default=None
)


def get_current_tenant() -> Optional[TenantContext]:
    """
    Get the current tenant context.

    Returns:
        TenantContext if set, None otherwise.
    """
    return _current_tenant.get()


def set_current_tenant(tenant: Optional[TenantContext]) -> None:
    """
    Set the current tenant context.

    Args:
        tenant: TenantContext to set, or None to clear.
    """
    _current_tenant.set(tenant)


def require_tenant() -> TenantContext:
    """
    Get current tenant context or raise an error if not set.

    Returns:
        TenantContext for the current request.

    Raises:
        ValueError: If no tenant context is available.
    """
    tenant = get_current_tenant()
    if not tenant:
        raise ValueError("No tenant context available. Ensure tenant middleware is configured.")
    return tenant


def extract_subdomain_from_host(
    host: str,
    base_domain: str = "kahflane.com"
) -> Optional[str]:
    """
    Extract subdomain from host header.

    Args:
        host: The host header value (e.g., "acme.kahflane.com:8000")
        base_domain: The base domain to match against

    Returns:
        Subdomain if found and valid, None otherwise.

    Examples:
        - "acme.kahflane.com" -> "acme"
        - "techcorp.kahflane.com:8080" -> "techcorp"
        - "kahflane.com" -> None
        - "localhost:8000" -> None
        - "acme.localhost" -> "acme" (development)
    """
    if not host:
        return None

    # Remove port if present
    host = host.split(":")[0].lower().strip()

    # Handle localhost for development (no subdomain)
    if host in ("localhost", "127.0.0.1"):
        return None

    # Handle development subdomains (e.g., acme.localhost)
    if host.endswith(".localhost"):
        subdomain = host[:-10]  # Remove ".localhost"
        if _is_valid_subdomain(subdomain):
            return subdomain
        return None

    # Check if host ends with base domain
    base_domain = base_domain.lower()
    if host.endswith(f".{base_domain}"):
        subdomain = host[:-(len(base_domain) + 1)]
        if _is_valid_subdomain(subdomain):
            return subdomain

    return None


def _is_valid_subdomain(subdomain: str) -> bool:
    """
    Validate subdomain format.

    Valid subdomains:
    - Alphanumeric and hyphens only
    - Cannot start or end with hyphen
    - 1-63 characters
    """
    if not subdomain or len(subdomain) > 63:
        return False

    # RFC 1123 compliant subdomain validation
    pattern = r'^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$'
    return bool(re.match(pattern, subdomain))


def generate_schema_name(slug: str) -> str:
    """
    Generate PostgreSQL schema name from tenant slug.

    Args:
        slug: The tenant's subdomain slug

    Returns:
        Schema name in format "tenant_{slug}" with hyphens replaced by underscores.

    Raises:
        ValueError: If slug is invalid.
    """
    if not _is_valid_subdomain(slug):
        raise ValueError(f"Invalid tenant slug: {slug}")

    # Replace hyphens with underscores for valid PostgreSQL identifier
    safe_slug = slug.replace("-", "_")
    return f"tenant_{safe_slug}"


def validate_schema_name(schema_name: str) -> bool:
    """
    Validate that a schema name is safe and follows our naming convention.

    This is a security check to prevent SQL injection via schema names.

    Args:
        schema_name: The schema name to validate

    Returns:
        True if valid, False otherwise.
    """
    if not schema_name:
        return False

    # Must start with "tenant_" prefix
    if not schema_name.startswith("tenant_"):
        return False

    # Must only contain valid characters (alphanumeric and underscore)
    pattern = r'^tenant_[a-z0-9_]{1,56}$'
    return bool(re.match(pattern, schema_name))
