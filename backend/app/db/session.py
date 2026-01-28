"""
Database session management with multi-tenant schema isolation.

This module provides session factories for both public schema (auth, tenants)
and tenant-specific schemas using SQLAlchemy's schema_translate_map feature.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import logging

from app.core.config import settings
from app.core.tenant import (
    TenantContext,
    get_current_tenant,
    validate_schema_name,
)

logger = logging.getLogger(__name__)

# Create async engine for database connections
# Using NullPool for better connection handling in async context
engine: AsyncEngine = create_async_engine(
    settings.async_database_url,
    echo=settings.DEBUG,
    poolclass=NullPool,
    future=True,
)

# Session factory for public schema operations
PublicSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


@asynccontextmanager
async def get_public_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session for the public schema.

    Use this for operations on shared tables:
    - tenant, user, tenant_membership, tenant_invitation

    Yields:
        AsyncSession bound to the public schema.
    """
    async with PublicSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def get_tenant_session(
    tenant: Optional[TenantContext] = None
) -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session for a tenant's schema.

    Uses SQLAlchemy's schema_translate_map to route all queries
    to the tenant's specific schema.

    Args:
        tenant: TenantContext to use. If None, uses current context.

    Yields:
        AsyncSession bound to the tenant's schema.

    Raises:
        ValueError: If no tenant context is available or schema is invalid.
    """
    if tenant is None:
        tenant = get_current_tenant()

    if not tenant:
        raise ValueError("No tenant context available for tenant session")

    if not tenant.schema_name:
        raise ValueError("Tenant context missing schema_name")

    # Validate schema name to prevent SQL injection
    if not validate_schema_name(tenant.schema_name):
        raise ValueError(f"Invalid schema name: {tenant.schema_name}")

    # Create schema translation map
    # Maps None (default schema) to tenant's schema
    schema_translate_map = {None: tenant.schema_name}

    async with engine.connect() as connection:
        # Apply schema translation to this connection
        connection = await connection.execution_options(
            schema_translate_map=schema_translate_map
        )

        async with AsyncSession(
            bind=connection,
            expire_on_commit=False,
            autoflush=False,
        ) as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise


async def create_tenant_schema(schema_name: str) -> None:
    """
    Create a new PostgreSQL schema for a tenant.

    This should only be called during tenant provisioning.

    Args:
        schema_name: The schema name to create (must be valid).

    Raises:
        ValueError: If schema name is invalid.
    """
    if not validate_schema_name(schema_name):
        raise ValueError(f"Invalid schema name: {schema_name}")

    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
        logger.info(f"Created schema: {schema_name}")


async def drop_tenant_schema(schema_name: str) -> None:
    """
    Drop a tenant's PostgreSQL schema.

    WARNING: This will delete all data in the schema. Use with extreme caution.

    Args:
        schema_name: The schema name to drop.

    Raises:
        ValueError: If schema name is invalid or doesn't start with tenant_.
    """
    if not validate_schema_name(schema_name):
        raise ValueError(f"Invalid schema name: {schema_name}")

    if not schema_name.startswith("tenant_"):
        raise ValueError("Can only drop schemas starting with 'tenant_'")

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE"))
        logger.warning(f"Dropped schema: {schema_name}")


async def schema_exists(schema_name: str) -> bool:
    """
    Check if a PostgreSQL schema exists.

    Args:
        schema_name: The schema name to check.

    Returns:
        True if schema exists, False otherwise.
    """
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                "SELECT EXISTS(SELECT 1 FROM information_schema.schemata WHERE schema_name = :name)"
            ),
            {"name": schema_name}
        )
        row = result.fetchone()
        return bool(row and row[0])


# FastAPI dependency functions

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting a tenant-scoped database session.

    This dependency automatically uses the current tenant context
    set by the TenantMiddleware.

    Usage:
        @router.get("/items")
        async def get_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with get_tenant_session() as session:
        yield session


async def get_public_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for getting a public schema database session.

    Use this for auth operations and tenant management.

    Usage:
        @router.post("/login")
        async def login(db: AsyncSession = Depends(get_public_db)):
            ...
    """
    async with get_public_session() as session:
        yield session
