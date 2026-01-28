"""
Pytest configuration and shared fixtures.

Provides test database, tenant context, and authenticated clients.
All tests use real services: PostgreSQL (5433), Redis (6380), Qdrant (6334).
"""
import os

# Set test environment BEFORE importing app modules
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "5433"
os.environ["DB_USER"] = "kahflane_test"
os.environ["DB_PASSWORD"] = "kahflane_test"
os.environ["DB_NAME"] = "kahflane_test"
os.environ["REDIS_HOST"] = "localhost"
os.environ["REDIS_PORT"] = "6380"
os.environ["QDRANT_HOST"] = "localhost"
os.environ["QDRANT_PORT"] = "6334"
os.environ["QDRANT_KEY"] = ""

# Reload config to pick up test environment
import importlib
import app.core.config as config_module
config_module.get_settings.cache_clear()
importlib.reload(config_module)

import pytest
from typing import AsyncGenerator
from uuid import uuid4
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, AsyncEngine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

from app.main import app
from app.core.config import settings
from app.core.tenant import TenantContext, set_current_tenant
from app.core.security import create_access_token, get_password_hash
from app.models.tenant import Tenant, User, TenantMembership

# Verify test config
assert settings.DB_PORT == 5433, f"Expected DB_PORT=5433, got {settings.DB_PORT}"
assert settings.REDIS_PORT == 6380, f"Expected REDIS_PORT=6380, got {settings.REDIS_PORT}"
assert settings.QDRANT_PORT == 6334, f"Expected QDRANT_PORT=6334, got {settings.QDRANT_PORT}"

TEST_SCHEMA_PREFIX = "test_tenant_"


def _get_db_url() -> str:
    return (
        f"postgresql+asyncpg://{os.environ['DB_USER']}:{os.environ['DB_PASSWORD']}"
        f"@{os.environ['DB_HOST']}:{os.environ['DB_PORT']}/{os.environ['DB_NAME']}"
    )


@pytest.fixture
async def test_engine() -> AsyncGenerator[AsyncEngine, None]:
    """Create test database engine and ensure public tables exist."""
    engine = create_async_engine(
        _get_db_url(),
        echo=False,
        pool_pre_ping=True,
    )

    async with engine.begin() as conn:
        await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.tenant (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                slug VARCHAR(63) UNIQUE NOT NULL,
                name VARCHAR(255) NOT NULL,
                domain VARCHAR(255),
                schema_name VARCHAR(63) UNIQUE NOT NULL,
                plan_type VARCHAR(50) DEFAULT 'free',
                is_active BOOLEAN DEFAULT TRUE,
                settings JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public."user" (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                full_name VARCHAR(255) NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                is_email_verified BOOLEAN DEFAULT FALSE,
                last_login_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.tenant_membership (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES public."user"(id) ON DELETE CASCADE,
                tenant_id UUID NOT NULL REFERENCES public.tenant(id) ON DELETE CASCADE,
                role VARCHAR(50) NOT NULL DEFAULT 'member',
                is_default BOOLEAN DEFAULT FALSE,
                invited_by_id UUID REFERENCES public."user"(id),
                joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(user_id, tenant_id)
            )
        """))

    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncGenerator[AsyncSession, None]:
    """Get database session for tests."""
    async_session = sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def test_user(db_session: AsyncSession) -> AsyncGenerator[User, None]:
    """Create a real test user in PostgreSQL."""
    user_id = uuid4()
    email = f"test_{uuid4().hex[:8]}@example.com"

    await db_session.execute(
        text("""
            INSERT INTO public."user" (id, email, password_hash, full_name, is_active, is_email_verified)
            VALUES (:id, :email, :password_hash, :full_name, TRUE, TRUE)
        """),
        {
            "id": user_id,
            "email": email,
            "password_hash": get_password_hash("testpassword123"),
            "full_name": "Test User",
        }
    )
    await db_session.commit()

    result = await db_session.execute(
        text('SELECT * FROM public."user" WHERE id = :id'),
        {"id": user_id}
    )
    row = result.fetchone()

    user = User(
        id=row.id,
        email=row.email,
        password_hash=row.password_hash,
        full_name=row.full_name,
        is_active=row.is_active,
        is_email_verified=row.is_email_verified,
    )

    yield user

    await db_session.execute(
        text('DELETE FROM public."user" WHERE id = :id'),
        {"id": user_id}
    )
    await db_session.commit()


@pytest.fixture
async def test_tenant(db_session: AsyncSession) -> AsyncGenerator[Tenant, None]:
    """Create a real test tenant with schema in PostgreSQL."""
    tenant_id = uuid4()
    tenant_slug = f"test_{uuid4().hex[:8]}"
    schema_name = f"tenant_{tenant_slug}"

    await db_session.execute(
        text("""
            INSERT INTO public.tenant (id, slug, name, schema_name, is_active)
            VALUES (:id, :slug, :name, :schema_name, TRUE)
        """),
        {
            "id": tenant_id,
            "slug": tenant_slug,
            "name": f"Test Tenant {tenant_slug}",
            "schema_name": schema_name,
        }
    )

    await db_session.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))
    await _create_tenant_tables(db_session, schema_name)
    await db_session.commit()

    tenant = Tenant(
        id=tenant_id,
        slug=tenant_slug,
        name=f"Test Tenant {tenant_slug}",
        schema_name=schema_name,
        is_active=True,
    )

    yield tenant

    await db_session.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
    await db_session.execute(
        text('DELETE FROM public.tenant WHERE id = :id'),
        {"id": tenant_id}
    )
    await db_session.commit()


async def _create_tenant_tables(session: AsyncSession, schema_name: str):
    """Create all tenant schema tables."""
    await session.execute(text(f'''
        CREATE TABLE IF NOT EXISTS "{schema_name}".organization (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            logo_url VARCHAR(500),
            settings JSONB DEFAULT '{{}}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    '''))

    await session.execute(text(f'''
        CREATE TABLE IF NOT EXISTS "{schema_name}".organization_member (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id UUID NOT NULL REFERENCES "{schema_name}".organization(id) ON DELETE CASCADE,
            user_id UUID NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'member',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(org_id, user_id)
        )
    '''))

    await session.execute(text(f'''
        CREATE TABLE IF NOT EXISTS "{schema_name}".team (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            org_id UUID NOT NULL REFERENCES "{schema_name}".organization(id) ON DELETE CASCADE,
            name VARCHAR(255) NOT NULL,
            description TEXT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    '''))

    await session.execute(text(f'''
        CREATE TABLE IF NOT EXISTS "{schema_name}".team_member (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            team_id UUID NOT NULL REFERENCES "{schema_name}".team(id) ON DELETE CASCADE,
            user_id UUID NOT NULL,
            role VARCHAR(50) NOT NULL DEFAULT 'member',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            UNIQUE(team_id, user_id)
        )
    '''))

    await session.execute(text(f'''
        CREATE TABLE IF NOT EXISTS "{schema_name}".document (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            uploader_id UUID NOT NULL,
            org_id UUID NOT NULL REFERENCES "{schema_name}".organization(id) ON DELETE CASCADE,
            team_id UUID REFERENCES "{schema_name}".team(id) ON DELETE SET NULL,
            title VARCHAR(500) NOT NULL,
            file_path VARCHAR(1000) NOT NULL,
            file_type VARCHAR(50) NOT NULL,
            file_size_bytes BIGINT,
            scope VARCHAR(50) NOT NULL DEFAULT 'team',
            status VARCHAR(50) NOT NULL DEFAULT 'pending',
            error_message TEXT,
            qdrant_collection_name VARCHAR(255),
            metadata JSONB DEFAULT '{{}}',
            upload_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            processed_at TIMESTAMP WITH TIME ZONE
        )
    '''))

    await session.execute(text(f'''
        CREATE TABLE IF NOT EXISTS "{schema_name}".processing_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES "{schema_name}".document(id) ON DELETE CASCADE,
            stage VARCHAR(100) NOT NULL,
            status VARCHAR(50) NOT NULL,
            message TEXT,
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    '''))


@pytest.fixture
async def test_membership(
    db_session: AsyncSession,
    test_user: User,
    test_tenant: Tenant,
) -> AsyncGenerator[TenantMembership, None]:
    """Create tenant membership for test user."""
    membership_id = uuid4()

    await db_session.execute(
        text("""
            INSERT INTO public.tenant_membership (id, user_id, tenant_id, role, is_default)
            VALUES (:id, :user_id, :tenant_id, :role, TRUE)
        """),
        {
            "id": membership_id,
            "user_id": test_user.id,
            "tenant_id": test_tenant.id,
            "role": "owner",
        }
    )
    await db_session.commit()

    membership = TenantMembership(
        id=membership_id,
        user_id=test_user.id,
        tenant_id=test_tenant.id,
        role="owner",
        is_default=True,
    )

    yield membership


@pytest.fixture
def tenant_context(test_tenant: Tenant) -> TenantContext:
    """Create tenant context for tests."""
    return TenantContext(
        tenant_id=str(test_tenant.id),
        tenant_slug=test_tenant.slug,
        schema_name=test_tenant.schema_name,
    )


@pytest.fixture
def auth_token(test_user: User, test_tenant: Tenant) -> str:
    """Generate JWT token for authenticated requests."""
    return create_access_token(
        user_id=str(test_user.id),
        tenant_id=str(test_tenant.id),
        tenant_slug=test_tenant.slug,
        role="owner",
    )


@pytest.fixture
async def authenticated_client(
    auth_token: str,
    test_tenant: Tenant,
) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client with authentication headers."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url=f"http://{test_tenant.slug}.localhost:8000",
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Host": f"{test_tenant.slug}.localhost",
            "X-Tenant-Slug": test_tenant.slug,
        },
    ) as client:
        yield client


@pytest.fixture
async def unauthenticated_client(test_tenant: Tenant) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client without authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url=f"http://{test_tenant.slug}.localhost:8000",
        headers={
            "Host": f"{test_tenant.slug}.localhost",
            "X-Tenant-Slug": test_tenant.slug,
        },
    ) as client:
        yield client


@pytest.fixture
async def test_organization(db_session: AsyncSession, test_tenant: Tenant, test_user: User):
    """Create a test organization in tenant schema."""
    org_id = uuid4()
    schema = test_tenant.schema_name

    await db_session.execute(text(f'''
        INSERT INTO "{schema}".organization (id, name, description)
        VALUES (:id, :name, :description)
    '''), {"id": org_id, "name": "Test Organization", "description": "Test org description"})

    await db_session.execute(text(f'''
        INSERT INTO "{schema}".organization_member (id, org_id, user_id, role)
        VALUES (:id, :org_id, :user_id, :role)
    '''), {"id": uuid4(), "org_id": org_id, "user_id": test_user.id, "role": "owner"})

    await db_session.commit()

    return {"id": org_id, "name": "Test Organization", "schema": schema}


@pytest.fixture
async def test_team(db_session: AsyncSession, test_organization: dict, test_user: User):
    """Create a test team in tenant schema."""
    team_id = uuid4()
    schema = test_organization["schema"]
    org_id = test_organization["id"]

    await db_session.execute(text(f'''
        INSERT INTO "{schema}".team (id, org_id, name, description)
        VALUES (:id, :org_id, :name, :description)
    '''), {"id": team_id, "org_id": org_id, "name": "Test Team", "description": "Test team description"})

    await db_session.execute(text(f'''
        INSERT INTO "{schema}".team_member (id, team_id, user_id, role)
        VALUES (:id, :team_id, :user_id, :role)
    '''), {"id": uuid4(), "team_id": team_id, "user_id": test_user.id, "role": "owner"})

    await db_session.commit()

    return {"id": team_id, "name": "Test Team", "org_id": org_id, "schema": schema}


# Redis client fixture
@pytest.fixture
async def redis_client():
    """Get real Redis client for tests."""
    import redis.asyncio as redis

    client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6380")),
        decode_responses=True,
    )

    yield client

    # Cleanup test keys
    async for key in client.scan_iter("kahflane:test_*"):
        await client.delete(key)

    await client.aclose()


# Qdrant client fixture
@pytest.fixture
def qdrant_client():
    """Get real Qdrant client for tests."""
    from qdrant_client import QdrantClient

    client = QdrantClient(
        host=os.environ.get("QDRANT_HOST", "localhost"),
        port=int(os.environ.get("QDRANT_PORT", "6334")),
    )

    yield client

    # Cleanup test collections
    collections = client.get_collections().collections
    for col in collections:
        if col.name.startswith("kahflane_test_"):
            client.delete_collection(col.name)


# Sample test data
@pytest.fixture
def sample_text_content() -> str:
    """Sample text content for testing."""
    return """
    Kahflane Knowledge Management System

    Kahflane is an AI-powered knowledge management platform that helps organizations
    efficiently store, retrieve, and leverage their collective knowledge.

    Key Features:
    - Multi-tenant architecture with complete data isolation
    - Intelligent document processing and indexing
    - Semantic search powered by vector embeddings
    - Role-based access control
    - Team and organization management

    This document serves as a test file for the document processing pipeline.
    It contains multiple paragraphs to test chunking functionality.

    The system supports various file formats including PDF, DOCX, TXT, and audio files.
    Audio files are transcribed using AI-powered speech recognition.

    Each tenant has their own isolated database schema, ensuring complete data separation.
    Vector embeddings are stored in Qdrant with tenant-specific filtering.
    """


@pytest.fixture
def sample_text_file(sample_text_content: str) -> bytes:
    """Sample text file bytes."""
    return sample_text_content.encode("utf-8")
