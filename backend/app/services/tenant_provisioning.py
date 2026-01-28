"""
Tenant provisioning service.

Handles the creation of new tenant schemas and their initial setup.
"""
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
import logging

from app.core.tenant import validate_schema_name, generate_schema_name, TenantContext
from app.db.session import get_public_session, get_tenant_session, create_tenant_schema

logger = logging.getLogger(__name__)


# SQL template for creating tenant schema tables
TENANT_SCHEMA_TABLES_SQL = """
-- Organization table
CREATE TABLE IF NOT EXISTS organization (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    logo_url VARCHAR(500),
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- Organization Member table
CREATE TABLE IF NOT EXISTS organization_member (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'member',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(org_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_org_member_user ON organization_member(user_id);
CREATE INDEX IF NOT EXISTS idx_org_member_org ON organization_member(org_id);

-- Team table
CREATE TABLE IF NOT EXISTS team (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_team_org ON team(org_id);

-- Team Member table
CREATE TABLE IF NOT EXISTS team_member (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    team_id UUID NOT NULL REFERENCES team(id) ON DELETE CASCADE,
    user_id UUID NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'member',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    UNIQUE(team_id, user_id)
);
CREATE INDEX IF NOT EXISTS idx_team_member_user ON team_member(user_id);
CREATE INDEX IF NOT EXISTS idx_team_member_team ON team_member(team_id);

-- Document table
CREATE TABLE IF NOT EXISTS document (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    uploader_id UUID NOT NULL,
    org_id UUID NOT NULL REFERENCES organization(id) ON DELETE CASCADE,
    team_id UUID REFERENCES team(id) ON DELETE SET NULL,
    title VARCHAR(500) NOT NULL,
    file_path VARCHAR(1000) NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size_bytes BIGINT,
    scope VARCHAR(50) NOT NULL DEFAULT 'team',
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    error_message TEXT,
    qdrant_collection_name VARCHAR(255),
    metadata JSONB NOT NULL DEFAULT '{}',
    upload_date TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    CONSTRAINT document_scope_check CHECK (scope IN ('personal', 'team', 'organization')),
    CONSTRAINT document_status_check CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
);
CREATE INDEX IF NOT EXISTS idx_document_org ON document(org_id);
CREATE INDEX IF NOT EXISTS idx_document_team ON document(team_id);
CREATE INDEX IF NOT EXISTS idx_document_uploader ON document(uploader_id);
CREATE INDEX IF NOT EXISTS idx_document_status ON document(status);

-- Processing Log table
CREATE TABLE IF NOT EXISTS processing_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES document(id) ON DELETE CASCADE,
    stage VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_processing_log_document ON processing_log(document_id);
"""


async def provision_tenant_schema(schema_name: str) -> None:
    """
    Create and provision a new tenant schema with all required tables.

    This function:
    1. Creates the PostgreSQL schema
    2. Sets the search_path to the new schema
    3. Creates all tenant-specific tables

    Args:
        schema_name: Name of the schema to create (must start with 'tenant_')

    Raises:
        ValueError: If schema name is invalid
    """
    # Validate schema name to prevent SQL injection
    if not validate_schema_name(schema_name):
        raise ValueError(f"Invalid schema name: {schema_name}")

    logger.info(f"Provisioning tenant schema: {schema_name}")

    # Create the schema
    await create_tenant_schema(schema_name)

    # Create tables in the new schema
    from app.db.session import engine
    async with engine.begin() as conn:
        # Set search path to the new schema
        await conn.execute(text(f"SET search_path TO {schema_name}"))

        # Create all tables
        await conn.execute(text(TENANT_SCHEMA_TABLES_SQL))

        # Reset search path
        await conn.execute(text("SET search_path TO public"))

    logger.info(f"Successfully provisioned schema: {schema_name}")


async def create_default_organization(
    tenant_context: TenantContext,
    org_name: str,
    owner_user_id: UUID,
) -> UUID:
    """
    Create the default organization for a new tenant.

    This is called during tenant registration to set up the initial
    organization and add the owner as a member.

    Args:
        tenant_context: The tenant context for schema routing
        org_name: Name for the organization (usually same as tenant name)
        owner_user_id: User ID of the tenant owner

    Returns:
        UUID of the created organization
    """
    from app.models.organization import Organization, OrganizationMember

    async with get_tenant_session(tenant_context) as session:
        # Create organization
        org = Organization(name=org_name)
        session.add(org)
        await session.flush()

        # Add owner as organization member
        org_member = OrganizationMember(
            org_id=org.id,
            user_id=owner_user_id,
            role="owner",
        )
        session.add(org_member)
        await session.commit()

        logger.info(f"Created default organization '{org_name}' for tenant '{tenant_context.tenant_slug}'")
        return org.id


async def create_default_team(
    tenant_context: TenantContext,
    org_id: UUID,
    team_name: str,
    lead_user_id: UUID,
) -> UUID:
    """
    Create a default team within an organization.

    Args:
        tenant_context: The tenant context for schema routing
        org_id: Organization UUID
        team_name: Name for the team
        lead_user_id: User ID of the team lead

    Returns:
        UUID of the created team
    """
    from app.models.organization import Team, TeamMember

    async with get_tenant_session(tenant_context) as session:
        # Create team
        team = Team(
            org_id=org_id,
            name=team_name,
            description=f"Default team for {team_name}",
        )
        session.add(team)
        await session.flush()

        # Add lead as team member
        team_member = TeamMember(
            team_id=team.id,
            user_id=lead_user_id,
            role="lead",
        )
        session.add(team_member)
        await session.commit()

        return team.id


async def deprovision_tenant_schema(schema_name: str) -> None:
    """
    Remove a tenant's schema and all its data.

    WARNING: This permanently deletes all tenant data. Use with extreme caution.

    Args:
        schema_name: Name of the schema to drop
    """
    if not validate_schema_name(schema_name):
        raise ValueError(f"Invalid schema name: {schema_name}")

    if not schema_name.startswith("tenant_"):
        raise ValueError("Safety check: Can only deprovision schemas starting with 'tenant_'")

    logger.warning(f"Deprovisioning tenant schema: {schema_name}")

    from app.db.session import drop_tenant_schema
    await drop_tenant_schema(schema_name)

    logger.warning(f"Successfully deprovisioned schema: {schema_name}")
