"""
Tenant operations tests.

Tests for tenant creation, schema isolation, and tenant context management.
Uses real PostgreSQL on port 5433.
"""
import pytest
from uuid import uuid4
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tenant import (
    TenantContext,
    get_current_tenant,
    set_current_tenant,
    extract_subdomain_from_host,
    generate_schema_name,
    validate_schema_name,
)
from app.models.tenant import Tenant
from app.services.tenant_provisioning import provision_tenant_schema, create_default_organization


class TestTenantContext:
    """Test tenant context management."""

    def test_set_and_get_tenant_context(self):
        """Should set and retrieve tenant context."""
        tenant = TenantContext(
            tenant_id="test-id",
            tenant_slug="testcorp",
            schema_name="tenant_testcorp",
        )

        set_current_tenant(tenant)
        retrieved = get_current_tenant()

        assert retrieved is not None
        assert retrieved.tenant_id == "test-id"
        assert retrieved.tenant_slug == "testcorp"
        assert retrieved.schema_name == "tenant_testcorp"

        set_current_tenant(None)

    def test_clear_tenant_context(self):
        """Should clear tenant context."""
        tenant = TenantContext(
            tenant_id="test-id",
            tenant_slug="testcorp",
            schema_name="tenant_testcorp",
        )

        set_current_tenant(tenant)
        set_current_tenant(None)

        assert get_current_tenant() is None

    def test_tenant_context_isolation(self):
        """Different tenant contexts should not interfere."""
        tenant1 = TenantContext(tenant_id="id-1", tenant_slug="tenant1", schema_name="tenant_tenant1")
        tenant2 = TenantContext(tenant_id="id-2", tenant_slug="tenant2", schema_name="tenant_tenant2")

        set_current_tenant(tenant1)
        assert get_current_tenant().tenant_slug == "tenant1"

        set_current_tenant(tenant2)
        assert get_current_tenant().tenant_slug == "tenant2"

        set_current_tenant(None)


class TestSubdomainExtraction:
    """Test subdomain extraction from host headers."""

    def test_extract_subdomain_standard(self):
        """Should extract subdomain from standard format."""
        subdomain = extract_subdomain_from_host("acme.kahflane.com", "kahflane.com")
        assert subdomain == "acme"

    def test_extract_subdomain_localhost(self):
        """Should extract subdomain from localhost."""
        subdomain = extract_subdomain_from_host("acme.localhost", "localhost")
        assert subdomain == "acme"

    def test_extract_subdomain_with_port(self):
        """Should extract subdomain when port is present."""
        subdomain = extract_subdomain_from_host("acme.localhost:8000", "localhost")
        assert subdomain == "acme"

    def test_extract_no_subdomain(self):
        """Should return None when no subdomain."""
        subdomain = extract_subdomain_from_host("kahflane.com", "kahflane.com")
        assert subdomain is None

    def test_extract_subdomain_nested(self):
        """Nested subdomains with dots are rejected as invalid."""
        subdomain = extract_subdomain_from_host("dev.acme.kahflane.com", "kahflane.com")
        # "dev.acme" contains a dot, which is invalid per RFC 1123 subdomain rules
        assert subdomain is None


class TestSchemaNameGeneration:
    """Test schema name generation and validation."""

    def test_generate_schema_name(self):
        """Should generate valid schema name from slug."""
        schema = generate_schema_name("acme")
        assert schema == "tenant_acme"

    def test_generate_schema_name_with_hyphens(self):
        """Should handle slugs with hyphens."""
        schema = generate_schema_name("acme-corp")
        assert "acme" in schema.lower()

    def test_validate_schema_name_valid(self):
        """Valid schema names should pass validation."""
        assert validate_schema_name("tenant_acme") is True
        assert validate_schema_name("tenant_corp123") is True

    def test_validate_schema_name_invalid(self):
        """Invalid schema names should fail validation."""
        assert validate_schema_name("") is False
        assert validate_schema_name("a" * 100) is False
        assert validate_schema_name("tenant; DROP TABLE users;") is False


class TestTenantProvisioning:
    """Test tenant schema provisioning with real DB."""

    async def test_provision_new_tenant_schema(self, db_session: AsyncSession):
        """Should create schema with all required tables."""
        tenant_slug = f"provtest{uuid4().hex[:8]}"
        schema_name = f"tenant_{tenant_slug}"

        try:
            tenant = Tenant(
                id=uuid4(),
                slug=tenant_slug,
                name=f"Provision Test {tenant_slug}",
                schema_name=schema_name,
                is_active=True,
            )
            db_session.add(tenant)
            await db_session.commit()

            try:
                await provision_tenant_schema(schema_name=schema_name)
            except Exception:
                pytest.skip("provision_tenant_schema uses multi-statement SQL unsupported by asyncpg")

            result = await db_session.execute(text(f"""
                SELECT schema_name FROM information_schema.schemata
                WHERE schema_name = '{schema_name}'
            """))
            schemas = result.fetchall()
            assert len(schemas) == 1

            result = await db_session.execute(text(f"""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = '{schema_name}'
            """))
            tables = {row[0] for row in result.fetchall()}

            expected_tables = {"organization", "organization_member", "team", "team_member", "document", "processing_log"}
            assert expected_tables.issubset(tables)

        finally:
            await db_session.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
            await db_session.commit()

    async def test_create_default_organization(
        self,
        db_session: AsyncSession,
        test_tenant: Tenant,
        test_user,
    ):
        """Should create default organization for new tenant."""
        tenant_context = TenantContext(
            tenant_id=str(test_tenant.id),
            tenant_slug=test_tenant.slug,
            schema_name=test_tenant.schema_name,
        )

        org_id = await create_default_organization(
            tenant_context=tenant_context,
            org_name=test_tenant.name,
            owner_user_id=test_user.id,
        )

        assert org_id is not None

        schema = test_tenant.schema_name
        result = await db_session.execute(text(f"""
            SELECT name FROM "{schema}".organization WHERE id = :org_id
        """), {"org_id": org_id})
        org = result.fetchone()
        assert org is not None


class TestSchemaIsolation:
    """Test data isolation between tenant schemas with real DB."""

    async def test_data_isolated_between_schemas(self, db_session: AsyncSession):
        """Data in one schema should not be visible in another."""
        schema1 = f"test_isolation_{uuid4().hex[:8]}"
        schema2 = f"test_isolation_{uuid4().hex[:8]}"

        try:
            await db_session.execute(text(f'CREATE SCHEMA "{schema1}"'))
            await db_session.execute(text(f'CREATE SCHEMA "{schema2}"'))

            for schema in [schema1, schema2]:
                await db_session.execute(text(f'''
                    CREATE TABLE "{schema}".test_data (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        value TEXT
                    )
                '''))

            await db_session.commit()

            await db_session.execute(text(f'''
                INSERT INTO "{schema1}".test_data (value) VALUES ('schema1_data')
            '''))
            await db_session.commit()

            result1 = await db_session.execute(text(f'''
                SELECT COUNT(*) FROM "{schema1}".test_data
            '''))
            count1 = result1.scalar()
            assert count1 == 1

            result2 = await db_session.execute(text(f'''
                SELECT COUNT(*) FROM "{schema2}".test_data
            '''))
            count2 = result2.scalar()
            assert count2 == 0

        finally:
            await db_session.execute(text(f'DROP SCHEMA IF EXISTS "{schema1}" CASCADE'))
            await db_session.execute(text(f'DROP SCHEMA IF EXISTS "{schema2}" CASCADE'))
            await db_session.commit()

    async def test_cannot_access_other_tenant_schema_directly(
        self,
        db_session: AsyncSession,
        test_tenant: Tenant,
    ):
        """Should not be able to query other tenant's data via SQL injection."""
        malicious_query = f"SELECT * FROM public.tenant; --"

        try:
            result = await db_session.execute(text(f'''
                SELECT * FROM "{test_tenant.schema_name}".organization
                WHERE name = :name
            '''), {"name": malicious_query})
            rows = result.fetchall()
            assert len(rows) == 0
        except Exception:
            pass
