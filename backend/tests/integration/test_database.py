"""
Database integration tests.

Tests real PostgreSQL operations with tenant schemas.
"""
import pytest
from uuid import uuid4
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant, User


pytestmark = pytest.mark.integration


class TestDatabaseConnection:
    """Test database connectivity and basic operations."""

    @pytest.mark.asyncio
    async def test_database_connection(self, db_session: AsyncSession):
        """Should connect to PostgreSQL successfully."""
        result = await db_session.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        assert row.test == 1

    @pytest.mark.asyncio
    async def test_can_query_public_schema(self, db_session: AsyncSession):
        """Should query public schema tables."""
        result = await db_session.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        )
        tables = [row.table_name for row in result.fetchall()]

        assert "tenant" in tables
        assert "user" in tables
        assert "tenant_membership" in tables


class TestTenantSchema:
    """Test tenant schema operations."""

    @pytest.mark.asyncio
    async def test_tenant_schema_created(self, db_session: AsyncSession, test_tenant: Tenant):
        """Should create tenant schema with all tables."""
        # Check schema exists
        result = await db_session.execute(
            text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = :schema"),
            {"schema": test_tenant.schema_name}
        )
        assert result.fetchone() is not None

    @pytest.mark.asyncio
    async def test_tenant_tables_created(self, db_session: AsyncSession, test_tenant: Tenant):
        """Should create all required tables in tenant schema."""
        result = await db_session.execute(
            text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = :schema
            """),
            {"schema": test_tenant.schema_name}
        )
        tables = [row.table_name for row in result.fetchall()]

        assert "organization" in tables
        assert "organization_member" in tables
        assert "team" in tables
        assert "team_member" in tables
        assert "document" in tables
        assert "processing_log" in tables

    @pytest.mark.asyncio
    async def test_insert_organization(self, db_session: AsyncSession, test_tenant: Tenant):
        """Should insert organization into tenant schema."""
        org_id = uuid4()
        schema = test_tenant.schema_name

        await db_session.execute(
            text(f'''
                INSERT INTO "{schema}".organization (id, name, description)
                VALUES (:id, :name, :description)
            '''),
            {"id": org_id, "name": "Test Org", "description": "Test description"}
        )
        await db_session.commit()

        result = await db_session.execute(
            text(f'SELECT * FROM "{schema}".organization WHERE id = :id'),
            {"id": org_id}
        )
        org = result.fetchone()

        assert org is not None
        assert org.name == "Test Org"

    @pytest.mark.asyncio
    async def test_tenant_isolation(self, db_session: AsyncSession, test_tenant: Tenant):
        """Should isolate data between tenant schemas."""
        # Create another tenant
        other_tenant_id = uuid4()
        other_slug = f"other_{uuid4().hex[:8]}"
        other_schema = f"tenant_{other_slug}"

        await db_session.execute(
            text("""
                INSERT INTO public.tenant (id, slug, name, schema_name)
                VALUES (:id, :slug, :name, :schema_name)
            """),
            {
                "id": other_tenant_id,
                "slug": other_slug,
                "name": "Other Tenant",
                "schema_name": other_schema,
            }
        )
        await db_session.execute(text(f'CREATE SCHEMA "{other_schema}"'))
        await db_session.execute(text(f'''
            CREATE TABLE "{other_schema}".organization (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL
            )
        '''))
        await db_session.commit()

        try:
            # Insert into first tenant
            org1_id = uuid4()
            await db_session.execute(
                text(f'''
                    INSERT INTO "{test_tenant.schema_name}".organization (id, name)
                    VALUES (:id, :name)
                '''),
                {"id": org1_id, "name": "Org in Tenant 1"}
            )

            # Insert into second tenant
            org2_id = uuid4()
            await db_session.execute(
                text(f'''
                    INSERT INTO "{other_schema}".organization (id, name)
                    VALUES (:id, :name)
                '''),
                {"id": org2_id, "name": "Org in Tenant 2"}
            )
            await db_session.commit()

            # Verify isolation - tenant 1 shouldn't see tenant 2's data
            result1 = await db_session.execute(
                text(f'SELECT COUNT(*) FROM "{test_tenant.schema_name}".organization')
            )
            result2 = await db_session.execute(
                text(f'SELECT COUNT(*) FROM "{other_schema}".organization')
            )

            count1 = result1.scalar()
            count2 = result2.scalar()

            # Each schema should only have its own org
            assert count1 >= 1
            assert count2 == 1

            # Verify org names are different
            r1 = await db_session.execute(
                text(f'SELECT name FROM "{test_tenant.schema_name}".organization WHERE id = :id'),
                {"id": org1_id}
            )
            r2 = await db_session.execute(
                text(f'SELECT name FROM "{other_schema}".organization WHERE id = :id'),
                {"id": org2_id}
            )

            assert r1.fetchone().name == "Org in Tenant 1"
            assert r2.fetchone().name == "Org in Tenant 2"

        finally:
            # Cleanup
            await db_session.execute(text(f'DROP SCHEMA IF EXISTS "{other_schema}" CASCADE'))
            await db_session.execute(
                text('DELETE FROM public.tenant WHERE id = :id'),
                {"id": other_tenant_id}
            )
            await db_session.commit()


class TestUserOperations:
    """Test user CRUD operations."""

    @pytest.mark.asyncio
    async def test_create_user(self, db_session: AsyncSession):
        """Should create user in public schema."""
        user_id = uuid4()
        email = f"newuser_{uuid4().hex[:8]}@example.com"

        await db_session.execute(
            text("""
                INSERT INTO public."user" (id, email, password_hash, full_name)
                VALUES (:id, :email, :password_hash, :full_name)
            """),
            {
                "id": user_id,
                "email": email,
                "password_hash": "hashed_password",
                "full_name": "New User",
            }
        )
        await db_session.commit()

        result = await db_session.execute(
            text('SELECT * FROM public."user" WHERE id = :id'),
            {"id": user_id}
        )
        user = result.fetchone()

        assert user is not None
        assert user.email == email
        assert user.full_name == "New User"

        # Cleanup
        await db_session.execute(
            text('DELETE FROM public."user" WHERE id = :id'),
            {"id": user_id}
        )
        await db_session.commit()

    @pytest.mark.asyncio
    async def test_user_email_unique(self, db_session: AsyncSession):
        """Should enforce unique email constraint."""
        from sqlalchemy.exc import IntegrityError

        # Create first user with unique email (without using password hashing)
        user_id = uuid4()
        email = f"unique_test_{uuid4().hex[:8]}@example.com"

        await db_session.execute(
            text("""
                INSERT INTO public."user" (id, email, password_hash, full_name)
                VALUES (:id, :email, :password_hash, :full_name)
            """),
            {
                "id": user_id,
                "email": email,
                "password_hash": "dummy_hash",
                "full_name": "First User",
            }
        )
        await db_session.commit()

        try:
            # Try to create second user with same email
            with pytest.raises(IntegrityError):
                await db_session.execute(
                    text("""
                        INSERT INTO public."user" (id, email, password_hash, full_name)
                        VALUES (:id, :email, :password_hash, :full_name)
                    """),
                    {
                        "id": uuid4(),
                        "email": email,  # Duplicate email
                        "password_hash": "dummy_hash",
                        "full_name": "Duplicate User",
                    }
                )
                await db_session.commit()

            await db_session.rollback()
        finally:
            # Cleanup
            await db_session.execute(
                text('DELETE FROM public."user" WHERE id = :id'),
                {"id": user_id}
            )
            await db_session.commit()
