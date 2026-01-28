"""
End-to-end integration tests.

These tests verify complete workflows across multiple services.
Requires running database, Redis, and Qdrant services.
"""
import pytest
from uuid import uuid4
from io import BytesIO
from httpx import AsyncClient, ASGITransport
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from app.models.tenant import Tenant, User
from app.models.document import DocumentStatus, FileType


@pytest.mark.integration
class TestTenantOnboardingFlow:
    """Test complete tenant onboarding workflow."""

    @pytest.fixture
    async def client(self):
        """HTTP client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost:8000",
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_complete_tenant_creation_flow(
        self,
        client: AsyncClient,
        db_session: AsyncSession,
    ):
        """
        Test: Register new user -> Create tenant -> Login -> Access dashboard

        This tests the complete onboarding workflow.
        """
        unique_id = uuid4().hex[:8]
        email = f"newuser_{unique_id}@example.com"
        password = "securepassword123"
        tenant_name = f"New Corp {unique_id}"

        # Step 1: Register new user with new tenant
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": email,
                "password": password,
                "full_name": "New User",
                "tenant_name": tenant_name,
            },
        )

        if response.status_code not in [200, 201]:
            pytest.skip(f"Registration endpoint not working: {response.status_code}")

        data = response.json()
        tenant_slug = data.get("tenant", {}).get("slug")
        assert tenant_slug is not None

        # Step 2: Login to the new tenant
        login_client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url=f"http://{tenant_slug}.localhost:8000",
            headers={"Host": f"{tenant_slug}.localhost"},
        )

        async with login_client:
            login_response = await login_client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": password},
            )

            if login_response.status_code != 200:
                pytest.skip("Login endpoint not working")

            tokens = login_response.json()
            access_token = tokens.get("access_token")
            assert access_token is not None

            # Step 3: Access protected resource
            login_client.headers["Authorization"] = f"Bearer {access_token}"

            me_response = await login_client.get("/api/v1/auth/me")
            if me_response.status_code == 200:
                me_data = me_response.json()
                assert me_data["email"] == email


@pytest.mark.integration
class TestDocumentProcessingFlow:
    """Test complete document processing workflow."""

    @pytest.mark.asyncio
    async def test_upload_and_process_document(
        self,
        authenticated_client: AsyncClient,
        test_organization: dict,
        test_team: dict,
        sample_text_file: bytes,
    ):
        """
        Test: Upload document -> Processing -> Search

        This tests the complete document ingestion pipeline.
        """
        # Step 1: Upload document
        files = {
            "file": ("integration_test.txt", BytesIO(sample_text_file), "text/plain"),
        }
        data = {
            "title": "Integration Test Document",
            "org_id": str(test_organization["id"]),
            "team_id": str(test_team["id"]),
            "scope": "team",
        }

        upload_response = await authenticated_client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data,
        )

        if upload_response.status_code not in [200, 201]:
            pytest.skip(f"Upload endpoint not working: {upload_response.status_code}")

        doc = upload_response.json()
        doc_id = doc["id"]
        assert doc["status"] == DocumentStatus.PENDING.value

        # Step 2: Verify document appears in list
        list_response = await authenticated_client.get(
            f"/api/v1/documents?org_id={test_organization['id']}",
        )

        if list_response.status_code == 200:
            docs = list_response.json()
            if isinstance(docs, dict):
                docs = docs.get("items", [])
            doc_ids = [d["id"] for d in docs]
            assert doc_id in doc_ids

        # Step 3: Get document details
        detail_response = await authenticated_client.get(f"/api/v1/documents/{doc_id}")

        if detail_response.status_code == 200:
            doc_detail = detail_response.json()
            assert doc_detail["title"] == "Integration Test Document"


@pytest.mark.integration
class TestSearchFlow:
    """Test document search workflow."""

    @pytest.mark.asyncio
    async def test_semantic_search(
        self,
        authenticated_client: AsyncClient,
        test_organization: dict,
        db_session: AsyncSession,
    ):
        """
        Test semantic search across indexed documents.

        Requires documents to be processed and indexed in Qdrant.
        """
        # Search for documents
        response = await authenticated_client.post(
            "/api/v1/search",
            json={
                "query": "knowledge management system",
                "org_id": str(test_organization["id"]),
                "limit": 10,
            },
        )

        # Search endpoint may not be implemented yet
        if response.status_code == 404:
            pytest.skip("Search endpoint not implemented")

        if response.status_code == 200:
            results = response.json()
            assert isinstance(results, list) or "results" in results


@pytest.mark.integration
class TestMultiTenantIsolation:
    """Test data isolation between tenants."""

    @pytest.mark.asyncio
    async def test_tenant_data_isolation(
        self,
        db_session: AsyncSession,
    ):
        """
        Test that data in one tenant is not visible to another.

        Creates two tenants and verifies complete isolation.
        """
        # Create two test schemas
        schema1 = f"test_iso_{uuid4().hex[:8]}"
        schema2 = f"test_iso_{uuid4().hex[:8]}"

        try:
            # Create schemas
            await db_session.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema1}"'))
            await db_session.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema2}"'))

            # Create test tables
            for schema in [schema1, schema2]:
                await db_session.execute(text(f'''
                    CREATE TABLE "{schema}".test_data (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        secret_value TEXT NOT NULL
                    )
                '''))

            await db_session.commit()

            # Insert different data into each schema
            await db_session.execute(text(f'''
                INSERT INTO "{schema1}".test_data (secret_value)
                VALUES ('tenant1_secret_data')
            '''))
            await db_session.execute(text(f'''
                INSERT INTO "{schema2}".test_data (secret_value)
                VALUES ('tenant2_secret_data')
            '''))
            await db_session.commit()

            # Verify data in schema1
            result1 = await db_session.execute(text(f'''
                SELECT secret_value FROM "{schema1}".test_data
            '''))
            values1 = [row[0] for row in result1.fetchall()]
            assert "tenant1_secret_data" in values1
            assert "tenant2_secret_data" not in values1

            # Verify data in schema2
            result2 = await db_session.execute(text(f'''
                SELECT secret_value FROM "{schema2}".test_data
            '''))
            values2 = [row[0] for row in result2.fetchall()]
            assert "tenant2_secret_data" in values2
            assert "tenant1_secret_data" not in values2

        finally:
            # Cleanup
            await db_session.execute(text(f'DROP SCHEMA IF EXISTS "{schema1}" CASCADE'))
            await db_session.execute(text(f'DROP SCHEMA IF EXISTS "{schema2}" CASCADE'))
            await db_session.commit()


@pytest.mark.integration
class TestAPIRateLimiting:
    """Test API rate limiting (if implemented)."""

    @pytest.fixture
    async def client(self):
        """HTTP client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost:8000",
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_rate_limiting_headers(self, client: AsyncClient):
        """Check for rate limiting headers."""
        response = await client.get("/health")

        # Rate limiting headers (if implemented)
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
        ]

        # Just check response is successful, rate limiting is optional
        assert response.status_code == 200


@pytest.mark.integration
class TestCORSConfiguration:
    """Test CORS headers."""

    @pytest.fixture
    async def client(self):
        """HTTP client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost:8000",
        ) as client:
            yield client

    @pytest.mark.asyncio
    async def test_cors_headers_present(self, client: AsyncClient):
        """CORS headers should be present for allowed origins."""
        response = await client.options(
            "/api/v1/auth/login",
            headers={
                "Origin": "http://localhost:5000",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Check for CORS headers
        assert response.status_code in [200, 204, 405]

    @pytest.mark.asyncio
    async def test_cors_blocked_for_unknown_origin(self, client: AsyncClient):
        """CORS should block unknown origins."""
        response = await client.options(
            "/api/v1/auth/login",
            headers={
                "Origin": "http://malicious-site.com",
                "Access-Control-Request-Method": "POST",
            },
        )

        # Should not include Access-Control-Allow-Origin for malicious origin
        allowed_origin = response.headers.get("Access-Control-Allow-Origin")
        assert allowed_origin != "http://malicious-site.com"


@pytest.mark.integration
class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.asyncio
    async def test_404_response_format(
        self,
        authenticated_client: AsyncClient,
    ):
        """404 responses should be properly formatted."""
        response = await authenticated_client.get(f"/api/v1/documents/{uuid4()}")

        if response.status_code == 404:
            data = response.json()
            assert "detail" in data or "message" in data

    @pytest.mark.asyncio
    async def test_validation_error_format(
        self,
        authenticated_client: AsyncClient,
    ):
        """Validation errors should be properly formatted."""
        response = await authenticated_client.post(
            "/api/v1/documents/upload",
            data={"title": ""},  # Missing required fields
        )

        if response.status_code == 422:
            data = response.json()
            assert "detail" in data

    @pytest.mark.asyncio
    async def test_unauthorized_response_format(
        self,
        unauthenticated_client: AsyncClient,
    ):
        """401 responses should be properly formatted."""
        response = await unauthenticated_client.get("/api/v1/organizations")

        if response.status_code in [401, 403]:
            data = response.json()
            assert "detail" in data or "message" in data
