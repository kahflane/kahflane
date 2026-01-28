"""
Health check endpoint tests.

Tests for application health, readiness, and liveness probes.
Uses real PostgreSQL, Redis, and Qdrant services.
"""
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.fixture
    async def client(self):
        """HTTP client for health checks (no auth needed)."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost:8000",
        ) as client:
            yield client

    async def test_basic_health_check(self, client: AsyncClient):
        """Basic health endpoint should always return healthy."""
        response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "kahflane-api"
        assert "version" in data

    async def test_healthz_endpoint(self, client: AsyncClient):
        """Kubernetes healthz endpoint should return ok."""
        response = await client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    async def test_root_endpoint(self, client: AsyncClient):
        """Root endpoint should return API info."""
        response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Kahflane API"
        assert "version" in data
        assert "health" in data


class TestReadinessEndpoint:
    """Test readiness check against real services."""

    @pytest.fixture
    async def client(self):
        """HTTP client for readiness checks."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost:8000",
        ) as client:
            yield client

    async def test_readiness_with_real_services(self, client: AsyncClient):
        """Readiness should return ready when real services are healthy."""
        response = await client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"
        assert data["checks"]["database"] == "ok"
        assert data["checks"]["redis"] == "ok"
        assert data["checks"]["qdrant"] == "ok"

    async def test_readiness_response_format(self, client: AsyncClient):
        """Readiness response should have all required fields."""
        response = await client.get("/ready")

        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert isinstance(data["checks"], dict)
        assert "database" in data["checks"]
        assert "redis" in data["checks"]
        assert "qdrant" in data["checks"]


class TestHealthCheckResponseFormat:
    """Test health check response format consistency."""

    @pytest.fixture
    async def client(self):
        """HTTP client."""
        transport = ASGITransport(app=app)
        async with AsyncClient(
            transport=transport,
            base_url="http://localhost:8000",
        ) as client:
            yield client

    async def test_health_response_has_required_fields(self, client: AsyncClient):
        """Health response should have all required fields."""
        response = await client.get("/health")

        data = response.json()
        required_fields = ["status", "service", "version"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    async def test_health_content_type(self, client: AsyncClient):
        """Health responses should be JSON."""
        response = await client.get("/health")

        assert "application/json" in response.headers["content-type"]

    async def test_healthz_content_type(self, client: AsyncClient):
        """Healthz responses should be JSON."""
        response = await client.get("/healthz")

        assert "application/json" in response.headers["content-type"]
