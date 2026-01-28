"""
Redis service tests.

Tests for Redis operations with tenant-prefixed key isolation.
Uses real Redis on port 6380.
"""
import pytest
from uuid import uuid4
from datetime import timedelta

from app.core.tenant import TenantContext, set_current_tenant
from app.services.redis_service import RedisService, GLOBAL_PREFIX


class TestRedisKeyGeneration:
    """Test Redis key generation with tenant prefixes."""

    @pytest.fixture
    def redis_service(self):
        """Get Redis service."""
        return RedisService()

    @pytest.fixture
    def sample_tenant_context(self) -> TenantContext:
        """Sample tenant context."""
        return TenantContext(
            tenant_id=str(uuid4()),
            tenant_slug="testcorp",
            schema_name="tenant_testcorp",
        )

    def test_build_key_format(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should build key with correct format."""
        set_current_tenant(sample_tenant_context)
        try:
            key = redis_service._build_key("user:123", namespace="cache")
            assert key == "kahflane:testcorp:cache:user:123"
        finally:
            set_current_tenant(None)

    def test_build_key_different_namespaces(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should handle different namespaces."""
        set_current_tenant(sample_tenant_context)
        try:
            cache_key = redis_service._build_key("item", namespace="cache")
            session_key = redis_service._build_key("item", namespace="session")
            rate_key = redis_service._build_key("item", namespace="ratelimit")

            assert cache_key == "kahflane:testcorp:cache:item"
            assert session_key == "kahflane:testcorp:session:item"
            assert rate_key == "kahflane:testcorp:ratelimit:item"
        finally:
            set_current_tenant(None)

    def test_build_key_no_tenant_raises_error(self, redis_service: RedisService):
        """Should raise error without tenant context."""
        set_current_tenant(None)

        with pytest.raises(ValueError, match="No tenant context"):
            redis_service._build_key("key")

    def test_build_global_key(self, redis_service: RedisService):
        """Should build global key without tenant prefix."""
        key = redis_service._build_global_key("config", namespace="global")
        assert key == "kahflane:global:global:config"


class TestRedisBasicOperations:
    """Test basic Redis operations with real Redis."""

    @pytest.fixture
    def redis_service(self):
        """Redis service connected to real Redis."""
        return RedisService()

    @pytest.fixture
    def sample_tenant_context(self) -> TenantContext:
        """Sample tenant context with unique slug for test isolation."""
        slug = f"test_{uuid4().hex[:8]}"
        return TenantContext(
            tenant_id=str(uuid4()),
            tenant_slug=slug,
            schema_name=f"tenant_{slug}",
        )

    async def test_set_and_get(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should set and get value with tenant-prefixed key."""
        set_current_tenant(sample_tenant_context)
        try:
            await redis_service.set("mykey", "myvalue")
            result = await redis_service.get("mykey")
            assert result == "myvalue"
        finally:
            await redis_service.delete("mykey")
            set_current_tenant(None)
            await redis_service.close()

    async def test_set_with_ttl(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should set value with TTL."""
        set_current_tenant(sample_tenant_context)
        try:
            await redis_service.set("ttlkey", "ttlvalue", ttl=300)
            result = await redis_service.get("ttlkey")
            assert result == "ttlvalue"

            remaining = await redis_service.ttl("ttlkey")
            assert 0 < remaining <= 300
        finally:
            await redis_service.delete("ttlkey")
            set_current_tenant(None)
            await redis_service.close()

    async def test_set_with_timedelta_ttl(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should set value with timedelta TTL."""
        set_current_tenant(sample_tenant_context)
        try:
            await redis_service.set("tdkey", "tdvalue", ttl=timedelta(hours=1))
            remaining = await redis_service.ttl("tdkey")
            assert 0 < remaining <= 3600
        finally:
            await redis_service.delete("tdkey")
            set_current_tenant(None)
            await redis_service.close()

    async def test_delete(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should delete a key."""
        set_current_tenant(sample_tenant_context)
        try:
            await redis_service.set("delkey", "delvalue")
            result = await redis_service.delete("delkey")
            assert result == 1

            value = await redis_service.get("delkey")
            assert value is None
        finally:
            set_current_tenant(None)
            await redis_service.close()

    async def test_exists(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should check key existence."""
        set_current_tenant(sample_tenant_context)
        try:
            await redis_service.set("existkey", "value")
            assert await redis_service.exists("existkey") is True
            assert await redis_service.exists("nonexistent") is False
        finally:
            await redis_service.delete("existkey")
            set_current_tenant(None)
            await redis_service.close()

    async def test_get_nonexistent(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should return None for non-existent key."""
        set_current_tenant(sample_tenant_context)
        try:
            result = await redis_service.get("nonexistent_key")
            assert result is None
        finally:
            set_current_tenant(None)
            await redis_service.close()


class TestRedisJSONOperations:
    """Test JSON serialization operations with real Redis."""

    @pytest.fixture
    def redis_service(self):
        return RedisService()

    @pytest.fixture
    def sample_tenant_context(self) -> TenantContext:
        slug = f"test_{uuid4().hex[:8]}"
        return TenantContext(
            tenant_id=str(uuid4()),
            tenant_slug=slug,
            schema_name=f"tenant_{slug}",
        )

    async def test_set_and_get_json(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should serialize and deserialize JSON."""
        set_current_tenant(sample_tenant_context)
        try:
            data = {"name": "test", "count": 42, "items": [1, 2, 3]}
            await redis_service.set_json("jsonkey", data)

            result = await redis_service.get_json("jsonkey")
            assert result == data
        finally:
            await redis_service.delete("jsonkey")
            set_current_tenant(None)
            await redis_service.close()

    async def test_get_json_nonexistent(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should return None for non-existent JSON key."""
        set_current_tenant(sample_tenant_context)
        try:
            result = await redis_service.get_json("missing_json")
            assert result is None
        finally:
            set_current_tenant(None)
            await redis_service.close()


class TestRedisSessionManagement:
    """Test session management with real Redis."""

    @pytest.fixture
    def redis_service(self):
        return RedisService()

    @pytest.fixture
    def sample_tenant_context(self) -> TenantContext:
        slug = f"test_{uuid4().hex[:8]}"
        return TenantContext(
            tenant_id=str(uuid4()),
            tenant_slug=slug,
            schema_name=f"tenant_{slug}",
        )

    async def test_set_and_get_session(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should set and get session data."""
        set_current_tenant(sample_tenant_context)
        try:
            session_data = {"user_id": "123", "role": "admin"}
            await redis_service.set_session("sess_abc", session_data)

            result = await redis_service.get_session("sess_abc")
            assert result == session_data
        finally:
            await redis_service.delete_session("sess_abc")
            set_current_tenant(None)
            await redis_service.close()

    async def test_session_has_ttl(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Session should have TTL set."""
        set_current_tenant(sample_tenant_context)
        try:
            await redis_service.set_session("sess_ttl", {"user_id": "123"})
            remaining = await redis_service.ttl("sess_ttl", namespace="session")
            # Default TTL is 24 hours = 86400 seconds
            assert 0 < remaining <= 86400
        finally:
            await redis_service.delete_session("sess_ttl")
            set_current_tenant(None)
            await redis_service.close()


class TestRedisRateLimiting:
    """Test rate limiting with real Redis."""

    @pytest.fixture
    def redis_service(self):
        return RedisService()

    @pytest.fixture
    def sample_tenant_context(self) -> TenantContext:
        slug = f"test_{uuid4().hex[:8]}"
        return TenantContext(
            tenant_id=str(uuid4()),
            tenant_slug=slug,
            schema_name=f"tenant_{slug}",
        )

    async def test_rate_limit_allowed(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should allow requests within rate limit."""
        set_current_tenant(sample_tenant_context)
        key = f"api:user:{uuid4().hex[:8]}"
        try:
            allowed, count = await redis_service.check_rate_limit(
                key, limit=10, window_seconds=60,
            )
            assert allowed is True
            assert count == 1
        finally:
            await redis_service.delete(key, namespace="ratelimit")
            set_current_tenant(None)
            await redis_service.close()

    async def test_rate_limit_exceeded(
        self,
        redis_service: RedisService,
        sample_tenant_context: TenantContext,
    ):
        """Should block requests exceeding rate limit."""
        set_current_tenant(sample_tenant_context)
        key = f"api:user:{uuid4().hex[:8]}"
        try:
            # Make 11 requests with limit of 10
            for _ in range(10):
                await redis_service.check_rate_limit(key, limit=10, window_seconds=60)

            allowed, count = await redis_service.check_rate_limit(
                key, limit=10, window_seconds=60,
            )
            assert allowed is False
            assert count == 11
        finally:
            await redis_service.delete(key, namespace="ratelimit")
            set_current_tenant(None)
            await redis_service.close()


class TestRedisTenantIsolation:
    """Test tenant isolation with real Redis."""

    @pytest.fixture
    def redis_service(self):
        return RedisService()

    def test_different_tenants_different_keys(self, redis_service: RedisService):
        """Different tenants should have different key prefixes."""
        tenant1 = TenantContext(tenant_id="id-1", tenant_slug="acme", schema_name="tenant_acme")
        tenant2 = TenantContext(tenant_id="id-2", tenant_slug="globex", schema_name="tenant_globex")

        set_current_tenant(tenant1)
        key1 = redis_service._build_key("user:123", namespace="cache")

        set_current_tenant(tenant2)
        key2 = redis_service._build_key("user:123", namespace="cache")

        set_current_tenant(None)

        assert key1 == "kahflane:acme:cache:user:123"
        assert key2 == "kahflane:globex:cache:user:123"
        assert key1 != key2

    async def test_tenant_data_isolation(self, redis_service: RedisService):
        """Tenant data should be isolated in real Redis."""
        slug1 = f"test_{uuid4().hex[:8]}"
        slug2 = f"test_{uuid4().hex[:8]}"
        tenant1 = TenantContext(tenant_id="id-1", tenant_slug=slug1, schema_name=f"tenant_{slug1}")
        tenant2 = TenantContext(tenant_id="id-2", tenant_slug=slug2, schema_name=f"tenant_{slug2}")

        try:
            # Set data for tenant1
            set_current_tenant(tenant1)
            await redis_service.set("shared_key", "tenant1_value")

            # Set data for tenant2
            set_current_tenant(tenant2)
            await redis_service.set("shared_key", "tenant2_value")

            # Verify isolation
            set_current_tenant(tenant1)
            val1 = await redis_service.get("shared_key")
            assert val1 == "tenant1_value"

            set_current_tenant(tenant2)
            val2 = await redis_service.get("shared_key")
            assert val2 == "tenant2_value"
        finally:
            set_current_tenant(tenant1)
            await redis_service.delete("shared_key")
            set_current_tenant(tenant2)
            await redis_service.delete("shared_key")
            set_current_tenant(None)
            await redis_service.close()


class TestRedisHealthCheck:
    """Test Redis health check with real Redis."""

    @pytest.fixture
    def redis_service(self):
        return RedisService()

    async def test_health_check_healthy(self, redis_service: RedisService):
        """Should return True when Redis is healthy."""
        is_healthy = await redis_service.health_check()
        assert is_healthy is True
        await redis_service.close()
