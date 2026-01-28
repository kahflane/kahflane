"""
Redis integration tests.

Tests real Redis operations with tenant-prefixed keys.
"""
import pytest
import json
from uuid import uuid4

from app.core.tenant import TenantContext, set_current_tenant


pytestmark = pytest.mark.integration


class TestRedisConnection:
    """Test Redis connectivity."""

    @pytest.mark.asyncio
    async def test_redis_ping(self, redis_client):
        """Should connect to Redis successfully."""
        result = await redis_client.ping()
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_set_get(self, redis_client):
        """Should set and get values."""
        key = f"test_key_{uuid4().hex[:8]}"
        value = "test_value"

        await redis_client.set(key, value)
        result = await redis_client.get(key)

        assert result == value

        # Cleanup
        await redis_client.delete(key)


class TestRedisOperations:
    """Test Redis operations used by the application."""

    @pytest.mark.asyncio
    async def test_redis_json_storage(self, redis_client):
        """Should store and retrieve JSON data."""
        key = f"json_test_{uuid4().hex[:8]}"
        data = {"message": "Hello from integration test", "count": 42}

        # Store JSON
        await redis_client.set(key, json.dumps(data), ex=60)

        # Retrieve and parse
        result_str = await redis_client.get(key)
        result = json.loads(result_str)

        assert result["message"] == "Hello from integration test"
        assert result["count"] == 42

        # Cleanup
        await redis_client.delete(key)

    @pytest.mark.asyncio
    async def test_redis_delete(self, redis_client):
        """Should delete values."""
        key = f"delete_test_{uuid4().hex[:8]}"
        await redis_client.set(key, "test_data")

        # Verify it exists
        assert await redis_client.get(key) is not None

        # Delete it
        await redis_client.delete(key)

        # Verify it's gone
        assert await redis_client.get(key) is None

    @pytest.mark.asyncio
    async def test_redis_ttl(self, redis_client):
        """Should expire keys after TTL."""
        key = f"ttl_test_{uuid4().hex[:8]}"

        # Set with 1 second TTL
        await redis_client.set(key, "expiring_value", ex=1)

        # Should exist immediately
        assert await redis_client.get(key) is not None

        # Wait for expiry
        import asyncio
        await asyncio.sleep(1.5)

        # Should be gone
        assert await redis_client.get(key) is None

    @pytest.mark.asyncio
    async def test_redis_key_prefixing(self, redis_client, tenant_context: TenantContext):
        """Should support tenant-prefixed keys."""
        prefix = f"kahflane:{tenant_context.tenant_slug}:cache"
        key = f"{prefix}:test_{uuid4().hex[:8]}"
        data = {"isolated": True}

        await redis_client.set(key, json.dumps(data))

        # Verify the key exists with full prefix
        result_str = await redis_client.get(key)
        assert result_str is not None

        result = json.loads(result_str)
        assert result["isolated"] is True

        # Cleanup
        await redis_client.delete(key)

    @pytest.mark.asyncio
    async def test_redis_increment(self, redis_client):
        """Should support atomic increment for rate limiting."""
        key = f"incr_test_{uuid4().hex[:8]}"

        # Increment multiple times
        for i in range(5):
            result = await redis_client.incr(key)
            assert result == i + 1

        # Cleanup
        await redis_client.delete(key)

    @pytest.mark.asyncio
    async def test_redis_pipeline(self, redis_client):
        """Should support pipelining for batch operations."""
        keys = [f"pipeline_test_{i}_{uuid4().hex[:8]}" for i in range(10)]
        values = [f"value_{i}" for i in range(10)]

        # Batch set using pipeline
        async with redis_client.pipeline() as pipe:
            for key, value in zip(keys, values):
                pipe.set(key, value, ex=60)
            await pipe.execute()

        # Batch get
        async with redis_client.pipeline() as pipe:
            for key in keys:
                pipe.get(key)
            results = await pipe.execute()

        # Verify all values retrieved
        assert len(results) == 10
        assert all(r is not None for r in results)

        # Cleanup
        await redis_client.delete(*keys)


class TestRedisPerformance:
    """Test Redis performance characteristics."""

    @pytest.mark.asyncio
    async def test_bulk_operations(self, redis_client):
        """Should handle bulk operations efficiently."""
        import time

        keys = [f"bulk_test_{i}_{uuid4().hex[:8]}" for i in range(100)]
        values = [f"value_{i}" for i in range(100)]

        start = time.time()

        # Bulk set using pipeline
        async with redis_client.pipeline() as pipe:
            for key, value in zip(keys, values):
                pipe.set(key, value, ex=60)
            await pipe.execute()

        set_time = time.time() - start

        start = time.time()

        # Bulk get
        async with redis_client.pipeline() as pipe:
            for key in keys:
                pipe.get(key)
            results = await pipe.execute()

        get_time = time.time() - start

        # Verify all values retrieved
        assert len(results) == 100
        assert all(r is not None for r in results)

        # Performance should be reasonable (< 1 second for 100 operations)
        assert set_time < 1.0, f"Bulk set too slow: {set_time}s"
        assert get_time < 1.0, f"Bulk get too slow: {get_time}s"

        # Cleanup
        await redis_client.delete(*keys)
