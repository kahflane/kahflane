"""
Redis service with tenant-isolated key prefixes.

All keys are prefixed with tenant identifier for complete data isolation.
Key format: kahflane:{tenant_slug}:{namespace}:{key}
"""
import logging
import json
from typing import Optional, Any, List, Dict, Union
from datetime import timedelta
import redis.asyncio as redis

from app.core.config import settings
from app.core.tenant import get_current_tenant, TenantContext

logger = logging.getLogger(__name__)

# Global key prefix
GLOBAL_PREFIX = "kahflane"


class RedisService:
    """
    Redis service with tenant-aware key management.

    All operations automatically prefix keys with tenant identifier
    to ensure complete data isolation between tenants.

    Key format: kahflane:{tenant_slug}:{namespace}:{key}

    Example:
        - kahflane:acme:cache:user:123
        - kahflane:acme:session:abc-def-ghi
        - kahflane:acme:ratelimit:api:192.168.1.1
    """

    def __init__(self):
        """Initialize Redis connection pool."""
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def _get_client(self) -> redis.Redis:
        """Get or create Redis client with connection pool."""
        if self._client is None:
            self._pool = redis.ConnectionPool.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                max_connections=10,
            )
            self._client = redis.Redis(connection_pool=self._pool)
        return self._client

    def _build_key(
        self,
        key: str,
        namespace: str = "data",
        tenant: Optional[TenantContext] = None,
    ) -> str:
        """
        Build a tenant-prefixed key.

        Args:
            key: The actual key
            namespace: Key namespace (cache, session, ratelimit, etc.)
            tenant: Optional tenant context, uses current if not provided

        Returns:
            Full key with tenant prefix: kahflane:{tenant_slug}:{namespace}:{key}
        """
        if tenant is None:
            tenant = get_current_tenant()

        if not tenant:
            raise ValueError("No tenant context for Redis key")

        return f"{GLOBAL_PREFIX}:{tenant.tenant_slug}:{namespace}:{key}"

    def _build_global_key(self, key: str, namespace: str = "global") -> str:
        """
        Build a global key (not tenant-specific).

        Used for cross-tenant data like global rate limits.

        Args:
            key: The actual key
            namespace: Key namespace

        Returns:
            Global key: kahflane:global:{namespace}:{key}
        """
        return f"{GLOBAL_PREFIX}:global:{namespace}:{key}"

    # ============ Basic Operations ============

    async def get(
        self,
        key: str,
        namespace: str = "cache",
    ) -> Optional[str]:
        """Get a value by key."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.get(full_key)

    async def set(
        self,
        key: str,
        value: str,
        namespace: str = "cache",
        ttl: Optional[Union[int, timedelta]] = None,
    ) -> bool:
        """
        Set a value with optional TTL.

        Args:
            key: Key name
            value: String value
            namespace: Key namespace
            ttl: Time-to-live (seconds or timedelta)

        Returns:
            True if successful
        """
        client = await self._get_client()
        full_key = self._build_key(key, namespace)

        if ttl is not None:
            if isinstance(ttl, timedelta):
                ttl = int(ttl.total_seconds())
            return await client.setex(full_key, ttl, value)
        else:
            return await client.set(full_key, value)

    async def delete(self, key: str, namespace: str = "cache") -> int:
        """Delete a key."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.delete(full_key)

    async def exists(self, key: str, namespace: str = "cache") -> bool:
        """Check if a key exists."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.exists(full_key) > 0

    async def expire(
        self,
        key: str,
        ttl: Union[int, timedelta],
        namespace: str = "cache",
    ) -> bool:
        """Set TTL on existing key."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        if isinstance(ttl, timedelta):
            ttl = int(ttl.total_seconds())
        return await client.expire(full_key, ttl)

    async def ttl(self, key: str, namespace: str = "cache") -> int:
        """Get remaining TTL for a key (-1 if no TTL, -2 if key doesn't exist)."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.ttl(full_key)

    # ============ JSON Operations ============

    async def get_json(
        self,
        key: str,
        namespace: str = "cache",
    ) -> Optional[Any]:
        """Get and deserialize JSON value."""
        value = await self.get(key, namespace)
        if value is None:
            return None
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON for key: {key}")
            return None

    async def set_json(
        self,
        key: str,
        value: Any,
        namespace: str = "cache",
        ttl: Optional[Union[int, timedelta]] = None,
    ) -> bool:
        """Serialize and set JSON value."""
        json_value = json.dumps(value)
        return await self.set(key, json_value, namespace, ttl)

    # ============ Hash Operations ============

    async def hget(
        self,
        key: str,
        field: str,
        namespace: str = "hash",
    ) -> Optional[str]:
        """Get a hash field value."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.hget(full_key, field)

    async def hset(
        self,
        key: str,
        field: str,
        value: str,
        namespace: str = "hash",
    ) -> int:
        """Set a hash field value."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.hset(full_key, field, value)

    async def hgetall(
        self,
        key: str,
        namespace: str = "hash",
    ) -> Dict[str, str]:
        """Get all hash fields and values."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.hgetall(full_key)

    async def hdel(
        self,
        key: str,
        *fields: str,
        namespace: str = "hash",
    ) -> int:
        """Delete hash fields."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.hdel(full_key, *fields)

    async def hincrby(
        self,
        key: str,
        field: str,
        amount: int = 1,
        namespace: str = "hash",
    ) -> int:
        """Increment a hash field value."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.hincrby(full_key, field, amount)

    # ============ List Operations ============

    async def lpush(
        self,
        key: str,
        *values: str,
        namespace: str = "list",
    ) -> int:
        """Push values to the left of a list."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.lpush(full_key, *values)

    async def rpush(
        self,
        key: str,
        *values: str,
        namespace: str = "list",
    ) -> int:
        """Push values to the right of a list."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.rpush(full_key, *values)

    async def lrange(
        self,
        key: str,
        start: int = 0,
        end: int = -1,
        namespace: str = "list",
    ) -> List[str]:
        """Get a range of list elements."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.lrange(full_key, start, end)

    async def llen(self, key: str, namespace: str = "list") -> int:
        """Get list length."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.llen(full_key)

    # ============ Set Operations ============

    async def sadd(
        self,
        key: str,
        *members: str,
        namespace: str = "set",
    ) -> int:
        """Add members to a set."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.sadd(full_key, *members)

    async def srem(
        self,
        key: str,
        *members: str,
        namespace: str = "set",
    ) -> int:
        """Remove members from a set."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.srem(full_key, *members)

    async def smembers(
        self,
        key: str,
        namespace: str = "set",
    ) -> set:
        """Get all set members."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.smembers(full_key)

    async def sismember(
        self,
        key: str,
        member: str,
        namespace: str = "set",
    ) -> bool:
        """Check if member exists in set."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.sismember(full_key, member)

    # ============ Counter / Rate Limiting ============

    async def incr(self, key: str, namespace: str = "counter") -> int:
        """Increment a counter."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.incr(full_key)

    async def incrby(
        self,
        key: str,
        amount: int,
        namespace: str = "counter",
    ) -> int:
        """Increment a counter by amount."""
        client = await self._get_client()
        full_key = self._build_key(key, namespace)
        return await client.incrby(full_key, amount)

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """
        Check if rate limit is exceeded using sliding window.

        Args:
            key: Rate limit key (e.g., "api:user:123")
            limit: Maximum requests allowed
            window_seconds: Time window in seconds

        Returns:
            Tuple of (is_allowed, current_count)
        """
        client = await self._get_client()
        full_key = self._build_key(key, namespace="ratelimit")

        # Increment counter
        current = await client.incr(full_key)

        # Set expiry on first request
        if current == 1:
            await client.expire(full_key, window_seconds)

        is_allowed = current <= limit
        return is_allowed, current

    # ============ Session Management ============

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return await self.get_json(session_id, namespace="session")

    async def set_session(
        self,
        session_id: str,
        data: Dict[str, Any],
        ttl: Union[int, timedelta] = timedelta(hours=24),
    ) -> bool:
        """Set session data with TTL."""
        return await self.set_json(session_id, data, namespace="session", ttl=ttl)

    async def delete_session(self, session_id: str) -> int:
        """Delete a session."""
        return await self.delete(session_id, namespace="session")

    async def extend_session(
        self,
        session_id: str,
        ttl: Union[int, timedelta] = timedelta(hours=24),
    ) -> bool:
        """Extend session TTL."""
        return await self.expire(session_id, ttl, namespace="session")

    # ============ Cache Operations ============

    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cached value (JSON)."""
        return await self.get_json(key, namespace="cache")

    async def cache_set(
        self,
        key: str,
        value: Any,
        ttl: Union[int, timedelta] = timedelta(minutes=15),
    ) -> bool:
        """Set cached value with TTL."""
        return await self.set_json(key, value, namespace="cache", ttl=ttl)

    async def cache_delete(self, key: str) -> int:
        """Delete cached value."""
        return await self.delete(key, namespace="cache")

    async def cache_invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache keys matching a pattern.

        Args:
            pattern: Key pattern (e.g., "user:*" to invalidate all user cache)

        Returns:
            Number of keys deleted
        """
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context")

        client = await self._get_client()
        full_pattern = f"{GLOBAL_PREFIX}:{tenant.tenant_slug}:cache:{pattern}"

        # Use SCAN to find matching keys (non-blocking)
        deleted = 0
        async for key in client.scan_iter(match=full_pattern):
            await client.delete(key)
            deleted += 1

        return deleted

    # ============ Tenant Operations ============

    async def delete_tenant_data(self) -> int:
        """
        Delete ALL Redis data for the current tenant.

        Use with extreme caution - this removes all cached data,
        sessions, counters, etc. for the tenant.

        Returns:
            Number of keys deleted
        """
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context")

        client = await self._get_client()
        pattern = f"{GLOBAL_PREFIX}:{tenant.tenant_slug}:*"

        deleted = 0
        async for key in client.scan_iter(match=pattern):
            await client.delete(key)
            deleted += 1

        logger.warning(f"Deleted {deleted} Redis keys for tenant {tenant.tenant_slug}")
        return deleted

    async def get_tenant_key_count(self) -> int:
        """Get count of all keys for current tenant."""
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context")

        client = await self._get_client()
        pattern = f"{GLOBAL_PREFIX}:{tenant.tenant_slug}:*"

        count = 0
        async for _ in client.scan_iter(match=pattern):
            count += 1

        return count

    # ============ Health & Utilities ============

    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def close(self):
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._client = None
        if self._pool:
            await self._pool.disconnect()
            self._pool = None


# Singleton instance
_redis_service: Optional[RedisService] = None


def get_redis_service() -> RedisService:
    """Get or create Redis service singleton."""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service
