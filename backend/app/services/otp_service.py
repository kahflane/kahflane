"""
Redis-backed OTP service for email verification and password reset.

Provides both sync (for Dramatiq workers) and async (for FastAPI endpoints) classes.
"""
import secrets
import redis
import redis.asyncio
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class OTPService:
    """Synchronous Redis OTP service for use in Dramatiq workers."""

    def __init__(self):
        self.client = redis.from_url(settings.REDIS_URL)

    def generate_otp(self, purpose: str, identifier: str) -> str:
        """Generate a 6-digit OTP and store in Redis with TTL."""
        otp = f"{secrets.randbelow(1000000):06d}"
        key = f"otp:{purpose}:{identifier}"
        self.client.setex(key, settings.OTP_EXPIRE_MINUTES * 60, otp)
        logger.info(f"Generated OTP for {purpose}:{identifier}")
        return otp

    def verify_otp(self, purpose: str, identifier: str, otp: str) -> bool:
        """Verify and consume an OTP (single-use)."""
        key = f"otp:{purpose}:{identifier}"
        stored = self.client.get(key)
        if stored and stored.decode() == otp:
            self.client.delete(key)
            return True
        return False


class AsyncOTPService:
    """Async Redis OTP service for use in FastAPI endpoints."""

    def __init__(self):
        self.client = redis.asyncio.from_url(settings.REDIS_URL)

    async def generate_otp(self, purpose: str, identifier: str) -> str:
        """Generate a 6-digit OTP and store in Redis with TTL."""
        otp = f"{secrets.randbelow(1000000):06d}"
        key = f"otp:{purpose}:{identifier}"
        await self.client.setex(key, settings.OTP_EXPIRE_MINUTES * 60, otp)
        logger.info(f"Generated OTP for {purpose}:{identifier}")
        return otp

    async def verify_otp(self, purpose: str, identifier: str, otp: str) -> bool:
        """Verify and consume an OTP (single-use)."""
        key = f"otp:{purpose}:{identifier}"
        stored = await self.client.get(key)
        if stored and stored.decode() == otp:
            await self.client.delete(key)
            return True
        return False


def get_async_otp_service() -> AsyncOTPService:
    return AsyncOTPService()
