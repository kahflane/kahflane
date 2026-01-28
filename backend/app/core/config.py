"""
Application configuration using pydantic-settings.

Environment variables are loaded from .env file and can be overridden
by actual environment variables.
"""
from typing import Optional, List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import computed_field
from functools import lru_cache
from urllib.parse import quote_plus


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    APP_NAME: str = "Kahflane"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Multi-tenancy
    BASE_DOMAIN: str = "kahflane.com"

    # Security
    SECRET_KEY: str = "change-me-in-production-use-openssl-rand-hex-32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Database
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_USER: str = "kahflane"
    DB_PASSWORD: str = "kahflane"
    DB_NAME: str = "kahflane"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_USER: Optional[str] = None
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "kahflane_documents"

    # AI Services
    GOOGLE_API_KEY: Optional[str] = None
    ELEVENLABS_API_KEY: Optional[str] = None

    # Storage
    STORAGE_TYPE: str = "s3"  # "local" or "s3"
    STORAGE_LOCAL_PATH: str = "./uploads"
    S3_BUCKET: Optional[str] = None
    S3_REGION: Optional[str] = None
    S3_ENDPOINT: Optional[str] = None  # For S3-compatible services
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # SMTP
    SMTP_HOST: str = "localhost"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: str = "noreply@kahflane.com"
    SMTP_USE_TLS: bool = True

    # OTP
    OTP_EXPIRE_MINUTES: int = 10
    INVITATION_EXPIRE_DAYS: int = 7

    # Frontend (for email links)
    FRONTEND_BASE_URL: str = "https://kahflane.com"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5000",
        "http://localhost:8000",
        "https://*.kahflane.com",
    ]

    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        """Construct database URL from components."""
        password = quote_plus(self.DB_PASSWORD)
        return f"postgresql+asyncpg://{self.DB_USER}:{password}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @computed_field
    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL from components."""
        if self.REDIS_USER and self.REDIS_PASSWORD:
            password = quote_plus(self.REDIS_PASSWORD)
            return f"redis://{self.REDIS_USER}:{password}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        elif self.REDIS_PASSWORD:
            password = quote_plus(self.REDIS_PASSWORD)
            return f"redis://:{password}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        else:
            return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @computed_field
    @property
    def QDRANT_URL(self) -> str:
        """Construct Qdrant URL from components."""
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def async_database_url(self) -> str:
        """Alias for DATABASE_URL (already async)."""
        return self.DATABASE_URL


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Using lru_cache ensures settings are only loaded once.
    """
    return Settings()


# Global settings instance
settings = get_settings()
