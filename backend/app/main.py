"""
Kahflane API - FastAPI Application Entry Point

Multi-tenant Knowledge Management System API.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.core.config import settings
from app.middleware.tenant import TenantMiddleware
from app.api.v1 import router as api_v1_router
from app.db.session import engine

import app.db.base  # noqa: F401

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("Starting Kahflane API...")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Base domain: {settings.BASE_DOMAIN}")

    from sqlmodel import SQLModel
    public_tables = [
        t for t in SQLModel.metadata.sorted_tables
        if t.schema == "public"
    ]
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all, tables=public_tables)
    logger.info("Public schema tables ensured.")

    try:
        yield
    finally:
        logger.info("Shutting down Kahflane API...")
        await engine.dispose()


# Create FastAPI application
app = FastAPI(
    title="Kahflane API",
    description="AI-powered Knowledge Management System with Multi-Tenant Architecture",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "auth", "description": "Authentication and user management"},
        {"name": "tenants", "description": "Tenant management"},
        {"name": "organizations", "description": "Organization and team management"},
        {"name": "documents", "description": "Document upload, management, and retrieval"},
    ],
)

# Add CORS middleware - must be added before other middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Tenant-ID", "X-Tenant-Slug"],
)

# Add tenant middleware for subdomain-based tenant resolution
app.add_middleware(TenantMiddleware, base_domain=settings.BASE_DOMAIN)


# Include API routes
app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)


# Health check endpoints (excluded from tenant middleware)

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "kahflane-api",
        "version": "1.0.0",
    }


@app.get("/healthz")
async def healthz():
    """Kubernetes-style health check."""
    return {"status": "ok"}


@app.get("/ready")
async def readiness_check():
    """
    Readiness check for Kubernetes.

    Verifies connectivity to all required services.
    """
    import redis.asyncio as redis
    from sqlalchemy import text
    from app.db.session import get_public_session
    from app.services.vector_store import get_vector_store

    checks = {}
    all_healthy = True

    # Check Database
    try:
        async with get_public_session() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)[:100]}"
        all_healthy = False
        logger.error(f"Database health check failed: {e}")

    # Check Redis
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        await redis_client.close()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)[:100]}"
        all_healthy = False
        logger.error(f"Redis health check failed: {e}")

    # Check Qdrant
    try:
        vector_store = get_vector_store()
        is_healthy = await vector_store.health_check()
        checks["qdrant"] = "ok" if is_healthy else "unhealthy"
        if not is_healthy:
            all_healthy = False
    except Exception as e:
        checks["qdrant"] = f"error: {str(e)[:100]}"
        all_healthy = False
        logger.error(f"Qdrant health check failed: {e}")

    status_code = 200 if all_healthy else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_healthy else "not_ready",
            "checks": checks,
        }
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Kahflane API",
        "version": "1.0.0",
        "docs": "/docs" if settings.DEBUG else "disabled",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
