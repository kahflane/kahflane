"""
SQLModel base configuration and metadata.

This module sets up the base SQLModel configuration used across all models.
"""
from sqlmodel import SQLModel

# Import all models here to ensure they are registered with SQLModel metadata
# This is required for Alembic migrations and table creation

# Public schema models (imported first as they're referenced by tenant models)
from app.models.tenant import Tenant, User, TenantMembership, TenantInvitation

# Tenant schema models
from app.models.organization import Organization, OrganizationMember, Team, TeamMember
from app.models.document import Document, ProcessingLog

__all__ = [
    "SQLModel",
    # Public schema
    "Tenant",
    "User",
    "TenantMembership",
    "TenantInvitation",
    # Tenant schema
    "Organization",
    "OrganizationMember",
    "Team",
    "TeamMember",
    "Document",
    "ProcessingLog",
]
