"""
Tenant schema models for organization and team management.

These models live in each tenant's dedicated schema (e.g., tenant_acme).
The schema is determined at runtime via schema_translate_map.
"""
from datetime import datetime, timezone
from typing import Optional, List, TYPE_CHECKING
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, String, Boolean, DateTime, Text, text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB

if TYPE_CHECKING:
    from app.models.document import Document


class Organization(SQLModel, table=True):
    """
    Organization within a tenant.

    Each tenant can have multiple organizations, though typically
    a tenant represents a single organization.
    """
    __tablename__ = "organization"
    # No schema specified - will use schema_translate_map at runtime

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    name: str = Field(
        max_length=255,
        sa_column=Column(String(255), nullable=False)
    )
    description: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    logo_url: Optional[str] = Field(
        default=None,
        max_length=500,
        sa_column=Column(String(500), nullable=True)
    )
    settings: dict = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default=text("'{}'")),
        description="Organization-specific settings"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )

    # Relationships
    members: List["OrganizationMember"] = Relationship(back_populates="organization")
    teams: List["Team"] = Relationship(back_populates="organization")
    documents: List["Document"] = Relationship(back_populates="organization")


class OrganizationMember(SQLModel, table=True):
    """
    Organization membership within a tenant.

    Links users (from public.user) to organizations within the tenant.
    Note: user_id references public.user but we don't use FK constraint
    across schemas.
    """
    __tablename__ = "organization_member"

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    org_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("organization.id"), nullable=False, index=True)
    )
    user_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), nullable=False, index=True),
        description="References public.user.id (no FK constraint across schemas)"
    )
    role: str = Field(
        default="member",
        max_length=50,
        sa_column=Column(String(50), nullable=False, default="member"),
        description="User's role: owner, admin, member"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )

    # Relationships (within tenant schema only)
    organization: "Organization" = Relationship(back_populates="members")


class Team(SQLModel, table=True):
    """
    Team within an organization.

    Teams group users for document access control.
    """
    __tablename__ = "team"

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    org_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("organization.id"), nullable=False, index=True)
    )
    name: str = Field(
        max_length=255,
        sa_column=Column(String(255), nullable=False)
    )
    description: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )

    # Relationships
    organization: "Organization" = Relationship(back_populates="teams")
    members: List["TeamMember"] = Relationship(back_populates="team")
    documents: List["Document"] = Relationship(back_populates="team")


class TeamMember(SQLModel, table=True):
    """
    Team membership within a tenant.

    Links users to teams for access control.
    """
    __tablename__ = "team_member"

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    team_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("team.id"), nullable=False, index=True)
    )
    user_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), nullable=False, index=True),
        description="References public.user.id"
    )
    role: str = Field(
        default="member",
        max_length=50,
        sa_column=Column(String(50), nullable=False, default="member"),
        description="User's role: lead, member"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )

    # Relationships
    team: "Team" = Relationship(back_populates="members")
