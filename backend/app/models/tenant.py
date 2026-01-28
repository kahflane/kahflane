"""
Public schema models for authentication and tenant management.

These models live in the 'public' schema and are shared across all tenants.
"""
from datetime import datetime, timezone
from typing import Optional, List, TYPE_CHECKING
from uuid import UUID, uuid4
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, String, Boolean, DateTime, text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB

if TYPE_CHECKING:
    pass


class Tenant(SQLModel, table=True):
    """
    Tenant registry in public schema.

    Each tenant represents an organization/company using the platform,
    identified by their unique subdomain.
    """
    __tablename__ = "tenant"
    __table_args__ = {"schema": "public"}

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    slug: str = Field(
        max_length=63,
        sa_column=Column(String(63), unique=True, index=True, nullable=False),
        description="Subdomain identifier (e.g., 'acme' for acme.kahflane.com)"
    )
    name: str = Field(
        max_length=255,
        sa_column=Column(String(255), nullable=False),
        description="Display name for the tenant"
    )
    domain: Optional[str] = Field(
        default=None,
        max_length=255,
        sa_column=Column(String(255), nullable=True),
        description="Optional custom domain"
    )
    schema_name: str = Field(
        max_length=63,
        sa_column=Column(String(63), unique=True, nullable=False),
        description="PostgreSQL schema name (e.g., 'tenant_acme')"
    )
    plan_type: str = Field(
        default="free",
        max_length=50,
        sa_column=Column(String(50), nullable=False, default="free"),
        description="Subscription tier (free, pro, enterprise)"
    )
    is_active: bool = Field(
        default=True,
        sa_column=Column(Boolean, nullable=False, default=True),
        description="Whether the tenant is active"
    )
    settings: dict = Field(
        default_factory=dict,
        sa_column=Column(JSONB, nullable=False, server_default=text("'{}'")),
        description="Tenant-specific settings"
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
    memberships: List["TenantMembership"] = Relationship(back_populates="tenant")
    invitations: List["TenantInvitation"] = Relationship(back_populates="tenant")


class User(SQLModel, table=True):
    """
    User authentication in public schema.

    Users can belong to multiple tenants via TenantMembership.
    Authentication is shared - users log in once and can switch tenants.
    """
    __tablename__ = "user"
    __table_args__ = {"schema": "public"}

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    email: str = Field(
        max_length=255,
        sa_column=Column(String(255), unique=True, index=True, nullable=False)
    )
    password_hash: str = Field(
        max_length=255,
        sa_column=Column(String(255), nullable=False)
    )
    full_name: str = Field(
        max_length=255,
        sa_column=Column(String(255), nullable=False)
    )
    is_active: bool = Field(
        default=True,
        sa_column=Column(Boolean, nullable=False, default=True)
    )
    is_email_verified: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False)
    )
    last_login_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True)
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
    memberships: List["TenantMembership"] = Relationship(
        back_populates="user",
        sa_relationship_kwargs={"foreign_keys": "TenantMembership.user_id"}
    )
    sent_invitations: List["TenantInvitation"] = Relationship(
        back_populates="invited_by_user",
        sa_relationship_kwargs={"foreign_keys": "TenantInvitation.invited_by_id"}
    )


class TenantMembership(SQLModel, table=True):
    """
    User-Tenant membership linking table.

    Defines which users have access to which tenants and their roles.
    """
    __tablename__ = "tenant_membership"
    __table_args__ = {"schema": "public"}

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    user_id: UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("public.user.id"),
            nullable=False,
            index=True
        )
    )
    tenant_id: UUID = Field(
        sa_column=Column(
            PG_UUID(as_uuid=True),
            ForeignKey("public.tenant.id"),
            nullable=False,
            index=True
        )
    )
    role: str = Field(
        default="member",
        max_length=50,
        sa_column=Column(String(50), nullable=False, default="member"),
        description="User's role: owner, admin, member"
    )
    is_default: bool = Field(
        default=False,
        sa_column=Column(Boolean, nullable=False, default=False),
        description="User's default tenant for login"
    )
    invited_by_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("public.user.id"), nullable=True)
    )
    joined_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )

    # Relationships
    user: "User" = Relationship(
        back_populates="memberships",
        sa_relationship_kwargs={"foreign_keys": "[TenantMembership.user_id]"}
    )
    tenant: "Tenant" = Relationship(back_populates="memberships")


class TenantInvitation(SQLModel, table=True):
    """
    Pending invitations to join a tenant.

    Invitations are sent via email with a unique token.
    """
    __tablename__ = "tenant_invitation"
    __table_args__ = {"schema": "public"}

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    tenant_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("public.tenant.id"), nullable=False, index=True)
    )
    email: str = Field(
        max_length=255,
        sa_column=Column(String(255), nullable=False, index=True)
    )
    role: str = Field(
        default="member",
        max_length=50,
        sa_column=Column(String(50), nullable=False, default="member")
    )
    token: str = Field(
        max_length=255,
        sa_column=Column(String(255), unique=True, nullable=False, index=True)
    )
    invited_by_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("public.user.id"), nullable=False)
    )
    expires_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), nullable=False)
    )
    accepted_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True)
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )

    # Relationships
    tenant: "Tenant" = Relationship(back_populates="invitations")
    invited_by_user: "User" = Relationship(back_populates="sent_invitations")
