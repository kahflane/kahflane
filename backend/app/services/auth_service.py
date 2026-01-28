"""
Authentication service.

Handles user registration, login, and token management.
"""
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import User, Tenant, TenantMembership
from app.core.security import (
    get_password_hash,
    verify_password,
    create_token_pair,
    TokenPair,
)
from app.core.tenant import TenantContext, generate_schema_name
from app.services.tenant_provisioning import (
    provision_tenant_schema,
    create_default_organization,
)
from app.db.session import get_public_session
import logging

logger = logging.getLogger(__name__)


class AuthService:
    """Service for authentication operations."""

    def __init__(self, session: AsyncSession):
        """
        Initialize auth service with a database session.

        Args:
            session: Public schema database session
        """
        self.session = session

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email address.

        Args:
            email: User's email address

        Returns:
            User if found, None otherwise
        """
        result = await self.session.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()

    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """
        Get user by ID.

        Args:
            user_id: User's UUID

        Returns:
            User if found, None otherwise
        """
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def create_user(
        self,
        email: str,
        password: str,
        full_name: str,
    ) -> User:
        """
        Create a new user.

        Args:
            email: User's email address
            password: Plain text password (will be hashed)
            full_name: User's display name

        Returns:
            Created User instance
        """
        user = User(
            email=email.lower(),
            password_hash=get_password_hash(password),
            full_name=full_name,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def authenticate(
        self,
        email: str,
        password: str,
    ) -> Optional[User]:
        """
        Authenticate user with email and password.

        Args:
            email: User's email address
            password: Plain text password

        Returns:
            User if credentials are valid, None otherwise
        """
        user = await self.get_user_by_email(email)
        if not user:
            return None

        if not verify_password(password, user.password_hash):
            return None

        if not user.is_active:
            return None

        # Update last login
        user.last_login_at = datetime.now(timezone.utc)
        await self.session.flush()

        return user

    async def get_tenant_by_slug(self, slug: str) -> Optional[Tenant]:
        """
        Get tenant by subdomain slug.

        Args:
            slug: Tenant's subdomain

        Returns:
            Tenant if found, None otherwise
        """
        result = await self.session.execute(
            select(Tenant).where(Tenant.slug == slug.lower())
        )
        return result.scalar_one_or_none()

    async def get_user_membership(
        self,
        user_id: UUID,
        tenant_id: UUID,
    ) -> Optional[TenantMembership]:
        """
        Get user's membership in a tenant.

        Args:
            user_id: User's UUID
            tenant_id: Tenant's UUID

        Returns:
            TenantMembership if exists, None otherwise
        """
        result = await self.session.execute(
            select(TenantMembership).where(
                TenantMembership.user_id == user_id,
                TenantMembership.tenant_id == tenant_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_user_tenants(self, user_id: UUID) -> list[dict]:
        """
        Get all tenants a user has access to.

        Args:
            user_id: User's UUID

        Returns:
            List of tenant info dicts
        """
        result = await self.session.execute(
            select(TenantMembership, Tenant)
            .join(Tenant, TenantMembership.tenant_id == Tenant.id)
            .where(TenantMembership.user_id == user_id)
        )
        rows = result.all()

        return [
            {
                "tenant_id": str(tenant.id),
                "slug": tenant.slug,
                "name": tenant.name,
                "role": membership.role,
                "is_default": membership.is_default,
            }
            for membership, tenant in rows
        ]

    async def create_tenant_with_owner(
        self,
        tenant_name: str,
        owner_email: str,
        owner_password: str,
        owner_full_name: str,
    ) -> tuple[Tenant, User]:
        """
        Create a new tenant with its first owner.

        This is the main registration flow for new organizations.

        Args:
            tenant_name: Display name for the tenant
            owner_email: Owner's email address
            owner_password: Owner's password
            owner_full_name: Owner's display name

        Returns:
            Tuple of (Tenant, User) created

        Raises:
            ValueError: If email already exists or slug is taken
        """
        # Generate slug from tenant name
        slug = tenant_name.lower().replace(" ", "-")
        # Remove any non-alphanumeric characters except hyphens
        slug = "".join(c for c in slug if c.isalnum() or c == "-")
        schema_name = generate_schema_name(slug)

        # Check if email already exists
        existing_user = await self.get_user_by_email(owner_email)
        if existing_user:
            raise ValueError("Email address already registered")

        # Check if slug already exists
        existing_tenant = await self.get_tenant_by_slug(slug)
        if existing_tenant:
            raise ValueError(f"Tenant slug '{slug}' is already taken")

        # Create user
        user = await self.create_user(
            email=owner_email,
            password=owner_password,
            full_name=owner_full_name,
        )

        # Create tenant
        tenant = Tenant(
            slug=slug,
            name=tenant_name,
            schema_name=schema_name,
        )
        self.session.add(tenant)
        await self.session.flush()

        # Create membership
        membership = TenantMembership(
            user_id=user.id,
            tenant_id=tenant.id,
            role="owner",
            is_default=True,
        )
        self.session.add(membership)
        await self.session.flush()

        # Provision schema (outside transaction to allow separate commit)
        await provision_tenant_schema(schema_name)

        # Create default organization in tenant schema
        tenant_context = TenantContext(
            tenant_id=str(tenant.id),
            tenant_slug=tenant.slug,
            schema_name=tenant.schema_name,
        )
        await create_default_organization(
            tenant_context=tenant_context,
            org_name=tenant_name,
            owner_user_id=user.id,
        )

        logger.info(f"Created tenant '{tenant_name}' (slug: {slug}) with owner {owner_email}")

        return tenant, user

    async def login_to_tenant(
        self,
        email: str,
        password: str,
        tenant_slug: str,
    ) -> Optional[tuple[User, Tenant, TenantMembership, TokenPair]]:
        """
        Authenticate user and verify access to a specific tenant.

        Args:
            email: User's email address
            password: User's password
            tenant_slug: Tenant subdomain to log into

        Returns:
            Tuple of (User, Tenant, Membership, Tokens) if successful, None otherwise
        """
        # Authenticate user
        user = await self.authenticate(email, password)
        if not user:
            return None

        # Get tenant
        tenant = await self.get_tenant_by_slug(tenant_slug)
        if not tenant or not tenant.is_active:
            return None

        # Check membership
        membership = await self.get_user_membership(user.id, tenant.id)
        if not membership:
            return None

        # Create tokens
        tokens = create_token_pair(
            user_id=str(user.id),
            tenant_id=str(tenant.id),
            tenant_slug=tenant.slug,
            role=membership.role,
        )

        return user, tenant, membership, tokens
