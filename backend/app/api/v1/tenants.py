"""
Tenant management API endpoints.

Handles tenant info, settings, invitations, membership management.
"""
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from uuid import UUID

from app.db.session import get_public_db
from app.models.tenant import Tenant, TenantMembership, TenantInvitation, User
from app.api.deps import get_current_user, require_admin, require_owner
from app.api.schemas import ErrorResponse
from app.core.config import settings
from app.core.security import get_password_hash

router = APIRouter()


# Schemas

class TenantResponse(BaseModel):
    """Tenant information response."""
    id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    slug: str = Field(examples=["acme-corp"])
    name: str = Field(examples=["Acme Corp"])
    domain: Optional[str] = Field(default=None, examples=["acme.com"])
    plan_type: str = Field(examples=["free"])
    is_active: bool = Field(examples=[True])
    settings: dict = Field(examples=[{"feature_flags": {"ai_search": True}}])

    class Config:
        from_attributes = True


class UpdateTenantRequest(BaseModel):
    """Request to update tenant settings."""
    name: Optional[str] = Field(None, min_length=2, max_length=255, examples=["Acme Corporation"])
    domain: Optional[str] = Field(None, max_length=255, examples=["acme.com"])
    settings: Optional[dict] = Field(default=None, examples=[{"feature_flags": {"ai_search": True}}])


class InviteMemberRequest(BaseModel):
    """Request to invite a member to the tenant."""
    email: EmailStr = Field(examples=["jane@example.com"])
    role: str = Field(default="member", pattern="^(admin|member)$", examples=["member"])


class InvitationResponse(BaseModel):
    """Response for created invitation."""
    id: str
    email: str
    role: str
    token: str
    expires_at: str
    message: str


class AcceptInvitationRequest(BaseModel):
    """Request to accept a tenant invitation."""
    # Optional fields for new user registration
    password: Optional[str] = Field(None, min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, min_length=1, max_length=255)


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str


class SwitchDefaultTenantRequest(BaseModel):
    """Request to switch default tenant."""
    tenant_id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])


# Endpoints

@router.get("/{slug}", response_model=TenantResponse, responses={404: {"model": ErrorResponse}})
async def get_tenant(
    slug: str,
    db: AsyncSession = Depends(get_public_db),
):
    """
    Get tenant information by slug.

    This endpoint is public to allow checking tenant existence.
    """
    result = await db.execute(
        select(Tenant).where(Tenant.slug == slug.lower())
    )
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tenant '{slug}' not found",
        )

    return TenantResponse(
        id=str(tenant.id),
        slug=tenant.slug,
        name=tenant.name,
        domain=tenant.domain,
        plan_type=tenant.plan_type,
        is_active=tenant.is_active,
        settings=tenant.settings,
    )


@router.get("/", response_model=TenantResponse)
async def get_current_tenant_info(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_public_db),
):
    """
    Get current tenant information.

    Requires authentication.
    """
    result = await db.execute(
        select(Tenant).where(Tenant.id == UUID(current_user["tenant_id"]))
    )
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    return TenantResponse(
        id=str(tenant.id),
        slug=tenant.slug,
        name=tenant.name,
        domain=tenant.domain,
        plan_type=tenant.plan_type,
        is_active=tenant.is_active,
        settings=tenant.settings,
    )


@router.patch("/", response_model=TenantResponse)
async def update_tenant(
    request: UpdateTenantRequest,
    current_user: dict = Depends(require_owner),
    db: AsyncSession = Depends(get_public_db),
):
    """
    Update current tenant settings.

    Only tenant owners can update settings.
    """
    result = await db.execute(
        select(Tenant).where(Tenant.id == UUID(current_user["tenant_id"]))
    )
    tenant = result.scalar_one_or_none()

    if not tenant:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Tenant not found",
        )

    # Update fields
    if request.name is not None:
        tenant.name = request.name
    if request.domain is not None:
        tenant.domain = request.domain if request.domain else None
    if request.settings is not None:
        tenant.settings = request.settings

    await db.commit()
    await db.refresh(tenant)

    return TenantResponse(
        id=str(tenant.id),
        slug=tenant.slug,
        name=tenant.name,
        domain=tenant.domain,
        plan_type=tenant.plan_type,
        is_active=tenant.is_active,
        settings=tenant.settings,
    )


@router.post("/invite", response_model=InvitationResponse, status_code=status.HTTP_201_CREATED, responses={400: {"model": ErrorResponse}})
async def invite_member(
    request: InviteMemberRequest,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_public_db),
):
    """
    Invite a user to join the current tenant. Requires admin or owner role.
    """
    tenant_id = UUID(current_user["tenant_id"])

    # Check if user is already a member
    existing_user = await db.execute(
        select(User).where(User.email == request.email.lower())
    )
    user = existing_user.scalar_one_or_none()

    if user:
        existing_membership = await db.execute(
            select(TenantMembership).where(
                TenantMembership.user_id == user.id,
                TenantMembership.tenant_id == tenant_id,
            )
        )
        if existing_membership.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User is already a member of this tenant",
            )

    # Check for existing pending invitation
    existing_invite = await db.execute(
        select(TenantInvitation).where(
            TenantInvitation.tenant_id == tenant_id,
            TenantInvitation.email == request.email.lower(),
            TenantInvitation.accepted_at.is_(None),
            TenantInvitation.expires_at > datetime.now(timezone.utc),
        )
    )
    if existing_invite.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A pending invitation already exists for this email",
        )

    # Get tenant name for the email
    tenant_result = await db.execute(
        select(Tenant).where(Tenant.id == tenant_id)
    )
    tenant = tenant_result.scalar_one()

    # Create invitation
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.INVITATION_EXPIRE_DAYS)

    invitation = TenantInvitation(
        tenant_id=tenant_id,
        email=request.email.lower(),
        role=request.role,
        token=token,
        invited_by_id=UUID(current_user["user_id"]),
        expires_at=expires_at,
    )
    db.add(invitation)
    await db.commit()
    await db.refresh(invitation)

    # Enqueue invitation email
    from app.workers.tasks import send_invitation_email_task
    send_invitation_email_task.send(
        request.email.lower(),
        tenant.name,
        token,
        current_user["full_name"],
    )

    return InvitationResponse(
        id=str(invitation.id),
        email=invitation.email,
        role=invitation.role,
        token=invitation.token,
        expires_at=invitation.expires_at.isoformat(),
        message="Invitation sent successfully",
    )


@router.post("/invitations/{token}/accept", response_model=MessageResponse, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def accept_invitation(
    token: str,
    request: AcceptInvitationRequest,
    db: AsyncSession = Depends(get_public_db),
):
    """
    Accept a tenant invitation. If the user doesn't exist, password and
    full_name are required to create an account.
    """
    # Look up invitation
    result = await db.execute(
        select(TenantInvitation).where(TenantInvitation.token == token)
    )
    invitation = result.scalar_one_or_none()

    if not invitation:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invitation not found",
        )

    if invitation.accepted_at is not None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation has already been accepted",
        )

    if invitation.expires_at < datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invitation has expired",
        )

    # Find or create user
    user_result = await db.execute(
        select(User).where(User.email == invitation.email)
    )
    user = user_result.scalar_one_or_none()

    if not user:
        # New user — password and full_name required
        if not request.password or not request.full_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="password and full_name are required for new users",
            )
        user = User(
            email=invitation.email,
            password_hash=get_password_hash(request.password),
            full_name=request.full_name,
            is_email_verified=True,  # Verified by accepting email invitation
        )
        db.add(user)
        await db.flush()

    # Check if already a member (race condition guard)
    existing_membership = await db.execute(
        select(TenantMembership).where(
            TenantMembership.user_id == user.id,
            TenantMembership.tenant_id == invitation.tenant_id,
        )
    )
    if existing_membership.scalar_one_or_none():
        invitation.accepted_at = datetime.now(timezone.utc)
        await db.commit()
        return MessageResponse(message="You are already a member of this tenant")

    # Create membership
    membership = TenantMembership(
        user_id=user.id,
        tenant_id=invitation.tenant_id,
        role=invitation.role,
        invited_by_id=invitation.invited_by_id,
    )
    db.add(membership)

    # Mark invitation as accepted
    invitation.accepted_at = datetime.now(timezone.utc)
    await db.commit()

    return MessageResponse(message="Invitation accepted successfully")


@router.post("/leave", response_model=MessageResponse, responses={400: {"model": ErrorResponse}})
async def leave_tenant(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_public_db),
):
    """
    Leave the current tenant. Owners cannot leave if they are the last owner.
    """
    user_id = UUID(current_user["user_id"])
    tenant_id = UUID(current_user["tenant_id"])

    # Get membership
    result = await db.execute(
        select(TenantMembership).where(
            TenantMembership.user_id == user_id,
            TenantMembership.tenant_id == tenant_id,
        )
    )
    membership = result.scalar_one_or_none()

    if not membership:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not a member of this tenant",
        )

    # If owner, check there's at least one other owner
    if membership.role == "owner":
        owner_count = await db.execute(
            select(func.count()).select_from(TenantMembership).where(
                TenantMembership.tenant_id == tenant_id,
                TenantMembership.role == "owner",
            )
        )
        if owner_count.scalar_one() <= 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot leave: you are the only owner. Transfer ownership first.",
            )

    await db.delete(membership)
    await db.commit()

    return MessageResponse(message="Successfully left the tenant")


@router.put("/default", response_model=MessageResponse, responses={400: {"model": ErrorResponse}})
async def switch_default_tenant(
    request: SwitchDefaultTenantRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_public_db),
):
    """
    Switch the user's default tenant.
    """
    user_id = UUID(current_user["user_id"])
    target_tenant_id = UUID(request.tenant_id)

    # Verify membership in target tenant
    result = await db.execute(
        select(TenantMembership).where(
            TenantMembership.user_id == user_id,
            TenantMembership.tenant_id == target_tenant_id,
        )
    )
    target_membership = result.scalar_one_or_none()

    if not target_membership:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Not a member of the target tenant",
        )

    # Clear all defaults for this user
    all_memberships = await db.execute(
        select(TenantMembership).where(TenantMembership.user_id == user_id)
    )
    for m in all_memberships.scalars().all():
        m.is_default = False

    # Set new default
    target_membership.is_default = True
    await db.commit()

    return MessageResponse(message="Default tenant updated")
