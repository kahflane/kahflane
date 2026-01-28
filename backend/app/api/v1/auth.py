"""
Authentication API endpoints.

Handles user registration, login, token refresh, email verification,
password reset, and password change.
"""
from fastapi import APIRouter, HTTPException, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from uuid import UUID

from app.db.session import get_public_db
from app.services.auth_service import AuthService
from app.services.otp_service import get_async_otp_service, AsyncOTPService
from app.core.security import (
    decode_token,
    create_token_pair,
    get_password_hash,
    verify_password,
)
from app.core.tenant import get_current_tenant
from app.api.deps import get_current_user
from app.api.schemas import ErrorResponse

router = APIRouter()


# Request/Response schemas

class RegisterRequest(BaseModel):
    """Request body for new tenant registration."""
    email: EmailStr = Field(examples=["john@example.com"])
    password: str = Field(min_length=8, max_length=128, examples=["SecureP@ss123"])
    full_name: str = Field(min_length=1, max_length=255, examples=["John Doe"])
    tenant_name: str = Field(min_length=2, max_length=255, examples=["Acme Corp"])


class RegisterResponse(BaseModel):
    """Response for successful registration."""
    user_id: str = Field(examples=["550e8400-e29b-41d4-a716-446655440000"])
    email: str = Field(examples=["john@example.com"])
    tenant_id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    tenant_slug: str = Field(examples=["acme-corp"])
    message: str = Field(examples=["Account created. Please verify your email."])


class LoginRequest(BaseModel):
    """Request body for login."""
    email: EmailStr = Field(examples=["john@example.com"])
    password: str = Field(examples=["SecureP@ss123"])


class LoginResponse(BaseModel):
    """Response for successful login."""
    access_token: str = Field(examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."])
    refresh_token: str = Field(examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."])
    token_type: str = Field(default="bearer", examples=["bearer"])
    tenant_slug: str = Field(examples=["acme-corp"])
    user: dict = Field(examples=[{"id": "550e8400-e29b-41d4-a716-446655440000", "email": "john@example.com", "full_name": "John Doe", "role": "owner"}])


class RefreshRequest(BaseModel):
    """Request body for token refresh."""
    refresh_token: str = Field(examples=["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."])


class TenantInfo(BaseModel):
    """Tenant information for user."""
    tenant_id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    slug: str = Field(examples=["acme-corp"])
    name: str = Field(examples=["Acme Corp"])
    role: str = Field(examples=["owner"])
    is_default: bool = Field(examples=[True])


class UserTenantsResponse(BaseModel):
    """Response listing user's tenants."""
    tenants: List[TenantInfo]


class MeResponse(BaseModel):
    """Response for current user info."""
    user_id: str = Field(examples=["550e8400-e29b-41d4-a716-446655440000"])
    email: str = Field(examples=["john@example.com"])
    full_name: str = Field(examples=["John Doe"])
    tenant_id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    tenant_slug: str = Field(examples=["acme-corp"])
    role: str = Field(examples=["owner"])


class MessageResponse(BaseModel):
    """Generic message response."""
    message: str


class VerifyEmailRequest(BaseModel):
    """Request to verify email with OTP."""
    email: EmailStr = Field(examples=["john@example.com"])
    otp: str = Field(min_length=6, max_length=6, examples=["123456"])


class ResendOTPRequest(BaseModel):
    """Request to resend verification OTP."""
    email: EmailStr = Field(examples=["john@example.com"])


class ForgotPasswordRequest(BaseModel):
    """Request to initiate password reset."""
    email: EmailStr = Field(examples=["john@example.com"])


class ResetPasswordRequest(BaseModel):
    """Request to reset password with OTP."""
    email: EmailStr = Field(examples=["john@example.com"])
    otp: str = Field(min_length=6, max_length=6, examples=["123456"])
    new_password: str = Field(min_length=8, max_length=128, examples=["NewSecureP@ss123"])


class ChangePasswordRequest(BaseModel):
    """Request to change password (authenticated)."""
    current_password: str = Field(examples=["OldP@ss123"])
    new_password: str = Field(min_length=8, max_length=128, examples=["NewSecureP@ss123"])


# Endpoints

@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED, responses={400: {"model": ErrorResponse}})
async def register(
    request: RegisterRequest,
    db: AsyncSession = Depends(get_public_db),
):
    """
    Register a new tenant and owner account.

    Creates user, tenant, membership, provisions schema, and sends
    a verification OTP to the provided email.
    """
    auth_service = AuthService(db)

    try:
        tenant, user = await auth_service.create_tenant_with_owner(
            tenant_name=request.tenant_name,
            owner_email=request.email,
            owner_password=request.password,
            owner_full_name=request.full_name,
        )
        await db.commit()
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Generate OTP and enqueue verification email
    otp_service = get_async_otp_service()
    otp = await otp_service.generate_otp("email_verify", user.email)

    from app.workers.tasks import send_verification_email
    send_verification_email.send(user.email, otp)

    return RegisterResponse(
        user_id=str(user.id),
        email=user.email,
        tenant_id=str(tenant.id),
        tenant_slug=tenant.slug,
        message="Account created. Please check your email for a verification code.",
    )


@router.post("/login", response_model=LoginResponse, responses={401: {"model": ErrorResponse}, 400: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
async def login(
    request: LoginRequest,
    http_request: Request,
    db: AsyncSession = Depends(get_public_db),
):
    """
    Login to a specific tenant.

    The tenant is determined by subdomain or X-Tenant-Slug header.
    Login is blocked until email is verified.
    """
    auth_service = AuthService(db)

    # Get tenant from request context (set by middleware)
    tenant_context = get_current_tenant()

    if not tenant_context:
        # No tenant specified - return user's available tenants
        user = await auth_service.authenticate(request.email, request.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )

        tenants = await auth_service.get_user_tenants(user.id)
        if not tenants:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User has no tenant access",
            )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Please login via tenant subdomain or specify X-Tenant-Slug header",
                "available_tenants": tenants,
            },
        )

    # Login to specific tenant
    result = await auth_service.login_to_tenant(
        email=request.email,
        password=request.password,
        tenant_slug=tenant_context.tenant_slug,
    )

    if not result:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials or no access to this tenant",
        )

    user, tenant, membership, tokens = result

    # Block login if email not verified
    if not user.is_email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email not verified. Please verify your email before logging in.",
        )

    await db.commit()

    return LoginResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        tenant_slug=tenant.slug,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": membership.role,
        },
    )


@router.post("/verify-email", response_model=MessageResponse, responses={400: {"model": ErrorResponse}})
async def verify_email(
    request: VerifyEmailRequest,
    db: AsyncSession = Depends(get_public_db),
):
    """
    Verify email address with OTP code.
    """
    otp_service = get_async_otp_service()
    valid = await otp_service.verify_otp("email_verify", request.email.lower(), request.otp)

    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification code",
        )

    auth_service = AuthService(db)
    user = await auth_service.get_user_by_email(request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not found",
        )

    user.is_email_verified = True
    await db.commit()

    return MessageResponse(message="Email verified successfully")


@router.post("/resend-otp", response_model=MessageResponse)
async def resend_otp(
    request: ResendOTPRequest,
    db: AsyncSession = Depends(get_public_db),
):
    """
    Resend email verification OTP.
    """
    auth_service = AuthService(db)
    user = await auth_service.get_user_by_email(request.email)

    if user and not user.is_email_verified:
        otp_service = get_async_otp_service()
        otp = await otp_service.generate_otp("email_verify", user.email)

        from app.workers.tasks import send_verification_email
        send_verification_email.send(user.email, otp)

    # Always return success to prevent email enumeration
    return MessageResponse(message="If the email exists and is not yet verified, a new code has been sent.")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    request: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_public_db),
):
    """
    Request a password reset OTP. Always returns success to prevent email enumeration.
    """
    auth_service = AuthService(db)
    user = await auth_service.get_user_by_email(request.email)

    if user and user.is_active:
        otp_service = get_async_otp_service()
        otp = await otp_service.generate_otp("password_reset", user.email)

        from app.workers.tasks import send_password_reset_email
        send_password_reset_email.send(user.email, otp)

    return MessageResponse(message="If the email exists, a password reset code has been sent.")


@router.post("/reset-password", response_model=MessageResponse, responses={400: {"model": ErrorResponse}})
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_public_db),
):
    """
    Reset password using OTP code.
    """
    otp_service = get_async_otp_service()
    valid = await otp_service.verify_otp("password_reset", request.email.lower(), request.otp)

    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset code",
        )

    auth_service = AuthService(db)
    user = await auth_service.get_user_by_email(request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not found",
        )

    user.password_hash = get_password_hash(request.new_password)
    await db.commit()

    return MessageResponse(message="Password reset successfully")


@router.post("/change-password", response_model=MessageResponse, responses={400: {"model": ErrorResponse}})
async def change_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_public_db),
):
    """
    Change password for authenticated user. Requires current password.
    """
    auth_service = AuthService(db)
    user = await auth_service.get_user_by_id(UUID(current_user["user_id"]))

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not found",
        )

    if not verify_password(request.current_password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect",
        )

    user.password_hash = get_password_hash(request.new_password)
    await db.commit()

    return MessageResponse(message="Password changed successfully")


@router.post("/refresh", response_model=LoginResponse, responses={401: {"model": ErrorResponse}})
async def refresh_token(
    request: RefreshRequest,
    db: AsyncSession = Depends(get_public_db),
):
    """
    Refresh access token using a refresh token.
    """
    try:
        payload = decode_token(request.refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    if payload.type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    # Verify user still exists and is active
    auth_service = AuthService(db)
    user = await auth_service.get_user_by_id(UUID(payload.sub))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    # Verify tenant membership still exists
    tenant = await auth_service.get_tenant_by_slug(payload.tenant_slug)
    if not tenant or not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant not found or inactive",
        )

    membership = await auth_service.get_user_membership(user.id, tenant.id)
    if not membership:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User no longer has access to this tenant",
        )

    # Create new tokens
    tokens = create_token_pair(
        user_id=str(user.id),
        tenant_id=str(tenant.id),
        tenant_slug=tenant.slug,
        role=membership.role,
    )

    return LoginResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        tenant_slug=tenant.slug,
        user={
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": membership.role,
        },
    )


@router.get("/me", response_model=MeResponse)
async def get_me(
    current_user: dict = Depends(get_current_user),
):
    """
    Get current authenticated user info.
    """
    return MeResponse(
        user_id=current_user["user_id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        tenant_id=current_user["tenant_id"],
        tenant_slug=current_user["tenant_slug"],
        role=current_user["role"],
    )


@router.get("/tenants", response_model=UserTenantsResponse)
async def get_user_tenants(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_public_db),
):
    """
    Get all tenants the current user has access to.
    """
    auth_service = AuthService(db)
    tenants = await auth_service.get_user_tenants(UUID(current_user["user_id"]))

    return UserTenantsResponse(
        tenants=[TenantInfo(**t) for t in tenants]
    )
