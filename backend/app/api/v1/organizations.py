"""
Organization and Team management API endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID

from app.db.session import get_db
from app.models.organization import Organization, OrganizationMember, Team, TeamMember
from app.api.deps import get_current_user, require_admin
from app.api.schemas import ErrorResponse

router = APIRouter()


# Schemas

class OrganizationResponse(BaseModel):
    """Organization information."""
    id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    name: str = Field(examples=["Engineering"])
    description: Optional[str] = Field(default=None, examples=["Engineering department"])
    logo_url: Optional[str] = Field(default=None, examples=["https://cdn.kahflane.com/logos/eng.png"])
    settings: dict = Field(examples=[{"max_members": 50}])

    class Config:
        from_attributes = True


class OrganizationMemberResponse(BaseModel):
    """Organization member info."""
    id: str = Field(examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"])
    user_id: str = Field(examples=["550e8400-e29b-41d4-a716-446655440000"])
    role: str = Field(examples=["admin"])


class TeamResponse(BaseModel):
    """Team information."""
    id: str = Field(examples=["d4e5f6a7-b8c9-0123-4567-890abcdef012"])
    org_id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    name: str = Field(examples=["Backend Team"])
    description: Optional[str] = Field(default=None, examples=["Backend engineering team"])

    class Config:
        from_attributes = True


class TeamMemberResponse(BaseModel):
    """Team member info."""
    id: str = Field(examples=["b2c3d4e5-f6a7-8901-2345-67890abcdef0"])
    user_id: str = Field(examples=["550e8400-e29b-41d4-a716-446655440000"])
    role: str = Field(examples=["lead"])


class CreateTeamRequest(BaseModel):
    """Request to create a team."""
    name: str = Field(min_length=1, max_length=255, examples=["Backend Team"])
    description: Optional[str] = Field(default=None, examples=["Backend engineering team"])


class AddMemberRequest(BaseModel):
    """Request to add a member."""
    user_id: str = Field(examples=["550e8400-e29b-41d4-a716-446655440000"])
    role: str = Field(default="member", examples=["member"])


# Organization Endpoints

@router.get("/", response_model=List[OrganizationResponse])
async def list_organizations(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List organizations in current tenant that user has access to.
    """
    user_id = UUID(current_user["user_id"])

    result = await db.execute(
        select(Organization)
        .join(OrganizationMember, Organization.id == OrganizationMember.org_id)
        .where(OrganizationMember.user_id == user_id)
    )
    orgs = result.scalars().all()

    return [
        OrganizationResponse(
            id=str(org.id),
            name=org.name,
            description=org.description,
            logo_url=org.logo_url,
            settings=org.settings,
        )
        for org in orgs
    ]


@router.get("/{org_id}", response_model=OrganizationResponse, responses={403: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def get_organization(
    org_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get organization details.
    """
    # Verify user is member
    result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.org_id == UUID(org_id),
            OrganizationMember.user_id == UUID(current_user["user_id"]),
        )
    )
    if not result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not a member of this organization",
        )

    result = await db.execute(
        select(Organization).where(Organization.id == UUID(org_id))
    )
    org = result.scalar_one_or_none()

    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    return OrganizationResponse(
        id=str(org.id),
        name=org.name,
        description=org.description,
        logo_url=org.logo_url,
        settings=org.settings,
    )


@router.get("/{org_id}/members", response_model=List[OrganizationMemberResponse])
async def list_organization_members(
    org_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List organization members.
    """
    result = await db.execute(
        select(OrganizationMember).where(OrganizationMember.org_id == UUID(org_id))
    )
    members = result.scalars().all()

    return [
        OrganizationMemberResponse(
            id=str(m.id),
            user_id=str(m.user_id),
            role=m.role,
        )
        for m in members
    ]


# Team Endpoints

@router.get("/{org_id}/teams", response_model=List[TeamResponse])
async def list_teams(
    org_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List teams in an organization.
    """
    result = await db.execute(
        select(Team).where(Team.org_id == UUID(org_id))
    )
    teams = result.scalars().all()

    return [
        TeamResponse(
            id=str(t.id),
            org_id=str(t.org_id),
            name=t.name,
            description=t.description,
        )
        for t in teams
    ]


@router.post("/{org_id}/teams", response_model=TeamResponse, status_code=status.HTTP_201_CREATED)
async def create_team(
    org_id: str,
    request: CreateTeamRequest,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new team in an organization.

    Requires admin role.
    """
    team = Team(
        org_id=UUID(org_id),
        name=request.name,
        description=request.description,
    )
    db.add(team)
    await db.flush()

    # Add creator as team lead
    member = TeamMember(
        team_id=team.id,
        user_id=UUID(current_user["user_id"]),
        role="lead",
    )
    db.add(member)
    await db.commit()

    return TeamResponse(
        id=str(team.id),
        org_id=str(team.org_id),
        name=team.name,
        description=team.description,
    )


@router.get("/{org_id}/teams/{team_id}", response_model=TeamResponse, responses={404: {"model": ErrorResponse}})
async def get_team(
    org_id: str,
    team_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get team details.
    """
    result = await db.execute(
        select(Team).where(
            Team.id == UUID(team_id),
            Team.org_id == UUID(org_id),
        )
    )
    team = result.scalar_one_or_none()

    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Team not found",
        )

    return TeamResponse(
        id=str(team.id),
        org_id=str(team.org_id),
        name=team.name,
        description=team.description,
    )


@router.get("/{org_id}/teams/{team_id}/members", response_model=List[TeamMemberResponse])
async def list_team_members(
    org_id: str,
    team_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List team members.
    """
    result = await db.execute(
        select(TeamMember).where(TeamMember.team_id == UUID(team_id))
    )
    members = result.scalars().all()

    return [
        TeamMemberResponse(
            id=str(m.id),
            user_id=str(m.user_id),
            role=m.role,
        )
        for m in members
    ]


@router.post("/{org_id}/members", response_model=OrganizationMemberResponse, status_code=status.HTTP_201_CREATED, responses={400: {"model": ErrorResponse}})
async def add_organization_member(
    org_id: str,
    request: AddMemberRequest,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Add a member to an organization. The user must already be a tenant member.

    Requires admin role.
    """
    user_id = UUID(request.user_id)

    # Check if already an org member
    result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.org_id == UUID(org_id),
            OrganizationMember.user_id == user_id,
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already an organization member",
        )

    member = OrganizationMember(
        org_id=UUID(org_id),
        user_id=user_id,
        role=request.role,
    )
    db.add(member)
    await db.commit()

    return OrganizationMemberResponse(
        id=str(member.id),
        user_id=str(member.user_id),
        role=member.role,
    )


@router.delete("/{org_id}/members/{user_id}", status_code=status.HTTP_200_OK, responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}})
async def remove_organization_member(
    org_id: str,
    user_id: str,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a member from an organization and all its teams.

    Requires admin role.
    """
    target_user_id = UUID(user_id)
    org_uuid = UUID(org_id)

    # Find org membership
    result = await db.execute(
        select(OrganizationMember).where(
            OrganizationMember.org_id == org_uuid,
            OrganizationMember.user_id == target_user_id,
        )
    )
    org_member = result.scalar_one_or_none()

    if not org_member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this organization",
        )

    # Remove from all teams in this org
    teams_result = await db.execute(
        select(Team).where(Team.org_id == org_uuid)
    )
    team_ids = [t.id for t in teams_result.scalars().all()]

    if team_ids:
        team_members_result = await db.execute(
            select(TeamMember).where(
                TeamMember.team_id.in_(team_ids),
                TeamMember.user_id == target_user_id,
            )
        )
        for tm in team_members_result.scalars().all():
            await db.delete(tm)

    # Remove org membership
    await db.delete(org_member)
    await db.commit()

    return {"message": "Member removed from organization and all its teams"}


@router.post("/{org_id}/teams/{team_id}/members", response_model=TeamMemberResponse, status_code=status.HTTP_201_CREATED, responses={400: {"model": ErrorResponse}})
async def add_team_member(
    org_id: str,
    team_id: str,
    request: AddMemberRequest,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Add a member to a team.

    Requires admin role.
    """
    # Check if already a member
    result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == UUID(team_id),
            TeamMember.user_id == UUID(request.user_id),
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User is already a team member",
        )

    member = TeamMember(
        team_id=UUID(team_id),
        user_id=UUID(request.user_id),
        role=request.role,
    )
    db.add(member)
    await db.commit()

    return TeamMemberResponse(
        id=str(member.id),
        user_id=str(member.user_id),
        role=member.role,
    )


@router.delete("/{org_id}/teams/{team_id}/members/{user_id}", status_code=status.HTTP_200_OK, responses={404: {"model": ErrorResponse}})
async def remove_team_member(
    org_id: str,
    team_id: str,
    user_id: str,
    current_user: dict = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    """
    Remove a member from a team.

    Requires admin role.
    """
    result = await db.execute(
        select(TeamMember).where(
            TeamMember.team_id == UUID(team_id),
            TeamMember.user_id == UUID(user_id),
        )
    )
    member = result.scalar_one_or_none()

    if not member:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User is not a member of this team",
        )

    await db.delete(member)
    await db.commit()

    return {"message": "Member removed from team"}
