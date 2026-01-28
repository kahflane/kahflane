"""
Search API endpoints for semantic document search.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import Optional, List
import logging

from app.db.session import get_db
from app.api.deps import get_current_user
from app.api.schemas import ErrorResponse
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

router = APIRouter()


# Request Schemas


class SearchRequest(BaseModel):
    """Request for semantic search across documents."""

    query: str = Field(
        min_length=1,
        max_length=2000,
        examples=["How does the onboarding process work?"],
    )
    org_id: Optional[str] = Field(
        default=None,
        examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"],
        description="Filter by organization ID",
    )
    team_id: Optional[str] = Field(
        default=None,
        examples=["d4e5f6a7-b8c9-0123-4567-890abcdef012"],
        description="Filter by team ID",
    )
    scope: Optional[str] = Field(
        default=None,
        examples=["team"],
        description="Filter by scope: personal, team, organization",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        examples=[10],
        description="Maximum number of results to return",
    )
    score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        examples=[0.7],
        description="Minimum similarity score (0.0 to 1.0)",
    )


class DocumentSearchRequest(BaseModel):
    """Request for search within a single document."""

    query: str = Field(
        min_length=1,
        max_length=2000,
        examples=["What are the key findings?"],
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        examples=[10],
        description="Maximum number of results to return",
    )
    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        examples=[0.5],
        description="Minimum similarity score (0.0 to 1.0)",
    )


# Response Schemas


class SearchResultDocument(BaseModel):
    """Document metadata in search result."""

    id: str = Field(examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"])
    title: str = Field(examples=["Q4 Financial Report"])
    file_type: str = Field(examples=["PDF"])
    scope: str = Field(examples=["team"])
    org_id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    team_id: Optional[str] = Field(
        default=None, examples=["d4e5f6a7-b8c9-0123-4567-890abcdef012"]
    )


class SearchResult(BaseModel):
    """A single search result."""

    text: str = Field(examples=["The onboarding process consists of three phases..."])
    score: float = Field(examples=[0.92])
    chunk_index: Optional[int] = Field(default=None, examples=[3])
    document: SearchResultDocument


class SearchResponse(BaseModel):
    """Search response containing results."""

    results: List[SearchResult]
    total: int = Field(examples=[5])
    query: str = Field(examples=["How does the onboarding process work?"])


# Endpoints


@router.post(
    "/",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Search documents",
    description="Semantic search across documents accessible to the current user.",
)
async def search_documents(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Semantic search across documents accessible to the current user.

    Searches all documents the user has permission to view,
    filtered by optional org_id, team_id, and scope parameters.
    Results are ranked by semantic similarity to the query.

    Access control:
    - personal scope: only the uploader can see
    - team scope: only team members can see
    - organization scope: any organization member can see
    """
    try:
        service = SearchService(db=db, user_id=current_user["user_id"])
        results = await service.search(
            query=request.query,
            org_id=request.org_id,
            team_id=request.team_id,
            scope=request.scope,
            limit=request.limit,
            score_threshold=request.score_threshold,
        )

        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            total=len(results),
            query=request.query,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed. Please try again.",
        )


@router.post(
    "/documents/{document_id}",
    response_model=SearchResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Search within document",
    description="Semantic search within a specific document.",
)
async def search_within_document(
    document_id: str,
    request: DocumentSearchRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Semantic search within a specific document.

    The user must have access to the document based on its scope.
    Results are chunks from the document ranked by semantic similarity.
    """
    try:
        service = SearchService(db=db, user_id=current_user["user_id"])
        results = await service.search_document(
            document_id=document_id,
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold,
        )

        return SearchResponse(
            results=[SearchResult(**r) for r in results],
            total=len(results),
            query=request.query,
        )

    except HTTPException:
        # Re-raise 404/403 from service
        raise
    except Exception as e:
        logger.error(f"Document search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed. Please try again.",
        )
