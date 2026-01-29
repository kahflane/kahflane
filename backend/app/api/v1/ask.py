"""
Ask API endpoints for RAG-based Q&A over documents.

Provides natural language question-answering with source attribution.
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
import json

from app.db.session import get_db
from app.api.deps import get_current_user
from app.api.schemas import ErrorResponse
from app.services.rag_service import RAGService
from app.services.llm_service import GenerationConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# ============ Request Schemas ============


class AskRequest(BaseModel):
    """Request to ask a question over documents."""

    question: str = Field(
        min_length=1,
        max_length=2000,
        examples=["What are the key findings in the Q4 report?"],
        description="Natural language question",
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
    document_ids: Optional[List[str]] = Field(
        default=None,
        examples=[["a1b2c3d4-e5f6-7890-abcd-ef1234567890"]],
        description="Search only within specific documents",
    )
    conversation_id: Optional[str] = Field(
        default=None,
        examples=["conv-abc-123"],
        description="Conversation ID for follow-up questions",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        examples=[5],
        description="Number of document chunks to use as context",
    )
    score_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        examples=[0.6],
        description="Minimum relevance score for retrieved chunks",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        examples=[0.7],
        description="LLM temperature (lower = more focused, higher = more creative)",
    )
    max_tokens: int = Field(
        default=2048,
        ge=100,
        le=8192,
        examples=[2048],
        description="Maximum tokens in response",
    )


class SummarizeRequest(BaseModel):
    """Request to summarize a document."""

    max_length: int = Field(
        default=500,
        ge=100,
        le=2000,
        examples=[500],
        description="Approximate maximum words in summary",
    )
    temperature: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        examples=[0.3],
        description="LLM temperature for summary generation",
    )


# ============ Response Schemas ============


class SourceResponse(BaseModel):
    """A source document referenced in the answer."""

    document_id: str = Field(examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"])
    title: str = Field(examples=["Q4 Financial Report"])
    chunk_text: str = Field(examples=["The company achieved 15% growth..."])
    relevance_score: float = Field(examples=[0.92])
    chunk_index: Optional[int] = Field(default=None, examples=[3])


class AskResponse(BaseModel):
    """Response from the Q&A endpoint."""

    answer: str = Field(
        examples=[
            "According to the Q4 report, the key findings include: 1) Revenue increased by 15%..."
        ]
    )
    sources: List[SourceResponse] = Field(
        description="Documents used to generate the answer"
    )
    conversation_id: str = Field(
        examples=["conv-abc-123"],
        description="ID for continuing the conversation",
    )


class SummarizeResponse(BaseModel):
    """Response from document summarization."""

    summary: str = Field(
        examples=["This document covers the company's financial performance in Q4..."]
    )
    document_id: str = Field(examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"])


# ============ Endpoints ============


@router.post(
    "/",
    response_model=AskResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Ask a question",
    description="Ask a natural language question and get an AI-generated answer based on your documents.",
)
async def ask_question(
    request: AskRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question over accessible documents using RAG.

    The answer is generated based on relevant content from documents
    the user has permission to access. Sources are cited in the response.

    Supports:
    - Filtering by organization, team, or specific documents
    - Conversation history for follow-up questions
    - Configurable retrieval and generation parameters
    """
    try:
        config = GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
        )

        service = RAGService(db=db, user_id=current_user["user_id"])
        response = await service.ask(
            question=request.question,
            org_id=request.org_id,
            team_id=request.team_id,
            document_ids=request.document_ids,
            conversation_id=request.conversation_id,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            config=config,
        )

        return AskResponse(
            answer=response.answer,
            sources=[
                SourceResponse(
                    document_id=s.document_id,
                    title=s.title,
                    chunk_text=s.chunk_text,
                    relevance_score=s.relevance_score,
                    chunk_index=s.chunk_index,
                )
                for s in response.sources
            ],
            conversation_id=response.conversation_id,
        )

    except Exception as e:
        logger.error(f"Ask failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process your question. Please try again.",
        )


@router.post(
    "/stream",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Ask a question (streaming)",
    description="Ask a question with Server-Sent Events for streaming response.",
)
async def ask_question_stream(
    request: AskRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Ask a question with streaming response via Server-Sent Events (SSE).

    Events:
    - `{"type": "token", "content": "..."}` - Answer text chunks
    - `{"type": "sources", "sources": [...]}` - Source documents
    - `{"type": "done", "conversation_id": "..."}` - Completion signal

    Use this endpoint for real-time answer display.
    """

    async def event_generator():
        try:
            config = GenerationConfig(
                temperature=request.temperature,
                max_output_tokens=request.max_tokens,
            )

            service = RAGService(db=db, user_id=current_user["user_id"])

            async for event in service.ask_stream(
                question=request.question,
                org_id=request.org_id,
                team_id=request.team_id,
                document_ids=request.document_ids,
                conversation_id=request.conversation_id,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                config=config,
            ):
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.error(f"Stream ask failed: {e}", exc_info=True)
            error_event = {"type": "error", "message": "Failed to process question"}
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/documents/{document_id}/summarize",
    response_model=SummarizeResponse,
    responses={
        400: {"model": ErrorResponse},
        403: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    summary="Summarize a document",
    description="Generate an AI summary of a specific document.",
)
async def summarize_document(
    document_id: str,
    request: SummarizeRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a summary of a document.

    The user must have access to the document based on its scope.
    Returns a concise AI-generated summary of the document content.
    """
    try:
        config = GenerationConfig(
            temperature=request.temperature,
            max_output_tokens=2048,
        )

        service = RAGService(db=db, user_id=current_user["user_id"])
        summary = await service.summarize_document(
            document_id=document_id,
            max_length=request.max_length,
            config=config,
        )

        return SummarizeResponse(
            summary=summary,
            document_id=document_id,
        )

    except HTTPException:
        # Re-raise 404/403 from service
        raise
    except Exception as e:
        logger.error(f"Summarize failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate summary. Please try again.",
        )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        404: {"model": ErrorResponse},
    },
    summary="Delete conversation",
    description="Delete a conversation history.",
)
async def delete_conversation(
    conversation_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a conversation history.

    Removes all messages and context for the specified conversation.
    """
    service = RAGService(db=db, user_id=current_user["user_id"])
    deleted = await service.delete_conversation(conversation_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )
