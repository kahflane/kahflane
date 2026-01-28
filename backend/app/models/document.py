"""
Tenant schema models for document management.

These models live in each tenant's dedicated schema.
"""
from datetime import datetime, timezone
from typing import Optional, List, TYPE_CHECKING
from uuid import UUID, uuid4
from enum import Enum
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, String, DateTime, Text, BigInteger, text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB

if TYPE_CHECKING:
    from app.models.organization import Organization, Team


class FileType(str, Enum):
    """Supported file types for document processing."""
    PDF = "PDF"
    DOCX = "DOCX"
    DOC = "DOC"
    TXT = "TXT"
    MD = "MD"
    AUDIO_MP3 = "AUDIO_MP3"
    AUDIO_WAV = "AUDIO_WAV"
    AUDIO_M4A = "AUDIO_M4A"
    VIDEO_MP4 = "VIDEO_MP4"
    VIDEO_WEBM = "VIDEO_WEBM"


class DocumentScope(str, Enum):
    """Document visibility scope."""
    PERSONAL = "personal"     # Only uploader can access
    TEAM = "team"             # Team members can access
    ORGANIZATION = "organization"  # All org members can access


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"       # Uploaded, waiting for processing
    PROCESSING = "processing" # Currently being processed
    COMPLETED = "completed"   # Successfully processed and indexed
    FAILED = "failed"         # Processing failed


class Document(SQLModel, table=True):
    """
    Document metadata and processing state.

    Actual file content is stored in S3/local storage.
    Vector embeddings are stored in Qdrant.
    """
    __tablename__ = "document"

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    uploader_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), nullable=False, index=True),
        description="References public.user.id"
    )
    org_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("organization.id"), nullable=False, index=True)
    )
    team_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("team.id"), nullable=True, index=True)
    )
    title: str = Field(
        max_length=500,
        sa_column=Column(String(500), nullable=False)
    )
    file_path: str = Field(
        max_length=1000,
        sa_column=Column(String(1000), nullable=False),
        description="Path to file in storage (S3 key or local path)"
    )
    file_type: str = Field(
        max_length=50,
        sa_column=Column(String(50), nullable=False),
        description="File type from FileType enum"
    )
    file_size_bytes: Optional[int] = Field(
        default=None,
        sa_column=Column(BigInteger, nullable=True)
    )
    scope: str = Field(
        default=DocumentScope.TEAM.value,
        max_length=50,
        sa_column=Column(String(50), nullable=False, default="team"),
        description="Visibility scope: personal, team, organization"
    )
    status: str = Field(
        default=DocumentStatus.PENDING.value,
        max_length=50,
        sa_column=Column(String(50), nullable=False, default="pending", index=True),
        description="Processing status"
    )
    error_message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Error message if processing failed"
    )
    qdrant_collection_name: Optional[str] = Field(
        default=None,
        max_length=255,
        sa_column=Column(String(255), nullable=True),
        description="Qdrant collection where vectors are stored"
    )
    extra_metadata: dict = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False, server_default=text("'{}'")),
        description="Additional metadata (page count, duration, etc.)"
    )
    upload_date: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )
    processed_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True)
    )

    # Relationships
    organization: "Organization" = Relationship(back_populates="documents")
    team: Optional["Team"] = Relationship(back_populates="documents")
    processing_logs: List["ProcessingLog"] = Relationship(back_populates="document")


class ProcessingLog(SQLModel, table=True):
    """
    Log entries for document processing stages.

    Tracks the progress of document ingestion through various stages.
    """
    __tablename__ = "processing_log"

    id: UUID = Field(
        default_factory=uuid4,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    )
    document_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), ForeignKey("document.id"), nullable=False, index=True)
    )
    stage: str = Field(
        max_length=100,
        sa_column=Column(String(100), nullable=False),
        description="Processing stage: UPLOAD, PARSING, CHUNKING, EMBEDDING, INDEXING"
    )
    status: str = Field(
        max_length=50,
        sa_column=Column(String(50), nullable=False),
        description="Stage status: started, completed, failed"
    )
    message: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
        description="Status message or error details"
    )
    extra_metadata: dict = Field(
        default_factory=dict,
        sa_column=Column("metadata", JSONB, nullable=False, server_default=text("'{}'")),
        description="Stage-specific metadata (chunk count, vector count, etc.)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(DateTime(timezone=True), nullable=False, server_default=text("NOW()"))
    )

    # Relationships
    document: "Document" = Relationship(back_populates="processing_logs")
