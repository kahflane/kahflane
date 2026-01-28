"""
Document management API endpoints.
"""
from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel, Field
from typing import Optional, List
from uuid import UUID
from datetime import datetime
import logging

from app.db.session import get_db
from app.models.document import Document, DocumentScope, DocumentStatus, FileType
from app.api.deps import get_current_user
from app.services.storage_service import storage_service
from app.core.tenant import get_current_tenant
from app.api.schemas import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# File type mapping
FILE_TYPE_MAP = {
    'application/pdf': FileType.PDF.value,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': FileType.DOCX.value,
    'application/msword': FileType.DOC.value,
    'text/plain': FileType.TXT.value,
    'text/markdown': FileType.MD.value,
    'audio/mpeg': FileType.AUDIO_MP3.value,
    'audio/mp3': FileType.AUDIO_MP3.value,
    'audio/wav': FileType.AUDIO_WAV.value,
    'audio/x-wav': FileType.AUDIO_WAV.value,
    'audio/m4a': FileType.AUDIO_M4A.value,
    'audio/x-m4a': FileType.AUDIO_M4A.value,
    'video/mp4': FileType.VIDEO_MP4.value,
    'video/webm': FileType.VIDEO_WEBM.value,
}

ALLOWED_EXTENSIONS = {
    '.pdf', '.docx', '.doc', '.txt', '.md',
    '.mp3', '.wav', '.m4a',
    '.mp4', '.webm',
}


# Schemas

class DocumentResponse(BaseModel):
    """Document information."""
    id: str = Field(examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"])
    title: str = Field(examples=["Q4 Financial Report"])
    file_path: str = Field(examples=["documents/2024/report.pdf"])
    file_type: str = Field(examples=["pdf"])
    file_size_bytes: Optional[int] = Field(default=None, examples=[1048576])
    scope: str = Field(examples=["team"])
    status: str = Field(examples=["processed"])
    error_message: Optional[str] = Field(default=None, examples=[None])
    upload_date: datetime = Field(examples=["2024-12-01T10:30:00Z"])
    processed_at: Optional[datetime] = Field(default=None, examples=["2024-12-01T10:31:00Z"])
    uploader_id: str = Field(examples=["550e8400-e29b-41d4-a716-446655440000"])
    org_id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    team_id: Optional[str] = Field(default=None, examples=["d4e5f6a7-b8c9-0123-4567-890abcdef012"])

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """List of documents."""
    documents: List[DocumentResponse]
    total: int = Field(examples=[42])


class CreateDocumentRequest(BaseModel):
    """Request to create document metadata."""
    title: str = Field(min_length=1, max_length=500, examples=["Q4 Financial Report"])
    file_type: str = Field(examples=["pdf"])
    file_path: str = Field(examples=["documents/2024/report.pdf"])
    file_size_bytes: Optional[int] = Field(default=None, examples=[1048576])
    org_id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    team_id: Optional[str] = Field(default=None, examples=["d4e5f6a7-b8c9-0123-4567-890abcdef012"])
    scope: str = Field(default=DocumentScope.TEAM.value, examples=["team"])


class UpdateDocumentRequest(BaseModel):
    """Request to update document."""
    title: Optional[str] = Field(None, min_length=1, max_length=500, examples=["Updated Report Title"])
    scope: Optional[str] = Field(default=None, examples=["organization"])


class UploadUrlRequest(BaseModel):
    """Request for presigned upload URL."""
    filename: str = Field(examples=["report.pdf"])
    content_type: Optional[str] = Field(default=None, examples=["application/pdf"])
    org_id: str = Field(examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"])
    team_id: Optional[str] = Field(default=None, examples=["d4e5f6a7-b8c9-0123-4567-890abcdef012"])


class UploadUrlResponse(BaseModel):
    """Response with presigned upload URL."""
    upload_url: str = Field(examples=["https://s3.amazonaws.com/kahflane/documents/..."])
    file_key: str = Field(examples=["documents/2024/abc123_report.pdf"])
    fields: dict = Field(examples=[{"key": "documents/2024/abc123_report.pdf"}])
    content_type: str = Field(examples=["application/pdf"])


class DownloadUrlResponse(BaseModel):
    """Response with presigned download URL."""
    download_url: str = Field(examples=["https://s3.amazonaws.com/kahflane/documents/...?X-Amz-Signature=..."])
    expires_in: int = Field(examples=[3600])


def _get_file_type(content_type: str, filename: str) -> str:
    """Determine file type from content type or extension."""
    # Try content type first
    if content_type in FILE_TYPE_MAP:
        return FILE_TYPE_MAP[content_type]

    # Fall back to extension
    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    ext_to_type = {
        '.pdf': FileType.PDF.value,
        '.docx': FileType.DOCX.value,
        '.doc': FileType.DOC.value,
        '.txt': FileType.TXT.value,
        '.md': FileType.MD.value,
        '.mp3': FileType.AUDIO_MP3.value,
        '.wav': FileType.AUDIO_WAV.value,
        '.m4a': FileType.AUDIO_M4A.value,
        '.mp4': FileType.VIDEO_MP4.value,
        '.webm': FileType.VIDEO_WEBM.value,
    }
    return ext_to_type.get(ext, 'UNKNOWN')


def _validate_file_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ALLOWED_EXTENSIONS


def _doc_to_response(doc: Document) -> DocumentResponse:
    """Convert Document model to response."""
    return DocumentResponse(
        id=str(doc.id),
        title=doc.title,
        file_path=doc.file_path,
        file_type=doc.file_type,
        file_size_bytes=doc.file_size_bytes,
        scope=doc.scope,
        status=doc.status,
        error_message=doc.error_message,
        upload_date=doc.upload_date,
        processed_at=doc.processed_at,
        uploader_id=str(doc.uploader_id),
        org_id=str(doc.org_id),
        team_id=str(doc.team_id) if doc.team_id else None,
    )


# Endpoints

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    org_id: Optional[str] = None,
    team_id: Optional[str] = None,
    doc_status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List documents accessible to the current user.

    Filters by organization, team, and status.
    """
    # Build query
    query = select(Document)

    if org_id:
        query = query.where(Document.org_id == UUID(org_id))
    if team_id:
        query = query.where(Document.team_id == UUID(team_id))
    if doc_status:
        query = query.where(Document.status == doc_status)

    # Get total count first
    count_query = select(func.count(Document.id))
    if org_id:
        count_query = count_query.where(Document.org_id == UUID(org_id))
    if team_id:
        count_query = count_query.where(Document.team_id == UUID(team_id))
    if doc_status:
        count_query = count_query.where(Document.status == doc_status)

    count_result = await db.execute(count_query)
    total = count_result.scalar() or 0

    # Get paginated results
    query = query.order_by(Document.upload_date.desc())
    query = query.offset(offset).limit(limit)

    result = await db.execute(query)
    documents = result.scalars().all()

    return DocumentListResponse(
        documents=[_doc_to_response(doc) for doc in documents],
        total=total,
    )


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    org_id: str = Form(...),
    team_id: Optional[str] = Form(None),
    scope: str = Form(DocumentScope.TEAM.value),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document file.

    This endpoint:
    1. Validates the file
    2. Uploads to S3 storage
    3. Creates database record
    4. Enqueues for processing
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    if not _validate_file_extension(file.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Use filename as title if not provided
    doc_title = title or file.filename.rsplit('.', 1)[0]

    # Determine file type
    content_type = file.content_type or 'application/octet-stream'
    file_type = _get_file_type(content_type, file.filename)

    if file_type == 'UNKNOWN':
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type",
        )

    # Read file content
    file_content = await file.read()
    file_size = len(file_content)

    # Check file size (max 100MB)
    max_size = 100 * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size is {max_size // (1024*1024)}MB",
        )

    try:
        # Upload to S3
        upload_result = await storage_service.upload_bytes(
            data=file_content,
            filename=file.filename,
            folder="documents",
            content_type=content_type,
            metadata={
                'uploader_id': current_user["user_id"],
                'org_id': org_id,
                'team_id': team_id or '',
            },
        )

        # Create database record
        document = Document(
            uploader_id=UUID(current_user["user_id"]),
            org_id=UUID(org_id),
            team_id=UUID(team_id) if team_id else None,
            title=doc_title,
            file_path=upload_result['file_key'],
            file_type=file_type,
            file_size_bytes=file_size,
            scope=scope,
            status=DocumentStatus.PENDING.value,
        )
        db.add(document)
        await db.commit()
        await db.refresh(document)

        logger.info(f"Uploaded document {document.id}: {file.filename}")

        # Enqueue for processing
        try:
            from app.workers.tasks import enqueue_document_ingestion
            enqueue_document_ingestion(str(document.id))
        except Exception as e:
            logger.warning(f"Failed to enqueue document processing: {e}")

        return _doc_to_response(document)

    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document",
        )


@router.post("/upload-url", response_model=UploadUrlResponse)
async def get_upload_url(
    request: UploadUrlRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Get a presigned URL for direct client upload.

    Use this for large files to upload directly to S3.
    """
    if not _validate_file_extension(request.filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    try:
        result = await storage_service.generate_upload_url(
            filename=request.filename,
            folder="documents",
            content_type=request.content_type,
            expires_in=3600,  # 1 hour
        )

        return UploadUrlResponse(
            upload_url=result['upload_url'],
            file_key=result['file_key'],
            fields=result['fields'],
            content_type=result['content_type'],
        )

    except Exception as e:
        logger.error(f"Failed to generate upload URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate upload URL",
        )


@router.post("/confirm-upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def confirm_upload(
    file_key: str = Form(...),
    title: str = Form(...),
    org_id: str = Form(...),
    team_id: Optional[str] = Form(None),
    scope: str = Form(DocumentScope.TEAM.value),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Confirm a direct upload and create database record.

    Call this after successfully uploading via presigned URL.
    """
    # Verify file exists in S3
    file_info = await storage_service.get_file_info(file_key)

    if not file_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found in storage. Upload may have failed.",
        )

    # Determine file type from key
    filename = file_key.rsplit('/', 1)[-1]
    file_type = _get_file_type(file_info['content_type'], filename)

    # Create database record
    document = Document(
        uploader_id=UUID(current_user["user_id"]),
        org_id=UUID(org_id),
        team_id=UUID(team_id) if team_id else None,
        title=title,
        file_path=file_key,
        file_type=file_type,
        file_size_bytes=file_info['size'],
        scope=scope,
        status=DocumentStatus.PENDING.value,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    logger.info(f"Confirmed upload for document {document.id}")

    # Enqueue for processing
    try:
        from app.workers.tasks import enqueue_document_ingestion
        enqueue_document_ingestion(str(document.id))
    except Exception as e:
        logger.warning(f"Failed to enqueue document processing: {e}")

    return _doc_to_response(document)


@router.post("/", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_document(
    request: CreateDocumentRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create document metadata manually.

    Use upload endpoints for file upload with automatic metadata creation.
    """
    document = Document(
        uploader_id=UUID(current_user["user_id"]),
        org_id=UUID(request.org_id),
        team_id=UUID(request.team_id) if request.team_id else None,
        title=request.title,
        file_path=request.file_path,
        file_type=request.file_type,
        file_size_bytes=request.file_size_bytes,
        scope=request.scope,
        status=DocumentStatus.PENDING.value,
    )
    db.add(document)
    await db.commit()
    await db.refresh(document)

    return _doc_to_response(document)


@router.get("/{document_id}", response_model=DocumentResponse, responses={404: {"model": ErrorResponse}})
async def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get document details.
    """
    result = await db.execute(
        select(Document).where(Document.id == UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return _doc_to_response(document)


@router.get("/{document_id}/download-url", response_model=DownloadUrlResponse, responses={404: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def get_download_url(
    document_id: str,
    expires_in: int = 3600,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a presigned URL to download the document.
    """
    result = await db.execute(
        select(Document).where(Document.id == UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    try:
        url = await storage_service.generate_presigned_url(
            file_key=document.file_path,
            expires_in=expires_in,
        )

        return DownloadUrlResponse(
            download_url=url,
            expires_in=expires_in,
        )

    except Exception as e:
        logger.error(f"Failed to generate download URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate download URL",
        )


@router.get("/{document_id}/download")
async def download_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Download document file directly (streams from S3).
    """
    result = await db.execute(
        select(Document).where(Document.id == UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    try:
        # Get file info for content type
        file_info = await storage_service.get_file_info(document.file_path)

        if not file_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found in storage",
            )

        # Stream the file
        async def file_stream():
            async for chunk in storage_service.download_file_stream(document.file_path):
                yield chunk

        # Extract filename from path
        filename = document.file_path.rsplit('/', 1)[-1]

        return StreamingResponse(
            file_stream(),
            media_type=file_info['content_type'],
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"',
                'Content-Length': str(file_info['size']),
            }
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found in storage",
        )
    except Exception as e:
        logger.error(f"Failed to download document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to download document",
        )


@router.patch("/{document_id}", response_model=DocumentResponse, responses={404: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
async def update_document(
    document_id: str,
    request: UpdateDocumentRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update document metadata.
    """
    result = await db.execute(
        select(Document).where(Document.id == UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Only uploader can update
    if str(document.uploader_id) != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the uploader can update this document",
        )

    if request.title is not None:
        document.title = request.title
    if request.scope is not None:
        document.scope = request.scope

    await db.commit()
    await db.refresh(document)

    return _doc_to_response(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT, responses={404: {"model": ErrorResponse}, 403: {"model": ErrorResponse}})
async def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a document and its file from storage.
    """
    result = await db.execute(
        select(Document).where(Document.id == UUID(document_id))
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Only uploader or admin can delete
    if str(document.uploader_id) != current_user["user_id"] and current_user["role"] not in ("admin", "owner"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied",
        )

    # Delete from S3
    try:
        await storage_service.delete_file(document.file_path)
    except Exception as e:
        logger.warning(f"Failed to delete file from S3: {e}")

    # Enqueue vector deletion
    try:
        from app.workers.tasks import enqueue_vector_deletion
        enqueue_vector_deletion(str(document.id))
    except Exception as e:
        logger.warning(f"Failed to enqueue vector deletion: {e}")

    # Delete from database
    await db.delete(document)
    await db.commit()

    logger.info(f"Deleted document {document_id}")
