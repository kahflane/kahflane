"""
Dramatiq background tasks for document processing.

All tasks must receive tenant_context to operate on the correct schema.
"""
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import CurrentMessage
import asyncio
import logging
from typing import Dict, Any

from app.core.config import settings
from app.core.tenant import TenantContext, set_current_tenant

logger = logging.getLogger(__name__)

# Configure Redis broker
redis_broker = RedisBroker(url=settings.REDIS_URL)
dramatiq.set_broker(redis_broker)


class TenantContextMiddleware(dramatiq.Middleware):
    """
    Middleware to restore tenant context in worker tasks.

    Every task receives a tenant_context dict that must be used to
    set the current tenant before processing.
    """

    def before_process_message(self, broker, message):
        """Extract and set tenant context from message options."""
        tenant_data = message.options.get("tenant_context")
        if tenant_data:
            tenant = TenantContext(
                tenant_id=tenant_data["tenant_id"],
                tenant_slug=tenant_data["tenant_slug"],
                schema_name=tenant_data["schema_name"],
            )
            set_current_tenant(tenant)
            logger.debug(f"Set tenant context: {tenant.tenant_slug}")

    def after_process_message(self, broker, message, *, result=None, exception=None):
        """Clear tenant context after processing."""
        set_current_tenant(None)
        logger.debug("Cleared tenant context")


# Register middleware
redis_broker.add_middleware(TenantContextMiddleware())
redis_broker.add_middleware(CurrentMessage())


def run_async(coro):
    """Helper to run async code in sync dramatiq tasks."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@dramatiq.actor(max_retries=3, min_backoff=1000, max_backoff=60000)
def ingest_document(document_id: str, tenant_context: Dict[str, str]):
    """
    Process and index a document.

    This task:
    1. Updates document status to PROCESSING
    2. Parses document content (using Docling for PDFs, ElevenLabs for audio)
    3. Chunks the content
    4. Generates embeddings (using Gemini)
    5. Indexes vectors in Qdrant
    6. Updates document status to COMPLETED

    Args:
        document_id: UUID of the document to process
        tenant_context: Dict with tenant_id, tenant_slug, schema_name
    """
    logger.info(f"Starting document ingestion: {document_id}")

    # Set tenant context
    tenant = TenantContext(**tenant_context)
    set_current_tenant(tenant)

    try:
        run_async(_ingest_document_async(document_id, tenant))
        logger.info(f"Completed document ingestion: {document_id}")
    except Exception as e:
        logger.error(f"Failed to ingest document {document_id}: {e}")
        raise
    finally:
        set_current_tenant(None)


async def _ingest_document_async(document_id: str, tenant: TenantContext):
    """Async implementation of document ingestion."""
    from app.db.session import get_tenant_session
    from app.models.document import Document, ProcessingLog, DocumentStatus
    from app.services.storage_service import get_storage_service
    from app.services.document_parser import get_document_parser
    from app.services.embedding_service import get_embedding_service
    from app.services.vector_store import get_vector_store
    from datetime import datetime, timezone
    from uuid import UUID

    storage = get_storage_service()
    parser = get_document_parser()
    embeddings = get_embedding_service()
    vector_store = get_vector_store()

    async with get_tenant_session(tenant) as session:
        # Get document
        from sqlalchemy import select
        result = await session.execute(
            select(Document).where(Document.id == UUID(document_id))
        )
        doc = result.scalar_one_or_none()

        if not doc:
            raise ValueError(f"Document {document_id} not found")

        # Update status to processing
        doc.status = DocumentStatus.PROCESSING.value
        await session.flush()

        # Log processing start
        log = ProcessingLog(
            document_id=doc.id,
            stage="STARTED",
            status="started",
            message="Document processing started",
        )
        session.add(log)
        await session.flush()

        try:
            # Step 1: Download file from storage
            logger.info(f"Downloading file: {doc.file_path}")
            file_data = await storage.download_file(doc.file_path)

            download_log = ProcessingLog(
                document_id=doc.id,
                stage="DOWNLOAD",
                status="completed",
                message=f"Downloaded {len(file_data)} bytes",
                metadata={"size_bytes": len(file_data)},
            )
            session.add(download_log)
            await session.flush()

            # Step 2: Parse content based on file_type
            logger.info(f"Parsing document: {doc.file_type}")
            text_content, parse_metadata = await parser.parse_document(
                file_data=file_data,
                file_type=doc.file_type,
                filename=doc.title,
            )

            parse_log = ProcessingLog(
                document_id=doc.id,
                stage="PARSING",
                status="completed",
                message=f"Extracted {len(text_content)} characters",
                metadata=parse_metadata,
            )
            session.add(parse_log)
            await session.flush()

            # Step 3: Chunk content
            logger.info("Chunking content")
            chunks = parser.chunk_text(text_content)

            if not chunks:
                raise ValueError("No content extracted from document")

            chunk_log = ProcessingLog(
                document_id=doc.id,
                stage="CHUNKING",
                status="completed",
                message=f"Created {len(chunks)} chunks",
                metadata={"chunk_count": len(chunks)},
            )
            session.add(chunk_log)
            await session.flush()

            # Step 4: Generate embeddings
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            chunk_texts = [c["text"] for c in chunks]
            chunk_embeddings = await embeddings.embed_documents(chunk_texts)

            # Add embeddings to chunks
            for chunk, embedding in zip(chunks, chunk_embeddings):
                chunk["embedding"] = embedding

            embed_log = ProcessingLog(
                document_id=doc.id,
                stage="EMBEDDING",
                status="completed",
                message=f"Generated {len(chunk_embeddings)} embeddings",
                metadata={"embedding_count": len(chunk_embeddings)},
            )
            session.add(embed_log)
            await session.flush()

            # Step 5: Index in Qdrant
            logger.info("Indexing vectors in Qdrant")
            vector_count = await vector_store.index_document_chunks(
                document_id=doc.id,
                chunks=chunks,
                org_id=doc.org_id,
                team_id=doc.team_id,
                uploader_id=doc.uploader_id,
                scope=doc.scope,
                metadata={
                    "title": doc.title,
                    "file_type": doc.file_type,
                    **parse_metadata,
                },
            )

            index_log = ProcessingLog(
                document_id=doc.id,
                stage="INDEXING",
                status="completed",
                message=f"Indexed {vector_count} vectors",
                metadata={"vector_count": vector_count},
            )
            session.add(index_log)

            # Update document as completed
            doc.status = DocumentStatus.COMPLETED.value
            doc.processed_at = datetime.now(timezone.utc)
            doc.qdrant_collection_name = vector_store.collection_name
            doc.metadata = {
                **doc.metadata,
                "chunk_count": len(chunks),
                "vector_count": vector_count,
                **parse_metadata,
            }

            # Log completion
            complete_log = ProcessingLog(
                document_id=doc.id,
                stage="COMPLETED",
                status="completed",
                message="Document processing completed successfully",
            )
            session.add(complete_log)

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            doc.status = DocumentStatus.FAILED.value
            doc.error_message = str(e)

            # Log failure
            error_log = ProcessingLog(
                document_id=doc.id,
                stage="FAILED",
                status="failed",
                message=str(e),
            )
            session.add(error_log)
            raise

        await session.commit()


@dramatiq.actor(max_retries=2)
def delete_document_vectors(document_id: str, tenant_context: Dict[str, str]):
    """
    Delete document vectors from Qdrant.

    Called when a document is deleted.

    Args:
        document_id: UUID of the document
        tenant_context: Dict with tenant_id, tenant_slug, schema_name
    """
    logger.info(f"Deleting vectors for document: {document_id}")

    tenant = TenantContext(**tenant_context)
    set_current_tenant(tenant)

    try:
        from app.services.vector_store import get_vector_store
        from uuid import UUID

        vector_store = get_vector_store()
        run_async(vector_store.delete_document_vectors(UUID(document_id)))
        logger.info(f"Deleted vectors for document: {document_id}")
    except Exception as e:
        logger.error(f"Failed to delete vectors for {document_id}: {e}")
        raise
    finally:
        set_current_tenant(None)


@dramatiq.actor(max_retries=3, min_backoff=5000)
def send_verification_email(email: str, otp: str):
    """Send email verification OTP via SMTP."""
    from app.services.email_service import get_email_service
    get_email_service().send_otp_email(email, otp)


@dramatiq.actor(max_retries=3, min_backoff=5000)
def send_password_reset_email(email: str, otp: str):
    """Send password reset OTP via SMTP."""
    from app.services.email_service import get_email_service
    get_email_service().send_password_reset_email(email, otp)


@dramatiq.actor(max_retries=3, min_backoff=5000)
def send_invitation_email_task(email: str, tenant_name: str, invite_token: str, invited_by: str):
    """Send tenant invitation email via SMTP."""
    from app.services.email_service import get_email_service
    get_email_service().send_invitation_email(email, tenant_name, invite_token, invited_by)


def enqueue_document_ingestion(document_id: str) -> None:
    """
    Helper to enqueue document for processing with current tenant context.

    This should be called from API endpoints after document upload.

    Args:
        document_id: UUID of the document to process
    """
    from app.core.tenant import get_current_tenant

    tenant = get_current_tenant()
    if not tenant:
        raise ValueError("No tenant context for task enqueue")

    ingest_document.send(
        document_id=document_id,
        tenant_context={
            "tenant_id": tenant.tenant_id,
            "tenant_slug": tenant.tenant_slug,
            "schema_name": tenant.schema_name,
        }
    )
    logger.info(f"Enqueued document {document_id} for ingestion")


def enqueue_vector_deletion(document_id: str) -> None:
    """
    Helper to enqueue document vector deletion with current tenant context.

    Args:
        document_id: UUID of the document
    """
    from app.core.tenant import get_current_tenant

    tenant = get_current_tenant()
    if not tenant:
        raise ValueError("No tenant context for task enqueue")

    delete_document_vectors.send(
        document_id=document_id,
        tenant_context={
            "tenant_id": tenant.tenant_id,
            "tenant_slug": tenant.tenant_slug,
            "schema_name": tenant.schema_name,
        }
    )
    logger.info(f"Enqueued vector deletion for document {document_id}")
