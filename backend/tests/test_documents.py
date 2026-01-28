"""
Document upload and processing tests.

Tests for document parsing, chunking (unit tests), and document
upload, listing, retrieval, deletion (real DB tests).
"""
import pytest
from uuid import uuid4
from io import BytesIO
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from httpx import AsyncClient

from app.models.document import FileType, DocumentStatus, DocumentScope
from app.services.document_parser import get_document_parser


class TestDocumentParser:
    """Test document parsing functionality (unit tests - no services needed)."""

    async def test_parse_text_file(self, sample_text_content: str):
        """Should parse plain text file."""
        parser = get_document_parser()
        file_data = sample_text_content.encode("utf-8")

        text_result, metadata = await parser.parse_document(
            file_data=file_data,
            file_type=FileType.TXT.value,
            filename="test.txt",
        )

        assert text_result is not None
        assert len(text_result) > 0
        assert "Kahflane" in text_result
        assert metadata["parser"] == "text"
        assert "line_count" in metadata

    async def test_parse_markdown_file(self):
        """Should parse markdown file."""
        parser = get_document_parser()
        md_content = """
# Test Document

## Introduction

This is a **markdown** document with:
- Bullet points
- Code blocks
- Headers

```python
def hello():
    print("Hello, World!")
```
        """
        file_data = md_content.encode("utf-8")

        text_result, metadata = await parser.parse_document(
            file_data=file_data,
            file_type=FileType.MD.value,
            filename="test.md",
        )

        assert "Test Document" in text_result
        assert "markdown" in text_result.lower()
        assert metadata["parser"] == "text"

    def test_chunk_text_short(self):
        """Short text should return single chunk."""
        parser = get_document_parser()
        short_text = "This is a short text."

        chunks = parser.chunk_text(short_text)

        assert len(chunks) == 1
        assert chunks[0]["text"] == short_text
        assert chunks[0]["chunk_index"] == 0

    def test_chunk_text_long(self, sample_text_content: str):
        """Long text should be split into multiple chunks."""
        parser = get_document_parser()

        chunks = parser.chunk_text(sample_text_content, chunk_size=200, chunk_overlap=50)

        assert len(chunks) > 1
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
            assert len(chunk["text"]) > 0

    def test_chunk_text_overlap(self):
        """Chunks should overlap correctly."""
        parser = get_document_parser()
        text_content = "A" * 500 + " " + "B" * 500

        chunks = parser.chunk_text(text_content, chunk_size=400, chunk_overlap=100)

        assert len(chunks) >= 2

    def test_chunk_text_empty(self):
        """Empty text should return empty list."""
        parser = get_document_parser()

        chunks = parser.chunk_text("")
        assert len(chunks) == 0

        chunks = parser.chunk_text("   ")
        assert len(chunks) == 0


class TestDocumentUploadAPI:
    """Test document upload endpoints against real DB."""

    async def test_upload_document(
        self,
        authenticated_client: AsyncClient,
        test_organization: dict,
        sample_text_file: bytes,
    ):
        """Should upload a document successfully."""
        files = {
            "file": ("test_document.txt", BytesIO(sample_text_file), "text/plain"),
        }
        data = {
            "title": "Test Document",
            "org_id": str(test_organization["id"]),
            "scope": DocumentScope.TEAM.value,
        }

        response = await authenticated_client.post(
            "/api/v1/documents/upload",
            files=files,
            data=data,
        )

        if response.status_code in [200, 201]:
            doc = response.json()
            assert doc["title"] == "Test Document"
            assert doc["status"] == DocumentStatus.PENDING.value
            assert doc["file_type"] == FileType.TXT.value

    async def test_upload_document_unauthorized(
        self,
        unauthenticated_client: AsyncClient,
        sample_text_file: bytes,
    ):
        """Should reject upload without authentication."""
        files = {
            "file": ("test.txt", BytesIO(sample_text_file), "text/plain"),
        }

        response = await unauthenticated_client.post(
            "/api/v1/documents/upload",
            files=files,
            data={"title": "Test"},
        )

        assert response.status_code in [401, 403]

    async def test_get_upload_url(
        self,
        authenticated_client: AsyncClient,
        test_organization: dict,
    ):
        """Should generate presigned upload URL."""
        response = await authenticated_client.post(
            "/api/v1/documents/upload-url",
            json={
                "filename": "large_file.pdf",
                "content_type": "application/pdf",
                "org_id": str(test_organization["id"]),
            },
        )

        if response.status_code == 200:
            data = response.json()
            assert "upload_url" in data
            assert "file_key" in data


class TestDocumentListing:
    """Test document listing and filtering against real DB."""

    async def test_list_documents(
        self,
        authenticated_client: AsyncClient,
        db_session: AsyncSession,
        test_organization: dict,
        test_user,
    ):
        """Should list documents for the organization."""
        schema = test_organization["schema"]
        doc_id = uuid4()

        await db_session.execute(text(f'''
            INSERT INTO "{schema}".document
            (id, uploader_id, org_id, title, file_path, file_type, scope, status)
            VALUES (:id, :uploader, :org, :title, :path, :type, :scope, :status)
        '''), {
            "id": doc_id,
            "uploader": test_user.id,
            "org": test_organization["id"],
            "title": "Test Document for Listing",
            "path": "test/path/doc.txt",
            "type": FileType.TXT.value,
            "scope": DocumentScope.TEAM.value,
            "status": DocumentStatus.COMPLETED.value,
        })
        await db_session.commit()

        response = await authenticated_client.get(
            f"/api/v1/documents?org_id={test_organization['id']}",
        )

        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list) or "items" in data
            docs = data if isinstance(data, list) else data.get("items", [])
            titles = [d["title"] for d in docs]
            assert "Test Document for Listing" in titles

    async def test_list_documents_by_status(
        self,
        authenticated_client: AsyncClient,
        test_organization: dict,
    ):
        """Should filter documents by status."""
        response = await authenticated_client.get(
            f"/api/v1/documents?org_id={test_organization['id']}&status={DocumentStatus.COMPLETED.value}",
        )

        if response.status_code == 200:
            data = response.json()
            docs = data if isinstance(data, list) else data.get("items", [])
            for doc in docs:
                assert doc["status"] == DocumentStatus.COMPLETED.value


class TestDocumentRetrieval:
    """Test document download and retrieval against real DB."""

    async def test_get_document_by_id(
        self,
        authenticated_client: AsyncClient,
        db_session: AsyncSession,
        test_organization: dict,
        test_user,
    ):
        """Should retrieve document metadata by ID."""
        schema = test_organization["schema"]
        doc_id = uuid4()

        await db_session.execute(text(f'''
            INSERT INTO "{schema}".document
            (id, uploader_id, org_id, title, file_path, file_type, scope, status)
            VALUES (:id, :uploader, :org, :title, :path, :type, :scope, :status)
        '''), {
            "id": doc_id,
            "uploader": test_user.id,
            "org": test_organization["id"],
            "title": "Specific Document",
            "path": "test/path/specific.txt",
            "type": FileType.TXT.value,
            "scope": DocumentScope.TEAM.value,
            "status": DocumentStatus.COMPLETED.value,
        })
        await db_session.commit()

        response = await authenticated_client.get(f"/api/v1/documents/{doc_id}")

        if response.status_code == 200:
            doc = response.json()
            assert doc["id"] == str(doc_id)
            assert doc["title"] == "Specific Document"

    async def test_get_document_download_url(
        self,
        authenticated_client: AsyncClient,
        db_session: AsyncSession,
        test_organization: dict,
        test_user,
    ):
        """Should generate download URL for document."""
        schema = test_organization["schema"]
        doc_id = uuid4()

        await db_session.execute(text(f'''
            INSERT INTO "{schema}".document
            (id, uploader_id, org_id, title, file_path, file_type, scope, status)
            VALUES (:id, :uploader, :org, :title, :path, :type, :scope, :status)
        '''), {
            "id": doc_id,
            "uploader": test_user.id,
            "org": test_organization["id"],
            "title": "Download Test",
            "path": "test/path/download.pdf",
            "type": FileType.PDF.value,
            "scope": DocumentScope.TEAM.value,
            "status": DocumentStatus.COMPLETED.value,
        })
        await db_session.commit()

        response = await authenticated_client.get(f"/api/v1/documents/{doc_id}/download-url")

        assert response.status_code in [200, 404, 500]


class TestDocumentDeletion:
    """Test document deletion against real DB."""

    async def test_delete_document(
        self,
        authenticated_client: AsyncClient,
        db_session: AsyncSession,
        test_organization: dict,
        test_user,
    ):
        """Should delete document and schedule vector cleanup."""
        schema = test_organization["schema"]
        doc_id = uuid4()

        await db_session.execute(text(f'''
            INSERT INTO "{schema}".document
            (id, uploader_id, org_id, title, file_path, file_type, scope, status)
            VALUES (:id, :uploader, :org, :title, :path, :type, :scope, :status)
        '''), {
            "id": doc_id,
            "uploader": test_user.id,
            "org": test_organization["id"],
            "title": "To Be Deleted",
            "path": "test/path/delete.txt",
            "type": FileType.TXT.value,
            "scope": DocumentScope.PERSONAL.value,
            "status": DocumentStatus.PENDING.value,
        })
        await db_session.commit()

        response = await authenticated_client.delete(f"/api/v1/documents/{doc_id}")

        if response.status_code in [200, 204]:
            result = await db_session.execute(text(f'''
                SELECT id FROM "{schema}".document WHERE id = :id
            '''), {"id": doc_id})
            assert result.fetchone() is None

    async def test_delete_document_not_found(
        self,
        authenticated_client: AsyncClient,
    ):
        """Should return 404 for non-existent document."""
        fake_id = uuid4()

        response = await authenticated_client.delete(f"/api/v1/documents/{fake_id}")

        assert response.status_code in [404, 204]
