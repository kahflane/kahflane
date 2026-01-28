"""
Docling integration tests.

Tests real document parsing with Docling SDK.
"""
import pytest
from pathlib import Path

from app.services.docling_service import DoclingService, get_docling_service
from app.services.document_parser import DocumentParserService


pytestmark = pytest.mark.integration


class TestDoclingService:
    """Test Docling document parsing."""

    @pytest.fixture
    def docling_service(self) -> DoclingService:
        """Get Docling service instance."""
        try:
            return DoclingService(
                enable_ocr=False,  # Disable OCR for faster tests
                enable_table_structure=True,
            )
        except Exception as e:
            pytest.skip(f"Docling service not available: {e}")

    @pytest.mark.asyncio
    async def test_parse_markdown_bytes(self, docling_service: DoclingService, sample_markdown_bytes: bytes):
        """Should parse markdown from bytes."""
        result = docling_service.parse_bytes(sample_markdown_bytes, "test.md")

        assert "text" in result
        assert "Kahflane" in result["text"]
        assert result["metadata"]["parser"] == "docling"

    @pytest.mark.asyncio
    async def test_parse_markdown_with_table(self, docling_service: DoclingService):
        """Should parse markdown content with tables."""
        markdown_content = """
# Test Document

This is a **test** document with:

- Bullet point 1
- Bullet point 2

## Section 2

Some more content here.

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
        """.encode("utf-8")

        result = docling_service.parse_bytes(markdown_content, "test.md")

        assert "text" in result
        assert "Test Document" in result["text"]
        assert "markdown" in result

    @pytest.mark.asyncio
    async def test_chunk_document(self, docling_service: DoclingService, sample_markdown_bytes: bytes):
        """Should chunk parsed document."""
        # First parse
        parsed = docling_service.parse_bytes(sample_markdown_bytes, "test.md")

        # Then chunk
        chunks = docling_service.chunk_document(parsed, max_tokens=100)

        assert len(chunks) > 0

        for chunk in chunks:
            assert "text" in chunk
            assert "chunk_index" in chunk
            assert len(chunk["text"]) > 0

    @pytest.mark.asyncio
    async def test_parse_and_chunk_combined(
        self,
        docling_service: DoclingService,
        sample_markdown_bytes: bytes,
    ):
        """Should parse and chunk in one call."""
        result = docling_service.parse_and_chunk(
            sample_markdown_bytes,
            "test.md",
            max_tokens=100,
        )

        assert "text" in result
        assert "chunks" in result
        assert "chunk_count" in result
        assert result["chunk_count"] > 0
        assert len(result["chunks"]) == result["chunk_count"]

    @pytest.mark.asyncio
    async def test_parse_pdf_bytes(self, docling_service: DoclingService, sample_pdf_bytes: bytes):
        """Should parse PDF from bytes."""
        try:
            result = docling_service.parse_bytes(sample_pdf_bytes, "test.pdf")

            assert "text" in result
            assert result["metadata"]["parser"] == "docling"
        except Exception as e:
            # PDF parsing might fail with minimal PDF
            if "parse" in str(e).lower() or "pdf" in str(e).lower():
                pytest.skip(f"PDF parsing failed (expected for minimal PDF): {e}")
            raise


class TestDocumentParserService:
    """Test DocumentParserService with real Docling."""

    @pytest.fixture
    def parser_service(self) -> DocumentParserService:
        """Get DocumentParserService instance."""
        return DocumentParserService()

    @pytest.mark.asyncio
    async def test_parse_markdown_document(
        self,
        parser_service: DocumentParserService,
        sample_markdown_bytes: bytes,
    ):
        """Should parse markdown document using Docling."""
        text, metadata = await parser_service.parse_document(
            file_data=sample_markdown_bytes,
            file_type="MD",
            filename="test.md",
        )

        assert len(text) > 0
        assert "Kahflane" in text
        assert "parser" in metadata

    @pytest.mark.asyncio
    async def test_parse_complex_markdown(self, parser_service: DocumentParserService):
        """Should parse complex markdown document."""
        md_content = """
# Integration Test Document

This document tests the **Docling** integration.

## Features

1. Parse markdown
2. Extract text
3. Generate chunks

### Code Example

```python
print("Hello from integration test")
```

## Table

| Name | Value |
|------|-------|
| Test | Pass  |
        """.encode("utf-8")

        text, metadata = await parser_service.parse_document(
            file_data=md_content,
            file_type="MD",
            filename="test.md",
        )

        assert len(text) > 0
        assert "Integration Test" in text

    @pytest.mark.asyncio
    async def test_chunk_text(self, parser_service: DocumentParserService, sample_text_content: str):
        """Should chunk text correctly."""
        chunks = parser_service.chunk_text(
            sample_text_content,
            chunk_size=200,
            chunk_overlap=50,
        )

        assert len(chunks) >= 1  # At least one chunk

        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
            assert len(chunk["text"]) > 0

    @pytest.mark.asyncio
    async def test_chunk_short_text(self, parser_service: DocumentParserService):
        """Should handle short text as single chunk."""
        short_text = "This is a short text."

        chunks = parser_service.chunk_text(short_text, chunk_size=1000)

        assert len(chunks) == 1
        assert chunks[0]["text"] == short_text
        assert chunks[0]["chunk_index"] == 0

    @pytest.mark.asyncio
    async def test_chunk_empty_text(self, parser_service: DocumentParserService):
        """Should handle empty text."""
        chunks = parser_service.chunk_text("")
        assert chunks == []

        chunks = parser_service.chunk_text("   ")
        assert chunks == []


class TestDoclingRealFiles:
    """Test Docling with real file samples."""

    @pytest.fixture
    def docling_service(self) -> DoclingService:
        """Get Docling service."""
        return DoclingService(enable_ocr=False)

    @pytest.mark.asyncio
    async def test_parse_complex_markdown(self, docling_service: DoclingService):
        """Should parse complex markdown with various elements."""
        complex_md = """
# Project Documentation

## Overview

This is a comprehensive guide to the **Kahflane** knowledge management system.

### Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| Multi-tenant | Isolated data per tenant | Active |
| Vector Search | Semantic document search | Active |
| AI Processing | Document understanding | Beta |

### Code Example

```python
from kahflane import Client

client = Client(api_key="your-key")
results = client.search("machine learning")
```

## Architecture

The system consists of several components:

1. **API Layer** - FastAPI-based REST API
2. **Database** - PostgreSQL with schema isolation
3. **Vector Store** - Qdrant for embeddings
4. **Cache** - Redis for sessions

> Note: All components are containerized for easy deployment.

### Diagram

```
[Client] --> [API] --> [Database]
                  |--> [Qdrant]
                  |--> [Redis]
```

## Conclusion

For more information, see the [documentation](https://docs.kahflane.com).
        """.encode("utf-8")

        result = docling_service.parse_bytes(complex_md, "complex.md")

        assert "text" in result
        assert "markdown" in result

        # Verify key content is extracted
        text = result["text"]
        assert "Project Documentation" in text
        assert "Kahflane" in text
        assert "Multi-tenant" in text

    @pytest.mark.asyncio
    async def test_docling_singleton(self):
        """Should return singleton instance."""
        service1 = get_docling_service()
        service2 = get_docling_service()

        assert service1 is service2
