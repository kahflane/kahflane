"""
Docling service for document parsing and conversion.

Provides PDF, DOCX, HTML, and other document format parsing with OCR,
table extraction, and chunking capabilities.
"""
import logging
import tempfile
import os
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from io import BytesIO

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
)
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.pipeline.simple_pipeline import SimplePipeline
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.chunking import HybridChunker

from app.core.config import settings

logger = logging.getLogger(__name__)

# Default chunking configuration
DEFAULT_MAX_TOKENS = 512
DEFAULT_MERGE_PEERS = True


class DoclingService:
    """Service for document parsing using Docling."""

    def __init__(
        self,
        enable_ocr: bool = True,
        enable_table_structure: bool = True,
        table_mode: str = "accurate",
        ocr_languages: Optional[List[str]] = None,
    ):
        """
        Initialize the Docling service.

        Args:
            enable_ocr: Whether to enable OCR for scanned documents
            enable_table_structure: Whether to extract table structures
            table_mode: Table extraction mode ("accurate" or "fast")
            ocr_languages: List of language codes for OCR (default: ["en"])
        """
        self.enable_ocr = enable_ocr
        self.enable_table_structure = enable_table_structure
        self.table_mode = table_mode
        self.ocr_languages = ocr_languages or ["en"]

        # Initialize converter
        self._converter = self._create_converter()

    def _create_converter(self) -> DocumentConverter:
        """Create and configure the document converter."""
        # Configure PDF pipeline options
        pdf_pipeline_options = PdfPipelineOptions()
        pdf_pipeline_options.do_ocr = self.enable_ocr
        pdf_pipeline_options.do_table_structure = self.enable_table_structure

        # Configure table structure options
        if self.enable_table_structure:
            table_mode = (
                TableFormerMode.ACCURATE
                if self.table_mode == "accurate"
                else TableFormerMode.FAST
            )
            pdf_pipeline_options.table_structure_options = TableStructureOptions(
                do_cell_matching=True,
                mode=table_mode,
            )

        # Create converter with format-specific options
        converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.MD,
                InputFormat.ASCIIDOC,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                ),
                InputFormat.DOCX: WordFormatOption(
                    pipeline_cls=SimplePipeline,
                ),
            },
        )

        return converter

    @property
    def converter(self) -> DocumentConverter:
        """Get the document converter instance."""
        return self._converter

    def parse_file(
        self,
        file_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Parse a document from file path.

        Args:
            file_path: Path to the document file

        Returns:
            Dict containing:
                - text: Extracted plain text
                - markdown: Markdown formatted text
                - html: HTML formatted text
                - page_count: Number of pages (if applicable)
                - tables: List of extracted tables
                - metadata: Document metadata
        """
        try:
            result = self._converter.convert(str(file_path))
            return self._process_result(result)
        except Exception as e:
            logger.error(f"Failed to parse file {file_path}: {e}")
            raise

    def parse_bytes(
        self,
        file_data: bytes,
        filename: str,
    ) -> Dict[str, Any]:
        """
        Parse a document from bytes.

        Args:
            file_data: Raw file bytes
            filename: Original filename (used to determine format)

        Returns:
            Dict containing parsed document data
        """
        try:
            # Create DocumentStream from bytes
            buf = BytesIO(file_data)
            source = DocumentStream(name=filename, stream=buf)

            result = self._converter.convert(source)
            return self._process_result(result)
        except Exception as e:
            logger.error(f"Failed to parse bytes for {filename}: {e}")
            raise

    def parse_url(
        self,
        url: str,
    ) -> Dict[str, Any]:
        """
        Parse a document from URL.

        Args:
            url: URL to the document

        Returns:
            Dict containing parsed document data
        """
        try:
            result = self._converter.convert(url)
            return self._process_result(result)
        except Exception as e:
            logger.error(f"Failed to parse URL {url}: {e}")
            raise

    def _process_result(self, result) -> Dict[str, Any]:
        """Process conversion result into standardized output."""
        doc = result.document

        # Extract text in various formats
        text = doc.export_to_markdown(strict_text=True)
        markdown = doc.export_to_markdown()

        # Try to get HTML, fall back to markdown if not available
        try:
            html = doc.export_to_html()
        except Exception:
            html = None

        # Get document dict for metadata
        doc_dict = doc.export_to_dict()

        # Extract tables if present
        tables = []
        if hasattr(doc, 'tables') and doc.tables:
            for table in doc.tables:
                tables.append({
                    "data": table.export_to_dataframe().to_dict() if hasattr(table, 'export_to_dataframe') else None,
                    "html": table.export_to_html() if hasattr(table, 'export_to_html') else None,
                })

        # Count pages if available
        page_count = None
        if hasattr(doc, 'pages'):
            page_count = len(doc.pages)
        elif 'pages' in doc_dict:
            page_count = len(doc_dict.get('pages', []))

        return {
            "text": text,
            "markdown": markdown,
            "html": html,
            "page_count": page_count,
            "tables": tables,
            "table_count": len(tables),
            "metadata": {
                "parser": "docling",
                "format": doc_dict.get("origin", {}).get("mimetype") if "origin" in doc_dict else None,
            },
            "_document": doc,  # Keep reference for chunking
        }

    def chunk_document(
        self,
        parsed_result: Dict[str, Any],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        merge_peers: bool = DEFAULT_MERGE_PEERS,
        include_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Chunk a parsed document for RAG applications.

        Args:
            parsed_result: Result from parse_* methods
            max_tokens: Maximum tokens per chunk
            merge_peers: Whether to merge small adjacent chunks
            include_context: Whether to include hierarchical context

        Returns:
            List of chunk dicts with text and metadata
        """
        doc = parsed_result.get("_document")
        if doc is None:
            raise ValueError("Parsed result does not contain document reference")

        try:
            chunker = HybridChunker(
                merge_peers=merge_peers,
            )

            chunks = []
            for i, chunk in enumerate(chunker.chunk(dl_doc=doc)):
                chunk_data = {
                    "text": chunk.text,
                    "chunk_index": i,
                }

                # Add contextualized text if requested
                if include_context:
                    try:
                        chunk_data["contextualized_text"] = chunker.contextualize(chunk=chunk)
                    except Exception:
                        chunk_data["contextualized_text"] = chunk.text

                # Add metadata if available
                if hasattr(chunk, 'meta'):
                    try:
                        chunk_data["metadata"] = chunk.meta.export_json_dict()
                    except Exception:
                        pass

                chunks.append(chunk_data)

            logger.info(f"Created {len(chunks)} chunks from document")
            return chunks

        except Exception as e:
            logger.error(f"Failed to chunk document: {e}")
            raise

    def parse_and_chunk(
        self,
        file_data: bytes,
        filename: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        merge_peers: bool = DEFAULT_MERGE_PEERS,
    ) -> Dict[str, Any]:
        """
        Parse document and return both full text and chunks.

        Args:
            file_data: Raw file bytes
            filename: Original filename
            max_tokens: Maximum tokens per chunk
            merge_peers: Whether to merge small adjacent chunks

        Returns:
            Dict with text, markdown, chunks, and metadata
        """
        # Parse document
        parsed = self.parse_bytes(file_data, filename)

        # Chunk document
        chunks = self.chunk_document(
            parsed,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
        )

        # Remove internal document reference before returning
        result = {k: v for k, v in parsed.items() if not k.startswith("_")}
        result["chunks"] = chunks
        result["chunk_count"] = len(chunks)

        return result


# Singleton instance
_docling_service: Optional[DoclingService] = None


def get_docling_service(
    enable_ocr: bool = True,
    enable_table_structure: bool = True,
) -> DoclingService:
    """
    Get or create Docling service singleton.

    Args:
        enable_ocr: Whether to enable OCR
        enable_table_structure: Whether to enable table extraction

    Returns:
        DoclingService instance
    """
    global _docling_service
    if _docling_service is None:
        _docling_service = DoclingService(
            enable_ocr=enable_ocr,
            enable_table_structure=enable_table_structure,
        )
    return _docling_service
