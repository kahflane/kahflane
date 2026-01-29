"""
Embedding service using Google Gemini.

Generates text embeddings for document chunks using Google's gemini-embedding-001 model.
"""
import logging
from typing import List, Optional

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

# Model configuration
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072  # Gemini embedding model output dimension
MAX_BATCH_SIZE = 100  # Maximum texts per API call


class EmbeddingService:
    """Service for generating text embeddings using Google Gemini."""

    def __init__(self):
        """Initialize the embedding service."""
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for embedding service")

        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model = EMBEDDING_MODEL

    async def embed_text(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            task_type: Task type for embedding optimization
                - RETRIEVAL_DOCUMENT: For indexing documents
                - RETRIEVAL_QUERY: For search queries
                - SEMANTIC_SIMILARITY: For similarity comparisons
                - CLASSIFICATION: For classification tasks
                - CLUSTERING: For clustering tasks

        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def embed_texts(
        self,
        texts: List[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            task_type: Task type for embedding optimization

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = []

        # Process in batches
        for i in range(0, len(texts), MAX_BATCH_SIZE):
            batch = texts[i : i + MAX_BATCH_SIZE]
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                # Extract embedding values from response
                batch_embeddings = [emb.values for emb in response.embeddings]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                raise

        return embeddings

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.

        Uses RETRIEVAL_QUERY task type for optimal search performance.

        Args:
            query: Search query text

        Returns:
            Embedding vector
        """
        return await self.embed_text(query, task_type="RETRIEVAL_QUERY")

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents (for indexing).

        Uses RETRIEVAL_DOCUMENT task type for optimal retrieval performance.

        Args:
            documents: List of document texts

        Returns:
            List of embedding vectors
        """
        return await self.embed_texts(documents, task_type="RETRIEVAL_DOCUMENT")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return EMBEDDING_DIMENSION


# Singleton instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create embedding service singleton."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
