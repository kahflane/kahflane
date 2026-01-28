"""
Vector store service for Qdrant operations.

Uses per-tenant collections for complete data isolation.
Each tenant gets their own Qdrant collection: kahflane_{tenant_slug}
"""
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.core.tenant import get_current_tenant, TenantContext

logger = logging.getLogger(__name__)

# Vector dimension for text-embedding-004 model
VECTOR_DIMENSION = 768

# Collection name prefix
COLLECTION_PREFIX = "kahflane_"


class VectorStoreService:
    """
    Service for managing document vectors in Qdrant.

    Uses separate collections per tenant for complete isolation.
    Collection naming: kahflane_{tenant_slug}
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Qdrant client.

        Args:
            url: Full Qdrant URL (e.g., "http://localhost:6333" or "https://cloud.qdrant.io")
                 If not provided, constructed from settings
            api_key: Qdrant API key. Pass empty string "" to explicitly disable.
                     If None, uses settings.QDRANT_KEY
        """
        # Use get_settings() to get fresh config (important for tests)
        from app.core.config import get_settings
        current_settings = get_settings()

        # Determine URL
        if url:
            qdrant_url = url
        else:
            qdrant_url = current_settings.QDRANT_URL

        # Determine API key
        # Empty string "" means explicitly no API key (for testing)
        # None means use settings
        if api_key == "":
            effective_api_key = None
        elif api_key is not None:
            effective_api_key = api_key
        else:
            effective_api_key = current_settings.QDRANT_KEY if current_settings.QDRANT_KEY else None

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=effective_api_key,
        )
        # Cache of existing collections to avoid repeated checks
        self._existing_collections: set = set()

    def _get_collection_name(self, tenant: Optional[TenantContext] = None) -> str:
        """
        Get collection name for the current tenant.

        Args:
            tenant: Optional tenant context, uses current if not provided

        Returns:
            Collection name in format: kahflane_{tenant_slug}
        """
        if tenant is None:
            tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context for collection name")

        return f"{COLLECTION_PREFIX}{tenant.tenant_slug}"

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists."""
        if collection_name in self._existing_collections:
            return True

        try:
            collections = self.client.get_collections()
            existing = {c.name for c in collections.collections}
            self._existing_collections = existing
            return collection_name in existing
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            raise

    async def ensure_tenant_collection_exists(
        self,
        tenant: Optional[TenantContext] = None,
    ) -> str:
        """
        Create collection for tenant if it doesn't exist.

        Args:
            tenant: Optional tenant context, uses current if not provided

        Returns:
            The collection name
        """
        collection_name = self._get_collection_name(tenant)

        if await self.collection_exists(collection_name):
            return collection_name

        try:
            logger.info(f"Creating Qdrant collection: {collection_name}")

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_DIMENSION,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=True,
            )

            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="document_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="org_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="team_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="scope",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="uploader_id",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

            self._existing_collections.add(collection_name)
            logger.info(f"Created Qdrant collection with indexes: {collection_name}")

            return collection_name

        except Exception as e:
            # Check if it was created by another process
            if await self.collection_exists(collection_name):
                return collection_name
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise

    async def index_document_chunks(
        self,
        document_id: UUID,
        chunks: List[Dict[str, Any]],
        org_id: UUID,
        team_id: Optional[UUID] = None,
        uploader_id: Optional[UUID] = None,
        scope: str = "team",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Index document chunks with their embeddings.

        Args:
            document_id: Document UUID
            chunks: List of dicts with 'text', 'embedding', and optional 'chunk_index'
            org_id: Organization UUID
            team_id: Optional team UUID
            uploader_id: Optional uploader user UUID
            scope: Document scope (personal, team, organization)
            metadata: Additional metadata to store

        Returns:
            Number of vectors indexed
        """
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context for vector indexing")

        collection_name = await self.ensure_tenant_collection_exists(tenant)

        points = []
        for i, chunk in enumerate(chunks):
            chunk_index = chunk.get("chunk_index", i)
            # Generate deterministic UUID from document_id and chunk_index
            # This allows idempotent upserts (same doc+chunk = same point ID)
            import uuid
            point_id = uuid.uuid5(uuid.NAMESPACE_OID, f"{document_id}_{chunk_index}")

            payload = {
                "document_id": str(document_id),
                "org_id": str(org_id),
                "scope": scope,
                "chunk_index": chunk_index,
                "text": chunk["text"],
            }

            if team_id:
                payload["team_id"] = str(team_id)
            if uploader_id:
                payload["uploader_id"] = str(uploader_id)
            if metadata:
                payload["metadata"] = metadata

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk["embedding"],
                    payload=payload,
                )
            )

        # Batch upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch,
            )

        logger.info(
            f"Indexed {len(points)} chunks for document {document_id} "
            f"in collection {collection_name}"
        )
        return len(points)

    async def search(
        self,
        query_embedding: List[float],
        org_id: Optional[UUID] = None,
        team_id: Optional[UUID] = None,
        uploader_id: Optional[UUID] = None,
        scope: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in tenant's collection.

        Args:
            query_embedding: Query vector
            org_id: Filter by organization
            team_id: Filter by team
            uploader_id: Filter by uploader (for personal scope)
            scope: Filter by scope
            limit: Maximum results
            score_threshold: Minimum similarity score

        Returns:
            List of search results with text, score, and metadata
        """
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context for vector search")

        collection_name = self._get_collection_name(tenant)

        # Check if collection exists
        if not await self.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist, returning empty results")
            return []

        # Build filter conditions (no tenant filter needed - collection is tenant-specific)
        must_conditions = []

        if org_id:
            must_conditions.append(
                models.FieldCondition(
                    key="org_id",
                    match=models.MatchValue(value=str(org_id)),
                )
            )

        if team_id:
            must_conditions.append(
                models.FieldCondition(
                    key="team_id",
                    match=models.MatchValue(value=str(team_id)),
                )
            )

        if uploader_id and scope == "personal":
            must_conditions.append(
                models.FieldCondition(
                    key="uploader_id",
                    match=models.MatchValue(value=str(uploader_id)),
                )
            )

        if scope:
            must_conditions.append(
                models.FieldCondition(
                    key="scope",
                    match=models.MatchValue(value=scope),
                )
            )

        filter_conditions = models.Filter(must=must_conditions) if must_conditions else None

        # Use query_points (modern API) instead of deprecated search
        results = self.client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            query_filter=filter_conditions,
            limit=limit,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "text": hit.payload.get("text") if hit.payload else None,
                "document_id": hit.payload.get("document_id") if hit.payload else None,
                "chunk_index": hit.payload.get("chunk_index") if hit.payload else None,
                "org_id": hit.payload.get("org_id") if hit.payload else None,
                "team_id": hit.payload.get("team_id") if hit.payload else None,
                "scope": hit.payload.get("scope") if hit.payload else None,
                "metadata": hit.payload.get("metadata") if hit.payload else None,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    async def delete_document_vectors(self, document_id: UUID) -> int:
        """
        Delete all vectors for a document.

        Args:
            document_id: Document UUID

        Returns:
            Number of vectors deleted (approximate)
        """
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context for vector deletion")

        collection_name = self._get_collection_name(tenant)

        if not await self.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return 0

        self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=str(document_id)),
                        ),
                    ]
                )
            ),
        )

        logger.info(f"Deleted vectors for document {document_id} in collection {collection_name}")
        return 1

    async def delete_tenant_collection(self) -> bool:
        """
        Delete the entire collection for the current tenant.

        Use with caution - this removes all vectors for the tenant.

        Returns:
            True if deleted, False if collection didn't exist
        """
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context for collection deletion")

        collection_name = self._get_collection_name(tenant)

        if not await self.collection_exists(collection_name):
            return False

        self.client.delete_collection(collection_name=collection_name)
        self._existing_collections.discard(collection_name)

        logger.info(f"Deleted collection {collection_name}")
        return True

    async def get_document_chunk_count(self, document_id: UUID) -> int:
        """Get the number of chunks indexed for a document."""
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context")

        collection_name = self._get_collection_name(tenant)

        if not await self.collection_exists(collection_name):
            return 0

        result = self.client.count(
            collection_name=collection_name,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=str(document_id)),
                    ),
                ]
            ),
        )
        return result.count

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for the current tenant's collection."""
        tenant = get_current_tenant()
        if not tenant:
            raise ValueError("No tenant context")

        collection_name = self._get_collection_name(tenant)

        if not await self.collection_exists(collection_name):
            return {
                "exists": False,
                "collection_name": collection_name,
                "vectors_count": 0,
            }

        info = self.client.get_collection(collection_name=collection_name)

        return {
            "exists": True,
            "collection_name": collection_name,
            "vectors_count": info.points_count or 0,
            "points_count": info.points_count or 0,
            "indexed_vectors_count": info.indexed_vectors_count or 0,
            "status": info.status.value if info.status else "unknown",
        }

    async def health_check(self) -> bool:
        """Check if Qdrant is healthy."""
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    async def list_tenant_collections(self) -> List[str]:
        """List all tenant collections (kahflane_* prefix)."""
        try:
            collections = self.client.get_collections()
            return [
                c.name for c in collections.collections
                if c.name.startswith(COLLECTION_PREFIX)
            ]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []


# Singleton instance
_vector_store: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    """Get or create vector store service singleton."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store
