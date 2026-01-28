"""
Vector store tests.

Tests for Qdrant vector operations with per-tenant collection isolation.
Uses real Qdrant on port 6334.
"""
import pytest
from uuid import uuid4

from app.core.tenant import TenantContext, set_current_tenant
from app.services.vector_store import (
    VectorStoreService,
    VECTOR_DIMENSION,
    COLLECTION_PREFIX,
)


@pytest.fixture
def vector_service():
    """Get vector store service connected to real Qdrant."""
    return VectorStoreService()


@pytest.fixture
def sample_tenant_context() -> TenantContext:
    """Sample tenant context with unique slug for test isolation."""
    slug = f"test_{uuid4().hex[:8]}"
    return TenantContext(
        tenant_id=str(uuid4()),
        tenant_slug=slug,
        schema_name=f"tenant_{slug}",
    )


@pytest.fixture
def sample_chunks() -> list:
    """Sample document chunks with embeddings."""
    return [
        {
            "text": "First chunk of content about Kahflane.",
            "embedding": [0.1] * VECTOR_DIMENSION,
            "chunk_index": 0,
        },
        {
            "text": "Second chunk about knowledge management.",
            "embedding": [0.2] * VECTOR_DIMENSION,
            "chunk_index": 1,
        },
        {
            "text": "Third chunk about AI and search.",
            "embedding": [0.3] * VECTOR_DIMENSION,
            "chunk_index": 2,
        },
    ]


class TestVectorStoreCollectionNaming:
    """Test collection name generation."""

    def test_collection_name_format(
        self,
        vector_service: VectorStoreService,
        sample_tenant_context: TenantContext,
    ):
        """Should generate correct collection name for tenant."""
        set_current_tenant(sample_tenant_context)
        try:
            collection_name = vector_service._get_collection_name()
            assert collection_name == f"{COLLECTION_PREFIX}{sample_tenant_context.tenant_slug}"
        finally:
            set_current_tenant(None)

    def test_no_tenant_raises_error(self, vector_service: VectorStoreService):
        """Should raise error without tenant context."""
        set_current_tenant(None)
        with pytest.raises(ValueError, match="No tenant context"):
            vector_service._get_collection_name()


class TestVectorStoreOperations:
    """Test vector store CRUD operations with real Qdrant."""

    async def test_ensure_collection_creates_new(
        self,
        vector_service: VectorStoreService,
        sample_tenant_context: TenantContext,
    ):
        """Should create collection if it doesn't exist."""
        set_current_tenant(sample_tenant_context)
        try:
            collection_name = await vector_service.ensure_tenant_collection_exists()
            expected = f"{COLLECTION_PREFIX}{sample_tenant_context.tenant_slug}"
            assert collection_name == expected

            # Verify it actually exists in Qdrant
            exists = await vector_service.collection_exists(collection_name)
            assert exists is True
        finally:
            await vector_service.delete_tenant_collection()
            set_current_tenant(None)

    async def test_ensure_collection_idempotent(
        self,
        vector_service: VectorStoreService,
        sample_tenant_context: TenantContext,
    ):
        """Should not fail when collection already exists."""
        set_current_tenant(sample_tenant_context)
        try:
            name1 = await vector_service.ensure_tenant_collection_exists()
            name2 = await vector_service.ensure_tenant_collection_exists()
            assert name1 == name2
        finally:
            await vector_service.delete_tenant_collection()
            set_current_tenant(None)

    async def test_index_and_search(
        self,
        vector_service: VectorStoreService,
        sample_tenant_context: TenantContext,
        sample_chunks: list,
    ):
        """Should index chunks and search them."""
        set_current_tenant(sample_tenant_context)
        doc_id = uuid4()
        org_id = uuid4()

        try:
            count = await vector_service.index_document_chunks(
                document_id=doc_id,
                chunks=sample_chunks,
                org_id=org_id,
                scope="team",
            )
            assert count == len(sample_chunks)

            # Search with the same embedding as first chunk
            results = await vector_service.search(
                query_embedding=[0.1] * VECTOR_DIMENSION,
                limit=10,
                score_threshold=0.0,
            )

            assert len(results) > 0
            # Results should have expected fields
            assert "text" in results[0]
            assert "score" in results[0]
            assert "document_id" in results[0]
        finally:
            await vector_service.delete_tenant_collection()
            set_current_tenant(None)

    async def test_index_no_tenant_raises_error(
        self,
        vector_service: VectorStoreService,
        sample_chunks: list,
    ):
        """Should raise error without tenant context."""
        set_current_tenant(None)

        with pytest.raises(ValueError, match="No tenant context"):
            await vector_service.index_document_chunks(
                document_id=uuid4(),
                chunks=sample_chunks,
                org_id=uuid4(),
            )

    async def test_search_empty_collection(
        self,
        vector_service: VectorStoreService,
        sample_tenant_context: TenantContext,
    ):
        """Should return empty results if collection doesn't exist."""
        set_current_tenant(sample_tenant_context)
        try:
            results = await vector_service.search(
                query_embedding=[0.1] * VECTOR_DIMENSION,
                limit=10,
            )
            assert results == []
        finally:
            set_current_tenant(None)

    async def test_delete_document_vectors(
        self,
        vector_service: VectorStoreService,
        sample_tenant_context: TenantContext,
        sample_chunks: list,
    ):
        """Should delete vectors for a specific document."""
        set_current_tenant(sample_tenant_context)
        doc_id = uuid4()
        org_id = uuid4()

        try:
            await vector_service.index_document_chunks(
                document_id=doc_id,
                chunks=sample_chunks,
                org_id=org_id,
                scope="team",
            )

            await vector_service.delete_document_vectors(doc_id)

            # Verify chunks are deleted
            chunk_count = await vector_service.get_document_chunk_count(doc_id)
            assert chunk_count == 0
        finally:
            await vector_service.delete_tenant_collection()
            set_current_tenant(None)

    async def test_delete_tenant_collection(
        self,
        vector_service: VectorStoreService,
        sample_tenant_context: TenantContext,
    ):
        """Should delete entire tenant collection."""
        set_current_tenant(sample_tenant_context)
        try:
            await vector_service.ensure_tenant_collection_exists()
            result = await vector_service.delete_tenant_collection()
            assert result is True

            # Collection should no longer exist
            vector_service._existing_collections.clear()
            collection_name = vector_service._get_collection_name()
            exists = await vector_service.collection_exists(collection_name)
            assert exists is False
        finally:
            set_current_tenant(None)

    async def test_get_collection_stats(
        self,
        vector_service: VectorStoreService,
        sample_tenant_context: TenantContext,
        sample_chunks: list,
    ):
        """Should return collection statistics."""
        set_current_tenant(sample_tenant_context)
        try:
            await vector_service.index_document_chunks(
                document_id=uuid4(),
                chunks=sample_chunks,
                org_id=uuid4(),
                scope="team",
            )

            stats = await vector_service.get_collection_stats()
            assert stats["exists"] is True
            assert stats["collection_name"] == f"{COLLECTION_PREFIX}{sample_tenant_context.tenant_slug}"
            assert stats["vectors_count"] >= len(sample_chunks)
        finally:
            await vector_service.delete_tenant_collection()
            set_current_tenant(None)

    async def test_collection_stats_nonexistent(
        self,
        vector_service: VectorStoreService,
        sample_tenant_context: TenantContext,
    ):
        """Should return exists=False for nonexistent collection."""
        set_current_tenant(sample_tenant_context)
        try:
            stats = await vector_service.get_collection_stats()
            assert stats["exists"] is False
            assert stats["vectors_count"] == 0
        finally:
            set_current_tenant(None)


class TestVectorStoreTenantIsolation:
    """Test tenant isolation with real Qdrant."""

    async def test_different_tenants_different_collections(
        self,
        vector_service: VectorStoreService,
    ):
        """Different tenants should use different collections."""
        slug1 = f"test_{uuid4().hex[:8]}"
        slug2 = f"test_{uuid4().hex[:8]}"
        tenant1 = TenantContext(tenant_id="id-1", tenant_slug=slug1, schema_name=f"tenant_{slug1}")
        tenant2 = TenantContext(tenant_id="id-2", tenant_slug=slug2, schema_name=f"tenant_{slug2}")

        set_current_tenant(tenant1)
        collection1 = vector_service._get_collection_name()
        set_current_tenant(tenant2)
        collection2 = vector_service._get_collection_name()
        set_current_tenant(None)

        assert collection1 != collection2
        assert collection1 == f"kahflane_{slug1}"
        assert collection2 == f"kahflane_{slug2}"

    async def test_tenant_data_isolation(
        self,
        sample_chunks: list,
    ):
        """Tenant data should be isolated in separate collections."""
        service = VectorStoreService()
        slug1 = f"test_{uuid4().hex[:8]}"
        slug2 = f"test_{uuid4().hex[:8]}"
        tenant1 = TenantContext(tenant_id="id-1", tenant_slug=slug1, schema_name=f"tenant_{slug1}")
        tenant2 = TenantContext(tenant_id="id-2", tenant_slug=slug2, schema_name=f"tenant_{slug2}")

        try:
            # Index data for tenant1
            set_current_tenant(tenant1)
            await service.index_document_chunks(
                document_id=uuid4(),
                chunks=sample_chunks,
                org_id=uuid4(),
                scope="team",
            )

            # Tenant2 should see empty results
            set_current_tenant(tenant2)
            results = await service.search(
                query_embedding=[0.1] * VECTOR_DIMENSION,
                limit=10,
                score_threshold=0.0,
            )
            assert results == []
        finally:
            set_current_tenant(tenant1)
            await service.delete_tenant_collection()
            set_current_tenant(tenant2)
            await service.delete_tenant_collection()
            set_current_tenant(None)


class TestVectorStoreHealthCheck:
    """Test Qdrant health check."""

    async def test_health_check_healthy(self, vector_service: VectorStoreService):
        """Should return True when Qdrant is healthy."""
        is_healthy = await vector_service.health_check()
        assert is_healthy is True
