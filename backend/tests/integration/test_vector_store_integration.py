"""
Vector store integration tests.

Tests real Qdrant operations with per-tenant collections.
"""
import pytest
from uuid import uuid4

from qdrant_client.models import Distance, VectorParams

from app.core.tenant import TenantContext, set_current_tenant
from app.services.vector_store import VectorStoreService, VECTOR_DIMENSION, COLLECTION_PREFIX


pytestmark = pytest.mark.integration


class TestQdrantConnection:
    """Test Qdrant connectivity."""

    @pytest.mark.asyncio
    async def test_qdrant_health(self, qdrant_client):
        """Should connect to Qdrant successfully."""
        # Get collections list (verifies connection)
        collections = qdrant_client.get_collections()
        assert collections is not None

    @pytest.mark.asyncio
    async def test_create_collection(self, qdrant_client):
        """Should create and delete collection."""
        collection_name = f"test_collection_{uuid4().hex[:8]}"

        # Create collection
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=VECTOR_DIMENSION,
                distance=Distance.COSINE,
            ),
        )

        # Verify it exists
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        assert collection_name in collection_names

        # Delete collection
        qdrant_client.delete_collection(collection_name)

        # Verify it's gone
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]
        assert collection_name not in collection_names


class TestVectorStoreServiceIntegration:
    """Test VectorStoreService with real Qdrant."""

    @pytest.fixture
    def vector_service(self) -> VectorStoreService:
        """Get VectorStoreService with test configuration."""
        # Explicitly pass test configuration to avoid .env file conflicts
        return VectorStoreService(
            url="http://localhost:6334",
            api_key="",  # Empty string to explicitly disable API key
        )

    @pytest.fixture
    def sample_chunks(self) -> list:
        """Sample document chunks with real-ish embeddings."""
        import random
        random.seed(42)

        return [
            {
                "text": "Kahflane is an AI-powered knowledge management system.",
                "embedding": [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)],
                "chunk_index": 0,
            },
            {
                "text": "The platform supports multi-tenant architecture.",
                "embedding": [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)],
                "chunk_index": 1,
            },
            {
                "text": "Documents are processed using advanced AI models.",
                "embedding": [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)],
                "chunk_index": 2,
            },
        ]

    @pytest.mark.asyncio
    async def test_ensure_tenant_collection(
        self,
        vector_service: VectorStoreService,
        tenant_context: TenantContext,
        qdrant_client,
    ):
        """Should create tenant collection if not exists."""
        set_current_tenant(tenant_context)
        collection_name = None

        try:
            collection_name = await vector_service.ensure_tenant_collection_exists()

            # Verify collection was created
            assert collection_name == f"{COLLECTION_PREFIX}{tenant_context.tenant_slug}"

            # Verify in Qdrant
            collections = qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            assert collection_name in collection_names

        finally:
            set_current_tenant(None)
            # Cleanup
            if collection_name:
                try:
                    qdrant_client.delete_collection(collection_name)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_upsert_vectors(
        self,
        vector_service: VectorStoreService,
        tenant_context: TenantContext,
        sample_chunks: list,
        qdrant_client,
    ):
        """Should upsert vectors into tenant collection."""
        set_current_tenant(tenant_context)
        collection_name = None

        try:
            document_id = str(uuid4())
            org_id = str(uuid4())
            uploader_id = str(uuid4())

            # Ensure collection exists
            collection_name = await vector_service.ensure_tenant_collection_exists()

            # Upsert vectors
            indexed_count = await vector_service.index_document_chunks(
                document_id=document_id,
                chunks=sample_chunks,
                org_id=org_id,
                team_id=None,
                uploader_id=uploader_id,
                scope="organization",
            )

            assert indexed_count == len(sample_chunks)

            # Verify points in Qdrant
            collection_info = qdrant_client.get_collection(collection_name)
            assert collection_info.points_count >= len(sample_chunks)

        finally:
            set_current_tenant(None)
            # Cleanup
            if collection_name:
                try:
                    qdrant_client.delete_collection(collection_name)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_search_vectors(
        self,
        vector_service: VectorStoreService,
        tenant_context: TenantContext,
        sample_chunks: list,
        qdrant_client,
    ):
        """Should search for similar vectors."""
        set_current_tenant(tenant_context)
        collection_name = None

        try:
            document_id = str(uuid4())
            org_id = str(uuid4())
            uploader_id = str(uuid4())

            # Setup: ensure collection and upsert vectors
            collection_name = await vector_service.ensure_tenant_collection_exists()
            await vector_service.index_document_chunks(
                document_id=document_id,
                chunks=sample_chunks,
                org_id=org_id,
                team_id=None,
                uploader_id=uploader_id,
                scope="organization",
            )

            # Search using first chunk's embedding as query
            query_embedding = sample_chunks[0]["embedding"]

            results = await vector_service.search(
                query_embedding=query_embedding,
                org_id=org_id,
                limit=5,
            )

            assert len(results) > 0

            # First result should be very similar (same embedding)
            first_result = results[0]
            assert first_result["score"] > 0.9  # High similarity
            assert "text" in first_result["payload"]

        finally:
            set_current_tenant(None)
            # Cleanup
            if collection_name:
                try:
                    qdrant_client.delete_collection(collection_name)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_delete_document_vectors(
        self,
        vector_service: VectorStoreService,
        tenant_context: TenantContext,
        sample_chunks: list,
        qdrant_client,
    ):
        """Should delete document vectors."""
        set_current_tenant(tenant_context)
        collection_name = None

        try:
            document_id = str(uuid4())
            org_id = str(uuid4())
            uploader_id = str(uuid4())

            # Setup
            collection_name = await vector_service.ensure_tenant_collection_exists()
            await vector_service.index_document_chunks(
                document_id=document_id,
                chunks=sample_chunks,
                org_id=org_id,
                team_id=None,
                uploader_id=uploader_id,
                scope="organization",
            )

            # Verify points exist
            info_before = qdrant_client.get_collection(collection_name)
            count_before = info_before.points_count

            # Delete document vectors
            await vector_service.delete_document_vectors(document_id)

            # Verify points deleted
            info_after = qdrant_client.get_collection(collection_name)
            count_after = info_after.points_count

            assert count_after < count_before

        finally:
            set_current_tenant(None)
            # Cleanup
            if collection_name:
                try:
                    qdrant_client.delete_collection(collection_name)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_tenant_collection_isolation(
        self,
        vector_service: VectorStoreService,
        tenant_context: TenantContext,
        sample_chunks: list,
        qdrant_client,
    ):
        """Should isolate vectors by tenant collection."""
        # Create second tenant context
        tenant2_context = TenantContext(
            tenant_id=str(uuid4()),
            tenant_slug=f"test_{uuid4().hex[:8]}",
            schema_name=f"tenant_test_{uuid4().hex[:8]}",
        )

        document_id = str(uuid4())
        org_id = str(uuid4())
        uploader_id = str(uuid4())

        collection1 = None
        collection2 = None

        try:
            # Insert into tenant 1
            set_current_tenant(tenant_context)
            collection1 = await vector_service.ensure_tenant_collection_exists()
            await vector_service.index_document_chunks(
                document_id=document_id,
                chunks=sample_chunks,
                org_id=org_id,
                team_id=None,
                uploader_id=uploader_id,
                scope="organization",
            )

            # Insert into tenant 2
            set_current_tenant(tenant2_context)
            collection2 = await vector_service.ensure_tenant_collection_exists()
            await vector_service.index_document_chunks(
                document_id=str(uuid4()),  # Different document
                chunks=sample_chunks[:1],  # Only one chunk
                org_id=org_id,
                team_id=None,
                uploader_id=uploader_id,
                scope="organization",
            )

            # Verify collections are different
            assert collection1 != collection2

            # Verify each collection has correct count
            info1 = qdrant_client.get_collection(collection1)
            info2 = qdrant_client.get_collection(collection2)

            assert info1.points_count == len(sample_chunks)
            assert info2.points_count == 1

        finally:
            set_current_tenant(None)
            # Cleanup both collections
            for col in [collection1, collection2]:
                if col:
                    try:
                        qdrant_client.delete_collection(col)
                    except Exception:
                        pass


class TestVectorStorePerformance:
    """Test vector store performance."""

    @pytest.fixture
    def vector_service(self) -> VectorStoreService:
        """Get VectorStoreService."""
        # Explicitly pass test configuration to avoid .env file conflicts
        return VectorStoreService(
            url="http://localhost:6334",
            api_key="",  # Empty string to explicitly disable API key
        )

    @pytest.mark.asyncio
    async def test_bulk_upsert_performance(
        self,
        vector_service: VectorStoreService,
        tenant_context: TenantContext,
        qdrant_client,
    ):
        """Should handle bulk upserts efficiently."""
        import random
        import time

        random.seed(42)
        set_current_tenant(tenant_context)
        collection_name = None

        try:
            # Generate 100 chunks
            chunks = [
                {
                    "text": f"Chunk number {i} with some content for testing.",
                    "embedding": [random.uniform(-1, 1) for _ in range(VECTOR_DIMENSION)],
                    "chunk_index": i,
                }
                for i in range(100)
            ]

            document_id = str(uuid4())
            org_id = str(uuid4())
            uploader_id = str(uuid4())

            collection_name = await vector_service.ensure_tenant_collection_exists()

            start = time.time()

            await vector_service.index_document_chunks(
                document_id=document_id,
                chunks=chunks,
                org_id=org_id,
                team_id=None,
                uploader_id=uploader_id,
                scope="organization",
            )

            elapsed = time.time() - start

            # Should complete in reasonable time
            assert elapsed < 10.0, f"Bulk upsert too slow: {elapsed}s for 100 vectors"

            # Verify all points inserted
            info = qdrant_client.get_collection(collection_name)
            assert info.points_count >= 100

        finally:
            set_current_tenant(None)
            if collection_name:
                try:
                    qdrant_client.delete_collection(collection_name)
                except Exception:
                    pass
