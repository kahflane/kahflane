"""
RAG (Retrieval-Augmented Generation) integration tests.

Tests the full Q&A pipeline with real services:
- Qdrant for vector search
- Google Gemini for embeddings and text generation
- Redis for conversation history
- PostgreSQL for document metadata

Run with: pytest tests/integration/test_rag_integration.py -v -m integration
"""
import pytest
from uuid import uuid4
from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, AsyncEngine

from app.core.tenant import TenantContext, set_current_tenant
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import VectorStoreService, VECTOR_DIMENSION, COLLECTION_PREFIX
from app.services.llm_service import LLMService, GenerationConfig
from app.services.rag_service import RAGService
from app.services.redis_service import RedisService


pytestmark = pytest.mark.integration


@pytest.fixture
async def tenant_db_session(
    integration_engine: AsyncEngine,
    tenant_context: TenantContext,
) -> AsyncGenerator[AsyncSession, None]:
    """
    Get a tenant-scoped database session with schema_translate_map.

    This is required for RAGService which uses SQLModel queries
    that need schema translation.
    """
    schema_translate_map = {None: tenant_context.schema_name}

    async with integration_engine.connect() as connection:
        connection = await connection.execution_options(
            schema_translate_map=schema_translate_map
        )
        async with AsyncSession(
            bind=connection,
            expire_on_commit=False,
            autoflush=False,
        ) as session:
            yield session
            await session.rollback()


class TestLLMServiceIntegration:
    """Test LLM service with real Google Gemini API."""

    @pytest.fixture
    def llm_service(self) -> LLMService:
        """Get LLM service instance."""
        return LLMService()

    @pytest.mark.asyncio
    async def test_generate_simple_response(self, llm_service: LLMService):
        """Should generate a response to a simple prompt."""
        response = await llm_service.generate(
            prompt="What is 2 + 2? Answer with just the number.",
            config=GenerationConfig(temperature=0.1, max_output_tokens=10),
        )

        assert response is not None
        assert len(response) > 0
        assert "4" in response

    @pytest.mark.asyncio
    async def test_generate_with_system_instruction(self, llm_service: LLMService):
        """Should follow system instructions."""
        response = await llm_service.generate(
            prompt="What is your name?",
            system_instruction="You are an assistant named Alex. Always introduce yourself by name.",
            config=GenerationConfig(temperature=0.3, max_output_tokens=100),
        )

        assert response is not None
        assert "Alex" in response

    @pytest.mark.asyncio
    async def test_generate_with_context(self, llm_service: LLMService):
        """Should use provided context to answer questions."""
        context_chunks = [
            {
                "text": "The Kahflane project started in January 2024. It was founded by a team of AI researchers.",
                "score": 0.95,
                "document": {"id": "doc-1", "title": "About Kahflane"},
            },
            {
                "text": "Kahflane specializes in knowledge management using AI and multi-tenant architecture.",
                "score": 0.88,
                "document": {"id": "doc-1", "title": "About Kahflane"},
            },
        ]

        response = await llm_service.generate_with_context(
            question="When was Kahflane started?",
            context_chunks=context_chunks,
            config=GenerationConfig(temperature=0.3),
        )

        assert response is not None
        assert "2024" in response or "January" in response

    @pytest.mark.asyncio
    async def test_generate_stream(self, llm_service: LLMService):
        """Should stream responses token by token."""
        chunks = []

        async for chunk in llm_service.generate_stream(
            prompt="Count from 1 to 5.",
            config=GenerationConfig(temperature=0.1, max_output_tokens=50),
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert "1" in full_response
        assert "5" in full_response

    @pytest.mark.asyncio
    async def test_summarize_document(self, llm_service: LLMService):
        """Should summarize document content."""
        document_text = """
        Kahflane is an AI-powered Knowledge Management System.

        Key Features:
        - Multi-tenant architecture with PostgreSQL schema isolation
        - Vector search using Qdrant
        - Document processing with Docling
        - LLM integration with Google Gemini

        The system allows organizations to upload documents, which are then
        parsed, chunked, and indexed for semantic search. Users can ask
        natural language questions and get AI-generated answers based on
        their organization's documents.

        Security is a priority, with complete data isolation between tenants
        and role-based access control within organizations.
        """

        summary = await llm_service.summarize_document(
            document_text=document_text,
            title="Kahflane Overview",
            max_length=100,
            config=GenerationConfig(temperature=0.3),
        )

        assert summary is not None
        assert len(summary) > 50
        # Summary should mention key topics
        assert any(
            keyword in summary.lower()
            for keyword in ["knowledge", "ai", "document", "search", "tenant"]
        )


class TestRAGPipelineIntegration:
    """Test full RAG pipeline with real services."""

    @pytest.fixture
    def vector_service(self) -> VectorStoreService:
        """Get VectorStoreService with test configuration."""
        return VectorStoreService(
            url="http://localhost:6334",
            api_key="",
        )

    @pytest.fixture
    def redis_service(self) -> RedisService:
        """Get Redis service instance."""
        return RedisService()

    @pytest.fixture
    async def indexed_documents(
        self,
        vector_service: VectorStoreService,
        tenant_context: TenantContext,
        db_session,
        test_user,
        qdrant_client,
    ):
        """
        Set up indexed documents for RAG testing.

        Creates:
        - An organization
        - Sample documents in the database
        - Indexed vectors in Qdrant
        """
        set_current_tenant(tenant_context)
        embedding_service = get_embedding_service()
        collection_name = None

        try:
            # Create organization
            org_id = uuid4()
            await db_session.execute(
                text(f'''
                    INSERT INTO "{tenant_context.schema_name}".organization
                    (id, name, description)
                    VALUES (:id, :name, :description)
                '''),
                {"id": org_id, "name": "Test Org", "description": "Test organization"},
            )

            # Add user as org member
            await db_session.execute(
                text(f'''
                    INSERT INTO "{tenant_context.schema_name}".organization_member
                    (id, org_id, user_id, role)
                    VALUES (:id, :org_id, :user_id, :role)
                '''),
                {
                    "id": uuid4(),
                    "org_id": org_id,
                    "user_id": test_user.id,
                    "role": "owner",
                },
            )

            # Create sample documents
            documents = [
                {
                    "id": uuid4(),
                    "title": "Company Policies",
                    "content": [
                        "Employees are entitled to 25 days of annual leave per year.",
                        "Remote work is allowed up to 3 days per week with manager approval.",
                        "All employees must complete security training within 30 days of joining.",
                    ],
                },
                {
                    "id": uuid4(),
                    "title": "Product Roadmap Q1 2024",
                    "content": [
                        "Feature A: Implement vector search with Qdrant - Target: January",
                        "Feature B: Add multi-tenant support - Target: February",
                        "Feature C: Launch RAG Q&A pipeline - Target: March",
                    ],
                },
                {
                    "id": uuid4(),
                    "title": "Technical Architecture",
                    "content": [
                        "The backend uses FastAPI with Python 3.10+.",
                        "Database is PostgreSQL with schema-per-tenant isolation.",
                        "Vector embeddings are stored in Qdrant using cosine similarity.",
                    ],
                },
            ]

            # Ensure collection exists
            collection_name = await vector_service.ensure_tenant_collection_exists()

            # Index each document
            for doc in documents:
                # Insert into database
                await db_session.execute(
                    text(f'''
                        INSERT INTO "{tenant_context.schema_name}".document
                        (id, uploader_id, org_id, title, file_path, file_type, scope, status)
                        VALUES (:id, :uploader_id, :org_id, :title, :file_path, :file_type, :scope, :status)
                    '''),
                    {
                        "id": doc["id"],
                        "uploader_id": test_user.id,
                        "org_id": org_id,
                        "title": doc["title"],
                        "file_path": f"/test/{doc['id']}.txt",
                        "file_type": "TXT",
                        "scope": "organization",
                        "status": "completed",
                    },
                )

                # Generate embeddings and index
                chunks = []
                for i, content in enumerate(doc["content"]):
                    embedding = await embedding_service.embed_text(
                        content, task_type="RETRIEVAL_DOCUMENT"
                    )
                    chunks.append({
                        "text": content,
                        "embedding": embedding,
                        "chunk_index": i,
                    })

                await vector_service.index_document_chunks(
                    document_id=str(doc["id"]),
                    chunks=chunks,
                    org_id=str(org_id),
                    team_id=None,
                    uploader_id=str(test_user.id),
                    scope="organization",
                )

            await db_session.commit()

            yield {
                "org_id": org_id,
                "documents": documents,
                "collection_name": collection_name,
            }

        finally:
            set_current_tenant(None)
            # Cleanup collection
            if collection_name:
                try:
                    qdrant_client.delete_collection(collection_name)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_ask_about_leave_policy(
        self,
        indexed_documents,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
        redis_service: RedisService,
    ):
        """Should answer questions about leave policy from indexed documents."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(
                db=tenant_db_session,
                user_id=str(test_user.id),
                redis_service=redis_service,
            )

            response = await service.ask(
                question="How many days of annual leave do employees get?",
                org_id=str(indexed_documents["org_id"]),
                top_k=3,
            )

            assert response.answer is not None
            assert "25" in response.answer
            assert len(response.sources) > 0
            assert response.conversation_id is not None

            # Check source contains relevant document
            source_titles = [s.title for s in response.sources]
            assert "Company Policies" in source_titles

        finally:
            set_current_tenant(None)

    @pytest.mark.asyncio
    async def test_ask_about_technical_stack(
        self,
        indexed_documents,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
        redis_service: RedisService,
    ):
        """Should answer questions about technical architecture."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(
                db=tenant_db_session,
                user_id=str(test_user.id),
                redis_service=redis_service,
            )

            response = await service.ask(
                question="What database does the system use?",
                org_id=str(indexed_documents["org_id"]),
                top_k=3,
            )

            assert response.answer is not None
            assert "PostgreSQL" in response.answer or "postgres" in response.answer.lower()
            assert len(response.sources) > 0

        finally:
            set_current_tenant(None)

    @pytest.mark.asyncio
    async def test_ask_about_roadmap(
        self,
        indexed_documents,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
        redis_service: RedisService,
    ):
        """Should answer questions about product roadmap."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(
                db=tenant_db_session,
                user_id=str(test_user.id),
                redis_service=redis_service,
            )

            response = await service.ask(
                question="What features are planned for Q1 2024?",
                org_id=str(indexed_documents["org_id"]),
                top_k=5,
            )

            assert response.answer is not None
            # Should mention at least one feature
            assert any(
                keyword in response.answer.lower()
                for keyword in ["vector", "search", "tenant", "rag", "q&a"]
            )

        finally:
            set_current_tenant(None)

    @pytest.mark.asyncio
    async def test_conversation_continuity(
        self,
        indexed_documents,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
        redis_service: RedisService,
    ):
        """Should maintain context across conversation turns."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(
                db=tenant_db_session,
                user_id=str(test_user.id),
                redis_service=redis_service,
            )

            # First question
            response1 = await service.ask(
                question="Tell me about the remote work policy.",
                org_id=str(indexed_documents["org_id"]),
            )

            assert response1.conversation_id is not None
            conversation_id = response1.conversation_id

            # Follow-up question using conversation context
            response2 = await service.ask(
                question="How many days per week is that?",
                org_id=str(indexed_documents["org_id"]),
                conversation_id=conversation_id,
            )

            # Should understand "that" refers to remote work
            assert response2.answer is not None
            assert "3" in response2.answer

            # Verify conversation was saved
            conversation = await service.get_conversation(conversation_id)
            assert conversation is not None
            assert len(conversation.messages) == 4  # 2 user + 2 assistant

        finally:
            set_current_tenant(None)

    @pytest.mark.asyncio
    async def test_no_relevant_content(
        self,
        indexed_documents,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
        redis_service: RedisService,
    ):
        """Should handle questions with no relevant content gracefully."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(
                db=tenant_db_session,
                user_id=str(test_user.id),
                redis_service=redis_service,
            )

            response = await service.ask(
                question="What is the recipe for chocolate cake?",
                org_id=str(indexed_documents["org_id"]),
                score_threshold=0.9,  # High threshold to ensure no matches
            )

            assert response.answer is not None
            # Should indicate no relevant information found
            assert any(
                phrase in response.answer.lower()
                for phrase in ["couldn't find", "no relevant", "not found", "no information"]
            )

        finally:
            set_current_tenant(None)

    @pytest.mark.asyncio
    async def test_streaming_response(
        self,
        indexed_documents,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
        redis_service: RedisService,
    ):
        """Should stream response tokens correctly."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(
                db=tenant_db_session,
                user_id=str(test_user.id),
                redis_service=redis_service,
            )

            events = []
            async for event in service.ask_stream(
                question="What is the leave policy?",
                org_id=str(indexed_documents["org_id"]),
            ):
                events.append(event)

            # Should have token events
            token_events = [e for e in events if e["type"] == "token"]
            assert len(token_events) > 0

            # Should have sources event
            sources_events = [e for e in events if e["type"] == "sources"]
            assert len(sources_events) == 1
            assert len(sources_events[0]["sources"]) > 0

            # Should have done event
            done_events = [e for e in events if e["type"] == "done"]
            assert len(done_events) == 1
            assert "conversation_id" in done_events[0]

        finally:
            set_current_tenant(None)


class TestRAGAccessControl:
    """Test that RAG respects document access control."""

    @pytest.fixture
    def vector_service(self) -> VectorStoreService:
        """Get VectorStoreService."""
        return VectorStoreService(url="http://localhost:6334", api_key="")

    @pytest.fixture
    async def access_controlled_docs(
        self,
        vector_service: VectorStoreService,
        tenant_context: TenantContext,
        db_session,
        test_user,
        integration_engine,
        qdrant_client,
    ):
        """
        Set up documents with different access scopes.
        """
        set_current_tenant(tenant_context)
        embedding_service = get_embedding_service()
        collection_name = None

        # Create a second user
        other_user_id = uuid4()
        await db_session.execute(
            text('''
                INSERT INTO public."user" (id, email, password_hash, full_name, is_active)
                VALUES (:id, :email, :password_hash, :full_name, TRUE)
            '''),
            {
                "id": other_user_id,
                "email": f"other_{uuid4().hex[:8]}@example.com",
                "password_hash": "hashed",
                "full_name": "Other User",
            },
        )

        try:
            # Create organization
            org_id = uuid4()
            await db_session.execute(
                text(f'''
                    INSERT INTO "{tenant_context.schema_name}".organization
                    (id, name) VALUES (:id, :name)
                '''),
                {"id": org_id, "name": "Access Test Org"},
            )

            # Add test_user as org member
            await db_session.execute(
                text(f'''
                    INSERT INTO "{tenant_context.schema_name}".organization_member
                    (id, org_id, user_id, role) VALUES (:id, :org_id, :user_id, :role)
                '''),
                {"id": uuid4(), "org_id": org_id, "user_id": test_user.id, "role": "member"},
            )

            # Create team
            team_id = uuid4()
            await db_session.execute(
                text(f'''
                    INSERT INTO "{tenant_context.schema_name}".team
                    (id, org_id, name) VALUES (:id, :org_id, :name)
                '''),
                {"id": team_id, "org_id": org_id, "name": "Engineering"},
            )

            # Add test_user to team
            await db_session.execute(
                text(f'''
                    INSERT INTO "{tenant_context.schema_name}".team_member
                    (id, team_id, user_id, role) VALUES (:id, :team_id, :user_id, :role)
                '''),
                {"id": uuid4(), "team_id": team_id, "user_id": test_user.id, "role": "member"},
            )

            collection_name = await vector_service.ensure_tenant_collection_exists()

            # Document 1: Organization scope (accessible to test_user)
            org_doc_id = uuid4()
            await db_session.execute(
                text(f'''
                    INSERT INTO "{tenant_context.schema_name}".document
                    (id, uploader_id, org_id, title, file_path, file_type, scope, status)
                    VALUES (:id, :uploader_id, :org_id, :title, :file_path, :file_type, :scope, :status)
                '''),
                {
                    "id": org_doc_id,
                    "uploader_id": other_user_id,
                    "org_id": org_id,
                    "title": "Org Document",
                    "file_path": "/test/org.txt",
                    "file_type": "TXT",
                    "scope": "organization",
                    "status": "completed",
                },
            )

            org_embedding = await embedding_service.embed_text(
                "The company budget for 2024 is $10 million.", task_type="RETRIEVAL_DOCUMENT"
            )
            await vector_service.index_document_chunks(
                document_id=str(org_doc_id),
                chunks=[{"text": "The company budget for 2024 is $10 million.", "embedding": org_embedding, "chunk_index": 0}],
                org_id=str(org_id),
                team_id=None,
                uploader_id=str(other_user_id),
                scope="organization",
            )

            # Document 2: Personal scope (NOT accessible to test_user)
            personal_doc_id = uuid4()
            await db_session.execute(
                text(f'''
                    INSERT INTO "{tenant_context.schema_name}".document
                    (id, uploader_id, org_id, title, file_path, file_type, scope, status)
                    VALUES (:id, :uploader_id, :org_id, :title, :file_path, :file_type, :scope, :status)
                '''),
                {
                    "id": personal_doc_id,
                    "uploader_id": other_user_id,
                    "org_id": org_id,
                    "title": "Personal Document",
                    "file_path": "/test/personal.txt",
                    "file_type": "TXT",
                    "scope": "personal",
                    "status": "completed",
                },
            )

            personal_embedding = await embedding_service.embed_text(
                "My secret password is hunter2.", task_type="RETRIEVAL_DOCUMENT"
            )
            await vector_service.index_document_chunks(
                document_id=str(personal_doc_id),
                chunks=[{"text": "My secret password is hunter2.", "embedding": personal_embedding, "chunk_index": 0}],
                org_id=str(org_id),
                team_id=None,
                uploader_id=str(other_user_id),
                scope="personal",
            )

            await db_session.commit()

            yield {
                "org_id": org_id,
                "team_id": team_id,
                "org_doc_id": org_doc_id,
                "personal_doc_id": personal_doc_id,
                "other_user_id": other_user_id,
                "collection_name": collection_name,
            }

        finally:
            set_current_tenant(None)
            # Cleanup
            await db_session.execute(
                text('DELETE FROM public."user" WHERE id = :id'),
                {"id": other_user_id},
            )
            await db_session.commit()
            if collection_name:
                try:
                    qdrant_client.delete_collection(collection_name)
                except Exception:
                    pass

    @pytest.mark.asyncio
    async def test_can_access_org_scoped_document(
        self,
        access_controlled_docs,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
    ):
        """Should be able to query organization-scoped documents."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(db=tenant_db_session, user_id=str(test_user.id))

            response = await service.ask(
                question="What is the company budget?",
                org_id=str(access_controlled_docs["org_id"]),
            )

            # Should find the org document
            assert response.answer is not None
            assert "$10 million" in response.answer or "10 million" in response.answer

        finally:
            set_current_tenant(None)

    @pytest.mark.asyncio
    async def test_cannot_access_personal_document(
        self,
        access_controlled_docs,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
    ):
        """Should NOT be able to query another user's personal documents."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(db=tenant_db_session, user_id=str(test_user.id))

            response = await service.ask(
                question="What is the secret password?",
                org_id=str(access_controlled_docs["org_id"]),
                score_threshold=0.5,  # Lower threshold to catch any matches
            )

            # Should NOT contain the secret
            assert "hunter2" not in response.answer.lower()

        finally:
            set_current_tenant(None)


class TestConversationHistory:
    """Test conversation history management."""

    @pytest.fixture
    def redis_service(self) -> RedisService:
        """Get Redis service."""
        return RedisService()

    @pytest.mark.asyncio
    async def test_save_and_load_conversation(
        self,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
        redis_service: RedisService,
    ):
        """Should persist and retrieve conversation history."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(
                db=tenant_db_session,
                user_id=str(test_user.id),
                redis_service=redis_service,
            )

            # Create a conversation (this will be empty since no documents, but tests storage)
            response = await service.ask(
                question="Hello, how are you?",
                score_threshold=0.99,  # Ensure no documents match
            )

            conversation_id = response.conversation_id

            # Load the conversation
            conversation = await service.get_conversation(conversation_id)

            assert conversation is not None
            assert conversation.id == conversation_id
            assert len(conversation.messages) == 2  # user + assistant

        finally:
            set_current_tenant(None)

    @pytest.mark.asyncio
    async def test_delete_conversation(
        self,
        tenant_context: TenantContext,
        tenant_db_session,
        test_user,
        redis_service: RedisService,
    ):
        """Should delete conversation history."""
        set_current_tenant(tenant_context)

        try:
            service = RAGService(
                db=tenant_db_session,
                user_id=str(test_user.id),
                redis_service=redis_service,
            )

            # Create a conversation
            response = await service.ask(
                question="Test question",
                score_threshold=0.99,
            )

            conversation_id = response.conversation_id

            # Delete it
            deleted = await service.delete_conversation(conversation_id)
            assert deleted is True

            # Should not exist anymore
            conversation = await service.get_conversation(conversation_id)
            assert conversation is None

        finally:
            set_current_tenant(None)

    @pytest.mark.asyncio
    async def test_conversation_isolation(
        self,
        tenant_context: TenantContext,
        db_session,
        tenant_db_session,
        test_user,
        redis_service: RedisService,
        integration_engine,
    ):
        """Conversations should be isolated per user."""
        set_current_tenant(tenant_context)

        # Create second user (using public schema db_session)
        other_user_id = uuid4()
        await db_session.execute(
            text('''
                INSERT INTO public."user" (id, email, password_hash, full_name, is_active)
                VALUES (:id, :email, :password_hash, :full_name, TRUE)
            '''),
            {
                "id": other_user_id,
                "email": f"isolation_{uuid4().hex[:8]}@example.com",
                "password_hash": "hashed",
                "full_name": "Isolation Test User",
            },
        )
        await db_session.commit()

        try:
            # User 1 creates conversation
            service1 = RAGService(
                db=tenant_db_session,
                user_id=str(test_user.id),
                redis_service=redis_service,
            )

            response1 = await service1.ask(
                question="User 1 question",
                score_threshold=0.99,
            )

            conversation_id = response1.conversation_id

            # User 2 should NOT be able to access it
            service2 = RAGService(
                db=tenant_db_session,
                user_id=str(other_user_id),
                redis_service=redis_service,
            )

            conversation = await service2.get_conversation(conversation_id)
            # Should be None because Redis key includes user_id
            assert conversation is None

        finally:
            set_current_tenant(None)
            # Cleanup
            await db_session.execute(
                text('DELETE FROM public."user" WHERE id = :id'),
                {"id": other_user_id},
            )
            await db_session.commit()
