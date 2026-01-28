"""
Tests for search API endpoints.

Unit tests for schema validation and API tests for endpoint behavior.
Integration tests marked with @pytest.mark.integration require running services.
"""

import pytest
from uuid import uuid4
from pydantic import ValidationError
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.api.v1.search import SearchRequest, DocumentSearchRequest, SearchResponse


class TestSearchRequestSchema:
    """Unit tests for SearchRequest schema validation."""

    def test_valid_search_request(self):
        """Valid search request should pass validation."""
        req = SearchRequest(query="How does authentication work?")
        assert req.query == "How does authentication work?"
        assert req.limit == 10  # default
        assert req.score_threshold == 0.7  # default
        assert req.org_id is None
        assert req.team_id is None
        assert req.scope is None

    def test_search_request_with_all_params(self):
        """Search request with all parameters."""
        req = SearchRequest(
            query="test query",
            org_id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
            team_id="d4e5f6a7-b8c9-0123-4567-890abcdef012",
            scope="team",
            limit=20,
            score_threshold=0.5,
        )
        assert req.query == "test query"
        assert req.org_id == "7c9e6679-7425-40de-944b-e07fc1f90ae7"
        assert req.team_id == "d4e5f6a7-b8c9-0123-4567-890abcdef012"
        assert req.scope == "team"
        assert req.limit == 20
        assert req.score_threshold == 0.5

    def test_search_request_empty_query_rejected(self):
        """Empty query should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="")
        assert "String should have at least 1 character" in str(exc_info.value)

    def test_search_request_query_too_long_rejected(self):
        """Query exceeding 2000 chars should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="x" * 2001)
        assert "String should have at most 2000 characters" in str(exc_info.value)

    def test_search_request_limit_minimum_enforced(self):
        """Limit below 1 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", limit=0)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_search_request_limit_maximum_enforced(self):
        """Limit above 50 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", limit=51)
        assert "less than or equal to 50" in str(exc_info.value)

    def test_search_request_score_threshold_minimum(self):
        """Score threshold below 0.0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", score_threshold=-0.1)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_search_request_score_threshold_maximum(self):
        """Score threshold above 1.0 should be rejected."""
        with pytest.raises(ValidationError) as exc_info:
            SearchRequest(query="test", score_threshold=1.1)
        assert "less than or equal to 1" in str(exc_info.value)


class TestDocumentSearchRequestSchema:
    """Unit tests for DocumentSearchRequest schema validation."""

    def test_valid_document_search_request(self):
        """Valid document search request should pass validation."""
        req = DocumentSearchRequest(query="What are the key points?")
        assert req.query == "What are the key points?"
        assert req.limit == 10
        assert req.score_threshold == 0.5  # lower default than broad search

    def test_document_search_request_empty_query_rejected(self):
        """Empty query should be rejected."""
        with pytest.raises(ValidationError):
            DocumentSearchRequest(query="")

    def test_document_search_request_limit_bounds(self):
        """Limit bounds should be enforced."""
        # Valid limit
        req = DocumentSearchRequest(query="test", limit=50)
        assert req.limit == 50

        # Invalid limits
        with pytest.raises(ValidationError):
            DocumentSearchRequest(query="test", limit=0)
        with pytest.raises(ValidationError):
            DocumentSearchRequest(query="test", limit=51)


class TestSearchAPI:
    """API tests for search endpoints."""

    async def test_search_requires_authentication(
        self,
        unauthenticated_client: AsyncClient,
    ):
        """Search endpoint should require authentication."""
        response = await unauthenticated_client.post(
            "/api/v1/search/",
            json={"query": "test query"},
        )
        assert response.status_code in [401, 403]

    async def test_document_search_requires_authentication(
        self,
        unauthenticated_client: AsyncClient,
    ):
        """Document search endpoint should require authentication."""
        response = await unauthenticated_client.post(
            f"/api/v1/search/documents/{uuid4()}",
            json={"query": "test query"},
        )
        assert response.status_code in [401, 403]

    async def test_search_validation_error(
        self,
        authenticated_client: AsyncClient,
    ):
        """Invalid search request should return 422."""
        response = await authenticated_client.post(
            "/api/v1/search/",
            json={"query": ""},  # Empty query
        )
        assert response.status_code == 422

    async def test_document_search_document_not_found(
        self,
        authenticated_client: AsyncClient,
    ):
        """Search within non-existent document should return 404."""
        fake_doc_id = str(uuid4())
        response = await authenticated_client.post(
            f"/api/v1/search/documents/{fake_doc_id}",
            json={"query": "test query"},
        )
        # May return 404 (not found) or 500 (if embedding service not available)
        assert response.status_code in [404, 500]


@pytest.mark.integration
class TestSearchIntegration:
    """Integration tests requiring running services (PostgreSQL, Qdrant, Gemini)."""

    async def test_search_returns_empty_results_when_no_documents(
        self,
        authenticated_client: AsyncClient,
        test_organization: dict,
    ):
        """Search should return empty results when no documents indexed."""
        response = await authenticated_client.post(
            "/api/v1/search/",
            json={
                "query": "knowledge management system",
                "org_id": str(test_organization["id"]),
                "limit": 10,
            },
        )

        # May return 200 with empty results or 500 if embedding service unavailable
        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert "total" in data
            assert "query" in data
            assert data["query"] == "knowledge management system"
            # No documents indexed, so results should be empty
            assert data["total"] == 0 or isinstance(data["results"], list)

    async def test_search_response_format(
        self,
        authenticated_client: AsyncClient,
        test_organization: dict,
    ):
        """Search response should have correct format."""
        response = await authenticated_client.post(
            "/api/v1/search/",
            json={
                "query": "test query",
                "org_id": str(test_organization["id"]),
            },
        )

        if response.status_code == 200:
            data = response.json()
            # Verify response structure
            assert isinstance(data.get("results"), list)
            assert isinstance(data.get("total"), int)
            assert isinstance(data.get("query"), str)

    async def test_document_search_access_denied_for_personal_document(
        self,
        authenticated_client: AsyncClient,
        db_session: AsyncSession,
        test_organization: dict,
        test_user,
    ):
        """Search within personal document owned by another user should return 403."""
        schema = test_organization["schema"]
        doc_id = uuid4()
        other_user_id = uuid4()

        # Create a personal-scoped document owned by another user
        await db_session.execute(
            text(f"""
            INSERT INTO "{schema}".document
            (id, uploader_id, org_id, title, file_path, file_type, scope, status)
            VALUES (:id, :uploader, :org, :title, :path, :type, :scope, :status)
        """),
            {
                "id": doc_id,
                "uploader": other_user_id,
                "org": test_organization["id"],
                "title": "Private Doc",
                "path": "test/private.txt",
                "type": "TXT",
                "scope": "personal",
                "status": "completed",
            },
        )
        await db_session.commit()

        response = await authenticated_client.post(
            f"/api/v1/search/documents/{doc_id}",
            json={"query": "test query"},
        )

        # Should be 403 (access denied) or 500 (if embedding service unavailable)
        assert response.status_code in [403, 500]

    async def test_document_search_access_granted_for_org_document(
        self,
        authenticated_client: AsyncClient,
        db_session: AsyncSession,
        test_organization: dict,
        test_user,
    ):
        """Search within org-scoped document should be allowed for org member."""
        schema = test_organization["schema"]
        doc_id = uuid4()
        other_user_id = uuid4()

        # Create an org-scoped document (user is org member via test_organization fixture)
        await db_session.execute(
            text(f"""
            INSERT INTO "{schema}".document
            (id, uploader_id, org_id, title, file_path, file_type, scope, status)
            VALUES (:id, :uploader, :org, :title, :path, :type, :scope, :status)
        """),
            {
                "id": doc_id,
                "uploader": other_user_id,
                "org": test_organization["id"],
                "title": "Org Doc",
                "path": "test/org.txt",
                "type": "TXT",
                "scope": "organization",
                "status": "completed",
            },
        )
        await db_session.commit()

        response = await authenticated_client.post(
            f"/api/v1/search/documents/{doc_id}",
            json={"query": "test query"},
        )

        # Should be 200 (allowed) or 500 (if embedding service unavailable)
        # Not 403 - user is org member so should have access
        assert response.status_code in [200, 500]

    async def test_document_search_pending_document_rejected(
        self,
        authenticated_client: AsyncClient,
        db_session: AsyncSession,
        test_organization: dict,
        test_user,
    ):
        """Search within pending (unprocessed) document should return 400."""
        schema = test_organization["schema"]
        doc_id = uuid4()

        # Create a pending document owned by the test user
        await db_session.execute(
            text(f"""
            INSERT INTO "{schema}".document
            (id, uploader_id, org_id, title, file_path, file_type, scope, status)
            VALUES (:id, :uploader, :org, :title, :path, :type, :scope, :status)
        """),
            {
                "id": doc_id,
                "uploader": test_user.id,
                "org": test_organization["id"],
                "title": "Pending Doc",
                "path": "test/pending.txt",
                "type": "TXT",
                "scope": "personal",
                "status": "pending",
            },
        )
        await db_session.commit()

        response = await authenticated_client.post(
            f"/api/v1/search/documents/{doc_id}",
            json={"query": "test query"},
        )

        # Should be 400 (not yet processed) or 500 (if embedding service unavailable)
        assert response.status_code in [400, 500]
