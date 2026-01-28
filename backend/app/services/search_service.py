"""
Search service for semantic search with access control.

Orchestrates query embedding, vector search, access filtering, and result enrichment.
"""

import logging
from typing import List, Dict, Any, Optional, Set
from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.document import Document, DocumentStatus
from app.models.organization import OrganizationMember, TeamMember
from app.services.embedding_service import get_embedding_service
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class SearchService:
    """
    Service for semantic search with access control.

    Coordinates embedding, vector search, and PostgreSQL access checks
    to return only documents the user is authorized to view.
    """

    def __init__(self, db: AsyncSession, user_id: str):
        """
        Initialize search service.

        Args:
            db: Tenant-scoped database session
            user_id: Current user's ID
        """
        self.db = db
        self.user_id = UUID(user_id)
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()
        # Cache for user memberships (loaded lazily)
        self._user_org_ids: Optional[Set[UUID]] = None
        self._user_team_ids: Optional[Set[UUID]] = None

    async def search(
        self,
        query: str,
        org_id: Optional[str] = None,
        team_id: Optional[str] = None,
        scope: Optional[str] = None,
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across accessible documents.

        Pipeline:
        1. Embed query text
        2. Search Qdrant (over-fetch to account for access filtering)
        3. Filter results by user's access permissions
        4. Enrich with document metadata
        5. Trim to requested limit

        Args:
            query: Search query text
            org_id: Optional organization filter
            team_id: Optional team filter
            scope: Optional scope filter (personal, team, organization)
            limit: Maximum results to return
            score_threshold: Minimum similarity score

        Returns:
            List of search results with document metadata
        """
        # Step 1: Embed query
        query_embedding = await self.embedding_service.embed_query(query)

        # Step 2: Search Qdrant (over-fetch 3x to account for access filtering)
        raw_results = await self.vector_store.search(
            query_embedding=query_embedding,
            org_id=UUID(org_id) if org_id else None,
            team_id=UUID(team_id) if team_id else None,
            scope=scope,
            limit=limit * 3,
            score_threshold=score_threshold,
        )

        if not raw_results:
            return []

        # Step 3: Get unique document IDs and load accessible documents
        doc_ids = {UUID(r["document_id"]) for r in raw_results if r.get("document_id")}
        accessible_docs = await self._get_accessible_documents(doc_ids)
        accessible_doc_ids = {str(doc.id) for doc in accessible_docs}
        doc_map = {str(doc.id): doc for doc in accessible_docs}

        # Step 4: Filter and enrich results
        results = []
        for hit in raw_results:
            doc_id = hit.get("document_id")
            if doc_id not in accessible_doc_ids:
                continue

            doc = doc_map[doc_id]
            results.append(
                {
                    "text": hit.get("text", ""),
                    "score": hit.get("score", 0.0),
                    "chunk_index": hit.get("chunk_index"),
                    "document": {
                        "id": str(doc.id),
                        "title": doc.title,
                        "file_type": doc.file_type,
                        "scope": doc.scope,
                        "org_id": str(doc.org_id),
                        "team_id": str(doc.team_id) if doc.team_id else None,
                    },
                }
            )

            if len(results) >= limit:
                break

        return results

    async def search_document(
        self,
        document_id: str,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search within a single document.

        Args:
            document_id: Document to search within
            query: Search query text
            limit: Maximum results to return
            score_threshold: Minimum similarity score

        Returns:
            List of search results from the document

        Raises:
            HTTPException: 404 if document not found, 403 if access denied
        """
        # Step 1: Verify document exists and user has access
        doc = await self._get_document_with_access_check(UUID(document_id))

        # Step 2: Embed query
        query_embedding = await self.embedding_service.embed_query(query)

        # Step 3: Search Qdrant with document_id filter
        raw_results = await self.vector_store.search(
            query_embedding=query_embedding,
            document_id=doc.id,
            limit=limit,
            score_threshold=score_threshold,
        )

        # Step 4: Format results
        results = []
        for hit in raw_results:
            results.append(
                {
                    "text": hit.get("text", ""),
                    "score": hit.get("score", 0.0),
                    "chunk_index": hit.get("chunk_index"),
                    "document": {
                        "id": str(doc.id),
                        "title": doc.title,
                        "file_type": doc.file_type,
                        "scope": doc.scope,
                        "org_id": str(doc.org_id),
                        "team_id": str(doc.team_id) if doc.team_id else None,
                    },
                }
            )

        return results

    async def _get_accessible_documents(
        self,
        candidate_doc_ids: Set[UUID],
    ) -> List[Document]:
        """
        Filter candidate documents by user's access permissions.

        Access rules:
        - personal: only the uploader can access
        - team: only team members can access
        - organization: any organization member can access

        Args:
            candidate_doc_ids: Set of document IDs to check

        Returns:
            List of documents the user can access
        """
        if not candidate_doc_ids:
            return []

        # Load candidate documents (only completed ones have indexed vectors)
        result = await self.db.execute(
            select(Document).where(
                Document.id.in_(candidate_doc_ids),
                Document.status == DocumentStatus.COMPLETED.value,
            )
        )
        candidates = list(result.scalars().all())

        if not candidates:
            return []

        # Load user's memberships for batch access checking
        user_org_ids = await self._get_user_org_ids()
        user_team_ids = await self._get_user_team_ids()

        # Filter by access
        accessible = []
        for doc in candidates:
            if self._user_can_access_document(doc, user_org_ids, user_team_ids):
                accessible.append(doc)

        return accessible

    async def _get_document_with_access_check(self, document_id: UUID) -> Document:
        """
        Load a document and verify the user has access.

        Args:
            document_id: Document ID to load

        Returns:
            Document if accessible

        Raises:
            HTTPException: 404 if not found, 400 if not processed, 403 if denied
        """
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()

        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        if doc.status != DocumentStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document has not been processed yet",
            )

        # Check access
        user_org_ids = await self._get_user_org_ids()
        user_team_ids = await self._get_user_team_ids()

        if not self._user_can_access_document(doc, user_org_ids, user_team_ids):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have access to this document",
            )

        return doc

    def _user_can_access_document(
        self,
        doc: Document,
        user_org_ids: Set[UUID],
        user_team_ids: Set[UUID],
    ) -> bool:
        """
        Check if user can access a document based on its scope.

        Args:
            doc: Document to check
            user_org_ids: Set of org IDs the user belongs to
            user_team_ids: Set of team IDs the user belongs to

        Returns:
            True if user can access the document
        """
        if doc.scope == "personal":
            return doc.uploader_id == self.user_id
        elif doc.scope == "team":
            return doc.team_id is not None and doc.team_id in user_team_ids
        elif doc.scope == "organization":
            return doc.org_id in user_org_ids
        else:
            # Unknown scope - deny by default
            return False

    async def _get_user_org_ids(self) -> Set[UUID]:
        """Get all organization IDs the user belongs to (cached)."""
        if self._user_org_ids is None:
            result = await self.db.execute(
                select(OrganizationMember.org_id).where(
                    OrganizationMember.user_id == self.user_id
                )
            )
            self._user_org_ids = {row[0] for row in result.all()}
        return self._user_org_ids

    async def _get_user_team_ids(self) -> Set[UUID]:
        """Get all team IDs the user belongs to (cached)."""
        if self._user_team_ids is None:
            result = await self.db.execute(
                select(TeamMember.team_id).where(TeamMember.user_id == self.user_id)
            )
            self._user_team_ids = {row[0] for row in result.all()}
        return self._user_team_ids
