"""
RAG (Retrieval-Augmented Generation) service.

Orchestrates the Q&A pipeline:
1. Query understanding and embedding
2. Vector retrieval from Qdrant
3. Access control filtering
4. Context assembly
5. LLM generation
6. Source attribution
"""
import logging
import json
from datetime import timedelta
from typing import Optional, List, Dict, Any, AsyncIterator
from uuid import uuid4
from dataclasses import dataclass, asdict

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.search_service import SearchService
from app.services.llm_service import (
    get_llm_service,
    LLMService,
    GenerationConfig,
    Source,
    RAGResponse,
)
from app.services.redis_service import get_redis_service, RedisService

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_TOP_K = 5
DEFAULT_SCORE_THRESHOLD = 0.6
CONVERSATION_TTL = timedelta(hours=24)
MAX_CONVERSATION_MESSAGES = 20


@dataclass
class ConversationMessage:
    """A message in conversation history."""

    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[Dict]] = None


@dataclass
class Conversation:
    """Conversation state with history."""

    id: str
    messages: List[ConversationMessage]
    created_at: str
    last_updated: str


class RAGService:
    """
    Service for RAG-based Q&A over documents.

    Coordinates retrieval and generation with:
    - Tenant isolation (inherited from SearchService)
    - Access control (user can only query accessible documents)
    - Conversation history (stored in Redis)
    """

    def __init__(
        self,
        db: AsyncSession,
        user_id: str,
        llm_service: Optional[LLMService] = None,
        redis_service: Optional[RedisService] = None,
    ):
        """
        Initialize RAG service.

        Args:
            db: Tenant-scoped database session
            user_id: Current user's ID
            llm_service: Optional LLM service (uses singleton if not provided)
            redis_service: Optional Redis service (uses singleton if not provided)
        """
        self.search_service = SearchService(db=db, user_id=user_id)
        self.llm_service = llm_service or get_llm_service()
        self.redis_service = redis_service or get_redis_service()
        self.user_id = user_id

    async def ask(
        self,
        question: str,
        org_id: Optional[str] = None,
        team_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        config: Optional[GenerationConfig] = None,
    ) -> RAGResponse:
        """
        Answer a question using RAG over accessible documents.

        Pipeline:
        1. Load conversation history (if conversation_id provided)
        2. Retrieve relevant chunks via SearchService
        3. Assemble context with conversation history
        4. Generate answer via LLMService
        5. Save to conversation history
        6. Return answer with sources

        Args:
            question: User's question
            org_id: Optional organization filter
            team_id: Optional team filter
            document_ids: Optional specific document IDs to search
            conversation_id: Optional ID for conversation continuity
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score
            config: Generation configuration

        Returns:
            RAGResponse with answer and sources
        """
        # Step 1: Load or create conversation
        conversation = None
        if conversation_id:
            conversation = await self._load_conversation(conversation_id)

        if conversation is None:
            conversation_id = str(uuid4())
            conversation = Conversation(
                id=conversation_id,
                messages=[],
                created_at=self._now_iso(),
                last_updated=self._now_iso(),
            )

        # Step 2: Retrieve relevant chunks
        if document_ids:
            # Search within specific documents
            context_chunks = await self._search_documents(
                question=question,
                document_ids=document_ids,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        else:
            # Search across all accessible documents
            context_chunks = await self.search_service.search(
                query=question,
                org_id=org_id,
                team_id=team_id,
                limit=top_k,
                score_threshold=score_threshold,
            )

        if not context_chunks:
            # No relevant content found - still save the conversation
            no_content_answer = "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing or ensure the relevant documents have been uploaded."
            conversation.messages.append(
                ConversationMessage(role="user", content=question)
            )
            conversation.messages.append(
                ConversationMessage(role="assistant", content=no_content_answer, sources=[])
            )
            conversation.last_updated = self._now_iso()
            await self._save_conversation(conversation)

            return RAGResponse(
                answer=no_content_answer,
                sources=[],
                conversation_id=conversation_id,
            )

        # Step 3: Build prompt with conversation history
        prompt = self._build_prompt_with_history(
            question=question,
            context_chunks=context_chunks,
            conversation=conversation,
        )

        # Step 4: Generate answer
        system_prompt = self._get_rag_system_prompt()
        answer = await self.llm_service.generate(
            prompt=prompt,
            system_instruction=system_prompt,
            config=config,
        )

        # Step 5: Extract sources
        sources = self._extract_sources(context_chunks)

        # Step 6: Save to conversation history
        conversation.messages.append(
            ConversationMessage(role="user", content=question)
        )
        conversation.messages.append(
            ConversationMessage(
                role="assistant",
                content=answer,
                sources=[asdict(s) for s in sources],
            )
        )

        # Trim conversation if too long
        if len(conversation.messages) > MAX_CONVERSATION_MESSAGES:
            conversation.messages = conversation.messages[-MAX_CONVERSATION_MESSAGES:]

        conversation.last_updated = self._now_iso()
        await self._save_conversation(conversation)

        return RAGResponse(
            answer=answer,
            sources=sources,
            conversation_id=conversation_id,
        )

    async def ask_stream(
        self,
        question: str,
        org_id: Optional[str] = None,
        team_id: Optional[str] = None,
        document_ids: Optional[List[str]] = None,
        conversation_id: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Answer a question with streaming response.

        Yields SSE events:
        - {"type": "token", "content": "..."}
        - {"type": "sources", "sources": [...]}
        - {"type": "done", "conversation_id": "..."}

        Args:
            question: User's question
            org_id: Optional organization filter
            team_id: Optional team filter
            document_ids: Optional specific document IDs
            conversation_id: Optional conversation ID
            top_k: Number of chunks to retrieve
            score_threshold: Minimum similarity score
            config: Generation configuration

        Yields:
            SSE event dictionaries
        """
        # Load or create conversation
        conversation = None
        if conversation_id:
            conversation = await self._load_conversation(conversation_id)

        if conversation is None:
            conversation_id = str(uuid4())
            conversation = Conversation(
                id=conversation_id,
                messages=[],
                created_at=self._now_iso(),
                last_updated=self._now_iso(),
            )

        # Retrieve relevant chunks
        if document_ids:
            context_chunks = await self._search_documents(
                question=question,
                document_ids=document_ids,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        else:
            context_chunks = await self.search_service.search(
                query=question,
                org_id=org_id,
                team_id=team_id,
                limit=top_k,
                score_threshold=score_threshold,
            )

        if not context_chunks:
            no_content_answer = "I couldn't find any relevant information in the documents to answer your question."
            yield {"type": "token", "content": no_content_answer}
            yield {"type": "sources", "sources": []}

            # Save conversation even with no content
            conversation.messages.append(
                ConversationMessage(role="user", content=question)
            )
            conversation.messages.append(
                ConversationMessage(role="assistant", content=no_content_answer, sources=[])
            )
            conversation.last_updated = self._now_iso()
            await self._save_conversation(conversation)

            yield {"type": "done", "conversation_id": conversation_id}
            return

        # Build prompt and stream response
        prompt = self._build_prompt_with_history(
            question=question,
            context_chunks=context_chunks,
            conversation=conversation,
        )

        system_prompt = self._get_rag_system_prompt()
        full_answer = []

        async for chunk in self.llm_service.generate_stream(
            prompt=prompt,
            system_instruction=system_prompt,
            config=config,
        ):
            full_answer.append(chunk)
            yield {"type": "token", "content": chunk}

        # Extract and send sources
        sources = self._extract_sources(context_chunks)
        yield {
            "type": "sources",
            "sources": [asdict(s) for s in sources],
        }

        # Save conversation
        answer = "".join(full_answer)
        conversation.messages.append(
            ConversationMessage(role="user", content=question)
        )
        conversation.messages.append(
            ConversationMessage(
                role="assistant",
                content=answer,
                sources=[asdict(s) for s in sources],
            )
        )

        if len(conversation.messages) > MAX_CONVERSATION_MESSAGES:
            conversation.messages = conversation.messages[-MAX_CONVERSATION_MESSAGES:]

        conversation.last_updated = self._now_iso()
        await self._save_conversation(conversation)

        yield {"type": "done", "conversation_id": conversation_id}

    async def summarize_document(
        self,
        document_id: str,
        max_length: int = 500,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate a summary of a document.

        Args:
            document_id: Document to summarize
            max_length: Approximate max words in summary
            config: Generation configuration

        Returns:
            Document summary

        Raises:
            HTTPException: If document not found or access denied
        """
        # Search to get document chunks (this validates access)
        chunks = await self.search_service.search_document(
            document_id=document_id,
            query="*",  # Get all chunks
            limit=50,
            score_threshold=0.0,
        )

        if not chunks:
            return "Document has no indexed content to summarize."

        # Get document title and concatenate text
        title = chunks[0]["document"]["title"] if chunks else "Document"
        full_text = "\n\n".join([c["text"] for c in chunks])

        # Generate summary
        return await self.llm_service.summarize_document(
            document_text=full_text,
            title=title,
            max_length=max_length,
            config=config,
        )

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Conversation]:
        """Get conversation by ID."""
        return await self._load_conversation(conversation_id)

    async def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        key = self._conversation_key(conversation_id)
        deleted = await self.redis_service.delete(key, namespace="conversation")
        return deleted > 0

    # ============ Private Methods ============

    async def _search_documents(
        self,
        question: str,
        document_ids: List[str],
        top_k: int,
        score_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Search within specific documents."""
        all_results = []

        for doc_id in document_ids:
            try:
                results = await self.search_service.search_document(
                    document_id=doc_id,
                    query=question,
                    limit=top_k,
                    score_threshold=score_threshold,
                )
                all_results.extend(results)
            except Exception as e:
                # Skip documents that fail (access denied, not found, etc.)
                logger.warning(f"Could not search document {doc_id}: {e}")
                continue

        # Sort by score and limit
        all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        return all_results[:top_k]

    def _build_prompt_with_history(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        conversation: Conversation,
    ) -> str:
        """Build prompt including conversation history."""
        parts = []

        # Add conversation history (last few exchanges)
        if conversation.messages:
            history_parts = []
            for msg in conversation.messages[-6:]:  # Last 3 exchanges
                role = "User" if msg.role == "user" else "Assistant"
                history_parts.append(f"{role}: {msg.content}")

            if history_parts:
                parts.append("CONVERSATION HISTORY:")
                parts.append("\n".join(history_parts))
                parts.append("")

        # Add context
        parts.append("RELEVANT CONTEXT FROM DOCUMENTS:")
        for i, chunk in enumerate(context_chunks, 1):
            doc_info = chunk.get("document", {})
            title = doc_info.get("title", "Unknown Document")
            text = chunk.get("text", "")
            parts.append(f"\n[Source {i}: {title}]")
            parts.append(text)

        parts.append("")
        parts.append(f"CURRENT QUESTION: {question}")
        parts.append("")
        parts.append("ANSWER:")

        return "\n".join(parts)

    def _get_rag_system_prompt(self) -> str:
        """Get system prompt for RAG."""
        return """You are a helpful knowledge assistant for an organization.
Your role is to answer questions using ONLY the information provided in the context.
Do not make up information or use external knowledge.

Guidelines:
- Be accurate and cite your sources (e.g., "According to Source 1...")
- If information is not in the context, clearly state that
- Consider the conversation history for follow-up questions
- Keep answers concise but complete
- Use bullet points for multiple items
- Maintain a professional, helpful tone"""

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Source]:
        """Extract source objects from chunks."""
        sources = []
        seen_docs = set()

        for chunk in chunks:
            doc_info = chunk.get("document", {})
            doc_id = doc_info.get("id", "")

            # Deduplicate by document
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            sources.append(
                Source(
                    document_id=doc_id,
                    title=doc_info.get("title", "Unknown"),
                    chunk_text=chunk.get("text", "")[:200] + "...",
                    relevance_score=chunk.get("score", 0.0),
                    chunk_index=chunk.get("chunk_index"),
                )
            )

        return sources

    def _conversation_key(self, conversation_id: str) -> str:
        """Build Redis key for conversation."""
        return f"conv:{self.user_id}:{conversation_id}"

    async def _load_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Conversation]:
        """Load conversation from Redis."""
        key = self._conversation_key(conversation_id)
        data = await self.redis_service.get_json(key, namespace="conversation")

        if data is None:
            return None

        messages = [
            ConversationMessage(
                role=m["role"],
                content=m["content"],
                sources=m.get("sources"),
            )
            for m in data.get("messages", [])
        ]

        return Conversation(
            id=data["id"],
            messages=messages,
            created_at=data["created_at"],
            last_updated=data["last_updated"],
        )

    async def _save_conversation(self, conversation: Conversation) -> bool:
        """Save conversation to Redis."""
        key = self._conversation_key(conversation.id)
        data = {
            "id": conversation.id,
            "messages": [asdict(m) for m in conversation.messages],
            "created_at": conversation.created_at,
            "last_updated": conversation.last_updated,
        }
        return await self.redis_service.set_json(
            key,
            data,
            namespace="conversation",
            ttl=CONVERSATION_TTL,
        )

    def _now_iso(self) -> str:
        """Get current time as ISO string."""
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
