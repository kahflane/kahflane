"""
LLM service for text generation using Google Gemini.

Provides text generation, streaming, and chat capabilities
separate from the embedding service.
"""
import logging
from typing import Optional, List, AsyncIterator, Dict, Any
from dataclasses import dataclass

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

# Model configuration
GENERATION_MODEL = "gemini-2.0-flash"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TEMPERATURE = 0.7


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = DEFAULT_TEMPERATURE
    max_output_tokens: int = DEFAULT_MAX_TOKENS
    top_p: float = 0.95
    top_k: int = 40


@dataclass
class Source:
    """A source document chunk used in RAG response."""

    document_id: str
    title: str
    chunk_text: str
    relevance_score: float
    chunk_index: Optional[int] = None


@dataclass
class RAGResponse:
    """Response from RAG Q&A pipeline."""

    answer: str
    sources: List[Source]
    conversation_id: Optional[str] = None
    model: str = GENERATION_MODEL


class LLMService:
    """Service for text generation using Google Gemini."""

    def __init__(self):
        """Initialize the LLM service."""
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required for LLM service")

        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
        self.model = GENERATION_MODEL

    async def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The user prompt/question
            system_instruction: Optional system instruction
            config: Generation configuration

        Returns:
            Generated text response
        """
        if config is None:
            config = GenerationConfig()

        try:
            gen_config = types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
            )

            if system_instruction:
                gen_config.system_instruction = system_instruction

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=gen_config,
            )

            return response.text or ""

        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming response.

        Args:
            prompt: The user prompt/question
            system_instruction: Optional system instruction
            config: Generation configuration

        Yields:
            Text chunks as they are generated
        """
        if config is None:
            config = GenerationConfig()

        try:
            gen_config = types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_output_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
            )

            if system_instruction:
                gen_config.system_instruction = system_instruction

            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=prompt,
                config=gen_config,
            ):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Failed to stream text: {e}")
            raise

    async def generate_with_context(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate answer using RAG context.

        Args:
            question: The user's question
            context_chunks: Retrieved document chunks with text and metadata
            system_instruction: Optional custom system instruction
            config: Generation configuration

        Returns:
            Generated answer
        """
        # Build the RAG prompt
        prompt = self._build_rag_prompt(question, context_chunks)

        if system_instruction is None:
            system_instruction = self._get_default_rag_system_prompt()

        return await self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            config=config,
        )

    async def generate_with_context_stream(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Generate answer using RAG context with streaming.

        Args:
            question: The user's question
            context_chunks: Retrieved document chunks
            system_instruction: Optional custom system instruction
            config: Generation configuration

        Yields:
            Answer text chunks
        """
        prompt = self._build_rag_prompt(question, context_chunks)

        if system_instruction is None:
            system_instruction = self._get_default_rag_system_prompt()

        async for chunk in self.generate_stream(
            prompt=prompt,
            system_instruction=system_instruction,
            config=config,
        ):
            yield chunk

    async def summarize_document(
        self,
        document_text: str,
        title: str,
        max_length: int = 500,
        config: Optional[GenerationConfig] = None,
    ) -> str:
        """
        Generate a summary of a document.

        Args:
            document_text: Full document text or concatenated chunks
            title: Document title
            max_length: Approximate max length of summary in words
            config: Generation configuration

        Returns:
            Document summary
        """
        system_instruction = """You are a helpful assistant that summarizes documents.
Create clear, concise summaries that capture the main points and key information.
Use bullet points for multiple distinct topics."""

        prompt = f"""Please summarize the following document titled "{title}".
Keep the summary under approximately {max_length} words.
Focus on the main topics, key findings, and important details.

DOCUMENT:
{document_text}

SUMMARY:"""

        if config is None:
            config = GenerationConfig(temperature=0.3)

        return await self.generate(
            prompt=prompt,
            system_instruction=system_instruction,
            config=config,
        )

    def _build_rag_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Build RAG prompt with context chunks.

        Args:
            question: User question
            context_chunks: Retrieved chunks with text, score, document info

        Returns:
            Formatted prompt string
        """
        # Format context chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            doc_info = chunk.get("document", {})
            title = doc_info.get("title", "Unknown Document")
            text = chunk.get("text", "")

            context_parts.append(f"[Source {i}: {title}]\n{text}")

        context_text = "\n\n---\n\n".join(context_parts)

        prompt = f"""Based on the following context from the organization's documents, answer the question.
If the answer cannot be found in the context, say so clearly.
Always cite which source(s) you used.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

        return prompt

    def _get_default_rag_system_prompt(self) -> str:
        """Get default system prompt for RAG."""
        return """You are a helpful knowledge assistant for an organization.
Your role is to answer questions using ONLY the information provided in the context.
Do not make up information or use external knowledge.

Guidelines:
- Be accurate and cite your sources (e.g., "According to Source 1...")
- If information is not in the context, clearly state that
- Keep answers concise but complete
- Use bullet points for multiple items
- Maintain a professional, helpful tone"""


# Singleton instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create LLM service singleton."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
