"""pgvector adapter — implements RetrieverPort. Embeds query_text with
OpenAI, then finds the nearest KnowledgeChunk rows by cosine distance.
The only module that knows the embedding model or the pgvector query
shape; everything downstream only ever sees RetrievedChunk.
"""

from __future__ import annotations

from langfuse.openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lead_qualifier.core.logging import get_logger
from lead_qualifier.domain.models import RetrievedChunk
from lead_qualifier.domain.ports import RetrieverPort
from lead_qualifier.infrastructure.db.orm_models import KnowledgeChunk

logger = get_logger(__name__)


class PgVectorRetriever(RetrieverPort):
    """Queries knowledge_chunks (populated by infrastructure/rag/ingest.py)
    for the rows nearest a lead's embedded notes via pgvector's cosine
    distance operator. Built fresh per request in api/deps.py, sharing
    that request's AsyncSession rather than opening its own.
    """

    def __init__(self, session: AsyncSession, api_key: str, embedding_model: str) -> None:
        self._session = session
        self._client = AsyncOpenAI(api_key=api_key)
        self._embedding_model = embedding_model

    async def retrieve(
        self, query_text: str, top_k: int = 5, trace_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Embed query_text and return its top_k nearest knowledge_chunks.

        Short-circuits on blank query_text (see RetrieverPort's docstring)
        rather than spending an embedding call on an empty string, which
        OpenAI's embeddings endpoint rejects outright.
        """
        if not query_text.strip():
            return []

        # trace_id isn't part of the real OpenAI embeddings API — it's a
        # langfuse.openai-specific kwarg, intercepted and stripped by that
        # wrapper before the real request goes out. mypy type-checks
        # against openai's own stubs, which don't know about it.
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=query_text,
            trace_id=trace_id,  # type: ignore[call-arg]
        )
        query_embedding = response.data[0].embedding

        # cosine_distance ranges 0 (identical) to 2 (opposite); converted
        # to the more intuitive similarity (1 identical, -1 opposite)
        # before crossing the port boundary, so callers never see distance.
        distance = KnowledgeChunk.embedding.cosine_distance(query_embedding)
        stmt = select(KnowledgeChunk, distance.label("distance")).order_by(distance).limit(top_k)
        rows = (await self._session.execute(stmt)).all()

        return [
            RetrievedChunk(
                id=str(chunk.id),
                source=chunk.source,
                content=chunk.content,
                similarity=1 - chunk_distance,
            )
            for chunk, chunk_distance in rows
        ]
