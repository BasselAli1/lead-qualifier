"""Adapter tests for infrastructure/rag/retriever.py's PgVectorRetriever.

FakeSession stands in for AsyncSession — no real Postgres is available in
this environment (see pyproject.toml's addopts comment on
pytest-postgresql), and pgvector's Vector column type has no SQLite
equivalent to fall back on anyway. This still exercises everything that
doesn't require an actual database round-trip: the blank-query
short-circuit, the embed-then-query sequencing, and the
distance-to-similarity conversion.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lead_qualifier.infrastructure.db.orm_models import KnowledgeChunk
from lead_qualifier.infrastructure.rag.retriever import PgVectorRetriever


class FakeSession:
    """Records the statement it was asked to execute and returns
    pre-baked (KnowledgeChunk, distance) rows."""

    def __init__(self, rows: list[tuple[KnowledgeChunk, float]]) -> None:
        self._rows = rows
        self.executed_statements: list = []

    async def execute(self, stmt):
        self.executed_statements.append(stmt)
        return SimpleNamespace(all=lambda: self._rows)


def _fake_embedding_client(embedding: list[float]) -> AsyncMock:
    client = AsyncMock()
    client.embeddings.create = AsyncMock(
        return_value=SimpleNamespace(data=[SimpleNamespace(embedding=embedding)])
    )
    return client


def _make_retriever(
    rows: list[tuple[KnowledgeChunk, float]],
) -> tuple[PgVectorRetriever, FakeSession]:
    session = FakeSession(rows)
    retriever = PgVectorRetriever(session, api_key="test", embedding_model="text-embedding-3-small")
    retriever._client = _fake_embedding_client([0.1] * 5)
    return retriever, session


async def test_retrieve_short_circuits_on_blank_query_text():
    retriever, session = _make_retriever(rows=[])

    result = await retriever.retrieve("   ")

    assert result == []
    assert session.executed_statements == []  # never even queried the DB
    retriever._client.embeddings.create.assert_not_called()


async def test_retrieve_maps_rows_to_retrieved_chunks_with_similarity():
    chunk = KnowledgeChunk(
        id=uuid.uuid4(), source="icp.md", content="Target: SaaS.", embedding=[0.1] * 5
    )
    retriever, session = _make_retriever(rows=[(chunk, 0.2)])  # cosine_distance 0.2

    result = await retriever.retrieve("Looking for a SaaS vendor.")

    assert len(result) == 1
    assert result[0].id == str(chunk.id)
    assert result[0].source == "icp.md"
    assert result[0].content == "Target: SaaS."
    assert result[0].similarity == pytest.approx(0.8)  # 1 - 0.2
    assert len(session.executed_statements) == 1


async def test_retrieve_embeds_query_text_and_forwards_trace_id():
    retriever, _ = _make_retriever(rows=[])

    await retriever.retrieve("Looking for a SaaS vendor.", trace_id="lead-1:2026-01-01")

    retriever._client.embeddings.create.assert_awaited_once()
    call_kwargs = retriever._client.embeddings.create.call_args.kwargs
    assert call_kwargs["input"] == "Looking for a SaaS vendor."
    assert call_kwargs["trace_id"] == "lead-1:2026-01-01"


async def test_retrieve_returns_multiple_chunks_in_query_order():
    chunk_a = KnowledgeChunk(id=uuid.uuid4(), source="a.md", content="A", embedding=[0.1] * 5)
    chunk_b = KnowledgeChunk(id=uuid.uuid4(), source="b.md", content="B", embedding=[0.2] * 5)
    retriever, _ = _make_retriever(rows=[(chunk_a, 0.1), (chunk_b, 0.4)])

    result = await retriever.retrieve("query")

    assert [chunk.source for chunk in result] == ["a.md", "b.md"]
