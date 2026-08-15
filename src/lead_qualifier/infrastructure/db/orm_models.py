"""SQLAlchemy ORM models — the Postgres-specific mirror of the domain
models in domain/models.py. Deliberately kept separate: domain/models.py
is what the rest of the app works with, these are just how that data gets
stored. lead_repository.py is the only place that converts between them.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# text-embedding-3-small is 1536-dimensional by default. If
# OPENAI_EMBEDDING_MODEL ever changes to a model (or a custom `dimensions`
# value) with a different output size, both this constant and the Alembic
# migration that creates the embedding column need to change together, or
# inserts will fail with a dimension mismatch.
EMBEDDING_DIM = 1536


class Base(DeclarativeBase):
    """Shared base class SQLAlchemy uses to discover all ORM models and
    generate migrations against — every table class below inherits from
    this."""


class LeadRecord(Base):
    """One qualification result for one version of a lead.

    The unique constraint on (external_id, lead_updated_at) is the actual
    idempotency mechanism: a duplicate webhook delivery for the same lead
    version will violate this constraint on insert, which
    lead_repository.py treats as "already processed" rather than an error.
    """

    __tablename__ = "lead_records"
    __table_args__ = (UniqueConstraint("external_id", "lead_updated_at", name="uq_lead_version"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    external_id: Mapped[str] = mapped_column(String, index=True)
    lead_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    tier: Mapped[str] = mapped_column(String)
    final_score: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(String)

    # Stored as JSON rather than normalized into columns: this is an audit
    # trail of what was actually evaluated, not a table we need to query
    # by individual lead fields. Keeping it as one JSON blob per result
    # also means adding a field to Lead/RuleResult/LLMResult later doesn't
    # require a migration here.
    lead_snapshot: Mapped[dict] = mapped_column(JSON)
    rule_result: Mapped[dict] = mapped_column(JSON)
    llm_result: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    pushed_to_hubspot: Mapped[bool] = mapped_column(default=False)
    notified_slack: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class KnowledgeChunk(Base):
    """One embedded chunk of RAG source material — ICP definition, sales
    playbook, or past deal notes from knowledge/*.md. Populated by
    infrastructure/rag/ingest.py, queried by infrastructure/rag/retriever.py.
    """

    __tablename__ = "knowledge_chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    source: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
