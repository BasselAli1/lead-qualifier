"""Initial schema: lead_records, knowledge_chunks

Revision ID: 72b584e88e4c
Revises:
Create Date: 2026-08-14

"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "72b584e88e4c"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Hardcoded rather than importing infrastructure/db/orm_models.py's
# EMBEDDING_DIM — migrations are a frozen historical record of schema
# changes, not something that should shift if the model file changes
# later. See that constant's own comment: this value and
# OPENAI_EMBEDDING_MODEL (core/config.py) must be changed together, via a
# new migration, not by editing this one.
_EMBEDDING_DIM = 1536


def upgrade() -> None:
    # knowledge_chunks.embedding (pgvector's Vector column type) requires
    # this extension; nothing else in this migration does.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "lead_records",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("external_id", sa.String(), nullable=False),
        sa.Column("lead_updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("tier", sa.String(), nullable=False),
        sa.Column("final_score", sa.Float(), nullable=False),
        sa.Column("rationale", sa.String(), nullable=False),
        sa.Column("lead_snapshot", sa.JSON(), nullable=False),
        sa.Column("rule_result", sa.JSON(), nullable=False),
        sa.Column("llm_result", sa.JSON(), nullable=True),
        sa.Column("pushed_to_hubspot", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("notified_slack", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        # Matches orm_models.py's LeadRecord.__table_args__ — the actual
        # idempotency mechanism a duplicate webhook delivery relies on.
        sa.UniqueConstraint("external_id", "lead_updated_at", name="uq_lead_version"),
    )
    op.create_index("ix_lead_records_external_id", "lead_records", ["external_id"])

    op.create_table(
        "knowledge_chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("content", sa.String(), nullable=False),
        sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("knowledge_chunks")
    op.drop_index("ix_lead_records_external_id", table_name="lead_records")
    op.drop_table("lead_records")
    # The vector extension is deliberately left installed on downgrade —
    # dropping it would break any other table/database sharing it, and
    # CREATE EXTENSION IF NOT EXISTS on a future upgrade is a no-op either
    # way if it's still present.
