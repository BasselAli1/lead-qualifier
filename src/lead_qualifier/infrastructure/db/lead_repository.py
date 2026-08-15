"""Postgres implementation of LeadRepositoryPort. The only place that
converts between domain models (QualificationResult) and the ORM row
shape (LeadRecord) — see orm_models.py for why they're kept separate.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from lead_qualifier.core.logging import get_logger
from lead_qualifier.domain.models import Lead, LLMResult, QualificationResult, RuleResult, Tier
from lead_qualifier.domain.ports import LeadRepositoryPort
from lead_qualifier.infrastructure.db.orm_models import LeadRecord

logger = get_logger(__name__)


def _to_domain(row: LeadRecord) -> QualificationResult:
    """Rebuild a QualificationResult from one LeadRecord row.

    Pydantic's model_validate() parses the stored JSON dicts (and ISO
    datetime strings within them) back into typed Lead/RuleResult/
    LLMResult objects — no manual field-by-field conversion needed.
    """
    return QualificationResult(
        lead=Lead.model_validate(row.lead_snapshot),
        tier=Tier(row.tier),
        final_score=row.final_score,
        rule_result=RuleResult.model_validate(row.rule_result),
        llm_result=LLMResult.model_validate(row.llm_result) if row.llm_result else None,
        rationale=row.rationale,
        created_at=row.created_at,
    )


class PostgresLeadRepository(LeadRepositoryPort):
    """Postgres-backed implementation of LeadRepositoryPort, used in
    production via api/deps.py. Tests use a fake/in-memory implementation
    of the same port instead, so they don't need a real database."""

    def __init__(self, session: AsyncSession) -> None:
        """Take an already-open AsyncSession rather than opening its own —
        this keeps one DB transaction per request, managed by whoever
        constructs this (api/deps.py), instead of this class owning
        connection lifecycle."""
        self._session = session

    async def is_duplicate(self, external_id: str, updated_at: datetime) -> bool:
        """Check whether this exact lead version has already been saved.

        A plain SELECT, not the actual enforcement mechanism — the
        UniqueConstraint on LeadRecord is what's actually relied on to be
        correct under concurrent requests. This method exists so the
        common case (no race) can skip re-processing early, before doing
        any RAG/LLM work, rather than only catching the race afterward.
        """
        stmt = select(LeadRecord.id).where(
            LeadRecord.external_id == external_id,
            LeadRecord.lead_updated_at == updated_at,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def save(self, result: QualificationResult) -> None:
        """Persist a qualification result.

        If two requests for the same lead version race past the
        is_duplicate() check above, the database's UniqueConstraint
        rejects the second insert with an IntegrityError — caught here
        and treated as "already saved," not as a failure.
        """
        row = LeadRecord(
            external_id=result.lead.external_id,
            lead_updated_at=result.lead.updated_at,
            tier=result.tier.value,
            final_score=result.final_score,
            rationale=result.rationale,
            lead_snapshot=result.lead.model_dump(mode="json"),
            rule_result=result.rule_result.model_dump(mode="json"),
            llm_result=result.llm_result.model_dump(mode="json") if result.llm_result else None,
            created_at=result.created_at,
        )
        self._session.add(row)
        try:
            await self._session.commit()
        except IntegrityError:
            await self._session.rollback()
            logger.info(
                "Duplicate lead version, skipping save",
                extra={"external_id": result.lead.external_id},
            )

    async def get(self, external_id: str) -> QualificationResult | None:
        """Return the most recent qualification result for a lead.

        A lead can have multiple LeadRecord rows over time (one per
        version that was processed) — this returns the latest one, which
        is what the manual API and MCP get_qualification tool care about.
        """
        stmt = (
            select(LeadRecord)
            .where(LeadRecord.external_id == external_id)
            .order_by(LeadRecord.created_at.desc())
            .limit(1)
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return _to_domain(row) if row else None

    async def list_hot_leads(self, since: datetime | None = None) -> list[QualificationResult]:
        """Return Hot-tier leads, newest first, optionally only those
        created at/after `since`. Backs the MCP list_hot_leads tool."""
        stmt = select(LeadRecord).where(LeadRecord.tier == Tier.HOT.value)
        if since is not None:
            stmt = stmt.where(LeadRecord.created_at >= since)
        stmt = stmt.order_by(LeadRecord.created_at.desc())
        rows = (await self._session.execute(stmt)).scalars().all()
        return [_to_domain(row) for row in rows]
