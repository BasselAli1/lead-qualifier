"""Orchestrates one lead through the full qualification pipeline:
normalize -> idempotency check -> rules -> (RAG + LLM, unless hard-
disqualified) -> combine -> persist -> push back to CRM -> notify if Hot.

This is the only module that calls more than one port in sequence — every
adapter only knows its own port; this is where they're composed. Depends
only on domain/ports.py and services/, never on a concrete infrastructure
adapter, so it can be unit-tested with fakes for all five ports and no
real database, LLM, or HTTP call.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from lead_qualifier.core.logging import get_logger
from lead_qualifier.domain.models import Lead, QualificationResult, Tier
from lead_qualifier.domain.ports import (
    CRMPort,
    LeadRepositoryPort,
    LLMPort,
    NotifierPort,
    RetrieverPort,
)
from lead_qualifier.services import rules_engine, scoring
from lead_qualifier.services.rules_engine import RulesConfig

logger = get_logger(__name__)


class QualifyLeadUseCase:
    """The single orchestrator in the app. Constructed once per request in
    api/deps.py with concrete adapters; every dependency here is a port,
    so tests construct this with fakes and never touch a real vendor.
    """

    def __init__(
        self,
        crm: CRMPort,
        retriever: RetrieverPort,
        llm: LLMPort,
        repository: LeadRepositoryPort,
        notifier: NotifierPort,
        rules_config: RulesConfig,
        rag_top_k: int = 5,
    ) -> None:
        self._crm = crm
        self._retriever = retriever
        self._llm = llm
        self._repository = repository
        self._notifier = notifier
        self._rules_config = rules_config
        self._rag_top_k = rag_top_k

    async def qualify_from_webhook_event(self, event: dict) -> QualificationResult | None:
        """Run one CRM webhook event through the full pipeline.

        Returns None if this exact lead version was already processed
        (a duplicate webhook delivery) — the caller (api/routes/webhooks.py)
        is expected to still respond 200 in that case, since the sender
        should not be told to retry a delivery we've already handled.

        Signature verification is assumed to have already happened
        (CRMPort.verify_signature is sync HMAC-only and has no need for
        anything built here) — that's the caller's job, before this is
        ever invoked.
        """
        lead = await self._crm.lead_from_webhook_event(event)

        if await self._repository.is_duplicate(lead.external_id, lead.updated_at):
            logger.info(
                "Duplicate lead version, skipping", extra={"external_id": lead.external_id}
            )
            return None

        # Passed through as the Langfuse trace id (see domain/ports.py) so
        # a lead's LLM trace can be found from its Cloud Logging
        # correlation id (webhooks.py sets that from the same
        # external_id) and vice versa. Langfuse requires a 32-char
        # lowercase-hex trace id — passing the raw "external_id:iso8601"
        # string crashes inside its client (int(trace_id, 16)) — so this
        # deterministically hashes that identifying pair into a valid
        # UUID5 hex instead of using it directly.
        trace_id = uuid.uuid5(
            uuid.NAMESPACE_URL, f"{lead.external_id}:{lead.updated_at.isoformat()}"
        ).hex

        rule_result = rules_engine.evaluate(lead, self._rules_config)

        llm_result = None
        if not rule_result.hard_disqualified:
            chunks = await self._retriever.retrieve(
                lead.notes.strip(), top_k=self._rag_top_k, trace_id=trace_id
            )
            llm_result = await self._llm.score(
                self._build_lead_summary(lead), chunks, trace_id=trace_id
            )

        final_score, tier, rationale = scoring.combine(rule_result, llm_result, self._rules_config)

        result = QualificationResult(
            lead=lead,
            tier=tier,
            final_score=final_score,
            rule_result=rule_result,
            llm_result=llm_result,
            rationale=rationale,
            created_at=datetime.now(UTC),
        )

        await self._repository.save(result)
        await self._crm.push_qualification(lead.external_id, tier, final_score, rationale)

        if tier == Tier.HOT:
            await self._notifier.notify_hot_lead(result)

        return result

    @staticmethod
    def _build_lead_summary(lead: Lead) -> str:
        """Human-readable summary of the lead for the LLM prompt.

        Unlike the retriever query (just `notes` — see RetrieverPort's
        docstring on why an empty one should short-circuit), the LLM
        benefits from every structured field even when notes is empty, so
        this includes all of them.
        """
        lines = [
            f"Company: {lead.company_name or 'unknown'}",
            f"Industry: {lead.industry or 'unknown'}",
            f"Company size: {lead.company_size or 'unknown'}",
            f"Job title: {lead.job_title or 'unknown'}",
            f"Budget: {lead.budget or 'unknown'}",
            f"Country: {lead.country or 'unknown'}",
        ]
        if lead.notes:
            lines.append(f"Notes: {lead.notes}")
        return "\n".join(lines)
