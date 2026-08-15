"""Hand-written fakes for every port in domain/ports.py — not a mocking
library, so tests exercise real (if trivial) behavior: FakeLeadRepository
actually tracks what's been saved rather than just recording that save()
was called. application/qualify_lead.py's own docstring calls this out
as the intended way to unit-test the orchestrator without a real
database, LLM, or HTTP call.
"""

from __future__ import annotations

from datetime import datetime

from lead_qualifier.domain.models import Lead, LLMResult, QualificationResult, RetrievedChunk, Tier
from lead_qualifier.domain.ports import (
    CRMPort,
    LeadRepositoryPort,
    LLMPort,
    NotifierPort,
    RetrieverPort,
)


class FakeCRM(CRMPort):
    """Always returns the same pre-built Lead regardless of the webhook
    event passed in — parsing a raw event into a Lead is HubSpotCRM's own
    concern (see infrastructure/crm/hubspot.py), not something
    qualify_lead.py's orchestration logic needs to exercise."""

    def __init__(self, lead: Lead) -> None:
        self._lead = lead
        self.pushed: list[tuple[str, Tier, float, str]] = []

    def verify_signature(
        self, method: str, uri: str, body: str, timestamp: str, signature: str
    ) -> bool:
        return True

    async def lead_from_webhook_event(self, event: dict) -> Lead:
        return self._lead

    async def push_qualification(
        self, external_id: str, tier: Tier, final_score: float, rationale: str
    ) -> None:
        self.pushed.append((external_id, tier, final_score, rationale))


class FakeRetriever(RetrieverPort):
    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or []
        self.queries: list[str] = []

    async def retrieve(
        self, query_text: str, top_k: int = 5, trace_id: str | None = None
    ) -> list[RetrievedChunk]:
        self.queries.append(query_text)
        return self._chunks


class FakeLLM(LLMPort):
    def __init__(self, result: LLMResult) -> None:
        self._result = result
        self.trace_ids: list[str | None] = []

    async def score(
        self, lead_summary: str, context_chunks: list[RetrievedChunk], trace_id: str | None = None
    ) -> LLMResult:
        self.trace_ids.append(trace_id)
        return self._result


class FakeNotifier(NotifierPort):
    def __init__(self) -> None:
        self.notified: list[QualificationResult] = []

    async def notify_hot_lead(self, result: QualificationResult) -> None:
        self.notified.append(result)


class FakeLeadRepository(LeadRepositoryPort):
    def __init__(self, duplicate_of: tuple[str, datetime] | None = None) -> None:
        self._duplicate_of = duplicate_of
        self.saved: list[QualificationResult] = []

    async def is_duplicate(self, external_id: str, updated_at: datetime) -> bool:
        return (external_id, updated_at) == self._duplicate_of

    async def save(self, result: QualificationResult) -> None:
        self.saved.append(result)

    async def get(self, external_id: str) -> QualificationResult | None:
        for result in reversed(self.saved):
            if result.lead.external_id == external_id:
                return result
        return None

    async def list_hot_leads(self, since: datetime | None = None) -> list[QualificationResult]:
        return [r for r in self.saved if r.tier == Tier.HOT]
