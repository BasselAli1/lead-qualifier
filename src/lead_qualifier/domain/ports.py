"""Interfaces for every piece of infrastructure the application layer talks
to: CRM, LLM, retriever, notifier, repository. application/qualify_lead.py
depends only on these abstractions, never on a concrete vendor module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from lead_qualifier.domain.models import Lead, LLMResult, QualificationResult, RetrievedChunk, Tier

# `trace_id` on LLMPort/RetrieverPort is derived from the same
# (external_id, updated_at) pair as the log correlation id (core/logging.py)
# — see application/qualify_lead.py — so a lead's Cloud Logging entries and
# its Langfuse trace can still be cross-referenced, even though the two ids
# aren't literally the same string (Langfuse requires trace_id to be a
# 32-char lowercase-hex value; the correlation id is just the bare
# external_id).


class CRMPort(ABC):
    """Everything the application layer needs from a CRM.

    Implemented by infrastructure/crm/hubspot.py's HubSpotCRM. A second
    CRM would implement this same interface rather than requiring changes
    anywhere else in the app.
    """

    @abstractmethod
    def verify_signature(
        self, method: str, uri: str, body: str, timestamp: str, signature: str
    ) -> bool:
        """Return True if a webhook request is authentically from this CRM.

        Sync, not async — this is pure HMAC computation, no I/O. Called
        before any other processing so an unverified request never reaches
        the qualification pipeline.
        """
        ...

    @abstractmethod
    async def lead_from_webhook_event(self, event: dict) -> Lead:
        """Turn one raw webhook event into a normalized Lead.

        This is the only place a CRM's specific payload shape is allowed
        to leak into the app — everything downstream only ever sees Lead.
        """
        ...

    @abstractmethod
    async def push_qualification(
        self, external_id: str, tier: Tier, final_score: float, rationale: str
    ) -> None:
        """Write a qualification result back onto the CRM record so sales
        reps see the tier/score/rationale without leaving the CRM."""
        ...


class LLMPort(ABC):
    """Scores a lead's fit/intent using an LLM, grounded in retrieved
    context. Implemented by infrastructure/llm/openai_client.py's
    OpenAIScorer."""

    @abstractmethod
    async def score(
        self, lead_summary: str, context_chunks: list[RetrievedChunk], trace_id: str | None = None
    ) -> LLMResult:
        """Return a 0-100 score and rationale for one lead.

        `context_chunks` is whatever the retriever found relevant (ICP,
        playbook, past deals) — the implementation is expected to include
        it in the prompt so scoring is grounded rather than generic.
        `trace_id` links this call to the same Langfuse trace and Cloud
        Logging correlation id as the rest of that lead's pipeline run.
        """
        ...


class RetrieverPort(ABC):
    """Finds knowledge-base context relevant to a lead. Implemented by
    infrastructure/rag/retriever.py's PgVectorRetriever."""

    @abstractmethod
    async def retrieve(
        self, query_text: str, top_k: int = 5, trace_id: str | None = None
    ) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks for query_text.

        An empty/near-empty query_text (e.g. a lead with no free-text
        notes) is expected to short-circuit to an empty list rather than
        embedding an empty string.
        """
        ...


class NotifierPort(ABC):
    """Alerts a human about a qualification result. Implemented by
    infrastructure/notifications/slack_notifier.py's SlackNotifier."""

    @abstractmethod
    async def notify_hot_lead(self, result: QualificationResult) -> None:
        """Notify about a Hot-tier lead. Only ever called for Hot leads —
        the tier check happens in application/qualify_lead.py, not here."""
        ...


class LeadRepositoryPort(ABC):
    """Persists qualification results and answers the queries the API and
    MCP server need. Implemented by
    infrastructure/db/lead_repository.py's PostgresLeadRepository."""

    @abstractmethod
    async def is_duplicate(self, external_id: str, updated_at: datetime) -> bool:
        """Return True if this exact lead version was already processed —
        the idempotency check that guards against webhook retries causing
        double LLM/embedding spend."""
        ...

    @abstractmethod
    async def save(self, result: QualificationResult) -> None:
        """Persist a qualification result. Expected to be safe to call
        even if the same (external_id, updated_at) is saved twice — treat
        it as already-saved rather than raising."""
        ...

    @abstractmethod
    async def get(self, external_id: str) -> QualificationResult | None:
        """Return the most recent qualification result for a lead, or
        None if it's never been qualified."""
        ...

    @abstractmethod
    async def list_hot_leads(self, since: datetime | None = None) -> list[QualificationResult]:
        """Return Hot-tier leads, optionally only those created at/after
        `since`. Backs the MCP list_hot_leads tool."""
        ...
