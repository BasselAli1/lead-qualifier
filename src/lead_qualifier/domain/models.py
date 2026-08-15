"""Core domain types. No framework or vendor imports beyond pydantic for validation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class Tier(StrEnum):
    """The four possible outcomes of qualifying a lead.

    String-valued (not a plain IntEnum) so it serializes directly to JSON,
    to the database, and to the HubSpot `lead_qualifier_tier` property
    without any conversion step.
    """

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"
    DISQUALIFIED = "disqualified"


class Lead(BaseModel):
    """CRM-agnostic representation of a lead.

    HubSpot's payload shape is mapped into this in
    infrastructure/crm/hubspot.py and nowhere else — every other module
    (rules engine, retriever, LLM scorer, repository) only ever sees this
    type, never a raw HubSpot dict.
    """

    external_id: str
    source: str = "hubspot"
    email: str | None = None
    company_name: str | None = None
    company_size: int | None = None
    industry: str | None = None
    job_title: str | None = None
    budget: float | None = None
    country: str | None = None
    notes: str = ""
    # Required (no default): half of the (external_id, updated_at)
    # idempotency key used by the repository to detect duplicate webhook
    # deliveries for the same lead version.
    updated_at: datetime


class RuleResult(BaseModel):
    """Output of services/rules_engine.py evaluating config/rules.yaml
    against one Lead."""

    hard_disqualified: bool
    disqualify_reason: str | None = None
    rule_score: float = Field(ge=0, le=100)
    matched_rules: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """One piece of context pulled from the RAG store (ICP definition,
    sales playbook, or past deal notes) for a given lead."""

    id: str
    source: str
    content: str
    similarity: float


class LLMResult(BaseModel):
    """Output of the OpenAI scoring call, plus everything needed for
    cost/latency observability without a separate lookup in Langfuse."""

    llm_score: float = Field(ge=0, le=100)
    rationale: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)


class QualificationResult(BaseModel):
    """The final record for one qualified lead: what gets persisted to
    Postgres, pushed to HubSpot, and — for Hot leads — sent to Slack."""

    lead: Lead
    tier: Tier
    final_score: float = Field(ge=0, le=100)
    rule_result: RuleResult
    llm_result: LLMResult | None = None
    rationale: str
    created_at: datetime
