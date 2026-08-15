"""FastAPI dependency providers. This is the one place concrete
infrastructure adapters get constructed and wired into ports — every
route imports only the `*Dep` aliases below, never a concrete adapter
class directly.

NOTE: infrastructure/{crm,llm,rag,notifications}/* don't exist yet (see
domain/ports.py for the classes they're expected to provide). The
constructor calls below are provisional — update them once those modules
land; the port-typed return values here are what matters and shouldn't
need to change.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from lead_qualifier.application.qualify_lead import QualifyLeadUseCase
from lead_qualifier.core.config import settings
from lead_qualifier.domain.ports import (
    CRMPort,
    LeadRepositoryPort,
    LLMPort,
    NotifierPort,
    RetrieverPort,
)
from lead_qualifier.infrastructure.crm.hubspot import HubSpotCRM
from lead_qualifier.infrastructure.db.lead_repository import PostgresLeadRepository
from lead_qualifier.infrastructure.db.session import get_session
from lead_qualifier.infrastructure.llm.openai_client import OpenAIScorer
from lead_qualifier.infrastructure.notifications.slack_notifier import SlackNotifier
from lead_qualifier.infrastructure.rag.retriever import PgVectorRetriever
from lead_qualifier.services.rules_engine import RulesConfig, load_rules_config

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@lru_cache
def get_rules_config() -> RulesConfig:
    """Parsed once per process and cached — config/rules.yaml is expected
    to change rarely enough that picking up edits via a restart is an
    acceptable tradeoff for not re-parsing YAML on every request."""
    return load_rules_config(settings.RULES_PATH)


RulesConfigDep = Annotated[RulesConfig, Depends(get_rules_config)]


def get_crm() -> CRMPort:
    return HubSpotCRM(
        api_key=settings.HUBSPOT_API_KEY, webhook_secret=settings.HUBSPOT_WEBHOOK_SECRET
    )


CRMDep = Annotated[CRMPort, Depends(get_crm)]


def get_retriever(session: SessionDep) -> RetrieverPort:
    """Needs a session to query knowledge_chunks (see orm_models.py) and
    an API key to embed the query text before doing the pgvector lookup."""
    return PgVectorRetriever(
        session, api_key=settings.OPENAI_API_KEY, embedding_model=settings.OPENAI_EMBEDDING_MODEL
    )


RetrieverDep = Annotated[RetrieverPort, Depends(get_retriever)]


def get_llm() -> LLMPort:
    return OpenAIScorer(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)


LLMDep = Annotated[LLMPort, Depends(get_llm)]


def get_notifier() -> NotifierPort:
    return SlackNotifier(webhook_url=settings.SLACK_WEBHOOK_URL)


NotifierDep = Annotated[NotifierPort, Depends(get_notifier)]


def get_lead_repository(session: SessionDep) -> LeadRepositoryPort:
    return PostgresLeadRepository(session)


LeadRepositoryDep = Annotated[LeadRepositoryPort, Depends(get_lead_repository)]


def get_qualify_lead_use_case(
    crm: CRMDep,
    retriever: RetrieverDep,
    llm: LLMDep,
    repository: LeadRepositoryDep,
    notifier: NotifierDep,
    rules_config: RulesConfigDep,
) -> QualifyLeadUseCase:
    """Built fresh per request (all its port dependencies are themselves
    per-request, tied to this request's DB session) — never cached."""
    return QualifyLeadUseCase(
        crm=crm,
        retriever=retriever,
        llm=llm,
        repository=repository,
        notifier=notifier,
        rules_config=rules_config,
        rag_top_k=settings.RAG_TOP_K,
    )


QualifyLeadUseCaseDep = Annotated[QualifyLeadUseCase, Depends(get_qualify_lead_use_case)]
