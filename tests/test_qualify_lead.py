"""Tests for application/qualify_lead.py's orchestration, using the fakes
in tests/fakes.py for all five ports — no real database, LLM, or HTTP
call, per that module's own docstring.
"""

from __future__ import annotations

import uuid

from fakes import FakeCRM, FakeLeadRepository, FakeLLM, FakeNotifier, FakeRetriever

from lead_qualifier.application.qualify_lead import QualifyLeadUseCase
from lead_qualifier.domain.models import LLMResult, Tier
from lead_qualifier.services.rules_engine import RulesConfig


def _llm_result(score: float = 90.0) -> LLMResult:
    return LLMResult(
        llm_score=score,
        rationale="Strong intent.",
        model="gpt-test",
        input_tokens=1,
        output_tokens=1,
        latency_ms=1.0,
    )


def _build(lead, rules_config: RulesConfig, *, repository=None, llm_result=None):
    crm = FakeCRM(lead)
    retriever = FakeRetriever()
    llm = FakeLLM(llm_result or _llm_result())
    repository = repository or FakeLeadRepository()
    notifier = FakeNotifier()
    use_case = QualifyLeadUseCase(
        crm=crm,
        retriever=retriever,
        llm=llm,
        repository=repository,
        notifier=notifier,
        rules_config=rules_config,
        rag_top_k=5,
    )
    return use_case, crm, retriever, llm, repository, notifier


async def test_duplicate_lead_short_circuits_before_any_scoring(rules_config, make_lead):
    lead = make_lead()
    repository = FakeLeadRepository(duplicate_of=(lead.external_id, lead.updated_at))
    use_case, crm, retriever, llm, repository, notifier = _build(
        lead, rules_config, repository=repository
    )

    result = await use_case.qualify_from_webhook_event({"objectId": lead.external_id})

    assert result is None
    assert repository.saved == []
    assert retriever.queries == []
    assert crm.pushed == []


async def test_hard_disqualified_lead_skips_retriever_and_llm(rules_config, make_lead):
    lead = make_lead(country="DE")  # fails the served-geography hard filter
    use_case, crm, retriever, llm, repository, notifier = _build(lead, rules_config)

    result = await use_case.qualify_from_webhook_event({"objectId": lead.external_id})

    assert result is not None
    assert result.tier == Tier.DISQUALIFIED
    assert result.llm_result is None
    assert retriever.queries == []
    assert llm.trace_ids == []
    assert repository.saved == [result]
    assert crm.pushed == [(lead.external_id, Tier.DISQUALIFIED, 0.0, result.rationale)]
    assert notifier.notified == []


async def test_hot_lead_triggers_notification(rules_config, make_lead):
    lead = make_lead()  # matches every scoring rule -> rule_score 100
    use_case, crm, retriever, llm, repository, notifier = _build(
        lead, rules_config, llm_result=_llm_result(100)
    )

    result = await use_case.qualify_from_webhook_event({"objectId": lead.external_id})

    assert result.tier == Tier.HOT
    assert notifier.notified == [result]


async def test_non_hot_lead_does_not_trigger_notification(rules_config, make_lead):
    lead = make_lead()
    # final = rule_score(100)*0.4 + llm_score(10)*0.6 = 46 -> warm, not hot
    use_case, crm, retriever, llm, repository, notifier = _build(
        lead, rules_config, llm_result=_llm_result(10)
    )

    result = await use_case.qualify_from_webhook_event({"objectId": lead.external_id})

    assert result.tier == Tier.WARM
    assert notifier.notified == []


async def test_retriever_queried_with_stripped_notes(rules_config, make_lead):
    lead = make_lead(notes="  Looking for a replacement.  ")
    use_case, crm, retriever, llm, repository, notifier = _build(lead, rules_config)

    await use_case.qualify_from_webhook_event({"objectId": lead.external_id})

    assert retriever.queries == ["Looking for a replacement."]


async def test_trace_id_shared_across_retriever_and_llm(rules_config, make_lead):
    lead = make_lead()
    use_case, crm, retriever, llm, repository, notifier = _build(lead, rules_config)

    await use_case.qualify_from_webhook_event({"objectId": lead.external_id})

    expected_trace_id = uuid.uuid5(
        uuid.NAMESPACE_URL, f"{lead.external_id}:{lead.updated_at.isoformat()}"
    ).hex
    assert llm.trace_ids == [expected_trace_id]
    # Langfuse requires exactly this shape (int(trace_id, 16) internally) —
    # this is the actual property that broke in production before uuid5
    # replaced the old "external_id:isoformat" trace_id.
    assert len(expected_trace_id) == 32
    int(expected_trace_id, 16)


async def test_result_is_saved_and_pushed_to_crm(rules_config, make_lead):
    lead = make_lead()
    use_case, crm, retriever, llm, repository, notifier = _build(lead, rules_config)

    result = await use_case.qualify_from_webhook_event({"objectId": lead.external_id})

    assert repository.saved == [result]
    assert crm.pushed == [(lead.external_id, result.tier, result.final_score, result.rationale)]
