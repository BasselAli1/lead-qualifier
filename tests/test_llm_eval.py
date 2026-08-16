"""LLM eval suite — runs real leads through the real OpenAIScorer (real
OpenAI API calls) against a small hand-built "golden dataset", grounded
in the same knowledge/*.md content the LLM is actually prompted with in
production. Excluded from the default test run (see pyproject.toml's
`-m 'not eval'`) since it costs real money and isn't deterministic — run
explicitly with `pytest -m eval`.

Everything except the LLM is faked (FakeCRM/FakeRetriever/FakeLeadRepository/
FakeNotifier from tests/fakes.py) — the rules engine and combine() step
run for real (deterministic), so each case's actual QualificationResult
reflects the real end-to-end decision, not just a raw isolated LLM score.

Assertions are intentionally loose (score ranges / a set of acceptable
tiers, not exact values) since LLM output varies run to run — the point
is to catch clear regressions (a strong lead scoring near-zero, a
disqualifying-signal lead scoring near-100), not to pin an exact number.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from fakes import FakeCRM, FakeLeadRepository, FakeNotifier, FakeRetriever

from lead_qualifier.application.qualify_lead import QualifyLeadUseCase
from lead_qualifier.core.config import settings
from lead_qualifier.domain.models import Lead, RetrievedChunk, Tier
from lead_qualifier.infrastructure.llm.openai_client import OpenAIScorer
from lead_qualifier.services.rules_engine import load_rules_config

pytestmark = pytest.mark.eval


def _chunk(source: str, content: str) -> RetrievedChunk:
    return RetrievedChunk(id=source, source=source, content=content, similarity=0.9)


# Verbatim excerpts from knowledge/*.md, chunked the same way
# infrastructure/rag/ingest.py actually chunks them (one "## " section
# per chunk, header included) — kept in sync by hand since these are
# fixed eval fixtures, not a live retrieval.
_ICP_FIRMOGRAPHICS = _chunk(
    "icp.md",
    "## Firmographics\n\n"
    "We sell best into mid-market and enterprise software/SaaS/technology\n"
    "companies. Fewer than 5 employees is an auto-disqualify (not enough\n"
    "budget authority or process to close). 200+ employees is a strong\n"
    "positive signal — these companies have dedicated budget for tooling in\n"
    "our category and a real evaluation process.",
)
_ICP_INDUSTRY = _chunk(
    "icp.md",
    "## Industry fit\n\n"
    "Best fit: Software, SaaS, Technology. We have supporting case studies and\n"
    "proven ROI in these verticals. Companies outside these industries aren't\n"
    "disqualified, but they should be treated as an unproven fit unless the\n"
    "lead's notes describe a specific, well-understood use case.",
)
_ICP_RED_FLAGS = _chunk(
    "icp.md",
    "## Red flags\n\n"
    "Vague or empty notes with no stated problem, budget, or timeline are a\n"
    "weak signal on their own — score conservatively rather than assuming\n"
    "intent that isn't actually there in the lead's notes.",
)
_PLAYBOOK_STRONG = _chunk(
    "sales_playbook.md",
    "## Strong buying signals\n\n"
    "Notes that mention a specific trigger event (new funding round, recent\n"
    "leadership hire, a compliance deadline, an expiring contract with a\n"
    "competitor) indicate active, time-boxed intent rather than casual\n"
    "browsing. A stated timeline (\"need this live by Q3\") or an explicit\n"
    "comparison against named competitors are both strong positive signals —\n"
    "the lead is already evaluating, not just researching.",
)
_PLAYBOOK_WEAK = _chunk(
    "sales_playbook.md",
    "## Weak or neutral signals\n\n"
    "Generic requests (\"interested in learning more\", \"send me pricing\") with\n"
    "no context about company size, team, or problem being solved are weak on\n"
    "their own. Don't penalize these leads outright — the CRM data (company\n"
    "size, industry, title) may still make them a good fit even if the free\n"
    "text notes are thin — but don't inflate the score based on enthusiasm of\n"
    "tone alone.",
)
_PLAYBOOK_DISQUALIFY = _chunk(
    "sales_playbook.md",
    "## Disqualifying signals\n\n"
    "Notes indicating the lead is a student, is doing academic research, is a\n"
    "competitor doing research, or is explicitly job-hunting/recruiting\n"
    "(rather than evaluating the product) should score very low regardless of\n"
    "firmographic fit.",
)
_DEAL_VP_CHAMPION = _chunk(
    "past_deals.md",
    "## Closed-won: mid-market SaaS, VP-level champion\n\n"
    "A VP of Sales at a 250-person SaaS company reached out after a\n"
    "competitor's contract renewal fell through. Budget was pre-approved\n"
    "($40k/yr) and the timeline was driven by the competitor contract's\n"
    "expiration date. Closed in three weeks — fast cycles are common when the\n"
    "lead is already actively replacing an incumbent rather than evaluating\n"
    "from scratch.",
)
_DEAL_BOTTOM_UP = _chunk(
    "past_deals.md",
    "## Closed-won: technology company, bottom-up then exec sponsor\n\n"
    "A Director of Engineering at a 500-person technology company first\n"
    "inquired with a vague \"just exploring\" note. Follow-up revealed a real\n"
    "project (migrating an internal tool) with a VP sponsor who joined the\n"
    "second call. Lesson: thin initial notes don't always mean low intent —\n"
    "firmographic fit (title, company size, industry) can outweigh weak notes\n"
    "at first contact.",
)


@dataclass
class EvalCase:
    name: str
    lead: Lead
    context: list[RetrievedChunk]
    acceptable_tiers: set[Tier]
    llm_score_min: float = 0.0
    llm_score_max: float = 100.0
    note: str = ""


def _make_lead(**overrides: object) -> Lead:
    defaults: dict[str, object] = {
        "external_id": "eval-lead",
        "source": "hubspot",
        "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return Lead(**defaults)  # type: ignore[arg-type]


# Each case's acceptable_tiers/llm_score bounds are derived from the real
# combine() formula (rule_weight=0.4, llm_weight=0.6, hot>=75, warm>=45 —
# see config/rules.yaml) given that case's *fixed* rule_score, not guessed
# — e.g. case 3's rule_score is 100, so even llm_score=0 floors final_score
# at 40 (Cold), and llm_score_max=35 caps it at 61 (at most Warm; Hot is
# arithmetically impossible here).
GOLDEN_DATASET = [
    EvalCase(
        name="strong_hot_champion",
        lead=_make_lead(
            company_name="Acme SaaS Co",
            company_size=250,
            industry="Software",
            job_title="VP of Sales",
            budget=40000.0,
            country="US",
            notes=(
                "Reaching out after our current vendor's contract renewal fell "
                "through. Budget is pre-approved and we need this live before "
                "the contract expires."
            ),
        ),
        context=[_ICP_FIRMOGRAPHICS, _PLAYBOOK_STRONG, _DEAL_VP_CHAMPION],
        acceptable_tiers={Tier.HOT, Tier.WARM},
        llm_score_min=55,
        note="Near-identical to the documented VP-champion closed-won deal; shouldn't score low.",
    ),
    EvalCase(
        name="thin_notes_good_firmographic_fit",
        lead=_make_lead(
            company_name="BigTech Inc",
            company_size=500,
            industry="Technology",
            job_title="Director of Engineering",
            country="US",
            notes="Just exploring options.",
        ),
        context=[_ICP_FIRMOGRAPHICS, _PLAYBOOK_WEAK, _DEAL_BOTTOM_UP],
        acceptable_tiers={Tier.HOT, Tier.WARM},
        llm_score_min=35,
        note=(
            "Thin notes shouldn't tank the score when firmographics are "
            "strong (playbook says so explicitly)."
        ),
    ),
    EvalCase(
        name="disqualifying_signal_despite_good_title",
        lead=_make_lead(
            company_name="Acme SaaS Co",
            company_size=250,
            industry="Software",
            job_title="VP of Engineering",
            budget=50000.0,
            country="US",
            notes=(
                "I'm a graduate student researching B2B SaaS pricing models "
                "for my thesis and would love to see how your product works."
            ),
        ),
        context=[_ICP_FIRMOGRAPHICS, _PLAYBOOK_DISQUALIFY],
        acceptable_tiers={Tier.COLD, Tier.WARM},
        llm_score_max=35,
        note=(
            "Academic research is an explicit disqualifying signal, "
            "regardless of the good title/firmographics."
        ),
    ),
    EvalCase(
        name="generic_pricing_inquiry_no_context",
        lead=_make_lead(
            company_name="Some Co",
            company_size=50,
            industry="Retail",
            job_title="Manager",
            country="US",
            notes="Can you send me pricing?",
        ),
        context=[_ICP_RED_FLAGS, _PLAYBOOK_WEAK],
        acceptable_tiers={Tier.COLD},
        llm_score_max=60,
        note=(
            "Generic, context-free request — playbook says not to inflate "
            "score on tone/enthusiasm alone."
        ),
    ),
    EvalCase(
        name="wrong_industry_but_clear_use_case",
        lead=_make_lead(
            company_name="Acme Manufacturing",
            company_size=300,
            industry="Manufacturing",
            job_title="Head of Operations",
            budget=25000.0,
            country="US",
            notes=(
                "We need to replace our manual inventory-tracking spreadsheets "
                "with an API-driven system before our Q4 audit; evaluating "
                "three vendors now."
            ),
        ),
        context=[_ICP_INDUSTRY, _PLAYBOOK_STRONG],
        acceptable_tiers={Tier.HOT, Tier.WARM},
        llm_score_min=40,
        note=(
            "icp.md says a specific, well-understood use case should "
            "offset a non-target industry."
        ),
    ),
]


@pytest.fixture
def rules_config():
    """Loads the real config/rules.yaml, unlike conftest.py's fixed
    rules_config fixture — evals are meant to reflect actual deployed
    decision-making, not be insulated from it."""
    return load_rules_config(settings.RULES_PATH)


@pytest.mark.parametrize("case", GOLDEN_DATASET, ids=[c.name for c in GOLDEN_DATASET])
async def test_golden_dataset(case: EvalCase, rules_config) -> None:
    crm = FakeCRM(case.lead)
    retriever = FakeRetriever(chunks=case.context)
    llm = OpenAIScorer(api_key=settings.OPENAI_API_KEY, model=settings.OPENAI_MODEL)
    repository = FakeLeadRepository()
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

    result = await use_case.qualify_from_webhook_event({"objectId": case.lead.external_id})

    assert result is not None
    assert result.llm_result is not None, "expected the LLM to actually run for this case"

    llm_score = result.llm_result.llm_score
    failure_context = (
        f"[{case.name}] llm_score={llm_score}, tier={result.tier}, "
        f"final_score={result.final_score}, rationale={result.llm_result.rationale!r}. "
        f"{case.note}"
    )
    assert case.llm_score_min <= llm_score <= case.llm_score_max, failure_context
    assert result.tier in case.acceptable_tiers, failure_context
