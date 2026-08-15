"""Tests for infrastructure/db/lead_repository.py's pure conversion logic.

The rest of PostgresLeadRepository (is_duplicate/save/get/list_hot_leads)
needs a real Postgres connection to test meaningfully — this environment
has no libpq installed (see pyproject.toml's addopts comment disabling
the pytest-postgresql plugin), so those are left for a real
DB-integration pass once that's available.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from lead_qualifier.domain.models import Tier
from lead_qualifier.infrastructure.db.lead_repository import _to_domain
from lead_qualifier.infrastructure.db.orm_models import LeadRecord


def _make_row(**overrides) -> LeadRecord:
    defaults: dict = {
        "id": uuid.uuid4(),
        "external_id": "42",
        "lead_updated_at": datetime(2026, 1, 1, tzinfo=UTC),
        "tier": "hot",
        "final_score": 88.0,
        "rationale": "Strong fit.",
        "lead_snapshot": {
            "external_id": "42",
            "source": "hubspot",
            "notes": "",
            "updated_at": "2026-01-01T00:00:00Z",
        },
        "rule_result": {"hard_disqualified": False, "rule_score": 80.0, "matched_rules": []},
        "llm_result": None,
        "created_at": datetime(2026, 1, 2, tzinfo=UTC),
    }
    defaults.update(overrides)
    return LeadRecord(**defaults)


def test_to_domain_round_trips_a_saved_result():
    row = _make_row()

    result = _to_domain(row)

    assert result.lead.external_id == "42"
    assert result.tier == Tier.HOT
    assert result.final_score == 88.0
    assert result.rationale == "Strong fit."
    assert result.llm_result is None


def test_to_domain_includes_llm_result_when_present():
    row = _make_row(
        tier="warm",
        llm_result={
            "llm_score": 70.0,
            "rationale": "Some intent.",
            "model": "gpt-test",
            "input_tokens": 1,
            "output_tokens": 1,
            "latency_ms": 1.0,
            "retrieved_chunks": [],
        },
    )

    result = _to_domain(row)

    assert result.tier == Tier.WARM
    assert result.llm_result is not None
    assert result.llm_result.llm_score == 70.0
