"""Shared fixtures. asyncio_mode = "auto" (pyproject.toml) means async
test functions run directly, with no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from lead_qualifier.domain.models import Lead
from lead_qualifier.services.rules_engine import RulesConfig


@pytest.fixture
def rules_config() -> RulesConfig:
    """A fixed, self-contained RulesConfig — deliberately not loaded from
    config/rules.yaml, so tuning that file for production can't silently
    change what these tests assert. Thresholds/weights mirror it anyway,
    since these values are what the docstrings/tests below reason about.
    """
    return RulesConfig(
        hard_filters=[
            {
                "field": "country",
                "operator": "not_in",
                "value": ["US", "CA"],
                "reason": "Outside served geography",
            },
            {
                "field": "company_size",
                "operator": "lt",
                "value": 5,
                "reason": "Company too small",
            },
        ],
        scoring_rules=[
            {
                "field": "company_size",
                "operator": "gte",
                "value": 200,
                "points": 25,
                "label": "Mid-market or larger",
            },
            {
                "field": "industry",
                "operator": "in",
                "value": ["Software", "SaaS"],
                "points": 20,
                "label": "Target industry",
            },
            {
                "field": "job_title",
                "operator": "contains_any",
                "value": ["VP", "Director", "Chief"],
                "points": 30,
                "label": "Senior title",
            },
            {
                "field": "budget",
                "operator": "gte",
                "value": 10000,
                "points": 25,
                "label": "Budget meets minimum",
            },
        ],
        rule_weight=0.4,
        llm_weight=0.6,
        hot_threshold=75,
        warm_threshold=45,
    )


@pytest.fixture
def make_lead():
    """Factory for a Lead that matches every scoring rule in
    `rules_config` above and fails none of its hard filters. Individual
    tests override just the field(s) they care about instead of
    repeating every required field.
    """

    def _make(**overrides: Any) -> Lead:
        defaults: dict[str, Any] = {
            "external_id": "12345",
            "source": "hubspot",
            "email": "jane@example.com",
            "company_name": "Acme Corp",
            "company_size": 250,
            "industry": "Software",
            "job_title": "VP of Engineering",
            "budget": 50000.0,
            "country": "US",
            "notes": "Looking to replace our current vendor by Q3.",
            "updated_at": datetime(2026, 1, 1, tzinfo=UTC),
        }
        defaults.update(overrides)
        return Lead(**defaults)

    return _make
