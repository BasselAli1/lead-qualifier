"""Unit tests for services/rules_engine.py's evaluate()."""

from __future__ import annotations

import dataclasses

import pytest

from lead_qualifier.services.rules_engine import evaluate


def test_hard_filter_short_circuits_scoring(rules_config, make_lead):
    lead = make_lead(country="DE")  # not in ["US", "CA"]

    result = evaluate(lead, rules_config)

    assert result.hard_disqualified is True
    assert result.disqualify_reason == "Outside served geography"
    assert result.rule_score == 0
    assert result.matched_rules == []


def test_hard_filter_company_too_small(rules_config, make_lead):
    lead = make_lead(company_size=2)

    result = evaluate(lead, rules_config)

    assert result.hard_disqualified is True
    assert result.disqualify_reason == "Company too small"


def test_first_matching_hard_filter_wins(rules_config, make_lead):
    """A lead failing two hard filters at once still returns exactly one
    reason — whichever filter is listed first in config."""
    lead = make_lead(country="DE", company_size=2)

    result = evaluate(lead, rules_config)

    assert result.disqualify_reason == "Outside served geography"


def test_all_scoring_rules_match(rules_config, make_lead):
    lead = make_lead()  # matches every scoring rule via the fixture's defaults

    result = evaluate(lead, rules_config)

    assert result.hard_disqualified is False
    assert result.rule_score == 100  # 25 + 20 + 30 + 25
    assert set(result.matched_rules) == {
        "Mid-market or larger",
        "Target industry",
        "Senior title",
        "Budget meets minimum",
    }


def test_no_scoring_rules_match(rules_config, make_lead):
    lead = make_lead(
        company_size=50,  # below the 200 bonus, but above the hard-filter floor of 5
        industry="Retail",
        job_title="Sales Associate",
        budget=500.0,
    )

    result = evaluate(lead, rules_config)

    assert result.hard_disqualified is False
    assert result.rule_score == 0
    assert result.matched_rules == []


def test_missing_field_never_matches(rules_config, make_lead):
    """A rule about a field the lead didn't provide simply doesn't apply
    — it isn't treated as a match or an error."""
    lead = make_lead(budget=None)

    result = evaluate(lead, rules_config)

    assert "Budget meets minimum" not in result.matched_rules


def test_rule_score_caps_at_100(rules_config, make_lead):
    """Points aren't capped at 100 in config itself — evaluate() clamps
    the total, protecting RuleResult.rule_score's 0-100 contract even if
    someone tunes rules.yaml's points to sum past 100."""
    heavy_config = dataclasses.replace(
        rules_config,
        hard_filters=[],
        scoring_rules=[
            {"field": "country", "operator": "eq", "value": "US", "points": 80, "label": "a"},
            {"field": "source", "operator": "eq", "value": "hubspot", "points": 80, "label": "b"},
        ],
    )
    lead = make_lead()

    result = evaluate(lead, heavy_config)

    assert result.rule_score == 100


def test_unknown_operator_raises(rules_config, make_lead):
    bad_config = dataclasses.replace(
        rules_config,
        hard_filters=[],
        scoring_rules=[
            {"field": "country", "operator": "bogus", "value": "US", "points": 10, "label": "x"}
        ],
    )
    lead = make_lead()

    with pytest.raises(ValueError, match="Unknown rule operator"):
        evaluate(lead, bad_config)
