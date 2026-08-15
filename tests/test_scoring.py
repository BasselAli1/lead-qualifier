"""Unit tests for services/scoring.py's combine()."""

from __future__ import annotations

import pytest

from lead_qualifier.domain.models import LLMResult, RuleResult, Tier
from lead_qualifier.services.scoring import combine


def _llm_result(score: float, rationale: str = "Strong fit.") -> LLMResult:
    return LLMResult(
        llm_score=score,
        rationale=rationale,
        model="gpt-test",
        input_tokens=10,
        output_tokens=10,
        latency_ms=100.0,
    )


def test_hard_disqualified_short_circuits_without_llm_result(rules_config):
    rule_result = RuleResult(
        hard_disqualified=True, disqualify_reason="Outside served geography", rule_score=0
    )

    final_score, tier, rationale = combine(rule_result, None, rules_config)

    assert final_score == 0.0
    assert tier == Tier.DISQUALIFIED
    assert rationale == "Outside served geography"


def test_missing_llm_result_raises_when_not_disqualified(rules_config):
    rule_result = RuleResult(hard_disqualified=False, rule_score=50)

    with pytest.raises(ValueError, match="llm_result is required"):
        combine(rule_result, None, rules_config)


@pytest.mark.parametrize(
    "rule_score,llm_score,expected_tier",
    [
        (100, 100, Tier.HOT),  # final = 100 -> >= hot_threshold (75)
        (50, 50, Tier.WARM),  # final = 50 -> >= warm_threshold (45), < 75
        (0, 20, Tier.COLD),  # final = 12 -> < 45
    ],
)
def test_tier_thresholds(rules_config, rule_score, llm_score, expected_tier):
    rule_result = RuleResult(hard_disqualified=False, rule_score=rule_score, matched_rules=["x"])
    llm_result = _llm_result(llm_score)

    _, tier, _ = combine(rule_result, llm_result, rules_config)

    assert tier == expected_tier


def test_final_score_is_weighted_blend(rules_config):
    rule_result = RuleResult(hard_disqualified=False, rule_score=80, matched_rules=["a", "b"])
    llm_result = _llm_result(60)

    final_score, _, _ = combine(rule_result, llm_result, rules_config)

    assert final_score == pytest.approx(80 * 0.4 + 60 * 0.6)


def test_rationale_includes_both_scores_and_matched_rules(rules_config):
    rule_result = RuleResult(
        hard_disqualified=False, rule_score=80, matched_rules=["Target industry"]
    )
    llm_result = _llm_result(60, rationale="Clear intent to buy.")

    _, _, rationale = combine(rule_result, llm_result, rules_config)

    assert "80/100" in rationale
    assert "Target industry" in rationale
    assert "60/100" in rationale
    assert "Clear intent to buy." in rationale


def test_rationale_notes_no_scoring_rules_matched(rules_config):
    rule_result = RuleResult(hard_disqualified=False, rule_score=0, matched_rules=[])
    llm_result = _llm_result(10)

    _, _, rationale = combine(rule_result, llm_result, rules_config)

    assert "no scoring rules matched" in rationale
