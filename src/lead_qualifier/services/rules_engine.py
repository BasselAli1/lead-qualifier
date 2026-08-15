"""Evaluates config/rules.yaml against a Lead. Hard filters short-circuit
qualification before any LLM/embedding spend; scoring rules produce a
weighted 0-100 rule_score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml

from lead_qualifier.domain.models import Lead, RuleResult


@dataclass(frozen=True)
class RulesConfig:
    """Parsed, ready-to-use form of config/rules.yaml.

    Kept as a plain dataclass (not a pydantic model) since this is purely
    internal to the services layer, not something crossing an API/DB
    boundary that needs validation or serialization.
    """

    hard_filters: list[dict[str, Any]]
    scoring_rules: list[dict[str, Any]]
    rule_weight: float
    llm_weight: float
    hot_threshold: float
    warm_threshold: float


def load_rules_config(path: str) -> RulesConfig:
    """Read and parse a rules.yaml file into a RulesConfig.

    Called once at startup (api/deps.py) rather than on every request —
    rules.yaml is expected to change rarely enough that a process restart
    to pick up edits is an acceptable tradeoff for not re-parsing YAML on
    every lead.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    combine = raw.get("combine", {})
    tiers = raw.get("tiers", {})
    return RulesConfig(
        hard_filters=raw.get("hard_filters", []),
        scoring_rules=raw.get("scoring_rules", []),
        rule_weight=combine.get("rule_weight", 0.5),
        llm_weight=combine.get("llm_weight", 0.5),
        hot_threshold=tiers.get("hot", 75),
        warm_threshold=tiers.get("warm", 45),
    )


def _matches(field_value: Any, operator: str, expected: Any) -> bool:
    """Evaluate one rule's operator against one field's value on a lead.

    A missing field (None) never matches any operator — a rule about
    `budget` simply doesn't apply to a lead where budget wasn't captured,
    rather than raising or matching by accident on None comparisons.
    """
    if field_value is None:
        return False

    if operator == "eq":
        return bool(field_value == expected)
    if operator == "ne":
        return bool(field_value != expected)
    if operator == "lt":
        return bool(field_value < expected)
    if operator == "lte":
        return bool(field_value <= expected)
    if operator == "gt":
        return bool(field_value > expected)
    if operator == "gte":
        return bool(field_value >= expected)
    if operator == "in":
        return field_value in expected
    if operator == "not_in":
        return field_value not in expected
    if operator == "contains":
        return str(expected).lower() in str(field_value).lower()
    if operator == "contains_any":
        haystack = str(field_value).lower()
        return any(str(needle).lower() in haystack for needle in expected)

    raise ValueError(f"Unknown rule operator: {operator}")


def evaluate(lead: Lead, config: RulesConfig) -> RuleResult:
    """Run every hard filter, then every scoring rule, against one lead.

    Hard filters are checked first and short-circuit on the first match —
    order in config/rules.yaml doesn't otherwise matter, but the first
    matching hard filter's `reason` is what ends up in the result.
    Scoring rules always run in full (no short-circuit) so the rationale
    can list every rule that matched, not just the first.
    """
    lead_fields = lead.model_dump()

    for rule in config.hard_filters:
        field_value = lead_fields.get(rule["field"])
        if _matches(field_value, rule["operator"], rule["value"]):
            return RuleResult(
                hard_disqualified=True,
                disqualify_reason=rule.get("reason", f"Failed hard filter on {rule['field']}"),
                rule_score=0,
                matched_rules=[],
            )

    matched: list[str] = []
    total_points = 0.0
    for rule in config.scoring_rules:
        field_value = lead_fields.get(rule["field"])
        if _matches(field_value, rule["operator"], rule["value"]):
            total_points += rule.get("points", 0)
            matched.append(rule.get("label", rule["field"]))

    rule_score = min(total_points, 100.0)
    return RuleResult(
        hard_disqualified=False,
        rule_score=rule_score,
        matched_rules=matched,
    )
