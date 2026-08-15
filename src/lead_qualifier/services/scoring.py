"""Combines the rule-based score and the RAG-grounded LLM score into a
final tier. Kept separate from rules_engine.py since this step doesn't
touch the lead itself, only the two upstream results.
"""

from __future__ import annotations

from lead_qualifier.domain.models import LLMResult, RuleResult, Tier
from lead_qualifier.services.rules_engine import RulesConfig


def combine(
    rule_result: RuleResult,
    llm_result: LLMResult | None,
    config: RulesConfig,
) -> tuple[float, Tier, str]:
    """Blend a RuleResult and an (optional) LLMResult into one outcome.

    Returns (final_score, tier, rationale). If the lead was hard
    disqualified, the LLM step never ran (see the short-circuit in
    rules_engine.evaluate / qualify_lead.py), so llm_result is expected to
    be None in that case and is not required — passing None when the lead
    was NOT disqualified is a programming error, not a valid state, which
    is why it raises rather than silently defaulting to something.
    """
    if rule_result.hard_disqualified:
        return 0.0, Tier.DISQUALIFIED, rule_result.disqualify_reason or "Failed a hard filter"

    if llm_result is None:
        raise ValueError("llm_result is required when the lead is not hard-disqualified")

    final_score = (
        rule_result.rule_score * config.rule_weight + llm_result.llm_score * config.llm_weight
    )

    if final_score >= config.hot_threshold:
        tier = Tier.HOT
    elif final_score >= config.warm_threshold:
        tier = Tier.WARM
    else:
        tier = Tier.COLD

    rationale = (
        f"Rule score {rule_result.rule_score:.0f}/100 "
        f"({', '.join(rule_result.matched_rules) or 'no scoring rules matched'}); "
        f"LLM score {llm_result.llm_score:.0f}/100 — {llm_result.rationale}"
    )
    return final_score, tier, rationale
