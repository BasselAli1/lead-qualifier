"""Slack adapter — implements NotifierPort. Posts a Hot-tier lead summary
to a Slack Incoming Webhook. The only module that knows Slack's payload
shape; everything upstream only ever deals in QualificationResult.
"""

from __future__ import annotations

import httpx

from lead_qualifier.core.logging import get_logger
from lead_qualifier.domain.models import QualificationResult
from lead_qualifier.domain.ports import NotifierPort

logger = get_logger(__name__)


class SlackNotifier(NotifierPort):
    """Posts to a Slack Incoming Webhook URL. Built fresh per request in
    api/deps.py, mirroring the other adapters even though it holds no
    per-request state."""

    def __init__(self, webhook_url: str | None) -> None:
        self._webhook_url = webhook_url

    async def notify_hot_lead(self, result: QualificationResult) -> None:
        """Post a Hot-tier lead summary to Slack.

        Two deliberate non-raising cases, both logged instead: no
        webhook_url configured (SLACK_WEBHOOK_URL is Optional in
        core/config.py — Slack is a nice-to-have, not a required
        integration) and a failed POST. By this point in
        application/qualify_lead.py the result is already saved and
        pushed to HubSpot, so a Slack outage shouldn't turn into a 500 for
        a webhook delivery that otherwise succeeded.
        """
        if self._webhook_url is None:
            logger.warning(
                "SLACK_WEBHOOK_URL not configured, skipping notification",
                extra={"external_id": result.lead.external_id},
            )
            return

        payload = {"text": self._build_message(result)}
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self._webhook_url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError:
            logger.exception(
                "Failed to post Slack notification",
                extra={"external_id": result.lead.external_id},
            )

    @staticmethod
    def _build_message(result: QualificationResult) -> str:
        """Build the Slack message body using mrkdwn (Slack's slightly
        nonstandard markdown dialect — *bold*, not **bold**)."""
        lead = result.lead
        company = lead.company_name or "Unknown company"
        contact = lead.email or lead.external_id
        return (
            f":fire: *Hot lead: {company}* ({contact})\n"
            f"Score: *{result.final_score:.0f}/100*\n"
            f"{result.rationale}"
        )
