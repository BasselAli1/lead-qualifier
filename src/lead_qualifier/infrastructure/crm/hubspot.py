"""HubSpot adapter — implements CRMPort. The only module in the app that
knows HubSpot's webhook signature scheme, API shapes, and property names;
everything downstream only ever sees a normalized Lead (domain/models.py)
or a Tier to push back, never a raw HubSpot payload.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from datetime import UTC, datetime

import httpx

from lead_qualifier.core.logging import get_logger
from lead_qualifier.domain.models import Lead, Tier
from lead_qualifier.domain.ports import CRMPort

logger = get_logger(__name__)

_API_BASE = "https://api.hubapi.com"

# A request older than this is rejected even with an otherwise-valid
# signature — without an age check, a captured request stays replayable
# forever since its signature never expires on its own.
_MAX_SIGNATURE_AGE_SECONDS = 300

# Contact properties fetched for every lead. `company_size`, `budget`, and
# `lead_notes` are custom properties this app expects the portal to have
# (HubSpot has no stock contact fields for these); the rest are stock.
_LEAD_PROPERTIES = [
    "email",
    "company",
    "company_size",
    "industry",
    "jobtitle",
    "budget",
    "country",
    "lead_notes",
    "hs_lastmodifieddate",
]


def _parse_float(value: str | None) -> float | None:
    """HubSpot's API returns every property value as a string (or null),
    regardless of the property's configured type on the portal."""
    return float(value) if value else None


def _parse_int(value: str | None) -> int | None:
    return int(float(value)) if value else None


class HubSpotCRM(CRMPort):
    """Talks to HubSpot's CRM v3 API and validates its v3 webhook
    signature scheme. Built fresh per request in api/deps.py."""

    def __init__(self, access_token: str, webhook_secret: str) -> None:
        self._access_token = access_token
        self._webhook_secret = webhook_secret

    def verify_signature(
        self, method: str, uri: str, body: str, timestamp: str, signature: str
    ) -> bool:
        """HubSpot v3 signature: HMAC-SHA256 of `method + uri + body +
        timestamp`, keyed by the app's webhook secret, base64-encoded.

        Also rejects a stale timestamp — see _MAX_SIGNATURE_AGE_SECONDS —
        since HMAC alone only proves the request wasn't tampered with, not
        that it's fresh.
        """
        try:
            request_age_seconds = time.time() - int(timestamp) / 1000
        except ValueError:
            return False
        if request_age_seconds > _MAX_SIGNATURE_AGE_SECONDS:
            return False

        source_string = f"{method}{uri}{body}{timestamp}"
        expected = base64.b64encode(
            hmac.new(self._webhook_secret.encode(), source_string.encode(), hashlib.sha256).digest()
        ).decode()
        return hmac.compare_digest(expected, signature)

    async def lead_from_webhook_event(self, event: dict) -> Lead:
        """Fetch the full contact from HubSpot's API and normalize it.

        The webhook payload only carries enough to identify what changed
        (`objectId`) — not the property values this app needs — so this
        always does a follow-up API read rather than parsing fields out
        of the event body itself.
        """
        contact_id = str(event["objectId"])
        async with httpx.AsyncClient(base_url=_API_BASE, headers=self._auth_header()) as client:
            response = await client.get(
                f"/crm/v3/objects/contacts/{contact_id}",
                params={"properties": ",".join(_LEAD_PROPERTIES)},
            )
        response.raise_for_status()
        props = response.json()["properties"]

        updated_at = (
            datetime.fromtimestamp(int(props["hs_lastmodifieddate"]) / 1000, tz=UTC)
            if props.get("hs_lastmodifieddate")
            else datetime.now(UTC)
        )

        return Lead(
            external_id=contact_id,
            source="hubspot",
            email=props.get("email"),
            company_name=props.get("company"),
            company_size=_parse_int(props.get("company_size")),
            industry=props.get("industry"),
            job_title=props.get("jobtitle"),
            budget=_parse_float(props.get("budget")),
            country=props.get("country"),
            notes=props.get("lead_notes") or "",
            updated_at=updated_at,
        )

    async def push_qualification(
        self, external_id: str, tier: Tier, final_score: float, rationale: str
    ) -> None:
        """Write the result back as contact properties so sales reps see
        it in HubSpot without leaving the CRM. `lead_qualifier_tier` is
        the same property name domain/models.py's Tier docstring assumes
        its string values serialize to directly, with no conversion step.
        """
        async with httpx.AsyncClient(base_url=_API_BASE, headers=self._auth_header()) as client:
            response = await client.patch(
                f"/crm/v3/objects/contacts/{external_id}",
                json={
                    "properties": {
                        "lead_qualifier_tier": tier.value,
                        "lead_qualifier_score": final_score,
                        "lead_qualifier_rationale": rationale,
                    }
                },
            )
        response.raise_for_status()

    def _auth_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._access_token}"}
