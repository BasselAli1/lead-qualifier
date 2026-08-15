"""Adapter tests for infrastructure/crm/hubspot.py's HubSpotCRM.

Uses httpx.MockTransport (built into httpx, not a mocking library) to
intercept the real network calls hubspot.py makes, so these exercise the
actual request/response handling code rather than mocking it away.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import httpx
import pytest

from lead_qualifier.domain.models import Tier
from lead_qualifier.infrastructure.crm import hubspot as hubspot_module
from lead_qualifier.infrastructure.crm.hubspot import HubSpotCRM


def _sign(secret: str, method: str, uri: str, body: str, timestamp: str) -> str:
    source_string = f"{method}{uri}{body}{timestamp}"
    return base64.b64encode(
        hmac.new(secret.encode(), source_string.encode(), hashlib.sha256).digest()
    ).decode()


def _mock_transport(monkeypatch, handler):
    """hubspot.py builds its own httpx.AsyncClient per call, with no seam
    to inject one — so this patches httpx.AsyncClient itself (module-
    global for the test's duration, restored by monkeypatch afterward) to
    route through a fake transport instead of the real network.

    real_async_client is captured before patching: httpx is a single
    shared module object, so patching httpx.AsyncClient also changes what
    `httpx.AsyncClient` resolves to inside factory() itself — without
    this, factory would call itself instead of the real client.
    """
    real_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(hubspot_module.httpx, "AsyncClient", factory)


@pytest.fixture
def crm() -> HubSpotCRM:
    return HubSpotCRM(api_key="test-api-key", webhook_secret="test-webhook-secret")


class TestVerifySignature:
    def test_accepts_a_correctly_signed_request(self, crm):
        timestamp = str(int(time.time() * 1000))
        signature = _sign("test-webhook-secret", "POST", "/webhooks/hubspot", "{}", timestamp)

        assert crm.verify_signature("POST", "/webhooks/hubspot", "{}", timestamp, signature)

    def test_rejects_wrong_secret(self, crm):
        timestamp = str(int(time.time() * 1000))
        signature = _sign("wrong-secret", "POST", "/webhooks/hubspot", "{}", timestamp)

        assert not crm.verify_signature("POST", "/webhooks/hubspot", "{}", timestamp, signature)

    def test_rejects_tampered_body(self, crm):
        timestamp = str(int(time.time() * 1000))
        signature = _sign("test-webhook-secret", "POST", "/webhooks/hubspot", "{}", timestamp)

        tampered_body = '{"tampered": true}'
        assert not crm.verify_signature(
            "POST", "/webhooks/hubspot", tampered_body, timestamp, signature
        )

    def test_rejects_stale_timestamp(self, crm):
        stale_timestamp = str(int((time.time() - 3600) * 1000))  # one hour old
        signature = _sign("test-webhook-secret", "POST", "/webhooks/hubspot", "{}", stale_timestamp)

        assert not crm.verify_signature(
            "POST", "/webhooks/hubspot", "{}", stale_timestamp, signature
        )

    def test_rejects_malformed_timestamp(self, crm):
        assert not crm.verify_signature(
            "POST", "/webhooks/hubspot", "{}", "not-a-number", "any-signature"
        )


class TestLeadFromWebhookEvent:
    async def test_maps_hubspot_properties_onto_lead(self, crm, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/crm/v3/objects/contacts/999"
            return httpx.Response(
                200,
                json={
                    "properties": {
                        "email": "jane@example.com",
                        "company": "Acme Corp",
                        "company_size": "250",
                        "industry": "Software",
                        "jobtitle": "VP of Engineering",
                        "budget": "50000",
                        "country": "US",
                        "lead_notes": "Interested in Q3.",
                        "hs_lastmodifieddate": "1750000000000",
                    }
                },
            )

        _mock_transport(monkeypatch, handler)

        lead = await crm.lead_from_webhook_event({"objectId": 999})

        assert lead.external_id == "999"
        assert lead.source == "hubspot"
        assert lead.email == "jane@example.com"
        assert lead.company_name == "Acme Corp"
        assert lead.company_size == 250  # HubSpot's string "250" -> int
        assert lead.budget == 50000.0  # HubSpot's string "50000" -> float
        assert lead.notes == "Interested in Q3."

    async def test_missing_optional_properties_default_sensibly(self, crm, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"properties": {}})

        _mock_transport(monkeypatch, handler)

        lead = await crm.lead_from_webhook_event({"objectId": 1})

        assert lead.notes == ""
        assert lead.company_size is None
        assert lead.budget is None

    async def test_raises_on_non_2xx_response(self, crm, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"message": "not found"})

        _mock_transport(monkeypatch, handler)

        with pytest.raises(httpx.HTTPStatusError):
            await crm.lead_from_webhook_event({"objectId": 1})


class TestPushQualification:
    async def test_sends_tier_score_and_rationale_as_properties(self, crm, monkeypatch):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={})

        _mock_transport(monkeypatch, handler)

        await crm.push_qualification("42", Tier.HOT, 88.5, "Strong fit.")

        assert captured["path"] == "/crm/v3/objects/contacts/42"
        properties = captured["body"]["properties"]
        assert properties["lead_qualifier_tier"] == "hot"
        assert properties["lead_qualifier_score"] == 88.5
        assert properties["lead_qualifier_rationale"] == "Strong fit."

    async def test_raises_on_non_2xx_response(self, crm, monkeypatch):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"message": "bad request"})

        _mock_transport(monkeypatch, handler)

        with pytest.raises(httpx.HTTPStatusError):
            await crm.push_qualification("42", Tier.HOT, 88.5, "Strong fit.")
