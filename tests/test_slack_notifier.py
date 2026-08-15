"""Adapter tests for infrastructure/notifications/slack_notifier.py's
SlackNotifier. Uses httpx.MockTransport, same technique as
tests/test_hubspot.py.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx
import pytest

from lead_qualifier.domain.models import Lead, QualificationResult, RuleResult, Tier
from lead_qualifier.infrastructure.notifications import slack_notifier as slack_module
from lead_qualifier.infrastructure.notifications.slack_notifier import SlackNotifier


def _mock_transport(monkeypatch, handler):
    """See tests/test_hubspot.py's _mock_transport for why
    real_async_client must be captured before patching."""
    real_async_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(slack_module.httpx, "AsyncClient", factory)


def _make_result(**overrides) -> QualificationResult:
    lead = Lead(
        external_id="1",
        company_name="Acme Corp",
        email="jane@example.com",
        updated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    defaults: dict = {
        "lead": lead,
        "tier": Tier.HOT,
        "final_score": 88.0,
        "rule_result": RuleResult(hard_disqualified=False, rule_score=80),
        "rationale": "Strong fit.",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    defaults.update(overrides)
    return QualificationResult(**defaults)


@pytest.fixture
def webhook_url() -> str:
    return "https://hooks.slack.com/services/T000/B000/XXX"


async def test_posts_formatted_message_when_webhook_configured(monkeypatch, webhook_url):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"ok": True})

    _mock_transport(monkeypatch, handler)
    notifier = SlackNotifier(webhook_url=webhook_url)

    await notifier.notify_hot_lead(_make_result())

    assert captured["url"] == webhook_url
    assert "Acme Corp" in captured["body"]["text"]
    assert "88" in captured["body"]["text"]


async def test_no_request_made_when_webhook_not_configured(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should never make a request when webhook_url is None")

    _mock_transport(monkeypatch, handler)
    notifier = SlackNotifier(webhook_url=None)

    await notifier.notify_hot_lead(_make_result())  # must not raise


async def test_does_not_raise_when_slack_request_fails(monkeypatch, webhook_url):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    _mock_transport(monkeypatch, handler)
    notifier = SlackNotifier(webhook_url=webhook_url)

    await notifier.notify_hot_lead(_make_result())  # must not raise, only logged
