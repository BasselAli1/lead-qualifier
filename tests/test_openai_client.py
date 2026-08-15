"""Adapter tests for infrastructure/llm/openai_client.py's OpenAIScorer.

Patches the constructed langfuse.openai.AsyncOpenAI client's
chat.completions.create directly, rather than going through httpx —
that call is what OpenAIScorer's own logic (JSON parsing, clamping,
prompt construction) actually depends on, and duck-typed SimpleNamespace
responses are enough to exercise it without a real API call.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lead_qualifier.domain.models import RetrievedChunk
from lead_qualifier.infrastructure.llm.openai_client import OpenAIScorer


def _fake_completion(score: float, rationale: str, model: str = "gpt-test") -> SimpleNamespace:
    """Duck-typed stand-in for an OpenAI ChatCompletion — only the
    attributes OpenAIScorer.score() actually reads."""
    content = json.dumps({"score": score, "rationale": rationale})
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model=model,
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=7),
    )


@pytest.fixture
def scorer() -> OpenAIScorer:
    return OpenAIScorer(api_key="test-key", model="gpt-test")


async def test_score_parses_json_response_into_llm_result(scorer):
    scorer._client.chat.completions.create = AsyncMock(
        return_value=_fake_completion(72.0, "Good but not great fit.")
    )

    result = await scorer.score("Company: Acme", context_chunks=[], trace_id="abc")

    assert result.llm_score == 72.0
    assert result.rationale == "Good but not great fit."
    assert result.model == "gpt-test"
    assert result.input_tokens == 42
    assert result.output_tokens == 7
    assert result.retrieved_chunks == []


async def test_score_clamps_score_above_100(scorer):
    scorer._client.chat.completions.create = AsyncMock(return_value=_fake_completion(150.0, "x"))

    result = await scorer.score("Company: Acme", context_chunks=[])

    assert result.llm_score == 100.0


async def test_score_clamps_score_below_0(scorer):
    scorer._client.chat.completions.create = AsyncMock(return_value=_fake_completion(-10.0, "x"))

    result = await scorer.score("Company: Acme", context_chunks=[])

    assert result.llm_score == 0.0


async def test_score_defaults_rationale_when_missing(scorer):
    completion = _fake_completion(50.0, "")
    completion.choices[0].message.content = json.dumps({"score": 50.0})
    scorer._client.chat.completions.create = AsyncMock(return_value=completion)

    result = await scorer.score("Company: Acme", context_chunks=[])

    assert result.rationale == "No rationale returned."


async def test_score_passes_trace_id_through_to_create_call(scorer):
    mock_create = AsyncMock(return_value=_fake_completion(60.0, "ok"))
    scorer._client.chat.completions.create = mock_create

    await scorer.score("Company: Acme", context_chunks=[], trace_id="lead-42:2026-01-01")

    assert mock_create.call_args.kwargs["trace_id"] == "lead-42:2026-01-01"


async def test_prompt_includes_retrieved_chunks(scorer):
    mock_create = AsyncMock(return_value=_fake_completion(60.0, "ok"))
    scorer._client.chat.completions.create = mock_create
    chunks = [
        RetrievedChunk(id="1", source="icp.md", content="Target: SaaS companies.", similarity=0.9)
    ]

    await scorer.score("Company: Acme", context_chunks=chunks)

    user_message = mock_create.call_args.kwargs["messages"][1]["content"]
    assert "Target: SaaS companies." in user_message
    assert "[icp.md]" in user_message


async def test_prompt_notes_missing_context_when_no_chunks_retrieved(scorer):
    mock_create = AsyncMock(return_value=_fake_completion(60.0, "ok"))
    scorer._client.chat.completions.create = mock_create

    await scorer.score("Company: Acme", context_chunks=[])

    user_message = mock_create.call_args.kwargs["messages"][1]["content"]
    assert "(no relevant context retrieved)" in user_message
