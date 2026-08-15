"""OpenAI adapter — implements LLMPort. Scores a lead's fit/intent with a
single JSON-mode chat completion, grounded in whatever context_chunks the
retriever found. The only module that knows the prompt wording or
OpenAI's request/response shape; everything downstream only ever sees an
LLMResult.
"""

from __future__ import annotations

import json
import time

from langfuse.openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from lead_qualifier.core.logging import get_logger
from lead_qualifier.domain.models import LLMResult, RetrievedChunk
from lead_qualifier.domain.ports import LLMPort

logger = get_logger(__name__)

# Must literally mention "JSON" for OpenAI's response_format=json_object
# mode, which otherwise rejects the request.
_SYSTEM_PROMPT = """You are a B2B sales lead qualification assistant. \
Score how well the described lead fits our ideal customer profile and \
shows buying intent, using ONLY the provided context (ICP definition, \
sales playbook, and/or past deal notes) plus the lead details given. \
Respond with a JSON object: {"score": <0-100 number>, "rationale": \
"<one to two sentence explanation>"}. If the context doesn't clearly \
support a high score, score conservatively."""


class OpenAIScorer(LLMPort):
    """Wraps OpenAI's chat completions API via langfuse.openai's drop-in
    client — a subclass of openai.AsyncOpenAI that transparently captures
    every call as a Langfuse generation. Passing `trace_id=` straight
    into .create() (a Langfuse-specific kwarg, stripped before the real
    API call) is what links this call into the same trace as the rest of
    the lead's pipeline run, per domain/ports.py's trace_id convention.
    If LANGFUSE_PUBLIC_KEY/SECRET_KEY aren't set (core/config.py allows
    both to be None), the wrapper logs a warning and no-ops tracing
    rather than failing the actual OpenAI call — so this adapter behaves
    the same with or without Langfuse configured.
    """

    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def score(
        self, lead_summary: str, context_chunks: list[RetrievedChunk], trace_id: str | None = None
    ) -> LLMResult:
        started = time.monotonic()
        # name/trace_id aren't part of the real OpenAI chat completions API
        # — both are langfuse.openai-specific kwargs, intercepted and
        # stripped by that wrapper before the real request goes out. mypy
        # type-checks against openai's own stubs, which don't know about
        # either.
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(lead_summary, context_chunks),
            response_format={"type": "json_object"},
            name="lead-scoring",  # type: ignore[call-overload]
            trace_id=trace_id,
        )
        latency_ms = (time.monotonic() - started) * 1000

        # response_format=json_object is an OpenAI platform guarantee of
        # well-formed JSON, so no try/except around the parse — a
        # malformed body here would mean the API broke its own contract.
        parsed = json.loads(response.choices[0].message.content or "{}")
        score = max(0.0, min(100.0, float(parsed.get("score", 0))))
        rationale = str(parsed.get("rationale") or "No rationale returned.")

        usage = response.usage
        return LLMResult(
            llm_score=score,
            rationale=rationale,
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            retrieved_chunks=context_chunks,
        )

    def _build_messages(
        self, lead_summary: str, context_chunks: list[RetrievedChunk]
    ) -> list[ChatCompletionMessageParam]:
        """Ground the prompt in whatever the retriever found. An empty
        context_chunks list (e.g. a lead with no notes to embed — see
        RetrieverPort's docstring) still gets scored, just without any
        retrieved grounding beyond the lead's own structured fields."""
        if context_chunks:
            context_block = "\n\n".join(
                f"[{chunk.source}] {chunk.content}" for chunk in context_chunks
            )
        else:
            context_block = "(no relevant context retrieved)"

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"# Context\n{context_block}\n\n# Lead\n{lead_summary}"},
        ]
