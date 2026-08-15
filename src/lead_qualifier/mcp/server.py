"""MCP server exposing qualification results as read-only tools for MCP
clients (e.g. Claude Desktop). The read-side counterpart to
api/routes/leads.py — both call the same LeadRepositoryPort methods, so
results never diverge between the HTTP and MCP surfaces.

Run with:
    python -m lead_qualifier.mcp.server
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from mcp.server import MCPServer

from lead_qualifier.core.logging import get_logger, setup_logging
from lead_qualifier.domain.models import QualificationResult
from lead_qualifier.infrastructure.db.lead_repository import PostgresLeadRepository
from lead_qualifier.infrastructure.db.session import async_session_factory

logger = get_logger(__name__)

# MCPServer (formerly FastMCP, renamed in mcp 2.0.0) — same @mcp.tool()
# decorator API, still defaults .run() to stdio transport.
mcp = MCPServer("lead-qualifier")


def _to_tool_result(result: QualificationResult) -> dict[str, Any]:
    """JSON-serializable shape for a QualificationResult, trimmed to what
    a human or LLM reading a tool result actually wants — drops
    rule_result/llm_result's internal detail (matched rules, retrieved
    chunks, token counts) since this is meant to be read, not round-
    tripped back into a QualificationResult."""
    return {
        "external_id": result.lead.external_id,
        "company_name": result.lead.company_name,
        "email": result.lead.email,
        "tier": result.tier.value,
        "final_score": result.final_score,
        "rationale": result.rationale,
        "created_at": result.created_at.isoformat(),
    }


@mcp.tool()
async def list_hot_leads(since: str | None = None) -> list[dict[str, Any]]:
    """List Hot-tier leads, newest first.

    since: optional ISO 8601 timestamp (e.g. "2026-08-01T00:00:00Z") —
    only leads qualified at/after this time are returned. Omit to get all
    Hot leads.
    """
    since_dt = datetime.fromisoformat(since) if since else None
    async with async_session_factory() as session:
        repository = PostgresLeadRepository(session)
        results = await repository.list_hot_leads(since=since_dt)
    return [_to_tool_result(result) for result in results]


@mcp.tool()
async def get_qualification(external_id: str) -> dict[str, Any]:
    """Get the most recent qualification result for one lead.

    external_id: the CRM's id for the lead (e.g. a HubSpot contact id).
    Returns {"found": false} if this lead has never been qualified,
    rather than an error — "never qualified" is an expected outcome, not
    a failure.
    """
    async with async_session_factory() as session:
        repository = PostgresLeadRepository(session)
        result = await repository.get(external_id)
    if result is None:
        return {"found": False}
    return {"found": True, **_to_tool_result(result)}


def main() -> None:
    setup_logging()
    mcp.run()


if __name__ == "__main__":
    main()
