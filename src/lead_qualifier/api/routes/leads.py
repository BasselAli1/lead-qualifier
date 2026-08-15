"""Manual read endpoints for qualification results — the HTTP counterpart
to the MCP server's get_qualification/list_hot_leads tools, for anything
that isn't an MCP client (internal dashboards, curl, etc). Both surfaces
call the same LeadRepositoryPort methods, so results never diverge.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status

from lead_qualifier.api.deps import LeadRepositoryDep
from lead_qualifier.domain.models import QualificationResult

router = APIRouter(prefix="/leads", tags=["leads"])


@router.get("/hot")
async def list_hot_leads(
    repository: LeadRepositoryDep,
    since: datetime | None = Query(default=None),  # noqa: B008 — FastAPI requires this
) -> list[QualificationResult]:
    return await repository.list_hot_leads(since=since)


@router.get("/{external_id}")
async def get_lead(external_id: str, repository: LeadRepositoryDep) -> QualificationResult:
    result = await repository.get(external_id)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead not found")
    return result
