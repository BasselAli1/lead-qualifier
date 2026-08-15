"""HubSpot webhook endpoint. The only route that receives inherently
unauthenticated traffic (HubSpot, not a logged-in user) — signature
verification here is what stands in for auth.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from lead_qualifier.api.deps import CRMDep, QualifyLeadUseCaseDep
from lead_qualifier.core.logging import correlation_id_var, get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post("/hubspot", status_code=status.HTTP_200_OK)
async def hubspot_webhook(
    request: Request, crm: CRMDep, use_case: QualifyLeadUseCaseDep
) -> dict[str, str]:
    """Verify the request came from HubSpot, then run it through the
    qualification pipeline.

    Always returns 200 for anything past signature verification —
    including duplicates and disqualified leads — so HubSpot doesn't
    retry a delivery we've already handled correctly. Only an unverified
    signature gets a non-2xx, since that's the one case a retry can't fix.
    """
    body = await request.body()
    if not crm.verify_signature(
        method=request.method,
        uri=str(request.url),
        body=body.decode(),
        timestamp=request.headers.get("X-HubSpot-Request-Timestamp", ""),
        signature=request.headers.get("X-HubSpot-Signature-v3", ""),
    ):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    event = await request.json()

    # Correlation id for every log line this request produces (see
    # core/logging.py) — set from the raw event since the normalized Lead
    # doesn't exist until lead_from_webhook_event() runs inside use_case.
    token = correlation_id_var.set(str(event.get("objectId", "unknown")))
    try:
        result = await use_case.qualify_from_webhook_event(event)
    finally:
        correlation_id_var.reset(token)

    if result is None:
        return {"status": "duplicate_skipped"}
    return {"status": "qualified", "tier": result.tier.value}
