"""FastAPI application entrypoint. Run via
`uvicorn lead_qualifier.api.main:app` in dev; the Cloud Run container's
entrypoint runs the same target in production.
"""

from __future__ import annotations

from fastapi import FastAPI

from lead_qualifier.api.routes import leads, webhooks
from lead_qualifier.core.logging import setup_logging

setup_logging()

app = FastAPI(title="Lead Qualifier")
app.include_router(webhooks.router)
app.include_router(leads.router)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    """Cloud Run hits this to determine container health — no DB or
    downstream check on purpose, so a transient Postgres blip doesn't
    get the whole container killed and restarted."""
    return {"status": "ok"}
