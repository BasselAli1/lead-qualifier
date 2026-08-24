# Lead Qualifier

Hybrid rules + RAG-grounded LLM lead qualification service for HubSpot. A
HubSpot contact update triggers a webhook, the lead is scored by a
deterministic rules engine and an OpenAI call grounded in your own ICP/sales
playbook/past-deals content, and the result (tier, score, rationale) is
written back to HubSpot, persisted, and posted to Slack for Hot leads.

## How it works

```
HubSpot webhook
      │
      ▼
Verify signature (HMAC, HubSpot's webhook secret)
      │
      ▼
Fetch full contact from HubSpot API  ──►  normalize into a Lead
      │
      ▼
Rules engine (config/rules.yaml)
      │
      ├─ hard filter matched? ──► Disqualified (LLM never runs, saves cost)
      │
      ▼
RAG retrieval (pgvector, knowledge/*.md) + OpenAI scoring, grounded in the
retrieved context
      │
      ▼
Combine rule_score + llm_score → final_score, tier (Hot/Warm/Cold)
      │
      ▼
Persist to Postgres  →  push tier/score/rationale back to HubSpot  →  Slack
notification if Hot
```

## Architecture

Hexagonal / ports-and-adapters, under `src/lead_qualifier/`:

| Layer | Contains |
|---|---|
| `domain/` | Pydantic models (`Lead`, `QualificationResult`, ...) and `ports.py`: the ABCs every adapter implements. No framework/vendor imports. |
| `services/` | Pure business logic: `rules_engine.py` (evaluates `config/rules.yaml`), `scoring.py` (combines rule + LLM scores into a tier). |
| `application/` | `qualify_lead.py`: the one orchestrator that calls multiple ports in sequence. Depends only on `domain/` and `services/`, so it's unit-testable with fakes and no real infrastructure. |
| `infrastructure/` | Concrete adapters, one per port: `crm/hubspot.py`, `llm/openai_client.py`, `rag/retriever.py` + `rag/ingest.py`, `notifications/slack_notifier.py`, `db/` (SQLAlchemy models, session, repository). |
| `api/` | FastAPI app: `main.py`, `deps.py` (dependency injection wiring adapters to ports), `routes/` (`webhooks.py`, `leads.py`). |
| `mcp/` | MCP server (`server.py`) exposing `list_hot_leads`/`get_qualification` as tools. Uses the same `LeadRepositoryPort` the HTTP API uses, so results never diverge between the two surfaces. |
| `core/` | `config.py` (env-driven `Settings`), `logging.py` (structured JSON logs with a correlation id per request). |

Every adapter is built fresh per-request via FastAPI `Depends` (see `api/deps.py`). Nothing is a global singleton except the DB engine and connection pool.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Docker (for local Postgres+pgvector)
- Accounts/credentials for: OpenAI, HubSpot (Private App access token), Neon (or any Postgres+pgvector), Langfuse (optional), Slack (optional)

## Setup

```bash
uv sync --dev          # creates .venv, installs everything including dev tools
cp .env.example .env   # fill in real values, see comments in the file for gotchas
docker compose up -d   # local Postgres+pgvector for dev (not needed if pointing at Neon)
uv run alembic upgrade head          # create the schema
uv run python -m lead_qualifier.infrastructure.rag.ingest   # embed knowledge/*.md into pgvector
```

Two non-obvious things `.env.example` already documents inline, worth knowing up front if you're pointing at Neon:
- Use `?ssl=require`, not `?sslmode=require` (asyncpg-specific, Neon's dashboard shows the latter by default).
- Strip `&channel_binding=require` if Neon's connection string includes it. asyncpg has no such parameter and raises `TypeError` outright rather than ignoring it.
- If using Neon's *pooled* endpoint, `infrastructure/db/session.py` already sets `statement_cache_size=0` to avoid asyncpg/PgBouncer prepared-statement conflicts.

## Running locally

```bash
uv run fastapi dev              # dev server with reload, http://localhost:8000
# or
uv run fastapi run              # production mode, no reload
```

Both auto-detect the app via the `[tool.fastapi]` entrypoint in `pyproject.toml` (`lead_qualifier.api.main:app`). No path argument needed despite the src-layout.

MCP server: `uv run python -m lead_qualifier.mcp.server`

## Testing

```bash
uv run pytest                   # unit + adapter tests (fakes/mocks, no real services), excludes eval
uv run pytest -m eval            # LLM eval suite against a golden dataset (real OpenAI calls, costs money)
uv run ruff check .
uv run mypy src
```

Test layout (all flat under `tests/`, no subpackages):
- `test_rules_engine.py`, `test_scoring.py`: pure services logic.
- `test_qualify_lead.py`: orchestration, using hand-written fakes (`tests/fakes.py`) for all five ports.
- `test_hubspot.py`, `test_openai_client.py`, `test_retriever.py`, `test_slack_notifier.py`, `test_lead_repository.py`: adapters, using `httpx.MockTransport` for HTTP-based adapters and mocked SDK clients rather than a real network/DB.
- `test_llm_eval.py`: the real-OpenAI eval suite (`@pytest.mark.eval`, excluded from the default run via `pyproject.toml`'s `addopts`).

## Deployment

Deployed on [FastAPI Cloud](https://fastapicloud.com). Two GitHub Actions workflows:

- **`.github/workflows/ci.yml`**: ruff + mypy + pytest on every push/PR to `main`.
- **`.github/workflows/deploy.yml`**: triggers via `workflow_run` once `ci.yml` succeeds on `main` (not on push directly, so a failing check blocks the deploy), then runs `fastapi deploy` using `FASTAPI_CLOUD_TOKEN`/`FASTAPI_CLOUD_APP_ID` (encrypted repo secrets, provisioned via `fastapi cloud setup-ci`).

Manual deploy: `uv run fastapi deploy`. Manual migration against a real (e.g. Neon) database: `DATABASE_URL=... uv run alembic upgrade head`. Deploying the app and migrating the database are separate steps. Neither triggers the other.

A `Dockerfile`/`docker-compose.yml` also exist for a container-based deploy path (e.g. Cloud Run) if you ever move off FastAPI Cloud. Note that FastAPI Cloud itself ignores both and builds directly from `pyproject.toml`/`uv.lock`.

## Configuration

- **`config/rules.yaml`**: hard filters (auto-disqualify) and weighted scoring rules, plus `rule_weight`/`llm_weight` and tier thresholds for `services/scoring.py`'s `combine()`. Edit this file, not code, to retune qualification criteria.
- **`knowledge/*.md`**: ICP definition, sales playbook, past deal notes. Chunked on `## ` headers and embedded into pgvector by `infrastructure/rag/ingest.py`. Re-run that script after editing any of these files.
- **`.env`**: see `.env.example` for the full list of variables and inline notes on the less obvious ones.
