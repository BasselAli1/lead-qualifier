# Multi-stage build: install dependencies with uv in a throwaway builder
# stage, then copy just the resulting .venv into a slim runtime image —
# keeps uv itself and any build-time cruft out of what actually ships.

FROM python:3.11-slim AS builder

# Official static uv binary, not `pip install uv` — nothing else needs to
# be resolved/installed before uv itself is available.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependency manifests copied (and installed) before the rest of the
# source, so Docker's layer cache is only invalidated by a dependency
# change, not by every code edit — `uv sync` here only needs uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Now the project itself (fast — its dependencies are already installed
# in the layer above).
COPY src/ src/
RUN uv sync --frozen --no-dev


FROM python:3.11-slim AS runtime

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

COPY --from=builder /app/.venv .venv
COPY src/ src/
COPY config/ config/
COPY alembic.ini ./
COPY alembic/ alembic/
COPY knowledge/ knowledge/

ENV PATH="/app/.venv/bin:$PATH"
USER appuser

EXPOSE 8080

# Shell-form CMD (not exec-form/JSON-array) so ${PORT:-8080} actually
# expands at container start — Cloud Run injects PORT at runtime, so it
# can't be baked in at build time. The leading `exec` still makes uvicorn
# PID 1 despite the shell wrapper, so it gets Cloud Run's SIGTERM
# directly for graceful shutdown instead of the shell swallowing it.
CMD exec uvicorn lead_qualifier.api.main:app --host 0.0.0.0 --port ${PORT:-8080}
