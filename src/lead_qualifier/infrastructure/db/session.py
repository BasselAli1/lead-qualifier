"""SQLAlchemy async engine and session factory for Postgres (local
docker-compose Postgres in dev, Cloud SQL in production — both speak the
same asyncpg protocol via DATABASE_URL).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from lead_qualifier.core.config import settings

# One engine per process, reused across requests — engines manage a
# connection pool internally, so creating a new one per request would
# defeat pooling entirely. pool_pre_ping checks a connection is still
# alive before handing it out, which matters for Cloud SQL connections
# that can be silently dropped after periods of idleness.
#
# statement_cache_size=0 disables asyncpg's server-side prepared
# statement cache. DATABASE_URL is expected to point at Neon's *pooled*
# (PgBouncer, transaction-mode) endpoint — PgBouncer in that mode hands
# out a different underlying server connection per transaction, but
# asyncpg's prepared statements are tied to one specific server
# connection, so without this they intermittently fail with
# DuplicatePreparedStatementError. Not needed (but harmless) against a
# direct/unpooled connection.
engine = create_async_engine(
    settings.DATABASE_URL, pool_pre_ping=True, connect_args={"statement_cache_size": 0}
)

# expire_on_commit=False avoids a footgun specific to async sessions: by
# default, commit() expires an object's attributes, and reading an
# expired attribute normally triggers an implicit lazy-load SELECT — but
# async sessions can't do that implicitly (no way to await inside a plain
# attribute access), so it raises MissingGreenlet instead. This setting
# sidesteps that regardless of whether any code currently reads an object
# post-commit.
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency that yields one session per request.

    The `async with` ensures the session is closed (connection returned
    to the pool) even if the request handler raises — api/deps.py wires
    this in via FastAPI's Depends().
    """
    async with async_session_factory() as session:
        yield session
