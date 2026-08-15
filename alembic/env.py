"""Alembic environment — bridges Alembic's migration API to this project's
async engine. See infrastructure/db/session.py for the actual
engine/DATABASE_URL setup; that engine is reused here rather than built
again, so there's one less place that needs to agree on connection
settings.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from sqlalchemy import Connection

from alembic import context
from lead_qualifier.infrastructure.db.orm_models import Base
from lead_qualifier.infrastructure.db.session import engine

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Drives `alembic revision --autogenerate`, which diffs this against the
# live database to propose migration operations. Base (orm_models.py) is
# the single source of truth for the target schema.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it against a database — for
    `alembic upgrade --sql`, generating a script a DBA runs by hand rather
    than Alembic connecting directly."""
    context.configure(
        url=str(engine.url),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Migrations run through Alembic's synchronous API
    (do_run_migrations); connection.run_sync bridges this async
    connection to that sync call — asyncpg can't do that implicitly, the
    same consideration behind session.py's expire_on_commit comment, just
    on the connection/execution side rather than the ORM attribute side.
    """
    async with engine.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await engine.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
