"""Centralizes all environment-driven settings. Every module that needs
config imports `settings` from here rather than reading os.environ directly.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All environment-driven configuration for the app, in one place.

    Values are read from a .env file (or real environment variables) at
    process startup. Fields with no default are required — missing one
    raises a ValidationError immediately when `settings` below is
    constructed, rather than failing later mid-request.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database (Postgres + pgvector; Cloud SQL in production)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/lead_qualifier"

    # LLM + embeddings (OpenAI covers both, so one key for both)
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-5.6-luna"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # LLM observability
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # HubSpot
    HUBSPOT_API_KEY: str
    HUBSPOT_WEBHOOK_SECRET: str

    # Slack
    SLACK_WEBHOOK_URL: str | None = None

    # App
    RULES_PATH: str = "config/rules.yaml"
    LOG_LEVEL: str = "INFO"
    PORT: int = Field(default=8080)

    # RAG
    RAG_TOP_K: int = 5


# Instantiated eagerly (at import time, not on first use) so a missing
# required env var surfaces immediately rather than deep into a request.
settings = Settings()
