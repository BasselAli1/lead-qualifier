"""Structured JSON logging. Cloud Run ships stdout/stderr to Cloud Logging
and parses JSON automatically; `severity` and `message` are the fields it
looks for. A per-request correlation id (usually the HubSpot lead id) is
threaded through a contextvar so every log line for one webhook can be
grouped together in Cloud Logging.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar

from pythonjsonlogger import json as jsonlogger

from lead_qualifier.core.config import settings

# Holds the current request's correlation id. A ContextVar (rather than a
# plain global) is scoped per async task, so concurrent requests being
# handled at the same time never see each other's value.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


class CorrelationIdFilter(logging.Filter):
    """Stamps every log record with the current correlation id.

    Attached to the root logger's handler in setup_logging(), so it runs
    on every single log call in the app without each call site having to
    pass the id in manually.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Read the contextvar and attach it to this record.

        Always returns True (never filters a record out) — this class is
        used purely for its side effect of enriching the record, not for
        deciding what gets logged.
        """
        record.correlation_id = correlation_id_var.get()
        return True


class CloudLoggingFormatter(jsonlogger.JsonFormatter):
    """Emits JSON log lines in the shape Cloud Logging expects.

    Cloud Logging auto-parses structured (JSON) stdout/stderr from a
    Cloud Run container and specifically looks for a `severity` field to
    drive log-level filtering in the console — the stdlib's own
    `record.levelname` isn't picked up by name, so it has to be copied
    into `severity` explicitly here.
    """

    def add_fields(self, log_record: dict, record: logging.LogRecord, message_dict: dict) -> None:
        """Add/override fields on the JSON payload for a single log line.

        Called by the base JsonFormatter for every record; `log_record` is
        the dict that ultimately gets serialized to JSON. We call the
        parent implementation first to get the default fields, then layer
        our own on top.
        """
        super().add_fields(log_record, record, message_dict)
        log_record["severity"] = record.levelname
        log_record["logger"] = record.name
        log_record.setdefault("correlation_id", getattr(record, "correlation_id", "-"))


def setup_logging() -> None:
    """Configure the root logger for the whole process.

    Called once at startup by api/main.py, mcp/server.py, and
    infrastructure/rag/ingest.py — every module in the app then just does
    `get_logger(__name__)` and logs normally; this function is what makes
    those log calls come out as Cloud-Logging-shaped JSON instead of the
    stdlib's default plain text.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CloudLoggingFormatter("%(message)s"))
    handler.addFilter(CorrelationIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.LOG_LEVEL)


def get_logger(name: str) -> logging.Logger:
    """Return a standard library logger for the given module name.

    Thin wrapper so call sites do `get_logger(__name__)` instead of
    importing `logging` directly everywhere — keeps the structured-logging
    setup this module owns as the one place that knows how logging is
    configured.
    """
    return logging.getLogger(name)
