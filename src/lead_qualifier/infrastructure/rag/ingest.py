"""Rerunnable script that embeds knowledge/*.md and loads the result into
the knowledge_chunks table (see infrastructure/db/orm_models.py's
KnowledgeChunk) that infrastructure/rag/retriever.py's PgVectorRetriever
queries at runtime.

Run with:
    python -m lead_qualifier.infrastructure.rag.ingest

Chunking splits on top-level "## " markdown headers — knowledge/*.md is
expected to be a handful of curated documents (ICP definition, sales
playbook, past deal notes), not raw unstructured text, so headers are a
natural and sufficient chunk boundary without pulling in a
tokenizer/text-splitter dependency.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from openai import AsyncOpenAI
from sqlalchemy import delete

from lead_qualifier.core.config import settings
from lead_qualifier.core.logging import get_logger, setup_logging
from lead_qualifier.infrastructure.db.orm_models import KnowledgeChunk
from lead_qualifier.infrastructure.db.session import async_session_factory

logger = get_logger(__name__)

# knowledge/ lives at the project root, not inside src/ — it's editorial
# content, not code, so it's kept out of the installed package.
_KNOWLEDGE_DIR = Path(__file__).resolve().parents[4] / "knowledge"

# Zero-width split right before each "## " line, so the header itself
# stays with the section that follows it instead of being consumed by
# the split.
_SECTION_SPLIT = re.compile(r"(?m)^(?=## )")


def _chunk_markdown(text: str) -> list[str]:
    """Split into one chunk per "## " section. A file with no such
    headers becomes a single chunk of its full (trimmed) contents."""
    return [section.strip() for section in _SECTION_SPLIT.split(text) if section.strip()]


async def ingest() -> None:
    """Embed and (re)load every knowledge/*.md file into knowledge_chunks.

    Each source file's existing chunks are deleted before its new ones
    are inserted, so re-running this after editing a .md file reflects
    that edit exactly rather than accumulating stale chunks alongside it.
    """
    if not _KNOWLEDGE_DIR.is_dir():
        logger.warning(
            "Knowledge directory not found, nothing to ingest",
            extra={"path": str(_KNOWLEDGE_DIR)},
        )
        return

    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async with async_session_factory() as session:
        for path in sorted(_KNOWLEDGE_DIR.glob("*.md")):
            source = path.stem
            chunks = _chunk_markdown(path.read_text())
            if not chunks:
                logger.warning("No chunks extracted, skipping", extra={"source": source})
                continue

            response = await client.embeddings.create(
                model=settings.OPENAI_EMBEDDING_MODEL, input=chunks
            )
            # Sorted by index rather than trusting response order to match
            # input order — the API includes `index` on each embedding
            # specifically because that isn't a guarantee.
            sorted_data = sorted(response.data, key=lambda item: item.index)
            embeddings = [item.embedding for item in sorted_data]

            await session.execute(delete(KnowledgeChunk).where(KnowledgeChunk.source == source))
            session.add_all(
                KnowledgeChunk(source=source, content=chunk, embedding=embedding)
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            )
            logger.info(
                "Ingested knowledge source",
                extra={"source": source, "chunk_count": len(chunks)},
            )

        await session.commit()


def main() -> None:
    setup_logging()
    asyncio.run(ingest())


if __name__ == "__main__":
    main()
