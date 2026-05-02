"""Chunk every raw article, embed it via Ollama, and persist the vector index.

Usage:
    python -m scripts.build_index            # incremental build
    python -m scripts.build_index --reset    # rebuild from scratch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make repo root importable when running directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from config import (  # noqa: E402
    PEOPLE_INDEX_PATH,
    PLACES_INDEX_PATH,
    RAW_PEOPLE_DIR,
    RAW_PLACES_DIR,
    ensure_dirs,
)
from src.chunker import chunk_text  # noqa: E402
from src.db import (  # noqa: E402
    connect,
    init_db,
    insert_chunk,
    reset_for_article,
    upsert_article,
)
from src.embedder import embed_many, ping  # noqa: E402
from src.vector_store import StoreEntry, VectorStore  # noqa: E402


def _load_articles(directory: Path, entity_type: str) -> list[dict]:
    articles: list[dict] = []
    if not directory.exists():
        return articles
    for meta_path in sorted(directory.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text_path = meta_path.with_suffix(".txt")
        if not text_path.exists():
            continue
        articles.append(
            {
                "title": meta["title"],
                "canonical_title": meta.get("canonical_title", meta["title"]),
                "url": meta.get("url", ""),
                "entity_type": entity_type,
                "text": text_path.read_text(encoding="utf-8"),
            }
        )
    return articles


def _build_one(
    articles: list[dict],
    *,
    index_path: Path,
    entity_type: str,
    reset: bool,
) -> int:
    store = VectorStore(index_path)
    if reset:
        store.reset()
    else:
        store.load()

    if not articles:
        print(f"  no {entity_type} articles found in raw/, skipping")
        return 0

    chunks_total = 0
    with connect() as conn:
        if reset:
            conn.execute("DELETE FROM chunks WHERE entity_type = ?", (entity_type,))
            conn.execute("DELETE FROM articles WHERE entity_type = ?", (entity_type,))

        for article in articles:
            title = article["title"]
            chunks = chunk_text(
                article["text"],
                source_title=title,
                canonical_title=article["canonical_title"],
                url=article["url"],
                entity_type=entity_type,
            )
            if not chunks:
                continue

            print(f"  embedding {len(chunks):>3} chunks for {title}")
            vectors = embed_many([c.text for c in chunks], progress=False)

            entries = [
                StoreEntry(
                    id=f"{title}::{c.chunk_index}",
                    text=c.text,
                    source_title=c.source_title,
                    canonical_title=c.canonical_title,
                    url=c.url,
                    entity_type=c.entity_type,
                    chunk_index=c.chunk_index,
                )
                for c in chunks
            ]
            store.add(vectors, entries)

            reset_for_article(conn, title)
            upsert_article(
                conn,
                title=title,
                canonical_title=article["canonical_title"],
                url=article["url"],
                entity_type=entity_type,
                char_count=len(article["text"]),
                chunk_count=len(chunks),
            )
            for entry in entries:
                insert_chunk(
                    conn,
                    chunk_id=entry.id,
                    article_title=title,
                    chunk_index=entry.chunk_index,
                    entity_type=entity_type,
                    text=entry.text,
                )
            chunks_total += len(chunks)

    store.save()
    print(f"  -> wrote {len(store)} vectors (dim={store.dim}) to {index_path}")
    return chunks_total


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local vector index")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the existing index and rebuild from scratch",
    )
    args = parser.parse_args()

    ensure_dirs()
    init_db()

    if not ping():
        print(
            "ERROR: Ollama is not reachable on http://localhost:11434.\n"
            "       Start it with `ollama serve` and ensure the embedding model is pulled:\n"
            "       `ollama pull nomic-embed-text`\n"
        )
        return 2

    print("Building people index...")
    n_people = _build_one(
        _load_articles(RAW_PEOPLE_DIR, "person"),
        index_path=PEOPLE_INDEX_PATH,
        entity_type="person",
        reset=args.reset,
    )

    print("\nBuilding places index...")
    n_places = _build_one(
        _load_articles(RAW_PLACES_DIR, "place"),
        index_path=PLACES_INDEX_PATH,
        entity_type="place",
        reset=args.reset,
    )

    print(
        f"\nIndex build complete: {n_people} people chunks, "
        f"{n_places} places chunks indexed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
