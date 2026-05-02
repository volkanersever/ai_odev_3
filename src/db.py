"""SQLite metadata store for source articles and chunks.

The vector store on its own is enough to answer queries, but the brief
explicitly lists SQLite as part of the recommended stack.  Persisting article
and chunk metadata in a relational table makes administrative tasks (counting
documents, listing entities, deleting an article) trivial.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    title             TEXT PRIMARY KEY,
    canonical_title   TEXT NOT NULL,
    url               TEXT NOT NULL,
    entity_type       TEXT NOT NULL CHECK (entity_type IN ('person', 'place')),
    char_count        INTEGER NOT NULL,
    chunk_count       INTEGER NOT NULL DEFAULT 0,
    fetched_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id                TEXT PRIMARY KEY,
    article_title     TEXT NOT NULL REFERENCES articles(title) ON DELETE CASCADE,
    chunk_index       INTEGER NOT NULL,
    entity_type       TEXT NOT NULL,
    text              TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_article  ON chunks(article_title);
CREATE INDEX IF NOT EXISTS idx_chunks_type     ON chunks(entity_type);
"""


@contextmanager
def connect(path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path = DB_PATH) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)


def upsert_article(
    conn: sqlite3.Connection,
    *,
    title: str,
    canonical_title: str,
    url: str,
    entity_type: str,
    char_count: int,
    chunk_count: int,
) -> None:
    conn.execute(
        """
        INSERT INTO articles (title, canonical_title, url, entity_type, char_count, chunk_count)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(title) DO UPDATE SET
            canonical_title = excluded.canonical_title,
            url             = excluded.url,
            entity_type     = excluded.entity_type,
            char_count      = excluded.char_count,
            chunk_count     = excluded.chunk_count,
            fetched_at      = datetime('now')
        """,
        (title, canonical_title, url, entity_type, char_count, chunk_count),
    )


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    chunk_id: str,
    article_title: str,
    chunk_index: int,
    entity_type: str,
    text: str,
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO chunks (id, article_title, chunk_index, entity_type, text)
        VALUES (?, ?, ?, ?, ?)
        """,
        (chunk_id, article_title, chunk_index, entity_type, text),
    )


def reset_for_article(conn: sqlite3.Connection, article_title: str) -> None:
    conn.execute("DELETE FROM chunks WHERE article_title = ?", (article_title,))


def stats(path: Path = DB_PATH) -> dict:
    with connect(path) as conn:
        rows = conn.execute(
            """
            SELECT entity_type, COUNT(*) AS articles, COALESCE(SUM(chunk_count), 0) AS chunks
            FROM articles GROUP BY entity_type
            """
        ).fetchall()
        return {row["entity_type"]: dict(row) for row in rows}


def all_titles(path: Path = DB_PATH, entity_type: str | None = None) -> list[str]:
    with connect(path) as conn:
        if entity_type:
            cur = conn.execute(
                "SELECT title FROM articles WHERE entity_type = ? ORDER BY title",
                (entity_type,),
            )
        else:
            cur = conn.execute("SELECT title FROM articles ORDER BY title")
        return [row["title"] for row in cur.fetchall()]
