"""Local Wikipedia RAG package.

Modules implement each pipeline stage:

* ``ingest``     -- pull plain-text Wikipedia articles via the public REST API
* ``chunker``    -- split documents into overlapping chunks
* ``embedder``   -- compute embeddings via the local Ollama runtime
* ``vector_store`` -- a NumPy-backed vector index written entirely from scratch
* ``classifier`` -- decide whether a query targets people, places, or both
* ``retriever``  -- top-k semantic search with metadata filtering
* ``generator``  -- prompt a local LLM to produce grounded answers
* ``rag``        -- orchestrate the full pipeline
* ``db``         -- SQLite metadata store for chunks and source articles
"""

__all__ = [
    "ingest",
    "chunker",
    "embedder",
    "vector_store",
    "classifier",
    "retriever",
    "generator",
    "rag",
    "db",
]
