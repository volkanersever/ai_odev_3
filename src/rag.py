"""High-level RAG orchestrator.

A single ``RAGPipeline`` instance owns the vector stores and exposes the
question-answering interface used by both the CLI and the Streamlit app.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

from config import (
    DEFAULT_TOP_K,
    PEOPLE,
    PEOPLE_INDEX_PATH,
    PLACES,
    PLACES_INDEX_PATH,
)
from src import generator
from src.classifier import ClassificationResult
from src.retriever import HybridRetriever, RetrievalResult
from src.vector_store import SearchHit, VectorStore


@dataclass
class Answer:
    query: str
    text: str
    classification: ClassificationResult
    hits: list[SearchHit]
    searched_stores: list[str]
    latency_seconds: float = 0.0
    history: list[tuple[str, str]] = field(default_factory=list)


class RAGPipeline:
    """Glue between the retriever and the generator."""

    def __init__(
        self,
        people_store: VectorStore | None = None,
        places_store: VectorStore | None = None,
        people: list[str] | None = None,
        places: list[str] | None = None,
    ) -> None:
        self.people_store = people_store or VectorStore(PEOPLE_INDEX_PATH).load()
        self.places_store = places_store or VectorStore(PLACES_INDEX_PATH).load()
        self.retriever = HybridRetriever(
            self.people_store,
            self.places_store,
            people=people or PEOPLE,
            places=places or PLACES,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        return len(self.people_store) > 0 and len(self.places_store) > 0

    def index_summary(self) -> dict:
        return {
            "people_chunks": len(self.people_store),
            "places_chunks": len(self.places_store),
            "people_dim": self.people_store.dim,
            "places_dim": self.places_store.dim,
        }

    # ------------------------------------------------------------------
    # Retrieval-only entry point (useful for the "show context" feature)
    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k=top_k)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def answer(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> Answer:
        start = time.time()
        retrieval = self.retriever.retrieve(query, top_k=top_k)

        if not retrieval.hits:
            text = (
                "I don't know. The local index does not contain information "
                "relevant to that question."
            )
            return Answer(
                query=query,
                text=text,
                classification=retrieval.classification,
                hits=[],
                searched_stores=retrieval.searched_stores,
                latency_seconds=time.time() - start,
            )

        text = generator.generate(query, retrieval.hits)
        return Answer(
            query=query,
            text=text,
            classification=retrieval.classification,
            hits=retrieval.hits,
            searched_stores=retrieval.searched_stores,
            latency_seconds=time.time() - start,
        )

    def answer_stream(self, query: str, *, top_k: int = DEFAULT_TOP_K) -> Iterator[str]:
        retrieval = self.retriever.retrieve(query, top_k=top_k)
        if not retrieval.hits:
            yield "I don't know. The local index does not contain information relevant to that question."
            return
        yield from generator.stream(query, retrieval.hits)
