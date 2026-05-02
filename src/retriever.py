"""Top-level retrieval that combines the classifier with the vector stores.

Search is **hybrid**: a cosine-similarity pass over the dense vectors plus a
small keyword-overlap bonus.  Pure dense retrieval sometimes loses to obvious
lexical cues -- e.g. the query *"Which famous place is located in Turkey"*
ranks several South-American sites above Hagia Sophia because the
embedding's notion of "Turkey" is diffuse.  Adding a tiny BM25-lite term
boost recovers those cases without changing the architecture.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from src.classifier import ClassificationResult, QueryType, classify
from src.embedder import embed_one
from src.vector_store import SearchHit, StoreEntry, VectorStore

# A very small stopword list -- keep it tight so we don't over-prune signal.
_STOPWORDS = {
    "the", "and", "or", "of", "in", "on", "at", "to", "for", "is", "are",
    "was", "were", "be", "been", "being", "a", "an", "by", "with", "as",
    "that", "this", "these", "those", "it", "its", "from", "into", "about",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "do", "does", "did", "have", "has", "had", "can", "could", "would",
    "should", "may", "might", "will", "shall",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _query_keywords(query: str) -> set[str]:
    text = unicodedata.normalize("NFKD", query).lower()
    return {
        t for t in _TOKEN_RE.findall(text)
        if len(t) >= 4 and t not in _STOPWORDS
    }


def _proper_nouns(query: str) -> set[str]:
    """Lower-cased proper nouns lifted from the *original* query.

    Proper nouns are typically named entities (countries, cities, people)
    and carry far more retrieval signal than common nouns even when their
    corpus IDF is similar.  We use this to apply a second, stronger
    bonus on top of the IDF-weighted overlap.
    """
    # Skip the very first token because English sentences start with a
    # capital letter -- "Where" / "What" should not count as a proper noun.
    tokens = query.split()
    candidates: list[str] = []
    for i, tok in enumerate(tokens):
        clean = re.sub(r"[^A-Za-z]", "", tok)
        if not clean:
            continue
        if i == 0 and clean.lower() in _STOPWORDS:
            continue
        if i == 0 and len(clean) <= 5 and clean.lower() in {
            "what", "where", "who", "whom", "whose", "when", "why", "how",
            "which", "is", "are", "was", "were", "do", "does", "did",
            "can", "could", "should", "would", "may", "might", "tell",
            "compare", "describe",
        }:
            continue
        if _PROPER_NOUN_RE.fullmatch(clean):
            candidates.append(clean.lower())
    # Drop the leading capital of the very first surviving token if it
    # is also a stopword.
    return {c for c in candidates if c not in _STOPWORDS}


def _keyword_overlap_score(
    query_terms: set[str],
    chunk_text: str,
    *,
    term_weights: dict[str, float] | None = None,
) -> float:
    """IDF-weighted overlap of query keywords inside the chunk, in [0, 1].

    ``term_weights`` is an optional IDF table.  Rare terms (like a country
    name that only appears in one article) carry far more signal than
    common terms (*place*, *located*) and so should drive ranking when
    cosine alone is undecided.
    """
    if not query_terms:
        return 0.0
    text_lower = chunk_text.lower()
    if term_weights is None:
        # Fall back to an unweighted log-tempered fraction.
        hits = sum(1 for term in query_terms if term in text_lower)
        return math.log1p(hits) / math.log1p(len(query_terms)) if hits else 0.0

    total_weight = sum(term_weights.get(t, 1.0) for t in query_terms)
    if total_weight == 0:
        return 0.0
    earned = sum(term_weights.get(t, 1.0) for t in query_terms if t in text_lower)
    return earned / total_weight


# When a query has no entity match we cast a wider net before re-ranking.
_RERANK_POOL = 80
# Weight applied to the IDF-weighted overlap bonus.
_KEYWORD_WEIGHT = 0.20
# Additional weight applied per matched proper noun (e.g. country names).
# Proper nouns are unambiguous retrieval signals so they earn a flat boost
# that is enough to outrank a chunk lacking the same noun.
_PROPER_NOUN_WEIGHT = 0.20


@dataclass
class RetrievalResult:
    classification: ClassificationResult
    hits: list[SearchHit]
    searched_stores: list[str]


class HybridRetriever:
    """Routes queries between the people and places vector stores."""

    def __init__(
        self,
        people_store: VectorStore,
        places_store: VectorStore,
        *,
        people: list[str],
        places: list[str],
    ) -> None:
        self.people_store = people_store
        self.places_store = places_store
        self.people = people
        self.places = places
        # Per-store IDF tables, computed lazily on first use.
        self._idf: dict[int, dict[str, float]] = {}

    def _idf_for(self, store: VectorStore) -> dict[str, float]:
        key = id(store)
        cached = self._idf.get(key)
        if cached is not None:
            return cached

        n_docs = max(len(store), 1)
        df: dict[str, int] = {}
        for entry in store.entries:
            seen = set(_TOKEN_RE.findall(entry.text.lower()))
            for tok in seen:
                if len(tok) >= 4 and tok not in _STOPWORDS:
                    df[tok] = df.get(tok, 0) + 1
        # Standard IDF with smoothing: ln((N + 1) / (df + 1)) + 1.
        idf = {
            tok: math.log((n_docs + 1) / (count + 1)) + 1.0
            for tok, count in df.items()
        }
        self._idf[key] = idf
        return idf

    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        cls = classify(query, people=self.people, places=self.places)
        query_vec = embed_one(query)
        query_terms = _query_keywords(query)
        proper_nouns = _proper_nouns(query)

        searched: list[str] = []
        hits: list[SearchHit] = []

        if cls.query_type in (QueryType.PERSON, QueryType.BOTH):
            hits.extend(
                self._search_one(
                    self.people_store,
                    query_vec,
                    query_terms=query_terms,
                    proper_nouns=proper_nouns,
                    matched_titles=cls.matched_people,
                    top_k=top_k,
                )
            )
            searched.append("people")

        if cls.query_type in (QueryType.PLACE, QueryType.BOTH):
            hits.extend(
                self._search_one(
                    self.places_store,
                    query_vec,
                    query_terms=query_terms,
                    proper_nouns=proper_nouns,
                    matched_titles=cls.matched_places,
                    top_k=top_k,
                )
            )
            searched.append("places")

        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:top_k]

        return RetrievalResult(classification=cls, hits=hits, searched_stores=searched)

    # ------------------------------------------------------------------
    def _search_one(
        self,
        store: VectorStore,
        query_vec,
        *,
        query_terms: set[str],
        proper_nouns: set[str],
        matched_titles: list[str],
        top_k: int,
    ) -> list[SearchHit]:
        # Comparison queries: pull a per-entity slice so each subject gets
        # equal representation, then merge.  Without this, cosine bias toward
        # one of the two articles would drown the other.
        if len(matched_titles) >= 2:
            per_entity = max(1, top_k // len(matched_titles))
            extra = top_k - per_entity * len(matched_titles)
            collected: list[SearchHit] = []
            for i, title in enumerate(matched_titles):
                pred = self._entity_filter([title])
                limit = per_entity + (1 if i < extra else 0)
                collected.extend(store.search(query_vec, top_k=limit, predicate=pred))
            collected.sort(key=lambda h: h.score, reverse=True)
            return collected[:top_k]

        predicate = self._entity_filter(matched_titles)

        # Single named entity: trust the title filter and skip the
        # keyword re-rank -- pulling the right article is enough.
        if predicate is not None:
            return store.search(query_vec, top_k=top_k, predicate=predicate)

        if len(store) == 0:
            return []
        if not query_terms and not proper_nouns:
            return store.search(query_vec, top_k=top_k)

        # Compute cosine scores once for the full store; that lets us mix
        # the dense top-K with a keyword-only pass without paying a second
        # matrix multiply.
        cosine_scores = store.cosine_scores(query_vec)

        idf = self._idf_for(store)
        term_weights = {t: idf.get(t, 1.0) for t in query_terms}

        import numpy as np
        cosine_top_idx = np.argpartition(
            -cosine_scores, min(_RERANK_POOL, cosine_scores.size) - 1
        )[: _RERANK_POOL]

        # Candidate set: top-N by cosine PLUS any chunk that mentions a
        # query keyword or proper noun.  The full lexical scan over the
        # ~thousands-of-chunks store costs a few milliseconds, so it's a
        # cheap way to recover precision lost by a diffuse embedding.
        candidate_idx: set[int] = set(int(i) for i in cosine_top_idx)
        match_targets = query_terms | proper_nouns
        for i, entry in enumerate(store.entries):
            text_lower = entry.text.lower()
            if any(t in text_lower for t in match_targets):
                candidate_idx.add(i)

        rescored: list[SearchHit] = []
        for i in candidate_idx:
            entry = store.entries[i]
            text_lower = entry.text.lower()
            cosine = float(cosine_scores[i])
            bonus = _keyword_overlap_score(
                query_terms, entry.text, term_weights=term_weights
            )
            # Strong, flat boost per proper noun present in the chunk.
            proper_hits = sum(1 for n in proper_nouns if n in text_lower)
            score = (
                cosine
                + _KEYWORD_WEIGHT * bonus
                + _PROPER_NOUN_WEIGHT * proper_hits
            )
            rescored.append(SearchHit(entry=entry, score=score))
        rescored.sort(key=lambda h: h.score, reverse=True)
        return rescored[:top_k]

    # ------------------------------------------------------------------
    @staticmethod
    def _entity_filter(matched_titles: list[str]):
        """Build a strict predicate over canonical / source title.

        When the classifier identifies a specific entity by name we want the
        retriever to *only* pull chunks from that article; semantic similarity
        across articles introduces more confusion than help.
        """
        if not matched_titles:
            return None
        wanted_lower = {t.lower() for t in matched_titles}

        def predicate(entry: StoreEntry) -> bool:
            return (
                entry.source_title.lower() in wanted_lower
                or entry.canonical_title.lower() in wanted_lower
            )

        return predicate
