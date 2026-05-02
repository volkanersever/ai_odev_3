"""A from-scratch NumPy vector store.

The brief asks us to favour language-native primitives over fully-featured
libraries that solve the core problem out of the box.  Chroma is *suggested*
but not required, so we implement our own store: vectors live in a single
``float32`` matrix on disk, metadata in a JSONL file.  Search is a single
matrix multiplication followed by ``argpartition`` -- exactly what FAISS-Flat
does under the hood, but small enough to read in five minutes.

The store supports:

* normalised cosine similarity (dot product on normalised vectors)
* metadata filtering by predicate (e.g. ``entity_type == "person"``)
* persistence as ``vectors.npy`` + ``meta.jsonl``
* incremental ``add`` of new chunks without rebuilding from scratch
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


@dataclass
class StoreEntry:
    """One chunk stored in the index, mirrored on disk."""

    id: str
    text: str
    source_title: str
    canonical_title: str
    url: str
    entity_type: str
    chunk_index: int


@dataclass
class SearchHit:
    """A retrieval result with its similarity score."""

    entry: StoreEntry
    score: float


class VectorStore:
    """In-memory + on-disk vector store backed by NumPy arrays."""

    VECTORS_FILE: str = "vectors.npy"
    META_FILE: str = "meta.jsonl"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._vectors: np.ndarray | None = None
        self._entries: list[StoreEntry] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def load(self) -> "VectorStore":
        vec_path = self.path / self.VECTORS_FILE
        meta_path = self.path / self.META_FILE
        if not vec_path.exists() or not meta_path.exists():
            self._vectors = None
            self._entries = []
            return self
        self._vectors = np.load(vec_path)
        self._entries = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self._entries.append(StoreEntry(**json.loads(line)))
        if self._vectors.shape[0] != len(self._entries):
            raise RuntimeError(
                f"Index corrupt: {self._vectors.shape[0]} vectors vs {len(self._entries)} entries"
            )
        return self

    def save(self) -> None:
        if self._vectors is None or len(self._entries) == 0:
            return
        np.save(self.path / self.VECTORS_FILE, self._vectors.astype(np.float32))
        with (self.path / self.META_FILE).open("w", encoding="utf-8") as f:
            for entry in self._entries:
                f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def reset(self) -> None:
        """Remove on-disk index files and clear in-memory state."""
        for name in (self.VECTORS_FILE, self.META_FILE):
            p = self.path / name
            if p.exists():
                p.unlink()
        self._vectors = None
        self._entries = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def add(self, vectors: np.ndarray, entries: Sequence[StoreEntry]) -> None:
        if vectors.shape[0] != len(entries):
            raise ValueError("vectors and entries length mismatch")
        if vectors.size == 0:
            return
        vectors = vectors.astype(np.float32, copy=False)
        if self._vectors is None:
            self._vectors = vectors
        else:
            if self._vectors.shape[1] != vectors.shape[1]:
                raise ValueError(
                    f"Embedding dim mismatch: existing {self._vectors.shape[1]}, new {vectors.shape[1]}"
                )
            self._vectors = np.vstack([self._vectors, vectors])
        self._entries.extend(entries)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query_vec: np.ndarray,
        *,
        top_k: int = 5,
        predicate: Callable[[StoreEntry], bool] | None = None,
    ) -> list[SearchHit]:
        if self._vectors is None or len(self._entries) == 0:
            return []
        if query_vec.ndim != 1:
            raise ValueError("query_vec must be 1-D")

        # Vectors are stored normalised, so dot product == cosine similarity.
        scores = self._vectors @ query_vec.astype(np.float32)

        if predicate is not None:
            mask = np.fromiter(
                (predicate(entry) for entry in self._entries),
                count=len(self._entries),
                dtype=bool,
            )
            if not mask.any():
                return []
            # Suppress non-matching entries by sending their score to -inf.
            scores = np.where(mask, scores, -np.inf)

        k = min(top_k, scores.size)
        if k <= 0:
            return []
        # ``argpartition`` is O(n); ``argsort`` over the top-k is cheap.
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        hits: list[SearchHit] = []
        for i in idx:
            score = float(scores[int(i)])
            if not np.isfinite(score):
                continue
            hits.append(SearchHit(entry=self._entries[int(i)], score=score))
        return hits

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    @property
    def dim(self) -> int:
        return 0 if self._vectors is None else int(self._vectors.shape[1])

    @property
    def entries(self) -> list[StoreEntry]:
        return list(self._entries)

    def cosine_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Return the cosine score of every stored vector against ``query_vec``.

        The retriever uses this when it needs the cosine score of a chunk
        that the keyword scan surfaced but cosine top-K did not.
        """
        if self._vectors is None:
            return np.zeros((0,), dtype=np.float32)
        return self._vectors @ query_vec.astype(np.float32)
