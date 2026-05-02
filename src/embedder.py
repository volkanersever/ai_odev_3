"""Local embeddings via the Ollama HTTP API.

The brief mandates a local embedding model.  We call the Ollama ``embeddings``
endpoint over plain HTTP with ``urllib`` -- no Ollama Python SDK required, no
external API.

Embeddings are L2-normalised on creation so cosine similarity reduces to a dot
product downstream.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Iterable

import numpy as np

from config import EMBED_MODEL, OLLAMA_HOST


class OllamaError(RuntimeError):
    """Raised when the local Ollama runtime is unreachable or returns an error."""


def _post_json(url: str, body: dict, timeout: int = 120) -> dict:
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise OllamaError(f"Ollama HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Is `ollama serve` running?"
        ) from exc


def embed_one(text: str, *, model: str = EMBED_MODEL) -> np.ndarray:
    """Embed a single string.  Returns a normalised float32 vector."""
    if not text.strip():
        raise ValueError("Cannot embed empty text")
    data = _post_json(
        f"{OLLAMA_HOST}/api/embeddings",
        {"model": model, "prompt": text},
    )
    vec = data.get("embedding")
    if not vec:
        raise OllamaError(f"Ollama returned no embedding: {data}")
    arr = np.asarray(vec, dtype=np.float32)
    return _l2_normalise(arr)


def embed_many(
    texts: Iterable[str],
    *,
    model: str = EMBED_MODEL,
    progress: bool = True,
) -> np.ndarray:
    """Embed a batch of strings.

    Ollama's HTTP embeddings endpoint is single-shot, so we loop.  A short
    sleep between calls keeps memory pressure low on smaller machines.
    """
    texts = list(texts)
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    vectors: list[np.ndarray] = []
    total = len(texts)
    start = time.time()
    for i, text in enumerate(texts, 1):
        vectors.append(embed_one(text, model=model))
        if progress and (i % 25 == 0 or i == total):
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0.0
            print(f"    embedded {i}/{total}  ({rate:.1f} chunk/s)")
    return np.vstack(vectors).astype(np.float32)


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return vec
    return vec / norm


def ping() -> bool:
    """Return True if the Ollama runtime answers a model-list request."""
    try:
        request = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False
