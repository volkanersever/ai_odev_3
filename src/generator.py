"""Local LLM answer generation via Ollama.

Talks to ``/api/generate`` over plain HTTP.  Two modes are exposed: a blocking
``generate`` that returns the full string, and a ``stream`` generator that
yields tokens as they arrive (used by the Streamlit UI for a real-time feel).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Iterator

from config import LLM_MODEL, LLM_NUM_CTX, LLM_TEMPERATURE, OLLAMA_HOST
from src.embedder import OllamaError
from src.vector_store import SearchHit


SYSTEM_PROMPT = (
    "You are a careful assistant that answers questions strictly from the "
    "provided Wikipedia context.\n"
    "Rules:\n"
    "- Use ONLY the information in the CONTEXT block. Do not rely on outside knowledge.\n"
    "- If the context does not contain the answer, your ENTIRE reply must be exactly: I don't know.\n"
    "  Do not explain, do not apologise, do not list what is missing -- just \"I don't know.\"\n"
    "- Cite the source titles in square brackets at the end of relevant sentences, e.g. [Albert Einstein].\n"
    "- Be concise and factual. Do not invent dates, places, or numbers.\n"
    "- For comparison questions, address each subject in turn using only the context.\n"
)


def build_prompt(query: str, hits: list[SearchHit]) -> str:
    """Assemble the user prompt, embedding labelled context blocks."""
    if not hits:
        context_block = "(no context retrieved)"
    else:
        formatted = []
        for i, hit in enumerate(hits, 1):
            label = hit.entry.canonical_title or hit.entry.source_title
            formatted.append(f"[{i}] Source: {label}\n{hit.entry.text}")
        context_block = "\n\n---\n\n".join(formatted)

    return (
        f"CONTEXT:\n{context_block}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer using only the context above. If the context does not contain "
        "the answer, reply exactly: \"I don't know.\""
    )


def generate(
    query: str,
    hits: list[SearchHit],
    *,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    num_ctx: int = LLM_NUM_CTX,
) -> str:
    """Blocking call that returns the full answer string."""
    prompt = build_prompt(query, hits)
    body = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise OllamaError(f"Ollama HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Is `ollama serve` running?"
        ) from exc
    return (payload.get("response") or "").strip()


def stream(
    query: str,
    hits: list[SearchHit],
    *,
    model: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
    num_ctx: int = LLM_NUM_CTX,
) -> Iterator[str]:
    """Yield tokens as they arrive from Ollama."""
    prompt = build_prompt(query, hits)
    body = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                event = json.loads(line)
                token = event.get("response", "")
                if token:
                    yield token
                if event.get("done"):
                    return
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. Is `ollama serve` running?"
        ) from exc
