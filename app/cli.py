"""Minimal CLI chat interface.

Run with::

    python -m app.cli

Slash commands:
    /context on|off     show retrieved chunks alongside answers
    /reset              clear conversation history (the index is unchanged)
    /stats              show index statistics
    /help               list commands
    /exit               quit
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running ``python app/cli.py`` directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.embedder import OllamaError, ping  # noqa: E402
from src.rag import RAGPipeline  # noqa: E402

BANNER = """\
==============================================================
  Local Wikipedia RAG Assistant (CLI)
  Type a question, or /help for commands.  /exit to quit.
==============================================================
"""

HELP = """\
Commands:
  /context on|off   Toggle showing retrieved source chunks
  /reset            Clear chat history (vector index is unchanged)
  /stats            Show index statistics
  /help             Show this help
  /exit             Exit the program
"""


def main() -> int:
    print(BANNER)

    if not ping():
        print(
            "WARN: Ollama runtime not detected on http://localhost:11434.\n"
            "      Run `ollama serve` and `ollama pull llama3.2:3b nomic-embed-text` first.\n"
        )

    pipeline = RAGPipeline()
    if not pipeline.is_ready:
        print(
            "WARN: Vector index is empty. Run\n"
            "      `python -m scripts.ingest && python -m scripts.build_index`\n"
            "      before asking questions.\n"
        )
    else:
        summary = pipeline.index_summary()
        print(
            f"Loaded index: {summary['people_chunks']} people chunks, "
            f"{summary['places_chunks']} places chunks "
            f"(dim={summary['people_dim'] or summary['places_dim']}).\n"
        )

    show_context = False
    history: list[tuple[str, str]] = []

    while True:
        try:
            query = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue

        if query.startswith("/"):
            cmd, _, arg = query[1:].partition(" ")
            cmd = cmd.lower()
            if cmd in {"exit", "quit"}:
                break
            if cmd == "help":
                print(HELP)
                continue
            if cmd == "context":
                show_context = arg.strip().lower() in {"on", "true", "1", "yes"}
                print(f"context display: {'on' if show_context else 'off'}\n")
                continue
            if cmd == "reset":
                history.clear()
                print("history cleared\n")
                continue
            if cmd == "stats":
                print(pipeline.index_summary(), "\n")
                continue
            print(f"unknown command: /{cmd}\n")
            continue

        try:
            result = pipeline.answer(query)
        except OllamaError as exc:
            print(f"Ollama error: {exc}\n")
            continue

        print(f"\nrag> {result.text}\n")
        print(
            f"     [classified as {result.classification.query_type.value}; "
            f"searched {', '.join(result.searched_stores) or 'none'}; "
            f"{len(result.hits)} chunks; {result.latency_seconds:.2f}s]"
        )
        if show_context and result.hits:
            print("\n     --- retrieved context ---")
            for i, hit in enumerate(result.hits, 1):
                snippet = hit.entry.text[:240].replace("\n", " ")
                print(
                    f"     [{i}] {hit.entry.canonical_title} "
                    f"(score={hit.score:.3f})\n         {snippet}..."
                )
        print()
        history.append((query, result.text))

    print("bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
